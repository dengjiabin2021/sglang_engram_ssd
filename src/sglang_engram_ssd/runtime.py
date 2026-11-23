"""每个 rank 的 owner 查询、单套 CUDA 工作区与同步 io_uring 读取。"""

from __future__ import annotations

import atexit
import os
import threading
from time import perf_counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .catalog import EngramSsdCatalog, get_engram_ssd_catalog
from .config import get_config

if TYPE_CHECKING:
    from .hashing import RequestHasher
    from .query.native import NativeBatchRowReader, NativePrefetchRowPool
    from .sched_prefetch import SchedulerPrefetchController


@dataclass(frozen=True)
class EngramSsdTable:
    """保存单层只读表定位，不持有张量，也不负责 I/O。

    由 runtime 建立并持有至关闭；字段不可变，可供模型线程和后台线程并发读取。
    FD 的所有权仍在 runtime.fds，两个 reader 只借用这些值。
    """

    layer_id: int  # 模型层号，与 catalog 的键一致。
    rows: int  # 全局行数，合法行号为 [0,rows)。
    dim: int  # D：weight 每行字节数。
    row_bytes: int  # 一行 weight 加 scale 的有效字节数，尚不包含文件系统读放大。
    weight_fd: int  # runtime 打开的只读文件描述符，可以与 scale_fd 相同。
    scale_fd: int
    weight_offset: int  # weight 张量在文件中的绝对字节偏移。
    scale_offset: int  # scale 张量在文件中的绝对字节偏移。


@dataclass
class EngramSsdBuffer:
    """持有同步补读的一套 CPU／GPU 中转区，不建立行缓存。

    由 runtime 创建并持有，close 时归还；仅在 runtime.lock 内串行使用。
    CPU 填充或 GPU 覆盖前，由 runtime 的 chunk event 保证前一块 GPU 使用已完成。
    """

    host_weight: torch.Tensor  # CPU pinned uint8 [chunk_rows,D]，reader 直接写入。
    host_scale: torch.Tensor  # CPU pinned uint8 [chunk_rows,D/32]。
    device_weight: torch.Tensor  # GPU uint8 [chunk_rows,D]，gather 直接读取。
    device_scale: torch.Tensor  # GPU uint8 [chunk_rows,D/32]。


@dataclass
class PoolUse:
    """保留一次 mapped gather 的槽引用及索引，避免 GPU 仍读取时被回收。

    由 lookup_from_pool 建立、runtime 持有；collect_pool_uses 在 event 完成后释放。
    列表的读写由 runtime._pool_uses_lock 保护；event 记录失败时保留生产流供关闭阶段等待。
    """

    event: torch.cuda.Event | None  # gather 和输出回填之后记录；None 表示记录失败。
    handle: Any  # resolve 返回的独立原生引用，与请求段引用分开释放。
    slot_ids: torch.Tensor  # CPU／GPU int64 [Q]，保留实际传给 gather 的 GPU 索引生命周期。
    stream: torch.cuda.Stream  # 生产 gather 的流；event 缺失时仅关闭阶段同步它。


class EngramSsdRuntime:
    """持有单 rank 的表、只读 FD、前后台 reader 及 CUDA 工作区。

    由缓存工厂创建，模型与调度线程共用，进程退出或显式 close 时关闭。
    同步读取以 lock 保护单套中转区；后台使用独立 reader，仅行池元数据与模型线程共享。
    GPU 临时引用以独立锁保护，pool 内部负责槽状态互斥；后台线程不执行 TP 通信。
    """

    def __init__(
        self,
        catalog: EngramSsdCatalog,
        *,
        dim: int,
        tp_rank: int,
        tp_size: int,
        chunk_rows: int,
        io_uring_queue_depth: int,
        device: torch.device,
        dedup_enabled: bool = False,
        sched_prefetch_enabled: bool = False,
        pool_bytes: int = 2147483648,
        log_interval: int = 0,
    ) -> None:
        """建立本 rank 的同步资源，并按开关建立调度预取资源。

        输入：
            catalog：已解析的本地 checkpoint 表目录，借用其不可变元数据。
            dim：行宽 D，与 catalog 一致；tp_rank、tp_size：本 rank 编号及 TP 总数。
            chunk_rows：每次同步读取的最大行数，决定单套中转区容量。
            io_uring_queue_depth：每个独立 ring 的在途请求上限。
            device：已选定的 NVIDIA GPU；调用方保证当前 CUDA 设备与其一致。
            dedup_enabled：同步查询是否先去重；sched_prefetch_enabled：是否建立后台 reader／行池。
            pool_bytes：开启预取时的压缩 payload 预算；log_interval：正数时开启原生计数及补读计时。
        输出：无返回值；runtime 拥有 FD、reader、工作区及可选行池，调用方须最终 close。
        TP 配置非法抛 ValueError；初始化失败先关闭已经取得的资源，再传播原异常。
        """
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError("Engram SSD requires a valid TP rank and size")
        self.catalog = catalog
        self.dim = dim
        self.scale_dim = dim // 32
        self.capacity = chunk_rows  # 同步工作区行容量，不是预取池的槽数 S。
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.dedup_enabled = dedup_enabled
        self.hasher: RequestHasher | None = None  # 模型真实 Hasher 复制出的只读 CPU 参数。
        self.sched_prefetch_enabled = sched_prefetch_enabled
        self.stats_enabled = bool(log_interval)
        self.pool: NativePrefetchRowPool | None = None
        self.sched_prefetch: SchedulerPrefetchController | None = None
        self._pool_uses: list[PoolUse] | None = None  # 已提交 GPU 工作、尚未归还的临时槽引用。
        self._pool_uses_lock: threading.Lock | None = None
        self.tables: dict[int, EngramSsdTable] = {}
        self.fds: dict[Path, int] = {}
        self.lock = threading.Lock()  # 保护同步中转区复用；关闭资源时也必须持有。
        self.reader: NativeBatchRowReader | None = None
        self.prefetch_reader: NativeBatchRowReader | None = None
        try:
            for layer_id in catalog.layer_ids:
                self.init_table(layer_id=layer_id)
            from .query import NativeBatchRowReader, RowTableSpec

            table_specs = [
                RowTableSpec(
                    layer_id=t.layer_id, rows=t.rows, dim=t.dim,
                    weight_fd=t.weight_fd, scale_fd=t.scale_fd,
                    weight_offset=t.weight_offset, scale_offset=t.scale_offset,
                )
                for t in self.tables.values()
            ]
            # 两个 reader 都只复制表定位；FD 只由本 runtime 持有，不能由 reader 关闭。
            self.reader = NativeBatchRowReader(
                tables=table_specs,
                io_uring_queue_depth=io_uring_queue_depth,
            )
            host_weight, host_scale = self.reader.allocate_workspace(
                layer_id=catalog.layer_ids[0], capacity=chunk_rows,
            )
            self.buffer = EngramSsdBuffer(
                host_weight=host_weight, host_scale=host_scale,
                device_weight=torch.empty((chunk_rows, dim), dtype=torch.uint8, device=device),
                device_scale=torch.empty((chunk_rows, self.scale_dim), dtype=torch.uint8, device=device),
            )
            self.device_ids = torch.arange(chunk_rows, dtype=torch.int64, device=device)
            self._chunk_done = torch.cuda.Event()
            if sched_prefetch_enabled:
                from .query import NativePrefetchRowPool

                # 两个 reader 各持有独立 ring／锁，只借用同一组只读 FD。
                # forward 补读不等待后台整段读取，但两者仍竞争磁盘带宽。
                self.prefetch_reader = NativeBatchRowReader(
                    tables=table_specs, io_uring_queue_depth=io_uring_queue_depth,
                )
                self.pool = NativePrefetchRowPool(dim=dim, pool_bytes=pool_bytes, track_stats=self.stats_enabled)
                self._pool_uses = []
                self._pool_uses_lock = threading.Lock()
        except BaseException:
            self.close()
            raise

    def _open_shard(self, path: Path) -> int:
        """按路径复用只读 FD，避免 weight／scale 位于同一文件时重复打开。

        输入：path 为本地 shard 路径；仅初始化阶段由单线程调用。
        输出：借用的文件描述符；所有权保存在 self.fds，由 close 关闭，打开失败直接抛 OSError。
        """
        if path not in self.fds:
            self.fds[path] = os.open(path, os.O_RDONLY)
        return self.fds[path]

    def set_request_hasher(self, *, hasher: RequestHasher) -> None:
        """保存与 checkpoint 层顺序一致的 CPU hash 参数。

        输入：hasher 为模型初始化 Hook 生成的只读 RequestHasher，须在请求预取开始前设置。
        输出：无返回值；更新 self.hasher；层顺序不一致抛 ValueError，保留原状态。
        """
        if hasher.layer_ids != self.catalog.layer_ids:
            raise ValueError("Engram CPU hash layer order differs from the checkpoint catalog")
        self.hasher = hasher

    def get_sched_prefetch(self, *, scheduler: Any) -> SchedulerPrefetchController:
        """取得本 rank 的唯一预取控制器，并非阻塞地回收已完成的 GPU 引用。

        输入：scheduler 为本 rank 调度器，提供 CPU 通信组与 chunk 配置；须已开启调度预取。
        输出：runtime 持有的 SchedulerPrefetchController；首次调用会创建唯一后台线程。
        仅由调度主线程调用；此入口不读取 SSD，也不等待在途 GPU 工作。
        """
        from .sched_prefetch import get_manager

        self.collect_pool_uses()
        return get_manager(runtime=self, scheduler=scheduler)

    def collect_pool_uses(self, *, wait: bool = False) -> None:
        """按 CUDA 完成状态回收 mapped gather 的临时槽引用。

        输入：wait=False 仅 query event，不等待；wait=True 用于关闭阶段，等待 event 或原生产流。
        输出：无返回值；已完成引用归还 pool，并通知控制器容量可能变化，未完成项继续保留。
        引用列表受 _pool_uses_lock 保护；不在此等待 SSD，未开启行池时没有副作用。
        """
        if self.pool is None or self._pool_uses_lock is None or self._pool_uses is None:
            return
        released = False
        with self._pool_uses_lock:
            pending = []
            for use in self._pool_uses:
                if wait:
                    if use.event is None:
                        # event 未成功记录时只能在关闭阶段等待原生产流，不能提前归还槽。
                        use.stream.synchronize()
                    else:
                        use.event.synchronize()
                if wait or (use.event is not None and use.event.query()):
                    self.pool.release(handle=use.handle)
                    released = True
                else:
                    pending.append(use)
            self._pool_uses = pending
        # GPU 临时引用也可能是池满的最后一个原因；释放后允许后台重试占槽。
        # 在 lifetime 锁外通知，避免与控制器 condition 形成相反锁序。
        if released and self.sched_prefetch is not None:
            self.sched_prefetch.notify_pool_release()

    def init_table(self, layer_id: int) -> None:
        """为单层建立文件定位，复用同一 shard 的 FD。

        输入：layer_id 为 catalog 中的模型层号；仅初始化阶段单线程调用。
        输出：无返回值；在 self.tables 新增不可变表定位，重复调用不新增资源。
        不读取张量内容；缺少 catalog 项或打开 shard 失败时直接传播异常。
        """
        if layer_id in self.tables:
            return
        weight = self.catalog.views[f"model.layers.{layer_id}.engram.embed.weight"]
        scale = self.catalog.views[f"model.layers.{layer_id}.engram.embed.scale"]
        self.tables[layer_id] = EngramSsdTable(
            layer_id=layer_id, rows=weight.rows, dim=weight.width,
            row_bytes=weight.width + scale.width,
            weight_fd=self._open_shard(path=weight.path), scale_fd=self._open_shard(path=scale.path),
            weight_offset=weight.offset, scale_offset=scale.offset,
        )

    def open_table(self, layer_id: int) -> EngramSsdTable:
        """取得已初始化层的只读定位，不打开文件或提交 I/O。

        输入：layer_id 为模型层号；仅在 runtime 存活时使用。
        输出：借用的 EngramSsdTable，可供前后台并发读取；不存在的层抛 KeyError。
        """
        if layer_id not in self.tables:
            raise KeyError(f"Engram layer {layer_id} is not initialized")
        return self.tables[layer_id]

    def lookup(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """同步读取本 rank 所属行，恢复原查询形状，不在此执行 TP 通信。

        输入：layer_id 为表层号；indices 为 runtime GPU 上 int32／int64 [T,C] 或任意前导形状，
            不要求 contiguous，可重复，所有行号须在 [0,table.rows) 内。
        输出：GPU BF16 [*indices.shape,D]，非 owner 位置为零；后续由调用方 reduce_output。
        SSD 读取和中转区使用同步完成；输出回填在当前流排队。非法设备／dtype 抛 ValueError，
        行号越界抛 IndexError，前台 I/O 错误直接传播；后台错误只在调度的 TP MIN 边界传播。
        """
        self._check_indices(indices=indices)
        table = self.open_table(layer_id=layer_id)
        flat_ids = indices.reshape(-1)  # GPU int32／int64 [Q]，Q 为本次查询的全部位置数。
        if ((flat_ids < 0) | (flat_ids >= table.rows)).any().item():
            raise IndexError(f"Engram layer {layer_id}: row IDs outside [0,{table.rows})")
        if self.tp_size == 1:
            return self._read_rows_to_output(layer_id=layer_id, indices=indices)
        row_start = table.rows * self.tp_rank // self.tp_size
        row_end = table.rows * (self.tp_rank + 1) // self.tp_size
        # 保留原位置，避免按 owner 过滤后丢失 [T,C] 中的重复项或位置关系。
        positions = torch.nonzero((flat_ids >= row_start) & (flat_ids < row_end), as_tuple=False).flatten()
        out = torch.zeros((*indices.shape, table.dim), dtype=torch.bfloat16, device=indices.device)
        if positions.numel():
            values = self._read_rows_to_output(
                layer_id=layer_id, indices=flat_ids.index_select(0, positions),
            )
            out.view(-1, table.dim).index_copy_(0, positions, values)
        # 工作区已由 chunk event 保护；index_copy 与下游消费在同一 CUDA 流上有序执行。
        return out

    def lookup_from_pool(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """消费 READY 池槽，并用独立前台 reader 同步补齐未命中行。

        输入：layer_id、indices 的约束与 lookup 相同；indices 不要求 contiguous，允许重复。
        输出：GPU BF16 [*indices.shape,D] 的本 rank 局部结果，不做 TP 通信。
        READY 命中直接读取 CPU pinned 映射地址，不把 weight／scale 搬到 HBM；PENDING 视为 miss。
        resolve 取得的临时引用保留至 CUDA event 完成，调用方不负责释放。
        前台 I/O／CUDA 错误直接传播；后台错误保存在控制器，只在调度 MIN 边界统一传播。
        """
        if self.pool is None or not indices.numel():
            return self.lookup(layer_id=layer_id, indices=indices)
        self._check_indices(indices=indices)
        self.collect_pool_uses()
        table = self.open_table(layer_id=layer_id)
        # 真实 GPU 行号转为 CPU int64 [Q]，按原查询位置做 owner 过滤及池索引。
        flat_ids = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1).contiguous()
        if ((flat_ids < 0) | (flat_ids >= table.rows)).any().item():
            raise IndexError(f"Engram layer {layer_id}: row IDs outside [0,{table.rows})")
        lo = table.rows * self.tp_rank // self.tp_size
        hi = table.rows * (self.tp_rank + 1) // self.tp_size
        positions = torch.nonzero((flat_ids >= lo) & (flat_ids < hi), as_tuple=False).flatten()
        out = torch.zeros((*indices.shape, table.dim), dtype=torch.bfloat16, device=self.device)
        if not positions.numel():
            return out
        if self._pool_uses_lock is None or self._pool_uses is None:
            raise RuntimeError("Engram pinned pool lifetime tracking is not initialized")
        owner_ids = flat_ids.index_select(0, positions).contiguous()
        stream = torch.cuda.current_stream(device=self.device)
        event = torch.cuda.Event()
        # 只为 READY 槽加独立引用；PENDING 不被 GPU 使用，允许从前台 ring 重读。
        resolution = self.pool.resolve(layer_id=layer_id, row_ids=owner_ids)
        # 保留索引张量直到对应 CUDA event 完成；临时 handle 保证 mapped 槽不被覆盖。
        slot_ids = resolution.slot_ids
        failed = False
        try:
            hits = int(resolution.hit_mask.sum().item())
            misses = owner_ids.numel() - hits
            miss_ms = 0.0
            if hits:
                from sglang.kernels.ops.embeddings.engram_gather import engram_gather

                # 仅上传槽号；下面两个指针映射 CPU pinned 表，gather 直接跨总线读原始字节。
                slot_ids = slot_ids[resolution.hit_mask].to(device=self.device).contiguous()
                values = torch.empty((hits, self.dim), dtype=torch.bfloat16, device=self.device)
                engram_gather(
                    weight_ptr=self.pool.weight_device_ptr, scale_ptr=self.pool.scale_device_ptr,
                    ids=slot_ids, out=values, dim=self.dim, block_size=32,
                    row_lo=0, row_hi=self.pool.capacity,
                )
                destinations = positions[resolution.hit_mask].to(device=self.device)
                out.view(-1, self.dim).index_copy_(0, destinations, values)
            if misses:
                # 使用前台独立 ring，不等待后台整段 I/O；局部结果按原位置回填。
                missing = ~resolution.hit_mask
                started = perf_counter() if self.stats_enabled else 0.0
                values = self._read_rows_to_output(
                    layer_id=layer_id, indices=owner_ids[missing].contiguous(),
                )
                if self.stats_enabled:
                    miss_ms = (perf_counter() - started) * 1000
                destinations = positions[missing].to(device=self.device)
                out.view(-1, self.dim).index_copy_(0, destinations, values)
            if self.stats_enabled:
                self.pool.record_forward(hits=hits, misses=misses, miss_ms=miss_ms)
        except BaseException:
            failed = True
            raise
        finally:
            # 即使 miss 补读异常，先前提交的 mapped gather 仍可能在途，不能立即 release。
            use = PoolUse(event=None, handle=resolution.handle, slot_ids=slot_ids, stream=stream)
            try:
                event.record(stream)
                use.event = event
            except BaseException:
                # 不覆盖原读取异常，也不把未记录 event 当成 GPU 已完成。
                if not failed:
                    raise
            finally:
                with self._pool_uses_lock:
                    self._pool_uses.append(use)
        return out

    def _check_indices(self, *, indices: torch.Tensor) -> None:
        """校验模型侧查询张量的设备与整数类型，不复制或同步张量。

        输入：indices 为查询张量，须在 runtime 指定 GPU 上且 dtype 为 int32／int64。
        输出：无返回值；不合要求抛 ValueError；形状／连续性不限，行号范围由 lookup 校验。
        """
        if indices.device != self.device or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram SSD indices must be int32/int64 on the runtime CUDA device")

    def reduce_output(self, out: torch.Tensor) -> torch.Tensor:
        """在原 Engram 消费位置合并所有 owner 的局部结果。

        输入：out 为 GPU BF16 [T,C,D] 或带同类前导维度的局部结果，非 owner 位置为零。
        输出：同形状、同 dtype 的 TP 合并结果；TP=1 或空输出原样返回。
        仅模型线程调用；非空 TP 查询的所有 rank 必须按相同层顺序参与原生 all-reduce。
        """
        if self.tp_size == 1 or out.numel() == 0:
            return out
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(out)

    def _read_rows_to_output(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """通过前台 reader 分块读原始字节，在当前 CUDA 流 gather 成 BF16。

        输入：layer_id 为表层号；indices 为 CPU／runtime GPU int32／int64 查询张量，
            不要求 contiguous、允许重复，调用方已检查全局行号范围并筛选 owner。
        输出：GPU BF16 [*indices.shape,D]，恢复输入顺序和重复项；函数内不做 TP 通信。
        self.lock 保护整次中转区使用；每块等待 event 后才允许下一块覆盖，I/O 错误直接传播。
        dedup_enabled 时只读唯一行，再上传 inverse 以还原；关闭时不建立去重张量。
        """
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        table = self.open_table(layer_id=layer_id)
        requested_shape = indices.shape
        inverse = None
        if self.dedup_enabled and indices.numel():
            # 去重覆盖整个本地查询，而不是分别在每个 I/O chunk 中去重。
            indices, inverse = torch.unique(indices.reshape(-1), sorted=True, return_inverse=True)
        out = torch.empty((*indices.shape, table.dim), dtype=torch.bfloat16, device=self.device)
        if not indices.numel():
            return out
        # 同一套 host／device 工作区只能被一条前台查询使用；后台行池不占这把锁。
        with self.lock, torch.cuda.device(self.device):
            ids_cpu = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1).contiguous()
            if self.reader is None:
                raise RuntimeError("Engram SSD runtime is closed")
            buffer = self.buffer
            try:
                for start in range(0, ids_cpu.numel(), self.capacity):
                    end = min(start + self.capacity, ids_cpu.numel())
                    count = end - start
                    # 直接写 CPU pinned 中转区，返回前本块全部 I/O 已结束，不保留 reader 写入。
                    self.reader.read_rows(
                        layer_id=layer_id, row_ids=ids_cpu[start:end],
                        weight_out=buffer.host_weight, scale_out=buffer.host_scale,
                    )
                    # H2D 与 gather 在同一当前流排队；device_ids 是连续临时槽号，不是全局行号。
                    buffer.device_weight[:count].copy_(buffer.host_weight[:count], non_blocking=True)
                    buffer.device_scale[:count].copy_(buffer.host_scale[:count], non_blocking=True)
                    engram_gather(
                        weight_ptr=buffer.device_weight.data_ptr(),
                        scale_ptr=buffer.device_scale.data_ptr(),
                        ids=self.device_ids[:count], out=out.view(-1, self.dim)[start:end],
                        dim=self.dim, block_size=32, row_lo=0, row_hi=count,
                    )
                    # 等待本块 gather 读完 device 工作区；同时确保 H2D 已不再读取 host 工作区。
                    self._chunk_done.record(torch.cuda.current_stream(device=self.device))
                    self._chunk_done.synchronize()
            finally:
                # 包括异常路径，旧 GPU 使用结束前都不允许复用 host/device 工作区。
                self._chunk_done.record(torch.cuda.current_stream(device=self.device))
                self._chunk_done.synchronize()
        if inverse is not None:
            return out.index_select(0, inverse.to(device=self.device)).reshape(*requested_shape, table.dim)
        return out

    def close(self) -> None:
        """收齐后台 I/O 与 GPU 使用后，按所有权顺序释放 runtime 资源。

        输入：无显式参数；调用方须已停止新的调度／模型提交，不与 lookup 并发调用。
        输出：无返回值；先停止并 join 后台线程，再收齐 GPU 引用、关闭行池／reader／FD。
        已关闭对象可再次调用；行池引用未归还等关闭错误直接传播，不跳过仍存活资源。
        """
        if self.sched_prefetch is not None:
            self.sched_prefetch.close()
            self.sched_prefetch = None
        self.collect_pool_uses(wait=True)
        with self.lock:
            if self.pool is not None:
                if self.stats_enabled:
                    import json
                    import logging

                    stats = self.pool.stats()
                    stats.update(phase="shutdown", tp_rank=self.tp_rank)
                    logging.getLogger(__name__).info(
                        "Engram SSD sched prefetch stats: %s", json.dumps(stats, sort_keys=True),
                    )
                self.pool.close()
                self.pool = None
            # 后台线程已退出，两个 ring 都关闭后才归还它们借用的 FD。
            if self.prefetch_reader is not None:
                self.prefetch_reader.close()
                self.prefetch_reader = None
            if self.reader is not None:
                self.reader.close()
                self.reader = None
            for fd in self.fds.values():
                os.close(fd)
            self.fds.clear()
            self.tables.clear()
            # 关闭后的 factory 对象可能仍被 lru_cache 持有，显式归还大块工作区。
            self.buffer = None
            self.device_ids = None
            self._chunk_done = None
            self.hasher = None
            self._pool_uses = None
            self._pool_uses_lock = None


@lru_cache(maxsize=1)
def get_engram_ssd_runtime(dim: int) -> EngramSsdRuntime:
    """按进程缓存本 rank runtime，并登记退出时的 close。

    输入：dim 为 checkpoint 的 Engram 行宽 D；同一进程的所有调用使用同一行宽。
    输出：缓存的 EngramSsdRuntime，由工厂持有并管理退出生命周期，调用方只借用。
    仅在插件生效且当前 CUDA 设备已设置后调用；配置与扩展初始化错误直接传播。
    """
    from sglang.srt.runtime_context import get_parallel

    config = get_config()
    parallel = get_parallel()
    runtime = EngramSsdRuntime(
        catalog=get_engram_ssd_catalog(), dim=dim,
        tp_rank=parallel.tp_rank, tp_size=parallel.tp_size,
        chunk_rows=config.chunk_rows, io_uring_queue_depth=config.io_uring_queue_depth,
        device=torch.device("cuda", torch.cuda.current_device()),
        dedup_enabled=config.dedup_enabled,
        sched_prefetch_enabled=config.sched_prefetch_enabled, pool_bytes=config.pool_bytes,
        log_interval=config.log_interval,
    )
    atexit.register(runtime.close)
    return runtime
