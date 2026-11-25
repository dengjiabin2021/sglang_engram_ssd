"""管理本 rank 的 owner 查询、批次双缓冲和同步 io_uring 工作区。"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .catalog import EngramSsdCatalog, get_engram_ssd_catalog
from .config import get_config

if TYPE_CHECKING:
    from types import ModuleType

    from .hashing import RequestHasher
    from .query.native import NativeBatchRowReader, NativePrefetchRowPool
    from .sched_prefetch import SchedulerPrefetchController, UrgentReadRequest

logger = logging.getLogger(__name__)

# 每套主存和显存字节区、decode 的 CPU pinned int64 hash 与 int32 [1] 标志、标志的 GPU 地址。
BatchStorage = tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, int]


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


@dataclass(eq=False)
class Batch:
    """关联一次 forward、双缓冲的活跃视图和跨流事件。

    调度线程登记并在 forward 提交末尾记录 done；后台组装数据并发布 copied 或 decode 标志。
    runtime 与组装方持有至各自完成；同一套内存再次写入前等待上一次 done。
    """

    storage: BatchStorage
    queries: int  # 本批各层的查询总数 Q，包含非 owner 位置。
    rows: int  # 暂存行容量 K；prefill 用去重行数，decode 用查询数上界。
    previous: torch.cuda.Event | None  # 同一套内存上一个 batch 的 forward 结束事件。
    copied: torch.cuda.Event  # decode 记录在整批 H2D 末尾。
    done: torch.cuda.Event  # 记录在模型计算流上。
    decode: bool
    request: UrgentReadRequest | None = None  # prefill 组装请求。
    expected: torch.Tensor | None = None  # 调试用 CPU int64 [T,L,C] 全局行号。
    forward: Any = None  # 原生 ForwardBatch 身份，由 Hasher Hook 绑定。
    stream: torch.cuda.Stream | None = None  # forward 实际使用的计算流。
    hash_ids: torch.Tensor | None = None  # 预留 hash 内存的 CPU pinned int64 [T,L,C] 活跃视图。
    hash_done: torch.cuda.Event | None = None  # GPU hash 拷回与标志重置完成。
    queued: bool = False  # decode 由后台负责完成；否则由 forward 同步组装。
    waited: bool = False  # decode 已在第一个 Engram 层插入计算流依赖。
    layer_copied: dict[int, torch.cuda.Event] = field(default_factory=dict)  # prefill 各层 H2D 末尾的事件。
    waited_layers: set[int] = field(default_factory=set)  # prefill 已插入计算流依赖的层。
    urgent: dict[int, int] = field(default_factory=dict)  # prefill 各层本 rank 的紧急缺行数。


class EngramSsdRuntime:
    """持有单 rank 的表、只读 FD、前后台 reader 及 CUDA 工作区。

    由缓存工厂创建，模型与调度线程共用，进程退出或显式 close 时关闭。
    同步读取以 lock 保护单套中转区；后台使用独立 reader，仅行池元数据与模型线程共享。
    双缓冲由调度线程分配给 batch，后台与计算流按 event 和标志交接；TP 通信在模型或调度线程完成。
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
        pool_bytes: int | None = None,
        log_interval: int = 0,
        batch_tokens: int = 0,
        debug_check_rows: bool = False,
    ) -> None:
        """建立本 rank 的同步资源，并按开关建立调度预取资源。

        输入：
            catalog：已解析的本地 checkpoint 表目录，借用其不可变元数据。
            dim：行宽 D，与 catalog 一致；tp_rank、tp_size：本 rank 编号及 TP 总数。
            chunk_rows：每次同步读取的最大行数，决定单套中转区容量。
            io_uring_queue_depth：每个独立 ring 的在途请求上限。
            device：已选定的 NVIDIA GPU；调用方保证当前 CUDA 设备与其一致。
            dedup_enabled：同步查询是否先去重；sched_prefetch_enabled：是否建立后台 reader／行池。
            pool_bytes：显式的行数据字节数；为空时等 Hasher 参数就绪再按配置定容。
            log_interval：正数时开启原生计数及补读计时。
            batch_tokens：双缓冲启动 token 容量；debug_check_rows：是否逐位核对 prefill 行号。
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
        self.batch_tokens = batch_tokens
        self.debug_check_rows = debug_check_rows
        self._buffers: list[tuple[BatchStorage, torch.cuda.Event | None]] = []  # 两套内存和各自最后的 done。
        self._retired: list[tuple[BatchStorage, torch.cuda.Event]] = []  # 扩容后仍在途的旧内存。
        self._next_buffer = 0
        self._batch: tuple[Any, Batch] | None = None  # 正在 run_batch 的 ScheduleBatch 与其工作区。
        self._fallback_logged = 0  # lookup_batch 已记录的退回同步查询次数，只记前 20 次。
        self._bind_logged = 0  # bind_batch 已记录的对象身份不一致次数，只记前 20 次。
        self._copy_stream: torch.cuda.Stream | None = None
        self._hash_stream: torch.cuda.Stream | None = None
        self._stream_ops: ModuleType | None = None
        self.tables: dict[int, EngramSsdTable] = {}
        self.fds: dict[Path, int] = {}
        self.lock = threading.Lock()  # 保护同步中转区复用；关闭资源时也必须持有。
        self.reader: NativeBatchRowReader | None = None
        self.prefetch_reader: NativeBatchRowReader | None = None
        self.decode_reader: NativeBatchRowReader | None = None  # decode 专用 ring，不排在后台预取块之后
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

                # 三个 reader 各持有独立 ring／锁，只借用同一组只读 FD。
                # forward 补读和 decode 读取都不等待后台整段读取，但仍竞争磁盘带宽。
                self.prefetch_reader = NativeBatchRowReader(
                    tables=table_specs, io_uring_queue_depth=io_uring_queue_depth,
                )
                self.decode_reader = NativeBatchRowReader(
                    tables=table_specs, io_uring_queue_depth=io_uring_queue_depth,
                )
                if pool_bytes is not None:
                    self.pool = NativePrefetchRowPool(dim=dim, pool_bytes=pool_bytes, track_stats=self.stats_enabled)
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
        """保存 CPU hash 参数，并按模型的哈希列数初始化行池和双缓冲。

        输入：hasher 为模型初始化 Hook 生成的只读 RequestHasher，须在请求预取开始前设置。
        输出：无返回值；先校验并分配行池，再分配双缓冲。双缓冲或流内存操作不可用时走同步查询。
        """
        if hasher.layer_ids != self.catalog.layer_ids:
            raise ValueError("Engram CPU hash layer order differs from the checkpoint catalog")
        self.hasher = hasher
        self.columns = hasher.state.offsets.shape[1]
        if not self.sched_prefetch_enabled:
            return
        if self.pool is None:
            from sglang.srt.runtime_context import get_schedule

            from .query import NativePrefetchRowPool

            schedule = get_schedule()
            chunked = schedule.chunked_prefill_size or 0
            pool_bytes = get_config().resolve_pool_bytes(
                tokens=chunked if chunked > 0 else schedule.max_prefill_tokens,
                layers=len(hasher.layer_ids), columns=self.columns, dim=self.dim, tp_size=self.tp_size,
            )
            self.pool = NativePrefetchRowPool(dim=self.dim, pool_bytes=pool_bytes, track_stats=self.stats_enabled)
        if not self.batch_tokens:
            return
        from .query.native import load_native_module

        self._stream_ops = load_native_module()
        capacity = self.batch_tokens * len(hasher.layer_ids) * self.columns
        with torch.cuda.device(self.device):
            try:
                # 调度和模型流使用默认优先级池；旁路流使用独立的 CUDA 流池。
                self._copy_stream = torch.cuda.Stream(device=self.device, priority=-1)
                self._hash_stream = torch.cuda.Stream(device=self.device, priority=-1)
                self._buffers = [
                    (self._allocate_batch_storage(capacity=capacity, probe=(index == 0)), None) for index in range(2)
                ]
            except (RuntimeError, MemoryError) as error:
                logger.warning("Engram SSD batch buffer allocation failed on rank %s: %s", self.tp_rank, error)

    def _allocate_batch_storage(self, *, capacity: int, probe: bool = False) -> BatchStorage:
        """为一套双缓冲分配固定容量，并探测 decode 的流内存操作。

        输入：capacity 为查询数和行数上界；probe 仅在启动时实测流等待，扩容只映射内存。
        输出：CPU pinned 与 GPU 字节区、可选的 hash 和 H2D 放行标志及其映射地址，均由 runtime 持有。
        """
        size = capacity * (4 + self.dim + self.scale_dim)
        host = torch.empty(size, dtype=torch.uint8, device="cpu", pin_memory=True)
        device = torch.empty(size, dtype=torch.uint8, device=self.device)
        if self._hash_stream is not None:
            flags = torch.zeros(1, dtype=torch.int32, device="cpu", pin_memory=True)
            address = self._stream_ops.stream_flag_address(
                flags=flags, stream=self._hash_stream.cuda_stream, probe=probe,
            )
            if address:
                ids = torch.empty(capacity, dtype=torch.int64, device="cpu", pin_memory=True)
                return host, device, ids, flags, address
            self._hash_stream = None
            logger.warning(
                "Engram SSD stream memory operations unavailable on rank %s; decode uses lookup", self.tp_rank,
            )
        return host, device, None, None, 0

    def reserve_batch(self, *, batch: Any, queries: int, rows: int, decode: bool) -> Batch | None:
        """轮换一套缓冲区给本次 run_batch，保留复用需要等待的事件。

        输入：batch 为 ScheduleBatch；queries 为各层查询总数，rows 为保存行数据的上界；decode 指明模式。
        输出：当前批次的工作区；缓冲不可用或扩容失败返回 None。这里仅登记，组装方才等待上一批事件。
        """
        if not self._buffers or not queries or (decode and self._hash_stream is None):
            return None
        self._retired = [(storage, done) for storage, done in self._retired if not done.query()]
        index = self._next_buffer
        storage, previous = self._buffers[index]
        if queries * (4 + self.dim + self.scale_dim) > storage[0].numel():
            with torch.cuda.device(self.device):
                try:
                    grown = self._allocate_batch_storage(capacity=queries)
                except (RuntimeError, MemoryError) as error:
                    logger.warning("Engram SSD cannot grow batch buffer to %s queries: %s", queries, error)
                    return None
            if decode and self._hash_stream is None:
                return None
            if previous is not None:
                self._retired.append((storage, previous))
            storage, previous = grown, None
        current = Batch(
            storage=storage, queries=queries, rows=max(1, rows), previous=previous,
            copied=torch.cuda.Event(), done=torch.cuda.Event(), decode=decode,
        )
        self._batch = (batch, current)
        self._buffers[index] = (storage, current.done)
        self._next_buffer = 1 - index
        return current

    def batch_rows(
        self, *, ids: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[int, torch.Tensor], int]:
        """按层筛选并去重 owner 行，分配一套连续暂存下标。

        输入：ids 为 CPU int64 [T,L,C] 的全局行号，可重复，各层在表范围内，不要求连续。
        输出：CPU int32 [L*T*C] 映射、各层唯一行号和目标下标（连续 CPU int64 向量）、唯一行总数。
        映射按层优先、token 再哈希列排列，非 owner 位置为 -1。
        """
        tokens, layers, columns = ids.shape
        mapping = torch.full((layers, tokens * columns), -1, dtype=torch.int32, device="cpu")
        rows_by_layer, slots_by_layer = {}, {}
        start = 0
        for column, layer in enumerate(self.catalog.layer_ids):
            table = self.open_table(layer_id=layer)
            lo = table.rows * self.tp_rank // self.tp_size
            hi = table.rows * (self.tp_rank + 1) // self.tp_size
            rows = ids[:, column].reshape(-1)
            owner = (rows >= lo) & (rows < hi)
            rows, inverse = torch.unique(rows[owner], sorted=True, return_inverse=True)
            end = start + rows.numel()
            mapping[column, owner] = (inverse + start).to(torch.int32)
            rows_by_layer[layer] = rows
            slots_by_layer[layer] = torch.arange(start, end, dtype=torch.int64, device="cpu")
            start = end
        return mapping.flatten(), rows_by_layer, slots_by_layer, start

    def batch_host(self, *, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把本批 CPU 字节区切成映射、weight、scale 三张连续视图。

        输入：batch 提供查询数 Q 与行容量 K；写入方须先等同一套上一批的 forward 结束。
        输出：CPU pinned int32 [Q]、uint8 [K,D]、uint8 [K,D/32]，三者共用底层分配。
        """
        host = batch.storage[0]
        weight = 4 * batch.queries
        scale = weight + batch.rows * self.dim
        end = scale + batch.rows * self.scale_dim
        return (
            host[:weight].view(torch.int32),
            host[weight:scale].view(batch.rows, self.dim),
            host[scale:end].view(batch.rows, self.scale_dim),
        )

    def stage_prefill(self, *, batch: Batch, mapping: torch.Tensor) -> None:
        """等暂存区可覆盖，再写入 prefill 的 CPU int32 [Q] 映射。

        输入：batch 为 prefill 缓冲，mapping 为连续 CPU int32 [Q]，值为 -1 或暂存下标。
        输出：无返回值；后台或前台补读线程可继续填充同一暂存区的行数据。
        """
        if batch.previous is not None:
            batch.previous.synchronize()
        self.batch_host(batch=batch)[0].copy_(mapping)

    def copy_batch(self, *, batch: Batch) -> None:
        """在独立拷贝流上提交整批字节区的一次 H2D 并记录事件。

        输入：batch 提供 CPU pinned 源和 GPU 目标；流上已有依赖或主机已经整理完源数据。
        输出：无返回值；batch.copied 已记录，源和目标内存保留到本批 forward 结束。
        """
        host, device, _, _, _ = batch.storage
        size = 4 * batch.queries + batch.rows * (self.dim + self.scale_dim)
        with torch.cuda.device(self.device), torch.cuda.stream(self._copy_stream):
            device[:size].copy_(host[:size], non_blocking=True)
            device.record_stream(self._copy_stream)
            batch.copied.record(self._copy_stream)

    def copy_layer(self, *, batch: Batch, layer: int, slots: torch.Tensor) -> None:
        """在拷贝流上提交 prefill 单层 weight／scale 的 H2D，首个提交的层同时拷映射。

        输入：batch 为 prefill 缓冲；layer 为层号；slots 为该层连续的 CPU int64 暂存下标，可为空。
        输出：无返回值；batch.layer_copied[layer] 已记录。拷贝流按提交顺序执行，后面层的事件也覆盖映射。
        """
        host, device, _, _, _ = batch.storage
        weight = 4 * batch.queries
        scale = weight + batch.rows * self.dim
        lo, hi = (int(slots[0]), int(slots[-1]) + 1) if slots.numel() else (0, 0)
        event = torch.cuda.Event()
        with torch.cuda.device(self.device), torch.cuda.stream(self._copy_stream):
            if not batch.layer_copied:
                device[:weight].copy_(host[:weight], non_blocking=True)
            w_lo, w_hi = weight + lo * self.dim, weight + hi * self.dim
            s_lo, s_hi = scale + lo * self.scale_dim, scale + hi * self.scale_dim
            device[w_lo:w_hi].copy_(host[w_lo:w_hi], non_blocking=True)
            device[s_lo:s_hi].copy_(host[s_lo:s_hi], non_blocking=True)
            device.record_stream(self._copy_stream)
            event.record(self._copy_stream)
        batch.layer_copied[layer] = event

    def bind_batch(self, *, forward_batch: Any, hash_ids: torch.Tensor) -> None:
        """将调度登记与真正的 ForwardBatch 关联；decode 同时提交旁路 D2H 和有门控的 H2D。

        输入：forward_batch 来自模型 Hasher，hash_ids 为 GPU int64 [T,L,C]，允许重复。
        输出：无返回值；本次 run_batch 内首个 token 数相符的 hash 调用登记消费，decode 的 CPU 行号由唯一后台线程读取。
        """
        if self._batch is None:
            return
        scheduled, batch = self._batch
        # overlap 调度下 input_ids 会在 forward 前被替换成新张量，不能按对象身份匹配；
        # _batch 只在本次 run_batch 内有效，取第一个查询数相同的 hash 调用即可。
        if batch.forward is not None or hash_ids.numel() != batch.queries:
            return
        same_ids = forward_batch.input_ids is scheduled.input_ids
        if self._bind_logged < 20 and (not same_ids or forward_batch.forward_mode != scheduled.forward_mode):
            # 排查用：记录旧的对象身份匹配为何不成立，绑定本身已不依赖它。
            self._bind_logged += 1
            ids, sched_ids = forward_batch.input_ids, scheduled.input_ids
            logger.warning(
                "Engram SSD bind identity differs on rank %s: same_ids=%s fb_ids=%s sb_ids=%s fb_mode=%s sb_mode=%s",
                self.tp_rank, same_ids,
                None if ids is None else (tuple(ids.shape), ids.dtype, ids.device, ids.data_ptr()),
                None if sched_ids is None else (tuple(sched_ids.shape), sched_ids.dtype, sched_ids.device, sched_ids.data_ptr()),
                forward_batch.forward_mode, scheduled.forward_mode,
            )
        batch.forward = forward_batch
        batch.stream = torch.cuda.current_stream(device=self.device)
        if not batch.decode:
            return
        _, _, ids, _, address = batch.storage
        batch.hash_ids = ids[:batch.queries].view(hash_ids.shape)
        batch.hash_done = torch.cuda.Event()
        self._hash_stream.wait_stream(batch.stream)
        if batch.previous is not None:
            self._hash_stream.wait_event(batch.previous)
        with torch.cuda.stream(self._hash_stream):
            self._stream_ops.reset_stream_flag(stream=self._hash_stream.cuda_stream, address=address)
            batch.hash_ids.copy_(hash_ids, non_blocking=True)
            hash_ids.record_stream(self._hash_stream)
            batch.hash_done.record(self._hash_stream)
        # 提前记录 H2D 事件供计算流等待；组装方置位后，拷贝流才读取暂存区。
        self._copy_stream.wait_event(batch.hash_done)
        self._stream_ops.wait_stream_flag(stream=self._copy_stream.cuda_stream, address=address, value=1)
        self.copy_batch(batch=batch)
        batch.queued = self.sched_prefetch.submit_decode(batch=batch, iteration=scheduled.forward_iter)

    def publish_decode(self, *, batch: Batch) -> None:
        """放行已入队的 decode H2D。

        输入：batch 的 hash 事件已完成且暂存区已写完，或当前 forward 已中止。
        输出：无返回值；写入主机标志，计算流等待 copied 保证 H2D 完成。
        """
        self._stream_ops.set_stream_flag(flags=batch.storage[3])

    def end_batch(self, *, batch: Any) -> None:
        """在实际模型流上记录 forward 的结束，随后解除本次 run_batch 的绑定。

        输入：batch 为刚提交完成的 ScheduleBatch。输出：无返回值；对应缓冲区下次使用时等待其事件。
        """
        if self._batch is None or self._batch[0] is not batch:
            return
        _, current = self._batch
        current.done.record(current.stream or torch.cuda.current_stream(device=self.device))
        self._batch = None

    def get_sched_prefetch(self, *, scheduler: Any) -> SchedulerPrefetchController:
        """取得本 rank 的唯一预取控制器。

        输入：scheduler 为本 rank 调度器，提供 CPU 通信组与 chunk 配置；须已开启调度预取。
        输出：runtime 持有的 SchedulerPrefetchController；首次调用会创建唯一后台线程。
        仅由调度主线程调用；此入口不读取 SSD，也不等待在途 GPU 工作。
        """
        from .sched_prefetch import get_manager

        return get_manager(runtime=self, scheduler=scheduler)

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
            不要求 contiguous，可重复；行号由官方 Hasher 按表行数取模得到，必在 [0,table.rows) 内，此处不再检查。
        输出：GPU BF16 [*indices.shape,D]，非 owner 位置为零；后续由调用方 reduce_output。
        SSD 读取和中转区使用同步完成；输出回填在当前流排队。非法设备／dtype 抛 ValueError，
        前台 I/O 错误直接传播；后台错误只在调度的 TP MIN 边界传播。
        """
        self._check_indices(indices=indices)
        table = self.open_table(layer_id=layer_id)
        flat_ids = indices.reshape(-1)  # GPU int32／int64 [Q]，Q 为本次查询的全部位置数。
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

    def lookup_batch(self, *, layer_id: int, indices: torch.Tensor, forward_batch: Any) -> torch.Tensor:
        """等待已登记批次的 H2D，并按显存映射 gather 本 rank 的 BF16 输出。

        输入：layer_id 为模型层号，indices 为 GPU int32／int64 [T,C]，可重复也可不连续；
            forward_batch 为原生批次，未登记时使用同步查询。
        输出：GPU BF16 [T,C,D]；非 owner 位置为零。调试时逐位核对 prefill 的真实行号。
        """
        batch = self._batch[1] if self._batch is not None else None
        if batch is None or batch.forward is not forward_batch or forward_batch is None:
            if self._fallback_logged < 20:
                # 只记前 20 次退回同步查询的原因，预热请求之后的记录用于确认 forward 是否用上预取缓冲。
                self._fallback_logged += 1
                logger.warning(
                    "Engram SSD lookup_batch falls back to sync lookup on rank %s: registered=%s bound=%s mode=%s",
                    self.tp_rank, batch is not None, batch is not None and batch.forward is not None,
                    getattr(forward_batch, "forward_mode", None),
                )
            return self.lookup(layer_id=layer_id, indices=indices)

        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        column = self.catalog.layer_ids.index(layer_id)
        if batch.expected is not None:
            self.check_rows(layer_id=layer_id, indices=indices, expected=batch.expected[:, column])
        stream = torch.cuda.current_stream(device=self.device)
        if batch.decode:
            if not batch.waited:
                if not batch.queued:
                    # 交接前后台已停止，本线程用旁路流的 hash 结果同步补读。
                    self.sched_prefetch.read_decode(batch=batch, retry=True)
                stream.wait_event(batch.copied)
                batch.waited = True
        elif layer_id not in batch.waited_layers:
            # 每层只等自己的行：事件记录前不能让 CUDA 等待，先等后台发起这一层的复制。
            self.sched_prefetch.wait_urgent(request=batch.request, layer=layer_id)
            stream.wait_event(batch.layer_copied[layer_id])
            batch.waited_layers.add(layer_id)

        device = batch.storage[1]
        per_layer = batch.queries // len(self.catalog.layer_ids)
        if indices.numel() != per_layer:
            # 原生 decoder SWA replay 可只给晚层传入 extend 尾部，本批映射覆盖完整 extend。
            return self.lookup(layer_id=layer_id, indices=indices)
        device.record_stream(stream)
        weight = batch.queries * 4
        scale = weight + batch.rows * self.dim
        # 当前层的 [T*C] 映射连续；原生 gather 将 -1 掩码为零。
        ids = device[:weight].view(torch.int32)[column * per_layer:(column + 1) * per_layer]
        out = torch.empty((*indices.shape, self.dim), dtype=torch.bfloat16, device=self.device)
        engram_gather(
            weight_ptr=device.data_ptr() + weight, scale_ptr=device.data_ptr() + scale,
            ids=ids, out=out.view(-1, self.dim), dim=self.dim, block_size=32, row_lo=0, row_hi=batch.rows,
        )
        return out

    @staticmethod
    def check_rows(*, layer_id: int, indices: torch.Tensor, expected: torch.Tensor) -> None:
        """逐位比较 GPU 全局行号和 prefill 的 CPU 预测。

        输入：layer_id 为层号，indices 为 GPU int32／int64 [T,C]，expected 为 CPU int64 [T,C]；
            两者可重复、无需连续。
        输出：无返回值；形状或行号不一致时报告该层及首个不同位置。
        """
        actual = indices.detach().to(device="cpu", dtype=torch.int64)
        if actual.shape != expected.shape:
            raise RuntimeError(f"Engram layer {layer_id}: GPU hash shape {actual.shape} != CPU {expected.shape}")
        mismatch = (actual != expected).nonzero()
        if mismatch.numel():
            token, column = mismatch[0].tolist()
            raise RuntimeError(
                f"Engram layer {layer_id}: row mismatch at ({token}, {column}), "
                f"GPU={actual[token, column].item()} CPU={expected[token, column].item()}"
            )

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
        输出：无返回值；先停止并 join 后台线程，再收齐 CUDA 工作、关闭行池／reader／FD。
        已关闭对象可再次调用；行池引用未归还等关闭错误直接传播，不跳过仍存活资源。
        """
        if self.sched_prefetch is not None:
            self.sched_prefetch.close()
            if self._batch is not None:
                _, current = self._batch
                if current.hash_done is not None and not current.queued and not current.waited:
                    # 已中止的 forward 无需组装；解除未交接批次的 H2D 门控后即可回收。
                    current.hash_done.synchronize()
                    self.publish_decode(batch=current)
                if current.stream is not None:
                    current.stream.synchronize()
            self.sched_prefetch = None
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        for _, done in self._buffers + self._retired:
            if done is not None:
                done.synchronize()
        self._buffers.clear()
        self._retired.clear()
        self._batch = None
        with self.lock:
            if self.pool is not None:
                if self.stats_enabled:
                    import json
                    stats = self.pool.stats()
                    stats.update(phase="shutdown", tp_rank=self.tp_rank)
                    logger.info(
                        "Engram SSD sched prefetch stats: %s", json.dumps(stats, sort_keys=True),
                    )
                self.pool.close()
                self.pool = None
            # 后台线程已退出，所有 ring 都关闭后才归还它们借用的 FD。
            if self.prefetch_reader is not None:
                self.prefetch_reader.close()
                self.prefetch_reader = None
            if self.decode_reader is not None:
                self.decode_reader.close()
                self.decode_reader = None
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
            self._copy_stream = None
            self._hash_stream = None


@lru_cache(maxsize=1)
def get_engram_ssd_runtime(dim: int) -> EngramSsdRuntime:
    """按进程缓存本 rank runtime，并登记退出时的 close。

    输入：dim 为 checkpoint 的 Engram 行宽 D；同一进程的所有调用使用同一行宽。
    输出：缓存的 EngramSsdRuntime，由工厂持有并管理退出生命周期，调用方只借用。
    仅在插件生效且当前 CUDA 设备已设置后调用；配置与扩展初始化错误直接传播。
    """
    from sglang.srt.runtime_context import get_parallel, get_schedule

    config = get_config()
    parallel = get_parallel()
    batch_tokens = 0
    if config.sched_prefetch_enabled:
        schedule = get_schedule()
        batch_tokens = max(schedule.chunked_prefill_size or 0, schedule.max_prefill_tokens)
    runtime = EngramSsdRuntime(
        catalog=get_engram_ssd_catalog(), dim=dim,
        tp_rank=parallel.tp_rank, tp_size=parallel.tp_size,
        chunk_rows=config.chunk_rows, io_uring_queue_depth=config.io_uring_queue_depth,
        device=torch.device("cuda", torch.cuda.current_device()),
        dedup_enabled=config.dedup_enabled,
        sched_prefetch_enabled=config.sched_prefetch_enabled,
        log_interval=config.log_interval, batch_tokens=batch_tokens, debug_check_rows=config.debug_check_rows,
    )
    atexit.register(runtime.close)
    return runtime
