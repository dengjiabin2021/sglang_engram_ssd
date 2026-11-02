"""每个 rank 一份的有限 DDR 行缓存和同步 CUDA 查表工作区。"""

from __future__ import annotations

import atexit
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .catalog import EngramSsdCatalog, get_engram_ssd_catalog, read_exact_at
from .config import get_config

if TYPE_CHECKING:
    from .query.native import NativeBatchRowReader


@dataclass(frozen=True)
class EngramSsdTable:
    layer_id: int
    rows: int
    dim: int
    row_bytes: int
    weight_fd: int
    scale_fd: int
    weight_offset: int
    scale_offset: int


@dataclass
class EngramSsdBuffer:
    """只保存一个块的张量、流和事件，分配及调度由 runtime 负责。"""

    host_weight: torch.Tensor
    host_scale: torch.Tensor
    device_weight: torch.Tensor
    device_scale: torch.Tensor
    gather_stream: torch.cuda.Stream  # 初始化时新建的独立流，执行 GPU gather 与反量化。
    copy_stream: torch.cuda.Stream    # 执行 Host → GPU 搬运，A/B 共用一条拷贝流。
    h2d_done: torch.cuda.Event        # Host 数据已搬走，CPU 可以重写本槽。
    gather_done: torch.cuda.Event     # GPU 数据已读完，H2D 可以覆盖本槽。
    lookup_done: torch.cuda.Event     # 本槽在本次 lookup 中的拷贝和 gather 已结束。


class EngramSsdRuntime:
    def __init__(
        self,
        catalog: EngramSsdCatalog,
        *,
        dim: int,
        lookup_mode: str,
        tp_rank: int,
        tp_size: int,
        query_backend: str,
        pipeline_enabled: bool,
        prefetch_layer_ids: tuple[int, ...],
        cache_bytes: int,
        chunk_rows: int,
        num_threads: int | None,
        thread_pool_queue_depth: int | None,
        io_uring_queue_depth: int | None,
        device: torch.device,
    ) -> None:
        self.catalog = catalog
        self.dim = dim
        self.scale_dim = dim // 32
        self.capacity = chunk_rows
        self.cache_budget = cache_bytes
        self.cached_bytes = 0
        self.cache: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()
        self.tables: dict[int, EngramSsdTable] = {}
        self.fds: dict[Path, int] = {}
        self.device = device
        self.lookup_mode = lookup_mode
        # 在模型线程构造时保存 TP 信息，后台查询不再读取 SGLang 的线程上下文。
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.query_backend = query_backend
        self.pipeline_enabled = pipeline_enabled
        self.prefetch_layer_ids = prefetch_layer_ids
        # 只管理整层 lookup 任务，不额外创建读取后端或预取块池。
        self.prefetch_executor: ThreadPoolExecutor | None = None
        self.buffers: list[EngramSsdBuffer] = []
        # 原生后端只创建一个 reader，所有 Engram 层共用其 C++ 缓存和读取资源。
        # Python 后端保持为 None，不导入或加载原生扩展。
        self.reader: NativeBatchRowReader | None = None
        # 配置已在模型加载前校验；构造时绑定查询实现，统一入口不处理具体查表逻辑。
        # 查询模式决定 rank 的查询范围，query_backend 决定 CPU 如何读取这些行。
        self._lookup_impl = {
            "all_queries": self._lookup_all_queries,
            "owner_allreduce": self._lookup_owner_allreduce,
        }[lookup_mode]
        # 仅保护同一进程中的并发调用，不串行不同 TP rank 的 SSD I/O。
        self.lock = threading.Lock()

        try:
            # 构造时初始化所有表句柄，形状已经由 catalog 集中校验。
            for layer_id in catalog.layer_ids:
                self.init_table(layer_id=layer_id)
            self._init_query_backend(
                num_threads=num_threads,
                thread_pool_queue_depth=thread_pool_queue_depth,
                io_uring_queue_depth=io_uring_queue_depth,
            )
            # 关闭时仍只分配一套；开启时 A/B 两套交替使用，行缓存预算不变。
            for _ in range(2 if pipeline_enabled else 1):
                self.buffers.append(self._allocate_buffer())
            # 这是临时工作区的顺序槽号，不存原表的全局 row_id。
            self.device_ids = torch.arange(chunk_rows, dtype=torch.int64, device=device)
            torch.cuda.current_stream(device=device).synchronize()
            if prefetch_layer_ids:
                self.prefetch_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="engram-lookup",
                )
        except Exception:
            self.close()
            raise

    def _init_query_backend(
        self,
        *,
        num_threads: int | None,
        thread_pool_queue_depth: int | None,
        io_uring_queue_depth: int | None,
    ) -> None:
        if self.query_backend == "python":
            return  # Python 后端保留原行缓存和串行读取，不创建原生 reader。

        # 只有显式选择 C++ 时才加载扩展；未构建或初始化失败均直接报错。
        from .query import NativeBatchRowReader, RowTableSpec

        tables = [
            RowTableSpec(
                layer_id=table.layer_id, rows=table.rows, dim=table.dim,
                weight_fd=table.weight_fd, scale_fd=table.scale_fd,
                weight_offset=table.weight_offset, scale_offset=table.scale_offset,
            )
            for table in self.tables.values()
        ]
        io_backend = {
            "cpp_thread_pool": "pread",
            "cpp_io_uring": "io_uring",
        }[self.query_backend]
        self.reader = NativeBatchRowReader(
            tables=tables,
            cache_bytes=self.cache_budget,
            # config 只解析当前后端适用的参数；其他字段为 None，不干扰后端构造。
            num_threads=num_threads,
            thread_pool_queue_depth=thread_pool_queue_depth,
            io_uring_queue_depth=io_uring_queue_depth,
            io_backend=io_backend,
        )

    def _allocate_buffer(self) -> EngramSsdBuffer:
        if self.reader is None:
            host_weight = torch.empty(
                (self.capacity, self.dim), dtype=torch.uint8, device="cpu", pin_memory=True
            )
            host_scale = torch.empty(
                (self.capacity, self.scale_dim),
                dtype=torch.uint8, device="cpu", pin_memory=True,
            )
        else:
            # catalog 已保证各层行宽相同，任一层分配的工作区都可跨层复用。
            host_weight, host_scale = self.reader.allocate_workspace(
                layer_id=self.catalog.layer_ids[0], capacity=self.capacity,
            )

        # 每套工作区新建自己的固定 gather 流，A/B 及模型主流之间互不共用。
        gather_stream = torch.cuda.Stream(device=self.device)
        if not self.pipeline_enabled:
            copy_stream = gather_stream
        elif self.buffers:
            copy_stream = self.buffers[0].copy_stream
        else:
            copy_stream = torch.cuda.Stream(device=self.device)
        return EngramSsdBuffer(
            host_weight=host_weight,
            host_scale=host_scale,
            device_weight=torch.empty(
                (self.capacity, self.dim), dtype=torch.uint8, device=self.device,
            ),
            device_scale=torch.empty(
                (self.capacity, self.scale_dim), dtype=torch.uint8, device=self.device,
            ),
            gather_stream=gather_stream,
            copy_stream=copy_stream,
            h2d_done=torch.cuda.Event(),
            gather_done=torch.cuda.Event(),
            lookup_done=torch.cuda.Event(),
        )

    def _open_shard(self, path: Path) -> int:
        if path not in self.fds:
            self.fds[path] = os.open(path, os.O_RDONLY)
        return self.fds[path]

    def init_table(self, layer_id: int) -> None:
        if layer_id in self.tables:
            return
        weight = self.catalog.views[f"model.layers.{layer_id}.engram.embed.weight"]
        scale = self.catalog.views[f"model.layers.{layer_id}.engram.embed.scale"]
        self.tables[layer_id] = EngramSsdTable(
            layer_id=layer_id, rows=weight.rows, dim=weight.width,
            row_bytes=weight.width + scale.width,
            weight_fd=self._open_shard(path=weight.path),
            scale_fd=self._open_shard(path=scale.path),
            weight_offset=weight.offset, scale_offset=scale.offset,
        )

    def open_table(self, layer_id: int) -> EngramSsdTable:
        if layer_id not in self.tables:
            raise KeyError(f"Engram SSD table for layer {layer_id} is not initialized")
        return self.tables[layer_id]

    def get_row(self, layer_id: int, row_id: int) -> bytes:
        # 仅 Python 分支持有 self.lock 时调用；原生后端不填充这份 Python LRU。
        table = self.open_table(layer_id=layer_id)
        if not 0 <= row_id < table.rows:
            raise IndexError(f"Engram layer {layer_id}: row {row_id} outside [0,{table.rows})")
        key = (str(self.catalog.root), layer_id, row_id)
        payload = self.cache.pop(key, None)
        if payload is not None:
            self.cache[key] = payload
            return payload

        # 两张表使用各自的原 shard、绝对起点和行宽；owner 筛选不改变全局行号。
        weight = read_exact_at(
            fd=table.weight_fd, size=table.dim,
            offset=table.weight_offset + row_id * table.dim,
        )
        scale_dim = table.dim // 32
        scale = read_exact_at(
            fd=table.scale_fd, size=scale_dim,
            offset=table.scale_offset + row_id * scale_dim,
        )
        payload = weight + scale  # 仅在有限 DDR 缓存中组合，不生成新文件。
        if table.row_bytes <= self.cache_budget:
            while self.cached_bytes + table.row_bytes > self.cache_budget:
                _, evicted = self.cache.popitem(last=False)
                self.cached_bytes -= len(evicted)
            self.cache[key] = payload
            self.cached_bytes += len(payload)
        return payload

    def lookup(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """返回本 rank 的 BF16 [T,C,D]；owner 模式下其他 rank 负责的位置为零。"""
        if indices.device != self.device or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram SSD indices must be int32/int64 on the runtime CUDA device")
        return self._lookup_impl(layer_id=layer_id, indices=indices)

    def _lookup_owner_allreduce(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """只查询本 rank 的行区间，将结果填回原查询位置；此处不进行通信。"""
        if self.tp_size == 1:
            return self._lookup_all_queries(layer_id=layer_id, indices=indices)

        table = self.open_table(layer_id=layer_id)
        flat_ids = indices.reshape(-1)
        # 与原 Engram 相同的连续行划分：不要求总行数整除 TP，也允许空区间。
        row_start = table.rows * self.tp_rank // self.tp_size
        row_end = table.rows * (self.tp_rank + 1) // self.tp_size
        if ((flat_ids < 0) | (flat_ids >= table.rows)).any().item():
            # 越界行不能被所有 owner 静默丢弃后变成零向量。
            raise IndexError(f"Engram layer {layer_id}: row IDs outside [0,{table.rows})")

        # positions 记录原 [T,C] 展平后的槽位；重复行也保留各自的查询位置。
        positions = torch.nonzero(
            (flat_ids >= row_start) & (flat_ids < row_end), as_tuple=False,
        ).flatten()
        out = torch.zeros(
            (*indices.shape, table.dim), dtype=torch.bfloat16, device=indices.device,
        )
        if positions.numel():
            owner_ids = flat_ids.index_select(0, positions)
            # 直接复用原查询：同一 reader、缓存、分块、H2D 和 A/B gather 流水。
            # owner_ids 仍是原表全局行号，不能减去 row_start。
            owner_values = self._lookup_all_queries(layer_id=layer_id, indices=owner_ids)
            out.view(-1, table.dim).index_copy_(0, positions, owner_values)

        # 包括零命中的清零操作，Future 完成前须保证回填后的局部结果已在 GPU 就绪。
        torch.cuda.current_stream(device=self.device).synchronize()
        return out

    def reduce_output(self, out: torch.Tensor) -> torch.Tensor:
        """在模型线程原 Engram 调用点合并各 owner，后台线程不提交 TP 通信。"""
        if self.lookup_mode != "owner_allreduce" or self.tp_size == 1 or out.numel() == 0:
            return out
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        # 无本地命中的 rank 也带完整零张量参加；使用返回值兼容非原地通信实现。
        return tensor_model_parallel_all_reduce(out)

    def _read_chunk(
        self,
        layer_id: int,
        row_ids: torch.Tensor,
        buffer: EngramSsdBuffer,
    ) -> None:
        """同步填满本块 Host 数据；三个读取后端共用后续的 CUDA 流水。"""
        if self.reader is not None:
            # C++ 直接写 pinned Tensor；返回时读取完成，不转 Python 列表。
            self.reader.read_rows(
                layer_id=layer_id,
                row_ids=row_ids,
                weight_out=buffer.host_weight,
                scale_out=buffer.host_scale,
            )
            return

        table = self.open_table(layer_id=layer_id)
        for slot, row_id in enumerate(row_ids.tolist()):
            payload = self.get_row(layer_id=layer_id, row_id=row_id)
            raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
            buffer.host_weight[slot].copy_(raw[:table.dim])
            buffer.host_scale[slot].copy_(raw[table.dim:])

    def _lookup_all_queries(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        # TP=1 和多 rank 使用同一路径：每个 rank 按全局行号完成全部查询。
        # 输出不按 TP 切片，也不做查表结果归约；CPU 后端共用后续分块流水。
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        table = self.open_table(layer_id=layer_id)
        out = torch.empty(
            (*indices.shape, table.dim), dtype=torch.bfloat16, device=indices.device
        )
        if indices.numel() == 0:
            return out

        with self.lock, torch.cuda.device(self.device):
            # 同步 D2H 以后，CPU 才根据真实 hash IDs 读取原文件。
            # 原模型的层切片可能不连续；原生接口要求连续 CPU int64 [Q]。
            ids_cpu = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1).contiguous()
            flat_out = out.view(-1, table.dim)
            buffer_count = len(self.buffers)
            try:
                for chunk, start in enumerate(range(0, ids_cpu.numel(), self.capacity)):
                    end = min(start + self.capacity, ids_cpu.numel())
                    query_count = end - start
                    buffer = self.buffers[chunk % buffer_count]
                    reused = chunk >= buffer_count

                    if self.pipeline_enabled and reused:
                        # CPU 只需等该槽 H2D 读完，即可填入新数据，无需等 GPU gather。
                        buffer.h2d_done.synchronize()
                    self._read_chunk(
                        layer_id=layer_id, row_ids=ids_cpu[start:end], buffer=buffer,
                    )

                    with torch.cuda.stream(buffer.copy_stream):
                        if self.pipeline_enabled and reused:
                            # GPU 等旧 gather 读完后再覆盖设备槽；这里不阻塞 CPU。
                            buffer.copy_stream.wait_event(buffer.gather_done)
                        buffer.device_weight[:query_count].copy_(
                            buffer.host_weight[:query_count], non_blocking=True,
                        )
                        buffer.device_scale[:query_count].copy_(
                            buffer.host_scale[:query_count], non_blocking=True,
                        )
                        if self.pipeline_enabled:
                            buffer.h2d_done.record(buffer.copy_stream)

                    with torch.cuda.stream(buffer.gather_stream):
                        if self.pipeline_enabled:
                            # 当前块搬运完成后才能 gather；其他块的 H2D 仍可并行。
                            buffer.gather_stream.wait_event(buffer.h2d_done)
                        engram_gather(
                            weight_ptr=buffer.device_weight.data_ptr(),
                            scale_ptr=buffer.device_scale.data_ptr(),
                            ids=self.device_ids[:query_count],
                            out=flat_out[start:end],
                            dim=table.dim,
                            block_size=32,
                            row_lo=0,
                            row_hi=query_count,
                        )
                        buffer.gather_done.record(buffer.gather_stream)
                    if not self.pipeline_enabled:
                        # 关闭流水：保持单缓冲、同流执行和逐块等待。
                        buffer.gather_done.synchronize()
                    # 开启流水时直接读下一块，CPU 读取可与已提交的 H2D/gather 重叠。
            finally:
                # A/B 各有 gather 流，必须分别收齐；共享拷贝流也要先完成。
                # 包括只提交了部分 H2D 就出错的情况，完成后才释放锁和工作区。
                for buffer in self.buffers:
                    if self.pipeline_enabled:
                        buffer.gather_stream.wait_stream(buffer.copy_stream)
                    buffer.lookup_done.record(buffer.gather_stream)
                for buffer in self.buffers:
                    buffer.lookup_done.synchronize()
        return out

    def close(self) -> None:
        # forward 退出会收齐预取任务；先关闭后台线程，再释放其借用的工作区和 fd。
        if self.prefetch_executor is not None:
            self.prefetch_executor.shutdown(wait=True)
            self.prefetch_executor = None
        with self.lock:
            # reader 借用模型 fd，必须先回收其线程池/ring，再关闭 runtime 持有的 fd。
            if self.reader is not None:
                self.reader.close()
                self.reader = None
            for fd in self.fds.values():
                os.close(fd)
            self.fds.clear()
            self.tables.clear()
            self.cache.clear()
            self.cached_bytes = 0
            self.buffers.clear()


@lru_cache(maxsize=1)
def get_engram_ssd_runtime(dim: int) -> EngramSsdRuntime:
    from sglang.srt.runtime_context import get_parallel

    config = get_config()
    parallel = get_parallel()
    catalog = get_engram_ssd_catalog()
    # 未指定层号时，预取全部 Engram 层；关闭时所有层仍按需调用原 SSD lookup。
    prefetch_layer_ids = (
        config.prefetch_layer_ids or catalog.layer_ids
    ) if config.prefetch_enabled else ()
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=dim,
        lookup_mode=config.lookup_mode,
        tp_rank=parallel.tp_rank,
        tp_size=parallel.tp_size,
        query_backend=config.query_backend,
        pipeline_enabled=config.pipeline_enabled,
        prefetch_layer_ids=prefetch_layer_ids,
        cache_bytes=config.cache_bytes, chunk_rows=config.chunk_rows,
        num_threads=config.num_threads,
        thread_pool_queue_depth=config.thread_pool_queue_depth,
        io_uring_queue_depth=config.io_uring_queue_depth,
        # SGLang 已绑定当前 worker 的 GPU；CUDA 可见编号不必等于 TP rank。
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    atexit.register(runtime.close)
    return runtime
