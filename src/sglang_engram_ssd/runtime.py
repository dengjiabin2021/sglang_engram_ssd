"""每个 rank 一份的有限 DDR 行缓存和同步 CUDA 查表工作区。"""

from __future__ import annotations

import atexit
import os
import threading
from collections import OrderedDict
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


class EngramSsdRuntime:
    def __init__(
        self,
        catalog: EngramSsdCatalog,
        *,
        dim: int,
        lookup_mode: str,
        query_backend: str,
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
        self.query_backend = query_backend
        # 原生后端只创建一个 reader，所有 Engram 层共用其 C++ 缓存和读取资源。
        # Python 后端保持为 None，不导入或加载原生扩展。
        self.reader: NativeBatchRowReader | None = None
        # 配置已在模型加载前校验；构造时绑定查询实现，统一入口不处理具体查表逻辑。
        # 查询模式决定 rank 的查询范围，query_backend 决定 CPU 如何读取这些行。
        self._lookup_impl = {"all_queries": self._lookup_all_queries}[lookup_mode]
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
            self.device_weight = torch.empty((chunk_rows, dim), dtype=torch.uint8, device=device)
            self.device_scale = torch.empty(
                (chunk_rows, self.scale_dim), dtype=torch.uint8, device=device
            )
            # 这是临时工作区的顺序槽号，不存原表的全局 row_id。
            self.device_ids = torch.arange(chunk_rows, dtype=torch.int64, device=device)
            torch.cuda.current_stream(device=device).synchronize()
            self.done = torch.cuda.Event()
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
            # 保留原 Python 查缓存、逐行读取和填充 pinned 工作区的路径。
            self.host_weight = torch.empty(
                (self.capacity, self.dim), dtype=torch.uint8, device="cpu", pin_memory=True
            )
            self.host_scale = torch.empty(
                (self.capacity, self.scale_dim),
                dtype=torch.uint8, device="cpu", pin_memory=True,
            )
            return

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
        # 原 Python 串行读取已在上方返回；这里仅映射两个 C++ 并发后端。
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
        # catalog 已保证各层行宽相同，任选一层分配一次，两块 pinned Tensor 跨层复用。
        self.host_weight, self.host_scale = self.reader.allocate_workspace(
            layer_id=self.catalog.layer_ids[0], capacity=self.capacity,
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

        # 两张表使用各自的原 shard、绝对起点和行宽，不按 TP owner 过滤。
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
        """统一查询入口，返回保持输入顺序的 GPU BF16 [T,C,D] 张量。"""
        return self._lookup_impl(layer_id=layer_id, indices=indices)

    def _lookup_all_queries(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        # 每个 rank 独立处理全部查询；CPU 读取后端共用分块、传输和反量化流程。
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        table = self.open_table(layer_id=layer_id)
        if indices.device != self.device or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram SSD indices must be int32/int64 on the runtime CUDA device")
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
            stream = torch.cuda.current_stream(device=self.device)
            for start in range(0, ids_cpu.numel(), self.capacity):
                end = min(start + self.capacity, ids_cpu.numel())
                query_count = end - start
                # 不排序、不去重；每次查询独占一个槽位，天然保持输出顺序。
                if self.reader is None:
                    for slot, row_id in enumerate(ids_cpu[start:end].tolist()):
                        payload = self.get_row(layer_id=layer_id, row_id=row_id)
                        raw = torch.frombuffer(bytearray(payload), dtype=torch.uint8)
                        self.host_weight[slot].copy_(raw[:table.dim])
                        self.host_scale[slot].copy_(raw[table.dim:])
                else:
                    # 一次传入整块 ID，C++ 直接填充工作区前 q 行；返回时读取已全部完成。
                    # 不转 Python 列表，不经 bytearray/frombuffer，也不维护第二份行缓存。
                    self.reader.read_rows(
                        layer_id=layer_id,
                        row_ids=ids_cpu[start:end],
                        weight_out=self.host_weight,
                        scale_out=self.host_scale,
                    )

                try:
                    self.device_weight[:query_count].copy_(
                        self.host_weight[:query_count], non_blocking=True
                    )
                    self.device_scale[:query_count].copy_(
                        self.host_scale[:query_count], non_blocking=True
                    )
                    engram_gather(
                        weight_ptr=self.device_weight.data_ptr(),
                        scale_ptr=self.device_scale.data_ptr(),
                        ids=self.device_ids[:query_count],
                        out=flat_out[start:end],
                        dim=table.dim,
                        block_size=32,
                        row_lo=0,
                        row_hi=query_count,
                    )
                finally:
                    # 包括异常路径：所有已提交拷贝结束后才释放锁或覆盖工作区。
                    self.done.record(stream)
                    self.done.synchronize()
        return out

    def close(self) -> None:
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


@lru_cache(maxsize=1)
def get_engram_ssd_runtime(dim: int) -> EngramSsdRuntime:
    config = get_config()
    runtime = EngramSsdRuntime(
        catalog=get_engram_ssd_catalog(), dim=dim,
        lookup_mode=config.lookup_mode,
        query_backend=config.query_backend,
        cache_bytes=config.cache_bytes, chunk_rows=config.chunk_rows,
        num_threads=config.num_threads,
        thread_pool_queue_depth=config.thread_pool_queue_depth,
        io_uring_queue_depth=config.io_uring_queue_depth,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    atexit.register(runtime.close)
    return runtime
