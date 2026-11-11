"""每个 rank 的 owner 查询、单套 CUDA 工作区与同步 io_uring 读取。"""

from __future__ import annotations

import atexit
import os
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from .catalog import EngramSsdCatalog, get_engram_ssd_catalog
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
    """单套中转区，仅在前一块 CUDA event 完成后才重新填充。"""

    host_weight: torch.Tensor
    host_scale: torch.Tensor
    device_weight: torch.Tensor
    device_scale: torch.Tensor


class EngramSsdRuntime:
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
    ) -> None:
        if tp_size <= 0 or not 0 <= tp_rank < tp_size:
            raise ValueError("Engram SSD requires a valid TP rank and size")
        self.catalog = catalog
        self.dim = dim
        self.scale_dim = dim // 32
        self.capacity = chunk_rows
        self.device = device
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tables: dict[int, EngramSsdTable] = {}
        self.fds: dict[Path, int] = {}
        self.lock = threading.Lock()
        self.reader: NativeBatchRowReader | None = None
        try:
            for layer_id in catalog.layer_ids:
                self.init_table(layer_id=layer_id)
            from .query import NativeBatchRowReader, RowTableSpec

            self.reader = NativeBatchRowReader(
                tables=[
                    RowTableSpec(
                        layer_id=t.layer_id, rows=t.rows, dim=t.dim,
                        weight_fd=t.weight_fd, scale_fd=t.scale_fd,
                        weight_offset=t.weight_offset, scale_offset=t.scale_offset,
                    )
                    for t in self.tables.values()
                ],
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
        except BaseException:
            self.close()
            raise

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
            weight_fd=self._open_shard(path=weight.path), scale_fd=self._open_shard(path=scale.path),
            weight_offset=weight.offset, scale_offset=scale.offset,
        )

    def open_table(self, layer_id: int) -> EngramSsdTable:
        if layer_id not in self.tables:
            raise KeyError(f"Engram layer {layer_id} is not initialized")
        return self.tables[layer_id]

    def lookup(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """返回 GPU BF16 [...,D] 局部结果，非 owner 的查询位置为零；不做通信。"""
        self._check_indices(indices=indices)
        table = self.open_table(layer_id=layer_id)
        flat_ids = indices.reshape(-1)
        if ((flat_ids < 0) | (flat_ids >= table.rows)).any().item():
            raise IndexError(f"Engram layer {layer_id}: row IDs outside [0,{table.rows})")
        if self.tp_size == 1:
            return self._read_rows_to_output(layer_id=layer_id, indices=indices)
        row_start = table.rows * self.tp_rank // self.tp_size
        row_end = table.rows * (self.tp_rank + 1) // self.tp_size
        positions = torch.nonzero((flat_ids >= row_start) & (flat_ids < row_end), as_tuple=False).flatten()
        out = torch.zeros((*indices.shape, table.dim), dtype=torch.bfloat16, device=indices.device)
        if positions.numel():
            values = self._read_rows_to_output(
                layer_id=layer_id, indices=flat_ids.index_select(0, positions),
            )
            out.view(-1, table.dim).index_copy_(0, positions, values)
        torch.cuda.current_stream(device=self.device).synchronize()
        return out

    def _check_indices(self, *, indices: torch.Tensor) -> None:
        if indices.device != self.device or indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("Engram SSD indices must be int32/int64 on the runtime CUDA device")

    def reduce_output(self, out: torch.Tensor) -> torch.Tensor:
        """仅在模型线程原 Engram 调用位置执行一次 TP 合并。"""
        if self.tp_size == 1 or out.numel() == 0:
            return out
        from sglang.srt.distributed import tensor_model_parallel_all_reduce

        return tensor_model_parallel_all_reduce(out)

    def _read_rows_to_output(self, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """同步读取给定全局行号；只使用一套工作区和调用方当前 CUDA 流。"""
        from sglang.kernels.ops.embeddings.engram_gather import engram_gather

        table = self.open_table(layer_id=layer_id)
        out = torch.empty((*indices.shape, table.dim), dtype=torch.bfloat16, device=self.device)
        if not indices.numel():
            return out
        with self.lock, torch.cuda.device(self.device):
            ids_cpu = indices.detach().to(device="cpu", dtype=torch.int64).reshape(-1).contiguous()
            if self.reader is None:
                raise RuntimeError("Engram SSD runtime is closed")
            buffer = self.buffer
            try:
                for start in range(0, ids_cpu.numel(), self.capacity):
                    end = min(start + self.capacity, ids_cpu.numel())
                    count = end - start
                    self.reader.read_rows(
                        layer_id=layer_id, row_ids=ids_cpu[start:end],
                        weight_out=buffer.host_weight, scale_out=buffer.host_scale,
                    )
                    buffer.device_weight[:count].copy_(buffer.host_weight[:count], non_blocking=True)
                    buffer.device_scale[:count].copy_(buffer.host_scale[:count], non_blocking=True)
                    engram_gather(
                        weight_ptr=buffer.device_weight.data_ptr(),
                        scale_ptr=buffer.device_scale.data_ptr(),
                        ids=self.device_ids[:count], out=out.view(-1, self.dim)[start:end],
                        dim=self.dim, block_size=32, row_lo=0, row_hi=count,
                    )
                    self._chunk_done.record(torch.cuda.current_stream(device=self.device))
                    self._chunk_done.synchronize()
            finally:
                # 包括异常路径，旧 GPU 使用结束前都不允许复用 host/device 工作区。
                self._chunk_done.record(torch.cuda.current_stream(device=self.device))
                self._chunk_done.synchronize()
        return out

    def close(self) -> None:
        with self.lock:
            if self.reader is not None:
                self.reader.close()
                self.reader = None
            for fd in self.fds.values():
                os.close(fd)
            self.fds.clear()
            self.tables.clear()


@lru_cache(maxsize=1)
def get_engram_ssd_runtime(dim: int) -> EngramSsdRuntime:
    from sglang.srt.runtime_context import get_parallel

    config = get_config()
    parallel = get_parallel()
    runtime = EngramSsdRuntime(
        catalog=get_engram_ssd_catalog(), dim=dim,
        tp_rank=parallel.tp_rank, tp_size=parallel.tp_size,
        chunk_rows=config.chunk_rows, io_uring_queue_depth=config.io_uring_queue_depth,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    atexit.register(runtime.close)
    return runtime
