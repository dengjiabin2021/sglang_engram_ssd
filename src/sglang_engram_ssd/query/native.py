"""仅提供 io_uring 原生读取包装，扩展缺失或初始化失败直接报错。"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .interface import PoolResolution, RowTableSpec

if TYPE_CHECKING:
    import torch


def load_native_module() -> ModuleType:
    import torch  # noqa: F401

    try:
        return import_module("._C", package=__package__)
    except ModuleNotFoundError as error:
        if error.name != f"{__package__}._C":
            raise
        raise RuntimeError(
            "Engram SSD native extension is not built; follow query/README.md"
        ) from error


class NativeBatchRowReader:
    def __init__(
        self, tables: Sequence[RowTableSpec], *, io_uring_queue_depth: int = 32
    ) -> None:
        self._reader = load_native_module().BatchRowReader(
            tables=tables, io_uring_queue_depth=io_uring_queue_depth,
        )
        self._lock = Lock()

    def allocate_workspace(
        self, *, layer_id: int, capacity: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._lock:
            return self._reader.allocate_workspace(layer_id=layer_id, capacity=capacity)

    def read_rows(
        self, *, layer_id: int, row_ids: torch.Tensor,
        weight_out: torch.Tensor, scale_out: torch.Tensor,
    ) -> None:
        # 一套 ring 由同一实例串行使用；C++ 在等待读取时释放 GIL。
        with self._lock:
            self._reader.read_rows(
                layer_id=layer_id, row_ids=row_ids, weight_out=weight_out, scale_out=scale_out,
            )

    def close(self) -> None:
        with self._lock:
            self._reader.close()

    def read_rows_to_slots(
        self, *, layer_id: int, row_ids: torch.Tensor, slot_ids: torch.Tensor,
        weight_base: torch.Tensor, scale_base: torch.Tensor,
    ) -> None:
        with self._lock:
            self._reader.read_rows_to_slots(
                layer_id=layer_id, row_ids=row_ids, slot_ids=slot_ids,
                weight_base=weight_base, scale_base=scale_base,
            )


class NativePrefetchRowPool:
    """原生互斥行池包装；不在 Python 另建索引，不持有或关闭模型 FD。"""

    def __init__(self, *, dim: int, pool_bytes: int, track_stats: bool = False) -> None:
        self._pool = load_native_module().PrefetchRowPool(
            dim=dim, pool_bytes=pool_bytes, track_stats=track_stats,
        )

    @property
    def weight(self) -> torch.Tensor:
        return self._pool.weight

    @property
    def scale(self) -> torch.Tensor:
        return self._pool.scale

    @property
    def weight_device_ptr(self) -> int:
        return self._pool.weight_device_ptr

    @property
    def scale_device_ptr(self) -> int:
        return self._pool.scale_device_ptr

    @property
    def capacity(self) -> int:
        return self._pool.capacity

    @property
    def weight_stride(self) -> int:
        return self._pool.weight_stride

    @property
    def scale_stride(self) -> int:
        return self._pool.scale_stride

    def acquire(self, *, layer_id: int, row_ids: torch.Tensor) -> Any | None:
        return self._pool.acquire(layer_id=layer_id, row_ids=row_ids)

    def is_ready(self, *, handle: Any) -> bool:
        return self._pool.is_ready(handle=handle)

    def mark_ready(self, *, slot_ids: torch.Tensor) -> None:
        self._pool.mark_ready(slot_ids=slot_ids)

    def release(self, *, handle: Any) -> None:
        self._pool.release(handle=handle)

    def cancel(self, *, handle: Any) -> None:
        self._pool.cancel(handle=handle)

    def resolve(self, *, layer_id: int, row_ids: torch.Tensor) -> PoolResolution:
        resolution = self._pool.resolve(layer_id=layer_id, row_ids=row_ids)
        return PoolResolution(
            slot_ids=resolution.slot_ids, hit_mask=resolution.hit_mask, handle=resolution.handle,
        )

    def stats(self) -> dict[str, int | float]:
        return self._pool.stats()

    def record_forward(self, *, hits: int, misses: int, miss_ms: float) -> None:
        self._pool.record_forward(hits=hits, misses=misses, miss_ms=miss_ms)

    def record_deferred(self, *, count: int) -> None:
        self._pool.record_deferred(count=count)

    def close(self) -> None:
        self._pool.close()


def validate_native_reader(*, io_uring_queue_depth: int) -> None:
    """模型加载前仅初始化并关闭 ring，不分配模型或 GPU 工作区。"""
    reader = NativeBatchRowReader(tables=(), io_uring_queue_depth=io_uring_queue_depth)
    reader.close()
