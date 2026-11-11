"""仅提供 io_uring 原生读取包装，扩展缺失或初始化失败直接报错。"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING

from .interface import RowTableSpec

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


def validate_native_reader(*, io_uring_queue_depth: int) -> None:
    """模型加载前仅初始化并关闭 ring，不分配模型或 GPU 工作区。"""
    reader = NativeBatchRowReader(tables=(), io_uring_queue_depth=io_uring_queue_depth)
    reader.close()
