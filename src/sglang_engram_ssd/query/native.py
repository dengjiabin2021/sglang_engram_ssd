"""供 runtime 按配置加载的原生批量读取薄封装，不自动编译扩展。"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from threading import Lock
from typing import TYPE_CHECKING

from .interface import RowTableSpec

if TYPE_CHECKING:
    import torch


class NativeBatchRowReader:
    def __init__(
        self,
        tables: Sequence[RowTableSpec],
        *,
        cache_bytes: int,
        num_threads: int | None = None,
        thread_pool_queue_depth: int | None = None,
        io_uring_queue_depth: int | None = None,
        io_backend: str = "pread",
    ) -> None:
        # 显式创建原生 reader 时才加载 Torch 和动态库，先让 Torch 初始化其类型和依赖。
        import torch  # noqa: F401

        try:
            native = import_module("._C", package=__package__)
        except ModuleNotFoundError as error:
            if error.name != f"{__package__}._C":
                raise
            raise RuntimeError(
                "Engram SSD native extension is not built; "
                "follow the build instructions in query/README.md"
            ) from error

        # 缓存只由 C++ 管理，Python 薄封装不再建立一份行缓存。
        self._reader = native.BatchRowReader(
            tables=tables,
            cache_bytes=cache_bytes,
            num_threads=num_threads,
            thread_pool_queue_depth=thread_pool_queue_depth,
            io_uring_queue_depth=io_uring_queue_depth,
            io_backend=io_backend,
        )
        self._lock = Lock()

    def allocate_workspace(
        self,
        *,
        layer_id: int,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 只在初始化工作区时调用，后续查询重复使用返回的两个 Tensor。
        with self._lock:
            return self._reader.allocate_workspace(layer_id=layer_id, capacity=capacity)

    def read_rows(
        self,
        *,
        layer_id: int,
        row_ids: torch.Tensor,
        weight_out: torch.Tensor,
        scale_out: torch.Tensor,
    ) -> None:
        # C++ 读取期间会释放 GIL；此锁只保护当前实例，避免并发 close 清空表信息。
        with self._lock:
            self._reader.read_rows(
                layer_id=layer_id,
                row_ids=row_ids,
                weight_out=weight_out,
                scale_out=scale_out,
            )

    def close(self) -> None:
        # 等当前读取结束后再关闭 reader，不关闭由 runtime 持有的 fd。
        with self._lock:
            self._reader.close()
