"""Python 与 C++ 共用的 io_uring 行读取协议与借用文件句柄。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RowTableSpec:
    """全局行号的存储定位；FD 归 runtime，后续布局扩展不改变调用方行号。"""

    layer_id: int
    rows: int
    dim: int
    weight_fd: int
    scale_fd: int
    weight_offset: int
    scale_offset: int


class BatchRowReader(Protocol):
    def __init__(
        self, tables: Sequence[RowTableSpec], *, io_uring_queue_depth: int = 32
    ) -> None:
        """创建必需的 io_uring ring；深度为正，所有 FD 借用至 close 结束。"""
        ...

    def allocate_workspace(
        self, *, layer_id: int, capacity: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 CPU pinned uint8 [K,D]／[K,D/32]，由调用方持有和保护复用。"""
        ...

    def read_rows(
        self, *, layer_id: int, row_ids: torch.Tensor,
        weight_out: torch.Tensor, scale_out: torch.Tensor,
    ) -> None:
        """同步读取 CPU 连续 int64 [Q] 全局行号，保留输入顺序及重复位置。

        输出为 CPU 连续 uint8 [K,D]／[K,D/32]，Q<=K，仅填前 Q 行。
        原始 FP8/E8M0 字节不反量化；成功返回前全部 I/O 完成，失败抛错。
        不保留输入输出引用，不覆盖在途 GPU 使用的内存，不执行 TP 通信。
        """
        ...

    def close(self) -> None:
        """先结束读取再释放 ring；从不关闭 runtime 借出的 FD。"""
        ...
