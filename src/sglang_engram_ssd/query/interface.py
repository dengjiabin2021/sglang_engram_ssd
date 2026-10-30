"""Python/C++ 共用的 CPU 批量行查询约定，不包含读取实现。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RowTableSpec:
    """由已校验的 catalog 和 runtime 表句柄提供的单层元数据。"""

    layer_id: int
    rows: int                    # 原表全局行数，合法行号范围为 [0, rows)。
    dim: int                     # weight 行宽；scale 行宽固定为 dim // 32。
    weight_fd: int               # 借用 runtime 的 fd，reader 不负责关闭。
    scale_fd: int
    weight_offset: int           # weight 张量在原文件中的绝对字节偏移。
    scale_offset: int            # scale 使用其自身文件中的绝对字节偏移。


class BatchRowReader(Protocol):
    """同步批量读取协议；每个 rank 的 reader 由其 runtime 持有。"""

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
        """各层共用缓存；pread 下 num_threads 缺省为 0，正数启用有界线程池。

        线程池只使用 thread_pool_queue_depth，限制等待的行任务数，缺省为 32。
        io_uring 只使用 io_uring_queue_depth，限制在途文件读取数，缺省为 32。
        仅校验当前后端使用的参数；显式传入无关参数时 warning 后忽略。
        cache_bytes 非负，0 时不缓存；实际使用的队列容量为正。fd 借用到关闭前。
        """
        ...

    def allocate_workspace(
        self,
        *,
        layer_id: int,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """分配一对可复用的 CPU uint8 pinned 输出，capacity 必须为正数。

        返回 [capacity,D] 和 [capacity,D/32]，由调用方持有，不计入行缓存预算。
        相同行宽的层可共用；每次覆盖前须确保此前 H2D 和 GPU 使用已经完成。
        """
        ...

    def read_rows(
        self,
        *,
        layer_id: int,
        row_ids: torch.Tensor,
        weight_out: torch.Tensor,
        scale_out: torch.Tensor,
    ) -> None:
        """将本批行数据写入调用方提供的 CPU 工作区。

        row_ids：连续 CPU int64 [Q]，保存当前层的全局行号。
        weight_out：连续 CPU uint8 [K,D]；scale_out：连续 CPU uint8 [K,D/32]。
        要求 Q <= K，两个输出的 K 相同，D 等于当前层 dim；输出允许使用 pinned 内存。

        仅写前 Q 行，第 i 个输出槽对应 row_ids[i]，重复行号保留各自的输出槽。
        成功返回时所有行已读满，返回后不再访问这些缓冲区；Q=0 时不读写数据。
        输出是 FP8 weight 和 E8M0 scale 的原始字节，不执行反量化或 CUDA 操作。
        未知层号、非法行号、I/O 错误或最终未读满均抛出异常，失败后的输出不可用。

        输入与两个输出的内存区间不得重叠。调用方负责串行调用同一 reader，
        并在调用期间保持输入、输出有效且不被复用。
        输出的后续 H2D 和 GPU 使用完成之前，调用方仍不得覆盖对应工作区。
        """
        ...

    def close(self) -> None:
        """释放 reader 自身资源；调用方先结束读取，再关闭 reader，最后关闭借用的 fd。"""
        ...
