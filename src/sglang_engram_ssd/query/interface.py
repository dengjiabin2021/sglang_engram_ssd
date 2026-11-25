"""Python 与 C++ 共用的 io_uring 行读取协议与借用文件句柄。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class RowTableSpec:
    """承载一层表的文件定位元数据，不读取或关闭 FD。

    runtime 创建并持有，reader 复制其字段；只读数据可跨线程借用，FD 须存活到所有 reader 关闭。
    字段对应原生 RowTableSpec，布局解释集中在 reader，调用方始终提交全局行号。
    """

    layer_id: int
    rows: int  # 全局行数，合法行号为 [0,rows)。
    dim: int  # weight 每行字节数 D；scale 每行 D/32 字节。
    weight_fd: int  # runtime 持有的只读描述符，可与 scale_fd 相同。
    scale_fd: int
    weight_offset: int  # weight 张量在文件中的绝对字节偏移。
    scale_offset: int  # scale 张量在文件中的绝对字节偏移。


@dataclass(frozen=True)
class PoolResolution:
    """承载一次 resolve 的查询位置与临时槽引用，不管理 GPU 执行。

    batch 组装方持有结果，handle 保持槽稳定，完成暂存区复制后显式 release。
    元数据张量由结果持有；调用方只读使用，不与其他线程修改同一结果。
    """

    slot_ids: torch.Tensor  # CPU int64 [Q]，未命中为 -1，重复行保留相同槽号。
    hit_mask: torch.Tensor  # CPU bool [Q]，仅 READY 行为 True。
    handle: Any  # 每个命中槽独立加一次引用；丢弃对象不能代替 release。


class BatchRowReader(Protocol):
    """定义同步读取协议，不规定文件布局之外的缓存或 CUDA 行为。

    runtime 创建具体实现并负责关闭；实例调用由包装锁串行保护，多个实例可共享只读 FD。
    所有输入和输出内存在一次读取返回前都由调用方保留，reader 不长期保存 Tensor 指针。
    """

    def __init__(
        self, tables: Sequence[RowTableSpec], *, io_uring_queue_depth: int = 32
    ) -> None:
        """创建独立的 io_uring ring。

        输入：tables 为各层 RowTableSpec，可为空；io_uring_queue_depth 为正数在途上限。
        输出：复制元数据并持有 ring；所有 FD 借用至 close 结束，初始化失败直接抛错。
        """
        ...

    def allocate_workspace(
        self, *, layer_id: int, capacity: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """为指定层创建顺序行工作区。

        输入：layer_id 为已登记层，capacity 为正数查询容量 Q。
        输出：连续 CPU pinned uint8 [Q,D]／[Q,D/32]，由调用方持有并保护 GPU 使用期间的复用。
        """
        ...

    def read_rows(
        self, *, layer_id: int, row_ids: torch.Tensor,
        weight_out: torch.Tensor, scale_out: torch.Tensor,
    ) -> None:
        """同步读取原始行字节，保留输入顺序及重复位置。

        输入：layer_id 为已登记层；row_ids 为连续 CPU int64 [Q]，行号属于 [0,rows)，可重复。
            weight_out／scale_out 为连续 CPU uint8 [容量,D]／[容量,D/32]，容量至少 Q。
        输出：仅填输出前 Q 行，保留原始 FP8/E8M0 字节；返回前全部 I/O 完成，失败输出不可用。
        不保留输入输出引用，不覆盖在途 GPU 使用的内存，不执行 TP 通信。
        """
        ...

    def close(self) -> None:
        """释放 ring 和元数据。

        输入：无；调用方须先停止读取。
        输出：reader 不再可读，重复关闭无效果，从不关闭 runtime 借出的 FD。
        """
        ...

    def read_rows_to_slots(
        self, *, layer_id: int, row_ids: torch.Tensor, slot_ids: torch.Tensor,
        weight_base: torch.Tensor, scale_base: torch.Tensor,
    ) -> None:
        """按指定槽号同步写入行池或调用方的 batch 暂存区。

        输入：layer_id 为已登记层；row_ids 为连续 CPU int64 [Q] 全局行号，可重复且属于 [0,rows)。
            slot_ids 为等长连续 CPU int64 [Q]，槽号互不重复且属于 [0,S)。
            weight_base／scale_base 为连续 CPU uint8 [S,D]／[S,D/32]，目标槽已被调用方预留。
        输出：只写指定槽；池槽在成功返回后由调用方 mark_ready，失败先收齐 I/O 再抛错。
        原生读取释放 GIL，行步长从 Tensor 读取；调用期间槽不能回收，也不能被 GPU 读取。
        """
        ...
