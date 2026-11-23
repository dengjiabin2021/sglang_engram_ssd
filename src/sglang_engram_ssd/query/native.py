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
    """加载必需原生扩展，不构建扩展或选择其他后端。

    输入：无；先导入 Torch，使扩展依赖的动态库可被解析。
    输出：缓存于 Python 导入系统的 `_C` 模块；模块不存在时抛 RuntimeError，ABI／依赖错误原样传播。
    """
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
    """串行保护一个原生读取器，不负责缓存、CUDA 操作或 FD 所有权。

    runtime 创建并持有实例，在全部调用结束后 close；多个实例可借用相同只读 FD 并发读取。
    `_reader` 独占一个 ring，`_lock` 串行保护该实例的调用和关闭，不限制其他实例。
    """

    def __init__(
        self, tables: Sequence[RowTableSpec], *, io_uring_queue_depth: int = 32
    ) -> None:
        """复制表定位信息并创建独立 ring。

        输入：tables 为各层元数据，可为空；io_uring_queue_depth 为该 ring 的正数在途上限。
        输出：初始化原生 reader 和包装锁；FD 仍由 runtime 持有，初始化失败直接抛错。
        """
        self._reader = load_native_module().BatchRowReader(
            tables=tables, io_uring_queue_depth=io_uring_queue_depth,
        )
        self._lock = Lock()

    def allocate_workspace(
        self, *, layer_id: int, capacity: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """为指定层分配复用的行工作区。

        输入：layer_id 必须已登记；capacity 为正数查询容量 Q。
        输出：连续 CPU pinned uint8 [Q,D] 与 [Q,D/32]，由调用方持有；复用前须等 GPU 使用结束。
        """
        with self._lock:
            return self._reader.allocate_workspace(layer_id=layer_id, capacity=capacity)

    def read_rows(
        self, *, layer_id: int, row_ids: torch.Tensor,
        weight_out: torch.Tensor, scale_out: torch.Tensor,
    ) -> None:
        """同步把全局行的原始字节写入顺序工作区。

        输入：layer_id 为已登记层；row_ids 为连续 CPU int64 [Q]，可重复，范围 [0,rows)。
            weight_out／scale_out 为连续 CPU uint8 [容量,D]／[容量,D/32]，容量至少 Q，不能重叠输入。
        输出：仅覆盖两个输出的前 Q 行，顺序与重复位置保持不变；返回后不再写入这些缓冲区。
        该实例锁覆盖整次读取；原生 I/O 释放 GIL，失败先收齐在途请求再抛错，输出内容不可用。
        """
        # 一套 ring 由同一实例串行使用；C++ 在等待读取时释放 GIL。
        with self._lock:
            self._reader.read_rows(
                layer_id=layer_id, row_ids=row_ids, weight_out=weight_out, scale_out=scale_out,
            )

    def close(self) -> None:
        """串行关闭原生 ring，不关闭借用 FD。

        输入：无；调用方须先结束该实例的生产者，不能在关闭后继续提交读取。
        输出：释放 ring 和元数据，重复关闭无效果；锁会等待同实例的在途调用退出。
        """
        with self._lock:
            self._reader.close()

    def read_rows_to_slots(
        self, *, layer_id: int, row_ids: torch.Tensor, slot_ids: torch.Tensor,
        weight_base: torch.Tensor, scale_base: torch.Tensor,
    ) -> None:
        """同步把缺失行写进调用方预留的 PENDING 池槽。

        输入：layer_id 为已登记层；row_ids 为连续 CPU int64 [Q] 全局行号，可重复，范围 [0,rows)。
            slot_ids 为等长连续 CPU int64 [Q]，互不重复且属于 [0,S)。
            weight_base／scale_base 为连续 CPU uint8 [S,D]／[S,D/32]，指定槽不得被 GPU 使用。
        输出：仅写指定槽的数据，不发布 READY；调用方在成功返回后 mark_ready，失败先收齐 I/O 再抛错。
        """
        with self._lock:
            self._reader.read_rows_to_slots(
                layer_id=layer_id, row_ids=row_ids, slot_ids=slot_ids,
                weight_base=weight_base, scale_base=scale_base,
            )


class NativePrefetchRowPool:
    """转发原生 pinned 行池操作，不在 Python 另建索引或持有模型 FD。

    runtime 创建并持有池；停止后台与 GPU 使用、归还所有句柄后显式 close。
    原生池在互斥锁下维护槽状态，可供调度、后台和 forward 并发调用；Python 不另设状态锁。
    返回的表视图和映射指针仅在池开放期间有效；调用方负责引用及 CUDA event 生命周期。
    """

    def __init__(self, *, dim: int, pool_bytes: int, track_stats: bool = False) -> None:
        """分配压缩行池及 GPU 映射地址。

        输入：dim 为正数行宽 D，且为 32 的倍数；pool_bytes 为 payload 预算，至少容纳一行。
            track_stats 决定是否累加运行计数，不控制槽状态维护。
        输出：创建原生池，容量 S=pool_bytes//(D+D/32)；分配／映射失败直接抛错，不回退。
        """
        self._pool = load_native_module().PrefetchRowPool(
            dim=dim, pool_bytes=pool_bytes, track_stats=track_stats,
        )

    @property
    def weight(self) -> torch.Tensor:
        """借用 weight 表视图。

        输入：无；调用方只能写已预留且无 GPU 使用的 PENDING 槽。
        输出：连续 CPU pinned uint8 [S,D]；Tensor 可持有内存，但槽有效性仍由池引用保护。
        """
        return self._pool.weight

    @property
    def scale(self) -> torch.Tensor:
        """借用 scale 表视图。

        输入：无；写入限制与 weight 相同，同一槽号对应同一行。
        输出：连续 CPU pinned uint8 [S,D/32]，不复制数据。
        """
        return self._pool.scale

    @property
    def weight_device_ptr(self) -> int:
        """取得 weight 表在 GPU 地址空间的映射起点。

        输入：无；池必须开放且 GPU 读取期间持有槽引用。
        输出：整数地址；它指向 CPU pinned 内存，不代表 HBM 副本。
        """
        return self._pool.weight_device_ptr

    @property
    def scale_device_ptr(self) -> int:
        """取得 scale 表在 GPU 地址空间的映射起点。

        输入：无；池必须开放且 GPU 读取期间持有槽引用。
        输出：整数地址；与 weight_device_ptr 使用同一槽号和各自行步长。
        """
        return self._pool.scale_device_ptr

    @property
    def capacity(self) -> int:
        """读取固定槽容量。

        输入：无。
        输出：创建时确定的正整数 S，不表示当前可回收槽数。
        """
        return self._pool.capacity

    @property
    def weight_stride(self) -> int:
        """读取 weight 行步长。

        输入：无。
        输出：相邻槽 weight 起点的字节距离，当前为 D。
        """
        return self._pool.weight_stride

    @property
    def scale_stride(self) -> int:
        """读取 scale 行步长。

        输入：无。
        输出：相邻槽 scale 起点的字节距离，当前为 D/32。
        """
        return self._pool.scale_stride

    def acquire(self, *, layer_id: int, row_ids: torch.Tensor) -> Any | None:
        """为一批全局行登记段引用，不执行 I/O。

        输入：layer_id 非负；row_ids 为连续 CPU int64 [Q]，行号非负，可重复；上限由 reader 校验。
        输出：句柄包含去重后的槽引用及 missing_row_ids／missing_slot_ids；池满返回 None 且状态不变。
        调用方读取缺失行并发布 READY，最终 release／cancel；失败槽或非法输入直接抛错。
        """
        return self._pool.acquire(layer_id=layer_id, row_ids=row_ids)

    def is_ready(self, *, handle: Any) -> bool:
        """非阻塞检查句柄持有的槽是否全部 READY。

        输入：本池 acquire／resolve 返回、尚未归还的 handle。
        输出：全部 READY 或空句柄为 True，否则 False；句柄非法、已归还或引用失败槽时抛错。
        """
        return self._pool.is_ready(handle=handle)

    def mark_ready(self, *, slot_ids: torch.Tensor) -> None:
        """在数据写完后原子发布一批槽。

        输入：连续 CPU int64 [Q] 槽号，范围 [0,S)，允许重复；须均为健康 PENDING，且 I/O 已结束。
        输出：PENDING 变 READY；无引用的槽进入 LRU。任一输入非法时整批状态不变并抛错。
        """
        self._pool.mark_ready(slot_ids=slot_ids)

    def release(self, *, handle: Any) -> None:
        """归还句柄引用，不取消在途 I/O。

        输入：本池 handle；GPU 临时引用须在对应 CUDA event 完成后归还。
        输出：各槽引用减一，无引用 READY 进入 LRU；PENDING 保留，重复归还无效果。
        """
        self._pool.release(handle=handle)

    def cancel(self, *, handle: Any) -> None:
        """归还句柄并撤销其新建、未发布的槽。

        输入：本池 handle；其 I/O 必须尚未提交或已经收齐，不能与写槽并发。
        输出：新建 PENDING 标为失败，无其他引用则回收；复用槽仅减引用，重复取消无效果。
        """
        self._pool.cancel(handle=handle)

    def resolve(self, *, layer_id: int, row_ids: torch.Tensor) -> PoolResolution:
        """解析 forward 行号，临时固定所有 READY 命中槽。

        输入：layer_id 非负；row_ids 为连续 CPU int64 [Q] 非负全局行号，可重复。
        输出：CPU int64 [Q] 槽号、CPU bool [Q] 命中掩码及独立句柄，保留查询位置；PENDING 不命中。
        每个命中槽只加一次引用；GPU gather 完成后调用方必须 release，不能只丢弃返回对象。
        """
        resolution = self._pool.resolve(layer_id=layer_id, row_ids=row_ids)
        return PoolResolution(
            slot_ids=resolution.slot_ids, hit_mask=resolution.hit_mask, handle=resolution.handle,
        )

    def stats(self) -> dict[str, int | float]:
        """在原生池锁内复制统计快照。

        输入：无；累计计数仅在 track_stats=True 时增长。
        输出：槽占用、累计计数和状态字典；占用为当前快照，时间单位为毫秒，不执行 I/O。
        """
        return self._pool.stats()

    def record_forward(self, *, hits: int, misses: int, miss_ms: float) -> None:
        """登记一次 forward 的查询位置和同步补读耗时。

        输入：hits／misses 为非负位置数，含重复查询；miss_ms 为非负有限毫秒数。
        输出：开启统计时累加对应计数，否则不累加；补读时间由调用方定义，不是纯 SSD 时间。
        """
        self._pool.record_forward(hits=hits, misses=misses, miss_ms=miss_ms)

    def record_submitted(self, *, rows: int) -> None:
        """登记实际交给 reader 的预取缺失行。

        输入：rows 为本次提交的非负行数，不包括仅占槽后取消的行。
        输出：统计开启时累加 submitted_rows；不影响行池状态或提交 I/O。
        """
        self._pool.record_submitted(rows=rows)

    def record_deferred(self, *, count: int) -> None:
        """登记本轮调度暂缓的请求数。

        输入：count 为非负请求数，同一请求可在不同轮重复计入。
        输出：统计开启时累加 deferred_requests，不改变 waiting 队列。
        """
        self._pool.record_deferred(count=count)

    def close(self) -> None:
        """在所有使用结束后关闭行池。

        输入：无；调用方须停止后台、收齐 I/O 与 GPU 使用并归还全部句柄。
        输出：释放池持有的表和索引，重复关闭无效果；仍有引用或 PENDING 时抛错，不关闭任何 FD。
        """
        self._pool.close()


def validate_native_reader(*, io_uring_queue_depth: int) -> None:
    """模型加载前探测必需扩展和 ring 是否可用。

    输入：io_uring_queue_depth 为待使用的正数队列深度。
    输出：创建并立即关闭空表 reader，不分配模型或 GPU 工作区；扩展／初始化错误直接传播。
    """
    reader = NativeBatchRowReader(tables=(), io_uring_queue_depth=io_uring_queue_depth)
    reader.close()
