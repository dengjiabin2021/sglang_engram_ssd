"""管理分段预取、批次组装、紧急读取及 TP 失败同步。"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from bisect import insort_right
from collections import deque
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import chain
from operator import attrgetter, itemgetter
from traceback import clear_frames
from typing import TYPE_CHECKING, Any

import torch

from .config import get_config

if TYPE_CHECKING:
    from .query.interface import BatchRowReader, PoolResolution
    from .runtime import Batch, EngramSsdRuntime

logger = logging.getLogger(__name__)

URGENT = 0
NEAR = 1
Priority = tuple[int, int, int, int]  # 级别、预测批次序号、Engram 层序号、批内顺序。
_Span = tuple[Any, int, int]  # 请求及绝对 token 区间。
_Window = list[tuple[Any, int, int, int]]  # 请求、绝对区间与从 1 开始的批次序号。
_Chunk = tuple[torch.Tensor, torch.Tensor, int]  # 普通段的行号、槽号、扫描终点。
_UrgentChunk = tuple[Priority, int, torch.Tensor, torch.Tensor]  # 优先级、层、行号、暂存下标。
_priority = attrgetter("priority")
_key = itemgetter(0)


@dataclass
class PrefetchSegment:
    """单个 token 区间的预取状态，不持有文件句柄或 GPU 输出。

    由控制器切段时创建并随请求状态持有；消费或取消后归还各层任务的引用。
    调度线程负责提交/回收，唯一后台线程负责读取；共享状态在控制器 condition 下发布。
    """

    start: int  # 完整 token 序列中的包含起点，单位 token。
    end: int  # 完整 token 序列中的不包含终点。
    status: str = "new"  # new/queued/reading/ready/blocked/cancelled。
    tasks: list[_ReadTask] = field(default_factory=list)  # 按 Engram 层顺序排列，各自持有池引用。
    row_ids: torch.Tensor | None = None  # CPU int64 [T,L,C] 全局行号，发起实际批次时可复用。
    owner_rows: tuple[torch.Tensor, ...] | None = None  # 池满时保留各层 CPU int64 [Q] owner 行号。

    @property
    def cancelled(self) -> bool:
        """无显式输入，返回整段是否已撤销；正在读取的块由读取线程收尾。"""
        return self.status == "cancelled"


@dataclass
class RequestPrefetchState:
    """一个请求的不可变 token 快照及分段引用，不修改原生 Req 的 KV 状态。

    由入队 Hook 创建，控制器持有至完成、撤回重登或对账清理；request 强引用保护对象身份。
    撤回重登时清理旧段和批次快照；共享状态由控制器 condition 保护。
    """

    request: Any  # 借用原生 Req 并持强引用，存活期间 id(request) 不会被复用。
    token_ids: tuple[int, ...]  # 原始输入与已确认输出组成的完整快照，不随原生列表修改。
    segments: list[PrefetchSegment] = field(default_factory=list)  # 按 start 排列的窗口段。


@dataclass(eq=False)
class _ReadTask:
    """保存一个普通预取层的池引用和读块位置。

    段持有任务直到回收，后台在 condition 外读取当前块，进度和取消在 condition 内发布。
    """

    layer_id: int
    priority: Priority
    owner: tuple[RequestPrefetchState, PrefetchSegment]
    handle: Any = None  # 普通段 acquire 的槽引用。
    rows: torch.Tensor | None = None  # 连续 CPU int64 [Q]，占槽前为 None。
    pos: int = 0  # 已扫描到的行位置。
    done: bool = False
    cancelled: bool = False


@dataclass(eq=False)
class UrgentReadRequest:
    """保存一次紧急组装的目标表、待写块和 READY 槽引用。

    控制器持有到组装结束。后台或接手补读的前台线程串行推进，condition 负责交接和通知就绪。
    """

    staging: tuple[torch.Tensor, torch.Tensor]  # CPU pinned uint8 [K,D] 和 [K,D/32]。
    chunks: deque[_UrgentChunk] = field(default_factory=deque)  # 按层顺序排列。
    hits: list[tuple[int, PoolResolution, torch.Tensor]] = field(default_factory=list)  # 层、READY 引用及目标下标。
    pending: list[int] = field(default_factory=list)  # 尚未完成组装的层，按层顺序排列。
    done: set[int] = field(default_factory=set)  # 已完成组装（有 finish 时已提交该层 H2D）的层。
    prepare: Callable[[], None] | None = None  # 首次写入前准备可用的暂存区。
    finish: Callable[[int], None] | None = None  # 某层主存数据就绪后提交该层 H2D。
    ready: bool = False  # 所有层都已完成组装。


_active_schedule: ContextVar[tuple[SchedulerPrefetchController, Any] | None] = ContextVar(
    "engram_ssd_schedule", default=None,
)


def reconcile_active_window(waiting_queue: list[Any], running_batch: Any = None) -> None:
    """在原生排序后，用当前 raw 调度的队列更新预取窗口。

    输入：waiting_queue 为当前 scheduler 持有的原生可变 Req 列表；running_batch 提供运行中请求数。
    输出：无返回值；有 prefill 需求时同步后台失败，原生队列只读。
    """
    active = _active_schedule.get()
    if active is not None:
        controller, scheduler = active
        if waiting_queue is scheduler.waiting_queue:
            # 清空本次 raw 上下文，重复的 priority 调用无需再次对账或通信；本轮紧接着组 prefill 批。
            _active_schedule.set(None)
            controller._reconcile_prefill(scheduler=scheduler, running_batch=running_batch, forming=True)


def get_manager(runtime: EngramSsdRuntime, scheduler: Any) -> SchedulerPrefetchController:
    """取得当前 rank 唯一的调度预取控制器。

    输入：runtime 为已加载 CPU Hasher 和行池的本 rank runtime；scheduler 提供 chunk 大小与 CPU TP 组。
    输出：由 runtime 持有的控制器；首次调用创建一个后台线程，关闭开关不进入此函数。
    必须由同一调度线程串行调用；runtime.close 负责停止控制器。
    """
    manager = getattr(runtime, "sched_prefetch", None)
    if manager is None:
        manager = SchedulerPrefetchController(runtime=runtime, scheduler=scheduler)
        runtime.sched_prefetch = manager
    return manager


class SchedulerPrefetchController:
    """管理本 rank 请求预取、后台失败同步及行池引用，不执行模型或改写原生调度状态。

    runtime 惰性创建并持有本对象，关闭时先 join 唯一后台线程，再由 runtime 释放池和 reader。
    调度入口由原生调度线程串行调用；后台计算 hash、读 SSD 和组装批次，condition 保护任务/状态发布。
    开启统计时，在同一个 condition 内更新等待、读取和窗口计数。
    TP 通信只在 raw 调度边界执行，每次最多一次 CPU MIN。
    """

    def __init__(self, runtime: EngramSsdRuntime, scheduler: Any) -> None:
        """绑定本 rank 资源并启动唯一后台线程。

        输入：runtime 借出 CPU Hasher、行池和独立 prefetch_reader；scheduler 提供 prefill 预算与 CPU TP 组。
        输出：无返回值；初始化请求状态、任务队列、统计和线程；缺少 Hasher/行池时抛 RuntimeError。
        不在此打开文件或建立第二个线程池，借用资源均由 runtime 关闭。
        """
        self.runtime = runtime
        self.config = get_config()
        if runtime.hasher is None or runtime.pool is None:
            raise RuntimeError("Engram scheduler prefetch requires CPU hashing parameters and a row pool")
        self.chunked_tokens = max(0, scheduler.chunked_prefill_size or 0)
        # 原生每批同时受 chunk 和总输入预算约束，窗口按两者较小值切批。
        # GPU／TP8 合成调度器只提供 chunk 大小，此时它也是唯一预算。
        max_prefill_tokens = getattr(scheduler, "max_prefill_tokens", self.chunked_tokens)
        self.prefill_tokens = min(self.chunked_tokens or max_prefill_tokens, max_prefill_tokens)
        self.cpu_group = scheduler.tp_group.cpu_group  # 借用已有 gloo 组，不在后台建立通信组。
        self.states: dict[int, RequestPrefetchState] = {}  # id(req) -> 状态；值持有 Req 强引用。
        self._condition = threading.Condition()  # 保护任务队列、失败标志、段状态和统计；锁内不做 I/O。
        self._jobs: list[_ReadTask] = []  # 按优先级键升序排列，当前块返回后重新取队首。
        self._urgent: list[tuple[Priority, UrgentReadRequest]] = []  # 按下一读块优先级排序。
        self._decodes: deque[tuple[Priority, Batch]] = deque()  # 按原生发起顺序登记。
        self._active: _ReadTask | None = None  # 正在写池槽的普通任务。
        self._active_urgent: UrgentReadRequest | None = None  # 正在组装的 prefill 或独立紧急请求。
        self._batches: dict[int, tuple[tuple[Any, int, int], ...]] = {}  # forward_iter -> (Req, start, end) 列表快照。
        self._failure: BaseException | None = None  # 后台首个错误，调度 MIN 后统一传播。
        self._stopping = False  # 关闭或后台失败后停止接收与读取新任务。
        self._prefill_count = 0  # 仅开启统计时累计，用于 LOG_INTERVAL。
        self._window_hits = self._window_rows = 0  # prefill 发起时已就绪和查询的唯一 owner 行数。
        self._prefill_wait_ms = self._decode_wait_ms = 0.0  # 前台等待和 decode 放行前的累计毫秒数。
        self._prefill_waits = self._decode_waits = 0  # 已返回的等待调用数和已发布的 decode 批次数。
        self._urgent_rows = 0  # prefill 和 decode 组装发现的唯一 owner 缺行数。
        self._predicted_spans = self._matched_spans = 0  # 实际非空区间数和与窗口完全相等的区间数。
        self._read_ms = 0.0  # 后台 reader 完成读块的累计毫秒数。
        self._read_rows = 0  # 后台 reader 成功写入池槽或暂存区的行数。
        self._idle_logged = False  # 每次请求生命周期结束后只记录一次空闲归零快照。
        self._closing = False  # 关闭时置位，decode 线程处理完已登记的批次后退出。
        self._thread = threading.Thread(target=self._worker, name="engram-ssd-prefetch", daemon=True)
        self._thread.start()
        # decode 每步只读几十行，用独立线程和 ring 立即读取，不排在后台预取块之后。
        self._decode_thread = threading.Thread(target=self._decode_worker, name="engram-ssd-decode", daemon=True)
        self._decode_thread.start()

    def _synchronize_failure(self) -> None:
        """以一次 CPU MIN 同步后台失败。

        输入：无；调用方只按原生 waiting／chunked 状态决定是否进入。
        输出：无返回值；任一 rank 失败则所有 rank 抛 RuntimeError。
        """
        with self._condition:
            failure = self._failure
        flag = torch.tensor([-1 if failure is not None else 1], dtype=torch.int32, device="cpu")
        if self.runtime.tp_size > 1:
            import torch.distributed as dist

            # 有 prefill 需求的各 rank 都归约一个失败标志。
            dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=self.cpu_group)
        if flag.item() < 0:
            raise RuntimeError("Engram scheduler prefetch failed on a TP rank") from failure

    @staticmethod
    def _valid_request(req: Any) -> bool:
        """判断请求是否属于可预取的存活纯文本主路径。

        输入：req 为借用的原生 Req，不修改字段。
        输出：bool；多模态、已经结束或已标记待终止的请求返回 False。
        """
        return (
            getattr(req, "multimodal_inputs", None) is None
            and getattr(req, "finished_reason", None) is None
            and getattr(req, "to_finish", None) is None
        )

    def register_request(self, req: Any, is_retracted: bool = False) -> None:
        """登记请求 token 快照，撤回重登时丢弃旧段和批次范围。

        输入：req 为已实际进入 waiting 的原生 Req；is_retracted 表示重新排队并刷新快照。
        输出：无返回值；保存原始输入加已确认输出的 CPU token 快照，不在此切段或读 SSD。
        同一调度线程调用；后台已失败时不提前抛错，由下一次 raw 调度 MIN 统一传播。
        """
        self._idle_logged = False
        key = id(req)
        if not self._valid_request(req=req):
            return
        existing = self.states.get(key)
        is_retracted = is_retracted or bool(getattr(req, "is_retracted", False))
        if existing is not None and not is_retracted:
            return
        if existing is not None:
            self._discard_state(state=existing)
        tokens = tuple(req.origin_input_ids) + tuple(req.output_ids)
        with self._condition:
            if self._stopping:
                # 后台错误统一交给调度边界的 MIN，避免入队 Hook 提前退出、其他 rank 仍继续模型通信。
                if self._failure is not None:
                    return
                raise RuntimeError("Engram scheduler prefetch is closed")
            self.states[key] = RequestPrefetchState(request=req, token_ids=tokens)

    def request_urgent(
        self, *, rows_by_layer: dict[int, torch.Tensor], slots_by_layer: dict[int, torch.Tensor],
        staging: tuple[torch.Tensor, torch.Tensor], priority: Priority, mapping: torch.Tensor | None = None,
        prepare: Callable[[], None] | None = None, finish: Callable[[int], None] | None = None,
    ) -> UrgentReadRequest:
        """登记紧急读取及组装，不等待 SSD 或 CUDA。

        输入：rows_by_layer 为各层 CPU int64 [Q] 唯一 owner 行号，slots_by_layer 为等长目标下标；
            staging 为连续 CPU pinned uint8 [K,D] 和 [K,D/32]，下标在 [0,K) 且跨层不重复。
            priority 以首层为基准；mapping 为可选的 CPU int32 [所有查询]，用于统计原查询位置。
            prepare 在写入前执行；finish(layer) 在该层写完后执行。目标内存由调用方保留至组装和 GPU 使用结束。
        输出：可通过 wait_urgent 等待的请求；只给 READY 池行加引用。
        """
        request = self._make_urgent(
            rows_by_layer=rows_by_layer, slots_by_layer=slots_by_layer,
            staging=staging, priority=priority, mapping=mapping,
        )
        request.prepare, request.finish = prepare, finish
        with self._condition:
            key = request.chunks[0][0] if request.chunks else priority
            insort_right(self._urgent, (key, request), key=_key)
            self._condition.notify_all()
        return request

    def _make_urgent(
        self, *, rows_by_layer: dict[int, torch.Tensor], slots_by_layer: dict[int, torch.Tensor],
        staging: tuple[torch.Tensor, torch.Tensor], priority: Priority, mapping: torch.Tensor | None,
    ) -> UrgentReadRequest:
        """固定 READY 槽，并把其它行按调用方给的下标切为读块。

        输入：行号、下标、暂存表、优先级和统计映射与 request_urgent 相同。
        输出：尚未排队的组装请求；READY 槽引用随请求持有，出错时归还已取引用。
        """
        pool = self.runtime.pool
        request = UrgentReadRequest(staging=staging)
        level, batch, first, order = priority
        layers = [
            (column, layer) for column, layer in enumerate(self.runtime.hasher.layer_ids) if layer in rows_by_layer
        ]
        offset = first - layers[0][0] if layers else first
        try:
            for column, layer in layers:
                rows, slots = rows_by_layer[layer].contiguous(), slots_by_layer[layer]
                resolved = pool.resolve(layer_id=layer, row_ids=rows)
                request.hits.append((layer, resolved, slots))
                request.pending.append(layer)
                misses = ~resolved.hit_mask
                rows, slots = rows[misses], slots[misses]
                for start in range(0, rows.numel(), self.config.chunk_rows):
                    end = start + self.config.chunk_rows
                    key = (level, batch, offset + column, order)
                    request.chunks.append((key, layer, rows[start:end], slots[start:end]))
            if self.config.log_interval and mapping is not None:
                # 最后一项留给映射 -1，统计按原查询位置计，包含重复行。
                hit = torch.zeros(staging[0].shape[0] + 1, dtype=torch.bool, device="cpu")
                for _, resolved, slots in request.hits:
                    hit[slots] = resolved.hit_mask
                count = int(hit[mapping.to(torch.int64)].sum())
                pool.record_forward(hits=count, misses=int((mapping >= 0).sum()) - count, miss_ms=0.0)
        except BaseException:
            self._release_hits(request=request)
            raise
        return request

    def _release_hits(self, *, request: UrgentReadRequest) -> None:
        """归还组装请求固定的 READY 引用。

        输入：request 的池读取已结束。输出：归还全部引用并清空来源，重复调用无效果。
        """
        for _, resolved, _ in request.hits:
            self.runtime.pool.release(handle=resolved.handle)
        request.hits.clear()

    def _read_urgent(self, *, request: UrgentReadRequest, reader: BatchRowReader) -> bool:
        """推进一个紧急读块；某层缺行全部到达后拷贝该层 READY 槽并结束该层组装。

        输入：request 由当前线程独占，reader 为后台或前台 ring；调用方不持有 condition。
        输出：所有层写好且 finish 都已调用时为 True。只在整块成功后前移，失败块可用前台 reader 重试。
        仅给后台 reader 成功的读块累计耗时和行数。
        """
        if request.prepare is not None:
            request.prepare()
            request.prepare = None
        weight, scale = request.staging
        pool = self.runtime.pool
        if request.chunks:
            _, layer, rows, slots = request.chunks[0]
            track = self.config.log_interval and reader is self.runtime.prefetch_reader
            if track:
                nrows = rows.numel()
                pool.record_submitted(rows=nrows)
                started = time.perf_counter()
            # 紧急缺行的目的地是调用方的 batch 暂存区，reader 返回前收齐这一块。
            reader.read_rows_to_slots(
                layer_id=layer, row_ids=rows, slot_ids=slots, weight_base=weight, scale_base=scale,
            )
            if track:
                elapsed = (time.perf_counter() - started) * 1000
                # 读块计时止于 reader 返回，汇总在 condition 内更新。
                with self._condition:
                    self._read_ms += elapsed
                    self._read_rows += nrows
            request.chunks.popleft()
        # 读块按层顺序排列；队首层已没有待读块时，该层即可单独组装和 H2D，后面的层继续读。
        while request.pending and not (request.chunks and request.chunks[0][1] == request.pending[0]):
            layer = request.pending[0]
            _, resolved, slots = request.hits[0]  # hits 与 pending 同序，队首即本层。
            hit = resolved.hit_mask
            source, target = resolved.slot_ids[hit], slots[hit]
            if source.numel():
                weight.index_copy_(0, target, pool.weight.index_select(0, source))
                scale.index_copy_(0, target, pool.scale.index_select(0, source))
            if request.finish is not None:
                request.finish(layer)
            # 该层暂存区已完整并已提交 H2D，归还这层的池引用。
            request.hits.pop(0)
            pool.release(handle=resolved.handle)
            request.pending.pop(0)
            request.done.add(layer)
        if request.pending:
            return False
        request.finish = None
        return True

    def wait_urgent(
        self, *, request: UrgentReadRequest, layer: int | None = None, timeout: float | None = None,
    ) -> bool:
        """等指定层（None 为全部层）组装发布；后台停用时由当前线程补读并完成组装。

        输入：request 为已提交的紧急请求，layer 为 Engram 层号，timeout 为正常等待秒数，None 表示无限等待。
        输出：就绪时为 True，等待超时为 False；同步补读失败向调用方抛出。
        同一前台线程调用；condition 保护就绪发布、交接和计数，补读及 H2D 在锁外执行。
        开启统计时记录返回前的整段耗时，包含等待和可能的前台补读。
        """
        track = self.config.log_interval
        if track:
            started = time.perf_counter()

        def ready_now() -> bool:
            """目标层已发布，或整个请求已完成。"""
            return request.ready or (layer is not None and layer in request.done)

        with self._condition:
            ready = self._condition.wait_for(
                lambda: ready_now() or (self._stopping and self._active_urgent is None and self._active is None),
                timeout=timeout,
            )
            if not ready or ready_now():
                if track:
                    self._prefill_wait_ms += (time.perf_counter() - started) * 1000
                    self._prefill_waits += 1
                return ready
            self._urgent = [entry for entry in self._urgent if entry[1] is not request]
        try:
            if track:
                read_started = time.perf_counter()
            while not self._read_urgent(request=request, reader=self.runtime.reader):
                pass
            if track:
                # 既有池计数继续单独记录前台补读部分。
                self.runtime.pool.record_forward(hits=0, misses=0, miss_ms=(time.perf_counter() - read_started) * 1000)
        finally:
            self._release_hits(request=request)
        with self._condition:
            request.ready = True
            self._condition.notify_all()
            if track:
                self._prefill_wait_ms += (time.perf_counter() - started) * 1000
                self._prefill_waits += 1
        return True

    def start_batch(self, *, batch: Any, iteration: int) -> None:
        """在 run_batch 前按原生 token 顺序准备纯 prefill，或给 decode 登记缓冲区。

        输入：batch 为原生 ScheduleBatch；iteration 为它即将取得的原生 forward_iter。
        输出：无返回值；复用窗口行号并提交组装。mixed 与未登记的 prefill 走同步查询。
        开启统计时，在撤销窗口段及 worker 取块前登记区间与紧急缺行。
        """
        runtime = self.runtime
        mode = batch.forward_mode
        if mode.is_decode():
            queries = len(batch.reqs) * len(runtime.hasher.layer_ids) * runtime.columns
            runtime.reserve_batch(batch=batch, queries=queries, rows=queries, decode=True)
            return
        if not mode.is_extend() or mode.is_mixed() or mode.is_split_prefill():
            return
        if not batch.extend_num_tokens or any(id(req) not in self.states for req in batch.reqs):
            return

        # 与 ScheduleBatch.prepare_for_extend 的 req 顺序及原生 Engram history 使用的前驱一致。
        spans = [
            (req, start, start + length)
            for req, start, length in zip(batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True)
        ]
        ids = torch.cat([
            self._row_ids(state=self.states[id(req)], token_ids=req.full_untruncated_fill_ids, start=start, end=end)
            for req, start, end in spans
        ])
        mapping, rows, slots, count = runtime.batch_rows(ids=ids)
        current = runtime.reserve_batch(batch=batch, queries=ids.numel(), rows=count, decode=False)
        if current is None:
            return
        if runtime.debug_check_rows:
            current.expected = ids
        _, weight, scale = runtime.batch_host(batch=current)
        with self._condition:
            # 先固定 READY 行，再归还窗口引用；后台取到紧急任务前已撤销重叠读块。
            current.request = self.request_urgent(
                rows_by_layer=rows, slots_by_layer=slots, staging=(weight, scale),
                priority=(URGENT, iteration, 0, 0), mapping=mapping,
                prepare=lambda: runtime.stage_prefill(batch=current, mapping=mapping),
                finish=lambda layer: runtime.copy_layer(batch=current, layer=layer, slots=slots[layer]),
            )
            # 按层记录本 rank 的紧急缺行数，计时日志据此区分全命中和需要紧急读的 forward。
            for _, layer, rows, _ in current.request.chunks:
                current.urgent[layer] = current.urgent.get(layer, 0) + rows.numel()
            if self.config.log_interval:
                missing = sum(rows.numel() for _, _, rows, _ in current.request.chunks)
                self._urgent_rows += missing
                self._window_rows += count
                self._window_hits += count - missing
                # 统计实际非空区间，匹配只要求同请求的窗口段仍有效且边界完全相等。
                for req, start, end in spans:
                    if start < end:
                        self._predicted_spans += 1
                        self._matched_spans += any(
                            not segment.cancelled and (segment.start, segment.end) == (start, end)
                            for segment in self.states[id(req)].segments
                        )
            self._cancel_spans(spans=spans)

    def submit_decode(self, *, batch: Batch, iteration: int) -> bool:
        """把已提交旁路 hash D2H 的 decode 交给独立的 decode 线程。

        输入：batch 持有 D2H 和 H2D 事件，iteration 为原生批次序号。
        输出：已排队为 True；控制器关闭时为 False，由当前 forward 补读。
        后台预取失败不影响 decode 线程，它使用自己的 reader。
        """
        with self._condition:
            if self._closing:
                return False
            self._decodes.append(((URGENT, iteration, 0, 0), batch))
            self._condition.notify_all()
            return True

    def read_decode(self, *, batch: Batch, retry: bool = False) -> None:
        """用 GPU hash 拷回的行号组装 decode，decode reader 失败时用前台 reader 补齐后释放 GPU 等待。

        输入：batch 包含 CPU pinned int64 [T,L,C] hash 及两条流的事件；
            retry 表示直接使用前台 reader。
        输出：填满暂存区后放行 H2D。补读也失败时直接抛出异常。
        开启统计时记录唯一 owner 缺行和 hash 就绪至放行 H2D 的耗时。
        """
        runtime = self.runtime
        mapping_out, weight, scale = runtime.batch_host(batch=batch)
        request = None
        track = self.config.log_interval
        try:
            batch.hash_done.synchronize()
            if track:
                # 拷贝流也依赖此事件，D2H 完成前的时间不计入 Engram 放行耗时。
                started = time.perf_counter()
            mapping, rows, slots, _ = runtime.batch_rows(ids=batch.hash_ids)
            mapping_out.copy_(mapping)
            request = self._make_urgent(
                rows_by_layer=rows, slots_by_layer=slots, staging=(weight, scale),
                priority=(URGENT, 0, 0, 0), mapping=mapping,
            )
            if track:
                missing = sum(rows.numel() for _, _, rows, _ in request.chunks)
                with self._condition:
                    self._urgent_rows += missing
            if not retry:
                try:
                    # decode 专用 ring：池中 READY 行直接复制，缺行立即读取，不与后台预取排队。
                    while not self._read_urgent(request=request, reader=runtime.decode_reader):
                        pass
                except BaseException:
                    logger.exception("Engram SSD decode reader failed on rank %s; retrying", runtime.tp_rank)
                    retry = True
            if retry:
                # 计算流可能已经排入等待；用前台 ring 同步补读，保证它可以完成。
                if track:
                    read_started = time.perf_counter()
                while not self._read_urgent(request=request, reader=runtime.reader):
                    pass
                if track:
                    runtime.pool.record_forward(
                        hits=0, misses=0, miss_ms=(time.perf_counter() - read_started) * 1000,
                    )
        finally:
            if request is not None:
                self._release_hits(request=request)
        runtime.publish_decode(batch=batch)
        if track:
            # H2D 标志已经置位，拷贝流可继续，未等待拷贝完成。
            elapsed = (time.perf_counter() - started) * 1000
            with self._condition:
                self._decode_wait_ms += elapsed
                self._decode_waits += 1

    def _worker_decode(self, *, batch: Batch, retry: bool = False) -> None:
        """在 decode 线程处理 batch；retry 表示只可使用前台 ring。

        输出：正常时放行 H2D。最终异常留在当前线程报告并退出本 rank，避免计算流挂起。
        """
        try:
            self.read_decode(batch=batch, retry=retry)
        except BaseException:
            # daemon 线程单独退出无法结束 rank，计算流和其他 rank 可能还在等本批。
            logger.exception("Engram SSD decode read failed on rank %s", self.runtime.tp_rank)
            os._exit(1)

    @torch.inference_mode()
    def _decode_worker(self) -> None:
        """decode 线程按登记顺序逐批读取并放行 H2D，关闭时处理完已登记的批次再退出。

        输入：无；使用 runtime.decode_reader，不与后台预取线程共用 ring 或队列。
        输出：无返回值；每批读完后置位流标志，计算流据此继续。
        """
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._decodes or self._closing)
                if not self._decodes:
                    return
                _, batch = self._decodes.popleft()
            self._worker_decode(batch=batch)

    def _row_ids(
        self, *, state: RequestPrefetchState, token_ids: Sequence[int], start: int, end: int,
    ) -> torch.Tensor:
        """取得 CPU int64 [T,L,C] 全局行号；有同区间窗口段时保留计算结果。

        输入：state 为当前请求快照，token_ids 为同一请求完整输入，start/end 为绝对区间。
        输出：只读行号；仅精确匹配窗口区间时复用，hash 在 condition 锁外计算。
        """
        with self._condition:
            segment = next((s for s in state.segments if not s.cancelled and (s.start, s.end) == (start, end)), None)
            if segment is not None and segment.row_ids is not None:
                return segment.row_ids
        ids = self.runtime.hasher.compute_request_row_ids(token_ids=token_ids, start=start, end=end)
        if segment is not None:
            with self._condition:
                if not segment.cancelled:
                    segment.row_ids = ids
        return ids

    def _owner_rows(self, *, ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """把 CPU int64 [T,L,C] 全局行号拆成各层本 rank 的 [Q] 行号，保留重复项。"""
        rows_by_layer = []
        for column, layer in enumerate(self.runtime.hasher.layer_ids):
            table = self.runtime.open_table(layer_id=layer)
            lo = table.rows * self.runtime.tp_rank // self.runtime.tp_size
            hi = table.rows * (self.runtime.tp_rank + 1) // self.runtime.tp_size
            rows = ids[:, column].reshape(-1)
            rows_by_layer.append(rows[(rows >= lo) & (rows < hi)])
        return tuple(rows_by_layer)

    def cancel_task(self, *, task: _ReadTask) -> None:
        """通过一个普通层任务撤销其所属段。

        输入：task 为本控制器中的层任务，可由段的 tasks 取得。
        输出：无返回值；移除任务并归还引用，在途块由读取线程收尾，同槽的其它任务仍可读取。
        """
        with self._condition:
            if not task.owner[1].cancelled:
                self.cancel_segment(segment=task.owner[1])
                return
            task.cancelled = True
            if task in self._jobs:
                self._jobs.remove(task)
            if task is not self._active:
                self._drop_task(task=task)

    def _drop_task(self, *, task: _ReadTask) -> None:
        """归还普通任务的行池引用。

        输入：task 为已撤销或回滚占槽的任务，持有 condition 且任务当前没有在途写入。
        输出：无返回值；最后一个引用归还的 PENDING 可回收，共享槽仍可读；清空行号。
        """
        if task.handle is not None:
            # 普通段共享的未读槽仍可由其他段读取，最后一个段退出时才回收。
            self.runtime.pool.release(handle=task.handle, discard_pending=True)
            task.handle = None
        task.rows = None

    def cancel_segment(self, *, segment: PrefetchSegment) -> None:
        """撤销一个 token 区间的全部层任务。

        输入：segment 为本控制器切出的段。输出：无返回值；标记整段撤销、清除行号缓存并归还引用。
        正在写入的块由读取线程收尾，本函数不等待 I/O。
        """
        with self._condition:
            # 先标记段撤销，随后各层直接归还任务引用。
            segment.status = "cancelled"
            for task in segment.tasks:
                self.cancel_task(task=task)
            segment.tasks.clear()
            segment.row_ids = None
            segment.owner_rows = None
            self._condition.notify_all()

    def _cancel_spans(self, *, spans: Sequence[_Span]) -> None:
        """撤销与 spans 中实际计算或紧急补读区间重叠的窗口段。

        输入：spans 为请求和绝对 token 区间，调用方持有 condition。
        输出：无返回值；整段撤销，下轮窗口重新推算尚未消费的区间。
        """
        for req, start, end in spans:
            if start == end:
                continue
            for segment in self.states[id(req)].segments:
                if not segment.cancelled and segment.start < end and segment.end > start:
                    self.cancel_segment(segment=segment)

    def _discard_state(self, state: RequestPrefetchState) -> None:
        """丢弃请求及它在尚未处理的批次中的 token 区间。

        输入：state 为当前已登记请求，调度线程串行调用。
        输出：无返回值；取消全部段，移除索引和旧批次中的对应范围，保留同批其它请求。
        """
        with self._condition:
            for segment in state.segments:
                self.cancel_segment(segment=segment)
            del self.states[id(state.request)]
        # 撤回重登使用同一 Req；旧 batch 的结果不应触碰新登记的状态。
        for iteration, spans in list(self._batches.items()):
            kept = tuple(span for span in spans if span[0] is not state.request)
            if kept:
                self._batches[iteration] = kept
            else:
                del self._batches[iteration]

    def _submit(self, state: RequestPrefetchState, segment: PrefetchSegment, priority: Priority) -> None:
        """更新窗口段的优先级，并把可占槽的新段或待重试段各层加入后台队列。

        输入：state 为请求快照，segment 为其中一段；priority 是首层的键，后续层递增层序号。
        输出：无返回值；池满段仅在可用槽足够时重试。调用方持有 condition，统一排序后唤醒后台。
        """
        if segment.status == "blocked":
            # 其它段可能已固定共享行，按当前引用和可淘汰缓存重新计数。
            available = self.runtime.pool.available_slots()
            if self._slots_needed(segment=segment, limit=available) > available:
                return
        if not segment.tasks:
            owner = (state, segment)
            segment.tasks = [
                _ReadTask(layer_id=layer, priority=priority, owner=owner) for layer in self.runtime.hasher.layer_ids
            ]
        level, batch, first, order = priority
        for column, task in enumerate(segment.tasks):
            task.priority = (level, batch, first + column, order)
        if segment.status in ("new", "blocked"):
            self._jobs.extend(segment.tasks)
            segment.status = "queued"

    def _slots_needed(self, *, segment: PrefetchSegment, limit: int = -1) -> int:
        """按段已缓存的行号计算需要多少可用槽。

        输入：segment 持有各层连续 CPU int64 [Q] owner 行号，可重复；调用方持有 condition。
            limit 为非负槽数时允许提前结束，-1 计算完整需求。
        输出：各层需新占或保留的缓存槽数；超过 limit 时只需返回已计出的超额数。
        """
        needed = 0
        for task, rows in zip(segment.tasks, segment.owner_rows, strict=True):
            needed += self.runtime.pool.slots_needed(
                layer_id=task.layer_id, row_ids=rows, limit=limit - needed if limit >= 0 else -1,
            )
            if limit >= 0 and needed > limit:
                break
        return needed

    def _reserve_segment(self, *, state: RequestPrefetchState, segment: PrefetchSegment) -> None:
        """计算一个普通段各层的 owner 行号并为全部层占槽。

        输入：state 提供完整 CPU token 快照，segment 提供 [start,end) 和可能缓存的行号。
        输出：无返回值；各层任务持有自己的槽。容量不足时回滚并记录所需槽数，保留行号供重试。
        唯一后台线程调用；hash 在 condition 外做，占槽和发布在锁内完成。
        """
        pool = self.runtime.pool
        with self._condition:
            rows_by_layer = segment.owner_rows
        if rows_by_layer is None:
            # CPU int64 [T,L,C] -> 各层 [T*C]，仅保留本 rank 的全局行号。
            ids = self._row_ids(
                state=state, token_ids=state.token_ids, start=segment.start, end=segment.end,
            )
            rows_by_layer = self._owner_rows(ids=ids)
        with self._condition:
            if self._stopping or segment.cancelled:
                return
            for task, rows in zip(segment.tasks, rows_by_layer, strict=True):
                # 先占齐各层再发起读盘，池满的段不会留下部分槽长期占用。
                task.handle = pool.acquire(layer_id=task.layer_id, row_ids=rows)
                if task.handle is None:
                    for reserved in segment.tasks:
                        self._drop_task(task=reserved)
                        self._jobs.remove(reserved)
                    segment.owner_rows = rows_by_layer
                    segment.status = "blocked"
                    self._condition.notify_all()
                    return
                task.rows = task.handle.row_ids
            segment.owner_rows = None

    def _next_chunk(self, *, task: _ReadTask) -> _Chunk | None:
        """选出普通段下一块需要读取的池行。

        输入：task 已持有段的池槽引用，调用方持有 condition。
        输出：连续 CPU int64 [Q] 行号与槽号、扫描终点；已就绪的共享槽跳过，扫描完返回 None。
        """
        limit = self.config.chunk_rows
        while task.pos < task.rows.numel():
            end = min(task.pos + limit, task.rows.numel())
            rows = task.rows[task.pos:end]
            # 普通段可能共享 PENDING，按当前状态跳过其他段已发布的行。
            slots = self.runtime.pool.pending_slots(layer_id=task.layer_id, row_ids=rows)
            pending = slots >= 0
            if pending.any():
                return rows[pending], slots[pending], end
            task.pos = end
        return None

    def _read_chunk(self, *, task: _ReadTask, chunk: _Chunk) -> None:
        """读取并发布普通段的一个块，然后交出读取权。

        输入：task 为当前普通层任务，chunk 提供行号、池槽号和扫描终点。
        输出：无返回值；I/O 成功后发布 READY 并更新进度，撤销任务等在途读取结束再归还槽。
        开启统计时只累计成功的后台 reader 调用，计时不含 READY 发布。
        """
        rows, slots, end = chunk
        track = self.config.log_interval
        complete = False
        try:
            pool = self.runtime.pool
            if track:
                nrows = rows.numel()
                pool.record_submitted(rows=nrows)
                started = time.perf_counter()
            # 读取期间不持 condition，reader 收齐本块 I/O 后才允许发布或归还。
            self.runtime.prefetch_reader.read_rows_to_slots(
                layer_id=task.layer_id, row_ids=rows, slot_ids=slots, weight_base=pool.weight, scale_base=pool.scale,
            )
            if track:
                elapsed = (time.perf_counter() - started) * 1000
            complete = True
        finally:
            with self._condition:
                try:
                    if complete:
                        if track:
                            self._read_ms += elapsed
                            self._read_rows += nrows
                        pool.mark_ready(slot_ids=slots)
                        task.pos = end
                finally:
                    self._active = None
                    if task.cancelled:
                        self._drop_task(task=task)
                    self._condition.notify_all()

    def _stop_failed(self, *, error: BaseException) -> None:
        """保留后台异常并停止普通预取和后台紧急读。

        输入：error 为本 rank 的首个后台错误，当前读块已收齐。输出：发布失败并唤醒前台补读。
        """
        clear_frames(error.__traceback__)
        with self._condition:
            self._failure = error
            self._stopping = True
            self._jobs.clear()
            for state in self.states.values():
                for segment in state.segments:
                    if segment.status == "reading":
                        self.cancel_segment(segment=segment)
            self._condition.notify_all()

    @torch.inference_mode()
    def _worker(self) -> None:
        """唯一后台预取线程按优先级读取，每个普通块或紧急 prefill 块之后重新选队首。

        输入：无；使用后台 reader。decode 由独立线程处理，不在此队列中。
        输出：发布行和组装就绪，或记录后台错误并交由前台 reader 补读；不执行 TP 通信。
        """
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._stopping or self._jobs or self._urgent)
                    if self._stopping:
                        break
                    urgent = self._urgent[0] if self._urgent else None
                    if urgent is not None:
                        self._active_urgent = urgent[1]
                    else:
                        task = self._jobs[0]
                        prepare = task.handle is None
                        _, segment = task.owner
                        if prepare:
                            segment.status = "reading"
                        else:
                            chunk = self._next_chunk(task=task)
                            if chunk is None:
                                self._jobs.pop(0)
                                task.done = True
                                if all(t.done for t in segment.tasks):
                                    segment.status = "ready"
                                self._condition.notify_all()
                                continue
                            self._active = task
                if urgent is not None:
                    _, request = urgent
                    complete = False
                    try:
                        complete = self._read_urgent(request=request, reader=self.runtime.prefetch_reader)
                    finally:
                        with self._condition:
                            if complete:
                                self._urgent.remove(urgent)
                            elif request.chunks and request.chunks[0][0] != urgent[0]:
                                self._urgent.remove(urgent)
                                insort_right(self._urgent, (request.chunks[0][0], request), key=_key)
                            request.ready = complete
                            self._active_urgent = None
                            self._condition.notify_all()
                elif prepare:
                    state, segment = task.owner
                    self._reserve_segment(state=state, segment=segment)
                else:
                    self._read_chunk(task=task, chunk=chunk)
        except BaseException as error:
            self._stop_failed(error=error)

    def _request_slots(self, *, scheduler: Any, running_batch: Any) -> int:
        """估算每一预测批最多放几个请求。

        输入：scheduler 提供运行中请求上限；running_batch 保留原签名，不参与估算。
        输出：max_running_requests。运行中请求往往成批结束，下一批可能一次放满；
            真正限制每批大小的是 token 预算，按当前空槽估算会只预取队首一两个请求。
        """
        return scheduler.max_running_requests

    def _plan_window(self, *, waiting: Sequence[Any], chunked: Any, slots: int, batches: int) -> _Window:
        """按原生队列顺序推算未来 prefill 批次，保留截断请求的首位。

        输入：waiting 为排序后的原生请求，chunked 为延续请求或 None；slots/batches 限制每批请求数和批数。
        输出：(请求, start, end, 批次序号) 列表；依当前 prefix 估算，无副作用。
        """
        requests = chain((chunked,) if chunked is not None else (), waiting)
        spans = (
            (req, req.prefix_len, len(req.origin_input_ids) + len(req.output_ids))
            for req in requests
        )
        current = next(spans, None)
        window = []
        for batch in range(1, batches + 1):
            budget = self.prefill_tokens
            for _ in range(slots):
                while current is not None and current[1] >= current[2]:
                    current = next(spans, None)
                if current is None:
                    return window
                req, start, stop = current
                # 不启用分块时保留完整请求；原生允许首个请求独占一批并超出 token 预算。
                if not self.chunked_tokens and budget < self.prefill_tokens and stop - start > budget:
                    break
                end = min(stop, start + budget) if self.chunked_tokens else stop
                window.append((req, start, end, batch))
                budget -= end - start
                current = (req, end, stop) if end < stop else next(spans, None)
                if budget <= 0:
                    break
        return window

    def reconcile(self, scheduler: Any, running_batch: Any = None, forming: bool = False) -> None:
        """按本轮队列刷新预取窗口。

        输入：scheduler 借出 waiting_queue 和 chunked_req，running_batch 提供当前运行请求数；
            forming 表示本轮紧接着要组 prefill 批（calc_priority 入口），该批不再新建预取段。
        输出：无返回值；清理离队请求，撤销区间变化或移出窗口的段，再按预测优先级排队。
            预取的始终是尚未开始组批的前 WINDOW_BATCHES 个 prefill 批；正在组的批只保留已有段，缺行由紧急读补齐。
        统计开启时，等待运行批排空后再记录空闲快照，以收齐末尾 decode 计数。
        调度线程调用，只更新元数据，不等待 I/O 或发起 TP 通信。
        """
        with self._condition:
            if self._failure is not None:
                return
        waiting, chunked = scheduler.waiting_queue, scheduler.chunked_req
        active = {id(req) for req in waiting}
        if chunked is not None:
            active.add(id(chunked))
        inflight = {id(req) for spans in self._batches.values() for req, _, _ in spans}
        # waiting/chunked 之外的请求也可能仍有 overlap batch 在途，须保留至对应结果到达再清理。
        # 已结束/中止请求可丢弃段引用，组装方独立持有需要复制的 READY 引用。
        for key, state in list(self.states.items()):
            if not self._valid_request(req=state.request) or (key not in active and key not in inflight):
                self._discard_state(state=state)
        skip = 1 if forming else 0  # 正在组的批占预测的第 1 批
        batches = self.config.window_batches + skip
        window = self._plan_window(
            waiting=waiting, chunked=chunked,
            slots=self._request_slots(scheduler=scheduler, running_batch=running_batch),
            batches=batches,
        )
        planned: dict[int, dict[tuple[int, int], Priority]] = {}
        current: dict[int, set[tuple[int, int]]] = {}  # 正在组的批：已有同区间段保留，不新建
        orders = [0] * (batches + 1)
        for req, start, end, batch in window:
            if batch <= skip:
                current.setdefault(id(req), set()).add((start, end))
                continue
            planned.setdefault(id(req), {})[(start, end)] = (NEAR, batch, 0, orders[batch])
            orders[batch] += 1
        with self._condition:
            if self._stopping:
                return
            submits = []  # (请求状态, 段, 优先级)，撤销全部窗口外段之后再统一提交
            for key, state in self.states.items():
                wanted = planned.get(key, {})
                keep = current.get(key, set())
                kept, held = {}, []
                for segment in state.segments:
                    span = (segment.start, segment.end)
                    if segment.cancelled:
                        continue
                    if span in wanted:
                        kept[span] = segment
                    elif span in keep:
                        # 正在组的批沿用已读部分，并提到最前继续读；start_batch 固定 READY 行后撤销它。
                        held.append(segment)
                        submits.append((state, segment, (NEAR, 1, 0, len(held))))
                    else:
                        # 无交集即移出窗口；有交集但区间变动也要按新区间重建。
                        self.cancel_segment(segment=segment)
                fresh = []
                for span, priority in wanted.items():
                    segment = kept.get(span) or PrefetchSegment(*span)
                    fresh.append(segment)
                    submits.append((state, segment, priority))
                state.segments = held + fresh
            # 先释放全部窗口外引用，再按新窗口提交或重试，缓存仍由现有 LRU 管理。
            for state, segment, priority in submits:
                self._submit(state=state, segment=segment, priority=priority)
            self._jobs.sort(key=_priority)
            self._condition.notify_all()
        if (
            self.config.log_interval and not self._idle_logged and not self.states
            and not waiting and chunked is None and (running_batch is None or not running_batch.reqs)
        ):
            self._log_stats(phase="idle")

    def run_prefill(
        self, original_fn: Callable[..., Any], scheduler: Any,
        prefill_delayer_single_pass: Any, running_batch: Any,
    ) -> Any:
        """包围一次原生 raw prefill 调度，让窗口对账使用排序后的队列。

        输入：original_fn 为原生未绑定函数；scheduler 和其余参数原样传入。
        输出：原函数结果；未触发 priority Hook 时按返回时的队列对账，有 prefill 需求则同步失败。
        """
        token = _active_schedule.set((self, scheduler))
        try:
            result = original_fn(
                scheduler, prefill_delayer_single_pass=prefill_delayer_single_pass, running_batch=running_batch,
            )
            if _active_schedule.get() is not None:
                # 原生提前返回、本轮不组 prefill 批，下一个 prefill 批就是窗口第 1 批。
                self._reconcile_prefill(scheduler=scheduler, running_batch=running_batch, forming=False)
            return result
        finally:
            _active_schedule.reset(token)

    def _reconcile_prefill(self, *, scheduler: Any, running_batch: Any, forming: bool) -> None:
        """按原生队列对账，并在有 prefill 需求时同步后台失败。

        输入：scheduler 提供 waiting 队列和续段，running_batch 提供运行中请求数；
            forming 为 True 表示从 calc_priority 进入，本轮紧接着组 prefill 批。
        输出：无返回值；更新窗口，纯 decode 轮次跳过归约。
        """
        self.reconcile(scheduler=scheduler, running_batch=running_batch, forming=forming)
        if scheduler.waiting_queue or scheduler.chunked_req is not None:
            self._synchronize_failure()

    def record_batch(self, batch: Any) -> None:
        """记录已提交 forward 的请求与不可变 token 消费范围。

        输入：batch 为 run_batch 的原生 ScheduleBatch；借用 prefix_lens/extend_lens/forward_iter。
        输出：无返回值；只为 prefill 请求保存 (Req, start, end)，不修改 batch 或 Req。
        同一调度线程调用；overlap 的 batch.copy 保留 forward_iter，后续可准确对账旧结果。
        """
        if not batch.forward_mode.is_extend() or batch.prefix_lens is None or batch.extend_lens is None:
            return
        spans = []
        decoding = {id(req) for req in (batch.decoding_reqs or [])}
        for req, start, length in zip(batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True):
            state = self.states.get(id(req))
            if state is not None and id(req) not in decoding:
                spans.append((req, int(start), int(start + length)))
        self._batches[int(batch.forward_iter)] = tuple(spans)

    def release_batch(self, batch: Any) -> None:
        """根据已保存的 forward 快照归还完成段，不读取可变的 Req 消费区间。

        输入：batch 为原生结果处理完成的 batch，用其 forward_iter 查询提交时快照。
        输出：无返回值；归还已消费的段，完整请求删除状态，按配置频率记录统计。
        撤回重登时已清理该请求的旧范围；forward 使用自己持有的显存缓冲。
        """
        for req, _, end in self._batches.pop(int(batch.forward_iter), ()):
            state = self.states.get(id(req))
            if state is not None:
                for segment in state.segments:
                    if not segment.cancelled and segment.end <= end:
                        self.cancel_segment(segment=segment)
                if end >= len(state.token_ids):
                    self._discard_state(state=state)
        if not self.config.log_interval:
            return
        self._prefill_count += 1
        if self._prefill_count % self.config.log_interval == 0:
            self._log_stats(phase="prefill")

    def _log_stats(self, *, phase: str) -> None:
        """输出本 rank 的行池和控制器累计快照。

        输入：phase 为 prefill 或 idle；调用方只在统计开启时调用。
        输出：无返回值；零引用时才发出本轮唯一的 idle 快照。
        """
        with self._condition:
            # 按已有的 condition → 池 mutex 顺序获取快照，锁外编码日志。
            stats = self.runtime.pool.stats()
            if phase == "idle":
                if stats["referenced_slots"]:
                    return
                self._idle_logged = True
            stats.update(
                phase=phase, tp_rank=self.runtime.tp_rank,
                window_hit_rate=self._window_hits / self._window_rows if self._window_rows else 0.0,
                prefill_wait_ms=self._prefill_wait_ms, prefill_waits=self._prefill_waits,
                decode_wait_ms=self._decode_wait_ms, decode_waits=self._decode_waits,
                urgent_rows=self._urgent_rows, predicted_spans=self._predicted_spans, matched_spans=self._matched_spans,
                read_ms=self._read_ms, read_rows=self._read_rows,
            )
        logger.info("Engram SSD sched prefetch stats: %s", json.dumps(stats, sort_keys=True))

    def close(self) -> None:
        """停止调度预取并同步回收所有请求状态。

        输入：无显式参数；由 runtime 在行池、reader 与 FD 仍存活时调用。
        输出：无返回值；join 后台预取线程和 decode 线程，清空调度请求、批次和紧急请求。
        本函数会等待在途读取完成；仅用于关闭阶段，不能从 worker 自身或普通调度检查中调用。
        """
        with self._condition:
            self._stopping = True
            self._closing = True  # decode 线程处理完已登记的批次后退出
            self._jobs.clear()
            self._condition.notify_all()
        self._thread.join()
        self._decode_thread.join()
        for _, request in self._urgent:
            self._release_hits(request=request)
        self._urgent.clear()
        for state in list(self.states.values()):
            self._discard_state(state=state)
        self._batches.clear()
