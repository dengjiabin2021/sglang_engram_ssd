"""管理 waiting 请求的分段预取、TP 放行与跨 forward 的行池引用。"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from .config import get_config

if TYPE_CHECKING:
    from .runtime import EngramSsdRuntime

logger = logging.getLogger(__name__)


@dataclass
class PrefetchSegment:
    """单个 token 区间的预取状态，不持有文件句柄或 GPU 输出。

    由控制器切段时创建并随请求状态持有；消费或取消后释放 handles 与 hash 缓存。
    调度线程负责提交/回收，唯一后台线程负责读取；共享状态在控制器 condition 下发布。
    """

    start: int  # 完整 token 序列中的包含起点，单位 token。
    end: int  # 完整 token 序列中的不包含终点。
    status: str = "new"  # new/queued/reading/pending/ready/blocked；取消另由 cancelled 标记。
    submitted_at: float | None = None  # 首次入后台队列的 monotonic 秒，重试不重置。
    handles: list[Any] = field(default_factory=list)  # 按 Engram 层顺序持有池引用；READING 时仍由 worker 局部持有。
    cancelled: bool = False  # 取消后不可再次提交；已发起的读取仍须收齐。
    pool_blocked: bool = False  # 最近是否因池满失败，重试在途期间保留，供入队 deadline 判断。
    blocked_epoch: int = -1  # 最近池满时的释放版本，同版本不重试。
    row_ids: torch.Tensor | None = None  # 池满重试保留的 CPU int64 [T,L,C]，T 为本段 token 数。


@dataclass
class RequestPrefetchState:
    """一个请求版本的不可变 token 快照及分段引用，不修改原生 Req 的 KV 状态。

    由入队 Hook 创建，控制器持有至完成、撤回重登或对账清理；request 强引用保护对象身份。
    generation 隔离同一 Req 的撤回版本；状态发布由控制器 condition 与唯一调度线程协调。
    """

    rid: str  # 请求标签，仅用于识别，不作为可能重复的字典键。
    request: Any  # 借用原生 Req 并持强引用，存活期间 id(request) 不会被复用。
    generation: int  # 控制器内递增版本号；旧 batch 结果不可释放新版本。
    token_ids: tuple[int, ...]  # 原始输入与已确认输出组成的完整快照，不随原生列表修改。
    segments: list[PrefetchSegment]  # 按 start 升序，区间连续；最多两段同时占有预取资源。
    enqueued_at: float  # 本版本入队的 monotonic 秒。
    completed_until: int  # 已消费或命中 prefix 的 token 终点，只增不减。
    cancelled: bool = False  # 旧版本取消后不再提交；在途读取由 worker 收尾。
    initialized: bool = False  # 首次候选检查或原生调度早退时才按可见 prefix 切段。


@dataclass(frozen=True)
class QueueEntry:
    """持有入队时间及请求强引用的只读记录，不管理池引用。

    由控制器登记或候选检查创建，调度线程持有并在对账离队后删除；无需跨线程修改。
    """

    request: Any  # 强引用保证 id 不会在入队时间记录仍存在时被其他请求复用。
    entered_at: float  # monotonic 秒，供未提交请求的防饥饿 deadline 使用。


@dataclass(frozen=True)
class BatchSpan:
    """单次 forward 消费范围的只读快照，不再读取之后可能被修改的 Req 区间。

    由 run_batch AFTER 在调度线程创建，控制器按 forward_iter 持有至结果处理；不跨线程修改。
    """

    request_id: int  # 对应 states 的 id(req)，须与 generation 一起验证。
    generation: int  # 提交该 batch 时的请求版本。
    start: int  # 本次 prefill 输入的包含起点，单位 token。
    end: int  # 本次 prefill 输入的不包含终点。


@dataclass
class SchedulingFrame:
    """一次 raw 调度的队列别名与同步状态，不跨越该调用存活。

    由 run_prefill 创建并存入当前线程 ContextVar，退出时恢复；priority Hook 只在该上下文中筛选。
    """

    controller: SchedulerPrefetchController  # 本次调用所属控制器，防止误处理其他调度上下文。
    scheduler: Any  # 借用原生 scheduler，退出时归还 waiting_queue 的原始列表别名。
    original_queue: list[Any]  # 原始列表对象，其他持有该别名的代码也能看到恢复后的队列。
    snapshot: list[Any] = field(default_factory=list)  # 原生排序后、筛选前的完整候选顺序。
    blocked: list[Any] = field(default_factory=list)  # 本次归约后暂缓的请求。
    filtered: bool = False  # 是否完成候选筛选，供队列恢复使用。
    synchronized: bool = False  # 是否已经做过本次失败/放行 MIN，保证空队列也只通信一次。


_active_schedule: ContextVar[SchedulingFrame | None] = ContextVar("engram_ssd_schedule", default=None)


def filter_active_waiting_queue(waiting_queue: list[Any]) -> None:
    """将原生 priority 完成后的候选转交当前调度控制器。

    输入：waiting_queue 为原生可变 Req 列表，必须是当前 scheduler 持有的同一对象。
    输出：无返回值；仅在本插件的 raw 调度上下文中原地筛选，其他调用不产生副作用。
    """
    frame = _active_schedule.get()
    if frame is not None and waiting_queue is frame.scheduler.waiting_queue:
        frame.controller.filter_waiting_queue(waiting_queue=waiting_queue)


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
    """管理本 rank 请求预取、TP 一致放行及行池引用，不执行模型或改写原生 KV 状态。

    runtime 惰性创建并持有本对象，关闭时先 join 唯一后台线程，再由 runtime 释放池和 reader。
    调度入口由原生调度线程串行调用；后台仅计算 hash 与读 SSD，condition 保护任务/状态发布。
    TP 通信只在 raw 调度边界执行，每次最多一次 CPU MIN，后台与 forward 均不传播本地后台异常。
    """

    def __init__(self, runtime: EngramSsdRuntime, scheduler: Any) -> None:
        """绑定本 rank 资源并启动唯一后台线程。

        输入：runtime 借出 CPU Hasher、行池和独立 prefetch_reader；scheduler 借出 chunk 大小与 CPU TP 组。
        输出：无返回值；初始化请求状态、任务队列和线程；缺少 Hasher/行池时抛 RuntimeError。
        不在此打开文件或建立第二个线程池，借用资源均由 runtime 关闭。
        """
        self.runtime = runtime
        self.config = get_config()
        if runtime.hasher is None or runtime.pool is None:
            raise RuntimeError("Engram scheduler prefetch requires CPU hashing parameters and a row pool")
        configured = int(self.config.segment_tokens)
        chunked = int(getattr(scheduler, "chunked_prefill_size", 0) or 0)
        self.segment_tokens = configured if configured > 0 else (chunked if chunked > 0 else 8192)
        self.chunked_tokens = chunked  # <=0 表示不分 chunk，放行检查的上界为剩余完整 prompt。
        self.cpu_group = scheduler.tp_group.cpu_group  # 借用已有 gloo 组，不在后台建立通信组。
        self.states: dict[int, RequestPrefetchState] = {}  # id(req) -> 当前版本；值持有 Req 强引用。
        self._queue_entered: dict[int, QueueEntry] = {}  # 包括尚未登记预取状态的候选入队时刻。
        self._generation = 0  # 撤回重登时递增，隔离旧 batch 的异步结果。
        self._condition = threading.Condition()  # 保护任务队列、失败标志及段状态发布；锁内不做 I/O。
        self._jobs: deque[tuple[RequestPrefetchState, PrefetchSegment]] = deque()  # 唯一 worker 消费的有序任务。
        self._batches: dict[int, tuple[BatchSpan, ...]] = {}  # forward_iter -> 不可变消费范围。
        self._failure: BaseException | None = None  # 后台首个错误，调度 MIN 后统一传播。
        self._release_epoch = 0  # 任何实际引用归还后递增，池满段据此判断是否值得重试。
        self._stopping = False  # 关闭或后台失败后停止接收与读取新任务。
        self._prefill_count = 0  # 仅开启统计时累计，用于 LOG_INTERVAL。
        self._idle_logged = False  # 每次请求生命周期结束后只记录一次空闲归零快照。
        self._thread = threading.Thread(target=self._worker, name="engram-ssd-prefetch", daemon=True)
        self._thread.start()

    def _synchronize_flags(self, flags: Sequence[int], frame: SchedulingFrame) -> list[int]:
        """以一次 CPU MIN 同步后台失败和整个候选列表的放行标志。

        输入：flags 为按同一 waiting 顺序构造的 0／1 标志，可为空；frame 为本次 raw 调度上下文。
        输出：所有 TP rank 一致的放行标志列表；任一 rank 失败则各 rank 同时抛 RuntimeError。
        独立的首元素承载失败；有 chunked 请求但无候选时也可归约空 flags。
        调用方只按原生 waiting／chunked 状态决定是否通信，不依赖本地故障或行池状态。
        """
        with self._condition:
            failure = self._failure
        mask = torch.tensor([-1 if failure is not None else 1, *flags], dtype=torch.int32, device="cpu")
        if self.runtime.tp_size > 1:
            import torch.distributed as dist

            # 各 rank 必须在同一次 raw 调度使用相同长度和顺序；失败检测不另加第二次通信。
            dist.all_reduce(mask, op=dist.ReduceOp.MIN, group=self.cpu_group)
        frame.synchronized = True
        values = mask.tolist()
        if values[0] < 0:
            raise RuntimeError("Engram scheduler prefetch failed on a TP rank") from failure
        return values[1:]

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
        """登记请求 token 快照，撤回重登时释放旧版本。

        输入：req 为已实际进入 waiting 的原生 Req；is_retracted 表示重新排队，需生成新版本。
        输出：无返回值；保存原始输入加已确认输出的 CPU token 快照和入队时间，不在此切段或读 SSD。
        同一调度线程调用；后台已失败时不提前抛错，由下一次 raw 调度 MIN 统一传播。
        """
        self._idle_logged = False
        key = id(req)
        now = time.monotonic()
        if not self._valid_request(req=req):
            return
        entry = self._queue_entered.setdefault(key, QueueEntry(request=req, entered_at=now))
        entered = entry.entered_at
        existing = self.states.get(key)
        is_retracted = is_retracted or bool(getattr(req, "is_retracted", False))
        if is_retracted:
            entered = now
        if existing is not None and not is_retracted:
            return
        if existing is not None:
            self._discard_state(state=existing)
            entered = now
        tokens = tuple(req.origin_input_ids) + tuple(req.output_ids)
        self._generation += 1
        state = RequestPrefetchState(
            rid=str(req.rid), request=req, generation=self._generation, token_ids=tokens,
            segments=[], enqueued_at=entered, completed_until=0,
        )
        with self._condition:
            if self._stopping:
                # 后台错误统一交给调度边界的 MIN，避免入队 Hook 提前退出、其他 rank 仍继续模型通信。
                if self._failure is not None:
                    return
                raise RuntimeError("Engram scheduler prefetch is closed")
            self.states[key] = state
            self._queue_entered[key] = QueueEntry(request=req, entered_at=entered)

    def _initialize_segments(self, state: RequestPrefetchState) -> None:
        """按当前可见 prefix 快照为请求首次切段，不重新匹配 KV 前缀。

        输入：state 为控制器持有的当前请求版本；已初始化时保持不变。
        输出：无返回值；生成连续 token 区间，更新 completed_until/initialized，不提交 I/O。
        由调度线程在 priority 后或原生 raw 早退边界调用；早退时允许冗余读取以保留等待窗口。
        """
        if state.initialized:
            return
        # cache-aware 策略或支持 fast match 的缓存会在 calc_priority 内刷新 prefix；
        # 其他策略使用当前快照。原生 init_next_round_input 仍会复核，之后变化由补读处理。
        start = min(max(0, int(state.request.prefix_len)), len(state.token_ids))
        state.segments = [
            PrefetchSegment(start=p, end=min(p + self.segment_tokens, len(state.token_ids)))
            for p in range(start, len(state.token_ids), self.segment_tokens)
        ]
        state.completed_until = start
        state.initialized = True

    def notify_pool_release(self) -> None:
        """在池引用实际归还后推进容量版本。

        输入：无显式参数；调用方须已完成 release/cancel，包括 GPU event 完成后的临时引用。
        输出：无返回值；在 condition 下递增版本，使池满段在下一次调度可重试，不直接提交 I/O。
        """
        with self._condition:
            self._release_epoch += 1

    def _release(self, handles: Sequence[Any], *, cancel: bool = False, capacity_changed: bool = True) -> None:
        """归还一组池引用并按需通知重试。

        输入：handles 为池句柄序列；cancel=True 仅用于尚未提交或已收齐 I/O 的 PENDING；
            capacity_changed=False 表示仅回滚本次失败占槽，不算外部容量变化。
        输出：无返回值；原生池 release/cancel 引用，必要时推进释放版本；不关闭池或 FD。
        """
        for handle in handles:
            if cancel:
                self.runtime.pool.cancel(handle=handle)
            else:
                self.runtime.pool.release(handle=handle)
        # 占槽失败时只撤销本次部分占用，并未增加重试前的可用空间，不能触发自己的重试。
        if handles and capacity_changed:
            self.notify_pool_release()

    def _discard_state(self, state: RequestPrefetchState) -> None:
        """撤销请求版本并回收已发布的段引用。

        输入：state 为控制器当前或旧请求版本；调用方不得把仍在读取的局部句柄提前归还。
        输出：无返回值；标记全部段取消、清空 hash 缓存，从索引删除当前版本并归还已发布引用。
        在途句柄由 worker 独立持有，等同步读取结束后收尾，避免覆盖内核仍在写的槽。
        """
        handles = []
        with self._condition:
            state.cancelled = True
            for segment in state.segments:
                segment.cancelled = True
                # READING 的 handle 由后台持有，I/O 完成后才释放。
                handles.extend(segment.handles)
                segment.handles = []
                segment.row_ids = None
            if self.states.get(id(state.request)) is state:
                del self.states[id(state.request)]
                self._queue_entered.pop(id(state.request), None)
        self._release(handles=handles)

    def _submit(self, state: RequestPrefetchState, segment: PrefetchSegment) -> None:
        """把可执行段排入唯一后台线程的队列。

        输入：state/segment 为未取消的请求版本及其段；new 或释放版本变化后的 blocked 才可提交。
        输出：无返回值；最多保留当前段和下一段，首次提交记录 monotonic 秒并转为 queued。
        只在 condition 下更新队列，不等待文件读取或在调度线程计算 hash。
        """
        with self._condition:
            if self._stopping or state.cancelled or segment.cancelled or segment.status not in ("new", "blocked"):
                return
            if segment.status == "blocked" and segment.blocked_epoch == self._release_epoch:
                return
            held = sum(
                s.status in ("queued", "reading", "pending", "ready") and not s.cancelled for s in state.segments
            )
            if held >= 2:
                return
            if segment.submitted_at is None:
                segment.submitted_at = time.monotonic()
            segment.status = "queued"
            self._jobs.append((state, segment))
            self._condition.notify()

    def _worker(self) -> None:
        """由唯一后台线程依次完成段 hash、占槽和 io_uring 读取。

        输入：无显式参数；从受 condition 保护的 _jobs 取出请求段，借用 runtime 的后台 reader。
        输出：无返回值；成功发布 pending/ready，池满发布 blocked，取消后归还引用；
            错误保存到 _failure 并停止提交，由调度线程统一归约后报错。
        读取期间不持 condition、不发起 TP 通信，退出前不留下仍在写的 I/O。
        """
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or bool(self._jobs))
                if self._stopping:
                    return
                state, segment = self._jobs.popleft()
                if state.cancelled or segment.cancelled:
                    continue
                # queued -> reading：此后句柄由 worker 局部持有，调度取消不能提前回收在途槽。
                segment.status = "reading"
                attempt_epoch = self._release_epoch
            handles = []
            try:
                handles = self._read_segment(state=state, segment=segment)
                with self._condition:
                    cancelled = self._stopping or state.cancelled or segment.cancelled
                    if not cancelled:
                        segment.handles = handles
                        # reading -> blocked 表示未完成整段占槽；有句柄先发布 pending，再确认各层 READY。
                        segment.status = "pending" if handles else "blocked"
                        segment.pool_blocked = not bool(handles)
                        if not handles:
                            segment.blocked_epoch = attempt_epoch
                        if handles and all(self.runtime.pool.is_ready(handle=handle) for handle in handles):
                            # pending -> ready：所有层缺失行均已写完或复用现成 READY，可安全放行消费。
                            segment.status = "ready"
                        if handles:
                            segment.row_ids = None
                        handles = []
                    self._condition.notify_all()
            except BaseException as error:
                with self._condition:
                    # 当前 reader 已收齐在途写入，记录失败并停工；不在后台或单个模型 rank 直接退出。
                    self._failure = error
                    self._stopping = True
                    self._condition.notify_all()
                return
            finally:
                self._release(handles=handles)

    def _read_segment(self, state: RequestPrefetchState, segment: PrefetchSegment) -> list[Any]:
        """在后台为一段所有 Engram 层占槽，并同步读入缺失行。

        输入：state 提供完整 CPU token 快照；segment 给出 [start,end)；池满时复用其缓存行号。
        输出：按层排序的池句柄列表，调用方负责发布或归还；池满/取消返回空列表。
        CPU int64 [T,L,C] 哈希行号按本 rank owner 过滤；全部层先占槽，失败则撤销部分占用。
        读错在 reader 收齐 I/O 后取消 PENDING 并抛出；只允许唯一 worker 调用，不做 TP 通信。
        """
        # CPU int64 [segment_tokens,L,C]；所有层先占槽，池满时整段撤销以免占住无法完成的段。
        with self._condition:
            ids = segment.row_ids
        if ids is None:
            ids = self.runtime.hasher.compute_request_row_ids(
                token_ids=state.token_ids, start=segment.start, end=segment.end,
            )
            with self._condition:
                if self._stopping or state.cancelled or segment.cancelled:
                    return []
                segment.row_ids = ids
        # 局部引用独立持有张量；调度线程可以在取消时清空缓存，不能影响正在处理的局部输入。
        handles = []
        try:
            for column, layer_id in enumerate(self.runtime.hasher.layer_ids):
                if self._stopping or state.cancelled or segment.cancelled:
                    self._release(handles=handles, cancel=True)
                    return []
                table = self.runtime.open_table(layer_id=layer_id)
                lo = table.rows * self.runtime.tp_rank // self.runtime.tp_size
                hi = table.rows * (self.runtime.tp_rank + 1) // self.runtime.tp_size
                # CPU int64 [T,L,C] -> [T*C]，只占本 rank owner 区间，保持全局行号交给 reader 定位。
                layer_ids = ids[:, column].reshape(-1)
                owner_ids = layer_ids[(layer_ids >= lo) & (layer_ids < hi)].contiguous()
                handle = self.runtime.pool.acquire(layer_id=layer_id, row_ids=owner_ids)
                if handle is None:
                    self._release(handles=handles, cancel=True, capacity_changed=False)
                    return []
                handles.append(handle)
            for layer_id, handle in zip(self.runtime.hasher.layer_ids, handles, strict=True):
                if self._stopping or state.cancelled or segment.cancelled:
                    self._release(handles=handles, cancel=True)
                    return []
                if handle.missing_row_ids.numel():
                    if self.config.log_interval:
                        self.runtime.pool.record_submitted(rows=handle.missing_row_ids.numel())
                    # 将 CPU int64 [Q] 的缺失行写到池指定槽；同步返回后才可发布 PENDING -> READY。
                    self.runtime.prefetch_reader.read_rows_to_slots(
                        layer_id=layer_id, row_ids=handle.missing_row_ids, slot_ids=handle.missing_slot_ids,
                        weight_base=self.runtime.pool.weight, scale_base=self.runtime.pool.scale,
                    )
                    self.runtime.pool.mark_ready(slot_ids=handle.missing_slot_ids)
            return handles
        except BaseException:
            # 唯一 worker 不会在当前段完成前 acquire 下一段；resolve 只 pin READY。
            # 当前段自建 PENDING 没有其他生产句柄，取消后即回收；已 READY 的复用行仅归还引用。
            self._release(handles=handles, cancel=True)
            raise

    def _next_segment(self, state: RequestPrefetchState) -> PrefetchSegment | None:
        """定位请求尚未消费的首段，供后台提交优先级使用。

        输入：state 为已初始化的请求版本，位置取已消费终点、真实 prefix 与 already_computed 的最大值。
        输出：首个相交段或 None；prefix 缩至原覆盖范围之前时返回 None，不能把旧 READY 当完整覆盖。
        只检查调度元数据，不改变 Req 或池状态。
        """
        req = state.request
        first_start = state.segments[0].start if state.segments else len(state.token_ids)
        # waiting 期间 prefix 页被驱逐时，旧段不覆盖更早的需求，不能误报整段 READY。
        if int(req.prefix_len) < first_start:
            return None
        start = max(state.completed_until, int(req.prefix_len), int(getattr(req, "already_computed", 0)))
        return next((s for s in state.segments if not s.cancelled and s.end > start), None)

    def _segments_for_next_chunk(self, state: RequestPrefetchState) -> tuple[list[PrefetchSegment], bool]:
        """收集下一次 prefill 输入上界覆盖的全部段。

        输入：state 为请求版本；chunked_tokens>0 时取一个 chunk 上界，否则取剩余完整 prompt。
        输出：(按 token 起点排序的相交段列表, 是否无缺口覆盖目标区间)，不检查 READY 或执行 I/O。
        超过两段的长 prompt 不能全部同时预取，放行方须使用既有入队 deadline，而非永久等待。
        """
        req = state.request
        start = max(state.completed_until, int(req.prefix_len), int(getattr(req, "already_computed", 0)))
        if not state.initialized or int(req.prefix_len) < state.completed_until:
            return [], False
        end = len(state.token_ids)
        if self.chunked_tokens > 0:
            end = min(end, start + self.chunked_tokens)
        # 未启用 chunk 时，上界是完整剩余 prompt；超过两段资源上限时 wait_complete 也走入队超时。
        # 实际 adder 还会按剩余 batch/token 预算缩短 chunk；这里取上界，避免只检查前半段。
        segments = [s for s in state.segments if not s.cancelled and s.start < end and s.end > start]
        covered = start >= end or (
            bool(segments) and segments[0].start <= start and segments[-1].end >= end
            and all(left.end == right.start for left, right in zip(segments, segments[1:]))
        )
        return segments, covered

    def _retire_segments(self, state: RequestPrefetchState, end: int) -> None:
        """释放已完整消费或已命中 prefix 的段。

        输入：state 为目标请求版本，end 为完整 token 序列中的不包含终点。
        输出：无返回值；completed_until 只增不减，end 之前完整段取消并归还句柄、清空 hash 缓存。
        部分消费段保留至整段完成，GPU 仍使用的槽另由 resolve 临时引用保护。
        """
        handles = []
        with self._condition:
            state.completed_until = max(state.completed_until, end)
            for segment in state.segments:
                if segment.end <= end:
                    segment.cancelled = True
                    handles.extend(segment.handles)
                    segment.handles = []
                    segment.row_ids = None
        self._release(handles=handles)

    def reconcile(self, scheduler: Any) -> None:
        """对账请求生命周期，并按当前调度优先级补交后台任务。

        输入：scheduler 借出 waiting_queue、chunked_req 及其存活状态；只由调度线程调用。
        输出：无返回值；清理已结束/不再活动的请求引用、收集已完成 GPU 引用、重排任务和更新统计。
        仅提交到后台队列，不读文件、不等待 I/O，也不在此发起 TP 通信或本地抛出后台异常。
        """
        self.runtime.collect_pool_uses()
        with self._condition:
            if self._failure is not None:
                return
        waiting = list(scheduler.waiting_queue)
        chunked = getattr(scheduler, "chunked_req", None)
        active = {id(req) for req in waiting}
        if chunked is not None:
            active.add(id(chunked))
        inflight = {(span.request_id, span.generation) for spans in self._batches.values() for span in spans}
        # waiting/chunked 之外的请求也可能仍有 overlap batch 在途，须保留至对应结果到达再清理。
        # 已结束/中止请求可丢弃段引用，实际 GPU 使用由 runtime 的临时 event 引用独立保护。
        for key, state in list(self.states.items()):
            if not self._valid_request(req=state.request) or (
                key not in active and (key, state.generation) not in inflight
            ):
                self._discard_state(state=state)
        for key in list(self._queue_entered):
            if key not in active and key not in self.states:
                self._queue_entered.pop(key, None)
        # 先保障下一轮最可能消费的 continuation，再让 waiting 请求公平取得各自首段；
        # 最后补第二段，防止单个长请求抢占后台队列和整池容量。
        requests = ([chunked] if chunked is not None else []) + waiting
        candidates = []
        for req in requests:
            state = self.states.get(id(req))
            if state is None or not state.initialized:
                continue
            self._retire_segments(state=state, end=max(state.completed_until, int(req.prefix_len)))
            segment = self._next_segment(state=state)
            if segment is not None:
                candidates.append((state, segment))
                self._submit(state=state, segment=segment)
        following = []
        for state, first in candidates:
            index = state.segments.index(first)
            if index + 1 < len(state.segments):
                segment = state.segments[index + 1]
                following.append((state, segment))
                self._submit(state=state, segment=segment)
        # 请求优先级可能在排队后改变，重排尚未开始的任务使后台跟上本轮调度顺序；
        # 已开始的 I/O 必须读完再处理取消，不能为了优先级提前释放目标内存。
        desired = [(id(state), id(segment)) for state, segment in candidates]
        desired.extend((id(state), id(segment)) for state, segment in following)
        order = {key: index for index, key in enumerate(desired)}
        with self._condition:
            self._jobs = deque(sorted(
                self._jobs, key=lambda pair: order.get((id(pair[0]), id(pair[1])), len(order)),
            ))
        if self.config.log_interval and not self.states and not self._idle_logged:
            stats = self.runtime.pool.stats()
            if stats["referenced_slots"] == 0:
                stats["phase"] = "idle"
                stats["tp_rank"] = self.runtime.tp_rank
                logger.info("Engram SSD sched prefetch stats: %s", json.dumps(stats, sort_keys=True))
                self._idle_logged = True

    def run_prefill(
        self, original_fn: Callable[..., Any], scheduler: Any,
        prefill_delayer_single_pass: Any, running_batch: Any,
    ) -> Any:
        """包围一次原生 raw prefill 调度，维护队列别名和统一失败边界。

        输入：original_fn 为借用的原生未绑定函数；scheduler 为原生实例；
            prefill_delayer_single_pass/running_batch 按原参数原样传入，不复制其实现。
        输出：原函数结果；正常 priority 路径筛选候选，有 prefill 需求的早退路径补交预取并同步失败。
        无 waiting／chunked 请求时只做首次对账，跳过归约；积压后台错误留到下一次 prefill 需求传播。
        每次至多一次 CPU MIN；finally 恢复暂缓与新追加请求，原函数异常保持向上传播。
        """
        self.reconcile(scheduler=scheduler)
        frame = SchedulingFrame(controller=self, scheduler=scheduler, original_queue=scheduler.waiting_queue)
        token = _active_schedule.set(frame)
        try:
            result = original_fn(
                scheduler, prefill_delayer_single_pass=prefill_delayer_single_pass, running_batch=running_batch,
            )
            # 原生 raw 可在 batch 满/无请求槽时早退；不能因此丢掉 waiting 与当前 batch 重叠的窗口。
            # 尚未经过 priority 的请求按当前 prefix 快照切段，可能重复前缀，但不改原生缓存/LRU。
            # states 在本地后台失败后可能残留，不能据此决定各 rank 是否参加 collective。
            # 必须先调用原生 raw，因为它可能把 grammar-ready 请求加入 waiting 队列。
            if not frame.filtered and (
                scheduler.waiting_queue or getattr(scheduler, "chunked_req", None) is not None
            ):
                for req in scheduler.waiting_queue:
                    state = self.states.get(id(req))
                    if state is not None:
                        self._initialize_segments(state=state)
                self.reconcile(scheduler=scheduler)
                # 有 prefill 需求但没有候选检查时仍同步失败；纯 decode 不增加 CPU 归约。
                if not frame.synchronized:
                    self._synchronize_flags(flags=(), frame=frame)
            return result
        finally:
            if frame.filtered:
                # 原生 raw 可能重绑定列表或追加被抢占请求；合并快照中的暂缓项与原函数剩余/新增项，
                # 再写回原列表别名，避免请求丢失、重复或其他别名持有过期队列。
                remaining = list(scheduler.waiting_queue)
                remaining_ids = {id(req) for req in remaining}
                blocked_ids = {id(req) for req in frame.blocked}
                old_ids = {id(req) for req in frame.snapshot}
                restored = [r for r in frame.snapshot if id(r) in blocked_ids or id(r) in remaining_ids]
                restored.extend(r for r in remaining if id(r) not in old_ids)
                frame.original_queue[:] = restored
                scheduler.waiting_queue = frame.original_queue
            _active_schedule.reset(token)

    def filter_waiting_queue(self, waiting_queue: list[Any]) -> None:
        """在原生 priority 后以一次 CPU MIN 决定整个 waiting 列表的放行结果。

        输入：waiting_queue 为当前调度上下文中的可变 Req 列表，各 TP rank 的顺序必须一致。
        输出：无返回值；原地保留放行项并记录暂缓快照，后台失败与候选标志在同一次 MIN 传播。
        空列表仅在 chunked_req 存在时同步失败；无 prefill 需求则跳过，保留错误至下一次需求。
        不等待后台 I/O，重复 priority Hook 不发起第二次通信。
        """
        frame = _active_schedule.get()
        if frame is None or frame.controller is not self or frame.filtered:
            return
        # 只依赖各 TP rank 一致的原生调度状态，不能用可能残留的本地 states／failure 决定通信。
        if not waiting_queue and getattr(frame.scheduler, "chunked_req", None) is None:
            frame.snapshot = []
            frame.filtered = True
            return
        # 此时使用原生 priority 能提供的最新 prefix 快照，且包含刚追加的 grammar-ready 请求。
        for req in waiting_queue:
            state = self.states.get(id(req))
            if state is not None:
                self._initialize_segments(state=state)
        self.reconcile(scheduler=frame.scheduler)
        frame.snapshot = list(waiting_queue)
        now = time.monotonic()
        flags = []
        with self._condition:
            for req in waiting_queue:
                state = self.states.get(id(req))
                segments, covered = self._segments_for_next_chunk(state=state) if state is not None else ([], False)
                ready = not self._valid_request(req=req) or (
                    state is not None and int(req.prefix_len) >= len(state.token_ids)
                )
                ready = ready or (covered and all(
                    segment.status == "ready" or (
                        bool(segment.handles) and all(self.runtime.pool.is_ready(handle=h) for h in segment.handles)
                    ) for segment in segments
                ))
                entry = self._queue_entered.setdefault(id(req), QueueEntry(request=req, entered_at=now))
                entered = state.enqueued_at if state is not None else entry.entered_at
                started = [segment.submitted_at or entered for segment in segments]
                elapsed = max(0.0, now - max(started, default=entered)) * 1000
                # 段比 chunk 小时可能需三段以上；仍保持两段上限，按入队超时补读，避免永久等待。
                unsubmitted = not covered or any(
                    segment.status in ("new", "blocked") or segment.pool_blocked for segment in segments
                )
                queue_elapsed = max(0.0, now - entered) * 1000
                # best_effort 直接交给 forward 补读；READY 必须覆盖整个下一次输入区间。
                # timeout 等本区间最晚提交段超时；未登记/池满/两段不足覆盖则按入队 deadline 防饿死。
                allowed = (
                    self.config.policy == "best_effort" or ready
                    or (self.config.policy == "timeout" and elapsed >= self.config.timeout_ms)
                    or (unsubmitted and queue_elapsed >= self.config.timeout_ms)
                )
                flags.append(int(allowed))
        allowed = self._synchronize_flags(flags=flags, frame=frame)
        frame.blocked = [req for req, flag in zip(waiting_queue, allowed, strict=True) if not flag]
        waiting_queue[:] = [req for req, flag in zip(waiting_queue, allowed, strict=True) if flag]
        frame.filtered = True
        if self.config.log_interval:
            self.runtime.pool.record_deferred(count=len(frame.blocked))

    def record_batch(self, batch: Any) -> None:
        """记录已提交 forward 的请求版本与不可变 token 消费范围。

        输入：batch 为 run_batch 的原生 ScheduleBatch；借用 prefix_lens/extend_lens/forward_iter。
        输出：无返回值；只为 prefill 请求保存 BatchSpan，decode 请求跳过，不修改 batch 或 Req。
        同一调度线程调用；overlap 的 batch.copy 保留 forward_iter，后续可准确对账旧结果。
        """
        if not batch.forward_mode.is_extend() or batch.prefix_lens is None or batch.extend_lens is None:
            return
        spans = []
        decoding = {id(req) for req in (batch.decoding_reqs or [])}
        for req, start, length in zip(batch.reqs, batch.prefix_lens, batch.extend_lens, strict=True):
            state = self.states.get(id(req))
            if state is not None and id(req) not in decoding:
                spans.append(BatchSpan(
                    request_id=id(req), generation=state.generation, start=int(start), end=int(start + length),
                ))
        self._batches[int(batch.forward_iter)] = tuple(spans)

    def release_batch(self, batch: Any) -> None:
        """根据已保存的 forward 快照归还完成段，不读取可变的 Req 消费区间。

        输入：batch 为原生结果处理完成的 batch，用其 forward_iter 查询提交时快照。
        输出：无返回值；相同 generation 的段归还，完整请求删除状态，按配置频率记录统计。
        旧版本结果不会释放撤回重登的新版本；实际 GPU 访问仍由临时 event 引用保护。
        """
        for span in self._batches.pop(int(batch.forward_iter), ()):
            state = self.states.get(span.request_id)
            if state is not None and state.generation == span.generation:
                self._retire_segments(state=state, end=span.end)
                if state.completed_until >= len(state.token_ids):
                    self._discard_state(state=state)
        if not self.config.log_interval:
            return
        self._prefill_count += 1
        if self._prefill_count % self.config.log_interval == 0:
            stats = self.runtime.pool.stats()
            stats["phase"] = "prefill"
            stats["tp_rank"] = self.runtime.tp_rank
            logger.info(
                "Engram SSD sched prefetch stats: %s", json.dumps(stats, sort_keys=True),
            )

    def close(self) -> None:
        """停止调度预取并同步回收所有请求状态。

        输入：无显式参数；由 runtime 在行池、reader 与 FD 仍存活时调用。
        输出：无返回值；停止提交、join 唯一 worker、清空请求/批次/入队记录并归还请求引用。
        本函数会等待在途读取完成；仅用于关闭阶段，不能从 worker 自身或普通调度检查中调用。
        """
        with self._condition:
            self._stopping = True
            self._jobs.clear()
            self._condition.notify_all()
        self._thread.join()
        for state in list(self.states.values()):
            self._discard_state(state=state)
        self._batches.clear()
        self._queue_entered.clear()
