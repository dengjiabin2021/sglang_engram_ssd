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
    start: int
    end: int
    status: str = "new"
    submitted_at: float | None = None
    handles: list[Any] = field(default_factory=list)
    cancelled: bool = False
    pool_blocked: bool = False


@dataclass
class RequestPrefetchState:
    rid: str
    request: Any
    generation: int
    token_ids: tuple[int, ...]
    segments: list[PrefetchSegment]
    enqueued_at: float
    completed_until: int
    cancelled: bool = False


@dataclass(frozen=True)
class BatchSpan:
    request_id: int
    generation: int
    start: int
    end: int


@dataclass
class SchedulingFrame:
    controller: SchedulerPrefetchController
    scheduler: Any
    original_queue: list[Any]
    snapshot: list[Any] = field(default_factory=list)
    blocked: list[Any] = field(default_factory=list)
    filtered: bool = False


_active_schedule: ContextVar[SchedulingFrame | None] = ContextVar("engram_ssd_schedule", default=None)


def filter_active_waiting_queue(waiting_queue: list[Any]) -> None:
    """只在本插件包围的 raw prefill 调用中处理原生排序后的候选列表。"""
    frame = _active_schedule.get()
    if frame is not None and waiting_queue is frame.scheduler.waiting_queue:
        frame.controller.filter_waiting_queue(waiting_queue=waiting_queue)


def get_manager(runtime: EngramSsdRuntime, scheduler: Any) -> SchedulerPrefetchController:
    """模型与行池已初始化后惰性创建；关闭开关不进入这个入口。"""
    manager = getattr(runtime, "sched_prefetch", None)
    if manager is None:
        manager = SchedulerPrefetchController(runtime=runtime, scheduler=scheduler)
        runtime.sched_prefetch = manager
    return manager


class SchedulerPrefetchController:
    def __init__(self, runtime: EngramSsdRuntime, scheduler: Any) -> None:
        self.runtime = runtime
        self.config = get_config()
        if runtime.hasher is None or runtime.pool is None:
            raise RuntimeError("Engram scheduler prefetch requires CPU hashing parameters and a row pool")
        configured = int(self.config.segment_tokens)
        chunked = int(getattr(scheduler, "chunked_prefill_size", 0) or 0)
        self.segment_tokens = configured if configured > 0 else (chunked if chunked > 0 else 8192)
        self.cpu_group = scheduler.tp_group.cpu_group
        self.states: dict[int, RequestPrefetchState] = {}
        self._queue_entered: dict[int, float] = {}
        self._generation = 0
        self._condition = threading.Condition()
        self._jobs: deque[tuple[RequestPrefetchState, PrefetchSegment]] = deque()
        self._batches: dict[int, tuple[BatchSpan, ...]] = {}
        self._failure: BaseException | None = None
        self._stopping = False
        self._prefill_count = 0
        self._idle_logged = False
        self._thread = threading.Thread(target=self._worker, name="engram-ssd-prefetch", daemon=True)
        self._thread.start()

    def raise_if_failed(self) -> None:
        """后台异常在调度或 forward 主路径传播，不能转为未命中后静默补读。"""
        with self._condition:
            failure = self._failure
        if failure is not None:
            raise RuntimeError("Engram scheduler prefetch failed") from failure

    @staticmethod
    def _valid_request(req: Any) -> bool:
        return (
            getattr(req, "multimodal_inputs", None) is None
            and getattr(req, "finished_reason", None) is None
            and getattr(req, "to_finish", None) is None
        )

    def register_request(self, req: Any, is_retracted: bool = False) -> None:
        """成功入队后登记；使用完整序列快照，不读取尚未初始化的截断 fill_ids。"""
        self.raise_if_failed()
        self._idle_logged = False
        key = id(req)
        now = time.monotonic()
        entered = self._queue_entered.setdefault(key, now)
        if not self._valid_request(req=req):
            return
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
        start = min(max(0, int(req.prefix_len)), len(tokens))
        self._generation += 1
        state = RequestPrefetchState(
            rid=str(req.rid), request=req, generation=self._generation, token_ids=tokens,
            segments=[
                PrefetchSegment(start=p, end=min(p + self.segment_tokens, len(tokens)))
                for p in range(start, len(tokens), self.segment_tokens)
            ],
            enqueued_at=entered, completed_until=start,
        )
        with self._condition:
            if self._stopping:
                raise RuntimeError("Engram scheduler prefetch is closed")
            self.states[key] = state
            self._queue_entered[key] = entered
        if state.segments:
            self._submit(state=state, segment=state.segments[0])

    def _release(self, handles: Sequence[Any], *, cancel: bool = False) -> None:
        for handle in handles:
            if cancel:
                self.runtime.pool.cancel(handle=handle)
            else:
                self.runtime.pool.release(handle=handle)

    def _discard_state(self, state: RequestPrefetchState) -> None:
        handles = []
        with self._condition:
            state.cancelled = True
            for segment in state.segments:
                segment.cancelled = True
                # READING 的 handle 由后台持有，I/O 完成后才释放。
                handles.extend(segment.handles)
                segment.handles = []
            if self.states.get(id(state.request)) is state:
                del self.states[id(state.request)]
                self._queue_entered.pop(id(state.request), None)
        self._release(handles=handles)

    def _submit(self, state: RequestPrefetchState, segment: PrefetchSegment) -> None:
        with self._condition:
            if self._stopping or state.cancelled or segment.cancelled or segment.status not in ("new", "blocked"):
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
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._stopping or bool(self._jobs))
                if self._stopping:
                    return
                state, segment = self._jobs.popleft()
                if state.cancelled or segment.cancelled:
                    continue
                segment.status = "reading"
            handles = []
            try:
                handles = self._read_segment(state=state, segment=segment)
                with self._condition:
                    cancelled = self._stopping or state.cancelled or segment.cancelled
                    if not cancelled:
                        segment.handles = handles
                        segment.status = "pending" if handles else "blocked"
                        segment.pool_blocked = not bool(handles)
                        if handles and all(self.runtime.pool.is_ready(handle=handle) for handle in handles):
                            segment.status = "ready"
                        handles = []
                    self._condition.notify_all()
            except BaseException as error:
                with self._condition:
                    self._failure = error
                    self._stopping = True
                    self._condition.notify_all()
                return
            finally:
                self._release(handles=handles)

    def _read_segment(self, state: RequestPrefetchState, segment: PrefetchSegment) -> list[Any]:
        # CPU int64 [segment_tokens,L,C]；所有层先占槽，池满时整段撤销以免占住无法完成的段。
        ids = self.runtime.hasher.compute_request_row_ids(
            token_ids=state.token_ids, start=segment.start, end=segment.end,
        )
        handles = []
        try:
            for column, layer_id in enumerate(self.runtime.hasher.layer_ids):
                if self._stopping or state.cancelled or segment.cancelled:
                    self._release(handles=handles, cancel=True)
                    return []
                table = self.runtime.open_table(layer_id=layer_id)
                lo = table.rows * self.runtime.tp_rank // self.runtime.tp_size
                hi = table.rows * (self.runtime.tp_rank + 1) // self.runtime.tp_size
                layer_ids = ids[:, column].reshape(-1)
                owner_ids = layer_ids[(layer_ids >= lo) & (layer_ids < hi)].contiguous()
                handle = self.runtime.pool.acquire(layer_id=layer_id, row_ids=owner_ids)
                if handle is None:
                    self._release(handles=handles, cancel=True)
                    return []
                handles.append(handle)
            for layer_id, handle in zip(self.runtime.hasher.layer_ids, handles, strict=True):
                if self._stopping or state.cancelled or segment.cancelled:
                    self._release(handles=handles, cancel=True)
                    return []
                if handle.missing_row_ids.numel():
                    self.runtime.reader.read_rows_to_slots(
                        layer_id=layer_id, row_ids=handle.missing_row_ids, slot_ids=handle.missing_slot_ids,
                        weight_base=self.runtime.pool.weight, scale_base=self.runtime.pool.scale,
                    )
                    self.runtime.pool.mark_ready(slot_ids=handle.missing_slot_ids)
            return handles
        except BaseException:
            self._release(handles=handles, cancel=True)
            raise

    def _next_segment(self, state: RequestPrefetchState) -> PrefetchSegment | None:
        req = state.request
        first_start = state.segments[0].start if state.segments else len(state.token_ids)
        # waiting 期间 prefix 页被驱逐时，旧段不覆盖更早的需求，不能误报整段 READY。
        if int(req.prefix_len) < first_start:
            return None
        start = max(state.completed_until, int(req.prefix_len), int(getattr(req, "already_computed", 0)))
        return next((s for s in state.segments if not s.cancelled and s.end > start), None)

    def _retire_segments(self, state: RequestPrefetchState, end: int) -> None:
        handles = []
        with self._condition:
            state.completed_until = max(state.completed_until, end)
            for segment in state.segments:
                if segment.end <= end:
                    segment.cancelled = True
                    handles.extend(segment.handles)
                    segment.handles = []
        self._release(handles=handles)

    def reconcile(self, scheduler: Any) -> None:
        """只检查状态和释放引用；不在 scheduler 线程读文件或等待后台。"""
        self.raise_if_failed()
        self.runtime.collect_pool_uses()
        waiting = list(scheduler.waiting_queue)
        chunked = getattr(scheduler, "chunked_req", None)
        active = {id(req) for req in waiting}
        if chunked is not None:
            active.add(id(chunked))
        inflight = {(span.request_id, span.generation) for spans in self._batches.values() for span in spans}
        for key, state in list(self.states.items()):
            if not self._valid_request(req=state.request) or (
                key not in active and (key, state.generation) not in inflight
            ):
                self._discard_state(state=state)
        for key in list(self._queue_entered):
            if key not in active and key not in self.states:
                self._queue_entered.pop(key, None)
        # chunked continuation 先于 waiting；所有第一段先排，再补第二段。
        requests = ([chunked] if chunked is not None else []) + waiting
        candidates = []
        for req in requests:
            state = self.states.get(id(req))
            if state is None:
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
        # 已排队任务也按最新顺序调整；唯一正在读取的段继续完成，不取消在途 I/O。
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
        self.reconcile(scheduler=scheduler)
        frame = SchedulingFrame(controller=self, scheduler=scheduler, original_queue=scheduler.waiting_queue)
        token = _active_schedule.set(frame)
        try:
            return original_fn(
                scheduler, prefill_delayer_single_pass=prefill_delayer_single_pass, running_batch=running_batch,
            )
        finally:
            if frame.filtered:
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
        """原生 grammar 就绪和 priority 排序之后，一次 TP MIN 同步整个候选列表。"""
        self.raise_if_failed()
        frame = _active_schedule.get()
        if frame is None or frame.controller is not self or frame.filtered:
            return
        # 原生 priority 已完成最新 prefix 匹配，且可能刚追加 grammar-ready 请求。
        self.reconcile(scheduler=frame.scheduler)
        frame.snapshot = list(waiting_queue)
        if not waiting_queue:
            return
        now = time.monotonic()
        flags = []
        with self._condition:
            for req in waiting_queue:
                state = self.states.get(id(req))
                segment = self._next_segment(state=state) if state is not None else None
                ready = not self._valid_request(req=req) or (
                    state is not None and int(req.prefix_len) >= len(state.token_ids)
                )
                ready = ready or (segment is not None and (
                    segment.status == "ready"
                    or (bool(segment.handles) and all(self.runtime.pool.is_ready(handle=h) for h in segment.handles))
                ))
                entered = state.enqueued_at if state is not None else self._queue_entered.setdefault(id(req), now)
                started = segment.submitted_at if segment is not None else None
                if started is None:
                    started = entered
                elapsed = max(0.0, now - float(started)) * 1000
                unsubmitted = segment is None or segment.status in ("new", "blocked") or segment.pool_blocked
                queue_elapsed = max(0.0, now - entered) * 1000
                allowed = (
                    self.config.policy == "best_effort" or ready
                    or (self.config.policy == "timeout" and elapsed >= self.config.timeout_ms)
                    or (unsubmitted and queue_elapsed >= self.config.timeout_ms)
                )
                flags.append(int(allowed))
        mask = torch.tensor(flags, dtype=torch.int32, device="cpu")
        if self.runtime.tp_size > 1:
            import torch.distributed as dist

            dist.all_reduce(mask, op=dist.ReduceOp.MIN, group=self.cpu_group)
        allowed = mask.tolist()
        frame.blocked = [req for req, flag in zip(waiting_queue, allowed, strict=True) if not flag]
        waiting_queue[:] = [req for req, flag in zip(waiting_queue, allowed, strict=True) if flag]
        frame.filtered = True
        if self.config.log_interval:
            self.runtime.pool.record_deferred(count=len(frame.blocked))

    def record_batch(self, batch: Any) -> None:
        """run_batch 返回后保存不可变区间；overlap 的 batch.copy 保留 forward_iter。"""
        self.raise_if_failed()
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
        """结果处理后仅释放该次 generation 的已计算段，不读取可变的 req.extend_end。"""
        self.raise_if_failed()
        for span in self._batches.pop(int(batch.forward_iter), ()):
            state = self.states.get(span.request_id)
            if state is not None and state.generation == span.generation:
                self._retire_segments(state=state, end=span.end)
                if state.completed_until >= len(state.token_ids):
                    self._discard_state(state=state)
        self._prefill_count += 1
        if self.config.log_interval and self._prefill_count % self.config.log_interval == 0:
            stats = self.runtime.pool.stats()
            stats["phase"] = "prefill"
            stats["tp_rank"] = self.runtime.tp_rank
            logger.info(
                "Engram SSD sched prefetch stats: %s", json.dumps(stats, sort_keys=True),
            )

    def close(self) -> None:
        """停止提交并收齐唯一后台线程；行池和 reader 由 runtime 在此之后关闭。"""
        with self._condition:
            self._stopping = True
            self._jobs.clear()
            self._condition.notify_all()
        self._thread.join()
        for state in list(self.states.values()):
            self._discard_state(state=state)
        self._batches.clear()
        self._queue_entered.clear()
