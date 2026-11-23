"""用 CPU 行池替身验证分段预取、调度队列恢复和 overlap 结果回收。"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_engram_ssd import sched_prefetch


Environment = tuple[sched_prefetch.SchedulerPrefetchController, SimpleNamespace, SimpleNamespace, SimpleNamespace]


@dataclass
class FakeHandle:
    keys: tuple[tuple[int, int], ...]
    missing_row_ids: torch.Tensor
    missing_slot_ids: torch.Tensor
    released: bool = False


class FakePool:
    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self.entries: dict[tuple[int, int], list[Any]] = {}
        self.slot_keys: dict[int, tuple[int, int]] = {}
        self.weight = torch.empty(1, dtype=torch.uint8)
        self.scale = torch.empty(1, dtype=torch.uint8)
        self._lock = threading.Lock()
        self._next_slot = 0
        self.cancelled = 0
        self.deferred = 0
        self.submitted = 0

    def acquire(self, *, layer_id: int, row_ids: torch.Tensor) -> FakeHandle | None:
        with self._lock:
            keys = tuple(dict.fromkeys((layer_id, row) for row in row_ids.tolist()))
            missing = [key for key in keys if key not in self.entries]
            if len(self.entries) + len(missing) > self.capacity:
                return None
            rows, slots = [], []
            for key in keys:
                if key not in self.entries:
                    slot = self._next_slot
                    self._next_slot += 1
                    self.entries[key] = [slot, 0, False]
                    self.slot_keys[slot] = key
                    rows.append(key[1])
                    slots.append(slot)
                self.entries[key][1] += 1
            return FakeHandle(
                keys=keys, missing_row_ids=torch.tensor(rows, dtype=torch.int64),
                missing_slot_ids=torch.tensor(slots, dtype=torch.int64),
            )

    def mark_ready(self, *, slot_ids: torch.Tensor) -> None:
        with self._lock:
            for slot in slot_ids.tolist():
                self.entries[self.slot_keys[slot]][2] = True

    def is_ready(self, *, handle: FakeHandle) -> bool:
        with self._lock:
            return all(self.entries[key][2] for key in handle.keys)

    def release(self, *, handle: FakeHandle) -> None:
        with self._lock:
            if not handle.released:
                for key in handle.keys:
                    self.entries[key][1] -= 1
                    assert self.entries[key][1] >= 0
                handle.released = True

    def cancel(self, *, handle: FakeHandle) -> None:
        self.release(handle=handle)
        with self._lock:
            self.cancelled += 1
            for slot in handle.missing_slot_ids.tolist():
                key = self.slot_keys.get(slot)
                if key is not None and self.entries[key][1:] == [0, False]:
                    del self.entries[key]
                    del self.slot_keys[slot]

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "referenced": sum(entry[1] for entry in self.entries.values()),
                "referenced_slots": sum(entry[1] > 0 for entry in self.entries.values()),
            }

    def record_deferred(self, *, count: int) -> None:
        self.deferred += count

    def record_submitted(self, *, rows: int) -> None:
        self.submitted += rows


class FakeHasher:
    layer_ids = (1, 14)

    def compute_request_row_ids(self, token_ids: tuple[int, ...], start: int, end: int) -> torch.Tensor:
        rows = torch.tensor(token_ids[start:end], dtype=torch.int64) % 16
        return rows[:, None, None].expand(-1, 2, 1).contiguous()


class FakeReader:
    def __init__(self) -> None:
        self.calls: list[tuple[int, list[int]]] = []
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.error: BaseException | None = None

    def read_rows_to_slots(
        self, layer_id: int, row_ids: torch.Tensor, slot_ids: torch.Tensor,
        weight_base: torch.Tensor, scale_base: torch.Tensor,
    ) -> None:
        self.entered.set()
        if not self.proceed.wait(timeout=5):
            raise RuntimeError("test reader was not released")
        if self.error is not None:
            raise self.error
        self.calls.append((layer_id, row_ids.tolist()))


def _request(rid: str, tokens: list[int], prefix: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        rid=rid, origin_input_ids=tokens, output_ids=[], prefix_len=prefix, already_computed=0,
        finished_reason=None, to_finish=None, multimodal_inputs=None,
        time_stats=SimpleNamespace(wait_queue_entry_time=time.monotonic() - 1),
    )


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[Environment]:
    config = SimpleNamespace(segment_tokens=2, policy="timeout", timeout_ms=500, log_interval=0)
    monkeypatch.setattr(sched_prefetch, "get_config", lambda: config)
    runtime = SimpleNamespace(
        hasher=FakeHasher(), pool=FakePool(), reader=FakeReader(), prefetch_reader=FakeReader(), tp_size=1, tp_rank=0,
        open_table=lambda layer_id: SimpleNamespace(rows=16),
        collect_pool_uses=lambda: None,
    )
    scheduler = SimpleNamespace(
        waiting_queue=[], chunked_req=None, chunked_prefill_size=2, tp_group=SimpleNamespace(cpu_group=object()),
    )
    manager = sched_prefetch.SchedulerPrefetchController(runtime=runtime, scheduler=scheduler)
    try:
        yield manager, runtime, scheduler, config
    finally:
        runtime.prefetch_reader.proceed.set()
        manager.close()


def _wait_segment(manager: Any, req: Any, index: int = 0, status: str = "ready") -> Any:
    state = manager.states[id(req)]
    if not state.initialized:
        manager._initialize_segments(state=state)
        if state.segments:
            manager._submit(state=state, segment=state.segments[0])
    with manager._condition:
        assert manager._condition.wait_for(
            lambda: state.segments[index].status == status or manager._failure is not None, timeout=5,
        )
    assert manager._failure is None, f"Unexpected background error: {manager._failure!r}"
    assert state.segments[index].status == status
    return state


def _batch(req: Any, start: int, length: int, iteration: int) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=[req], prefix_lens=[start], extend_lens=[length], forward_iter=iteration,
        decoding_reqs=None, forward_mode=SimpleNamespace(is_extend=lambda: True),
    )


def _filter_once(manager: Any, scheduler: Any) -> list[str]:
    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> list[str]:
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        return [req.rid for req in self.waiting_queue]

    return manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    )


def test_registration_waits_for_priority_prefix_snapshot(environment: Environment) -> None:
    """A7/P3：初次入队不读 SSD，priority 更新 prefix 后才切分未命中的后缀。"""
    manager, runtime, scheduler, config = environment
    config.policy = "best_effort"
    req = _request("prefix-hit", [1, 2, 3, 4, 5, 6])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    state = manager.states[id(req)]
    assert not state.initialized
    assert not state.segments
    assert not runtime.prefetch_reader.calls
    req.prefix_len = 2  # 模拟支持 prefix 匹配的原生 priority 完成缓存查询。
    assert _filter_once(manager=manager, scheduler=scheduler) == [req.rid]
    _wait_segment(manager=manager, req=req)
    _wait_segment(manager=manager, req=req, index=1)
    assert [(segment.start, segment.end) for segment in state.segments] == [(2, 4), (4, 6)]
    assert all(row not in (1, 2) for _, rows in runtime.prefetch_reader.calls for row in rows)
    assert not runtime.reader.calls  # 后台独立 reader，不占用 forward 的 io_uring 实例。


def test_priority_without_prefix_refresh_uses_existing_snapshot(environment: Environment) -> None:
    """A7/P3：原生策略未刷新 prefix 时不自行改 KV 状态，使用当前快照预取。"""
    manager, _, scheduler, config = environment
    config.policy = "best_effort"
    req = _request("no-fast-prefix", [1, 2], prefix=0)
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    assert _filter_once(manager=manager, scheduler=scheduler) == [req.rid]
    state = _wait_segment(manager=manager, req=req)
    assert state.segments[0].start == 0


def test_cross_segment_chunk_requires_both_segments(environment: Environment) -> None:
    """A7/P4：chunk 从段内开始并跨越两段时，不能仅凭首段 READY 放行。"""
    manager, runtime, scheduler, config = environment
    config.policy = "wait_complete"
    manager.chunked_tokens = 3
    runtime.prefetch_reader.proceed.clear()
    req = _request("cross-segment", [1, 2, 3, 4])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = manager.states[id(req)]
    manager._initialize_segments(state=state)
    state.segments[0].status = "ready"
    req.prefix_len = 1
    assert _filter_once(manager=manager, scheduler=scheduler) == []
    segments, covered = manager._segments_for_next_chunk(state=state)
    assert covered
    assert segments == state.segments
    runtime.prefetch_reader.proceed.set()
    _wait_segment(manager=manager, req=req, index=1)
    assert _filter_once(manager=manager, scheduler=scheduler) == [req.rid]


@pytest.mark.parametrize("chunked_tokens", [0, 6])
def test_chunk_larger_than_two_segments_uses_entry_deadline(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, chunked_tokens: int,
) -> None:
    """A7/P4/N3：大 chunk 或未启用 chunk 的完整 prompt 超过两段时，按入队超时放行。"""
    manager, _, scheduler, config = environment
    clock = [100.0]
    monkeypatch.setattr(sched_prefetch.time, "monotonic", lambda: clock[0])
    config.policy = "wait_complete"
    manager.chunked_tokens = chunked_tokens
    req = _request("three-segments", [1, 2, 3, 4, 5, 6])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = _wait_segment(manager=manager, req=req)
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=req, index=1)
    assert state.segments[2].status == "new"
    assert _filter_once(manager=manager, scheduler=scheduler) == []
    clock[0] += 0.6
    assert _filter_once(manager=manager, scheduler=scheduler) == [req.rid]


def test_pool_full_retries_only_after_release_and_reuses_hash(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7/P5：容量未变化时不重试；释放通知后复用 CPU 行号，不重复计算 hash。"""
    manager, runtime, scheduler, _ = environment
    calls = []
    compute = runtime.hasher.compute_request_row_ids

    def counted(*, token_ids: tuple[int, ...], start: int, end: int) -> torch.Tensor:
        calls.append((start, end))
        return compute(token_ids=token_ids, start=start, end=end)

    monkeypatch.setattr(runtime.hasher, "compute_request_row_ids", counted)
    runtime.pool.capacity = 2  # 第一层能占槽，第二层失败，必须回滚且不能触发自己的重试。
    req = _request("retry", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = _wait_segment(manager=manager, req=req, status="blocked")
    cached_ids = state.segments[0].row_ids
    for _ in range(4):
        manager.reconcile(scheduler=scheduler)
    assert state.segments[0].status == "blocked"
    assert calls == [(0, 2)]
    assert state.segments[0].row_ids is cached_ids
    runtime.pool.capacity = 1024
    manager.notify_pool_release()  # 模拟段引用或 CUDA event 完成后的临时引用归还。
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=req)
    assert calls == [(0, 2)]
    assert state.segments[0].row_ids is None


def test_cancel_during_hash_does_not_restore_segment_cache(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7/P5：hash 计算期间取消请求，不重新挂回已清理状态，也不提交 SSD 读取。"""
    manager, runtime, scheduler, _ = environment
    entered = threading.Event()
    proceed = threading.Event()
    compute = runtime.hasher.compute_request_row_ids

    def delayed(*, token_ids: tuple[int, ...], start: int, end: int) -> torch.Tensor:
        entered.set()
        if not proceed.wait(timeout=5):
            raise RuntimeError("test hash was not released")
        return compute(token_ids=token_ids, start=start, end=end)

    monkeypatch.setattr(runtime.hasher, "compute_request_row_ids", delayed)
    req = _request("cancel-hash", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = manager.states[id(req)]
    manager._initialize_segments(state=state)
    manager._submit(state=state, segment=state.segments[0])
    try:
        assert entered.wait(timeout=5)
        scheduler.waiting_queue = []
        manager.reconcile(scheduler=scheduler)
        assert state.cancelled
    finally:
        proceed.set()
    manager.close()
    assert state.segments[0].row_ids is None
    assert not runtime.prefetch_reader.calls


@pytest.mark.parametrize("local_failure", [False, True])
def test_tp_background_failure_reaches_shared_candidate_min(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, local_failure: bool,
) -> None:
    """A7/P7：同一次非空候选检查中，任一 rank 后台失败经一次 MIN 传播到全部 rank。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    req = _request("failed-rank", [1, 2])
    scheduler.waiting_queue = [req]
    cause = OSError("synthetic background failure")
    if local_failure:
        with manager._condition:
            manager._failure = cause
        manager.register_request(req=req)  # 不允许入队 Hook 抢先抛异常、跳过后面的集合通信。
    calls = []

    def reduce(mask: torch.Tensor, op: Any, group: Any) -> None:
        calls.append(mask.clone())
        if local_failure:
            assert mask[0].item() == -1
        assert mask.numel() == 2  # 一个独立失败标志加一个候选，不增加第二次归约。
        mask[0] = -1  # 模拟任一远端 rank 的失败标记，不把它混入放行标志。

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        _filter_once(manager=manager, scheduler=scheduler)
    assert len(calls) == 1
    assert error.value.__cause__ is (cause if local_failure else None)
    assert scheduler.waiting_queue == [req]


@pytest.mark.parametrize("local_failure", [False, True])
def test_raw_early_return_still_propagates_background_failure(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, local_failure: bool,
) -> None:
    """A7/N1：仍有 waiting 请求的 raw 早退参加一次失败 MIN，本地或远端失败统一传播。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    scheduler.waiting_queue = [_request("raw-return", [1, 2])]
    cause = OSError("synthetic early-return failure")
    if local_failure:
        with manager._condition:
            manager._failure = cause
    calls = []

    def reduce(mask: torch.Tensor, op: Any, group: Any) -> None:
        calls.append(mask.clone())
        assert mask.tolist() == ([-1] if local_failure else [1])
        mask[0] = -1

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> None:
        return None

    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        )
    assert len(calls) == 1
    assert error.value.__cause__ is (cause if local_failure else None)
    assert sched_prefetch._active_schedule.get() is None


def test_full_batch_early_return_starts_waiting_prefetch(environment: Environment) -> None:
    """A7/N2：batch 满导致 priority 前早退时，仍在后台预取 waiting 首段并保留两段上限。"""
    manager, runtime, scheduler, _ = environment
    req = _request("waiting-behind-full-batch", [1, 2, 3, 4, 5, 6], prefix=2)
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    runtime.prefetch_reader.proceed.clear()
    state = manager.states[id(req)]
    assert not state.initialized

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> str:
        return "batch-full"  # 模拟原生早退，不调用 priority Hook。

    try:
        result = manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        )
        assert result == "batch-full"
        # run_prefill 已返回，后台 reader 才被允许完成；调度线程没有等待 SSD。
        assert runtime.prefetch_reader.entered.wait(timeout=5)
        assert state.initialized
        assert [(segment.start, segment.end) for segment in state.segments] == [(2, 4), (4, 6)]
        assert state.segments[0].status == "reading"
        assert scheduler.waiting_queue == [req]
        assert not runtime.reader.calls
    finally:
        runtime.prefetch_reader.proceed.set()
    _wait_segment(manager=manager, req=req)
    _wait_segment(manager=manager, req=req, index=1)


@pytest.mark.parametrize("use_priority", [False, True])
@pytest.mark.parametrize("local_failure", [False, True])
@pytest.mark.parametrize("residual_state", [False, True])
def test_decode_without_prefill_demand_skips_failure_min(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, use_priority: bool,
    local_failure: bool, residual_state: bool,
) -> None:
    """A7/N4：无 waiting／chunked 时只对账一次，不按本地错误或残留 state 决定通信。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    if residual_state:
        req = _request("residual-prefetch-state", [1, 2])
        scheduler.waiting_queue = [req]
        manager.register_request(req=req)
        scheduler.waiting_queue = []
    cause = OSError("synthetic deferred failure")
    if local_failure:
        with manager._condition:
            manager._failure = cause
            manager._stopping = True
    reconciled = []
    reconcile = manager.reconcile

    def counted_reconcile(*, scheduler: Any) -> None:
        reconciled.append(scheduler)
        reconcile(scheduler=scheduler)

    def reject_collective(mask: torch.Tensor, op: Any, group: Any) -> None:
        raise AssertionError("Pure decode must not issue a CPU failure collective")

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> None:
        if use_priority:
            sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        return None

    monkeypatch.setattr(manager, "reconcile", counted_reconcile)
    monkeypatch.setattr(torch.distributed, "all_reduce", reject_collective)
    assert manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    ) is None
    assert reconciled == [scheduler]
    assert manager._failure is (cause if local_failure else None)
    # 失败 rank 保留状态、健康 rank 已清理，仍必须在两边都跳过，而不是按 states 大小分流。
    assert bool(manager.states) == (residual_state and local_failure)
    assert sched_prefetch._active_schedule.get() is None


@pytest.mark.parametrize("local_failure", [False, True])
def test_new_waiting_request_resumes_deferred_failure_min(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, local_failure: bool,
) -> None:
    """A7/N4：纯 decode 暂存失败，新 waiting 请求到达后的第一次 raw 调度恢复一次 MIN。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    cause = OSError("synthetic deferred background failure")
    if local_failure:
        with manager._condition:
            manager._failure = cause
            manager._stopping = True
    calls = []

    def reduce(mask: torch.Tensor, op: Any, group: Any) -> None:
        calls.append(mask.clone())
        assert mask.tolist() == ([-1] if local_failure else [1])
        mask[0] = -1  # 健康 rank 也必须通过同一次 MIN 得知远端积压的错误。

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> None:
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    assert manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    ) is None
    assert not calls
    req = _request("first-waiting-after-decode", [0, 1])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    if local_failure:
        assert id(req) not in manager.states  # 失败后不再登记，但原生 waiting 仍触发统一归约。
    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        )
    assert len(calls) == 1
    assert error.value.__cause__ is (cause if local_failure else None)
    assert scheduler.waiting_queue == [req]
    assert sched_prefetch._active_schedule.get() is None


def test_grammar_added_inside_raw_resumes_waiting_prefetch(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7/N2/N4：入口无 waiting，原生内部追加 grammar-ready 后仍预取并恢复失败同步。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    req = _request("grammar-ready-during-raw", [0, 1, 0, 1])
    runtime.prefetch_reader.proceed.clear()
    calls = []

    def reduce(mask: torch.Tensor, op: Any, group: Any) -> None:
        calls.append(mask.clone())
        assert mask.tolist() == [1]

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> str:
        self.waiting_queue.append(req)
        manager.register_request(req=req)
        return "batch-full-after-grammar"  # 原生 grammar 追加后仍可能因 batch 满在 priority 前返回。

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    try:
        assert manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        ) == "batch-full-after-grammar"
        assert runtime.prefetch_reader.entered.wait(timeout=5)
        assert manager.states[id(req)].initialized
        assert scheduler.waiting_queue == [req]
        assert len(calls) == 1
    finally:
        runtime.prefetch_reader.proceed.set()
    _wait_segment(manager=manager, req=req)


@pytest.mark.parametrize("use_priority", [False, True])
@pytest.mark.parametrize("local_failure", [False, True])
def test_chunked_continuation_without_waiting_keeps_failure_min(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, use_priority: bool, local_failure: bool,
) -> None:
    """A7/N1/N4：waiting 为空但存在 chunked continuation，仍统一传播本地或远端后台错误。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    scheduler.chunked_req = _request("chunked-without-waiting", [0, 1, 0, 1])
    cause = OSError("synthetic chunked background failure")
    if local_failure:
        with manager._condition:
            manager._failure = cause
            manager._stopping = True
    calls = []

    def reduce(mask: torch.Tensor, op: Any, group: Any) -> None:
        calls.append(mask.clone())
        assert mask.tolist() == ([-1] if local_failure else [1])
        mask[0] = -1

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> None:
        if use_priority:
            sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        )
    assert len(calls) == 1
    assert error.value.__cause__ is (cause if local_failure else None)
    assert sched_prefetch._active_schedule.get() is None


def test_queue_entry_owns_request_and_cleans_inactive_identity(environment: Environment) -> None:
    """A7/P10：时间记录持有请求本身；相同 rid 的不同请求不混用身份或入队时间。"""
    manager, _, scheduler, _ = environment
    first = _request("same-rid", [1])
    second = _request("same-rid", [2])
    scheduler.waiting_queue = [first, second]
    manager.register_request(req=first)
    manager.register_request(req=second)
    assert manager._queue_entered[id(first)].request is first
    assert manager._queue_entered[id(second)].request is second
    assert manager.states[id(first)].request is first
    assert manager.states[id(second)].request is second
    scheduler.waiting_queue = [second]
    manager.reconcile(scheduler=scheduler)
    assert id(first) not in manager._queue_entered
    assert id(first) not in manager.states
    assert manager._queue_entered[id(second)].request is second


def test_background_owner_reads_and_two_segment_limit(environment: Environment) -> None:
    """A7：单后台线程只读本rank行，重复行不重复读取，每请求最多持有两段。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_rank, runtime.tp_size = 1, 2
    req = _request("owner", [1, 9, 9, 10, 11, 12])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = _wait_segment(manager=manager, req=req)
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=req, index=1)
    assert state.segments[2].status == "new"
    assert all(row >= 8 for _, rows in runtime.prefetch_reader.calls for row in rows)
    assert sum(rows.count(9) for _, rows in runtime.prefetch_reader.calls) == 2  # 两个不同层各一次。
    assert manager._thread.is_alive()


def test_queue_filter_restores_blocked_and_new_requests(environment: Environment) -> None:
    """A7：原生排序、移除已调度项和追加请求后，blocked及新项不丢失或重复。"""
    manager, _, scheduler, config = environment
    config.policy = "wait_complete"
    ready = _request("ready", [1, 2])
    blocked = _request("blocked", [3, 4])
    grammar = _request("grammar-ready-inside-original", [5, 6])
    appended = _request("preempted-inside-original", [7, 8])
    scheduler.waiting_queue = [blocked, ready]
    original_queue = scheduler.waiting_queue
    manager.register_request(req=ready)
    _wait_segment(manager=manager, req=ready)

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> str:
        self.waiting_queue.append(grammar)
        self.waiting_queue.reverse()
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        assert self.waiting_queue == [ready]
        self.waiting_queue = []
        self.waiting_queue.append(appended)
        return "original-result"

    result = manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    )
    assert result == "original-result"
    assert scheduler.waiting_queue is original_queue
    assert scheduler.waiting_queue == [grammar, blocked, appended]
    assert len({id(req) for req in scheduler.waiting_queue}) == 3


def test_tp_timeout_uses_one_vector_min(environment: Environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """A7：整个候选列表只做一次CPU MIN，远端未放行时本地超时也不能单独调度。"""
    manager, runtime, scheduler, config = environment
    config.timeout_ms = 0
    runtime.tp_size = 8
    scheduler.waiting_queue = [_request("one", [1]), _request("two", [2])]
    calls = []

    def reduce(mask: torch.Tensor, op: Any, group: Any) -> None:
        calls.append((mask.clone(), op, group))
        assert mask.tolist() == [1, 1, 1]
        mask[2] = 0

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> list[str]:
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        return [req.rid for req in self.waiting_queue]

    result = manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    )
    assert result == ["one"]
    assert len(calls) == 1
    assert calls[0][0].device.type == "cpu"
    assert calls[0][2] is scheduler.tp_group.cpu_group
    assert [req.rid for req in scheduler.waiting_queue] == ["one", "two"]


def test_empty_queue_skips_failure_sync_and_restores_context(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7/N4：无 prefill 需求的空候选不通信，原函数异常仍恢复调度上下文。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    calls = []

    def reduce(mask: torch.Tensor, op: Any, group: Any) -> None:
        calls.append(mask.clone())
        raise AssertionError("Empty decode scheduling must not issue a CPU collective")

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> None:
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        raise ValueError("original failure")

    with pytest.raises(ValueError, match="original failure"):
        manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        )
    assert sched_prefetch._active_schedule.get() is None
    assert not calls


def test_overlap_release_uses_batch_span_not_mutable_request(environment: Environment) -> None:
    """A7：下一chunk已修改Req时，旧结果仅释放旧batch明确完成的段。"""
    manager, runtime, scheduler, _ = environment
    req = _request("chunks", [1, 2, 3, 4])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = _wait_segment(manager=manager, req=req)
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=req, index=1)
    first = _batch(req=req, start=0, length=2, iteration=1)
    manager.record_batch(batch=first)
    req.prefix_len, req.extend_end = 2, 4
    manager.release_batch(batch=first)
    assert state.segments[0].cancelled
    assert not state.segments[1].cancelled
    assert runtime.pool.stats()["referenced"] == 4
    second = _batch(req=req, start=2, length=2, iteration=2)
    manager.record_batch(batch=second)
    manager.release_batch(batch=second)
    assert id(req) not in manager.states
    assert runtime.pool.stats()["referenced"] == 0


def test_retraction_old_result_cannot_release_new_generation(environment: Environment) -> None:
    """A7：同一Req撤回重登后，旧forward结果不能释放新generation的行池引用。"""
    manager, runtime, scheduler, _ = environment
    req = _request("retract", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    old = _wait_segment(manager=manager, req=req)
    batch = _batch(req=req, start=0, length=2, iteration=1)
    manager.record_batch(batch=batch)
    req.output_ids = [3]
    manager.register_request(req=req, is_retracted=True)
    new = _wait_segment(manager=manager, req=req)
    assert new.generation != old.generation
    referenced = runtime.pool.stats()["referenced"]
    manager.release_batch(batch=batch)
    assert runtime.pool.stats()["referenced"] == referenced
    assert not new.segments[0].cancelled


def test_partial_segment_reservation_rolls_back_pending(environment: Environment) -> None:
    """A7：跨层占槽中途池满时回收未读PENDING，避免无法完成的段永久占池。"""
    manager, runtime, scheduler, _ = environment
    runtime.pool.capacity = 2
    req = _request("full", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    _wait_segment(manager=manager, req=req, status="blocked")
    assert runtime.pool.stats()["referenced"] == 0
    assert not runtime.pool.entries
    assert not runtime.prefetch_reader.calls


def test_wait_complete_pool_full_uses_entry_timeout(environment: Environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """A7：wait_complete下未能占槽的段仍按入队超时放行，重提不能重置计时或饿死。"""
    manager, runtime, scheduler, config = environment
    clock = [100.0]
    monkeypatch.setattr(sched_prefetch.time, "monotonic", lambda: clock[0])
    config.policy = "wait_complete"
    runtime.pool.capacity = 0
    req = _request("full-with-deadline", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    _wait_segment(manager=manager, req=req, status="blocked")
    clock[0] += 0.6

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> list[str]:
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        return [candidate.rid for candidate in self.waiting_queue]

    result = manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    )
    assert result == [req.rid]


def test_prefix_shrink_is_not_mistaken_for_ready(environment: Environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """A7：prefix命中缩小时旧READY段不覆盖更早token，先暂缓再按入队超时补读。"""
    manager, _, scheduler, config = environment
    clock = [100.0]
    monkeypatch.setattr(sched_prefetch.time, "monotonic", lambda: clock[0])
    config.policy = "wait_complete"
    req = _request("shrinking-prefix", [1, 2, 3, 4], prefix=2)
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    _wait_segment(manager=manager, req=req)
    req.prefix_len = 0

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> list[str]:
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        return [candidate.rid for candidate in self.waiting_queue]

    first = manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    )
    assert first == []
    assert scheduler.waiting_queue == [req]
    clock[0] += 0.6
    second = manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    )
    assert second == [req.rid]


def test_abort_inflight_defers_release_until_read_finishes(environment: Environment) -> None:
    """A7：请求中止不提前复用在途I/O槽，读完成后唯一后台线程回收引用。"""
    manager, runtime, scheduler, _ = environment
    runtime.prefetch_reader.proceed.clear()
    req = _request("abort", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = manager.states[id(req)]
    manager._initialize_segments(state=state)
    manager._submit(state=state, segment=state.segments[0])
    assert runtime.prefetch_reader.entered.wait(timeout=5)
    scheduler.waiting_queue = []
    manager.reconcile(scheduler=scheduler)
    assert id(req) not in manager.states
    assert runtime.pool.stats()["referenced"] > 0
    runtime.prefetch_reader.proceed.set()
    manager.close()
    assert runtime.pool.stats()["referenced"] == 0


def test_background_error_is_rethrown(environment: Environment) -> None:
    """A7：后台SSD失败在下一次主路径传播，且不遗留PENDING或引用。"""
    manager, runtime, scheduler, _ = environment
    runtime.prefetch_reader.error = OSError("synthetic I/O error")
    req = _request("failure", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = manager.states[id(req)]
    manager._initialize_segments(state=state)
    manager._submit(state=state, segment=state.segments[0])
    with manager._condition:
        assert manager._condition.wait_for(lambda: manager._failure is not None, timeout=5)
    with pytest.raises(RuntimeError, match="failed on a TP rank"):
        _filter_once(manager=manager, scheduler=scheduler)
    assert runtime.pool.stats()["referenced"] == 0
    assert not runtime.pool.entries


def test_log_disabled_does_not_query_pool_stats(environment: Environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1：LOG_INTERVAL=0时调度不额外读取统计或生成idle日志。"""
    manager, runtime, scheduler, _ = environment
    monkeypatch.setattr(runtime.pool, "stats", lambda: pytest.fail("disabled logging requested pool stats"))
    manager.reconcile(scheduler=scheduler)
    manager.reconcile(scheduler=scheduler)


def test_idle_final_stats_logged_once_and_rearmed(
    environment: Environment, caplog: pytest.LogCaptureFixture,
) -> None:
    """A7：空闲且引用归零时只记录一次JSON，新请求生命周期结束后重新记录。"""
    manager, _, scheduler, config = environment
    config.log_interval = 1
    caplog.set_level(logging.INFO, logger=sched_prefetch.__name__)
    manager.reconcile(scheduler=scheduler)
    manager.reconcile(scheduler=scheduler)
    messages = [record.getMessage() for record in caplog.records if "sched prefetch stats:" in record.getMessage()]
    assert len(messages) == 1
    req = _request("new-cycle", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    _wait_segment(manager=manager, req=req)
    req.finished_reason = object()
    scheduler.waiting_queue = []
    manager.reconcile(scheduler=scheduler)
    manager.reconcile(scheduler=scheduler)
    messages = [record.getMessage() for record in caplog.records if "sched prefetch stats:" in record.getMessage()]
    assert len(messages) == 2
    for message in messages:
        stats = json.loads(message.removeprefix("Engram SSD sched prefetch stats: "))
        assert stats["phase"] == "idle"
        assert stats["referenced_slots"] == 0
