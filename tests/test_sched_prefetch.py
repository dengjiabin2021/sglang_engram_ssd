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
        hasher=FakeHasher(), pool=FakePool(), reader=FakeReader(), tp_size=1, tp_rank=0,
        open_table=lambda layer_id: SimpleNamespace(rows=16),
        collect_pool_uses=lambda: None,
    )
    scheduler = SimpleNamespace(waiting_queue=[], chunked_req=None, tp_group=SimpleNamespace(cpu_group=object()))
    manager = sched_prefetch.SchedulerPrefetchController(runtime=runtime, scheduler=scheduler)
    try:
        yield manager, runtime, scheduler, config
    finally:
        runtime.reader.proceed.set()
        manager.close()


def _wait_segment(manager: Any, req: Any, index: int = 0, status: str = "ready") -> Any:
    state = manager.states[id(req)]
    with manager._condition:
        assert manager._condition.wait_for(
            lambda: state.segments[index].status == status or manager._failure is not None, timeout=5,
        )
    manager.raise_if_failed()
    assert state.segments[index].status == status
    return state


def _batch(req: Any, start: int, length: int, iteration: int) -> SimpleNamespace:
    return SimpleNamespace(
        reqs=[req], prefix_lens=[start], extend_lens=[length], forward_iter=iteration,
        decoding_reqs=None, forward_mode=SimpleNamespace(is_extend=lambda: True),
    )


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
    assert all(row >= 8 for _, rows in runtime.reader.calls for row in rows)
    assert sum(rows.count(9) for _, rows in runtime.reader.calls) == 2  # 两个不同层各一次。
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
        assert mask.tolist() == [1, 1]
        mask[1] = 0

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


def test_empty_queue_does_not_call_collective(environment: Environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """A7：候选为空不发TP集合通信，原函数异常也恢复调度上下文。"""
    manager, runtime, scheduler, _ = environment
    runtime.tp_size = 8
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **k: pytest.fail("unexpected collective"))

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> None:
        sched_prefetch.filter_active_waiting_queue(waiting_queue=self.waiting_queue)
        raise ValueError("original failure")

    with pytest.raises(ValueError, match="original failure"):
        manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        )
    assert sched_prefetch._active_schedule.get() is None


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
    assert not runtime.reader.calls


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
    runtime.reader.proceed.clear()
    req = _request("abort", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    assert runtime.reader.entered.wait(timeout=5)
    scheduler.waiting_queue = []
    manager.reconcile(scheduler=scheduler)
    assert id(req) not in manager.states
    assert runtime.pool.stats()["referenced"] > 0
    runtime.reader.proceed.set()
    manager.close()
    assert runtime.pool.stats()["referenced"] == 0


def test_background_error_is_rethrown(environment: Environment) -> None:
    """A7：后台SSD失败在下一次主路径传播，且不遗留PENDING或引用。"""
    manager, runtime, scheduler, _ = environment
    runtime.reader.error = OSError("synthetic I/O error")
    req = _request("failure", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    with manager._condition:
        assert manager._condition.wait_for(lambda: manager._failure is not None, timeout=5)
    with pytest.raises(RuntimeError, match="scheduler prefetch failed"):
        manager.raise_if_failed()
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
