"""用 CPU 行池和 reader 替身验证分块预取、紧急行以及调度回收。"""

from __future__ import annotations

import json
import logging
import threading
from array import array
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_engram_ssd import sched_prefetch
from sglang_engram_ssd.query.interface import PoolResolution
from sglang_engram_ssd.runtime import Batch, EngramSsdRuntime


Environment = tuple[sched_prefetch.SchedulerPrefetchController, SimpleNamespace, SimpleNamespace, SimpleNamespace]


@dataclass
class FakeHandle:
    """由 FakePool 创建的槽引用，随任务存活至归还；引用变化由池锁保护。"""

    keys: tuple[tuple[int, int], ...]  # 所有已分配槽的 (层, 行) 键。
    row_ids: torch.Tensor  # 去重后的 CPU int64 [Q] 全局行号。
    slot_ids: torch.Tensor  # CPU int64 [Q] 槽号。
    released: bool = False


class FakePool:
    """为控制器测试提供 CPU 行池，fixture 创建，池锁保护引用、状态和 LRU。"""

    def __init__(self, capacity: int = 1024) -> None:
        """按 capacity 创建槽和空索引，输出由测试持有的 CPU 行表。"""
        self.capacity = capacity
        self.entries: dict[tuple[int, int], list[Any]] = {}  # (层, 行) -> [槽号, 引用数, 是否 READY]。
        self.slot_keys: dict[int, tuple[int, int]] = {}
        self.weight_stride, self.scale_stride = 32, 1
        self.weight = torch.empty((capacity, self.weight_stride), dtype=torch.uint8, device="cpu")
        self.scale = torch.empty((capacity, self.scale_stride), dtype=torch.uint8, device="cpu")
        self._lock = threading.Lock()
        self._lru: dict[tuple[int, int], None] = {}  # 最久未使用的槽在前。
        self.submitted = 0

    def acquire(self, *, layer_id: int, row_ids: torch.Tensor) -> FakeHandle | None:
        """为 layer_id 的 CPU int64 [Q] 全局行原子占槽，可重复；容量不足时返回 None。"""
        with self._lock:
            keys = tuple(dict.fromkeys((layer_id, row) for row in row_ids.tolist()))
            missing = [key for key in keys if key not in self.entries]
            room = self.capacity - len(self.entries) + len(self._lru) - sum(key in self._lru for key in keys)
            if len(missing) > room:
                return None
            for key in keys:
                if key in self.entries:
                    self.entries[key][1] += 1
                    self._lru.pop(key, None)
            for key in missing:
                if len(self.entries) == self.capacity:
                    self._remove(key=next(iter(self._lru)))
                slot = next(i for i in range(self.capacity) if i not in self.slot_keys)
                self.entries[key] = [slot, 1, False]
                self.slot_keys[slot] = key
            return FakeHandle(
                keys=keys, row_ids=_ids([row for _, row in keys]),
                slot_ids=_ids([self.entries[key][0] for key in keys]),
            )

    def resolve(self, *, layer_id: int, row_ids: torch.Tensor) -> PoolResolution:
        """按 layer_id 和 CPU int64 [Q] 行号解析 READY；返回槽号、掩码和本次引用。"""
        keys = [(layer_id, row) for row in row_ids.tolist()]
        with self._lock:
            ready = tuple(key for key in dict.fromkeys(keys) if key in self.entries and self.entries[key][2])
            for key in ready:
                self.entries[key][1] += 1
                self._lru.pop(key, None)
            slots = _ids([self.entries[key][0] if key in ready else -1 for key in keys])
            return PoolResolution(slot_ids=slots, hit_mask=slots >= 0, handle=FakeHandle(ready, _ids([]), _ids([])))

    def available_slots(self) -> int:
        """无参数，返回当前空闲槽及可淘汰的缓存槽总数。"""
        with self._lock:
            return self.capacity - len(self.entries) + len(self._lru)

    def slots_needed(self, *, layer_id: int, row_ids: torch.Tensor, limit: int = -1) -> int:
        """统计 layer_id 的 CPU int64 [Q] 行所需槽数，超过非负 limit 时返回 limit + 1。"""
        with self._lock:
            keys = {(layer_id, row) for row in row_ids.tolist()}
            needed = sum(key not in self.entries or key in self._lru for key in keys)
            return needed if limit < 0 else min(needed, limit + 1)

    def pending_slots(self, *, layer_id: int, row_ids: torch.Tensor) -> torch.Tensor:
        """查询 layer_id 的 CPU int64 [Q] 行，返回等长槽号，非 PENDING 为 -1。"""
        with self._lock:
            return _ids([
                self.entries[key][0] if key in self.entries and not self.entries[key][2] else -1
                for key in ((layer_id, row) for row in row_ids.tolist())
            ])

    def mark_ready(self, *, slot_ids: torch.Tensor) -> None:
        """将 CPU int64 [Q] 槽从 PENDING 发布为 READY，零引用槽加入 LRU。"""
        with self._lock:
            for slot in slot_ids.tolist():
                key = self.slot_keys[slot]
                entry = self.entries[key]
                assert not entry[2]
                entry[2] = True
                if not entry[1]:
                    self._lru[key] = None

    def is_ready(self, *, handle: FakeHandle) -> bool:
        """检查未归还的 handle，所有引用槽已发布时返回 True。"""
        with self._lock:
            assert not handle.released
            return all(self.entries[key][2] for key in handle.keys)

    def release(self, *, handle: FakeHandle, discard_pending: bool = False) -> None:
        """归还 handle；discard_pending 为真时回收引用归零的 PENDING。"""
        with self._lock:
            assert not handle.released
            for key in handle.keys:
                entry = self.entries[key]
                entry[1] -= 1
                assert entry[1] >= 0
                if not entry[1]:
                    if entry[2]:
                        self._lru[key] = None
                    elif discard_pending:
                        self._remove(key=key)
            handle.released = True

    def _remove(self, *, key: tuple[int, int]) -> None:
        """在池锁内移除 key 所在的零引用槽，更新空闲和 LRU 索引。"""
        slot, refs, _ = self.entries.pop(key)
        assert refs == 0
        del self.slot_keys[slot]
        self._lru.pop(key, None)

    def stats(self) -> dict[str, int]:
        """无参数，返回引用总数和有引用槽数，用于验证释放。"""
        with self._lock:
            return {
                "referenced": sum(entry[1] for entry in self.entries.values()),
                "referenced_slots": sum(entry[1] > 0 for entry in self.entries.values()),
            }

    def record_submitted(self, *, rows: int) -> None:
        """把 rows 条实际提交的行加入测试计数。"""
        self.submitted += rows

    def record_forward(self, *, hits: int, misses: int, miss_ms: float) -> None:
        """接收组装的命中、缺失和补读时间，测试用池不累计这三个原生字段。"""


class FakeHasher:
    """测试 runtime 持有的两层只读 hash 替身，由后台线程调用。"""

    layer_ids = (1, 14)

    def compute_request_row_ids(self, token_ids: tuple[int, ...], start: int, end: int) -> torch.Tensor:
        """用 token_ids 的 [start,end) 计算连续 CPU int64 [T,2,1] 行号，值在 [0,16)。"""
        rows = torch.tensor(token_ids[start:end], dtype=torch.int64) % 16
        return rows[:, None, None].expand(-1, 2, 1).contiguous()


class FakeReader:
    """测试 runtime 持有的 reader，可用事件暂停一次读取，前后台各用一个实例。"""

    def __init__(self) -> None:
        """无参数；创建读盘记录和测试门闩，默认允许读取直接完成。"""
        self.calls: list[tuple[int, list[int]]] = []
        self.threads: list[int] = []  # 对应每次成功读取的调用线程 ID。
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.proceed.set()
        self.error: BaseException | None = None

    def read_rows_to_slots(
        self, layer_id: int, row_ids: torch.Tensor, slot_ids: torch.Tensor,
        weight_base: torch.Tensor, scale_base: torch.Tensor,
    ) -> None:
        """读取 layer_id 的连续 CPU int64 [Q] 行到指定槽，行号与槽号一一对应、槽号不重复。

        weight_base 和 scale_base 为 CPU uint8 [S,D]／[S,D/32]；返回时已填槽并记录线程。
        proceed 事件允许测试控制完成时间，error 可模拟后台读失败。
        """
        self.entered.set()
        if not self.proceed.wait(timeout=5):
            raise RuntimeError("test reader was not released")
        if self.error is not None:
            raise self.error
        weight_base[slot_ids] = row_ids.to(torch.uint8)[:, None]
        scale_base[slot_ids] = (row_ids + layer_id).to(torch.uint8)[:, None]
        self.calls.append((layer_id, row_ids.tolist()))
        self.threads.append(threading.get_ident())


def _ids(rows: Sequence[int]) -> torch.Tensor:
    """将 rows 转为连续 CPU int64 [Q]；保留顺序和重复。"""
    return torch.tensor(rows, dtype=torch.int64, device="cpu")


def _request(rid: str, tokens: list[int], prefix: int = 0) -> SimpleNamespace:
    """按请求名、完整 token 和前缀长度构造调度用 CPU 请求替身。"""
    return SimpleNamespace(
        rid=rid, origin_input_ids=tokens, output_ids=[], prefix_len=prefix, already_computed=0,
        full_untruncated_fill_ids=tokens,
        finished_reason=None, to_finish=None, multimodal_inputs=None,
    )


@pytest.fixture
def environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[Environment]:
    """创建每个用例独立的控制器和测试 runtime；结束时放行 reader 并停止后台。"""
    config = SimpleNamespace(window_batches=2, chunk_rows=2, log_interval=0)
    monkeypatch.setattr(sched_prefetch, "get_config", lambda: config)
    runtime = SimpleNamespace(
        hasher=FakeHasher(), pool=FakePool(), reader=FakeReader(), prefetch_reader=FakeReader(),
        decode_reader=FakeReader(), tp_size=1, tp_rank=0,
        open_table=lambda layer_id: SimpleNamespace(rows=16), catalog=SimpleNamespace(layer_ids=(1, 14)),
        debug_check_rows=False, dim=32, scale_dim=1,
    )
    runtime.batch_rows = partial(EngramSsdRuntime.batch_rows, runtime)
    runtime.batch_host = partial(EngramSsdRuntime.batch_host, runtime)
    runtime.stage_prefill = partial(EngramSsdRuntime.stage_prefill, runtime)
    scheduler = SimpleNamespace(
        waiting_queue=[], chunked_req=None, chunked_prefill_size=2, max_running_requests=128,
        max_prefill_tokens=128, tp_group=SimpleNamespace(cpu_group=object()),
    )
    manager = sched_prefetch.SchedulerPrefetchController(runtime=runtime, scheduler=scheduler)
    try:
        yield manager, runtime, scheduler, config
    finally:
        runtime.prefetch_reader.proceed.set()
        manager.close()


def _wait_segment(manager: Any, req: Any, index: int = 0, status: str = "ready") -> Any:
    """等待 req 已提交段中指定下标达到目标状态，返回当前请求快照。"""
    state = manager.states[id(req)]
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


def _run_priority(manager: Any, scheduler: Any) -> list[str]:
    """模拟一轮经过 priority Hook 的原生调度，返回原生看到的请求名。"""
    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> list[str]:
        """触发排序后对账，返回当前队列中的请求名。"""
        sched_prefetch.reconcile_active_window(waiting_queue=self.waiting_queue, running_batch=running_batch)
        return [req.rid for req in self.waiting_queue]

    return manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    )


def _staging(rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    """为 CPU reader 替身分配 rows 行带哨兵值的 uint8 暂存表，返回 weight 和 scale。"""
    return torch.full((rows, 32), 255, dtype=torch.uint8), torch.full((rows, 1), 255, dtype=torch.uint8)


def _urgent(
    manager: sched_prefetch.SchedulerPrefetchController, *, rows: dict[int, list[int]],
    priority: sched_prefetch.Priority,
) -> sched_prefetch.UrgentReadRequest:
    """按 rows 生成各层连续目标下标，提交测试用紧急请求。"""
    start, slots = 0, {}
    for layer, values in rows.items():
        end = start + len(values)
        slots[layer] = torch.arange(start, end, dtype=torch.int64)
        start = end
    return manager.request_urgent(
        rows_by_layer={layer: _ids(values) for layer, values in rows.items()},
        slots_by_layer=slots, staging=_staging(start), priority=priority,
    )


def _submit_first(
    manager: sched_prefetch.SchedulerPrefetchController, req: Any,
    priority: sched_prefetch.Priority = (sched_prefetch.NEAR, 2, 0, 0),
) -> sched_prefetch.PrefetchSegment:
    """为读队列单元测试直接提交 req 的剩余区间，返回由 priority 排序的段。"""
    manager.register_request(req=req)
    state = manager.states[id(req)]
    segment = sched_prefetch.PrefetchSegment(start=req.prefix_len, end=len(state.token_ids))
    with manager._condition:
        state.segments.append(segment)
        manager._submit(state=state, segment=segment, priority=priority)
        manager._jobs.sort(key=lambda task: task.priority)
        manager._condition.notify_all()
    return segment


def _seed_ready(pool: FakePool, *, layer_id: int, row: int) -> FakeHandle:
    """在测试池写好 layer_id 的 row，返回持有引用的 READY 句柄。"""
    handle = pool.acquire(layer_id=layer_id, row_ids=_ids([row]))
    pool.weight[handle.slot_ids] = row
    pool.scale[handle.slot_ids] = row + layer_id
    pool.mark_ready(slot_ids=handle.slot_ids)
    return handle


def test_read_queue_checks_priority_after_each_chunk(environment: Environment) -> None:
    """完成标准 1：同一后台线程读有界块，块之间按键四项排序，层可独立插队。"""
    manager, runtime, _, config = environment
    reader = runtime.prefetch_reader
    reader.proceed.clear()
    normal = _request("normal", [1, 2, 3, 4, 5])
    _submit_first(manager=manager, req=normal)
    assert reader.entered.wait(timeout=5)
    near = _request("near", [12])
    _submit_first(manager=manager, req=near, priority=(sched_prefetch.NEAR, 1, 0, 0))
    urgent = sched_prefetch.URGENT
    jobs = [
        _urgent(manager, rows={1: [10]}, priority=(urgent, 2, 0, 0)),
        _urgent(manager, rows={14: [11]}, priority=(urgent, 1, 1, 0)),
        _urgent(manager, rows={1: [9]}, priority=(urgent, 1, 0, 2)),
        _urgent(manager, rows={14: [13], 1: [8]}, priority=(urgent, 1, 0, 1)),
    ]
    reader.proceed.set()
    assert all(manager.wait_urgent(request=job, timeout=5) for job in jobs)
    _wait_segment(manager=manager, req=normal)
    _wait_segment(manager=manager, req=near)
    assert reader.calls == [
        (1, [1, 2]), (1, [8]), (1, [9]), (14, [11]), (14, [13]), (1, [10]),
        (1, [12]), (14, [12]), (1, [3, 4]), (1, [5]), (14, [1, 2]), (14, [3, 4]), (14, [5]),
    ]
    assert all(len(rows) <= config.chunk_rows for _, rows in reader.calls)
    assert set(reader.threads) == {manager._thread.ident}


def test_window_cancels_moved_segments_and_reuses_ready_cache(environment: Environment) -> None:
    """完成标准 1：排序移出窗口后归还 READY 引用；重新进入时直接复用缓存。"""
    manager, runtime, scheduler, config = environment
    config.window_batches = 1
    first = _request("first", [1, 2])
    second = _request("second", [3, 4])
    scheduler.waiting_queue = [first, second]
    for req in scheduler.waiting_queue:
        manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    state = _wait_segment(manager=manager, req=first)
    old = state.segments[0]
    assert not manager.states[id(second)].segments
    scheduler.waiting_queue.reverse()
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=second)
    assert not state.segments and old.cancelled
    assert runtime.pool.entries[(1, 1)][1:] == [0, True]
    assert runtime.pool.entries[(14, 2)][1:] == [0, True]
    scheduler.waiting_queue.reverse()
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=first)
    assert runtime.prefetch_reader.calls == [(1, [1, 2]), (14, [1, 2]), (1, [3, 4]), (14, [3, 4])]


def test_window_places_continuation_first_and_respects_request_slots(environment: Environment) -> None:
    """完成标准 2：已有续段优先；新截断区间在下一批最前；每批请求数按运行上限估算，不扣除运行中请求。"""
    manager, _, scheduler, _ = environment
    manager.chunked_tokens = manager.prefill_tokens = 5
    chunked = scheduler.chunked_req = _request("chunked", [0, 1, 2, 3], prefix=2)
    first = _request("first", [4, 5, 6, 7])
    second = _request("second", [8, 9, 10, 11])
    waiting = [first, second]
    expected = [(chunked, 2, 4, 1), (first, 0, 3, 1), (first, 3, 4, 2), (second, 0, 4, 2)]
    assert manager._plan_window(waiting=waiting, chunked=chunked, slots=2, batches=2) == expected
    assert manager._plan_window(waiting=waiting, chunked=chunked, slots=1, batches=2) == [
        (chunked, 2, 4, 1), (first, 0, 4, 2),
    ]
    # 运行批满时仍按上限估算：请求常成批结束，下一批可能一次放满。
    scheduler.max_running_requests = 3
    running = SimpleNamespace(reqs=[object(), object(), object()])
    slots = manager._request_slots(scheduler=scheduler, running_batch=running)
    assert slots == 3
    assert manager._plan_window(waiting=waiting, chunked=chunked, slots=slots, batches=2) == expected


def test_window_caps_chunk_with_max_prefill_tokens(environment: Environment) -> None:
    """完成标准 3：8 token chunk 受 5 token 输入预算限制，截断余量排在下一批首位。"""
    _, runtime, scheduler, _ = environment
    scheduler.chunked_prefill_size, scheduler.max_prefill_tokens = 8, 5
    manager = sched_prefetch.SchedulerPrefetchController(runtime=runtime, scheduler=scheduler)
    try:
        first = _request("first", [1, 2, 3])
        truncated = _request("truncated", [4, 5, 6, 7, 8])
        last = _request("last", [9, 10])
        assert manager._plan_window(waiting=[first, truncated, last], chunked=None, slots=8, batches=2) == [
            (first, 0, 3, 1), (truncated, 0, 2, 1), (truncated, 2, 5, 2), (last, 0, 2, 2),
        ]
    finally:
        manager.close()


@pytest.mark.parametrize(
    ("lengths", "continuation", "slots"),
    [
        pytest.param([3, 7, 2], None, 8, id="shared-chunk"),
        pytest.param([2, 4], (7, 3), 8, id="continuation-first"),
        pytest.param([2, 2, 2], None, 2, id="request-limit"),
    ],
)
def test_first_window_batch_matches_native_adder(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
    lengths: list[int], continuation: tuple[int, int] | None, slots: int,
) -> None:
    """完成标准 3：按 lengths、续段（总长、前缀）和 slots 用真实 PrefillAdder 对照批 1。"""
    policy = pytest.importorskip("sglang.srt.managers.schedule_policy")
    from sglang.srt.mem_cache.prefill_budget import PrefillBudget
    from sglang.srt.sampling.sampling_params import SamplingParams

    manager, _, _, _ = environment
    manager.chunked_tokens = manager.prefill_tokens = 8
    # 用真实的 token 预算，替身只提供充足的 KV 容量、空 cache 与前缀锁接口。
    allocator = SimpleNamespace(page_size=1, available_size=lambda: 1024, request_slot_reserve=lambda **_: 0)
    allocator.create_prefill_budget = partial(PrefillBudget, allocator)
    cache = SimpleNamespace(
        supports_mamba=lambda: False, evictable_size=lambda: 0, lock=lambda _: None, unlock=lambda _: None,
        finish_storage_prefetch_admission=lambda _handle, **_: None,
    )
    exec_config = SimpleNamespace(features=SimpleNamespace(enable_encoder_swa_bounded_replay=False))
    monkeypatch.setattr(policy, "get_exec", lambda: exec_config)
    adder = policy.PrefillAdder(
        page_size=1, tree_cache=cache, token_to_kv_pool_allocator=allocator, running_batch=None,
        new_token_ratio=1.0, rem_input_tokens=8, rem_chunk_tokens=8, prefill_max_requests=slots,
    )

    def make_req(rid: str, length: int, prefix: int = 0) -> Any:
        """按请求标识、输入长度和已计算前缀构造原生 CPU Req，返回已填好输入序列的请求。"""
        req = policy.Req(
            rid=rid, origin_input_text="", origin_input_ids=array("q", range(length)),
            sampling_params=SamplingParams(max_new_tokens=1),
        )
        req.refresh_fill_ids()
        req.prefix_len = prefix
        return req

    waiting = [make_req(rid=str(i), length=length) for i, length in enumerate(lengths)]
    chunked = make_req("chunked", *continuation) if continuation is not None else None
    window = manager._plan_window(waiting=waiting, chunked=chunked, slots=slots, batches=2)
    # 沿用原生先加入续段、再按队列接纳请求的顺序，区间从实际接纳结果读取。
    remaining = adder.add_chunked_req(req=chunked) if chunked is not None else None
    for req in waiting:
        result = adder.add_one_req(req=req, has_chunked_req=remaining is not None, truncation_align_size=None)
        if result is not policy.AddReqResult.CONTINUE:
            break
    assert [(req, start, end) for req, start, end, batch in window if batch == 1] == [
        (req, req.prefix_len, req.extend_end) for req in adder.can_run_list
    ]


def test_window_without_chunking_uses_max_prefill_tokens(environment: Environment) -> None:
    """完成标准 2：无分块时按 max_prefill_tokens 分批，超长首请求完整独占一批。"""
    manager, runtime, scheduler, _ = environment
    scheduler.chunked_prefill_size, scheduler.max_prefill_tokens = None, 5
    unchunked = sched_prefetch.SchedulerPrefetchController(runtime=runtime, scheduler=scheduler)
    try:
        first = _request("first", [1, 2, 3])
        second = _request("second", [1, 2, 3, 4])
        third = _request("third", [1, 2])
        long = _request("long", list(range(8)), prefix=1)
        assert unchunked.prefill_tokens == 5
        assert unchunked._plan_window(waiting=[first, second, third, long], chunked=None, slots=8, batches=4) == [
            (first, 0, 3, 1), (second, 0, 4, 2), (third, 0, 2, 3), (long, 1, 8, 4),
        ]
    finally:
        unchunked.close()


def test_window_reads_all_first_layers_of_each_batch_first(environment: Environment) -> None:
    """完成标准 3：窗口按预测批次、层、批内请求的顺序读取。"""
    manager, runtime, scheduler, _ = environment
    manager.chunked_tokens = manager.prefill_tokens = 4
    waiting = [_request(str(i), [i, i + 1]) for i in range(0, 10, 2)]
    scheduler.waiting_queue = waiting
    for req in waiting:
        manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=waiting[3])
    assert runtime.prefetch_reader.calls == [
        (1, [0, 1]), (1, [2, 3]), (14, [0, 1]), (14, [2, 3]),
        (1, [4, 5]), (1, [6, 7]), (14, [4, 5]), (14, [6, 7]),
    ]
    assert not manager.states[id(waiting[4])].segments


def test_forming_round_skips_current_batch_and_keeps_its_segments(environment: Environment) -> None:
    """calc_priority 入口跳过正在组的批：不为它新建段，已有同区间段保留；预取其后的 WINDOW_BATCHES 批。"""
    manager, _, scheduler, config = environment
    config.window_batches = 1
    manager.chunked_tokens = manager.prefill_tokens = 2
    first, second, third = (_request(name, [i, i + 1]) for i, name in ((0, "first"), (2, "second"), (4, "third")))
    scheduler.waiting_queue = [first, second, third]
    for req in scheduler.waiting_queue:
        manager.register_request(req=req)
    # 首次就是组批轮：first 是正在组的批，没有旧段，不新建；只预取 second。
    manager.reconcile(scheduler=scheduler, forming=True)
    _wait_segment(manager=manager, req=second)
    assert not manager.states[id(first)].segments
    assert not manager.states[id(third)].segments
    # decode 轮：second 成为下一个 prefill 批，沿用已有段。
    old = manager.states[id(second)].segments[0]
    scheduler.waiting_queue = [second, third]
    manager.reconcile(scheduler=scheduler)
    assert manager.states[id(second)].segments == [old] and not old.cancelled
    # 下一个组批轮：second 正在组，保留已有段；新预取 third。
    manager.reconcile(scheduler=scheduler, forming=True)
    _wait_segment(manager=manager, req=third)
    assert manager.states[id(second)].segments == [old] and not old.cancelled


def test_urgent_writes_caller_slots_without_reserving_pool(environment: Environment) -> None:
    """完成标准 1：READY 复制、PENDING 和新行读入分散的调用方下标，池占用保持原值。"""
    manager, runtime, _, _ = environment
    pool = runtime.pool = FakePool(capacity=2)
    ready = _seed_ready(pool, layer_id=1, row=9)
    pool.release(handle=ready)
    pending = pool.acquire(layer_id=1, row_ids=_ids([3]))
    weight, scale = _staging(6)
    reader = runtime.prefetch_reader
    reader.proceed.clear()
    try:
        request = manager.request_urgent(
            rows_by_layer={1: _ids([9, 3, 8]), 14: _ids([7])},
            slots_by_layer={1: _ids([4, 0, 2]), 14: _ids([3])}, staging=(weight, scale),
            priority=(sched_prefetch.URGENT, 0, 0, 0),
        )
        assert reader.entered.wait(timeout=5)
        assert not manager.wait_urgent(request=request, timeout=0)
        assert torch.equal(weight, torch.full_like(weight, 255))
        assert pool.stats()["referenced"] == 2
        reader.proceed.set()
        assert manager.wait_urgent(request=request, timeout=5)
        assert reader.calls == [(1, [3, 8]), (14, [7])]
        assert torch.equal(weight[:, 0], torch.tensor([3, 255, 8, 7, 9, 255], dtype=torch.uint8))
        assert torch.equal(scale[:, 0], torch.tensor([4, 255, 9, 21, 10, 255], dtype=torch.uint8))
        assert set(pool.entries) == {(1, 9), (1, 3)}
        assert not pool.is_ready(handle=pending)
        assert pool.stats()["referenced"] == 1
    finally:
        pool.release(handle=pending, discard_pending=True)


def test_prefill_uses_native_batch_order_and_cancels_overlaps(
    environment: Environment, caplog: pytest.LogCaptureFixture,
) -> None:
    """验证同批有重叠段和精确匹配段时，按实际顺序组装并统计去重后的紧急缺行。"""
    manager, runtime, _, config = environment
    config.chunk_rows, config.log_interval = 1, 1
    caplog.set_level(logging.INFO, logger=sched_prefetch.__name__)
    runtime.tp_size, runtime.tp_rank = 2, 1
    runtime.debug_check_rows = True
    first = _request("first", [12, 10, 8, 7], prefix=1)
    second = _request("second", [5, 10, 1])
    cached = _seed_ready(runtime.pool, layer_id=14, row=8)
    runtime.pool.release(handle=cached)
    reader = runtime.prefetch_reader
    reader.proceed.clear()
    active = _submit_first(manager=manager, req=first)
    assert reader.entered.wait(timeout=5)
    queued = _submit_first(manager=manager, req=second)
    queued.row_ids = runtime.hasher.compute_request_row_ids(token_ids=second.origin_input_ids, start=0, end=3)
    compute = runtime.hasher.compute_request_row_ids
    hashes = []

    def counted(*, token_ids: Sequence[int], start: int, end: int) -> torch.Tensor:
        """记录实际发起时重算的区间，转交 CPU 哈希替身。"""
        hashes.append((start, end))
        return compute(token_ids=token_ids, start=start, end=end)

    runtime.hasher.compute_request_row_ids = counted
    batch = SimpleNamespace(
        reqs=[first, second], prefix_lens=[1, 0], extend_lens=[2, 3], extend_num_tokens=5,
        forward_mode=SimpleNamespace(is_decode=lambda: False, is_extend=lambda: True,
                                     is_mixed=lambda: False, is_split_prefill=lambda: False),
    )

    def reserve(*, batch: Any, queries: int, rows: int, decode: bool) -> Batch:
        """以 CPU 字节区模拟本批的 pinned / HBM 双缓冲，保留实际的批次视图布局。"""
        assert not decode and queries == 10 and rows == 4
        host = torch.zeros(queries * 4 + rows * 33, dtype=torch.uint8)
        runtime.batch = Batch((host, host.clone(), None, None, 0), queries, rows, None, None, None, decode)
        return runtime.batch

    layers = []

    def copied(*, batch: Batch, layer: int, slots: torch.Tensor) -> None:
        """捕获模拟的单层 H2D，验证调度锁已经释放、各层按顺序提交。"""
        assert not manager._condition._is_owned()
        layers.append(layer)
        batch.storage[1].copy_(batch.storage[0])

    runtime.reserve_batch, runtime.copy_layer = reserve, copied
    manager.start_batch(batch=batch, iteration=1)
    assert active.cancelled and queued.cancelled
    assert hashes == [(1, 3)]
    assert runtime.batch.urgent == {1: 2, 14: 1}
    assert not manager.wait_urgent(request=runtime.batch.request, timeout=0)
    assert not reader.calls
    reader.proceed.set()
    assert manager.wait_urgent(request=runtime.batch.request, timeout=5)
    assert layers == [1, 14] and runtime.batch.request.done == {1, 14}
    mapping, weight, scale = runtime.batch_host(batch=runtime.batch)
    assert torch.equal(mapping, torch.tensor([1, 0, -1, 1, -1, 3, 2, -1, 3, -1], dtype=torch.int32))
    assert torch.equal(weight[:, 0], torch.tensor([8, 10, 8, 10], dtype=torch.uint8))
    assert torch.equal(scale[:, 0], torch.tensor([9, 11, 22, 24], dtype=torch.uint8))
    assert torch.equal(runtime.batch.expected[:, 0, 0], _ids([10, 8, 5, 10, 1]))
    assert torch.equal(*runtime.batch.storage[:2])
    assert reader.calls == [(1, [10]), (1, [8]), (1, [10]), (14, [10])]
    assert runtime.pool.stats()["referenced"] == 0
    manager._log_stats(phase="prefill")
    message = caplog.records[-1].getMessage()
    stats = json.loads(message.removeprefix("Engram SSD sched prefetch stats: "))
    assert stats["window_hit_rate"] == 0.25
    assert stats["urgent_rows"] == 3  # 同一层内的重复 owner 行只补读一次。
    assert (stats["predicted_spans"], stats["matched_spans"]) == (2, 1)
    assert stats["prefill_waits"] == 2 and stats["prefill_wait_ms"] >= 0
    assert stats["read_rows"] == 4 and stats["read_ms"] >= 0  # 一个已发出的普通块和三个紧急块。


@pytest.mark.parametrize(
    ("failed_row", "background_rows", "fallback_rows"),
    [(6, [5], [6, 8, 9]), (9, [5, 6, 8], [9])],
)
def test_urgent_wait_resumes_only_unready_rows_after_background_error(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
    failed_row: int, background_rows: list[int], fallback_rows: list[int],
) -> None:
    """完成标准 6：按 failed_row 模拟失败，核对前后台各自读行及统计、释放结果。"""
    manager, runtime, _, config = environment
    config.chunk_rows, config.log_interval = 1, 1
    pool = runtime.pool = FakePool(capacity=3)
    cached = _seed_ready(pool, layer_id=1, row=7)
    pool.release(handle=cached)
    read = runtime.prefetch_reader.read_rows_to_slots
    error = OSError("synthetic background block failure")

    def fail_row(
        *, layer_id: int, row_ids: torch.Tensor, slot_ids: torch.Tensor,
        weight_base: torch.Tensor, scale_base: torch.Tensor,
    ) -> None:
        """用原 reader 替身写 CPU int64 [1] 行到目标表，遇到 failed_row 时抛出错误。"""
        if row_ids.tolist() == [failed_row]:
            raise error
        read(layer_id=layer_id, row_ids=row_ids, slot_ids=slot_ids, weight_base=weight_base, scale_base=scale_base)

    monkeypatch.setattr(runtime.prefetch_reader, "read_rows_to_slots", fail_row)
    request = _urgent(manager, rows={1: [7, 5, 6, 8, 9]}, priority=(sched_prefetch.URGENT, 0, 0, 0))
    assert manager.wait_urgent(request=request, timeout=5)
    assert manager._failure is error
    assert runtime.prefetch_reader.calls == [(1, [row]) for row in background_rows]
    assert runtime.reader.calls == [(1, [row]) for row in fallback_rows]
    assert runtime.reader.threads == [threading.get_ident()] * len(fallback_rows)
    assert manager._read_rows == len(background_rows)
    assert manager._prefill_waits == 1
    weight, scale = request.staging
    assert torch.equal(weight[:, 0], torch.tensor([7, 5, 6, 8, 9], dtype=torch.uint8))
    assert torch.equal(scale[:, 0], torch.tensor([8, 6, 7, 9, 10], dtype=torch.uint8))
    assert pool.stats()["referenced"] == 0


def test_decode_stats_measure_hash_ready_to_publish(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """用 CPU 暂存区和可控时钟核对 decode 缺行和放行等待的统计边界；decode reader 不计入后台读取。"""
    manager, runtime, _, config = environment
    config.log_interval = 1
    handle = _seed_ready(runtime.pool, layer_id=1, row=2)
    runtime.pool.release(handle=handle)
    ids = _ids([2, 5, 2, 5, 2, 7]).view(3, 2, 1)
    host = torch.empty(6 * 4 + 3 * 33, dtype=torch.uint8)
    batch = Batch((host, host, None, None, 0), 6, 3, None, None, None, True)
    batch.hash_ids = ids
    now = 0.0
    read = runtime.decode_reader.read_rows_to_slots

    def synchronize() -> None:
        """模拟无参数的 hash 事件等待，将时钟推进 1 秒。"""
        nonlocal now
        now += 1

    def read_block(**kwargs: Any) -> None:
        """将读块参数交给 CPU reader 写行，并模拟 125 毫秒读块耗时。"""
        nonlocal now
        read(**kwargs)
        now += 0.125

    def publish(*, batch: Batch) -> None:
        """检查 batch 的 CPU 暂存行均已就绪，并模拟放行标志的发布时间。"""
        nonlocal now
        _, weight, _ = runtime.batch_host(batch=batch)
        assert weight[:, 0].tolist() == [2, 5, 7]
        now += 0.25

    batch.hash_done = SimpleNamespace(synchronize=synchronize)
    runtime.publish_decode = publish
    monkeypatch.setattr(runtime.decode_reader, "read_rows_to_slots", read_block)
    monkeypatch.setattr(sched_prefetch, "time", SimpleNamespace(perf_counter=lambda: now))
    manager.read_decode(batch=batch)
    assert (manager._urgent_rows, manager._read_rows) == (2, 0)
    assert manager._read_ms == 0
    assert (manager._decode_waits, manager._decode_wait_ms) == (1, 375)


def test_decode_worker_reports_terminal_read_failure(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """完成标准 6：已交接的 decode 最终失败时，后台报告原异常并结束本 rank。"""
    manager, _, _, _ = environment
    error = OSError("decode synchronous read failed")

    def fail(*, batch: Batch, retry: bool) -> None:
        """模拟当前线程的同步补读失败。"""
        assert retry
        raise error

    def exit_rank(status: int) -> None:
        """拦截进程退出，允许测试核对状态码。"""
        raise SystemExit(status)

    monkeypatch.setattr(manager, "read_decode", fail)
    monkeypatch.setattr(sched_prefetch, "os", SimpleNamespace(_exit=exit_rank))
    with pytest.raises(SystemExit) as info:
        manager._worker_decode(batch=SimpleNamespace(), retry=True)
    assert info.value.code == 1
    assert "decode synchronous read failed" in caplog.text


def test_registration_waits_for_priority_prefix_snapshot(environment: Environment) -> None:
    """A7/P3：初次入队不读 SSD，priority 更新 prefix 后才切分未命中的后缀；正在组的 (2,4) 留给紧急读。"""
    manager, runtime, scheduler, _ = environment
    req = _request("prefix-hit", [1, 2, 3, 4, 5, 6])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    state = manager.states[id(req)]
    assert not state.segments
    assert not runtime.prefetch_reader.calls
    req.prefix_len = 2  # 模拟支持 prefix 匹配的原生 priority 完成缓存查询。
    runtime.prefetch_reader.proceed.clear()
    try:
        assert _run_priority(manager=manager, scheduler=scheduler) == [req.rid]
        assert not runtime.prefetch_reader.calls
    finally:
        runtime.prefetch_reader.proceed.set()
    _wait_segment(manager=manager, req=req)
    # 本轮正在组 (2,4)，只预取其后的 (4,6)。
    assert [(segment.start, segment.end) for segment in state.segments] == [(4, 6)]
    assert all(row not in (1, 2) for _, rows in runtime.prefetch_reader.calls for row in rows)
    assert not runtime.reader.calls  # 后台独立 reader，不占用 forward 的 io_uring 实例。


def test_priority_without_prefix_refresh_uses_existing_snapshot(environment: Environment) -> None:
    """A7/P3：原生策略未刷新 prefix 时不自行改 KV 状态，使用当前快照预取。"""
    manager, _, scheduler, _ = environment
    current = _request("forming", [5, 6])  # 本轮正在组的批，不新建预取段
    req = _request("no-fast-prefix", [1, 2], prefix=0)
    scheduler.waiting_queue = [current, req]
    manager.register_request(req=current)
    manager.register_request(req=req)
    assert _run_priority(manager=manager, scheduler=scheduler) == [current.rid, req.rid]
    state = _wait_segment(manager=manager, req=req)
    assert state.segments[0].start == 0
    assert not manager.states[id(current)].segments


def test_pool_full_retries_only_after_release_and_reuses_hash(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A7/P5：实际可用槽不足时不重试；空闲加 LRU 足够后复用 CPU 行号。"""
    manager, runtime, scheduler, _ = environment
    calls, attempts = [], []
    compute = runtime.hasher.compute_request_row_ids
    pool = runtime.pool = FakePool(capacity=4)
    first = pool.acquire(layer_id=1, row_ids=_ids([11]))
    second = pool.acquire(layer_id=1, row_ids=_ids([12]))
    pool.mark_ready(slot_ids=first.slot_ids)
    pool.mark_ready(slot_ids=second.slot_ids)
    acquire = pool.acquire

    def counted(*, token_ids: tuple[int, ...], start: int, end: int) -> torch.Tensor:
        """用原 hash 替身计算连续 CPU int64 [T,2,1]，记录输入 token_ids 的所查区间。"""
        calls.append((start, end))
        return compute(token_ids=token_ids, start=start, end=end)

    def track_acquire(*, layer_id: int, row_ids: torch.Tensor) -> FakeHandle | None:
        """记录尝试为 layer_id 的 CPU int64 [Q] 行占槽，返回真实替身结果。"""
        attempts.append(layer_id)
        return acquire(layer_id=layer_id, row_ids=row_ids)

    monkeypatch.setattr(runtime.hasher, "compute_request_row_ids", counted)
    monkeypatch.setattr(pool, "acquire", track_acquire)
    req = _request("retry", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    state = _wait_segment(manager=manager, req=req, status="blocked")
    segment = state.segments[0]
    cached_ids = segment.row_ids
    assert manager._slots_needed(segment=segment) == 4
    assert attempts == [1, 14]
    pool.release(handle=first)
    assert pool.available_slots() == 3
    for _ in range(4):
        manager.reconcile(scheduler=scheduler)
    assert segment.status == "blocked"
    assert attempts == [1, 14]
    assert calls == [(0, 2)]
    assert segment.row_ids is cached_ids
    pool.release(handle=second)
    assert pool.available_slots() == 4
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=req)
    assert calls == [(0, 2)]
    assert attempts == [1, 14, 1, 14]
    assert segment.row_ids is cached_ids
    assert segment.owner_rows is None


def test_blocked_segment_retries_when_another_segment_pins_shared_rows(environment: Environment) -> None:
    """完成标准 7：其它普通段固定共享行后，按当前实际需求重试，不必等共享引用归零。"""
    manager, runtime, scheduler, _ = environment
    pool = runtime.pool = FakePool(capacity=4)
    holders = [_seed_ready(pool, layer_id=1, row=row) for row in (9, 10)]
    waiting = _request("waiting-for-room", [1, 2])
    shared = _request("shared-row", [1])
    scheduler.waiting_queue = [waiting, shared]
    segment = _submit_first(manager=manager, req=waiting)
    _wait_segment(manager=manager, req=waiting, status="blocked")
    assert manager._slots_needed(segment=segment) == 4
    _submit_first(manager=manager, req=shared)
    _wait_segment(manager=manager, req=shared)
    for handle in holders:
        pool.release(handle=handle)
    assert pool.available_slots() == 2
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=waiting)
    assert runtime.prefetch_reader.calls == [(1, [1]), (14, [1]), (1, [2]), (14, [2])]


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
    manager.reconcile(scheduler=scheduler)
    segment = state.segments[0]
    try:
        assert entered.wait(timeout=5)
        scheduler.waiting_queue = []
        manager.reconcile(scheduler=scheduler)
        assert segment.cancelled
    finally:
        proceed.set()
    manager.close()
    assert segment.row_ids is None
    assert not runtime.prefetch_reader.calls


@pytest.mark.parametrize("local_failure", [False, True])
def test_priority_syncs_one_background_failure_flag(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, local_failure: bool,
) -> None:
    """完成标准 3：有 waiting 且进入 priority 时，任一 rank 失败经一次单标志 MIN 传播。"""
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

    def reduce(flag: torch.Tensor, op: Any, group: Any) -> None:
        """检查本 rank 的单个失败标志，模拟归约发现某 rank 失败。"""
        calls.append(flag.tolist())
        assert flag.tolist() == ([-1] if local_failure else [1])
        flag[0] = -1

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        _run_priority(manager=manager, scheduler=scheduler)
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
    """完成标准 1/2：原生运行槽已满时早退，仍按当前 prefix 为将来的两个批次预取。"""
    manager, runtime, scheduler, _ = environment
    req = _request("waiting-behind-full-batch", [1, 2, 3, 4, 5, 6], prefix=2)
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    runtime.prefetch_reader.proceed.clear()
    state = manager.states[id(req)]
    assert not state.segments

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> str:
        """模拟原生早退，不调用 priority Hook。"""
        return "batch-full"

    try:
        result = manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None,
            running_batch=SimpleNamespace(reqs=[object()] * scheduler.max_running_requests),
        )
        assert result == "batch-full"
        # run_prefill 已返回，后台 reader 才被允许完成；调度线程没有等待 SSD。
        assert runtime.prefetch_reader.entered.wait(timeout=5)
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

    def counted_reconcile(*, scheduler: Any, running_batch: Any, forming: bool = False) -> None:
        """记录原生调度状态的对账次数并调用原控制器。"""
        reconciled.append(scheduler)
        reconcile(scheduler=scheduler, running_batch=running_batch, forming=forming)

    def reject_collective(mask: torch.Tensor, op: Any, group: Any) -> None:
        raise AssertionError("Pure decode must not issue a CPU failure collective")

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> None:
        if use_priority:
            sched_prefetch.reconcile_active_window(waiting_queue=self.waiting_queue)
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
def test_new_waiting_request_resumes_failure_min(
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
        assert manager.states[id(req)].segments
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
            sched_prefetch.reconcile_active_window(waiting_queue=self.waiting_queue)
        return None

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
        manager.run_prefill(
            original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
        )
    assert len(calls) == 1
    assert error.value.__cause__ is (cause if local_failure else None)
    assert sched_prefetch._active_schedule.get() is None


def test_background_owner_reads_and_window_can_hold_three_segments(environment: Environment) -> None:
    """完成标准 1/7：只读本 rank 行，同一层重复行只读一次，单请求覆盖三批。"""
    manager, runtime, scheduler, config = environment
    config.window_batches = 3
    runtime.tp_rank, runtime.tp_size = 1, 2
    req = _request("owner", [1, 9, 9, 10, 11, 12, 13, 14])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    state = _wait_segment(manager=manager, req=req, index=2)
    assert [(s.start, s.end) for s in state.segments] == [(0, 2), (2, 4), (4, 6)]
    assert all(8 <= row < 13 for _, rows in runtime.prefetch_reader.calls for row in rows)
    assert sum(rows.count(9) for _, rows in runtime.prefetch_reader.calls) == 2
    assert manager._thread.is_alive()


def test_best_effort_preserves_native_queue_and_uses_priority_order(
    environment: Environment, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完成标准 2/3/5：按排序结果取窗口，正在组的批不预取，不等待缺行，不接管原生队列或重复归约。"""
    manager, runtime, scheduler, config = environment
    config.window_batches = 1
    runtime.tp_size = 8
    outside, next_req = _request("outside", [2, 3]), _request("next", [0, 1])
    first_queue = scheduler.waiting_queue = [outside, next_req]
    for req in first_queue:
        manager.register_request(req=req)
    remaining = [outside]
    calls = []
    runtime.prefetch_reader.proceed.clear()

    def reduce(flag: torch.Tensor, op: Any, group: Any) -> None:
        """验证归约只传一个 CPU 失败标志，健康 rank 继续组批。"""
        calls.append(flag.tolist())
        assert flag.device.type == "cpu" and flag.dtype == torch.int32
        assert op == torch.distributed.ReduceOp.MIN and group is scheduler.tp_group.cpu_group

    def original(self: Any, prefill_delayer_single_pass: Any, running_batch: Any) -> str:
        """模拟原生排序、重复 priority 回调及取走已组批请求后替换列表。"""
        assert not manager.states[id(outside)].segments
        assert not manager.states[id(next_req)].segments
        self.waiting_queue.reverse()
        for _ in range(2):
            sched_prefetch.reconcile_active_window(waiting_queue=self.waiting_queue, running_batch=running_batch)
            assert self.waiting_queue is first_queue and self.waiting_queue == [next_req, outside]
        # 排序后 next 是本轮正在组的批，不新建段；窗口 1 批预取其后的 outside。
        assert not manager.states[id(next_req)].segments
        assert [(seg.start, seg.end) for seg in manager.states[id(outside)].segments] == [(0, 2)]
        assert not runtime.prefetch_reader.calls
        self.waiting_queue = remaining
        return "original-result"

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    assert manager.run_prefill(
        original_fn=original, scheduler=scheduler, prefill_delayer_single_pass=None, running_batch=None,
    ) == "original-result"
    assert calls == [[1]]
    assert scheduler.waiting_queue is remaining and remaining == [outside]
    assert first_queue == [next_req, outside]


@pytest.mark.parametrize("use_priority", [False, True])
def test_empty_queue_skips_failure_sync_and_restores_context(
    environment: Environment, monkeypatch: pytest.MonkeyPatch, use_priority: bool,
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
        """模拟原生在 priority 前或之后报错。"""
        if use_priority:
            sched_prefetch.reconcile_active_window(waiting_queue=self.waiting_queue)
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
    manager.reconcile(scheduler=scheduler)
    state = _wait_segment(manager=manager, req=req)
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


def test_retraction_removes_only_its_old_batch_span(environment: Environment) -> None:
    """A7：同一 Req 撤回重登时删除它的旧区间；旧 batch 结果仍归还同批其它请求。"""
    manager, runtime, scheduler, _ = environment
    req = _request("retract", [1, 2])
    peer = _request("peer", [5, 6])
    scheduler.waiting_queue = [req, peer]
    manager.register_request(req=req)
    manager.register_request(req=peer)
    manager.reconcile(scheduler=scheduler)
    old = _wait_segment(manager=manager, req=req)
    _wait_segment(manager=manager, req=peer)
    batch = _batch(req=req, start=0, length=2, iteration=1)
    batch.reqs.append(peer)
    batch.prefix_lens.append(0)
    batch.extend_lens.append(2)
    manager.record_batch(batch=batch)
    req.output_ids = [3]
    manager.register_request(req=req, is_retracted=True)
    scheduler.waiting_queue = [peer, req]
    manager.reconcile(scheduler=scheduler)
    new = _wait_segment(manager=manager, req=req)
    assert old.segments[0].cancelled
    assert new.token_ids == (1, 2, 3)
    assert manager._batches[1] == ((peer, 0, 2),)
    assert runtime.pool.stats()["referenced"] == 8
    manager.release_batch(batch=batch)
    assert id(peer) not in manager.states
    assert runtime.pool.stats()["referenced"] == 4
    assert not new.segments[0].cancelled


def test_partial_segment_reservation_rolls_back_pending(environment: Environment) -> None:
    """A7：跨层占槽中途池满时回收未读PENDING，避免无法完成的段永久占池。"""
    manager, runtime, scheduler, _ = environment
    runtime.pool.capacity = 2
    req = _request("full", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=req, status="blocked")
    assert runtime.pool.stats()["referenced"] == 0
    assert not runtime.pool.entries
    assert not runtime.prefetch_reader.calls


def test_prefix_shrink_replaces_overlapping_window_span(environment: Environment) -> None:
    """完成标准 1/5：prefix 缩小时重建相交的窗口段，旧 READY 只命中重叠行。"""
    manager, runtime, scheduler, config = environment
    config.window_batches = 1
    req = _request("shrinking-prefix", [1, 2, 3, 4], prefix=1)
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    state = _wait_segment(manager=manager, req=req)
    old = state.segments[0]
    assert (old.start, old.end) == (1, 3)
    runtime.prefetch_reader.proceed.clear()
    req.prefix_len = 0
    # 本用例验证 prefix 变化后的重建，用 decode 轮入口（无正在组的批），窗口第 1 批就是该请求。
    manager.reconcile(scheduler=scheduler)
    new = state.segments[0]
    assert old.cancelled and (new.start, new.end) == (0, 2)
    runtime.prefetch_reader.proceed.set()
    _wait_segment(manager=manager, req=req)
    manager.reconcile(scheduler=scheduler)
    assert runtime.prefetch_reader.calls == [(1, [2, 3]), (14, [2, 3]), (1, [1]), (14, [1])]


def test_abort_inflight_defers_release_until_read_finishes(environment: Environment) -> None:
    """A7：中止请求后只收齐当前块，后台回收它的引用并跳过余下的块。"""
    manager, runtime, scheduler, config = environment
    config.chunk_rows = 1
    runtime.prefetch_reader.proceed.clear()
    req = _request("abort", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    assert runtime.prefetch_reader.entered.wait(timeout=5)
    scheduler.waiting_queue = []
    manager.reconcile(scheduler=scheduler)
    assert id(req) not in manager.states
    assert runtime.pool.stats()["referenced"] > 0
    runtime.prefetch_reader.proceed.set()
    manager.close()
    assert runtime.prefetch_reader.calls == [(1, [1])]
    assert runtime.pool.stats()["referenced"] == 0


def test_background_error_is_rethrown(environment: Environment) -> None:
    """A7：后台SSD失败在下一次主路径传播，且不遗留PENDING或引用。"""
    manager, runtime, scheduler, _ = environment
    runtime.prefetch_reader.error = OSError("synthetic I/O error")
    req = _request("failure", [1, 2])
    scheduler.waiting_queue = [req]
    manager.register_request(req=req)
    manager.reconcile(scheduler=scheduler)
    with manager._condition:
        assert manager._condition.wait_for(lambda: manager._failure is not None, timeout=5)
    with pytest.raises(RuntimeError, match="failed on a TP rank"):
        _run_priority(manager=manager, scheduler=scheduler)
    assert runtime.pool.stats()["referenced"] == 0
    assert not runtime.pool.entries


def test_log_disabled_does_not_query_pool_stats(environment: Environment, monkeypatch: pytest.MonkeyPatch) -> None:
    """A1：用禁止查询和计时的替身，验证关闭统计时仍能对账、读块和等待。"""
    manager, runtime, scheduler, _ = environment
    monkeypatch.setattr(runtime.pool, "stats", lambda: pytest.fail("disabled logging requested pool stats"))
    clock = SimpleNamespace(perf_counter=lambda: pytest.fail("disabled logging read the clock"))
    monkeypatch.setattr(sched_prefetch, "time", clock)
    manager.reconcile(scheduler=scheduler)
    manager.reconcile(scheduler=scheduler)
    request = _urgent(manager, rows={1: [1]}, priority=(sched_prefetch.URGENT, 1, 0, 0))
    assert manager.wait_urgent(request=request, timeout=5)
    manager.release_batch(batch=SimpleNamespace(forward_iter=1))


def test_idle_final_stats_logged_once_and_rearmed(
    environment: Environment, caplog: pytest.LogCaptureFixture,
) -> None:
    """A7：模拟请求完成及运行批排空，验证每轮空闲零引用日志只发一次。"""
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
    manager.reconcile(scheduler=scheduler)
    _wait_segment(manager=manager, req=req)
    req.finished_reason = object()
    scheduler.waiting_queue = []
    manager.reconcile(scheduler=scheduler, running_batch=SimpleNamespace(reqs=[object()]))
    assert sum("sched prefetch stats:" in record.getMessage() for record in caplog.records) == 1
    manager.reconcile(scheduler=scheduler)
    manager.reconcile(scheduler=scheduler)
    messages = [record.getMessage() for record in caplog.records if "sched prefetch stats:" in record.getMessage()]
    assert len(messages) == 2
    for message in messages:
        stats = json.loads(message.removeprefix("Engram SSD sched prefetch stats: "))
        assert stats["phase"] == "idle"
        assert stats["referenced_slots"] == 0
