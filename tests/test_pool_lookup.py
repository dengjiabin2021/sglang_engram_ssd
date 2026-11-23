"""验证 pinned 行池直接消费、同步补读以及 CUDA event 临时引用保护。"""

from __future__ import annotations

from collections.abc import Callable
import os
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_engram_ssd.catalog import EngramSsdCatalog
from sglang_engram_ssd.runtime import EngramSsdRuntime, PoolUse


@pytest.mark.parametrize("warm", ["none", "partial", "all"])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (8, 2), (8, 5)])
@pytest.mark.parametrize("dedup", [False, True])
def test_lookup_from_pool_matches_synchronous(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    warm: str,
    tp_size: int,
    tp_rank: int,
    dedup: bool,
) -> None:
    """A4：全命中／部分命中／全未命中、重复和owner恢复都逐位相同。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    scale = catalog.views["model.layers.1.engram.embed.scale"]
    # 各行各列不同的 E8M0 数据可以识别错误的 scale 槽位／步长。
    raw_scale = (torch.arange(23 * 8, dtype=torch.int64) % 7 + 124).to(torch.uint8)
    raw_scale[0] = 0
    with scale.path.open("r+b") as stream:
        stream.seek(scale.offset)
        stream.write(bytes(raw_scale.tolist()))
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=tp_rank, tp_size=tp_size,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        dedup_enabled=dedup, sched_prefetch_enabled=True, pool_bytes=264 * 32, log_interval=1,
    )
    pool = runtime.pool
    assert pool is not None and runtime.prefetch_reader is not None
    handle = None
    try:
        ids = torch.tensor([[0, 5, 6, 5], [22, 0, 6, 7]], dtype=torch.int64, device=cuda_native_device)
        expected = runtime.lookup(layer_id=1, indices=ids)
        lo, hi = 23 * tp_rank // tp_size, 23 * (tp_rank + 1) // tp_size
        rows = torch.arange(lo, hi, dtype=torch.int64)
        if warm == "none":
            rows = rows[:0]
        elif warm == "partial":
            rows = rows[::2]
        handle = pool.acquire(layer_id=1, row_ids=rows.contiguous())
        assert handle is not None
        runtime.prefetch_reader.read_rows_to_slots(
            layer_id=1, row_ids=handle.missing_row_ids, slot_ids=handle.missing_slot_ids,
            weight_base=pool.weight, scale_base=pool.scale,
        )
        pool.mark_ready(slot_ids=handle.missing_slot_ids)
        actual = runtime.lookup_from_pool(layer_id=1, indices=ids)
        # 请求引用先释放；resolve 的独立引用仍保护 GPU 正在读取的池内存。
        pool.release(handle=handle)
        handle = None
        assert torch.equal(actual.cpu(), expected.cpu())
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        owner_ids = ids.cpu().reshape(-1)
        owner_ids = owner_ids[(owner_ids >= lo) & (owner_ids < hi)]
        expected_hits = int(torch.isin(owner_ids, rows).sum())
        stats = pool.stats()
        assert stats["forward_hits"] == expected_hits
        assert stats["forward_misses"] == owner_ids.numel() - expected_hits
        runtime.collect_pool_uses(wait=True)
        assert pool.stats()["referenced_slots"] == 0
    finally:
        runtime.collect_pool_uses(wait=True)
        if handle is not None:
            pool.release(handle=handle)
        runtime.close()


def test_pending_rows_are_not_consumed(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
) -> None:
    """A4：PENDING 未读满的槽必须作为 miss，同步补读，不暴露未初始化数据。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=0, tp_size=1,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        sched_prefetch_enabled=True, pool_bytes=264 * 8,
    )
    assert runtime.pool is not None
    handle = runtime.pool.acquire(layer_id=1, row_ids=torch.tensor([2, 4], dtype=torch.int64))
    assert handle is not None
    try:
        ids = torch.tensor([[2, 4, 2]], dtype=torch.int64, device=cuda_native_device)
        actual = runtime.lookup_from_pool(layer_id=1, indices=ids)
        assert torch.equal(actual, runtime.lookup(layer_id=1, indices=ids))
    finally:
        runtime.collect_pool_uses(wait=True)
        runtime.pool.cancel(handle=handle)
        runtime.close()


def test_forward_defers_background_failure_to_scheduler_min(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
) -> None:
    """A4/A7/N1：后台已失败时 forward 仍完成同步／PENDING 补读，不在单个 rank 提前抛错。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=2, tp_size=8,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        sched_prefetch_enabled=True, pool_bytes=264 * 8,
    )
    pool = runtime.pool
    assert pool is not None
    handle = pool.acquire(layer_id=1, row_ids=torch.tensor([5, 6], dtype=torch.int64))
    assert handle is not None

    def reject_local_failure() -> None:
        raise AssertionError("Background failure must propagate through the scheduler MIN")

    try:
        ids = torch.tensor([[5, 6, 5, 0]], dtype=torch.int64, device=cuda_native_device)
        expected = runtime.lookup(layer_id=1, indices=ids)
        # 保留旧检查同名替身，确保未来重新加入本地 raise 时本用例确实失败。
        # 未就绪槽不被 GPU 使用；独立前台 reader 可以在后台失败后继续补读。
        runtime.sched_prefetch = SimpleNamespace(
            _failure=OSError("synthetic background failure"), raise_if_failed=reject_local_failure,
            notify_pool_release=lambda: None, close=lambda: None,
        )
        assert torch.equal(runtime.lookup(layer_id=1, indices=ids), expected)
        assert torch.equal(runtime.lookup_from_pool(layer_id=1, indices=ids), expected)
    finally:
        runtime.collect_pool_uses(wait=True)
        pool.cancel(handle=handle)
        runtime.close()


def test_forward_misses_do_not_wait_for_prefetch_reader(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
) -> None:
    """A4/A7：后台持有自己的 reader 锁时，前台 PENDING 补读仍可完成。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=0, tp_size=1,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        sched_prefetch_enabled=True, pool_bytes=264 * 8,
    )
    pool = runtime.pool
    background_reader = runtime.prefetch_reader
    assert pool is not None and background_reader is not None and runtime.reader is not None
    assert background_reader is not runtime.reader
    assert background_reader._lock is not runtime.reader._lock
    handle = pool.acquire(layer_id=1, row_ids=torch.tensor([2, 4], dtype=torch.int64))
    assert handle is not None
    holding_reader = threading.Event()
    release_background = threading.Event()
    hold_timed_out = threading.Event()
    finished = threading.Event()
    failures: list[BaseException] = []

    def prefetch() -> None:
        try:
            # 显式保持包装层的互斥锁，稳定复现整段 I/O 占用 reader 的窗口。
            # 超时只负责让错误实现退出测试；不能把超时后的完成当成并发成功。
            with background_reader._lock:
                holding_reader.set()
                if not release_background.wait(timeout=10):
                    hold_timed_out.set()
            background_reader.read_rows_to_slots(
                layer_id=1, row_ids=handle.missing_row_ids, slot_ids=handle.missing_slot_ids,
                weight_base=pool.weight, scale_base=pool.scale,
            )
            pool.mark_ready(slot_ids=handle.missing_slot_ids)
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()

    worker = threading.Thread(target=prefetch, name="test-engram-prefetch", daemon=True)
    try:
        ids = torch.tensor([[2, 4, 2]], dtype=torch.int64, device=cuda_native_device)
        expected = runtime.lookup(layer_id=1, indices=ids)
        worker.start()
        assert holding_reader.wait(timeout=10), "Background reader did not acquire its lock"
        # 槽尚为 PENDING，必须从前台 ring 补读，不能等后台读完才返回。
        actual = runtime.lookup_from_pool(layer_id=1, indices=ids)
        assert not hold_timed_out.is_set(), "Forward waited for the background reader lock"
        assert not finished.is_set()
        assert torch.equal(actual.cpu(), expected.cpu())
    finally:
        release_background.set()
        if worker.ident is not None:
            worker.join(timeout=10)
            assert not worker.is_alive(), "Background read did not finish before pool cleanup"
        runtime.collect_pool_uses(wait=True)
        # 已完成的 READY 与失败后已收齐 I/O 的 PENDING 都可通过 cancel 归还引用。
        pool.cancel(handle=handle)
        runtime.close()
    assert failures == []


@pytest.mark.parametrize("enabled", [False, True])
def test_prefetch_reader_lifetime_matches_feature_switch(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    enabled: bool,
) -> None:
    """A1/A3：关闭预取不分配第二 reader；关闭 runtime 后两个 reader 和借用 FD 均失效。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=0, tp_size=1,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        sched_prefetch_enabled=enabled, pool_bytes=264 * 8,
    )
    readers = [runtime.reader]
    fds = tuple(runtime.fds.values())
    try:
        assert runtime.reader is not None
        if enabled:
            assert runtime.prefetch_reader is not None
            assert runtime.prefetch_reader is not runtime.reader
            readers.append(runtime.prefetch_reader)
        else:
            assert runtime.prefetch_reader is None and runtime.pool is None
        for fd in fds:
            os.fstat(fd)
    finally:
        runtime.close()
    assert runtime.reader is None and runtime.prefetch_reader is None
    for reader in readers:
        assert reader is not None
        with pytest.raises(RuntimeError, match="reader is closed"):
            reader.allocate_workspace(layer_id=1, capacity=1)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


def test_decode_keeps_synchronous_embedding_path() -> None:
    """A1/A4：decode 不使用调度预取行池，CPU替身无需GPU或checkpoint。"""
    from sglang_engram_ssd.hooks.engram_embedding import forward_ssd_embedding

    expected = torch.ones((1, 1, 256), dtype=torch.bfloat16)
    calls: list[str] = []

    def lookup(*, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        calls.append("synchronous")
        return expected

    def lookup_from_pool(*, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        raise AssertionError("Decode must not consume the scheduler pool")

    runtime = SimpleNamespace(
        sched_prefetch_enabled=True, lookup=lookup, lookup_from_pool=lookup_from_pool,
        reduce_output=lambda out: out,
    )
    batch = SimpleNamespace(
        contains_mm_inputs=lambda: False, forward_mode=SimpleNamespace(is_extend=lambda: False),
    )
    actual = forward_ssd_embedding(
        original_fn=lambda *args, **kwargs: None, self=SimpleNamespace(ssd=runtime, layer_id=1),
        indices=torch.tensor([[2]], dtype=torch.int64), forward_batch=batch,
    )
    assert actual is expected
    assert calls == ["synchronous"]


def test_unfinished_gpu_use_retains_reference() -> None:
    """A3/A4：纯CPU事件替身证明未完成event不能触发release。"""
    released: list[object] = []
    runtime = object.__new__(EngramSsdRuntime)
    notifications: list[str] = []
    runtime.sched_prefetch = SimpleNamespace(notify_pool_release=lambda: notifications.append("release"))
    runtime.pool = SimpleNamespace(release=lambda *, handle: released.append(handle))
    runtime._pool_uses_lock = threading.Lock()
    ready = [False]
    handle = object()
    runtime._pool_uses = [PoolUse(
        event=SimpleNamespace(query=lambda: ready[0]), handle=handle,
        slot_ids=torch.empty(0, dtype=torch.int64), stream=SimpleNamespace(),
    )]
    runtime.collect_pool_uses()
    assert released == []
    assert notifications == []
    assert len(runtime._pool_uses) == 1
    ready[0] = True
    runtime.collect_pool_uses()
    assert released == [handle]
    assert notifications == ["release"]
    assert runtime._pool_uses == []
    runtime.collect_pool_uses()
    assert notifications == ["release"]


def test_unrecorded_event_is_only_released_after_stream_drain() -> None:
    """A4：event记录失败后保留lease，关闭时收齐原生产流才能回收。"""
    actions: list[str] = []
    runtime = object.__new__(EngramSsdRuntime)
    runtime.sched_prefetch = None
    runtime.pool = SimpleNamespace(release=lambda *, handle: actions.append("release"))
    runtime._pool_uses_lock = threading.Lock()
    runtime._pool_uses = [PoolUse(
        event=None, handle=object(), slot_ids=torch.empty(0, dtype=torch.int64),
        stream=SimpleNamespace(synchronize=lambda: actions.append("synchronize")),
    )]
    runtime.collect_pool_uses()
    assert actions == []
    runtime.collect_pool_uses(wait=True)
    assert actions == ["synchronize", "release"]
