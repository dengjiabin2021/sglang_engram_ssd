"""验证原生行池引用、取消、LRU 与 io_uring 按槽写入；需目标 NVIDIA 环境。"""

from __future__ import annotations

import importlib
import os
import sys
import threading
from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path
    from types import ModuleType
    from typing import Any

torch = pytest.importorskip("torch")
pytestmark = [
    pytest.mark.skipif(sys.platform != "linux", reason="The native reader requires Linux"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="The real pinned pool requires CUDA"),
]


@pytest.fixture(scope="module")
def native() -> ModuleType:
    """A3：原生扩展缺失时跳过，不给生产接口增加 CPU 分配回退。"""
    name = "sglang_engram_ssd.query._C"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as error:
        if error.name != name:
            raise
        pytest.skip("Build the native Engram SSD extension on the test machine")


@pytest.fixture
def make_pool(native: ModuleType) -> Iterator[Callable[..., Any]]:
    pools = []

    def create(slots: int = 4, track_stats: bool = True) -> Any:
        pool = native.PrefetchRowPool(dim=256, pool_bytes=slots * 264, track_stats=track_stats)
        pools.append(pool)
        return pool

    yield create
    for pool in pools:
        pool.close()


def ids(values: Sequence[int]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.int64, device="cpu")


def publish(pool: Any, handle: Any) -> None:
    pool.mark_ready(slot_ids=handle.missing_slot_ids)


def test_aligned_storage_and_capacity(make_pool: Callable[..., Any]) -> None:
    """A3：真实 pinned 表连续、容量正确、两个起点均为 4096 字节对齐。"""
    pool = make_pool(slots=3)
    assert pool.capacity == 3
    assert pool.weight.shape == (3, 256)
    assert pool.scale.shape == (3, 8)
    assert pool.weight.dtype == pool.scale.dtype == torch.uint8
    assert pool.weight.is_contiguous() and pool.scale.is_contiguous()
    assert pool.weight.is_pinned() and pool.scale.is_pinned()
    assert pool.weight.data_ptr() % 4096 == pool.scale.data_ptr() % 4096 == 0
    assert pool.weight_device_ptr > 0 and pool.scale_device_ptr > 0
    assert pool.weight_stride == pool.weight.stride(0) == 256
    assert pool.scale_stride == pool.scale.stride(0) == 8
    assert pool.stats()["payload_bytes"] == 3 * 264


def test_empty_owner_and_invalid_publication(make_pool: Callable[..., Any]) -> None:
    """A3：空 owner 立即就绪；非法批量发布不能提前发布其中的有效槽。"""
    pool = make_pool(slots=2)
    empty = pool.acquire(layer_id=1, row_ids=ids([]))
    assert pool.is_ready(handle=empty)
    pool.release(handle=empty)
    resolution = pool.resolve(layer_id=1, row_ids=ids([]))
    assert resolution.slot_ids.numel() == resolution.hit_mask.numel() == 0
    pool.release(handle=resolution.handle)
    handle = pool.acquire(layer_id=1, row_ids=ids([1]))
    free_slot = 1 - handle.missing_slot_ids.item()
    with pytest.raises(RuntimeError, match="PENDING"):
        pool.mark_ready(slot_ids=ids([handle.missing_slot_ids.item(), free_slot]))
    assert not pool.is_ready(handle=handle)
    with pytest.raises(IndexError, match="non-negative"):
        pool.acquire(layer_id=1, row_ids=ids([2, -1]))
    assert pool.stats()["occupied_slots"] == 1
    pool.cancel(handle=handle)


def test_unique_rows_layers_and_resolve_positions(make_pool: Callable[..., Any]) -> None:
    """A3：行键包含层号，重复行只占一个槽，resolve 保留重复位置。"""
    pool = make_pool(slots=3)
    first = pool.acquire(layer_id=1, row_ids=ids([7, 7, 9]))
    second = pool.acquire(layer_id=14, row_ids=ids([7]))
    assert torch.equal(first.missing_row_ids, ids([7, 9]))
    assert torch.equal(second.missing_row_ids, ids([7]))
    assert pool.stats()["occupied_slots"] == 3
    assert pool.stats()["referenced_slots"] == 3
    assert not pool.is_ready(handle=first)
    pending = pool.resolve(layer_id=1, row_ids=ids([7, 9]))
    assert not pending.hit_mask.any().item()
    pool.release(handle=pending.handle)
    publish(pool, first)
    publish(pool, second)
    result = pool.resolve(layer_id=1, row_ids=ids([9, 7, 9, 99]))
    assert torch.equal(result.hit_mask, torch.tensor([True, True, True, False]))
    assert torch.equal(
        result.slot_ids,
        ids([first.missing_slot_ids[1].item(), first.missing_slot_ids[0].item(),
             first.missing_slot_ids[1].item(), -1]),
    )
    pool.release(handle=first)
    pool.release(handle=second)
    assert pool.stats()["referenced_slots"] == 2
    pool.release(handle=result.handle)
    pool.release(handle=result.handle)
    assert pool.stats()["referenced_slots"] == 0


def test_acquire_is_atomic_when_existing_cache_must_be_kept(make_pool: Callable[..., Any]) -> None:
    """A3：空间不足不增加部分引用，不能淘汰本次 acquire 也需要的缓存行。"""
    pool = make_pool(slots=1)
    first = pool.acquire(layer_id=1, row_ids=ids([10]))
    publish(pool, first)
    pool.release(handle=first)
    before = pool.stats()
    assert pool.acquire(layer_id=1, row_ids=ids([10, 20])) is None
    assert pool.stats() == before
    result = pool.resolve(layer_id=1, row_ids=ids([10]))
    assert result.hit_mask.item()
    pool.release(handle=result.handle)
    replacement = pool.acquire(layer_id=1, row_ids=ids([20]))
    assert torch.equal(replacement.missing_row_ids, ids([20]))
    missing = pool.resolve(layer_id=1, row_ids=ids([10]))
    assert not missing.hit_mask.item()
    pool.release(handle=missing.handle)
    pool.cancel(handle=replacement)


def test_lru_only_evicts_unreferenced_ready(make_pool: Callable[..., Any]) -> None:
    """A3：resolve 临时引用保护 READY，释放后按最近使用次序淘汰。"""
    pool = make_pool(slots=2)
    for row in (10, 20):
        handle = pool.acquire(layer_id=1, row_ids=ids([row]))
        publish(pool, handle)
        pool.release(handle=handle)
    retained = pool.resolve(layer_id=1, row_ids=ids([10]))
    pool.release(handle=retained.handle)
    replacement = pool.acquire(layer_id=1, row_ids=ids([30]))
    result = pool.resolve(layer_id=1, row_ids=ids([10, 20]))
    assert torch.equal(result.hit_mask, torch.tensor([True, False]))
    assert pool.acquire(layer_id=1, row_ids=ids([40])) is None
    pool.release(handle=result.handle)
    pool.cancel(handle=replacement)


def test_pending_zero_references_stays_until_io_publication(make_pool: Callable[..., Any]) -> None:
    """A3：release 不覆盖可能仍被 I/O 写入的 PENDING；关闭必须等待读取完成。"""
    pool = make_pool(slots=1)
    handle = pool.acquire(layer_id=1, row_ids=ids([1]))
    slots = handle.missing_slot_ids.clone()
    pool.release(handle=handle)
    assert pool.stats()["pending_slots"] == 1
    assert pool.stats()["referenced_slots"] == 0
    assert pool.acquire(layer_id=1, row_ids=ids([2])) is None
    with pytest.raises(RuntimeError, match="completed reads"):
        pool.close()
    pool.mark_ready(slot_ids=slots)
    replacement = pool.acquire(layer_id=1, row_ids=ids([2]))
    pool.cancel(handle=replacement)


def test_cancel_rolls_back_new_pending_and_is_idempotent(make_pool: Callable[..., Any]) -> None:
    """A3：跨层占槽失败可撤销已新建但尚未读取的 PENDING，不留下池满泄漏。"""
    pool = make_pool(slots=2)
    handle = pool.acquire(layer_id=1, row_ids=ids([1, 2]))
    assert pool.acquire(layer_id=14, row_ids=ids([1])) is None
    pool.cancel(handle=handle)
    pool.cancel(handle=handle)
    pool.release(handle=handle)
    assert pool.stats()["occupied_slots"] == 0
    replacement = pool.acquire(layer_id=14, row_ids=ids([1, 2]))
    assert replacement is not None
    pool.cancel(handle=replacement)


def test_cancel_borrower_does_not_fail_shared_pending(make_pool: Callable[..., Any]) -> None:
    """A3：取消未创建该槽的借用句柄不会破坏生产者的共享 PENDING。"""
    pool = make_pool(slots=1)
    producer = pool.acquire(layer_id=1, row_ids=ids([3]))
    borrower = pool.acquire(layer_id=1, row_ids=ids([3, 3]))
    assert borrower.missing_row_ids.numel() == 0
    pool.cancel(handle=borrower)
    assert not pool.is_ready(handle=producer)
    publish(pool, producer)
    assert pool.is_ready(handle=producer)
    pool.release(handle=producer)


def test_cancel_producer_fails_shared_pending_without_permanent_wait(make_pool: Callable[..., Any]) -> None:
    """A3：生产者取消共享 PENDING 后显式失败，最后引用释放即回收。"""
    pool = make_pool(slots=1)
    producer = pool.acquire(layer_id=1, row_ids=ids([3]))
    borrower = pool.acquire(layer_id=1, row_ids=ids([3]))
    pool.cancel(handle=producer)
    with pytest.raises(RuntimeError, match="cancelled"):
        pool.is_ready(handle=borrower)
    with pytest.raises(RuntimeError, match="cancelled"):
        pool.acquire(layer_id=1, row_ids=ids([3]))
    with pytest.raises(RuntimeError, match="cancelled"):
        pool.resolve(layer_id=1, row_ids=ids([3]))
    pool.release(handle=borrower)
    assert pool.stats()["occupied_slots"] == 0
    replacement = pool.acquire(layer_id=1, row_ids=ids([3]))
    pool.cancel(handle=replacement)


def test_cancel_ready_only_releases_its_own_reference(make_pool: Callable[..., Any]) -> None:
    """A3：撤回已完成预取只减引用，不能删除其他请求或 forward 仍使用的 READY。"""
    pool = make_pool(slots=1)
    producer = pool.acquire(layer_id=1, row_ids=ids([3]))
    publish(pool, producer)
    borrower = pool.acquire(layer_id=1, row_ids=ids([3]))
    lease = pool.resolve(layer_id=1, row_ids=ids([3]))
    pool.cancel(handle=producer)
    assert pool.is_ready(handle=borrower)
    pool.release(handle=borrower)
    assert pool.acquire(layer_id=1, row_ids=ids([4])) is None
    pool.release(handle=lease.handle)
    assert pool.stats()["cached_slots"] == 1


def test_concurrent_shared_handles(make_pool: Callable[..., Any]) -> None:
    """A3：多个调用线程共享同一 PENDING/READY，引用归零且只提交唯一行。"""
    pool = make_pool(slots=2)
    producer = pool.acquire(layer_id=1, row_ids=ids([1, 2]))
    acquired = threading.Barrier(5, timeout=10)
    published = threading.Barrier(5, timeout=10)
    errors = []

    def consume() -> None:
        handle = None
        try:
            handle = pool.acquire(layer_id=1, row_ids=ids([2, 1, 2]))
            assert handle.missing_row_ids.numel() == 0
            acquired.wait()
            published.wait()
            assert pool.is_ready(handle=handle)
        except BaseException as error:
            errors.append(error)
            acquired.abort()
            published.abort()
        finally:
            if handle is not None:
                pool.release(handle=handle)

    threads = [threading.Thread(target=consume) for _ in range(4)]
    for thread in threads:
        thread.start()
    try:
        acquired.wait()
        publish(pool, producer)
        published.wait()
    finally:
        for thread in threads:
            thread.join(timeout=10)
        pool.release(handle=producer)
    assert all(not thread.is_alive() for thread in threads)
    assert not errors
    assert pool.stats()["referenced_slots"] == 0
    assert pool.stats()["submitted_rows"] == 2


def test_stats_optional_and_close_guard(make_pool: Callable[..., Any]) -> None:
    """A3：累计统计受开关控制，close 拒绝活动引用且成功后幂等。"""
    for enabled in (False, True):
        pool = make_pool(slots=2, track_stats=enabled)
        handle = pool.acquire(layer_id=1, row_ids=ids([1, 1, 2]))
        with pytest.raises(RuntimeError, match="released handles"):
            pool.close()
        publish(pool, handle)
        pool.record_forward(hits=3, misses=2, miss_ms=1.25)
        pool.record_deferred(count=4)
        pool.release(handle=handle)
        snapshot = pool.stats()
        assert snapshot["submitted_rows"] == (2 if enabled else 0)
        assert snapshot["dedup_saved_rows"] == (1 if enabled else 0)
        assert snapshot["forward_hits"] == (3 if enabled else 0)
        assert snapshot["forward_misses"] == (2 if enabled else 0)
        assert snapshot["miss_read_ms"] == (1.25 if enabled else 0)
        assert snapshot["deferred_requests"] == (4 if enabled else 0)
        pool.close()
        pool.close()
        pool.release(handle=handle)
        with pytest.raises(RuntimeError, match="closed"):
            pool.acquire(layer_id=1, row_ids=ids([1]))


def test_reader_slots_preserve_order_and_borrow_file_descriptors(
    native: ModuleType, tmp_path: Path, make_pool: Callable[..., Any],
) -> None:
    """A3/A0：按槽写入与原行字节相同，重复目标拒绝，关闭不关闭借用的 fd。"""
    rows, dim = 4, 256
    weight = torch.arange(rows * dim, dtype=torch.int64).remainder(251).to(torch.uint8).view(rows, dim)
    scale = torch.arange(rows * 8, dtype=torch.int64).remainder(29).to(torch.uint8).view(rows, 8)
    weight_path, scale_path = tmp_path / "weight.bin", tmp_path / "scale.bin"
    weight_path.write_bytes(b"w" * 37 + bytes(weight.reshape(-1).tolist()))
    scale_path.write_bytes(b"s" * 19 + bytes(scale.reshape(-1).tolist()))
    weight_fd, scale_fd = os.open(weight_path, os.O_RDONLY), os.open(scale_path, os.O_RDONLY)
    reader = None
    try:
        spec = SimpleNamespace(
            layer_id=1, rows=rows, dim=dim, weight_fd=weight_fd, scale_fd=scale_fd,
            weight_offset=37, scale_offset=19,
        )
        reader = native.BatchRowReader(tables=[spec], io_uring_queue_depth=2)
        weight_out, scale_out = reader.allocate_workspace(layer_id=1, capacity=4)
        weight_out.fill_(255)
        scale_out.fill_(255)
        reader.read_rows_to_slots(
            layer_id=1, row_ids=ids([2, 1]), slot_ids=ids([3, 0]),
            weight_base=weight_out, scale_base=scale_out,
        )
        assert torch.equal(weight_out[3], weight[2])
        assert torch.equal(weight_out[0], weight[1])
        assert torch.equal(scale_out[3], scale[2])
        assert torch.equal(scale_out[0], scale[1])
        assert torch.equal(weight_out[1:3], torch.full((2, dim), 255, dtype=torch.uint8))
        before_weight, before_scale = weight_out.clone(), scale_out.clone()
        with pytest.raises(ValueError, match="unique"):
            reader.read_rows_to_slots(
                layer_id=1, row_ids=ids([0, 1]), slot_ids=ids([1, 1]),
                weight_base=weight_out, scale_base=scale_out,
            )
        assert torch.equal(weight_out, before_weight)
        assert torch.equal(scale_out, before_scale)
        with pytest.raises(IndexError, match="outside"):
            reader.read_rows_to_slots(
                layer_id=1, row_ids=ids([0, rows]), slot_ids=ids([1, 2]),
                weight_base=weight_out, scale_base=scale_out,
            )
        assert torch.equal(weight_out, before_weight)
        reader.read_rows(layer_id=1, row_ids=ids([2, 2]), weight_out=weight_out, scale_out=scale_out)
        assert torch.equal(weight_out[:2], weight[2].expand(2, dim))
        assert torch.equal(scale_out[:2], scale[2].expand(2, 8))
        pool = make_pool(slots=4)
        handle = pool.acquire(layer_id=1, row_ids=ids([2, 1, 2]))
        reader.read_rows_to_slots(
            layer_id=1, row_ids=handle.missing_row_ids, slot_ids=handle.missing_slot_ids,
            weight_base=pool.weight, scale_base=pool.scale,
        )
        publish(pool, handle)
        result = pool.resolve(layer_id=1, row_ids=ids([1, 2, 1]))
        assert result.hit_mask.all().item()
        assert torch.equal(pool.weight.index_select(0, result.slot_ids), weight[ids([1, 2, 1])])
        assert torch.equal(pool.scale.index_select(0, result.slot_ids), scale[ids([1, 2, 1])])
        pool.release(handle=result.handle)
        pool.release(handle=handle)
        reader.close()
        assert os.pread(weight_fd, dim, 37) == bytes(weight[0].tolist())
        assert os.pread(scale_fd, 8, 19) == bytes(scale[0].tolist())
    finally:
        if reader is not None:
            reader.close()
        os.close(weight_fd)
        os.close(scale_fd)
