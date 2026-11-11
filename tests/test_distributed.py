"""在真实 torchrun TP8 中核验同步查询与行池查询的逐位一致性。"""

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import torch

from sglang_engram_ssd.catalog import EngramSsdCatalog
from sglang_engram_ssd.runtime import EngramSsdRuntime

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8,
    reason="A0/A4 requires torchrun --standalone --nproc-per-node=8",
)


@pytest.mark.parametrize("warm", ["none", "partial", "all"])
def test_real_tp8_pool_matches_sync(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    warm: str,
) -> None:
    """A0/A4：真实八 rank NCCL 合并，覆盖全未命中、部分命中及全命中。"""
    import torch.distributed as dist
    import sglang.srt.distributed as sglang_dist

    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    def reduce_tensor(value: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    monkeypatch.setattr(sglang_dist, "tensor_model_parallel_all_reduce", reduce_tensor)
    catalog, weights, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=rank, tp_size=world,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        sched_prefetch_enabled=True, pool_bytes=264 * 32,
    )
    pool = runtime.pool
    assert pool is not None and runtime.reader is not None
    handle = None
    try:
        ids = torch.tensor([[0, 5, 5, 22], [7, 12, 13, 0]], dtype=torch.int64, device=cuda_native_device)
        lo, hi = 23 * rank // world, 23 * (rank + 1) // world
        rows = torch.arange(lo, hi, dtype=torch.int64)
        if warm == "none":
            rows = rows[:0]
        elif warm == "partial":
            rows = rows[::2]
        handle = pool.acquire(layer_id=1, row_ids=rows.contiguous())
        assert handle is not None
        runtime.reader.read_rows_to_slots(
            layer_id=1, row_ids=handle.missing_row_ids, slot_ids=handle.missing_slot_ids,
            weight_base=pool.weight, scale_base=pool.scale,
        )
        pool.mark_ready(slot_ids=handle.missing_slot_ids)
        synchronous = runtime.reduce_output(out=runtime.lookup(layer_id=1, indices=ids))
        pooled = runtime.reduce_output(out=runtime.lookup_from_pool(layer_id=1, indices=ids))
        expected = weights.index_select(0, ids.cpu().reshape(-1)).view(torch.float8_e4m3fn)
        expected = expected.float().to(torch.bfloat16).view(2, 4, 256)
        assert torch.equal(synchronous.cpu(), expected)
        assert torch.equal(pooled.cpu(), expected)
        runtime.collect_pool_uses(wait=True)
        pool.release(handle=handle)
        handle = None
        assert pool.stats()["referenced_slots"] == 0
        dist.barrier()
    finally:
        runtime.collect_pool_uses(wait=True)
        if handle is not None:
            pool.release(handle=handle)
        runtime.close()
        if owned_group:
            dist.destroy_process_group()
