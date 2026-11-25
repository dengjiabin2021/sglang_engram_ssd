"""在真实 torchrun TP8 中核验显存批次查询与同步查询的逐位一致性。"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch

from sglang_engram_ssd import sched_prefetch
from sglang_engram_ssd.catalog import EngramSsdCatalog
from sglang_engram_ssd.runtime import EngramSsdRuntime

pytestmark = pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8,
    reason="A0/A4 requires torchrun --standalone --nproc-per-node=8",
)


@pytest.mark.parametrize("warm", ["none", "partial", "all"])
def test_real_tp8_batch_matches_sync(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    warm: str,
) -> None:
    """A0/A4：真实八 rank NCCL 合并，覆盖全未命中、部分命中及全命中。"""
    import torch.distributed as dist
    import sglang.srt.distributed as sglang_dist
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()

    def reduce_tensor(value: torch.Tensor) -> torch.Tensor:
        """对 GPU BF16 局部值执行真实 NCCL 合并并返回结果。"""
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        return value

    monkeypatch.setattr(sglang_dist, "tensor_model_parallel_all_reduce", reduce_tensor)
    catalog, weights, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=rank, tp_size=world,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        sched_prefetch_enabled=True, pool_bytes=264 * 32, batch_tokens=8,
    )
    pool = runtime.pool
    assert pool is not None and runtime.reader is not None
    handle = None
    try:
        cpu_ids = torch.tensor([[0, 5, 5, 22], [7, 12, 13, 0]], dtype=torch.int64)
        ids = cpu_ids.to(device=cuda_native_device)
        config = SimpleNamespace(window_batches=2, chunk_rows=3, log_interval=0)
        monkeypatch.setattr(sched_prefetch, "get_config", lambda: config)

        def predict(*, token_ids: list[int], start: int, end: int) -> torch.Tensor:
            """按测试的 token 下标返回 CPU int64 [T,1,4] hash。"""
            return cpu_ids[token_ids[start:end], None]

        runtime.set_request_hasher(hasher=SimpleNamespace(
            layer_ids=(1,), state=SimpleNamespace(offsets=cpu_ids[:1]), compute_request_row_ids=predict,
        ))
        scheduler = SimpleNamespace(chunked_prefill_size=8, tp_group=SimpleNamespace(cpu_group=None))
        manager = runtime.get_sched_prefetch(scheduler=scheduler)
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
        req = SimpleNamespace(origin_input_ids=[0, 1], output_ids=[], full_untruncated_fill_ids=[0, 1])
        scheduled = SimpleNamespace(
            reqs=[req], prefix_lens=[0], extend_lens=[2], extend_num_tokens=2,
            forward_mode=ForwardMode.EXTEND, input_ids=ids, forward_iter=1,
        )
        forward = SimpleNamespace(forward_mode=ForwardMode.EXTEND, input_ids=ids)
        manager.register_request(req=req)
        manager.start_batch(batch=scheduled, iteration=1)
        runtime.bind_batch(forward_batch=forward, hash_ids=ids[:, None])
        gathered = runtime.reduce_output(out=runtime.lookup_batch(layer_id=1, indices=ids, forward_batch=forward))
        runtime.end_batch(batch=scheduled)
        expected = weights.index_select(0, cpu_ids.reshape(-1)).view(torch.float8_e4m3fn)
        expected = expected.float().to(torch.bfloat16).view(2, 4, 256)
        assert torch.equal(synchronous.cpu(), expected)
        assert torch.equal(gathered.cpu(), expected)
        pool.release(handle=handle)
        handle = None
        assert pool.stats()["referenced_slots"] == 0
        dist.barrier()
    finally:
        if handle is not None:
            pool.release(handle=handle, discard_pending=True)
        runtime.close()
        if owned_group:
            dist.destroy_process_group()


def test_real_tp8_background_failure_uses_one_gloo_min() -> None:
    """完成标准 3：真实八 rank 对单个失败标志做 CPU MIN，统一报告一个 rank 的后台错误。"""
    import torch.distributed as dist

    # 固定较短通信超时，使漏发 collective 的错误实现明确失败，避免目标机无限等待。
    owned_group = not dist.is_initialized()
    if owned_group:
        dist.init_process_group(backend="gloo", timeout=timedelta(seconds=20))
    cpu_group = dist.group.WORLD if owned_group else dist.new_group(
        backend="gloo", timeout=timedelta(seconds=20),
    )
    rank = dist.get_rank()
    try:
        # 这里只需要真实归约，不创建 reader、模型或后台线程；只有 TP3 持有本地错误。
        controller = object.__new__(sched_prefetch.SchedulerPrefetchController)
        controller.runtime = SimpleNamespace(tp_size=dist.get_world_size())
        controller.cpu_group = cpu_group
        controller._condition = threading.Condition()
        cause = OSError("synthetic background failure on TP3") if rank == 3 else None
        controller._failure = cause
        with pytest.raises(RuntimeError, match="failed on a TP rank") as error:
            controller._synchronize_failure()
        assert error.value.__cause__ is cause

        # 每个 rank 都须走过相同异常边界；本测试不让健康 rank 提前进入下一模型通信。
        reached = torch.ones(1, dtype=torch.int32, device="cpu")
        dist.all_reduce(reached, op=dist.ReduceOp.SUM, group=cpu_group)
        assert reached.item() == 8
    finally:
        if not owned_group:
            dist.destroy_process_group(cpu_group)
        if owned_group:
            dist.destroy_process_group()
