"""验证批次显存 gather、双缓冲生命周期和 decode GPU 就绪标志。"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from sglang_engram_ssd import sched_prefetch
from sglang_engram_ssd.catalog import EngramSsdCatalog
from sglang_engram_ssd.config import EngramSsdConfig
from sglang_engram_ssd.runtime import EngramSsdRuntime


def _batch_runtime(
    *, catalog: EngramSsdCatalog, device: torch.device, ids: torch.Tensor, monkeypatch: pytest.MonkeyPatch,
    tp_rank: int = 0, tp_size: int = 1, debug: bool = False,
) -> tuple[EngramSsdRuntime, sched_prefetch.SchedulerPrefetchController]:
    """用合成表与 CPU int64 [T,1,C] 哈希替身建立原生 runtime 和后台控制器。"""
    config = SimpleNamespace(window_batches=2, chunk_rows=1, log_interval=1)
    monkeypatch.setattr(sched_prefetch, "get_config", lambda: config)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=tp_rank, tp_size=tp_size,
        chunk_rows=3, io_uring_queue_depth=4, device=device,
        sched_prefetch_enabled=True, pool_bytes=264 * 24, log_interval=1,
        batch_tokens=8, debug_check_rows=debug,
    )

    def predict(*, token_ids: list[int], start: int, end: int) -> torch.Tensor:
        """以 token_ids 起止区间选出该请求的测试行号，返回 CPU int64 [T,1,C]。"""
        return ids[token_ids[start:end]]

    runtime.set_request_hasher(
        hasher=SimpleNamespace(layer_ids=(1,), state=SimpleNamespace(offsets=ids[0]), compute_request_row_ids=predict),
    )
    scheduler = SimpleNamespace(chunked_prefill_size=2, tp_group=SimpleNamespace(cpu_group=object()))
    return runtime, runtime.get_sched_prefetch(scheduler=scheduler)


def _batch(*, tokens: int, device: torch.device, decode: bool) -> tuple[SimpleNamespace, SimpleNamespace]:
    """创建 share 同一 input_ids 的调度和模型批次；tokens 为本轮输入数。"""
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    mode = ForwardMode.DECODE if decode else ForwardMode.EXTEND
    ids = torch.arange(tokens, dtype=torch.int64, device=device)
    token_ids = list(range(tokens))
    req = SimpleNamespace(origin_input_ids=token_ids, output_ids=[], full_untruncated_fill_ids=token_ids)
    batch = SimpleNamespace(
        reqs=[object() for _ in range(tokens)] if decode else [req],
        prefix_lens=[0], extend_lens=[tokens], extend_num_tokens=tokens, forward_iter=1,
        input_ids=ids, forward_mode=mode,
    )
    return batch, SimpleNamespace(input_ids=ids, forward_mode=mode)


@pytest.mark.parametrize("chunked", [8, None, -1])
def test_pool_startup_uses_hasher_layout_and_schedule_budget(
    monkeypatch: pytest.MonkeyPatch, chunked: int | None,
) -> None:
    """完成标准 4：真实 Hasher 安装入口接入行池定容，禁用分块时使用原生 prefill 上限。"""
    from sglang.srt import runtime_context

    from sglang_engram_ssd import query, runtime as runtime_module

    config = EngramSsdConfig(enabled=True, io_uring_queue_depth=4, chunk_rows=3, sched_prefetch_enabled=True)
    monkeypatch.setattr(runtime_module, "get_config", lambda: config)
    monkeypatch.setattr(
        runtime_context, "get_schedule", lambda: SimpleNamespace(chunked_prefill_size=chunked, max_prefill_tokens=16),
    )
    monkeypatch.setattr(query, "NativePrefetchRowPool", SimpleNamespace)
    runtime = object.__new__(EngramSsdRuntime)
    runtime.catalog = SimpleNamespace(layer_ids=(1, 14))
    runtime.dim, runtime.tp_size = 256, 8
    runtime.sched_prefetch_enabled, runtime.stats_enabled = True, False
    runtime.pool, runtime.batch_tokens = None, 0
    runtime.set_request_hasher(
        hasher=SimpleNamespace(layer_ids=(1, 14), state=SimpleNamespace(offsets=torch.zeros((2, 4)))),
    )
    assert runtime.pool.pool_bytes == (9 if chunked == 8 else 18) * 3 * 264


@pytest.mark.parametrize("warm", ["none", "partial", "all"])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (8, 2), (8, 5)])
def test_prefill_batch_gather_matches_synchronous(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device, warm: str, tp_size: int, tp_rank: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完成标准 2/4/7：READY／缺失行汇集后逐位等于同步 lookup，包含重复和非 owner 位置。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    scale = catalog.views["model.layers.1.engram.embed.scale"]
    raw_scale = (torch.arange(23 * 8, dtype=torch.int64) % 7 + 124).to(torch.uint8)
    raw_scale[0] = 0
    with scale.path.open("r+b") as stream:
        stream.seek(scale.offset)
        stream.write(bytes(raw_scale.tolist()))
    ids = torch.tensor([[[0, 5, 6, 5]], [[22, 0, 6, 7]]], dtype=torch.int64)
    runtime, manager = _batch_runtime(
        catalog=catalog, device=cuda_native_device, ids=ids, monkeypatch=monkeypatch, tp_rank=tp_rank, tp_size=tp_size,
    )
    handle = None
    try:
        gpu = ids.to(device=cuda_native_device)
        expected = runtime.lookup(layer_id=1, indices=gpu[:, 0])
        lo, hi = 23 * tp_rank // tp_size, 23 * (tp_rank + 1) // tp_size
        rows = torch.arange(lo, hi, dtype=torch.int64)
        if warm == "none":
            rows = rows[:0]
        elif warm == "partial":
            rows = rows[::2].contiguous()
        pool = runtime.pool
        handle = pool.acquire(layer_id=1, row_ids=rows)
        runtime.prefetch_reader.read_rows_to_slots(
            layer_id=1, row_ids=handle.missing_row_ids, slot_ids=handle.missing_slot_ids,
            weight_base=pool.weight, scale_base=pool.scale,
        )
        pool.mark_ready(slot_ids=handle.missing_slot_ids)
        pool.release(handle=handle)
        handle = None
        scheduled, forward = _batch(tokens=2, device=cuda_native_device, decode=False)
        manager.register_request(req=scheduled.reqs[0])
        manager.start_batch(batch=scheduled, iteration=1)
        runtime.bind_batch(forward_batch=forward, hash_ids=gpu)
        indices = gpu[:, 0]
        tensor_to = torch.Tensor.to

        def forbid_copy(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
            """拦截由模型索引地址发起的转换，其他张量沿用原张量转换。"""
            assert not self.is_cuda or self.data_ptr() != indices.data_ptr()
            return tensor_to(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "to", forbid_copy)
            actual = runtime.lookup_batch(layer_id=1, indices=indices, forward_batch=forward)
        runtime.end_batch(batch=scheduled)
        assert torch.equal(actual.cpu(), expected.cpu())
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        owner = ids[(ids >= lo) & (ids < hi)]
        stats = pool.stats()
        hits = int(torch.isin(owner, rows).sum())
        assert stats["forward_hits"] == hits and stats["forward_misses"] == owner.numel() - hits
        assert stats["referenced_slots"] == 0
    finally:
        if handle is not None:
            runtime.pool.release(handle=handle, discard_pending=True)
        runtime.close()


def test_prefill_failed_reader_retries_then_gathers(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完成标准 6：prefill 后台错误交给前台 ring，成功 H2D 后仍逐位一致。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    ids = torch.tensor([[[2, 4, 2]]], dtype=torch.int64)
    runtime, manager = _batch_runtime(
        catalog=catalog, device=cuda_native_device, ids=ids, monkeypatch=monkeypatch, debug=True,
    )
    error = OSError("prefill read failed")

    def fail(**kwargs: Any) -> None:
        """收到后台槽位读取参数后抛出测试错误。"""
        raise error

    try:
        gpu = ids.to(device=cuda_native_device)
        expected = runtime.lookup(layer_id=1, indices=gpu[:, 0])
        monkeypatch.setattr(runtime.prefetch_reader, "read_rows_to_slots", fail)
        scheduled, forward = _batch(tokens=1, device=cuda_native_device, decode=False)
        manager.register_request(req=scheduled.reqs[0])
        manager.start_batch(batch=scheduled, iteration=1)
        runtime.bind_batch(forward_batch=forward, hash_ids=gpu)
        with pytest.raises(RuntimeError, match="row mismatch"):
            runtime.lookup_batch(layer_id=1, indices=gpu[:, 0] + 1, forward_batch=forward)
        actual = runtime.lookup_batch(layer_id=1, indices=gpu[:, 0], forward_batch=forward)
        runtime.end_batch(batch=scheduled)
        assert manager._failure is error
        assert torch.equal(actual, expected)
    finally:
        runtime.close()


@pytest.mark.parametrize("failure", ["none", "late", "window", "foreground"])
def test_decode_gpu_wait_and_reader_failure(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device, failure: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """完成标准 6：decode 用独立 reader 不等后台预取；decode reader 失败时前台补读，前台也失败时 lookup 抛出。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    ids = torch.tensor([[[0, 5, 6, 5]], [[22, 0, 6, 7]]], dtype=torch.int64)
    runtime, manager = _batch_runtime(
        catalog=catalog, device=cuda_native_device, ids=ids, monkeypatch=monkeypatch, tp_size=8, tp_rank=2,
    )
    entered, proceed = threading.Event(), threading.Event()
    background_entered, release = threading.Event(), threading.Event()
    decode_error = OSError("decode reader failed")
    foreground = OSError("decode foreground read failed")
    read = runtime.decode_reader.read_rows_to_slots

    def read_after_submit(**kwargs: Any) -> None:
        """暂停 decode 读至计算流提交等待，再按 failure 决定读入参数给定的目标或抛错。"""
        entered.set()
        assert proceed.wait(timeout=5)
        if failure == "late":
            raise decode_error
        read(**kwargs)

    def block_background(**kwargs: Any) -> None:
        """让后台预取读一直阻塞到测试结束，验证 decode 不排在它之后。"""
        background_entered.set()
        assert release.wait(timeout=10)

    def fail_retry(**kwargs: Any) -> None:
        """收到前台槽位读参数后模拟前台 reader 也失败。"""
        raise foreground

    try:
        if runtime._hash_stream is None:
            pytest.skip("完成标准 3 需要支持 CUDA stream memory operations 的设备及驱动")
        gpu = ids.to(device=cuda_native_device)
        expected = runtime.lookup(layer_id=1, indices=gpu[:, 0])
        monkeypatch.setattr(runtime.decode_reader, "read_rows_to_slots", read_after_submit)
        if failure == "window":
            monkeypatch.setattr(runtime.prefetch_reader, "read_rows_to_slots", block_background)
            req = SimpleNamespace(origin_input_ids=[0, 1], output_ids=[], prefix_len=0)
            manager.register_request(req=req)
            scheduler = SimpleNamespace(waiting_queue=[req], chunked_req=None, max_running_requests=8)
            manager.reconcile(scheduler=scheduler)
            assert background_entered.wait(timeout=5)
        if failure == "foreground":
            # 控制器关闭后不再交给 decode 线程，由 forward 自己用前台 reader 补读。
            with manager._condition:
                manager._closing = True
            monkeypatch.setattr(runtime.reader, "read_rows_to_slots", fail_retry)
        scheduled, forward = _batch(tokens=2, device=cuda_native_device, decode=True)
        manager.start_batch(batch=scheduled, iteration=1)
        runtime.bind_batch(forward_batch=forward, hash_ids=gpu)
        current = runtime._batch[1]
        if failure == "foreground":
            assert not current.queued
            with pytest.raises(OSError, match="decode foreground read failed") as info:
                runtime.lookup_batch(layer_id=1, indices=gpu[:, 0], forward_batch=forward)
            assert info.value is foreground
            return
        assert current.queued and entered.wait(timeout=5)
        actual = runtime.lookup_batch(layer_id=1, indices=gpu[:, 0], forward_batch=forward)
        assert current.waited
        proceed.set()
        runtime.end_batch(batch=scheduled)
        assert torch.equal(actual, expected)
        assert current.storage[3].tolist() == [1]
        assert manager._failure is None  # decode reader 失败不影响后台预取
    finally:
        proceed.set()
        release.set()
        runtime.close()


def test_batch_buffers_rotate_and_wait_only_before_reuse(monkeypatch: pytest.MonkeyPatch) -> None:
    """完成标准 5/7：第三批复用第一套，登记不等待，第一次写入前等待第一批计算流事件。"""
    runtime = object.__new__(EngramSsdRuntime)
    runtime.dim, runtime.scale_dim, runtime.device = 32, 1, torch.device("cpu")
    host = [torch.zeros(74, dtype=torch.uint8) for _ in range(2)]
    runtime._buffers = [((cpu, cpu.clone(), None, None, 0), None) for cpu in host]
    runtime._retired, runtime._next_buffer = [], 0
    runtime._batch, runtime._hash_stream = None, None
    waited, recorded = [], []
    stream = object()

    def event() -> SimpleNamespace:
        """创建记录等待次数和调用流的 CPU CUDA event 替身，输出每次创建的独立事件。"""
        result = SimpleNamespace()
        result.synchronize = lambda: waited.append(result)
        result.record = lambda source: recorded.append((result, source))
        return result

    monkeypatch.setattr(torch.cuda, "Event", event)
    a, b, c = [SimpleNamespace() for _ in range(3)]
    first = runtime.reserve_batch(batch=a, queries=2, rows=1, decode=False)
    runtime.stage_prefill(batch=first, mapping=torch.tensor([0, -1], dtype=torch.int32))
    first.stream = stream
    runtime.end_batch(batch=a)
    second = runtime.reserve_batch(batch=b, queries=2, rows=1, decode=False)
    second.stream = stream
    runtime.end_batch(batch=b)
    third = runtime.reserve_batch(batch=c, queries=2, rows=1, decode=False)
    assert first.storage[0] is third.storage[0] is host[0]
    assert second.storage[0] is host[1]
    assert third.previous is first.done
    assert waited == [] and recorded == [(first.done, stream), (second.done, stream)]
    runtime.stage_prefill(batch=third, mapping=torch.tensor([-1, 0], dtype=torch.int32))
    assert waited == [first.done]
    assert host[0][:8].view(torch.int32).tolist() == [-1, 0]


def test_stream_ops_probe_warns_once(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """完成标准 3：替身返回未映射地址，两套 prefill 缓冲仍可分配，只警告一次。"""
    runtime = object.__new__(EngramSsdRuntime)
    runtime.dim, runtime.scale_dim, runtime.tp_rank, runtime.device = 32, 1, 0, torch.device("cpu")
    runtime._hash_stream = SimpleNamespace(cuda_stream=1)
    runtime._stream_ops = SimpleNamespace(stream_flag_address=lambda **kwargs: 0)
    for name in ("empty", "zeros"):
        factory = getattr(torch, name)

        def allocate(*args: Any, op=factory, pin_memory: bool = False, **kwargs: Any) -> torch.Tensor:
            """以普通 CPU 内存模拟指定的 pinned 分配，保留其他分配参数和返回形状。"""
            return op(*args, **kwargs)

        monkeypatch.setattr(torch, name, allocate)
    for index in range(2):
        host, device, ids, flags, address = runtime._allocate_batch_storage(capacity=2, probe=(index == 0))
        assert host.numel() == device.numel() == 74
        assert ids is flags is None and address == 0
    assert sum("stream memory operations unavailable" in record.getMessage() for record in caplog.records) == 1


@pytest.mark.parametrize("mode", ["mixed", "unregistered", "unsupported_decode"])
def test_no_prepared_batch_uses_sync_embedding(mode: str) -> None:
    """完成标准 3/4：混合、未登记、设备不支持 decode 流标志时沿用原同步查询。"""
    from sglang_engram_ssd.hooks.engram_embedding import forward_ssd_embedding

    runtime = object.__new__(EngramSsdRuntime)
    runtime._batch = None
    runtime.sched_prefetch_enabled = True
    expected = torch.ones((1, 1, 256), dtype=torch.bfloat16)
    calls = []

    def lookup(*, layer_id: int, indices: torch.Tensor) -> torch.Tensor:
        """捕获回退传入的层和原索引，返回固定 BF16 [1,1,256] 输出。"""
        calls.append((layer_id, indices))
        return expected

    runtime.lookup = lookup
    runtime.reduce_output = lambda out: out
    forward = SimpleNamespace(contains_mm_inputs=lambda: False, mode=mode)
    if mode == "unsupported_decode":
        runtime._buffers, runtime._hash_stream = [object()], None
        assert runtime.reserve_batch(batch=forward, queries=1, rows=1, decode=True) is None
    indices = torch.tensor([[2]], dtype=torch.int64)
    actual = forward_ssd_embedding(
        original_fn=lambda *args, **kwargs: None, self=SimpleNamespace(ssd=runtime, layer_id=1),
        indices=indices, forward_batch=forward,
    )
    assert actual is expected and calls == [(1, indices)]


def test_debug_hash_rows_reports_first_difference(cuda_native_device: torch.device) -> None:
    """完成标准 4/7：GPU int32 和 CPU int64 允许逐位相同；报告首个行号或形状不一致。"""
    ids = torch.tensor([[3, 1], [2, 5]], dtype=torch.int64)
    gpu = ids.to(device=cuda_native_device, dtype=torch.int32)
    EngramSsdRuntime.check_rows(layer_id=14, indices=gpu, expected=ids)
    with pytest.raises(RuntimeError, match=r"layer 14: row mismatch at \(1, 0\), GPU=2 CPU=3"):
        EngramSsdRuntime.check_rows(layer_id=14, indices=gpu, expected=ids + torch.tensor([[0, 0], [1, 1]]))
    with pytest.raises(RuntimeError, match=r"layer 14: GPU hash shape"):
        EngramSsdRuntime.check_rows(layer_id=14, indices=gpu[:1], expected=ids)


@pytest.mark.parametrize("enabled", [False, True])
def test_prefetch_reader_lifetime_matches_feature_switch(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    enabled: bool,
) -> None:
    """A1/A3：关闭预取不分配后台和 decode reader；关闭 runtime 后所有 reader 和借用 FD 均失效。"""
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
            assert runtime.prefetch_reader is not None and runtime.decode_reader is not None
            assert len({id(runtime.reader), id(runtime.prefetch_reader), id(runtime.decode_reader)}) == 3
            readers += [runtime.prefetch_reader, runtime.decode_reader]
        else:
            assert runtime.prefetch_reader is None and runtime.decode_reader is None and runtime.pool is None
        for fd in fds:
            os.fstat(fd)
    finally:
        runtime.close()
    assert runtime.reader is None and runtime.prefetch_reader is None and runtime.decode_reader is None
    for reader in readers:
        assert reader is not None
        with pytest.raises(RuntimeError, match="reader is closed"):
            reader.allocate_workspace(layer_id=1, capacity=1)
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)
