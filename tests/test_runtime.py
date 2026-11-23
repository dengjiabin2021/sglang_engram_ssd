"""用合成文件及串行 Torch 参考验证精简后的 owner 查询语义。"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from sglang_engram_ssd.catalog import EngramSsdCatalog
from sglang_engram_ssd.runtime import EngramSsdRuntime


pytestmark = pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(),
    reason="A0 requires an NVIDIA CUDA device and the compiled io_uring extension",
)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (8, 2)])
def test_dedup_restores_positions_and_reduces_reads(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    tp_size: int,
    tp_rank: int,
) -> None:
    """A5：整次查询去重覆盖重复、跨chunk、owner过滤，恢复后逐位一致。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=tp_rank, tp_size=tp_size,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
    )
    counts: list[int] = []
    assert runtime.reader is not None
    read_rows = runtime.reader.read_rows

    def record_rows(**kwargs) -> None:
        counts.append(kwargs["row_ids"].numel())
        read_rows(**kwargs)

    runtime.reader.read_rows = record_rows
    indices = torch.tensor([[5, 6, 5, 2], [20, 5, 6, 2]], dtype=torch.int64, device=cuda_native_device)
    try:
        baseline = runtime.lookup(layer_id=1, indices=indices)
        baseline_count = sum(counts)
        counts.clear()
        runtime.dedup_enabled = True
        actual = runtime.lookup(layer_id=1, indices=indices)
        assert torch.equal(actual, baseline)
        owner = indices.reshape(-1)
        owner = owner[(owner >= 23 * tp_rank // tp_size) & (owner < 23 * (tp_rank + 1) // tp_size)]
        assert baseline_count == owner.numel()
        assert sum(counts) == owner.unique().numel()
        assert sum(counts) < baseline_count
    finally:
        runtime.close()


def _serial_owner_reference(
    *, weights: torch.Tensor, scales: torch.Tensor, indices: torch.Tensor, tp_rank: int, tp_size: int
) -> torch.Tensor:
    # 合成 scale 全为 127；单位缩放下，原 Torch 反量化就是 FP8 → FP32 → BF16。
    assert bool((scales == 127).all())
    rows, dim = weights.shape
    flat_ids = indices.to(device="cpu", dtype=torch.int64).reshape(-1)
    values = weights.index_select(0, flat_ids).view(torch.float8_e4m3fn).float().to(torch.bfloat16)
    row_start = rows * tp_rank // tp_size
    row_end = rows * (tp_rank + 1) // tp_size
    owned = (flat_ids >= row_start) & (flat_ids < row_end)
    return values.masked_fill(~owned.unsqueeze(-1), 0).view(*indices.shape, dim)


@pytest.mark.parametrize("rows,tp_size", [(23, 1), (23, 8), (3, 8)])
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_serial_owner_lookup_matches_reference(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    rows: int,
    tp_size: int,
    index_dtype: torch.dtype,
) -> None:
    """A0：TP1及模拟TP8覆盖非整除行区间、空owner、重复查询与跨块尾部。"""
    catalog, weights, scales = synthetic_catalog(rows=rows)
    flat = torch.cat([
        torch.arange(rows, dtype=torch.int64, device="cpu"),
        torch.tensor([0, rows - 1, 0, rows // 2, rows - 1], dtype=torch.int64, device="cpu"),
    ])
    cpu_indices = flat.view(2, -1)
    indices = cpu_indices.to(device=cuda_native_device, dtype=index_dtype)
    partials = []
    for rank in range(tp_size):
        runtime = EngramSsdRuntime(
            catalog=catalog, dim=256, tp_rank=rank, tp_size=tp_size,
            chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
        )
        try:
            actual = runtime.lookup(layer_id=1, indices=indices)
            expected = _serial_owner_reference(
                weights=weights, scales=scales, indices=cpu_indices, tp_rank=rank, tp_size=tp_size,
            )
            assert actual.device == cuda_native_device
            assert actual.dtype == torch.bfloat16
            assert actual.shape == (*indices.shape, 256)
            assert torch.equal(actual.cpu(), expected)
            partials.append(actual.cpu())
            if tp_size == 1:
                assert runtime.reduce_output(out=actual) is actual
        finally:
            runtime.close()

    # 在一张 GPU 顺序模拟 owner 求和，不替代真实 torchrun 的 TP8 集合通信验证。
    combined = torch.stack(partials).float().sum(dim=0).to(torch.bfloat16)
    reference = _serial_owner_reference(
        weights=weights, scales=scales, indices=cpu_indices, tp_rank=0, tp_size=1,
    )
    assert torch.equal(combined, reference)


def test_lookup_with_no_owned_queries(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
) -> None:
    """A0：非空 owner 行区间没有命中时，返回完整形状的零张量。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=7, tp_size=8,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
    )
    try:
        indices = torch.tensor([[0, 0, 1]], dtype=torch.int64, device=cuda_native_device)
        actual = runtime.lookup(layer_id=1, indices=indices)
        assert actual.shape == (1, 3, 256)
        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, torch.zeros_like(actual))
    finally:
        runtime.close()


@pytest.mark.parametrize("row_ids", [[5, 6, 5], [0, 0, 1]])
def test_owner_lookup_does_not_add_current_stream_wait(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
    row_ids: list[int],
) -> None:
    """A0/A4：owner 结果恢复和空 owner 路径不额外等待当前流；中转区仍由 event 保护。"""
    catalog, weights, scales = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=2, tp_size=8,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
    )

    def reject_stream_wait(self: torch.cuda.Stream) -> None:
        raise AssertionError("Owner output restoration must not synchronize the current stream")

    try:
        indices = torch.tensor([row_ids], dtype=torch.int64, device=cuda_native_device)
        # 只拦截显式流同步；Event.synchronize 继续保证 pinned/GPU 工作区可安全复用。
        with monkeypatch.context() as patch:
            patch.setattr(torch.cuda.Stream, "synchronize", reject_stream_wait)
            actual = runtime.lookup(layer_id=1, indices=indices)
        expected = _serial_owner_reference(
            weights=weights, scales=scales, indices=indices,
            tp_rank=2, tp_size=8,
        )
        assert torch.equal(actual.cpu(), expected)
    finally:
        runtime.close()


@pytest.mark.parametrize("tp_size", [1, 8])
def test_lookup_with_empty_queries(
    synthetic_catalog: Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]],
    cuda_native_device: torch.device,
    tp_size: int,
) -> None:
    """A0：空查询保留前导维度，且 TP8 的空输出无需发起归约。"""
    catalog, _, _ = synthetic_catalog(rows=23)
    runtime = EngramSsdRuntime(
        catalog=catalog, dim=256, tp_rank=0, tp_size=tp_size,
        chunk_rows=3, io_uring_queue_depth=4, device=cuda_native_device,
    )
    try:
        indices = torch.empty((0, 24), dtype=torch.int64, device=cuda_native_device)
        actual = runtime.lookup(layer_id=1, indices=indices)
        assert actual.shape == (0, 24, 256)
        assert actual.dtype == torch.bfloat16
        assert actual.device == cuda_native_device
        assert runtime.reduce_output(out=actual) is actual
    finally:
        runtime.close()
