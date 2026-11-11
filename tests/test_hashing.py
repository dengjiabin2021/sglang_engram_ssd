"""使用合成参数验证请求 CPU 行号与官方 Torch、CUDA 路径逐位一致。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from sglang_engram_ssd.hashing import HashingState, RequestHasher

if TYPE_CHECKING:
    from sglang.srt.layers.engram import EngramHasher


_LAYER_IDS = (1, 14)


def _make_hasher(max_ngram_size: int = 4) -> EngramHasher:
    from sglang.srt.layers.engram import EngramHasher, compute_hash_multipliers

    # 跳过会加载 tokenizer 的 __init__，仅为官方参考方法提供合成缓冲。
    hasher = EngramHasher.__new__(EngramHasher)
    torch.nn.Module.__init__(hasher)
    hasher.max_ngram_size = max_ngram_size
    token_map = (torch.arange(257, dtype=torch.int64) * 37 + 11) % 97
    hasher.pad_id = int(token_map[3])
    hasher.image_token_id = None
    primes = torch.tensor(
        [[[17, 19], [23, 29], [31, 37]], [[41, 43], [47, 53], [59, 61]]], dtype=torch.int64
    )[:, : max_ngram_size - 1].contiguous()
    flat_primes = primes.flatten(1)
    offsets = torch.cat([torch.zeros(2, 1, dtype=torch.int64), flat_primes[:, :-1].cumsum(dim=1)], dim=1)
    hasher.register_buffer("token_map", token_map, persistent=False)
    hasher.register_buffer(
        "multipliers",
        compute_hash_multipliers(layer_ids=_LAYER_IDS, max_ngram_size=max_ngram_size, vocab_size=97),
        persistent=False,
    )
    hasher.register_buffer("primes", primes, persistent=False)
    hasher.register_buffer("offsets", offsets, persistent=False)
    return hasher


def _segment_history(token_ids: torch.Tensor, start: int, n: int) -> torch.Tensor:
    history = torch.zeros(1, n - 1, dtype=torch.int64, device=token_ids.device)
    previous = token_ids[max(0, start - n + 1) : start]
    if previous.numel():
        history[0, -previous.numel() :] = previous
    return history


def _torch_reference(hasher: EngramHasher, token_ids: torch.Tensor, start: int, end: int) -> torch.Tensor:
    from sglang.kernels.ops.embeddings.engram_hash import MODE_EXTEND

    values, _ = hasher._torch_hash_ids(
        input_ids=token_ids[start:end],
        positions=torch.arange(start, end, dtype=torch.int64),
        kmode=MODE_EXTEND,
        history=_segment_history(token_ids=token_ids, start=start, n=hasher.max_ngram_size),
        num_real=end - start,
        block=1,
        row=torch.zeros(end - start, dtype=torch.int64),
        starts=torch.zeros(1, dtype=torch.int64),
    )
    return values


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize(
    "length,start,end",
    [(1, 0, 1), (2, 0, 2), (3, 1, 3), (7, 0, 4), (7, 1, 2), (17, 3, 17), (129, 63, 79), (513, 256, 513)],
)
def test_request_segments_match_official_torch(n: int, length: int, start: int, end: int) -> None:
    """A2：随机长度、非零起点与 blocked 前驱均与官方整段及分段 Torch 参考一致。"""
    hasher = _make_hasher(max_ngram_size=n)
    calculator = RequestHasher.from_hasher(hasher=hasher, layer_ids=_LAYER_IDS)
    generator = torch.Generator().manual_seed(42 + length + n)
    token_ids = torch.randint(0, 257, (length,), generator=generator, dtype=torch.int64)
    actual = calculator.compute_request_row_ids(token_ids=token_ids, start=start, end=end)
    segment_reference = _torch_reference(hasher=hasher, token_ids=token_ids, start=start, end=end)
    full_reference = _torch_reference(hasher=hasher, token_ids=token_ids, start=0, end=length)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.int64
    assert actual.shape == (end - start, 2, (n - 1) * 2)
    assert torch.equal(actual, segment_reference)
    assert torch.equal(actual, full_reference[start:end])


def test_request_segments_keep_preceding_tokens() -> None:
    """A2：分段保留段前真实 token，连续分段拼接与一次计算完全一致。"""
    calculator = RequestHasher.from_hasher(hasher=_make_hasher(), layer_ids=_LAYER_IDS)
    token_ids = torch.randint(0, 257, (131,), generator=torch.Generator().manual_seed(17), dtype=torch.int32)
    full = calculator.compute_request_row_ids(token_ids=token_ids, start=0, end=len(token_ids))
    boundaries = [0, 1, 2, 7, 31, 64, 131]
    parts = [
        calculator.compute_request_row_ids(token_ids=token_ids.tolist(), start=start, end=end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    assert torch.equal(full, torch.cat(parts))


def test_hasher_state_is_an_owned_cpu_snapshot() -> None:
    """A2：复制官方参数且保留 layout 层序，模型缓冲后续变化不污染 CPU 快照。"""
    hasher = _make_hasher()
    state = HashingState.from_hasher(hasher=hasher, layer_ids=_LAYER_IDS)
    assert state.layer_ids == _LAYER_IDS
    assert state.pad_id == hasher.pad_id
    assert state.max_ngram_size == 4
    assert state.n_heads == 2
    for name in ("token_map", "multipliers", "primes", "offsets"):
        original = getattr(hasher, name)
        snapshot = getattr(state, name)
        assert snapshot.device.type == "cpu"
        assert snapshot.dtype == torch.int64
        assert snapshot.is_contiguous()
        assert torch.equal(snapshot, original)
        original.add_(1)
        assert not torch.equal(snapshot, original)


@pytest.mark.parametrize("start,end", [(0, 0), (2, 2)])
def test_empty_request_segment(start: int, end: int) -> None:
    """A2：空序列和非零起点空段返回带完整层／列维度的 CPU int64 空张量。"""
    calculator = RequestHasher.from_hasher(hasher=_make_hasher(), layer_ids=_LAYER_IDS)
    values = calculator.compute_request_row_ids(token_ids=[7] * end, start=start, end=end)
    assert values.shape == (0, 2, 6)
    assert values.dtype == torch.int64
    assert values.device.type == "cpu"


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (0, 4)])
def test_invalid_request_segment(start: int, end: int) -> None:
    """A2：非法段边界明确报错，避免误用段内位置替代请求全局位置。"""
    calculator = RequestHasher.from_hasher(hasher=_make_hasher(), layer_ids=_LAYER_IDS)
    with pytest.raises(IndexError, match="request range"):
        calculator.compute_request_row_ids(token_ids=[1, 2, 3], start=start, end=end)


@pytest.mark.skipif(
    torch.version.cuda is None or not torch.cuda.is_available(),
    reason="A2 CUDA comparison requires an NVIDIA CUDA device and the official Triton kernels",
)
@pytest.mark.parametrize("start,end", [(0, 1), (1, 9), (7, 37)])
def test_request_segments_match_official_cuda(start: int, end: int) -> None:
    """A2：真实 CUDA hash kernel 与 CPU 请求计算器逐位一致，不依赖 checkpoint。"""
    from sglang.kernels.ops.embeddings.engram_hash import MODE_EXTEND, engram_hash_ids

    hasher = _make_hasher().to(device="cuda")
    calculator = RequestHasher.from_hasher(hasher=hasher, layer_ids=_LAYER_IDS)
    token_ids = torch.randint(0, 257, (73,), generator=torch.Generator().manual_seed(19), dtype=torch.int64)
    actual = calculator.compute_request_row_ids(token_ids=token_ids, start=start, end=end)
    gpu_ids = token_ids.to(device="cuda")
    expected, _ = engram_hash_ids(
        input_ids=gpu_ids[start:end],
        positions=torch.arange(start, end, dtype=torch.int64, device="cuda"),
        mode=MODE_EXTEND,
        history=_segment_history(token_ids=gpu_ids, start=start, n=hasher.max_ngram_size),
        token_map=hasher.token_map,
        multipliers=hasher.multipliers,
        primes=hasher.primes,
        offsets=hasher.offsets,
        pad_id=hasher.pad_id,
        num_real=end - start,
        row=torch.zeros(end - start, dtype=torch.int64, device="cuda"),
        starts=torch.zeros(1, dtype=torch.int64, device="cuda"),
    )
    assert torch.equal(actual, expected.cpu())
