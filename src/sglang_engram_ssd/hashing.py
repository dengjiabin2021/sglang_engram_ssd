"""复制官方哈希参数，在 CPU 上计算完整请求任意 token 段的 Engram 行号。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.engram import EngramHasher


@dataclass(frozen=True)
class HashingState:
    """由模型 hasher 复制的只读参数；张量独立持有，均为 CPU int64。"""

    token_map: torch.Tensor  # [vocab_size]
    multipliers: torch.Tensor  # [L, n]
    primes: torch.Tensor  # [L, n - 1, H]
    offsets: torch.Tensor  # [L, (n - 1) * H]
    pad_id: int  # 官方 hasher 已映射到压缩词表的 PAD，不再经过 token_map。
    max_ngram_size: int
    layer_ids: tuple[int, ...]
    n_heads: int

    @classmethod
    def from_hasher(cls, hasher: EngramHasher, *, layer_ids: Sequence[int]) -> HashingState:
        """在官方 hasher 构造后复制参数；层号必须使用构造时的 layout 顺序。"""
        layers = tuple(int(layer_id) for layer_id in layer_ids)
        n = int(hasher.max_ngram_size)
        token_map = hasher.token_map.detach().to(device="cpu", dtype=torch.int64, copy=True).contiguous()
        multipliers = hasher.multipliers.detach().to(device="cpu", dtype=torch.int64, copy=True).contiguous()
        primes = hasher.primes.detach().to(device="cpu", dtype=torch.int64, copy=True).contiguous()
        offsets = hasher.offsets.detach().to(device="cpu", dtype=torch.int64, copy=True).contiguous()
        if n < 2 or not layers:
            raise ValueError(f"Engram hashing requires n >= 2 and nonempty layers, got n={n}, layers={layers}")
        if token_map.ndim != 1 or token_map.numel() == 0:
            raise ValueError("Engram token_map must be a nonempty CPU vector")
        if primes.ndim != 3 or tuple(primes.shape[:2]) != (len(layers), n - 1):
            raise ValueError(f"Engram primes shape {tuple(primes.shape)} does not match n={n}, layers={layers}")
        heads = int(primes.shape[-1])
        if heads <= 0 or tuple(multipliers.shape) != (len(layers), n):
            raise ValueError(
                f"Engram multipliers shape {tuple(multipliers.shape)} does not match n={n}, layers={layers}"
            )
        if tuple(offsets.shape) != (len(layers), (n - 1) * heads):
            raise ValueError(f"Engram offsets shape {tuple(offsets.shape)} does not match n={n}, heads={heads}")
        return cls(
            token_map=token_map,
            multipliers=multipliers,
            primes=primes,
            offsets=offsets,
            pad_id=int(hasher.pad_id),
            max_ngram_size=n,
            layer_ids=layers,
            n_heads=heads,
        )


@dataclass(frozen=True)
class RequestHasher:
    """只读 CPU 参数的请求计算器，不持有 tokenizer、模型或请求可变状态。"""

    state: HashingState

    @classmethod
    def from_hasher(cls, hasher: EngramHasher, *, layer_ids: Sequence[int]) -> RequestHasher:
        """取得与模型完全相同的词表映射及哈希参数，不构造第二份 tokenizer。"""
        return cls(state=HashingState.from_hasher(hasher=hasher, layer_ids=layer_ids))

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return self.state.layer_ids

    def compute_request_row_ids(
        self, token_ids: Sequence[int] | torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        """完整请求序列的 [start,end) → CPU int64 [end-start,L,C]，保留段前历史。"""
        if isinstance(token_ids, torch.Tensor):
            if token_ids.device.type != "cpu" or token_ids.ndim != 1:
                raise ValueError("Engram request token_ids must be a one-dimensional CPU tensor")
            if token_ids.dtype not in (torch.int32, torch.int64):
                raise ValueError("Engram request token_ids must have int32 or int64 dtype")
        length = len(token_ids)
        if not 0 <= start <= end <= length:
            raise IndexError(f"Engram request range [{start},{end}) outside token sequence length {length}")
        state = self.state
        if start == end:
            return torch.empty(
                (0, len(state.layer_ids), (state.max_ngram_size - 1) * state.n_heads),
                dtype=torch.int64,
                device="cpu",
            )

        # 只复制当前段及最多 n-1 个前驱，避免每段都重新复制整个长 prompt。
        context_start = max(0, start - state.max_ngram_size + 1)
        context = torch.as_tensor(token_ids[context_start:end], dtype=torch.int64, device="cpu")
        if int(context.min()) < 0 or int(context.max()) >= state.token_map.numel():
            raise IndexError(f"Engram request token ID outside token_map size {state.token_map.numel()}")
        positions = torch.arange(start, end, dtype=torch.int64, device="cpu")
        shifts = torch.arange(state.max_ngram_size, dtype=torch.int64, device="cpu")
        blocked = positions.unsqueeze(-1) < shifts
        source = positions.unsqueeze(-1) - shifts - context_start
        tokens = context[source.clamp_min(0)]

        # 复用官方 CPU 公式，保留 int64 乘法、逐级 XOR 与取模的原始语义。
        from sglang.srt.layers.engram import compute_engram_hash_ids

        return compute_engram_hash_ids(
            tokens=tokens,
            blocked=blocked,
            pad_id=state.pad_id,
            token_map=state.token_map,
            multipliers=state.multipliers,
            primes=state.primes,
            offsets=state.offsets,
        )
