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
    """持有官方 Hasher 参数的独立 CPU 快照，不计算 hash 或维护请求 history。

    Hasher 构造后的 Hook 创建快照，runtime／RequestHasher 持有至关闭；无需显式释放资源。
    frozen 只禁止字段替换，张量由调用方按只读约定使用，允许后台并发读取，不允许原地修改。
    """

    token_map: torch.Tensor  # 连续 CPU int64 [vocab_size]，原始 token 到压缩词表的映射。
    multipliers: torch.Tensor  # 连续 CPU int64 [L,n]，n=max_ngram_size。
    primes: torch.Tensor  # 连续 CPU int64 [L,n-1,H]，H=n_heads，C=(n-1)*H。
    offsets: torch.Tensor  # 连续 CPU int64 [L,C]，各哈希列的全局表偏移。
    pad_id: int  # 官方 hasher 已映射到压缩词表的 PAD，不再经过 token_map。
    max_ngram_size: int
    layer_ids: tuple[int, ...]  # 与模型 layout 一致的 L 个层号顺序。
    n_heads: int  # 每个 n-gram 长度使用的哈希头数 H。

    @classmethod
    def from_hasher(cls, hasher: EngramHasher, *, layer_ids: Sequence[int]) -> HashingState:
        """在官方 Hasher 构造后复制参数并校验布局。

        输入：hasher 为已初始化的官方 EngramHasher；layer_ids 为构造时 layout 的有序非空层号。
            cls 为构造快照的类，张量来源可位于 GPU，但复制结果全部落在 CPU。
        输出：独立持有连续 CPU int64 张量的 HashingState；不持有原模型，非法形状抛 ValueError。
        """
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
    """用只读 CPU 参数计算请求行号，不持有 tokenizer、模型或请求可变 history。

    runtime 通过 Hasher Hook 创建并持有至关闭；state 为独立快照，没有后台资源需要 close。
    每次计算仅创建局部张量，多个调用可并发，前提是调用方不修改 state 内的张量。
    """

    state: HashingState

    @classmethod
    def from_hasher(cls, hasher: EngramHasher, *, layer_ids: Sequence[int]) -> RequestHasher:
        """创建与模型共用同一组数值参数的 CPU 计算器。

        输入：hasher 为已初始化官方 Hasher，layer_ids 为其 layout 层顺序，cls 为构造计算器的类。
        输出：持有独立 HashingState 的 RequestHasher，不复制模型或重新构造 tokenizer。
        """
        return cls(state=HashingState.from_hasher(hasher=hasher, layer_ids=layer_ids))

    @property
    def layer_ids(self) -> tuple[int, ...]:
        """读取哈希输出的层顺序。

        输入：无。
        输出：不可变层号元组，对应计算结果的 L 维，不复制参数张量。
        """
        return self.state.layer_ids

    def compute_request_row_ids(
        self, token_ids: Sequence[int] | torch.Tensor, start: int, end: int
    ) -> torch.Tensor:
        """计算完整请求中 [start,end) 段每个 token 的 Engram 全局行号。

        输入：token_ids 为完整请求的原始 token 序列，可重复；可传 CPU int32／int64 [T] 张量，
            或整数序列，值须属于 [0,vocab_size)，张量不要求 contiguous。
            start／end 为绝对位置，满足 0<=start<=end<=T，不是分段后局部位置。
        输出：CPU int64 [end-start,L,C]，C=(max_ngram_size-1)*n_heads；空段保留 L／C 维。
        只截取当前段和最多 n-1 个前驱；负的前驱位置通过 blocked 交给官方公式填 pad_id。
        不修改原始请求或 history；非法区间／token 抛 IndexError，非法张量类型／设备抛 ValueError。
        """
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
        # 绝对位置与前驱距离广播为 [end-start,n]；负位置虽暂取第一个 token，随后由 blocked 屏蔽。
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
