"""集中管理 EngramEmbedding 的构造、推理和加载完成替换。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.plugins.hook_registry import (
    HookType,
    plugin_hook as register_hook,
)

from ._registry import _EMBED_FINISH, _EMBED_FORWARD, _EMBED_INIT

if TYPE_CHECKING:
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@register_hook(_EMBED_INIT, type=HookType.AROUND)
def init_ssd_embedding(
    original_fn: Callable[..., Any],
    self: Any,
    num_embeddings: int,
    dim: int,
    layer_id: int,
) -> None:
    from torch import nn

    from ..runtime import get_engram_ssd_runtime

    # 跳过原构造函数，避免按原配置分配 DDR 表或 HBM 分片。
    nn.Module.__init__(self)
    self.dim = dim
    self.layer_id = layer_id
    # 同进程各层 dim 相同，缓存 factory 返回同一 runtime；构造时统一 init_table。
    self.ssd = get_engram_ssd_runtime(dim=dim)
    self.tp_size = self.ssd.tp_size
    if self.ssd.lookup_mode == "owner_allreduce":
        # 按全局行号划分 owner；这里只记录范围，不分配完整 DDR/HBM 表。
        self.row_start = num_embeddings * self.ssd.tp_rank // self.tp_size
        row_end = num_embeddings * (self.ssd.tp_rank + 1) // self.tp_size
        self.rows = row_end - self.row_start
    else:
        # all_queries 在任意 TP 规模下都处理全部查询。
        self.row_start = 0
        self.rows = num_embeddings
    self.host_table = None
    self.register_parameter("weight", None)
    self.register_parameter("scale", None)


@register_hook(_EMBED_FORWARD, type=HookType.AROUND)
def forward_ssd_embedding(
    original_fn: Callable[..., torch.Tensor],
    self: Any,
    indices: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    *,
    cp_all_tokens: bool = False,
) -> torch.Tensor:
    runtime = getattr(self, "ssd", None)
    if runtime is None:
        return original_fn(self, indices, forward_batch, cp_all_tokens=cp_all_tokens)
    if cp_all_tokens:
        raise ValueError("Engram SSD does not support context-parallel lookup")
    if forward_batch is not None and forward_batch.contains_mm_inputs():
        raise ValueError("Engram SSD currently supports text-only requests")
    context = None
    if runtime.prefetch_executor is not None:
        from ..prefetch import get_current_prefetch

        context = get_current_prefetch()
    if context is not None:
        # 后台预取只做本地查表，未选中层仍由 context 调用现有 SSD lookup。
        out = context.lookup(layer_id=self.layer_id, indices=indices)
    else:
        out = runtime.lookup(layer_id=self.layer_id, indices=indices)
    # owner 模式在模型原 Engram 位置每层合并一次，保持与主干 TP 通信的执行顺序。
    # all_queries 和 TP=1 在 reduce_output 内直接返回，不额外通信。
    return runtime.reduce_output(out)


@register_hook(_EMBED_FINISH, type=HookType.AROUND)
def finish_ssd_embedding_load(
    original_fn: Callable[..., Any], self: Any, label: str
) -> Any:
    if getattr(self, "ssd", None) is not None:
        return None  # 没有完整 DDR backing，无需执行其写入完成 barrier。
    return original_fn(self, label)
