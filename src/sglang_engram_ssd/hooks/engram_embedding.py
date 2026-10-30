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
    from sglang.srt.runtime_context import get_parallel

    from ..runtime import get_engram_ssd_runtime

    # 跳过原构造函数，避免按原配置分配 DDR 表或 HBM 分片。
    nn.Module.__init__(self)
    self.dim = dim
    self.layer_id = layer_id
    parallel = get_parallel()
    self.tp_size = parallel.tp_size
    self.row_start = num_embeddings * parallel.tp_rank // self.tp_size
    row_end = num_embeddings * (parallel.tp_rank + 1) // self.tp_size
    self.rows = row_end - self.row_start
    self.host_table = None
    # 同进程各层 dim 相同，缓存 factory 返回同一 runtime；构造时统一 init_table。
    self.ssd = get_engram_ssd_runtime(dim=dim)
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
    return runtime.lookup(layer_id=self.layer_id, indices=indices)


@register_hook(_EMBED_FINISH, type=HookType.AROUND)
def finish_ssd_embedding_load(
    original_fn: Callable[..., Any], self: Any, label: str
) -> Any:
    if getattr(self, "ssd", None) is not None:
        return None  # 没有完整 DDR backing，无需执行其写入完成 barrier。
    return original_fn(self, label)
