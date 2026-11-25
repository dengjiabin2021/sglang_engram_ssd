"""SSD 关闭时单独登记的计时 Hook：给原生 DDR／HBM 的 EngramEmbedding.forward 计时。

不放在 hooks 包内：导入 hooks 包会登记全部 SSD 替换。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.plugins.hook_registry import (
    HookType,
    plugin_hook as register_hook,
)

from .timing import get_timer

if TYPE_CHECKING:
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# sglang/python/sglang/srt/layers/engram.py：EngramEmbedding.forward(
#     self, indices: torch.Tensor, forward_batch: Optional[ForwardBatch] = None,
#     *, cp_all_tokens: bool = False) -> torch.Tensor
@register_hook("sglang.srt.layers.engram.EngramEmbedding.forward", type=HookType.AROUND)
def time_native_embedding(
    original_fn: Callable[..., torch.Tensor],
    self: Any,
    indices: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    *,
    cp_all_tokens: bool = False,
) -> torch.Tensor:
    """原样调用官方 forward（per_rank 布局含 owner gather 和 TP all-reduce），prefill 时记录 GPU 耗时。"""
    return get_timer().run(
        forward_batch=forward_batch,
        fn=lambda: original_fn(self, indices, forward_batch, cp_all_tokens=cp_all_tokens),
    )
