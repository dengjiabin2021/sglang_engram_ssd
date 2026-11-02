"""在原 hash IDs 生成后提交查表任务，不重复计算 hash 或更新 history。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt.plugins.hook_registry import HookType, plugin_hook as register_hook

from ..prefetch import get_current_prefetch
from ._registry import _HASH_FORWARD

if TYPE_CHECKING:
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


@register_hook(_HASH_FORWARD, type=HookType.AFTER)
def submit_engram_prefetch(
    hash_ids: torch.Tensor,
    self: Any,
    input_ids: torch.Tensor,
    forward_batch: ForwardBatch,
) -> None:
    context = get_current_prefetch()
    if context is not None:
        # 只提交后台任务；模型线程继续执行前面层的 Attention/MoE。
        context.submit(hash_ids)
    # 返回 None，保留原 Hasher 的返回值。
