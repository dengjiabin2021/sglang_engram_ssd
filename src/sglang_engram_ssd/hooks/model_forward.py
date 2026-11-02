"""为一次模型 forward 保存按层预取任务，并在退出时收齐后台工作。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from sglang.srt.plugins.hook_registry import HookType, plugin_hook as register_hook

from ._registry import _MODEL_FORWARD

if TYPE_CHECKING:
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors


@register_hook(_MODEL_FORWARD, type=HookType.AROUND)
def forward_with_engram_prefetch(
    original_fn: Callable[..., Any],
    self: Any,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor | None,
    pp_proxy_tensors: PPProxyTensors | None = None,
) -> torch.Tensor | PPProxyTensors:
    from ..prefetch import EngramPrefetchContext, reset_current_prefetch, set_current_prefetch
    from ..runtime import get_engram_ssd_runtime

    runtime = get_engram_ssd_runtime(dim=self.engram_layout.head_dim)
    # 每轮 forward 创建独立任务字典，查表工作区仍由 runtime.lock 保护。
    context = EngramPrefetchContext(runtime, layer_ids=tuple(self.engram_layout.layer_ids))
    token = set_current_prefetch(context)
    try:
        try:
            result = original_fn(
                self,
                input_ids=input_ids,
                positions=positions,
                forward_batch=forward_batch,
                input_embeds=input_embeds,
                pp_proxy_tensors=pp_proxy_tensors,
            )
        except BaseException:
            # 模型已有异常时仍等待后台退出，保留原模型异常。
            context.finish(raise_error=False)
            raise
        else:
            context.finish()
            return result
    finally:
        reset_current_prefetch(token)
