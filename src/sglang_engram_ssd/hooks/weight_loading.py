"""权重加载阶段：过滤由 SSD runtime 管理的 Engram 大表参数。"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

from sglang.srt.plugins.hook_registry import (
    HookType,
    plugin_hook as register_hook,
)

from ..catalog import get_engram_ssd_catalog
from ..weight_loader import engram_ssd_safetensors_weights_iterator
from ._registry import _LOAD_WEIGHTS, _WEIGHTS_ITERATOR

if TYPE_CHECKING:
    import torch
    from sglang.srt.model_loader.loader import DefaultModelLoader


@register_hook(_WEIGHTS_ITERATOR, type=HookType.AROUND)
def selective_weights_iterator(
    original_fn: Callable[..., Any],
    self: DefaultModelLoader,
    source: DefaultModelLoader.Source,
    *,
    resolved_source: Optional[DefaultModelLoader.ResolvedSource] = None,
    startup_prefetch_started: bool = False,
    startup_prefetch_active: bool = False,
) -> Any:
    from sglang.srt.model_loader.weight_utils import maybe_add_mtp_safetensors
    from sglang.srt.runtime_context import get_model

    if source.prefix or self.load_config.draft_model_idx is not None:
        raise ValueError("Engram SSD only supports the primary target weight source")
    if startup_prefetch_started or startup_prefetch_active:
        raise ValueError("Engram SSD does not support startup checkpoint prefetch")

    # 沿用原文件发现流程，随后直接选择专用 iterator，不能包装已产出 Tensor 的
    # buffered iterator 再过滤，否则 get_tensor 已经发生。
    if resolved_source is None:
        hf_folder, hf_weights_files, use_safetensors = self._prepare_weights(
            source.model_or_path, source.revision,
            source.fall_back_to_pt, source.allow_patterns_overrides,
        )
        if use_safetensors and source.model_config is not None:
            hf_weights_files = maybe_add_mtp_safetensors(
                hf_weights_files, hf_folder, "model.safetensors.index.json",
                source.model_config.hf_config,
            )
    else:
        hf_folder = resolved_source.hf_folder
        hf_weights_files = list(resolved_source.weight_files)
        use_safetensors = resolved_source.use_safetensors

    catalog = get_engram_ssd_catalog()
    if not use_safetensors or Path(hf_folder).resolve() != catalog.root:
        raise ValueError("Engram SSD iterator must use the catalog's original safetensors checkpoint")
    weights_iterator = engram_ssd_safetensors_weights_iterator(
        hf_weights_files=hf_weights_files,
        skip_names=catalog.raw_names,
        drop_cache_after_load=get_model().weight_loader_drop_cache_after_load,
    )
    if self.counter_before_loading_weights == 0.0:
        self.counter_before_loading_weights = time.perf_counter()
    return ((source.prefix + name, tensor) for name, tensor in weights_iterator)


@register_hook(_LOAD_WEIGHTS, type=HookType.AROUND)
def load_weights_with_external_engram(
    original_fn: Callable[..., Any],
    self: Any,
    weights: Iterable[Tuple[str, torch.Tensor]],
    is_nextn=False,
) -> Any:
    if is_nextn:
        raise ValueError("Engram SSD does not support NextN loading")
    external_names = get_engram_ssd_catalog().canonical_names

    def ordinary_weights() -> Iterable[Tuple[str, torch.Tensor]]:
        # 正常路径已在 get_tensor 前过滤；这里补充规范化名称的外置参数识别。
        for name, tensor in weights:
            canonical = self.remap_weight_name_to_dpsk_hf_format(
                name, is_nextn=is_nextn, num_hidden_layers=self.config.num_hidden_layers
            )
            if canonical not in external_names:
                yield name, tensor

    return original_fn(self, weights=ordinary_weights(), is_nextn=is_nextn)
