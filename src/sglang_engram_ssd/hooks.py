"""使用 SGLang 官方插件装饰器接入，不修改 SGLang 仓库文件。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Tuple

# 本地版本公开名称是 plugin_hook；register_hook 只是此插件中的别名。
from sglang.srt.plugins.hook_registry import (
    HookRegistry,
    HookType,
    plugin_hook as register_hook,
)

from .catalog import (
    build_engram_ssd_catalog,
    get_engram_ssd_catalog,
    install_engram_ssd_catalog,
)
from .config import get_config
from .weight_loader import engram_ssd_safetensors_weights_iterator

if TYPE_CHECKING:
    import torch
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_loader.loader import BaseModelLoader, DefaultModelLoader

logger = logging.getLogger(__name__)

_PREPARE_MODEL = "sglang.srt.model_executor.model_runner_components.load_model_utils.load_model_with_memory_saver"
_GET_LOADER = "sglang.srt.model_loader.loader.get_model_loader"
_WEIGHTS_ITERATOR = "sglang.srt.model_loader.loader.DefaultModelLoader._get_weights_iterator"
_LOAD_WEIGHTS = "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights"
_EMBED_INIT = "sglang.srt.layers.engram.EngramEmbedding.__init__"
_EMBED_FORWARD = "sglang.srt.layers.engram.EngramEmbedding.forward"
_EMBED_FINISH = "sglang.srt.layers.engram.EngramEmbedding.finish_load"
HOOK_TARGETS = (
    _PREPARE_MODEL, _GET_LOADER, _WEIGHTS_ITERATOR, _LOAD_WEIGHTS,
    _EMBED_INIT, _EMBED_FORWARD, _EMBED_FINISH,
)


def _check_hook_installation() -> None:
    # SGLang apply_hooks 会记录并继续处理失败项；模型加载前确认必要 hook 已安装，
    # 防止只安装“跳过权重”却未安装 SSD forward 等不完整组合。
    missing = [target for target in HOOK_TARGETS if target not in HookRegistry._patched]
    if missing:
        raise RuntimeError(f"Incomplete Engram SSD hook installation: {missing}")


@register_hook(_PREPARE_MODEL, type=HookType.AROUND)
def load_model_with_engram_ssd(
    original_fn: Callable[..., Any],
    *,
    model_config: ModelConfig,
    load_config: LoadConfig,
    device: str,
    gpu_id: int,
    memory_saver_adapter: Any,
    is_draft_worker: bool,
) -> Any:
    # 必须先在模型加载调用链内校验；此处异常不再经过插件注册的捕获逻辑。
    config = get_config()
    _check_hook_installation()

    import torch

    from sglang.srt.configs.load_config import LoadFormat
    from sglang.srt.environ import envs
    from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled
    from sglang.srt.models.deepseek_v4 import DeepseekV4ForCausalLM
    from sglang.srt.runtime_context import get_disagg, get_model, get_parallel, get_spec

    model = get_model()
    parallel = get_parallel()

    # 约束检查在原加载函数之前，早于 startup overlap 的整文件预取。
    if device != "cuda" or is_draft_worker or torch.version.hip is not None:
        raise ValueError("Engram SSD requires an NVIDIA CUDA target model")
    if model_config.hf_config.model_type != "deepseek_v41":
        raise ValueError("Engram SSD only supports DeepSeek V4.1")
    if not Path(model.model_path).is_dir():
        raise ValueError("Engram SSD requires an existing local checkpoint directory")
    if load_config.load_format != LoadFormat.SAFETENSORS:
        raise ValueError("Engram SSD requires --load-format safetensors")
    if model.is_startup_weight_load_overlap or model.weight_loader_prefetch_checkpoints:
        raise ValueError("Engram SSD requires serial startup without checkpoint prefetch")
    if (
        model.weight_loader_disable_mmap
        or model.weight_cache_mode != "off"
        or model.custom_weight_loader
    ):
        raise ValueError("Engram SSD requires the default selective safetensors loading path")
    if not envs.SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE.get():
        raise ValueError("Engram SSD requires SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1")
    if envs.SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT.get() != "shared":
        raise ValueError("Engram SSD requires the shared global-row configuration")
    if (
        parallel.nnodes != 1
        or parallel.pp_size != 1
        or parallel.attn_dp_size != 1
        or parallel.attn_cp_size != 1
    ):
        raise ValueError("Engram SSD currently supports single-node plain TP")
    if get_disagg().disaggregation_mode != "null":
        raise ValueError("Engram SSD currently does not support PD disaggregation")
    if get_spec().speculative_algorithm:
        raise ValueError("Engram SSD currently supports non-speculative prefill/decode")
    if not cuda_graph_fully_disabled():
        raise ValueError("Engram SSD requires both decode and prefill CUDA graphs disabled")

    # 复用原模型的名称映射；只在 catalog 阶段核对所有 shape/dtype/文件偏移。
    hf_config = model_config.hf_config
    layer_rows = dict(zip(
        hf_config.engram_layer_ids,
        hf_config.engram_num_embeddings,
        strict=True,
    ))
    name_mapper = partial(
        DeepseekV4ForCausalLM.remap_weight_name_to_dpsk_hf_format,
        is_nextn=False,
        num_hidden_layers=hf_config.num_hidden_layers,
    )
    catalog = build_engram_ssd_catalog(
        root=Path(model.model_path), name_mapper=name_mapper,
        layer_rows=layer_rows, dim=hf_config.engram_head_dim,
    )
    install_engram_ssd_catalog(catalog=catalog)
    logger.info(
        "Engram SSD enabled: TP rank=%s, layers=%s, cache_bytes=%s, chunk_rows=%s",
        parallel.tp_rank, catalog.layer_ids, config.cache_bytes, config.chunk_rows,
    )
    return original_fn(
        model_config=model_config, load_config=load_config, device=device,
        gpu_id=gpu_id, memory_saver_adapter=memory_saver_adapter,
        is_draft_worker=is_draft_worker,
    )


@register_hook(_GET_LOADER, type=HookType.AROUND)
def require_default_model_loader(
    original_fn: Callable[..., BaseModelLoader],
    load_config: LoadConfig,
    model_config: Optional[ModelConfig] = None,
) -> BaseModelLoader:
    from sglang.srt.model_loader.loader import DefaultModelLoader

    loader = original_fn(load_config=load_config, model_config=model_config)
    # safetensors 格式仍可能按 quantization 选择其他 loader，检查实际类型。
    if type(loader) is not DefaultModelLoader:
        raise ValueError("Engram SSD requires DefaultModelLoader")
    return loader


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

    from .runtime import get_engram_ssd_runtime

    # 跳过原构造函数：它会先分配完整 _HostTable，不能调用后再释放。
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
