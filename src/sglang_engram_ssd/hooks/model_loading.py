"""模型加载阶段：校验启动配置、准备 catalog 并限定模型加载器。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

# 官方装饰器名为 plugin_hook，register_hook 沿用插件中的调用名称。
from sglang.srt.plugins.hook_registry import (
    HookType,
    plugin_hook as register_hook,
)

from ..catalog import build_engram_ssd_catalog, install_engram_ssd_catalog
from ..config import get_config
from ._registry import _GET_LOADER, _PREPARE_MODEL, _check_hook_installation

if TYPE_CHECKING:
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.model_loader.loader import BaseModelLoader

logger = logging.getLogger(__name__)


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

    # 本 Hook 由独立的 SSD 开关启用，不读取原 HOST_TABLE 开关或 DDR 布局。

    import torch

    from sglang.srt.configs.load_config import LoadFormat
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
        "Engram SSD enabled: TP rank=%s, layers=%s, lookup_mode=%s, "
        "query_backend=%s, num_threads=%s, thread_pool_queue_depth=%s, "
        "io_uring_queue_depth=%s, cache_bytes=%s, chunk_rows=%s",
        parallel.tp_rank, catalog.layer_ids, config.lookup_mode, config.query_backend,
        config.num_threads, config.thread_pool_queue_depth, config.io_uring_queue_depth,
        config.cache_bytes, config.chunk_rows,
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
