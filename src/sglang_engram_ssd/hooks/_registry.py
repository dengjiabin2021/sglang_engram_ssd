"""集中维护 Hook 目标及安装完整性检查。"""

from sglang.srt.plugins.hook_registry import HookRegistry

# 本模块不导入具体 Hook，避免与各替换模块形成循环依赖。
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
