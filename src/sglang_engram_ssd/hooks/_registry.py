"""集中维护 Hook 目标及安装完整性检查。"""

from __future__ import annotations

from sglang.srt.plugins.hook_registry import HookRegistry

# 本模块不导入具体 Hook，避免与各替换模块形成循环依赖。
_PREPARE_MODEL = (
    "sglang.srt.model_executor.model_runner_components.load_model_utils.load_model_with_memory_saver"
)
_GET_LOADER = "sglang.srt.model_loader.loader.get_model_loader"
_WEIGHTS_ITERATOR = "sglang.srt.model_loader.loader.DefaultModelLoader._get_weights_iterator"
_LOAD_WEIGHTS = "sglang.srt.models.deepseek_v4.DeepseekV4ForCausalLM.load_weights"
_EMBED_INIT = "sglang.srt.layers.engram.EngramEmbedding.__init__"
_EMBED_FORWARD = "sglang.srt.layers.engram.EngramEmbedding.forward"
_EMBED_FINISH = "sglang.srt.layers.engram.EngramEmbedding.finish_load"
_HASH_INIT = "sglang.srt.layers.engram.EngramHasher.__init__"
_ADD_TO_QUEUE = "sglang.srt.managers.scheduler.Scheduler._add_request_to_queue"
_GET_PREFILL = "sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw"
_PREFILL_PRIORITY = "sglang.srt.managers.schedule_policy.SchedulePolicy.calc_priority"
_RUN_BATCH = "sglang.srt.managers.scheduler.Scheduler.run_batch"
_RESULT_PREFILL = (
    "sglang.srt.managers.scheduler_components.batch_result_processor."
    "SchedulerBatchResultProcessor.process_batch_result_prefill"
)
HOOK_TARGETS = (
    _PREPARE_MODEL, _GET_LOADER, _WEIGHTS_ITERATOR, _LOAD_WEIGHTS,
    _EMBED_INIT, _EMBED_FORWARD, _EMBED_FINISH,
)
SCHED_PREFETCH_HOOK_TARGETS = (
    _HASH_INIT, _ADD_TO_QUEUE, _GET_PREFILL, _PREFILL_PRIORITY, _RUN_BATCH, _RESULT_PREFILL,
)


def _check_hook_installation(*, sched_prefetch_enabled: bool = False) -> None:
    # SGLang apply_hooks 会记录并继续处理失败项；模型加载前确认必要 hook 已安装，
    # 防止只安装“跳过权重”却未安装 SSD forward 等不完整组合。
    targets = HOOK_TARGETS + SCHED_PREFETCH_HOOK_TARGETS if sched_prefetch_enabled else HOOK_TARGETS
    missing = [target for target in targets if target not in HookRegistry._patched]
    if missing:
        raise RuntimeError(f"Incomplete Engram SSD hook installation: {missing}")
