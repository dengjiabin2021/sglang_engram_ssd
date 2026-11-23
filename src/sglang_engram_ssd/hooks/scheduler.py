"""转发 scheduler 生命周期通知，将业务状态留在调度预取控制器中。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional, Union

from sglang.srt.plugins.hook_registry import HookType, plugin_hook as register_hook

from ._registry import _ADD_TO_QUEUE, _GET_PREFILL, _PREFILL_PRIORITY, _RESULT_PREFILL, _RUN_BATCH

if TYPE_CHECKING:
    from sglang.srt.managers.prefill_delayer import PrefillDelayerSinglePassExecutor
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.utils import EmbeddingBatchResult, GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import PPProxyTensors


def _runtime(owner: Any) -> Any:
    """按总开关取得已加载模型对应的 runtime。

    输入：owner 为提供 model_config 的原生 scheduler 或结果处理器；仅借用其模型配置。
    输出：关闭调度预取时返回 None，否则按 Engram 行宽取得进程内缓存 runtime，不新建模型。
    """
    from ..config import get_config
    from ..runtime import get_engram_ssd_runtime

    if not get_config().sched_prefetch_enabled:
        return None
    return get_engram_ssd_runtime(dim=owner.model_config.hf_config.engram_head_dim)


# sglang/python/sglang/srt/managers/scheduler.py:
# Scheduler._add_request_to_queue(self, req: Req, is_retracted: bool = False)
@register_hook(_ADD_TO_QUEUE, type=HookType.AFTER)
def register_waiting_request(
    function_result: Any, self: Any, req: Req, is_retracted: bool = False,
) -> None:
    """在原生请求实际进入 waiting 后转交预取登记。

    输入：function_result 为原函数结果，忽略；self 为 scheduler；req 为原生 Req；
        is_retracted 标记撤回重登，参数与原函数保持一致。
    输出：无返回值；只对普通非 PD 的 waiting 纯文本请求登记状态，不在 Hook 中执行读取。
    """
    from sglang.srt.disaggregation.utils import DisaggregationMode

    runtime = _runtime(owner=self)
    if (
        runtime is not None and self.disaggregation_mode == DisaggregationMode.NULL
        and any(candidate is req for candidate in self.waiting_queue)
    ):
        # 原生入队可能把请求送往其他队列；只有真实 waiting 项才能由此控制器持有。
        runtime.get_sched_prefetch(scheduler=self).register_request(
            req=req, is_retracted=is_retracted,
        )


# sglang/python/sglang/srt/managers/scheduler.py: Scheduler._get_new_batch_prefill_raw(
#     self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor], running_batch: ScheduleBatch)
@register_hook(_GET_PREFILL, type=HookType.AROUND)
def schedule_engram_prefill(
    original_fn: Callable[..., Any], self: Any,
    prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor], running_batch: ScheduleBatch,
) -> Any:
    """把一次原生 raw 调度交给控制器包围，关闭时原样调用。

    输入：original_fn 为原生未绑定函数，self 为 scheduler；
        prefill_delayer_single_pass/running_batch 与原函数同义并原样传递。
    输出：原函数返回的 batch/running_batch 结果；控制器负责筛选、早退预取和统一失败同步。
    所有分支均调用 original_fn，Hook 不复制原生调度实现。
    """
    runtime = _runtime(owner=self)
    if runtime is None:
        return original_fn(
            self, prefill_delayer_single_pass=prefill_delayer_single_pass, running_batch=running_batch,
        )
    return runtime.get_sched_prefetch(scheduler=self).run_prefill(
        original_fn=original_fn, scheduler=self,
        prefill_delayer_single_pass=prefill_delayer_single_pass, running_batch=running_batch,
    )


# sglang/python/sglang/srt/managers/schedule_policy.py: SchedulePolicy.calc_priority(
#     self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None, processed_tokens: int = 0)
@register_hook(_PREFILL_PRIORITY, type=HookType.AFTER)
def filter_engram_candidates(
    function_result: Any, self: Any, waiting_queue: list[Req],
    running_batch: Optional[ScheduleBatch] = None, processed_tokens: int = 0,
) -> None:
    """在原生 priority 排序后触发当前 raw 调度的候选筛选。

    输入：function_result 为原结果，忽略；self 为 SchedulePolicy；waiting_queue 为原生可变 Req 列表；
        running_batch/processed_tokens 保留原签名，Hook 不使用或修改二者。
    输出：无返回值；仅在插件包围的 raw 上下文中由控制器原地筛选，不改变 priority 返回值。
    """
    from ..sched_prefetch import filter_active_waiting_queue

    filter_active_waiting_queue(waiting_queue=waiting_queue)


# sglang/python/sglang/srt/managers/scheduler.py: Scheduler.run_batch(
#     self, batch: ScheduleBatch, pp_proxy_tensors: Optional[PPProxyTensors] = None)
@register_hook(_RUN_BATCH, type=HookType.AFTER)
def record_engram_batch(
    function_result: Any, self: Any, batch: ScheduleBatch,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> None:
    """在原生 forward 提交后记录不可变的消费快照。

    输入：function_result 为原 forward 结果，忽略；self 为 scheduler；batch 为本次 ScheduleBatch；
        pp_proxy_tensors 保留原签名，不使用；当前插件不支持 PP。
    输出：无返回值；启用时转交控制器保存版本与 token 区间，不修改模型输出。
    """
    runtime = _runtime(owner=self)
    if runtime is not None:
        runtime.get_sched_prefetch(scheduler=self).record_batch(batch=batch)


# sglang/python/sglang/srt/managers/scheduler_components/batch_result_processor.py:
# SchedulerBatchResultProcessor.process_batch_result_prefill(
#     self, batch: ScheduleBatch, result: Union[GenerationBatchResult, EmbeddingBatchResult])
@register_hook(_RESULT_PREFILL, type=HookType.AFTER)
def release_engram_prefill(
    function_result: Any, self: Any, batch: ScheduleBatch,
    result: Union[GenerationBatchResult, EmbeddingBatchResult],
) -> None:
    """在原生 prefill 结果处理后转交行池请求引用回收。

    输入：function_result 为原处理结果，忽略；self 为结果处理器；batch/result 为原函数参数，
        只使用 batch 的 forward_iter，result 原样保留且不修改。
    输出：无返回值；仅已有控制器时按提交快照释放已消费段，不在 frozen 结果处理器上挂状态。
    """
    runtime = _runtime(owner=self)
    manager = getattr(runtime, "sched_prefetch", None) if runtime is not None else None
    if manager is not None:
        manager.release_batch(batch=batch)
