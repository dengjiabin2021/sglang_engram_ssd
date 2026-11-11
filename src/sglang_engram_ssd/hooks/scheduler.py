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
    from ..config import get_config
    from ..runtime import get_engram_ssd_runtime

    if not get_config().sched_prefetch_enabled:
        return None
    return get_engram_ssd_runtime(dim=owner.model_config.hf_config.engram_head_dim)


# sglang/srt/managers/scheduler.py: Scheduler._add_request_to_queue(self, req: Req, is_retracted: bool = False)
@register_hook(_ADD_TO_QUEUE, type=HookType.AFTER)
def register_waiting_request(
    function_result: Any, self: Any, req: Req, is_retracted: bool = False,
) -> None:
    from sglang.srt.disaggregation.utils import DisaggregationMode

    runtime = _runtime(owner=self)
    if (
        runtime is not None and self.disaggregation_mode == DisaggregationMode.NULL
        and any(candidate is req for candidate in self.waiting_queue)
    ):
        runtime.get_sched_prefetch(scheduler=self).register_request(
            req=req, is_retracted=is_retracted,
        )


# sglang/srt/managers/scheduler.py: Scheduler._get_new_batch_prefill_raw(
#     self, prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor], running_batch: ScheduleBatch)
@register_hook(_GET_PREFILL, type=HookType.AROUND)
def schedule_engram_prefill(
    original_fn: Callable[..., Any], self: Any,
    prefill_delayer_single_pass: Optional[PrefillDelayerSinglePassExecutor], running_batch: ScheduleBatch,
) -> Any:
    runtime = _runtime(owner=self)
    if runtime is None:
        return original_fn(
            self, prefill_delayer_single_pass=prefill_delayer_single_pass, running_batch=running_batch,
        )
    return runtime.get_sched_prefetch(scheduler=self).run_prefill(
        original_fn=original_fn, scheduler=self,
        prefill_delayer_single_pass=prefill_delayer_single_pass, running_batch=running_batch,
    )


# sglang/srt/managers/schedule_policy.py: SchedulePolicy.calc_priority(
#     self, waiting_queue: List[Req], running_batch: Optional[ScheduleBatch] = None, processed_tokens: int = 0)
@register_hook(_PREFILL_PRIORITY, type=HookType.AFTER)
def filter_engram_candidates(
    function_result: Any, self: Any, waiting_queue: list[Req],
    running_batch: Optional[ScheduleBatch] = None, processed_tokens: int = 0,
) -> None:
    from ..sched_prefetch import filter_active_waiting_queue

    filter_active_waiting_queue(waiting_queue=waiting_queue)


# sglang/srt/managers/scheduler.py: Scheduler.run_batch(
#     self, batch: ScheduleBatch, pp_proxy_tensors: Optional[PPProxyTensors] = None)
@register_hook(_RUN_BATCH, type=HookType.AFTER)
def record_engram_batch(
    function_result: Any, self: Any, batch: ScheduleBatch,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
) -> None:
    runtime = _runtime(owner=self)
    if runtime is not None:
        runtime.get_sched_prefetch(scheduler=self).record_batch(batch=batch)


# sglang/srt/managers/scheduler_components/batch_result_processor.py:
# SchedulerBatchResultProcessor.process_batch_result_prefill(
#     self, batch: ScheduleBatch, result: Union[GenerationBatchResult, EmbeddingBatchResult])
@register_hook(_RESULT_PREFILL, type=HookType.AFTER)
def release_engram_prefill(
    function_result: Any, self: Any, batch: ScheduleBatch,
    result: Union[GenerationBatchResult, EmbeddingBatchResult],
) -> None:
    runtime = _runtime(owner=self)
    manager = getattr(runtime, "sched_prefetch", None) if runtime is not None else None
    if manager is not None:
        manager.release_batch(batch=batch)
