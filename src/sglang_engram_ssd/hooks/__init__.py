"""按替换对象组织 Hook，导入本包时统一完成装饰器登记。"""

from __future__ import annotations

from ..config import should_register_sched_prefetch_hooks
from ._registry import HOOK_TARGETS

# 先登记模型／权重加载，再登记 embedding；模型加载时会统一检查这些目标均已成功替换。
# 根包 register() 只需导入本包，各子模块的装饰器均在插件来源上下文内执行。
from .model_loading import (
    load_model_with_engram_ssd,
    require_default_model_loader,
)
from .weight_loading import (
    load_weights_with_external_engram,
    selective_weights_iterator,
)
from .engram_embedding import (
    finish_ssd_embedding_load,
    forward_ssd_embedding,
    init_ssd_embedding,
)

__all__ = [
    "HOOK_TARGETS",
    "load_model_with_engram_ssd",
    "require_default_model_loader",
    "selective_weights_iterator",
    "load_weights_with_external_engram",
    "init_ssd_embedding",
    "forward_ssd_embedding",
    "finish_ssd_embedding_load",
]

if should_register_sched_prefetch_hooks():
    # 开关关闭不导入新 Hook，避免仅导入插件就捕获 hash 参数或创建调度资源。
    from .engram_hasher import capture_batch_hash, capture_request_hasher
    from .scheduler import (
        prepare_engram_batch,
        reconcile_engram_window,
        record_engram_batch,
        register_waiting_request,
        release_engram_prefill,
        schedule_engram_prefill,
    )

    __all__.extend([
        "capture_request_hasher", "capture_batch_hash", "register_waiting_request", "schedule_engram_prefill",
        "reconcile_engram_window", "prepare_engram_batch", "record_engram_batch", "release_engram_prefill",
    ])
