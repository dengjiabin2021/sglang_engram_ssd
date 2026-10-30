"""按替换对象组织 Hook，导入本包时统一完成装饰器登记。"""

from ._registry import HOOK_TARGETS

# 保留原 hooks.py 的登记顺序：模型加载、权重加载、embedding。
# 根包 register() 仍只需导入 hooks，各子模块均在插件来源上下文内加载。
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
