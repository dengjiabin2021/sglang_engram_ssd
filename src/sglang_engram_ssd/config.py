"""插件自己的环境变量，不向 SGLang Envs 类添加字段。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


_ENABLED_VALUES = frozenset({"1", "true", "yes", "y"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "n"})


@dataclass(frozen=True)
class EngramSsdConfig:
    enabled: bool
    cache_bytes: int
    chunk_rows: int


def should_register_hooks() -> bool:
    # 只判断是否明确关闭，不做会抛异常的完整配置校验。
    # 非法开关值也先登记 Hook，随后由模型加载入口的 get_config 严格拒绝。
    raw = os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "0").lower()
    return raw not in _DISABLED_VALUES


@lru_cache(maxsize=1)
def get_config() -> EngramSsdConfig:
    # 由模型加载 Hook 首次调用；校验失败时中止模型加载，运行中不切换 backend。
    raw = os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "0").lower()
    if raw not in _ENABLED_VALUES and raw not in _DISABLED_VALUES:
        raise ValueError("SGLANG_ENABLE_DSV41_ENGRAM_SSD must be a boolean")
    enabled = raw in _ENABLED_VALUES
    if not enabled:
        return EngramSsdConfig(False, 512 * 1024 * 1024, 4096)

    cache_bytes = int(os.getenv("SGLANG_DSV41_ENGRAM_SSD_CACHE_BYTES", str(512 * 1024 * 1024)))
    chunk_rows = int(os.getenv("SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS", "4096"))
    if cache_bytes < 0 or chunk_rows <= 0:
        raise ValueError("Engram SSD requires cache_bytes >= 0 and chunk_rows > 0")
    return EngramSsdConfig(enabled, cache_bytes, chunk_rows)
