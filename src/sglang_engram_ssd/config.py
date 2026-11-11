"""统一解析插件开关与固定 io_uring 同步路径配置。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache

_ENABLED_VALUES = frozenset({"1", "true", "yes", "y"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "n"})
logger = logging.getLogger(__name__)

# 仅用于一次性迁移提示，不再解析或启用这些旧配置。
_RETIRED_NAMES = (
    "SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND",
    "SGLANG_DSV41_ENGRAM_SSD_NUM_THREADS",
    "SGLANG_DSV41_ENGRAM_SSD_THREAD_POOL_QUEUE_DEPTH",
    "SGLANG_DSV41_ENGRAM_SSD_QUEUE_DEPTH",
    "SGLANG_DSV41_ENGRAM_SSD_LOOKUP_MODE",
    "SGLANG_DSV41_ENGRAM_SSD_CACHE_BYTES",
    "SGLANG_ENABLE_DSV41_ENGRAM_SSD_PIPELINE",
    "SGLANG_ENABLE_DSV41_ENGRAM_SSD_PREFETCH",
    "SGLANG_DSV41_ENGRAM_SSD_PREFETCH_LAYERS",
)


@dataclass(frozen=True)
class EngramSsdConfig:
    enabled: bool
    io_uring_queue_depth: int
    chunk_rows: int
    dedup_enabled: bool = False


def should_register_hooks() -> bool:
    # 登记阶段不抛错；非法值在进入原模型加载函数前统一拒绝。
    return os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "0").lower() not in _DISABLED_VALUES


def _boolean(name: str, default: str = "0") -> bool:
    raw = os.getenv(name, default).lower()
    if raw not in _ENABLED_VALUES | _DISABLED_VALUES:
        raise ValueError(f"{name} must be a boolean")
    return raw in _ENABLED_VALUES


@lru_cache(maxsize=1)
def get_config() -> EngramSsdConfig:
    enabled = _boolean("SGLANG_ENABLE_DSV41_ENGRAM_SSD")
    if not enabled:
        raise RuntimeError("get_config() requires Engram SSD to be enabled")
    for name in _RETIRED_NAMES:
        if name in os.environ:
            logger.warning("Ignoring retired Engram SSD configuration %s", name)
    queue_depth = int(os.getenv("SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH", "32"))
    chunk_rows = int(os.getenv("SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS", "4096"))
    if queue_depth <= 0 or chunk_rows <= 0:
        raise ValueError("Engram SSD IO_URING_QUEUE_DEPTH and CHUNK_ROWS must be positive")
    return EngramSsdConfig(
        enabled=enabled, io_uring_queue_depth=queue_depth, chunk_rows=chunk_rows,
        dedup_enabled=_boolean("SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP"),
    )
