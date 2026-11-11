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
    sched_prefetch_enabled: bool = False
    pool_bytes: int = 2147483648
    segment_tokens: int = 0
    policy: str = "timeout"
    timeout_ms: int = 500
    log_interval: int = 0


def should_register_hooks() -> bool:
    # 登记阶段不抛错；非法值在进入原模型加载函数前统一拒绝。
    return os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "0").lower() not in _DISABLED_VALUES


def should_register_sched_prefetch_hooks() -> bool:
    return os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "0").lower() not in _DISABLED_VALUES


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
    sched_enabled = _boolean("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH")
    prefix = "SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_"
    names = ("POOL_BYTES", "SEGMENT_TOKENS", "POLICY", "TIMEOUT_MS", "LOG_INTERVAL")
    pool_bytes, segment_tokens, policy, timeout_ms, log_interval = 2147483648, 0, "timeout", 500, 0
    if sched_enabled:
        pool_bytes = int(os.getenv(prefix + "POOL_BYTES", "2147483648"))
        segment_tokens = int(os.getenv(prefix + "SEGMENT_TOKENS", "0"))
        policy = os.getenv(prefix + "POLICY", "timeout").lower()
        log_interval = int(os.getenv(prefix + "LOG_INTERVAL", "0"))
        if policy not in {"best_effort", "wait_complete", "timeout"}:
            raise ValueError(prefix + "POLICY must be best_effort, wait_complete, or timeout")
        if policy == "timeout":
            timeout_ms = int(os.getenv(prefix + "TIMEOUT_MS", "500"))
        elif prefix + "TIMEOUT_MS" in os.environ:
            logger.warning("Ignoring %sTIMEOUT_MS: POLICY is not timeout", prefix)
        if pool_bytes <= 0 or segment_tokens < 0 or log_interval < 0 or timeout_ms < 0:
            raise ValueError("Engram SSD prefetch requires positive POOL_BYTES and nonnegative segment/time/log values")
    else:
        for suffix in names:
            if prefix + suffix in os.environ:
                logger.warning("Ignoring %s%s: SCHED_PREFETCH is disabled", prefix, suffix)
    return EngramSsdConfig(
        enabled=enabled, io_uring_queue_depth=queue_depth, chunk_rows=chunk_rows,
        dedup_enabled=_boolean("SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP"),
        sched_prefetch_enabled=sched_enabled, pool_bytes=pool_bytes,
        segment_tokens=segment_tokens, policy=policy, timeout_ms=timeout_ms, log_interval=log_interval,
    )
