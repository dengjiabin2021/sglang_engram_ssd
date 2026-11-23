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
    """保存进程内已校验的只读配置，不创建 reader、行池或后台线程。

    由 get_config 首次成功调用时创建并缓存，供加载、runtime 和调度模块共享。
    字段冻结后可跨线程只读；启动后修改环境变量不会更新这个实例。
    """

    enabled: bool  # SSD 总开关；get_config 只在开启时返回实例。
    io_uring_queue_depth: int  # 每个 reader 的最大在途文件请求数，必须为正。
    chunk_rows: int  # 同步中转区最多容纳的查询行数 Q，必须为正。
    dedup_enabled: bool = False  # 仅控制同步查询的 unique／inverse，不影响行池自身去重。
    sched_prefetch_enabled: bool = False  # 控制调度预取相关 Hook、行池与后台 reader 的创建。
    pool_bytes: int = 2147483648  # 每 rank 的压缩行 payload 预算，单位字节，不含元数据／分配器开销。
    segment_tokens: int = 0  # 每段 token 数；0 由控制器取 chunk 长度或 8192。
    policy: str = "timeout"  # 取值为 best_effort、wait_complete 或 timeout。
    timeout_ms: int = 500  # timeout 策略等待毫秒数；其他策略未提交请求的防饥饿仍使用默认值。
    log_interval: int = 0  # 每多少次 prefill 输出统计；0 表示不启用采样与累计计时。


def should_register_hooks() -> bool:
    """在插件发现阶段判断 SSD 是否明确关闭，不执行完整配置校验。

    输入：无显式参数；只读取 SSD 总开关环境变量，缺省为关闭。
    输出：明确关闭返回 False，其他值返回 True；非法值留给模型加载前的 get_config 报错。
    """
    # 登记阶段不抛错；非法值在进入原模型加载函数前统一拒绝。
    return os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "0").lower() not in _DISABLED_VALUES


def should_register_sched_prefetch_hooks() -> bool:
    """判断是否登记调度预取 Hook，不创建运行期资源。

    输入：无显式参数；读取调度预取开关，调用方负责先检查 SSD 总开关。
    输出：明确关闭返回 False，其他值返回 True；这里只做登记门控，不校验从属参数。
    """
    return os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH", "0").lower() not in _DISABLED_VALUES


def _boolean(name: str, default: str = "0") -> bool:
    """按统一合法值集合解析一个布尔环境变量。

    输入：name 为完整环境变量名；default 为未设置时使用的字符串默认值。
    输出：合法值对应的 bool；不属于开启／关闭集合时抛 ValueError，不静默采用默认值。
    """
    raw = os.getenv(name, default).lower()
    if raw not in _ENABLED_VALUES | _DISABLED_VALUES:
        raise ValueError(f"{name} must be a boolean")
    return raw in _ENABLED_VALUES


@lru_cache(maxsize=1)
def get_config() -> EngramSsdConfig:
    """解析、校验并缓存当前生效的 SSD 配置。

    输入：无显式参数；集中读取本模块声明的环境变量，启动阶段调用。
    输出：冻结的 EngramSsdConfig；非法生效参数抛 ValueError，总开关关闭时抛 RuntimeError。
    退役参数或所属功能关闭的显式参数只警告并忽略；成功结果按进程缓存，调用方只读使用。
    """
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
        # 只有功能生效才解析其数值；关闭功能时的无效参数不能阻止原同步路径启动。
        pool_bytes = int(os.getenv(prefix + "POOL_BYTES", "2147483648"))
        segment_tokens = int(os.getenv(prefix + "SEGMENT_TOKENS", "0"))
        policy = os.getenv(prefix + "POLICY", "timeout").lower()
        log_interval = int(os.getenv(prefix + "LOG_INTERVAL", "0"))
        if policy not in {"best_effort", "wait_complete", "timeout"}:
            raise ValueError(prefix + "POLICY must be best_effort, wait_complete, or timeout")
        if policy == "timeout":
            timeout_ms = int(os.getenv(prefix + "TIMEOUT_MS", "500"))
        elif prefix + "TIMEOUT_MS" in os.environ:
            # 其他策略不采用显式等待值；保留默认值供未提交请求的防饥饿规则使用。
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
