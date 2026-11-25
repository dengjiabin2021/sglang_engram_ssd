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
    "SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_SEGMENT_TOKENS",
    "SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POLICY",
    "SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_TIMEOUT_MS",
    "SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_MAX_DEFER_MS",
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
    window_batches: int = 2  # 每轮最多推算的 prefill 批数。
    pool_batches: int | None = None  # 按批数定容；两个池配置都为空时取 window_batches + 1。
    pool_bytes: int | None = None  # 每 rank 的行数据字节预算，不含元数据和分配器开销。
    log_interval: int = 0  # 每多少次 prefill 输出统计；0 表示不启用采样与累计计时。
    debug_check_rows: bool = False  # 在 prefill 逐位核对 CPU 和 GPU 全局行号。

    def resolve_pool_bytes(self, *, tokens: int, layers: int, columns: int, dim: int, tp_size: int) -> int:
        """按模型布局和 prefill 预算计算本 rank 的行池大小。

        输入：tokens 为每批 token 预算，layers/columns 为 Engram 层数和哈希列数，dim 为行宽，
            tp_size 为 TP rank 数。输出：行数据字节数；实际槽容量必须严格大于窗口所需槽数。
        """
        row_bytes = dim + dim // 32
        # 每批预计查询行数加 10% owner 分布余量，向上取整到完整行。
        batch_rows = (tokens * layers * columns * 11 + tp_size * 10 - 1) // (tp_size * 10)
        size = self.pool_bytes
        if size is None:
            batches = self.pool_batches if self.pool_batches is not None else self.window_batches + 1
            size = batches * batch_rows * row_bytes
        if size // row_bytes <= self.window_batches * batch_rows:
            raise ValueError("Engram SSD SCHED_PREFETCH_POOL_BYTES must hold more rows than the prefetch window")
        return size


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
    names = ("WINDOW_BATCHES", "POOL_BATCHES", "POOL_BYTES", "LOG_INTERVAL")
    debug_name = "SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEBUG_CHECK_ROWS"
    window_batches, pool_batches, pool_bytes = 2, None, None
    log_interval = 0
    if sched_enabled:
        # 只有功能生效才解析其数值；关闭功能时的无效参数不能阻止原同步路径启动。
        window_batches = int(os.getenv(prefix + "WINDOW_BATCHES", "2"))
        batch_value = os.getenv(prefix + "POOL_BATCHES")
        byte_value = os.getenv(prefix + "POOL_BYTES")
        if batch_value is not None and byte_value is not None:
            raise ValueError(f"{prefix}POOL_BATCHES and {prefix}POOL_BYTES are mutually exclusive")
        pool_batches = int(batch_value) if batch_value is not None else None
        pool_bytes = int(byte_value) if byte_value is not None else None
        if window_batches <= 0 or (pool_bytes is not None and pool_bytes <= 0):
            raise ValueError(f"{prefix}WINDOW_BATCHES and {prefix}POOL_BYTES must be positive")
        if pool_batches is not None and pool_batches <= window_batches:
            raise ValueError(f"{prefix}POOL_BATCHES must exceed {prefix}WINDOW_BATCHES")

        log_interval = int(os.getenv(prefix + "LOG_INTERVAL", "0"))
        if log_interval < 0:
            raise ValueError(f"{prefix}LOG_INTERVAL must be nonnegative")
    else:
        for suffix in names:
            if prefix + suffix in os.environ:
                logger.warning("Ignoring %s%s: SCHED_PREFETCH is disabled", prefix, suffix)
        if debug_name in os.environ:
            logger.warning("Ignoring %s: SCHED_PREFETCH is disabled", debug_name)
    return EngramSsdConfig(
        enabled=enabled, io_uring_queue_depth=queue_depth, chunk_rows=chunk_rows,
        dedup_enabled=_boolean("SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP"),
        sched_prefetch_enabled=sched_enabled, window_batches=window_batches, pool_batches=pool_batches,
        pool_bytes=pool_bytes, log_interval=log_interval,
        debug_check_rows=sched_enabled and _boolean(debug_name),
    )
