"""插件自己的环境变量，不向 SGLang Envs 类添加字段。"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import lru_cache


_ENABLED_VALUES = frozenset({"1", "true", "yes", "y"})
_DISABLED_VALUES = frozenset({"0", "false", "no", "n"})
# 服务只开放三种读取方式，Python 串行路径作为原始实现保留。
_QUERY_BACKENDS = frozenset({"python", "cpp_thread_pool", "cpp_io_uring"})

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngramSsdConfig:
    enabled: bool
    lookup_mode: str
    query_backend: str
    num_threads: int | None
    thread_pool_queue_depth: int | None
    io_uring_queue_depth: int | None
    cache_bytes: int
    chunk_rows: int


def should_register_hooks() -> bool:
    # 只判断是否明确关闭，不做会抛异常的完整配置校验。
    # 非法开关值也先登记 Hook，随后由模型加载入口的 get_config 严格拒绝。
    raw = os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "0").lower()
    return raw not in _DISABLED_VALUES


def _read_backend_positive_int(
    name: str,
    default: int,
    *,
    backend: str,
    selected_backend: str,
) -> int | None:
    # 先判断适用后端，再解析数值；无关参数即使不是整数，也只警告并忽略。
    raw = os.getenv(name)
    if selected_backend != backend:
        if raw is not None:
            logger.warning("Ignoring %s: not used by query_backend=%s", name, selected_backend)
        return None
    value = int(raw) if raw is not None else default
    if value <= 0:
        raise ValueError(f"{name} must be positive for query_backend={selected_backend}")
    return value


@lru_cache(maxsize=1)
def get_config() -> EngramSsdConfig:
    # 由模型加载 Hook 首次调用；校验失败时中止模型加载，运行中不切换 backend。
    raw = os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_SSD", "0").lower()
    if raw not in _ENABLED_VALUES and raw not in _DISABLED_VALUES:
        raise ValueError("SGLANG_ENABLE_DSV41_ENGRAM_SSD must be a boolean")
    enabled = raw in _ENABLED_VALUES
    if not enabled:
        # 正常关闭时不会登记 Hook；此处只接受已启用 SSD 的调用。
        raise RuntimeError("get_config() requires Engram SSD to be enabled")

    # 当前只开放每个 rank 处理全部查询的模式；未实现的模式在模型加载前报错。
    lookup_mode = os.getenv("SGLANG_DSV41_ENGRAM_SSD_LOOKUP_MODE", "all_queries").lower()
    if lookup_mode != "all_queries":
        raise ValueError(
            f"Unsupported SGLANG_DSV41_ENGRAM_SSD_LOOKUP_MODE={lookup_mode!r}; "
            "currently only 'all_queries' is supported"
        )

    # 查询模式决定哪些行由当前 rank 处理，读取后端决定这些行如何从文件取出。
    # 默认沿用 Python 路径，只有显式选择 C++ 后端时才要求已构建原生扩展。
    query_backend = os.getenv("SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND", "python").lower()
    if query_backend not in _QUERY_BACKENDS:
        raise ValueError(
            f"Unsupported SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND={query_backend!r}; "
            "expected python, cpp_thread_pool or cpp_io_uring"
        )
    # 两类队列分别配置：线程池统计等待的行任务，io_uring 统计在途文件读取。
    # 不适用字段保存 None，不将它们的默认值或原始配置传给其他后端。
    num_threads = _read_backend_positive_int(
        "SGLANG_DSV41_ENGRAM_SSD_NUM_THREADS", 4,
        backend="cpp_thread_pool", selected_backend=query_backend,
    )
    thread_pool_queue_depth = _read_backend_positive_int(
        "SGLANG_DSV41_ENGRAM_SSD_THREAD_POOL_QUEUE_DEPTH", 32,
        backend="cpp_thread_pool", selected_backend=query_backend,
    )
    io_uring_queue_depth = _read_backend_positive_int(
        "SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH", 32,
        backend="cpp_io_uring", selected_backend=query_backend,
    )
    # 旧变量没有唯一含义，不再沿用；提示改用后端专属变量，但不阻止启动。
    if "SGLANG_DSV41_ENGRAM_SSD_QUEUE_DEPTH" in os.environ:
        logger.warning(
            "Ignoring SGLANG_DSV41_ENGRAM_SSD_QUEUE_DEPTH: use "
            "SGLANG_DSV41_ENGRAM_SSD_THREAD_POOL_QUEUE_DEPTH or "
            "SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH instead"
        )

    cache_bytes = int(os.getenv("SGLANG_DSV41_ENGRAM_SSD_CACHE_BYTES", str(512 * 1024 * 1024)))
    chunk_rows = int(os.getenv("SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS", "4096"))
    if cache_bytes < 0 or chunk_rows <= 0:
        raise ValueError("Engram SSD requires cache_bytes >= 0 and chunk_rows > 0")
    return EngramSsdConfig(
        enabled=enabled,
        lookup_mode=lookup_mode,
        query_backend=query_backend,
        num_threads=num_threads,
        thread_pool_queue_depth=thread_pool_queue_depth,
        io_uring_queue_depth=io_uring_queue_depth,
        cache_bytes=cache_bytes,
        chunk_rows=chunk_rows,
    )
