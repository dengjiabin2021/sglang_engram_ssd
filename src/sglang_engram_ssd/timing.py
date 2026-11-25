"""用 CUDA event 记录每个 prefill forward 中各 Engram 层查询（含 TP all-reduce）的 GPU 耗时。"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import torch

logger = logging.getLogger(__name__)


def timing_enabled() -> bool:
    """读取计时开关 SGLANG_ENABLE_DSV41_ENGRAM_TIMING，SSD 关闭时也可单独开启。"""
    return os.getenv("SGLANG_ENABLE_DSV41_ENGRAM_TIMING", "0").lower() in ("1", "true", "yes", "y")


class EngramTimer:
    """按 forward 收集各 Engram 层的起止 event，event 完成后再输出日志，不阻塞计算流。

    只由模型线程调用；同一 forward 内的层按调用顺序编号，各 rank 的 forward 序号一致。
    """

    def __init__(self) -> None:
        """初始化当前 forward 和等待 event 完成的记录。"""
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.forward: Any = None  # 当前正在记录的 ForwardBatch。
        self.index = -1  # 已记录的 prefill forward 序号。
        self.pending: list[dict[str, Any]] = []  # 尚未输出的记录，按 forward 顺序排列。

    def flush(self) -> None:
        """输出所有 event 已完成的记录；遇到未完成的记录即停止，保持输出顺序。"""
        while self.pending and self.pending[0]["events"][-1][1].query():
            record = self.pending.pop(0)
            line = {
                "rank": self.rank, "forward": record["forward"], "t": record["t"], "tokens": record["tokens"],
                "ms": [start.elapsed_time(end) for start, end in record.pop("events")],
                "urgent": record["urgent"],
            }
            logger.info("Engram timing: %s", json.dumps(line))

    def run(self, *, forward_batch: Any, fn: Callable[[], torch.Tensor], urgent: int | None = None) -> torch.Tensor:
        """执行一层 Engram 查询；纯 prefill 时在当前流前后记录 event。

        输入：forward_batch 为原生批次；fn 返回该层合并后的输出；urgent 为本层本 rank 的紧急缺行数，
            None 表示该层没有走预取缓冲（DDR、同步 SSD 或退回同步查询）。
        输出：fn 的返回值。decode 及 mixed 只输出已完成的记录，不计时。
        """
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None or not mode.is_extend() or mode.is_mixed():
            self.flush()
            return fn()
        if forward_batch is not self.forward:
            # 新 forward 的第一层：之前的记录都已收齐各层，可以输出；t 用于剔除服务就绪前的预热请求。
            self.flush()
            self.forward = forward_batch
            self.index += 1
            self.pending.append({
                "forward": self.index, "t": time.time(), "tokens": int(forward_batch.input_ids.numel()),
                "events": [], "urgent": [],
            })
        record = self.pending[-1]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn()
        end.record()
        record["events"].append((start, end))
        record["urgent"].append(urgent)
        return out


@lru_cache(maxsize=1)
def get_timer() -> EngramTimer | None:
    """返回进程内唯一的计时器；开关关闭时返回 None。"""
    return EngramTimer() if timing_enabled() else None
