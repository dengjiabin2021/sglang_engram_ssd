"""集中管理 EngramEmbedding 的构造、推理和加载完成替换。"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Optional

from sglang.srt.plugins.hook_registry import (
    HookType,
    plugin_hook as register_hook,
)

from ..timing import get_timer
from ._registry import _EMBED_FINISH, _EMBED_FORWARD, _EMBED_INIT

if TYPE_CHECKING:
    import torch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# sglang/python/sglang/srt/layers/engram.py：EngramEmbedding.__init__(
#     self, num_embeddings: int, dim: int, layer_id: int)
@register_hook(_EMBED_INIT, type=HookType.AROUND)
def init_ssd_embedding(
    original_fn: Callable[..., Any],
    self: Any,
    num_embeddings: int,
    dim: int,
    layer_id: int,
) -> None:
    """用 SSD runtime 元数据替代原 Embedding 的整表构造。

    输入：original_fn 为被替换的官方构造函数；self 为尚未初始化的 Embedding；
        num_embeddings 为全局行数，dim 为行宽 D，layer_id 为当前模型层号。
    输出：无返回值；初始化 nn.Module，记录本 rank owner 范围并借用共享 runtime，weight／scale 参数设为空。
    有意不调用 original_fn，防止同时分配完整 DDR 表或 HBM 分片；runtime 负责 reader 和工作区的释放。
    """
    from torch import nn

    from ..runtime import get_engram_ssd_runtime

    # 跳过原构造函数，避免按原配置分配 DDR 表或 HBM 分片。
    nn.Module.__init__(self)
    self.dim = dim
    self.layer_id = layer_id
    # 同进程各层 dim 相同，缓存 factory 返回同一 runtime；构造时统一 init_table。
    self.ssd = get_engram_ssd_runtime(dim=dim)
    self.tp_size = self.ssd.tp_size
    # 按全局行号划分 owner；这里只记录范围，不分配完整 DDR/HBM 表。
    self.row_start = num_embeddings * self.ssd.tp_rank // self.tp_size
    row_end = num_embeddings * (self.ssd.tp_rank + 1) // self.tp_size
    self.rows = row_end - self.row_start
    self.host_table = None
    self.register_parameter("weight", None)
    self.register_parameter("scale", None)


# sglang/python/sglang/srt/layers/engram.py：EngramEmbedding.forward(
#     self, indices: torch.Tensor, forward_batch: Optional[ForwardBatch] = None,
#     *, cp_all_tokens: bool = False) -> torch.Tensor
@register_hook(_EMBED_FORWARD, type=HookType.AROUND)
def forward_ssd_embedding(
    original_fn: Callable[..., torch.Tensor],
    self: Any,
    indices: torch.Tensor,
    forward_batch: Optional[ForwardBatch] = None,
    *,
    cp_all_tokens: bool = False,
) -> torch.Tensor:
    """在原 Engram 消费位置完成本 rank 查询与一次 TP 结果合并。

    输入：original_fn 为官方 forward；self 为 Embedding；indices 为 GPU int32／int64 [T,C] 全局行号，
        可重复、无需 contiguous，须在表范围内；forward_batch 为可选批元数据；cp_all_tokens 不允许为 True。
    输出：GPU BF16 [T,C,D] 合并结果；runtime 不存在时完全调用原 forward。
    已登记的纯 prefill 和 decode 从显存缓冲 gather，其余批次同步读取；TP 合并保持原调用顺序。
    """
    runtime = getattr(self, "ssd", None)
    if runtime is None:
        return original_fn(self, indices, forward_batch, cp_all_tokens=cp_all_tokens)
    if cp_all_tokens:
        raise ValueError("Engram SSD does not support context-parallel lookup")
    if forward_batch is not None and forward_batch.contains_mm_inputs():
        raise ValueError("Engram SSD currently supports text-only requests")

    def run() -> torch.Tensor:
        """本层查询加一次 TP 合并。"""
        if runtime.sched_prefetch_enabled and forward_batch is not None:
            out = runtime.lookup_batch(layer_id=self.layer_id, indices=indices, forward_batch=forward_batch)
        else:
            out = runtime.lookup(layer_id=self.layer_id, indices=indices)
        # 模型原 Engram 位置每层合并一次，保持与主干 TP 通信的执行顺序。
        return runtime.reduce_output(out)

    timer = get_timer()
    if timer is None:
        return run()
    # 本 forward 已绑定预取缓冲时带上本层紧急缺行数，计时日志据此区分全命中。
    batch = runtime._batch[1] if runtime._batch is not None else None
    urgent = batch.urgent.get(self.layer_id, 0) if batch is not None and batch.forward is forward_batch else None
    return timer.run(forward_batch=forward_batch, fn=run, urgent=urgent)


# sglang/python/sglang/srt/layers/engram.py：EngramEmbedding.finish_load(self, label: str)
@register_hook(_EMBED_FINISH, type=HookType.AROUND)
def finish_ssd_embedding_load(
    original_fn: Callable[..., Any], self: Any, label: str
) -> Any:
    """在 SSD Embedding 上跳过只适用于完整 DDR backing 的加载完成处理。

    输入：original_fn 为官方 finish_load；self 为 Embedding；label 为原加载阶段标识。
    输出：SSD 实例返回 None，不执行原 DDR barrier；非 SSD 实例转调原函数并透传其返回值。
    """
    if getattr(self, "ssd", None) is not None:
        return None  # 没有完整 DDR backing，无需执行其写入完成 barrier。
    return original_fn(self, label)
