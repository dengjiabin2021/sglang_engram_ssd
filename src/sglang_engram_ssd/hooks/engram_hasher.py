"""捕获官方 CPU hash 参数和 decode forward 产出的 GPU 行号。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt.plugins.hook_registry import HookType, plugin_hook as register_hook

from ._registry import _HASH_FORWARD, _HASH_INIT

if TYPE_CHECKING:
    import torch
    from sglang.srt.layers.engram import EngramLayout
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


# sglang/python/sglang/srt/layers/engram.py：EngramHasher.__init__(
#     self, layout: EngramLayout, tokenizer, pad_id: int, compressed_vocab_size: int)
@register_hook(_HASH_INIT, type=HookType.AFTER)
def capture_request_hasher(
    result: None,
    self: Any,
    layout: EngramLayout,
    tokenizer: Any,
    pad_id: int,
    compressed_vocab_size: int,
) -> None:
    """在官方 Hasher 构造完成后为调度预取保存只读 CPU hash 参数。

    输入：result 为原构造函数的 None；self 为已经初始化的官方 Hasher；layout 提供行宽 D 与层顺序；
        tokenizer、pad_id、compressed_vocab_size 为原构造参数，Hook 不重建 tokenizer，也不再次执行构造。
    输出：无返回值；将 RequestHasher 安装到当前 runtime，仅复制计算参数，不更新请求 history。
    本 Hook 只随调度预取开关登记，构造阶段同步完成；复制出的参数由 runtime 持有到关闭。
    """
    from ..hashing import RequestHasher
    from ..runtime import get_engram_ssd_runtime

    runtime = get_engram_ssd_runtime(dim=layout.head_dim)
    # 使用模型真实参数和 layout 的层顺序，避免根据 checkpoint 配置重算另一套 hash 参数。
    runtime.set_request_hasher(hasher=RequestHasher.from_hasher(hasher=self, layer_ids=layout.layer_ids))
    self._engram_ssd = runtime


# sglang/python/sglang/srt/layers/engram.py：EngramHasher.forward(
#     self, input_ids: torch.Tensor, forward_batch: ForwardBatch) -> torch.Tensor
@register_hook(_HASH_FORWARD, type=HookType.AFTER)
def capture_batch_hash(
    result: torch.Tensor, self: Any, input_ids: torch.Tensor, forward_batch: ForwardBatch,
) -> None:
    """在原生 hash 和 history 更新提交后绑定本批，交出 decode 的 GPU 行号。

    输入：result 为 GPU int64 [T,L,C]，各层行号可重复；self、input_ids、forward_batch 为原函数参数。
    输出：无返回值；已登记的 decode 排队拷贝行号，prefill 仅关联批次，原生输出仍由模型使用。
    """
    runtime = getattr(self, "_engram_ssd", None)
    if runtime is not None:
        runtime.bind_batch(forward_batch=forward_batch, hash_ids=result)
