"""在官方 Hasher 构造后捕获 CPU 参数，不提前调用 forward 或提交 history。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt.plugins.hook_registry import HookType, plugin_hook as register_hook

from ._registry import _HASH_INIT

if TYPE_CHECKING:
    from sglang.srt.layers.engram import EngramLayout


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
