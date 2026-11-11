"""在官方 Hasher 构造后捕获 CPU 参数，不提前调用 forward 或提交 history。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sglang.srt.plugins.hook_registry import HookType, plugin_hook as register_hook

from ._registry import _HASH_INIT

if TYPE_CHECKING:
    from sglang.srt.layers.engram import EngramLayout


# sglang/srt/layers/engram.py：EngramHasher.__init__(
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
    from ..hashing import RequestHasher
    from ..runtime import get_engram_ssd_runtime

    runtime = get_engram_ssd_runtime(dim=layout.head_dim)
    runtime.set_request_hasher(hasher=RequestHasher.from_hasher(hasher=self, layer_ids=layout.layer_ids))
