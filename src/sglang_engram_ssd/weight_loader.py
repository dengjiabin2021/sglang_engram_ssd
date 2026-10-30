"""在 get_tensor 之前过滤外置 Engram 大表，普通参数继续正常加载。"""

from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def engram_ssd_safetensors_weights_iterator(
    hf_weights_files: list[str],
    *,
    skip_names: frozenset[str],
    drop_cache_after_load: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    import safetensors

    for st_file in hf_weights_files:
        with safetensors.safe_open(st_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in skip_names:
                    continue  # 只跳过精确的 embed.weight/scale 原始参数名。
                yield name, handle.get_tensor(name)
        if drop_cache_after_load:
            # 保留原普通 loader 的可选 page-cache 回收语义。
            from sglang.srt.model_loader.weight_utils import _drop_file_cache_after_load

            _drop_file_cache_after_load(st_file)
