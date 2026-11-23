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
    """逐参数产出普通权重，在 get_tensor 前排除外置 Engram 大表。

    输入：hf_weights_files 为按加载顺序排列的本地 safetensors 路径；skip_names 为需跳过的原参数名集合；
        drop_cache_after_load 为是否在每个 shard 迭代完成后沿用原 loader 的页缓存回收行为。
    输出：惰性生成 (原参数名, CPU 原 dtype／形状 Tensor)，仅普通参数会物化；不修改 checkpoint。
    生成器持有当前文件上下文至该 shard 迭代结束或生成器关闭；调用方负责消费或关闭生成器。
    """
    import safetensors

    for st_file in hf_weights_files:
        with safetensors.safe_open(st_file, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in skip_names:
                    continue  # 只跳过精确的 embed.weight/scale 原始参数名。
                # 过滤必须发生在 get_tensor 前，避免被跳过的大表先占用主存。
                yield name, handle.get_tensor(name)
        if drop_cache_after_load:
            # 保留原普通 loader 的可选 page-cache 回收语义。
            from sglang.srt.model_loader.weight_utils import _drop_file_cache_after_load

            _drop_file_cache_after_load(st_file)
