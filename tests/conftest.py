"""提供不依赖真实 checkpoint 的合成 Engram 表与可选 CUDA 原生扩展环境。"""

from __future__ import annotations

import os
from collections.abc import Callable
from importlib import import_module, util
from pathlib import Path

import pytest
import torch

from sglang_engram_ssd.catalog import EngramSsdCatalog, SsdTensorView


@pytest.fixture(scope="session")
def cuda_native_device() -> torch.device:
    if torch.version.cuda is None or not torch.cuda.is_available():
        pytest.skip("A0 requires an NVIDIA CUDA device and the compiled io_uring extension")
    if util.find_spec("sglang_engram_ssd.query._C") is None:
        pytest.skip("A0 requires the compiled sglang_engram_ssd.query._C io_uring extension")
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    # 已构建扩展的 ABI 或依赖错误应报错，不伪装成缺少测试依赖。
    import_module("sglang_engram_ssd.query._C")
    return torch.device("cuda", torch.cuda.current_device())


@pytest.fixture
def synthetic_catalog(
    tmp_path: Path,
) -> Callable[..., tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]]:
    def create(*, rows: int, dim: int = 256) -> tuple[EngramSsdCatalog, torch.Tensor, torch.Tensor]:
        # 0..111 是有限 E4M3 编码；E8M0 字节 127 表示精确的单位缩放。
        weights = ((torch.arange(rows * dim, dtype=torch.int64, device="cpu") * 17 + 3) % 112)
        weights = weights.to(torch.uint8).view(rows, dim)
        weights[:, 0] = (torch.arange(rows, dtype=torch.int64, device="cpu") % 111 + 1).to(torch.uint8)
        scales = torch.full((rows, dim // 32), 127, dtype=torch.uint8, device="cpu")
        path = tmp_path / f"engram-{rows}-{dim}.bin"
        weight_offset = 37
        scale_offset = weight_offset + weights.numel() + 19
        path.write_bytes(
            b"W" * weight_offset + bytes(weights.flatten().tolist())
            + b"S" * 19 + bytes(scales.flatten().tolist()) + b"\0" * 11
        )
        views = {
            "model.layers.1.engram.embed.weight": SsdTensorView(
                path=path, raw_key="synthetic.weight", offset=weight_offset,
                rows=rows, width=dim, dtype="F8_E4M3",
            ),
            "model.layers.1.engram.embed.scale": SsdTensorView(
                path=path, raw_key="synthetic.scale", offset=scale_offset,
                rows=rows, width=dim // 32, dtype="F8_E8M0",
            ),
        }
        catalog = EngramSsdCatalog(root=tmp_path, views=views, layer_ids=(1,))
        return catalog, weights, scales

    return create
