"""只解析原 checkpoint 元数据，不导出、重排或复制权重文件。"""

from __future__ import annotations

import json
import os
import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SsdTensorView:
    path: Path
    raw_key: str
    offset: int  # 在原 shard 中的绝对字节偏移，包含 8 字节长度和 JSON header。
    rows: int
    width: int
    dtype: str


@dataclass(frozen=True)
class EngramSsdCatalog:
    root: Path
    views: dict[str, SsdTensorView]  # canonical 参数名 → 已校验的原张量位置。
    layer_ids: tuple[int, ...]      # 只保存配置中启用 Engram 的模型层号。

    @property
    def raw_names(self) -> frozenset[str]:
        return frozenset(view.raw_key for view in self.views.values())

    @property
    def canonical_names(self) -> frozenset[str]:
        return frozenset(self.views.keys())


def read_exact_at(fd: int, size: int, offset: int) -> bytes:
    """同步读满给定区间；短读继续读取，到文件末尾仍不足才报错。"""
    parts: list[bytes] = []
    received = 0
    while received < size:
        part = os.pread(fd, size - received, offset + received)
        if not part:
            raise EOFError(f"Checkpoint short read: offset={offset}, expected={size}, got={received}")
        parts.append(part)
        received += len(part)
    return b"".join(parts)


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_len = struct.unpack("<Q", prefix)[0]
        if header_len > os.fstat(stream.fileno()).st_size - 8:
            raise ValueError(f"Safetensors header exceeds file size: {path}")
        header = json.loads(stream.read(header_len))
    return header, 8 + header_len


def build_engram_ssd_catalog(
    root: Path,
    *,
    name_mapper: Callable[[str], str],
    layer_rows: dict[int, int],
    dim: int,
) -> EngramSsdCatalog:
    # 形状和量化布局只在目录构建阶段检查；runtime 不重复校验配置。
    if not layer_rows:
        raise ValueError("The model has no configured Engram layers")
    if dim <= 0 or dim & (dim - 1) or dim % 32:
        raise ValueError("Engram head_dim must be a positive power of two divisible by 32")
    root = root.resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    headers: dict[Path, tuple[dict, int]] = {}
    views: dict[str, SsdTensorView] = {}

    for raw_name, shard_name in index["weight_map"].items():
        canonical = name_mapper(raw_name)
        match = re.fullmatch(r"model\.layers\.(\d+)\.engram\.embed\.(weight|scale)", canonical)
        if match is None:
            continue  # 其他 .weight 参数不会进入这里，包括 Engram wkv。
        layer_id = int(match.group(1))
        if layer_id not in layer_rows:
            continue
        path = (root / shard_name).resolve()
        if path not in headers:
            headers[path] = read_safetensors_header(path=path)
        header, data_base = headers[path]
        meta = header[raw_name]
        is_weight = match.group(2) == "weight"
        expected_dtype = "F8_E4M3" if is_weight else "F8_E8M0"
        expected_width = dim if is_weight else dim // 32

        # U8 scale 不直接当成 E8M0，避免绕过原 loader 数值转换后改变结果。
        if meta["dtype"] != expected_dtype:
            raise ValueError(f"{raw_name}: expected {expected_dtype}, got {meta['dtype']}")
        if meta["shape"] != [layer_rows[layer_id], expected_width]:
            raise ValueError(
                f"{raw_name}: expected shape {[layer_rows[layer_id], expected_width]}, "
                f"got {meta['shape']}"
            )
        rows, width = meta["shape"]
        begin, end = meta["data_offsets"]
        if begin < 0 or end - begin != rows * width or data_base + end > path.stat().st_size:
            raise ValueError(f"{raw_name}: invalid tensor byte range in {path}")
        if canonical in views:
            raise ValueError(f"Duplicate canonical Engram parameter: {canonical}")
        views[canonical] = SsdTensorView(
            path=path, raw_key=raw_name, offset=data_base + begin,
            rows=rows, width=width, dtype=meta["dtype"],
        )

    # 缺失的 weight/scale 在同一个校验阶段报告，不留到 init_table 中检查形状。
    expected = {
        f"model.layers.{layer_id}.engram.embed.{kind}"
        for layer_id in layer_rows
        for kind in ("weight", "scale")
    }
    missing = expected.difference(views)
    if missing:
        raise ValueError(f"Missing Engram checkpoint parameters: {sorted(missing)}")
    return EngramSsdCatalog(root=root, views=views, layer_ids=tuple(layer_rows))


_CATALOG: EngramSsdCatalog | None = None


def install_engram_ssd_catalog(catalog: EngramSsdCatalog) -> None:
    global _CATALOG
    # 当前实现一个 worker 绑定一个不可变模型目录。
    if _CATALOG is not None and _CATALOG.root != catalog.root:
        raise RuntimeError("An Engram SSD worker cannot switch checkpoint directories")
    _CATALOG = catalog


def get_engram_ssd_catalog() -> EngramSsdCatalog:
    if _CATALOG is None:
        raise RuntimeError("Engram SSD catalog must be prepared before model construction")
    return _CATALOG
