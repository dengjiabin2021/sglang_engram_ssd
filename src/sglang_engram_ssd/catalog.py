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
    """描述一个原始 safetensors 张量的位置，不持有 FD、映射或张量数据。

    catalog 构建时创建，随后由 catalog 持有并供 runtime 只读共享；无资源关闭动作。
    字段冻结，可跨线程读取；weight 行宽为 D，scale 行宽为 D/32，二者均以原始字节定位。
    """

    path: Path  # 原 shard 的绝对路径，不是导出或重排后的文件。
    raw_key: str  # safetensors header 中的原参数名，用于加载阶段精确跳过大表。
    offset: int  # 在原 shard 中的绝对字节偏移，包含 8 字节长度和 JSON header。
    rows: int  # 张量全局行数；合法行号为 [0,rows)。
    width: int  # 每行原始字节数，weight 为 D，scale 为 D/32。
    dtype: str  # safetensors dtype 标记，仅允许已校验的 F8_E4M3 或 F8_E8M0。


@dataclass(frozen=True)
class EngramSsdCatalog:
    """汇总各 Engram 层原大表的元数据，不负责读取 payload 或持有文件资源。

    模型加载 Hook 构建并安装进当前 worker，runtime 借用到退出；无需显式 close。
    冻结字段中的 views 字典按只读约定共享，调用方不得并发修改；安装只在模型初始化阶段进行。
    """

    root: Path  # 当前 worker 绑定的原 checkpoint 绝对目录。
    views: dict[str, SsdTensorView]  # canonical 参数名 → 已校验的原张量位置。
    layer_ids: tuple[int, ...]      # 只保存配置中启用 Engram 的模型层号。

    @property
    def raw_names(self) -> frozenset[str]:
        """取得应在 safetensors get_tensor 前跳过的大表原参数名。

        输入：self 为已构建 catalog，无额外参数。
        输出：不可变的原参数名集合；不读取文件、不修改目录元数据。
        """
        return frozenset(view.raw_key for view in self.views.values())

    @property
    def canonical_names(self) -> frozenset[str]:
        """取得模型名称映射后的外置大表参数名。

        输入：self 为已构建 catalog，无额外参数。
        输出：不可变的 canonical 参数名集合，用于 load_weights 的第二层过滤。
        """
        return frozenset(self.views.keys())


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    """只读取一个 shard 的 safetensors header，不物化权重张量。

    输入：path 为本地 safetensors 文件路径，调用时须可读。
    输出：(header 字典, payload 的绝对起始字节偏移)；函数同步读取并自行关闭临时文件句柄。
    头长度或文件边界非法时抛 ValueError；文件和 JSON 错误原样传播。
    """
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
    """从 checkpoint 索引构建并校验 Engram weight／scale 的字节目录。

    输入：root 为本地 checkpoint 目录；name_mapper 将原参数名映射为模型 canonical 名；
        layer_rows 为启用层号到全局行数的映射，顺序决定 layer_ids；dim 为行宽 D，须为可被 32 整除的正二次幂。
    输出：包含原 shard 路径、偏移、形状和 dtype 的 catalog，不读取大表 payload，也不安装全局状态。
    文件在本调用内同步读取并关闭；缺表、形状／dtype／字节范围不符时抛 ValueError。
    """
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
    """在模型初始化阶段将目录安装到当前 worker。

    输入：catalog 为完整校验后的目录，调用方随后不得修改 views。
    输出：无返回值；替换进程级目录引用，不复制 payload。已绑定其他 root 时抛 RuntimeError。
    仅由串行模型加载路径调用，不提供运行期间切换模型或并发写保护。
    """
    global _CATALOG
    # 当前实现一个 worker 绑定一个不可变模型目录。
    if _CATALOG is not None and _CATALOG.root != catalog.root:
        raise RuntimeError("An Engram SSD worker cannot switch checkpoint directories")
    _CATALOG = catalog


def get_engram_ssd_catalog() -> EngramSsdCatalog:
    """借用当前 worker 已安装的只读目录。

    输入：无显式参数；调用前须由模型加载 Hook 安装 catalog。
    输出：进程共享的 EngramSsdCatalog 引用，不转移所有权；尚未安装时抛 RuntimeError。
    """
    if _CATALOG is None:
        raise RuntimeError("Engram SSD catalog must be prepared before model construction")
    return _CATALOG
