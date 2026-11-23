"""验证滑动窗口读取的字节一致性、EOF 收尾与错误后 ring 复用；仅在目标环境运行。"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
import torch

from sglang_engram_ssd.query.interface import RowTableSpec
from sglang_engram_ssd.query.native import NativeBatchRowReader

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("queue_depth", [1, 3, 32])
def test_more_requests_than_queue_depth(
    cuda_native_device: torch.device, tmp_path: Path, queue_depth: int,
) -> None:
    """A0/P2：请求数超过 QD，非对齐偏移与重复行仍按输入位置逐字节写入。"""
    rows, dim = 37, 256
    weight = torch.arange(rows * dim, dtype=torch.int64).remainder(251).to(torch.uint8).view(rows, dim)
    scale = torch.arange(rows * 8, dtype=torch.int64).remainder(113).to(torch.uint8).view(rows, 8)
    weight_path, scale_path = tmp_path / "weight.bin", tmp_path / "scale.bin"
    weight_path.write_bytes(b"w" * 13 + bytes(weight.flatten().tolist()))
    scale_path.write_bytes(b"s" * 29 + bytes(scale.flatten().tolist()))
    weight_fd, scale_fd = os.open(weight_path, os.O_RDONLY), os.open(scale_path, os.O_RDONLY)
    reader = None
    try:
        spec = RowTableSpec(
            layer_id=1, rows=rows, dim=dim, weight_fd=weight_fd, scale_fd=scale_fd,
            weight_offset=13, scale_offset=29,
        )
        reader = NativeBatchRowReader(tables=[spec], io_uring_queue_depth=queue_depth)
        row_ids = torch.tensor([36, 0, 17, 36, *range(rows)], dtype=torch.int64)
        weight_out, scale_out = reader.allocate_workspace(layer_id=1, capacity=row_ids.numel())
        reader.read_rows(layer_id=1, row_ids=row_ids, weight_out=weight_out, scale_out=scale_out)
        assert torch.equal(weight_out, weight.index_select(0, row_ids))
        assert torch.equal(scale_out, scale.index_select(0, row_ids))
    finally:
        if reader is not None:
            reader.close()
        os.close(weight_fd)
        os.close(scale_fd)


@pytest.mark.parametrize("error_mode", ["short_read_eof", "cqe_error"])
def test_read_error_drains_and_reader_can_be_reused(
    cuda_native_device: torch.device, tmp_path: Path, error_mode: str,
) -> None:
    """A0/P2：短读到 EOF 或 CQE 读错先收尾；修复源后复用 ring，不再改动旧输出。"""
    rows, dim = 17, 256
    weight = torch.arange(rows * dim, dtype=torch.int64).remainder(251).to(torch.uint8).view(rows, dim)
    scale = torch.arange(rows * 8, dtype=torch.int64).remainder(113).to(torch.uint8).view(rows, 8)
    weight_bytes, scale_bytes = bytes(weight.flatten().tolist()), bytes(scale.flatten().tolist())
    weight_path, scale_path = tmp_path / "weight.bin", tmp_path / "scale.bin"
    # 最后一行只保留半行，使内核先返回正短读，再在重提剩余部分时返回 EOF。
    weight_path.write_bytes(weight_bytes[:-128] if error_mode == "short_read_eof" else weight_bytes)
    scale_path.write_bytes(scale_bytes)
    weight_fd = os.open(weight_path, os.O_RDONLY)
    scale_fd = os.open(scale_path, os.O_WRONLY if error_mode == "cqe_error" else os.O_RDONLY)
    reader = None
    try:
        spec = RowTableSpec(
            layer_id=1, rows=rows, dim=dim, weight_fd=weight_fd, scale_fd=scale_fd,
            weight_offset=0, scale_offset=0,
        )
        reader = NativeBatchRowReader(tables=[spec], io_uring_queue_depth=4)
        row_ids = torch.tensor([rows - 1, *range(rows - 1)], dtype=torch.int64)
        old_weight, old_scale = reader.allocate_workspace(layer_id=1, capacity=rows)
        message = "unexpected EOF" if error_mode == "short_read_eof" else "io_uring read"
        with pytest.raises(RuntimeError, match=message):
            reader.read_rows(layer_id=1, row_ids=row_ids, weight_out=old_weight, scale_out=old_scale)

        # read_rows 抛出前必须已经收齐 CQE；随后调用方可以立即覆盖旧输出。
        old_weight.fill_(255)
        old_scale.fill_(255)
        if error_mode == "short_read_eof":
            weight_path.write_bytes(weight_bytes)
        else:
            # 保留 reader 借用的 fd 数值，只把只写描述符替换为可读描述符。
            replacement_fd = os.open(scale_path, os.O_RDONLY)
            try:
                os.dup2(replacement_fd, scale_fd)
            finally:
                os.close(replacement_fd)
        weight_out, scale_out = reader.allocate_workspace(layer_id=1, capacity=rows)
        reader.read_rows(layer_id=1, row_ids=row_ids, weight_out=weight_out, scale_out=scale_out)
        assert torch.equal(weight_out, weight.index_select(0, row_ids))
        assert torch.equal(scale_out, scale.index_select(0, row_ids))
        assert torch.equal(old_weight, torch.full_like(old_weight, 255))
        assert torch.equal(old_scale, torch.full_like(old_scale, 255))
    finally:
        if reader is not None:
            reader.close()
        os.close(weight_fd)
        os.close(scale_fd)
