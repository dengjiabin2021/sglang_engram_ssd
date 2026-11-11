"""批量行查询协议和必需原生实现；导入本包不加载动态库。"""

from __future__ import annotations

from .interface import BatchRowReader, RowTableSpec
from .native import NativeBatchRowReader, NativePrefetchRowPool

__all__ = ["BatchRowReader", "NativeBatchRowReader", "NativePrefetchRowPool", "RowTableSpec"]
