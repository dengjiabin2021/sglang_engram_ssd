"""批量行查询协议及可选原生实现；导入本包不加载原生动态库。"""

from .interface import BatchRowReader, RowTableSpec
from .native import NativeBatchRowReader

__all__ = ["BatchRowReader", "NativeBatchRowReader", "RowTableSpec"]
