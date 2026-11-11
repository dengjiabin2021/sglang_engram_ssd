# io_uring 原生读取模块

`BatchRowReader` 只负责同步文件读取，不做行缓存、CUDA gather 或 TP 通信。全局行地址与 FD 选择集中在 reader，调用方不依赖存储布局。

## 构建

以下命令仅供目标 Linux NVIDIA 环境执行，本地未编译。

```bash
cd /workspace/sglang_engram_ssd
cmake -S src/sglang_engram_ssd/query/csrc -B build/query \
  -DCMAKE_BUILD_TYPE=Release \
  -DENGRAM_SSD_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/query --parallel 4
```

需要当前环境的 Torch、Python 开发头文件和 liburing 开发文件。liburing 是必需依赖；可向 CMake 显式传入 `LIBURING_INCLUDE_DIR`、`LIBURING_LIBRARY` 指向已手动准备的文件。扩展写入 `query/_C*.so`，editable 安装不会自动编译。

## 同步接口

```python
from sglang_engram_ssd.query import NativeBatchRowReader

reader = NativeBatchRowReader(tables=tables, io_uring_queue_depth=32)
weight, scale = reader.allocate_workspace(layer_id=layer_id, capacity=128)
reader.read_rows(
    layer_id=layer_id,
    row_ids=row_ids,
    weight_out=weight,
    scale_out=scale,
)
reader.close()
```

`row_ids` 为连续 CPU int64 全局行号；两个输出是连续 CPU uint8 `[K,D]`、`[K,D/32]`，只写前 Q 行并保留重复与顺序。输入输出不得重叠。

调用同步返回前所有 CQE 已回收，短读、EINTR、完成项错误和未读满均显式处理。读取时释放 GIL；Python 包装保护单个 ring 的串行调用与关闭。reader 借用 runtime 的 FD，close 不关闭 FD。

返回的 pinned 内存只能在 GPU 已停止读取后复用；runtime 用逐块 CUDA event 保证单套工作区生命周期。
