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

## 按槽位读取与 pinned 行池

`read_rows_to_slots(layer_id=..., row_ids=..., slot_ids=..., weight_base=..., scale_base=...)`
将 CPU 连续 int64 `[Q]` 全局行号写入指定的不同槽位，基表为 CPU uint8 `[S,D]`／`[S,D/32]`。
它复用同一个 ring，释放 GIL 并同步收齐 I/O；文件定位与行步长只由 reader 解释，两个 FD 仍由 runtime 持有。

`NativePrefetchRowPool(dim=..., pool_bytes=..., track_stats=False)` 提供：

| 接口 | 生命周期约定 |
|---|---|
| `acquire(layer_id=..., row_ids=...)` | 去重并固定槽，返回 handle；空间不足返回 None，不阻塞；空行集合返回就绪空 handle |
| `handle.missing_row_ids`／`missing_slot_ids` | 只包含本次新建 PENDING 行；同步读取返回后调用 `mark_ready(slot_ids=...)` |
| `is_ready(handle=...)` | 全部槽 READY 才为真；失败 key 显式抛错 |
| `resolve(layer_id=..., row_ids=...)` | 返回 `slot_ids`、`hit_mask`、临时 `handle`；只命中 READY，并保留输入顺序与重复 |
| `release(handle=...)` | 归还引用；I/O 未完成的 PENDING 即使零引用也不能被复用 |
| `cancel(handle=...)` | 仅在未提交 I/O 或 I/O 已收齐失败后撤销自建 PENDING；共享失败 key 显式报错直到全部引用归还 |
| `stats()` | 锁内快照；累计计数只在 track_stats 时更新 |
| `close()` | 要求全部 I/O 与引用已结束，存在 PENDING 或引用时拒绝关闭 |

池的 `weight`／`scale` 为4096起点对齐的 pinned 表，`weight_device_ptr`／`scale_device_ptr` 供原生 gather 直接访问。
行步长单独保存；当前两张表连续，分别为 D 和 D/32。slab 起点对齐不代表未来任意行读已经满足 DIO 对齐。

每个 acquire／resolve handle 都由调用方释放。GPU gather 使用过的 resolve handle 必须等对应 CUDA event 完成后再 release；不能只因为 CPU 函数返回就归还槽位。
