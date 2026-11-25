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

需要当前环境的 Torch、CUDA Toolkit、Python 开发头文件和 liburing 开发文件。扩展链接 CUDA Driver 以提交 decode 的流标志操作。liburing 是必需依赖；可向 CMake 显式传入 `LIBURING_INCLUDE_DIR`、`LIBURING_LIBRARY` 指向已手动准备的文件。扩展写入 `query/_C*.so`，editable 安装不会自动编译。

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

调用同步返回前所有 CQE 已回收，短读、EINTR、完成项错误和未读满均显式处理。滑动窗口在完成项到达后补交剩余部分或下一请求，不等待同批最慢请求。出错后停止补交，先收齐在途请求再抛错，保证返回后不再写目标内存。

读取时释放 GIL；Python 包装保护单个 ring 的串行调用与关闭。每个实例独占 ring／锁，借用 runtime 的 FD，close 不关闭 FD。调度预取开启时，runtime 分别创建前台 `reader` 和后台 `prefetch_reader`；各自按配置 queue depth 提交，不共享 reader 锁，但共用磁盘带宽。

返回的 pinned 内存只能在 GPU 已停止读取后复用；runtime 用逐块 CUDA event 保证单套工作区生命周期。

## 按槽位读取与 pinned 行池

`read_rows_to_slots(layer_id=..., row_ids=..., slot_ids=..., weight_base=..., scale_base=...)`
将 CPU 连续 int64 `[Q]` 全局行号写入指定的不同槽位，基表为 CPU uint8 `[S,D]`／`[S,D/32]`。
同一实例的两种读取方法复用其 ring，释放 GIL 并同步收齐 I/O；runtime 的后台使用独立实例。文件定位与行步长只由 reader 解释，两个 FD 仍由 runtime 持有。

`NativePrefetchRowPool(dim=..., pool_bytes=..., track_stats=False)` 提供：

| 接口 | 生命周期约定 |
|---|---|
| `acquire(layer_id=..., row_ids=...)` | 为普通预取段去重并固定槽，全部行放得下时取得引用，空间不足时返回 None |
| `handle.row_ids`／`slot_ids` | 按首次出现顺序去重后的 CPU int64 全局行号与槽号 |
| `handle.missing_row_ids`／`missing_slot_ids` | 只包含本次新建 PENDING 行；同步读取返回后调用 `mark_ready(slot_ids=...)` |
| `pending_slots(layer_id=..., row_ids=...)` | 按输入位置返回 PENDING 槽，其它位置为 `-1`。仅查元数据，使用前须持有槽引用 |
| `available_slots()`／`slots_needed(layer_id=..., row_ids=..., limit=-1)` | 查询空闲加 LRU 槽数，以及这些行尚需占用的可用槽数。设定非负 `limit` 时，所需槽数超过它即可结束统计 |
| `is_ready(handle=...)` | 全部槽 READY 才为真；失败 key 显式抛错 |
| `resolve(layer_id=..., row_ids=...)` | 返回 `slot_ids`、`hit_mask`、临时 `handle`；只命中 READY，并保留输入顺序与重复 |
| `release(handle=..., discard_pending=False)` | 归还引用。普通段撤销时设为 `True` 以回收零引用 PENDING，共享槽由其它段继续读取。在途写入须始终有槽引用 |
| `cancel(handle=...)` | 仅在未提交 I/O 或 I/O 已收齐失败后撤销自建 PENDING；共享失败 key 显式报错直到全部引用归还 |
| `stats()` | 锁内快照；累计计数只在 track_stats 时更新 |
| `record_submitted(rows=...)` | 在实际交付 reader 前计数；acquire／取消占槽不算读取提交 |
| `close()` | 要求全部 I/O 与引用已结束，存在 PENDING 或引用时拒绝关闭 |

池的 `weight`／`scale` 为起点 4096 对齐的连续 pinned 表，行步长分别为 D 和 D/32。每个 acquire／resolve 句柄由调用方归还，批次组装方在 READY 行复制到调用方暂存区后归还 resolve 引用。

普通段仍使用 `pending_slots` 在块提交前跳过其它段已经发布的共享槽。紧急读取查询 READY，其余行通过同一个 `read_rows_to_slots` 直接写到调用方的 batch 暂存区。

decode 使用扩展的 `stream_flag_address` 探测映射标志及流等待支持。CPU pinned int32 `[1]` 标志由 `reset_stream_flag` 在旁路流清零。暂存数据写完后由 `set_stream_flag` 置位，拷贝流等到标志为 1 才执行 H2D，计算流只等 H2D event。
