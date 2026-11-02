# 批量行查询接口

| 文件 | 内容 |
| --- | --- |
| `interface.py` | Python 元数据类型与 `BatchRowReader` 协议。 |
| `native.py` | runtime 使用的 `NativeBatchRowReader`，负责延迟加载和实例内串行调用。 |
| `csrc/batch_reader.h` | 对应的 C++ 元数据和类接口声明。 |
| `csrc/batch_reader.cpp` | 同步 `pread` 批量读取实现。 |
| `csrc/thread_pool.h`、`csrc/thread_pool.cpp` | 固定线程数和有界任务队列。 |
| `csrc/io_uring_reader.h`、`csrc/io_uring_reader.cpp` | 可选的有界异步提交与完成回收。 |
| `csrc/bindings.cpp` | Python 参数转换与读取期间的 GIL 释放。 |
| `csrc/CMakeLists.txt` | 原生静态库与可选 Python 模块构建配置。 |

## 表初始化

构造 `NativeBatchRowReader` 时，传入当前 rank 各层的 `RowTableSpec` 和所有层共用的缓存预算。读取参数分别为 `num_threads`、`thread_pool_queue_depth`、`io_uring_queue_depth`，默认均为 `None`；`io_backend` 默认为 `pread`。只使用当前后端适用的参数，显式传入的无关参数会 warning 后忽略。表信息沿用 catalog 校验后的全局行数、行宽、绝对偏移和 runtime 已打开的 fd。reader 保存元数据并借用 fd，不重新解析 checkpoint，也不承担 fd 所有权。

`batch_reader.h` 完整定义 `BatchRowReader` 的成员与方法，`batch_reader.cpp` 实现这些方法。`tables_` 保存表元数据，`cache_lru_` 保存最近使用顺序，`cache_index_` 按层号和行号定位缓存，`closed_` 保存关闭状态。

## 批量读取

Python 接口为 `read_rows(*, layer_id, row_ids, weight_out, scale_out) -> None`，C++ 参数顺序与名称保持一致。

| 参数 | 约定 |
| --- | --- |
| `layer_id` | 选择构造时提供的 Engram 层。 |
| `row_ids` | 连续 CPU `int64 [Q]`，保存全局行号。 |
| `weight_out` | 连续 CPU `uint8 [K,D]`，由调用方分配。 |
| `scale_out` | 连续 CPU `uint8 [K,D/32]`，与 weight 使用相同 K。 |

要求 `Q <= K`，D 为当前层行宽；输出允许使用 pinned 内存。仅写前 Q 行，第 i 个槽对应 `row_ids[i]`，重复行号保留重复输出，剩余槽不变。`read_rows()` 不分配输出工作区，不执行查询去重或 TP owner 划分。

缓存未命中时，weight 从 `weight_offset + row_id * D` 读取 D 字节，scale 从其自身文件的 `scale_offset + row_id * (D/32)` 读取 D/32 字节。输出保留 FP8/E8M0 编码，不进行 CPU 反量化。

## Pinned 工作区

`allocate_workspace(*, layer_id, capacity)` 在 C++ 中一次性分配两个 CPU `uint8` pinned Tensor，形状为 `[capacity,D]` 和 `[capacity,D/32]`。容量必须为正，分配失败直接报错。相同行宽的层可共用，后续查询重复传入这两个 Tensor，不按层或按批次重新分配。

返回的是持有底层内存的 Tensor 句柄，不复制数据。Python 调用方负责保存工作区；工作区占用 `capacity * (D + D/32)` 字节，不计入 LRU 预算，reader 关闭不使已返回的 Tensor 失效。

C++ 使用输出 Tensor 的 `data_ptr()` 直接填充：未命中时 `pread` 写入 pinned 输出，命中时从 LRU 复制到 pinned 输出。不经过 Python `bytearray`、`frombuffer` 或额外的普通 CPU 输出张量；保留缓存副本时仍需将读取结果复制进 LRU。

创建工作区时持有 GIL，批量读取时仍按原约定释放 GIL。重新填充同一 pinned 工作区前，调用方必须等待此前从它发起的 H2D 完成；GPU 工作区的复用另由 runtime 等待对应 gather 完成。runtime 按流水开关分配一组或两组 pinned 输出，reader 接口本身不管理 CUDA stream 或双缓冲。

## 跨层行缓存

缓存完全由 C++ 管理，Python 薄封装不保留第二份行缓存。缓存键为当前 reader 中的层号和全局行号，一行保存 `D + D/32` 字节的 weight/scale 原始数据。

命中后复制到当前输出槽，最近使用顺序按原查询顺序更新。未命中时先读满两段数据，再缓存完整行；空间不足时从全局队尾淘汰，所有层共用同一预算。重复查询仍保留各自的输出槽，不做查询去重。

`cache_bytes` 必须非负，0 表示不保留行缓存；单行超过预算时也直接读输出。`cached_bytes_` 记录已缓存 payload 字节数，不包含索引、容器开销和系统 page cache。`close()` 清空缓存并将计数归零，不关闭借用 fd。

## 有界线程池

使用 `io_backend="pread"` 时，可配置以下参数：

| 构造参数 | 作用 |
| --- | --- |
| `num_threads` | 缺省 `None` 按 0 处理，使用底层 C++ 串行路径；正数创建固定数量的 worker，负数报错。 |
| `thread_pool_queue_depth` | 仅启用线程池时使用，缺省 `None` 按 32 处理，必须为正，限制等待执行的行任务数。 |

`io_uring_queue_depth` 在 `pread` 后端不适用，显式传入只 warning 并忽略。未启用线程池时，`thread_pool_queue_depth` 也只 warning 并忽略，不校验其数值。

每项任务读取一行的 weight 和 scale，并写入其原始输出槽。执行中的任务不超过 `num_threads`，等待队列不超过 `thread_pool_queue_depth`；队满时提交方等待，不为每次查询创建新线程。批次的 future 跟踪数量不超过 Q，Q 受输出工作区容量 K 约束。

调用线程先查看当前缓存并填充命中槽，将未命中的查询提交给 worker。worker 只读文件，不访问 Python 对象或 LRU。所有读取完成后，调用线程按原 slot 顺序更新缓存，保持最终 LRU 顺序；重复的未命中查询仍各自读取，不合并。

提交失败、非法行号或读取异常时，也先等待全部已提交任务结束，再抛出异常。`read_rows()` 返回后不会有 worker 继续写入工作区。`close()` 先排空队列并回收线程，再清理缓存和表信息，借用 fd 仍由调用方最后关闭。

## io_uring 批量读取

选择 `io_backend="io_uring"` 时，仅使用 `io_uring_queue_depth`，缺省 `None` 按 32 处理，必须为正。它限制在途文件读取数，weight 与 scale 各占一个请求，因此不是行数；深度为 1 也可使用。显式传入 `num_threads` 或 `thread_pool_queue_depth` 只 warning 并忽略，不要求清零，也不校验其数值。该路径使用原始只读 fd 和普通 buffered READ，支持小行及 scale 的非对齐偏移。

未命中的查询先构造描述符，再按固定窗口批量提交。请求索引通过 SQE 的 `user_data` 对应到输出地址，CQE 可乱序到达；每轮收齐后，短读和 `-EINTR` 的剩余部分进入下一轮。所有窗口完成后，仍按原 slot 顺序更新 LRU，不做查询去重。

ring 使用 `flags=0`，不启用 SQPOLL，按[实际提交数](https://github.com/axboe/liburing/blob/master/man/io_uring_submit.3)累计已接受请求。[等待结果与 I/O 结果](https://github.com/axboe/liburing/blob/master/man/io_uring_wait_cqe.3)分开处理；错误路径同样收齐已提交 CQE，不把[销毁 ring](https://github.com/axboe/liburing/blob/master/man/io_uring_queue_exit.3)当作缓冲区可复用的保证。

提交或等待控制错误会使 ring 在排空后关闭，不自动切回其他后端。等待控制错误持续发生时，调用可能持续阻塞；接口没有超时返回或取消功能，不能提前释放仍可能被写入的输出。

此后端需要构建时启用 liburing，并由目标 Linux 内核允许相应操作。未构建或初始化失败时直接报错，不静默回退到 `pread`。每个 reader 独立持有 ring，不关闭借用的模型文件 fd。

## 完成与资源释放

接口采用同步完成语义：成功返回时前 Q 行全部填满，内部并发读取也须在返回前完成；空查询不读写数据。短读继续读取，EOF 或 I/O 错误导致未读满时抛出异常，失败后的输出不可用。

同一 reader 由调用方串行调用。输入与两个输出的内存区间不得重叠，调用期间保持有效且不被复用；后续 H2D 与 CUDA kernel 的同步仍由 Python runtime 负责。

调用方先结束读取，再调用 `close()`，最后关闭 fd。reader 的缓存、线程池和 ring 不跨 rank 共享。

## 当前接入状态

已实现 C++ 串行 `pread`、有界线程池及可选 `io_uring`，共用 LRU 和 pinned 工作区。各路径均补齐短读、重试 `EINTR`，在 EOF 或 I/O 错误时抛出异常。只写前 Q 行，不排序、不去重。

服务提供 Python 串行、C++ 线程池和 C++ io_uring 三个后端，均支持 `all_queries` 和 `owner_allreduce`；owner 模式筛选后复用同一全查询实现。`SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND` 默认为 `python`；Python 路径不加载 `query._C`，原生模块只在选择 C++ 后端并创建 reader 时加载。

| `QUERY_BACKEND` 值 | 原生 `io_backend` | 使用的读取参数 |
| --- | --- | --- |
| `python` | 不创建原生 reader | 不使用后端专属读取参数。 |
| `cpp_thread_pool` | `pread` | `NUM_THREADS` 默认 `4`，`THREAD_POOL_QUEUE_DEPTH` 默认 `32`，均须为正。 |
| `cpp_io_uring` | `io_uring` | `IO_URING_QUEUE_DEPTH` 默认 `32`，须为正。 |

表中的环境变量均使用 `SGLANG_DSV41_ENGRAM_SSD_` 前缀。`THREAD_POOL_QUEUE_DEPTH` 限制等待的行任务数；`IO_URING_QUEUE_DEPTH` 限制在途文件请求数，weight、scale 各占一个请求。

服务仅解析、校验所选后端适用的环境变量；显式设置的无关参数只 warning 并忽略，即使值不是整数或为负数，也不会阻止启动。未设置的无关参数不产生 warning，runtime 以 `None` 表示不适用项。旧 `SGLANG_DSV41_ENGRAM_SSD_QUEUE_DEPTH` 已停用，显式设置会 warning 并忽略，须改用专属变量。

原生后端的 runtime 在表初始化后创建一个跨层 reader，借用所有层的 fd，并复用 C++ 分配的 pinned 工作区。每块直接传入连续 CPU `int64` 行号片段，原生路径不转 Python 列表、不填充 Python LRU；Python 后端继续使用原缓存、串行逐行读取与工作区填充。

三个后端均保留同步 D2H 和按 K 分块，H2D 与 gather 的调度由 runtime 的块间流水开关决定。按层预取开启时，同一个 lookup 在后台线程执行，保留这条数据路径。runtime 关闭时先结束预取线程，再释放 reader，最后关闭文件 fd。未构建扩展、未启用 liburing 或后端初始化失败时直接报错，不静默回退。

绑定在持有 GIL 时转换表字段和 Tensor 参数，仅在 C++ `read_rows()` 执行期间释放 GIL；返回或抛出异常后自动恢复 GIL。Python 实例锁串行化读取和关闭，避免 GIL 释放后另一线程清空表信息，不协调不同 rank。

`close()` 可重复调用，关闭后再次读取会报错，借用的 fd 仍由调用方关闭。

## Runtime owner 查询与归约

`SGLANG_DSV41_ENGRAM_SSD_LOOKUP_MODE` 默认 `all_queries`，也可设为 `owner_allreduce`。每层 N 行、P 个 TP rank 时，rank r 负责 `[N*r//P, N*(r+1)//P)`；N 不必整除 P，允许空区间。只划分读取责任，每节点仍使用同版本的完整 checkpoint，不产生分片文件。

`_lookup_owner_allreduce()` 在 GPU 展平 `[T,C]`，筛出本 rank 的原查询位置 `positions[M]`，用 `index_select` 得到仍为全局行号的 `owner_ids[M]`。原 `_lookup_all_queries()` 读取紧凑 BF16 `[M,D]` 后，通过 `index_copy_` 回填完整零张量 `[T,C,D]`；重复查询不去重，位置保持原顺序。清零与回填完成后才返回。reader、LRU、分块、H2D 和 A/B gather 全部复用，不新增读取后端或 C++ owner 逻辑。

`EngramEmbedding.forward` 在模型线程取得 lookup/Future 结果后，统一调用 `reduce_output()`，使用 `tensor_model_parallel_all_reduce()` 的返回值。每个实际 Engram 调用点归约一次完整输出，不按块通信；无本地命中的 rank 仍带完整零张量参加。TP=1 直接走全查询，全局空输出跳过归约。后台查询始终不通信，避免与 Attention/MoE 集体调用交错；`all_queries` 不增加归约。

三种读取后端、`PIPELINE` 和按层预取均可组合。owner 模式增加紧凑 `[M,D]` BF16、位置和行号临时张量，仍有完整输出及 all-reduce 的通信/缓冲开销。当前仅静态核对，实际 TP 与跨节点运行尚未验证。

## Runtime 块间流水

`SGLANG_ENABLE_DSV41_ENGRAM_SSD_PIPELINE` 默认 `0`，适用于全部三个服务后端。关闭时使用一套 CPU/GPU 工作区，在初始化时确定的同一条流上逐块传输和 gather，并在块末等待；开启时使用 A/B 两套工作区，独立的 `copy_stream` 执行 H2D，固定的 `gather_stream` 执行 gather。

`EngramSsdBuffer` 是保存张量、流和事件的数据类，不包含构造和调度方法。runtime 初始化时为 A/B 分别新建独立的固定 `gather_stream`，两者都与模型主流分开，后续 lookup 不重新绑定。开启流水时 A/B 共用一条独立 `copy_stream`，关闭时只有一套工作区，`copy_stream=gather_stream`。`h2d_done`、`gather_done` 保护各槽复用；结束时分别记录并等待各槽的 `lookup_done`，收齐两条 gather 流。

执行 lookup 的线程顺序读取各块，提交 GPU 工作后即可填充另一套 Host 工作区。`PIPELINE` 本身不创建后台线程；reader 仍只负责把当前块读满，runtime 按下列依赖保护缓冲区：

| 操作 | 等待条件 |
| --- | --- |
| CPU 重新填充 Host A/B | 该 Host 槽上一次 H2D 已完成。 |
| H2D 覆盖 GPU A/B | 该 GPU 槽上一次 gather 已完成。 |
| gather 读取 GPU A/B | 本块 H2D 已完成。 |

lookup 结束前排空本次 GPU 工作再释放 runtime 锁，保持同步返回。最终 BF16 输出和只读 `device_ids` 共用一份；仅 CPU/GPU 中转区从一套变成两套，行缓存预算不变。该开关只负责 lookup 内的块间重叠，跨层预取由独立开关控制；重叠收益需在目标设备确认。

在服务启动前设置 `export SGLANG_ENABLE_DSV41_ENGRAM_SSD_PIPELINE=1` 即可开启，改为 `0` 恢复逐块等待；修改后需重启服务。

## Runtime 按层预取

`SGLANG_ENABLE_DSV41_ENGRAM_SSD_PREFETCH` 默认 `0`。开启后，每个 runtime 使用一个持久后台线程直接执行原 `runtime.lookup()`，Future 保存整层形状的 BF16 GPU 输出；owner 模式下是尚未归约的局部结果。`SGLANG_DSV41_ENGRAM_SSD_PREFETCH_LAYERS` 使用逗号分隔的模型层号，如 `14`、`1,14`；未设置或为空表示全部 Engram 层。预取关闭时，显式设置层号只 warning 并忽略。

| 阶段 | 执行动作 |
| --- | --- |
| 模型进入 | AROUND Hook 建立本次 forward 上下文。 |
| hash 就绪 | AFTER Hook 保留原 hash/history 更新，立即按层号顺序统一提交所有选中层的任务；任务持有该层索引视图和就绪事件。 |
| 后台查询 | 以 A 的流等待 hash 就绪并准备索引，各块使用 A/B 各自的 gather 流；复用原 lookup、行缓存、reader 和 A/B 工作区。 |
| Embedding 消费 | 本层有 Future 则取结果，未完成则等待；没有任务则同步 lookup。随后在模型线程统一调用 `reduce_output()`，owner 多 rank 非空输出在此归约。 |
| forward 退出 | 正常或异常路径都收齐后台任务；关闭 runtime 时先停止线程，再释放 reader/fd。 |

上下文只保存 runtime、模型层号到 `hash_ids[T,L,C]` 中第二个维度 L 的位置映射、层号到 Future 的字典。任务统一提交后，由一个 worker 复用现有 A/B 工作区排队执行，不等待前层输出被消费，也不表示多层 lookup 并发。未预取层的同步 lookup 使用同一个 `runtime.lock`，后台另一层正在查询时会等锁。三个读取后端、行缓存和 `PIPELINE` 配置继续生效，不回到 SGLang 原表；不增加 CPU raw 块池或第二份缓存。

Future 完成时，本 rank 的查询与回填已在 GPU 就绪；owner 模式的 TP 归约仍留在模型线程，后台不做通信。前台用 `record_stream` 保护消费期间的存储寿命，并从字典移除已消费的 Future。多个已完成但尚未消费的层输出可以累积，每层 BF16 `[T,C,D]` 额外占用 `2*T*C*D` 字节。forward 结束时收齐其余任务并清空字典。

两个额外 Hook 仅在预取开启时登记并检查，关闭时保持按层 lookup。允许与 decoder SWA bounded replay 同开：当前第 1、14 层位于裁剪边界前时，形状不受影响；边界后各层的预取结果尚未接入 tail 行映射。原并行、投机和 graph 限制不变。配置示例：

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_SSD_PREFETCH=1
export SGLANG_DSV41_ENGRAM_SSD_PREFETCH_LAYERS=14
```

## 原生模块构建

以下命令供目标 Linux 容器执行，本次未在本地编译：

```bash
cd /workspace/sglang_engram_ssd
cmake -S src/sglang_engram_ssd/query/csrc -B build/query \
  -DCMAKE_BUILD_TYPE=Release \
  -DENGRAM_SSD_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/query --parallel
```

产物包括 `build/query/libengram_ssd_query.a` 和源码包 `query/` 下的 `_C*.so`。使用 C++20、当前 Torch 的 ABI 参数及同一环境的 Python 开发头文件，构建结果需在目标容器确认。

当前 `pip install -e` 不自动调用 CMake。原生模块构建后，现有 editable 安装即可加载；修改 C++ 或更换 Python/Torch 环境后需重新构建。只构建静态库时，将 `ENGRAM_SSD_BUILD_PYTHON` 设为 `OFF`。

需要 `io_uring` 时，在目标容器安装 liburing 开发包，再重新配置构建。Debian/Ubuntu 示例：

```bash
apt-get update
apt-get install -y liburing-dev

cmake -S src/sglang_engram_ssd/query/csrc -B build/query \
  -DENGRAM_SSD_ENABLE_IO_URING=ON
cmake --build build/query --parallel
```

以上沿用前一条配置命令保存的 Torch/Python 路径及 Python 模块开关。`ENGRAM_SSD_ENABLE_IO_URING` 默认为 `OFF`；关闭时不依赖 liburing，原有两条 `pread` 路径仍可构建。

## Python 调用

以下为接口调用示例，表元数据、行号及 CPU 输出工作区由调用方准备：

```python
from sglang_engram_ssd.query import NativeBatchRowReader

reader = NativeBatchRowReader(
    tables=tables,
    cache_bytes=1024 * 1024 * 1024,
    num_threads=4,
    thread_pool_queue_depth=32,
    io_backend="pread",
)
weight_out, scale_out = reader.allocate_workspace(layer_id=layer_id, capacity=128)
reader.read_rows(
    layer_id=layer_id,
    row_ids=row_ids,
    weight_out=weight_out,
    scale_out=scale_out,
)
reader.close()
```

示例中的 `row_ids` 为 CPU 连续 `int64` 张量，长度不超过 128。后续批次和相同行宽的层可复用 `weight_out`、`scale_out`。接口只负责 CPU 读取；服务中的 H2D、反量化和 CUDA 同步仍由 runtime 执行。

选择 `io_uring` 时，构造如下，后续工作区和读取调用不变：

```python
reader = NativeBatchRowReader(
    tables=tables,
    cache_bytes=1024 * 1024 * 1024,
    io_uring_queue_depth=32,
    io_backend="io_uring",
)
```

服务通过环境变量选择后端，无需手动构造 reader。例如，在原启动命令前设置：

```bash
# 使用 C++ 线程池；缓存预算和 CHUNK_ROWS 沿用原配置。
export SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND=cpp_thread_pool
export SGLANG_DSV41_ENGRAM_SSD_NUM_THREADS=4
export SGLANG_DSV41_ENGRAM_SSD_THREAD_POOL_QUEUE_DEPTH=32
```

改为 `cpp_io_uring` 时使用 `SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH`，无需清零线程池参数；改为 `python` 使用原 Python 串行读取。后端在 runtime 初始化时确定，修改后需重启服务。
