# SGLang Engram SSD 插件

将研究文档第三章的 eager SSD 方案实现为独立插件。基于本仓库 SGLang commit `baa9befa381511cd4d56f83980951e5a45fffc6d`；所有新增文件都在本目录。

本地 SGLang 的官方装饰器名为 `plugin_hook`，插件通过 `plugin_hook as register_hook` 导入，因此代码中的 `@register_hook(...)` 使用的是官方 HookRegistry。安装后由 `sglang.srt.plugins` entry point 自动发现，每个 TP worker 各登记和应用一遍。

## 安装与启动

在已经安装本地 SGLang 及其 CUDA 依赖的 Python 环境中执行：

```bash
python -m pip install --no-deps --no-build-isolation -e ./sglang-engram-ssd
```

启动示例，模型路径指向现有的本地 SSD checkpoint：

```bash
SGLANG_PLUGINS=engram_ssd \
SGLANG_ENABLE_DSV41_ENGRAM_SSD=1 \
SGLANG_DSV41_ENGRAM_SSD_LOOKUP_MODE=all_queries \
SGLANG_DSV41_ENGRAM_SSD_CACHE_BYTES=536870912 \
SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS=4096 \
python -m sglang.launch_server \
  --model-path /path/to/DeepSeek-V4.1-Flash \
  --trust-remote-code \
  --language-model-only \
  --device cuda \
  --tp-size 4 \
  --load-format safetensors \
  --startup-weight-load-mode serial \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --host 127.0.0.1 \
  --port 30000
```

`SGLANG_PLUGINS` 是可选的插件白名单；需要同时启用其他插件时按 SGLang 规则追加名称。TP=4 仅为示例，插件没有固定 TP 数量，也不限制节点数。当前范围为 NVIDIA CUDA、普通 TP、纯文本、非投机 prefill/decode；不支持 Attention DP/CP、PP、PD 分离和 CUDA graph。保留原 HTTP 接口及 CPU/GPU overlap 调度。

使用包含视觉配置的 V4.1 checkpoint 时，必须添加 `--language-model-only`，跳过视觉模块并使用文本启动预热。否则默认图片预热会触发插件的 `Engram SSD currently supports text-only requests` 检查。

`SGLANG_ENABLE_DSV41_ENGRAM_SSD` 默认为 0，关闭时插件不登记 hook。新增环境变量由插件自己的 `config.py` 读取，无需修改 SGLang `environ.py`。不要启用整 checkpoint prefetch、startup overlap、IPC weight cache、custom loader 或 `weight-loader-disable-mmap`；插件会在模型加载前报告不支持的配置。

SSD 开启时独立接管 Engram，不再要求 `SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1` 或 `SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared`；原 HOST_TABLE 开关和 DDR 布局不参与 SSD 路径选择。SSD 关闭时，由 SGLang 按原配置选择 HBM、DDR shared 或 DDR per_rank。

`SGLANG_DSV41_ENGRAM_SSD_LOOKUP_MODE` 默认为 `all_queries`，表示每个 rank 处理全部查询；也可选择 `owner_allreduce`，由各 rank 读取所属行并在模型到达对应 Engram 层时归约结果。其他模式会在模型加载前报错。启动日志显示实际 `lookup_mode`，修改配置需重启服务进程。

插件入口只识别明确关闭值，其余情况先登记 Hook。完整配置校验在模型加载 Hook 的开头执行；非法开关、不支持的查询模式、非整数预算、负预算或非正的 `CHUNK_ROWS` 会在原模型加载函数调用前抛错，阻止模型加载。配置解析错误不再发生于会被 SGLang 捕获并继续的插件注册阶段。

## TP 查询模式

`LOOKUP_MODE=all_queries` 支持 TP=1 及 SGLang 模型允许的多 rank 规模。每个 rank 都处理全部 `indices[T,C]`，用全表 row_id 查询并得到完整 `[T,C,D]`，不按 TP 切分 Engram 行，也不要求表行数整除 TP。该模式不增加 Engram all-reduce 或 all-gather；模型其他部分的 TP 通信照常执行。

沿用上面的 `all_queries` 完整启动命令，切换 TP 时只修改 `--tp-size`：

| 配置示例 | 含义 |
| --- | --- |
| `--tp-size 1` | 单 rank 全查询。模型权重、算子和显存仍须满足 SGLang 要求，并不表示该巨型模型一定能放入单卡。 |
| `--tp-size 8` | 8 个 rank 各自执行全查询；其他允许的 TP 规模同理。 |

三个读取后端、`PIPELINE` 和按层预取均在每个 rank 独立生效。`CACHE_BYTES=B` 是每 rank 的缓存 payload 预算；TP=P 时合计上限约为 `P*B`，不包含模型其他内存、工作区、索引对象或系统 page cache。CUDA 设备由 SGLang 绑定，不能假定 TP rank 等于 GPU 编号；启动日志中的 `tp_size` 和 `cuda_device` 分别显示实际 TP 规模与当前设备。

跨机普通 TP 由 SGLang 负责通信和设备绑定，每个节点须在本地 SSD 准备相同版本的完整 checkpoint。未扩展 EP 等其他并行方式；当前仅完成源码静态检查，实际多机运行尚未验证。

切换 owner 模式时，在相同启动命令前设置：

```bash
export SGLANG_DSV41_ENGRAM_SSD_LOOKUP_MODE=owner_allreduce
```

所有 TP rank 使用相同的 `LOOKUP_MODE` 与模型配置，保持各 Engram 调用点的集体通信一致。

每层总行数为 N、TP 规模为 P 时，rank r 负责 `[N*r//P, N*(r+1)//P)` 的连续全局行区间。N 不必整除 P，P>N 时允许空区间。runtime 在 GPU 展平 `[T,C]`，筛出所属行的原查询位置 `positions[M]`，再用全局 row_id 查询紧凑结果 `[M,D]`。该查询直接复用原 reader、缓存、分块、H2D 和 A/B gather；重复查询不去重，结果按原位置回填完整零张量 `[T,C,D]`。只划分读取责任，不切割或导出 checkpoint。

Embedding 在模型线程的实际 Engram 调用点，对完整局部输出执行一次 TP all-reduce，不按 chunk 归约。没有本地命中的 rank 也携带完整零张量参加；TP=1 直接走原全查询，全局空输出跳过通信。后台预取只产生未归约的局部结果，模型到层消费后统一归约，避免后台通信与 Attention/MoE 的集体调用交错。三个读取后端、`PIPELINE` 和按层预取均可组合。

owner 模式仍保留完整 `[T,C,D]` 输出，并增加紧凑 `[M,D]` BF16 结果、位置和行号临时张量；完整输出的 all-reduce 还有通信与缓冲成本，不能据此假定 P 倍加速。当前仅完成静态检查，实际 TP 与跨节点运行尚未验证。

## 查询后端选择

`LOOKUP_MODE` 决定每个 rank 处理全部查询还是所属行查询；`QUERY_BACKEND` 决定这些查询由哪种 CPU 读取实现执行，两者分别配置。

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND` | `python` | 支持下表中的三个后端，默认保留原 Python 串行路径。 |
| `SGLANG_DSV41_ENGRAM_SSD_NUM_THREADS` | `4` | 仅 `cpp_thread_pool` 使用，控制工作线程数，必须为正。 |
| `SGLANG_DSV41_ENGRAM_SSD_THREAD_POOL_QUEUE_DEPTH` | `32` | 仅 `cpp_thread_pool` 使用，限制等待执行的行任务数，必须为正。 |
| `SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH` | `32` | 仅 `cpp_io_uring` 使用，限制在途文件读取数，必须为正；weight、scale 各占一个请求。 |

只解析和校验当前后端使用的参数。其他后端的专属参数若显式设置，只输出 warning 并忽略；即使值不是整数或为负数，也不会阻止启动。未设置的不适用参数不产生 warning，Python 后端不使用以上三个读取参数。

旧 `SGLANG_DSV41_ENGRAM_SSD_QUEUE_DEPTH` 已停用，设置后会 warning 并忽略；请改用对应后端的专属参数。

| `QUERY_BACKEND` | 执行方式 |
| --- | --- |
| `python` | 原 Python 行缓存、串行逐行读取和 pinned 工作区填充，无需构建原生扩展。 |
| `cpp_thread_pool` | C++ 有界线程池并发读取，线程数由 `NUM_THREADS` 控制。 |
| `cpp_io_uring` | C++ 批量提交和收集 io_uring 请求，不启用插件线程池。 |

例如，在启动命令前选择线程池：

```bash
export SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND=cpp_thread_pool
export SGLANG_DSV41_ENGRAM_SSD_NUM_THREADS=4
export SGLANG_DSV41_ENGRAM_SSD_THREAD_POOL_QUEUE_DEPTH=32
```

选择 io_uring：

```bash
export SGLANG_DSV41_ENGRAM_SSD_QUERY_BACKEND=cpp_io_uring
export SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH=32
```

此前设置的线程数和线程池队列参数会被 warning 后忽略，无需改成 0。选择原 Python 串行读取时，将 `QUERY_BACKEND` 改为 `python`。选择原生后端前，按 [原生模块构建说明](src/sglang_engram_ssd/query/README.md#原生模块构建) 编译扩展；io_uring 还需启用 liburing。扩展缺失或后端初始化失败时直接报错，不自动回退。启动日志分别显示线程数、线程池队列深度和 io_uring 深度，不适用项为 `None`；修改配置需重启服务。

## 块间流水

`SGLANG_ENABLE_DSV41_ENGRAM_SSD_PIPELINE` 默认 `0`，三个读取后端均可使用。关闭时保留一套 CPU pinned/GPU 工作区，每块在 CPU 读取后，在初始化时确定的流上执行 H2D 和 gather 并等待。开启方式：

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_SSD_PIPELINE=1
```

开启后使用 A/B 两套工作区，CPU 填充下一块时，GPU 可以继续处理上一块。H2D 使用独立的 `copy_stream`，gather 固定使用初始化时确定的 `gather_stream`，通过事件控制数据就绪和工作区复用。每套都包含 weight、scale；最终 BF16 输出和只读 `device_ids` 仍各保留一份。实际重叠程度取决于设备和各阶段耗时。

`EngramSsdBuffer` 是简单的数据类，只保存工作区、`gather_stream`、`copy_stream` 和完成事件；分配与调度由 runtime 负责。runtime 初始化时为 A/B 分别新建独立的固定 gather 流，两者都与模型主流分开；后续 lookup 不重新绑定。开启流水时 A/B 共用一条独立拷贝流，lookup 返回前分别等待两条 gather 流完成；关闭流水时只有一套工作区，`copy_stream=gather_stream`。

仅开启块间流水时，调用线程仍按顺序读取各块，不增加后台读取线程；lookup 结束前排空本次 GPU 工作，保持同步返回。`PIPELINE` 只调整单次 lookup 内的分块调度，跨层提前查询由下面的独立开关控制。启动日志中的 `pipeline_enabled` 显示开关状态，修改后需重启服务。

## 按层预取

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_PREFETCH` | `0` | 在后台提前执行选中层的完整 SSD lookup。 |
| `SGLANG_DSV41_ENGRAM_SSD_PREFETCH_LAYERS` | 未设置 | 逗号分隔的模型层号，例如 `14` 或 `1,14`；未设置或为空时，选择全部 Engram 层。 |

在原启动命令前增加：

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_SSD_PREFETCH=1
export SGLANG_DSV41_ENGRAM_SSD_PREFETCH_LAYERS=14
```

关闭预取时按层调用原插件 lookup；显式设置 `PREFETCH_LAYERS` 只 warning 并忽略。开启后，每个 runtime 使用一个持久后台线程，直接调用同一个 `runtime.lookup()`，将整层形状的 BF16 GPU 输出保存在 Future 中；owner 模式保存尚未归约的局部结果。Embedding 到达该层时，完成则直接取结果，未完成则等待，随后统一调用 `reduce_output()`；未选中的层仍同步执行原插件 lookup，不回到 SGLang 原表。三个读取后端、现有行缓存和 `PIPELINE` 均继续生效。

本次 hash IDs 生成后，立即按层号顺序统一提交所有选中层的查询任务，不等待模型消费前层。一个后台 worker 复用同一套 A/B 工作区，任务排队逐层执行，统一提交不表示层间 lookup 并发。后台以 A 的流等待 hash 就绪并准备索引，各块由 A/B 各自的 gather 流执行；模型线程继续前面的 Attention/MoE。未预取层的同步 lookup 也使用 `runtime.lock`，若后台另一层正在占用工作区，则需等待该锁；实际重叠程度需在目标设备确认。

Future 完成时，本 rank 的该层查询输出已在 GPU 就绪，owner 模式的 TP 归约仍由模型线程执行；消费时用 `record_stream` 保护其在模型流上的使用寿命，并移除字典中的 Future。多个已完成但尚未消费的层输出可以累积，每层 BF16 `[T,C,D]` 额外占用 `2*T*C*D` 字节。没有额外的 pinned raw 块池，也不新增行缓存副本。

模型 forward 正常或异常退出时都会收齐后台任务；runtime 关闭时先停止预取线程，再关闭 reader 和文件句柄。原 hash/history 更新、wkv、gate 不变。允许预取与 decoder SWA bounded replay 同开：当前第 1、14 层位于裁剪边界前时，形状不受影响；边界后各层的预取结果尚未接入 tail 行映射。原 DP/CP、投机、PP、PD 分离和 CUDA graph 限制保持不变。修改配置需重启服务。

## 代码组织与 Hook 接入

| 文件 | 作用 |
| --- | --- |
| `src/sglang_engram_ssd/__init__.py` | 通用插件入口，启用后导入并登记 hooks。 |
| `config.py` | SSD、块间流水和按层预取配置，以及查询模式、读取参数、缓存预算和工作区容量。 |
| `catalog.py` | 原 index/header 解析、集中形状和编码校验、原始文件位置。 |
| `runtime.py` | 全查询或 owner 筛选回填、读取后端、A/B 流水、后台执行器及模型线程调用的结果归约。 |
| `prefetch.py` | hash 就绪时统一提交选中层、按层 Future 消费及 forward 任务回收。 |
| `query/` | 已接入 runtime 的 C++ 批量读取、LRU、pinned 工作区、线程池及可选 `io_uring`。 |
| `weight_loader.py` | 在 `get_tensor()` 前精确跳过 Engram 两张大表。 |
| `hooks/__init__.py` | 登记基础七个接入点；预取开启时额外登记模型与 Hasher Hook。 |
| `hooks/_registry.py` | Hook 目标及安装完整性检查。 |
| `hooks/model_loading.py` | 模型加载准备与加载器选择替换。 |
| `hooks/weight_loading.py` | 权重 iterator 与模型 `load_weights` 替换。 |
| `hooks/engram_embedding.py` | `EngramEmbedding` 构造、推理与加载完成替换。 |
| `hooks/model_forward.py` | AROUND Hook 管理本次 forward 的预取上下文与任务回收。 |
| `hooks/engram_hasher.py` | AFTER Hook 提交原 hash 结果，不重复 hash 或 history 更新。 |

| SGLang Hook 目标 | 接入动作 |
| --- | --- |
| `load_model_utils.load_model_with_memory_saver` | 在原模型构造和启动预取前验证配置、准备 catalog。 |
| `loader.get_model_loader` | 核对实际返回的 loader，防止量化配置选择其他加载路径。 |
| `DefaultModelLoader._get_weights_iterator` | 保留文件发现，改用提前过滤的选择性 iterator。 |
| `DeepseekV4ForCausalLM.load_weights` | 在原 dispatch 外补充 canonical 名称的外置参数过滤。 |
| `EngramEmbedding.__init__` | 创建空参数占位和共享 runtime，跳过完整 DDR 表分配。 |
| `EngramEmbedding.forward` | 按 layer_id 查询或消费 Future，再统一调用 `reduce_output()`；仅 owner 多 rank 非空输出进行归约。 |
| `EngramEmbedding.finish_load` | SSD 模式跳过完整 host 表的完成 barrier。 |
| `DeepseekV4Model.forward` | 仅预取开启时登记 AROUND Hook，建立上下文并在退出时收齐任务。 |
| `EngramHasher.forward` | 仅预取开启时登记 AFTER Hook，提交原 hash IDs 和就绪事件。 |

HookRegistry 会传播已导入的函数别名，因此 `ModelRunner` 之前导入的 `load_model_with_memory_saver` 也会指向 wrapper。在该入口 Hook 已成功应用的前提下，插件会检查其余必要 Hook；两个额外 Hook 仅在预取开启时纳入检查。入口 Hook 自身未安装或插件模块导入失败仍需查看 SGLang 插件加载日志。没有修改 SGLang 原文件；Hasher、history、wkv、gate 和 gather kernel 使用原实现。

## 数据路径与容量

启动解析原 `model.safetensors.index.json` 和相关 shard header，只将配置启用的 `model.layers.<id>.engram.embed.weight/scale` 加入目录。校验在 `build_engram_ssd_catalog()` 完成；catalog 保存 `layer_ids`，runtime 构造时循环 `init_table(layer_id)` 绑定句柄。`open_table(layer_id)` 在层号不存在时抛出带层号的 `KeyError`。

首版要求原生 `F8_E4M3` payload `[N,D]` 和 `F8_E8M0` scale `[N,D/32]`。U8 scale 不自动当作 E8M0 原始指数。weight 与 scale 可以位于不同 shard；每个 shard 的数据区起点分别是 `8 + header_len`，两张表各自保存绝对偏移。不会导出新表、转换权重或生成额外大文件。

`lookup()` 是统一查询入口，runtime 构造时按 `lookup_mode` 绑定实现。当前 `all_queries` 由 `_lookup_all_queries()` 执行，按原查询顺序处理 GPU `indices[T,C]`：同步 D2H → 每块最多 K 次查询 → 所选后端填充 pinned staging → 两次 H2D → 原 gather 反量化 → GPU BF16 `[T,C,D]`。不排序、不去重、不做 Engram lookup TP all-reduce。`device_ids=[0,K)` 仅是当前 GPU 工作区槽号；全局 row_id 只用于原文件读取。

Python 后端保留逐行查缓存、读取和工作区填充。原生后端在 runtime 初始化时创建一个跨层 `NativeBatchRowReader`，借用已打开的 fd，由 C++ 分配 pinned 工作区；每块直接传入连续 CPU `int64` 行号片段，不转 Python 列表、不填充 Python LRU。本块读取同步结束后，runtime 才提交其 H2D 和 gather；流水开启时，下一块读取可与已提交的 GPU 工作重叠。

设一行原始数据为 `R=D+D/32` 字节，工作区套数 `s` 在流水关闭时为 1、开启时为 2：每 rank 缓存 payload 预算仍为 B，pinned 工作区为 `s*K*R`，HBM 工作区为 `s*K*R+8*K`，每块 q 次查询的 H2D 仍为 `q*R`。缓存索引、对象、CPU 行号和完整 BF16 输出另计。流水不复制整张 Engram 表，不增加行缓存预算，也不复制最终 BF16 输出。Python 后端由 `self.cached_bytes` 计数，原生后端由 C++ `cached_bytes_` 计数；同一时刻只使用所选后端的行缓存。预算为零时仍保留有限 staging，按需读取但不保留 LRU 行数据。

各 rank 独立维护目录、LRU、fd 和 GPU 工作区。同一节点读取相同文件可能复用内核 page cache；应用层预算不限制全节点 page cache。同进程锁只保护跨层共享工作区的并发调用，不协调各 rank 的 I/O。流水开启时，CPU 工作区复用等待其 H2D 完成，GPU 工作区覆盖等待其 gather 完成；lookup 返回前完成本次工作。按层预取只改变 lookup 的发起时机，复用同一缓存与工作区，额外显存取决于已完成但尚未消费的各层 BF16 输出。

## 验证状态

本次仅按本地源码检查 Hook 接口、调用链和 Python 语法，未安装插件、导入运行 SGLang、执行测试、模型推理或性能测试。实际 checkpoint 编码、CUDA 数值一致性和吞吐收益仍需在目标 NVIDIA 环境验证。
