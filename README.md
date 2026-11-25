# SGLang Engram SSD 插件

基于官方 SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`，通过 HookRegistry 接入，所有修改均位于插件仓库。

## 固定读取路径

每个 TP rank 负责 `[N*r//P,N*(r+1)//P)` 的全局行区间。走同步路径时，模型到达 Engram 层后通过 io_uring 读取所属行，单套 pinned／GPU 工作区串行分块完成 H2D 和原生 `engram_gather`，再将局部 BF16 结果放回完整零张量。模型线程在原 Engram 调用点执行一次 TP all-reduce；空 owner 也参加，TP=1 不通信。

原 checkpoint 保持不变，weight／scale 的定位仅由 reader 解释。插件不分配完整 host table，不依赖原生 HOST_TABLE 布局选择。

## 构建与安装

先在目标 Linux NVIDIA 环境按 [原生读取模块](src/sglang_engram_ssd/query/README.md) 构建，再执行：

```bash
python -m pip install --no-deps --no-build-isolation -e ./sglang-engram-ssd
```

原生扩展、liburing 和可初始化的 io_uring 是必需依赖，失败直接报错，不自动回退。

## 配置

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD` | `0` | 总开关；关闭时不登记插件 Hook |
| `SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS` | `4096` | 同步中转区容量和单个后台读块的行数上限，必须为正 |
| `SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH` | `32` | 每个 reader 的在途文件请求上限，必须为正；weight、scale 各占一个请求 |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP` | `0` | 对整次本 rank 查询去重，读取唯一行后恢复重复位置 |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH` | `0` | 启用窗口预取与批次显存 gather |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_WINDOW_BATCHES` | `2` | 预取窗口的预测批数，必须为正 |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POOL_BATCHES` | 未设置 | 按批数设置行池，必须大于窗口批数 |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POOL_BYTES` | 未设置 | 每 rank 行池字节预算，与 `POOL_BATCHES` 二选一 |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_LOG_INTERVAL` | `0` | 每 N 次 prefill 打印统计，0 关闭 |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEBUG_CHECK_ROWS` | `0` | 随调度预取启用，逐位核对已登记 prefill 批次的 CPU 预测与 GPU hash |

去重在同步内部读取入口完成，保留原顺序、形状和 BF16 数值；关闭时不分配 unique／inverse 张量。

CPU 行号计算直接调用官方 `compute_engram_hash_ids`，参数从模型已构造的 Hasher 复制，不重建 tokenizer，也不提前更新请求 history。其 Hook 仅随调度预取总开关登记。

行池保存两张连续 CPU pinned 表，起点均 4096 对齐，每槽保存 `D + D/32` 字节。每批按 `token 预算 × Engram 层数 × 哈希列数 ÷ TP × 1.1` 向上取整到整行。两种容量配置均未指定时，按 `WINDOW_BATCHES + 1` 批分配。启动时校验行池的实际整行容量严格大于窗口所需行数。

同一层／行只占一个槽，零引用 READY 槽按 LRU 复用，PENDING 和有引用槽保留至读取或使用结束。行池配置以数据字节计，进程另有池元数据、对齐和批次暂存开销。

调度预取关闭时忽略其从属参数及行号核对开关，显式设置会警告。启用调度预取才创建行池、后台 reader、后台线程、CPU hash 副本及批次双缓冲。4096 对齐仅预留后续读取接口，本轮仍从原 checkpoint 正常读取，不包含 O_DIRECT 或合并格式。

旧版读取选择、查询模式、按层预取、流水和 reader 缓存配置均已退役；显式设置只在启动时警告一次并忽略。配置只由 `config.py` 解析，非法值在原模型加载前拒绝。

旧段长、调度策略、读取预算和暂缓时长配置也已退役，显式设置时警告并忽略。

## 启动

### 调度预取流程

请求进入 waiting 后登记纯文本 token 快照。每轮原生 `calc_priority` 排序后，用当时可见的 prefix 和运行中请求数推算预取窗口。每批先放已有续段，再按队列顺序填充 `chunked_prefill_size`，被截断的剩余部分转入下一批首位。未启用分块时用 `max_prefill_tokens` 组批，每个请求保留完整的剩余区间。原生 raw 提前返回时，用当前 prefix 和队列顺序重算窗口。

原生优先级策略不一定在这一阶段匹配前缀，提前返回时 prefix 也可能仍为零，因此仍可能重复读取部分已缓存前缀。发起批次时使用原生组批后的实际前缀长度与 extend 区间。

每 rank 只有一个持久后台线程，使用独立 `prefetch_reader` 读取本 rank 的缺失行；forward 的 `reader` 使用另一套 ring／锁，两者共用 runtime 持有的只读 FD。每套 ring 在请求完成后持续补交，按各自配置的队列深度限制在途数量，最多合计两套在途队列。独立 ring 消除前后台锁等待，实际仍竞争 SSD 带宽。

后台任务按推算的 token 区间和 Engram 层分别排队。优先级键为 `(级别, 预测批次序号, Engram 层序号, 批内顺序)`，按字典序从小到大读取。紧急读取优先，窗口内依次读取 N+1 第一层、N+1 第二层、N+2 第一层、N+2 第二层。后台每读完最多 `CHUNK_ROWS` 行便重新选队首。

移出窗口或区间变化的段每轮撤销，在途块收尾，READY 行归还引用并留作缓存。区间变化时按新区间建段。池满的窗口段保留行号，后续轮次有槽时再试。

调度固定采用 `best_effort`。priority AFTER Hook 读取原生排序后的队列来推算窗口，队列和组批由原生调度器管理。发起批次时提交缺行的紧急读取。`run_batch` AFTER Hook 标记双缓冲的 forward 结束，并按 `forward_iter` 保存 `(Req, start, end)` 快照。撤回重登时删除该请求的旧段与批次区间。

在 TP 大于 1 时，waiting 非空或存在 `chunked_req` 的 raw 调度对一个后台失败标志执行一次 CPU gloo `all_reduce(MIN)`。归约在原生排序后进行，原生提前返回时由外层 Hook 补做。纯 decode 跳过归约。

是否通信只依赖原生 waiting／chunked 状态，不依赖本地 `states`、后台任务或失败标志；本地后台失败后 `states` 可能与健康 rank 不同。失败标志为 `-1` 时归约后各 rank 统一报错，本 rank 保留原始异常原因。

后台失败时，已发起的批次用独立的前台 reader 补读。下一次 waiting／chunked 需求仍通过原调度归约统一报告后台错误。后台线程只处理本 rank 数据，模型继续在原 Engram 层执行 TP all-reduce。

abort、完成和 retract 释放请求引用，在途 I/O 返回后回收槽位。关闭时停止提交、收齐后台和 CUDA 工作，再关闭行池、reader 与 FD。

### 紧急行请求接口

`controller.request_urgent(rows_by_layer=rows, slots_by_layer=slots, staging=(weight, scale), priority=(URGENT, batch, layer_order, order))` 接收每层唯一的 CPU int64 owner 行号、对应的暂存下标，以及调用方持有的 CPU pinned uint8 `[K,D]`、`[K,D/32]` 表。下标在各层之间全局唯一。

READY 槽固定后由组装方复制，PENDING 和缺失行通过 io_uring 直接读到指定暂存下标。控制器在完成组装后归还池引用。前台需要等待时调用 `controller.wait_urgent(request=..., timeout=...)`，后台停止且在途块收齐后，该调用用前台 reader 补读剩余块。暂存表由调用方保留至 CUDA 使用结束。

### 服务命令

```bash
SGLANG_PLUGINS=engram_ssd \
SGLANG_ENABLE_DSV41_ENGRAM_SSD=1 \
python -m sglang.launch_server \
  --model-path /model \
  --trust-remote-code \
  --language-model-only \
  --device cuda \
  --tp-size 8 \
  --load-format safetensors \
  --startup-weight-load-mode serial \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled
```

当前只支持 NVIDIA CUDA、普通 TP、纯文本、非投机 prefill／decode；不支持 attention DP／CP、PP、PD、CUDA Graph。模型必须为本地 safetensors；禁止整 checkpoint 预取、startup overlap、IPC weight cache、自定义 loader 和禁用 mmap 的加载配置。

## 批次组装与显存 gather

- **Prefill** 在 `Scheduler.run_batch` 前用 `prefix_lens` 和 `extend_lens` 得到实际区间。精确匹配窗口段时复用行号，其余区间重新计算。每层按 owner 过滤、去重并建立 int32 查询映射，非本 rank 位置为 `-1`。发起时固定 READY 行并撤销重叠窗口段，在途块照常收尾。后台填好暂存区后在拷贝流一次 H2D，计算流等待该 event。若 event 尚未记录，首个 Engram 层等后台发起 H2D。
- **Decode** 在 `EngramHasher.forward` 返回后，将 GPU 行号从旁路流拷回 pinned 内存。H2D 提前登记，后台按实际行号写完暂存区后置位唯一的放行标志，计算流只等 H2D event。交接前的失败由 forward 补读，交接后的失败由后台使用前台 reader 补读。同步补读也失败时在读取线程报告错误，本 rank 退出。
- **双缓冲** 启动时按 `max(chunked_prefill_size, max_prefill_tokens) × L × C` 分配两套 pinned／GPU 缓冲，分别保存映射、weight、scale。超出容量时按需扩容。复用前等待上一次 forward 的结束 event。分配失败时警告并走同步查询。
- **Forward** 已登记批次用原生 `engram_gather` 读显存，`-1` 输出零，保持 BF16 形状和原 TP 合并。prefill 的 GPU hash 继续更新 history，开启调试开关才拷回行号作逐位核对。mixed、未登记的模型调用和未组装的查询形状走同步路径。设备启动探测不支持流内存操作时警告一次，decode 走同步路径。

## 验证状态

本地 macOS 仅做源码静态检查，未编译、未运行 pytest 或模型。测试位于 `tests/`，原生和 CUDA 测试在目标环境执行；实际 TP8 数值、端到端和性能结果仍待验证。

在已构建新扩展的目标环境执行单元测试：

```bash
cd /workspace/sglang_engram_ssd
python -m pytest -q tests/test_config.py tests/test_registration.py tests/test_hashing.py \
  tests/test_io_uring_reader.py tests/test_prefetch_row_pool.py tests/test_sched_prefetch.py \
  tests/test_runtime.py tests/test_pool_lookup.py
torchrun --standalone --nproc-per-node=8 -m pytest -q tests/test_distributed.py
```

TP8 测试使用合成表和真实 NCCL，不需要加载完整模型。单 GPU 的 owner 模拟测试不等于真实多 rank 验证。
服务端 A6～A8 按 [验收命令](tests/SERVICE_ACCEPTANCE.md) 执行，原路径和优化路径分开启动，最多同时驻留一套模型。

## 统计日志

`LOG_INTERVAL=N>0` 时每 N 次 prefill 输出 `Engram SSD sched prefetch stats: {JSON}`。

| 字段 | 口径 |
|---|---|
| `submitted_rows` | 实际交付 reader 的预取缺失行数；仅占槽后取消不会计入，不含同步补读 |
| `dedup_saved_rows` | 成功登记的输入行数减新占行数，包含段内重复与已有行复用；取消后不倒扣 |
| `forward_hits`／`forward_misses` | 组装时本 rank 原始查询位置数，分别按已就绪和缺失计数 |
| `window_hit_rate` | prefill 发起时已在行池就绪的本 rank 唯一行占比 |
| `miss_read_ms` | 后台失败后同步补读和组装的 wall time，包含前台 reader 等待 |
| `occupied_slots`／`referenced_slots` | 当前占用／有引用槽位数；READY 缓存可占槽但引用为零 |

日志带 `tp_rank` 和 `phase=prefill/idle/shutdown`；空闲引用归零记录每生命周期只打印一次，关闭时再取最终快照。
验收 A7 要看全部 rank 的本轮 idle 归零状态，不能把中途快照或单 rank 结果当作全局无泄漏证明。

完整交付范围、Hook 列表与待验证事项见 [DELIVERY.md](DELIVERY.md)。旧工作区基准脚本针对基线接口，需在基线 tag 使用；新分支使用插件内的验收脚本。
