# 调度级预取交付说明

本次将 Engram SSD 生产读取路径收敛为同步 io_uring owner 查询，并新增默认关闭的 waiting 分段预取、就绪感知调度、pinned 行池直接消费与独立去重。正式 Hasher 的 history 更新和原层 TP all-reduce 位置保留。

依据工作区根目录的 `ENGRAM_SSD_CODING_GUIDELINES.md`、`ENGRAM_SSD_REQUIREMENTS.md` 实现。所有交付修改位于插件仓库；官方 `sglang/` 只作静态参考。

**实现与测试代码已交付；本地没有编译、运行 pytest、调用模型或服务。A0～A8 的运行验收均未执行，不能将静态检查视为通过。**

## 1. 开发基线与提交

- 分支：`feature/engram-ssd-io-uring`。
- 基线 tag：`baseline-before-ssd-optimization`，插件提交 `800741958e6544b0827accc847859fde8b49c0c8`。
- 官方源码参考：SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`，未修改。
- 没有推送远端或创建 PR。

按需求第 9 节的七步组织提交：

| 顺序 | 内容 |
|---|---|
| 1 | R0：删除旧读取分支、旧预取、流水与 reader 缓存；必需 io_uring／liburing；A0 测试 |
| 2 | R7：同步 owner 查询 unique／inverse，恢复原顺序及重复位置 |
| 3 | R1：捕获模型真实 hash 参数、CPU 段行号及 A2 测试 |
| 4 | R2：对齐 pinned 行池、引用／LRU、reader 按槽写入与 A3 测试 |
| 5 | R3／R4／R6：后台分段、TP 一致放行、队列恢复、结果生命周期 |
| 6 | R5：mapped pinned gather、同步 miss、CUDA event 临时引用与 A4／真实 TP8 测试 |
| 7 | R8／R9：配置校验、统计、文档与目标环境服务验收工具 |

查看实际记录：

```bash
git log --reverse --oneline baseline-before-ssd-optimization..HEAD
```

## 2. 需求与交付文件

| 需求 | 实现位置 | 测试／验证入口 |
|---|---|---|
| R0 | `config.py`、`runtime.py`、`query/`、原有 Hook | `tests/test_runtime.py`、`test_config.py` |
| R1 | `hashing.py`、`hooks/engram_hasher.py` | `tests/test_hashing.py` |
| R2 | `query/csrc/prefetch_row_pool.{h,cpp}`、`batch_reader`、绑定与包装 | `tests/test_prefetch_row_pool.py` |
| R3 | `sched_prefetch.py` 的登记、两段上限与唯一后台线程 | `tests/test_sched_prefetch.py` |
| R4 | raw 调度 AROUND、priority AFTER、整批 CPU MIN | 同上，含 grammar 当轮加入与队列重绑定 |
| R5 | `runtime.lookup_from_pool`、Embedding Hook | `tests/test_pool_lookup.py`、`test_distributed.py` |
| R6 | 请求 generation、不可变 batch span、取消／退出与 GPU lease 回收 | scheduler／pool lookup 两组测试 |
| R7 | `runtime._read_rows_to_output` | `tests/test_runtime.py` 的 A5 |
| R8 | `config.py`，README 配置表 | `tests/test_config.py`、`test_registration.py` |
| R9 | 原生计数、JSON 快照、条件计时与 idle／shutdown 归零状态 | 原生池、scheduler 测试及服务日志采集 |

文件名默认相对 `src/sglang_engram_ssd/`。新原生源文件已加入 CMake；需要重新构建扩展，旧二进制接口不兼容。

退役环境变量名仅在 `config.py` 的迁移警告表和对应迁移测试中保留，不再参与路径选择或数值解析。测试客户端的并发 HTTP 线程不属于插件 I/O 后端；生产代码没有线程池或 pread 读取路径。

## 3. 新增 Hook 接入

目标全部集中在 `hooks/_registry.py`，仅 `SCHED_PREFETCH=1` 时登记，并纳入安装检查。

| 目标 | 类型 | 作用 |
|---|---|---|
| `sglang.srt.layers.engram.EngramHasher.__init__` | AFTER | 从模型实际 Hasher 复制只读 CPU 参数 |
| `sglang.srt.managers.scheduler.Scheduler._add_request_to_queue` | AFTER | 只登记真实入队、未终止的纯文本请求 |
| `sglang.srt.managers.scheduler.Scheduler._get_new_batch_prefill_raw` | AROUND | 对账、补交和恢复完整队列 |
| `sglang.srt.managers.schedule_policy.SchedulePolicy.calc_priority` | AFTER | 原生 grammar 加入及排序之后统一过滤；整批至多一次 TP MIN |
| `sglang.srt.managers.scheduler.Scheduler.run_batch` | AFTER | 以 forward_iter 保存本次不可变区间与 generation |
| `sglang.srt.managers.scheduler_components.batch_result_processor.SchedulerBatchResultProcessor.process_batch_result_prefill` | AFTER | 按本次区间释放已消费段 |

相对需求文本直接包围 raw 函数的示意，增加两个必要辅助接入：

1. **priority AFTER**：raw 内部会加入 grammar-ready 请求、排序并重绑定 waiting 列表；在外层提前过滤会漏掉当轮新请求。外层 finally 保留 blocked、原函数剩余及期间新入队项。
2. **run_batch AFTER**：overlap 结果使用 `batch.copy()`，Req 的 prefix／end 可能已被下一轮修改。`forward_iter` 随 copy 保留，因此用其记录的 generation／span 回收，避免旧结果释放新请求引用。

没有复制原生调度函数，也没有修改官方源码。后台不提交 TP collective；就绪、超时都先形成候选向量，再经 CPU 组 MIN 决定放行。

## 4. 配置与读取接口

完整默认值见 [README](README.md#配置)。两个独立开关默认都是 0：

- `SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH`：R1～R6。
- `SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP`：R7。

调度预取关闭时，不登记六个新增 Hook，不建立 CPU hash 副本、行池、后台 reader 或后台线程；从属数值参数只警告并忽略。其他策略不解析显式 TIMEOUT_MS，未提交请求的防饥饿 deadline 使用默认 500ms。

原生协议见 [query/README](src/sglang_engram_ssd/query/README.md) 及 `query/interface.py`。本次新增按槽读取、池 acquire／resolve／release／cancel／统计接口。IO 复用同一 reader 实现；修正后前后台各持有一个独立实例，pool 不定位或关闭 FD。

池基址对齐 4096，行步长保存在池中；这仅预留后续接口，**本轮没有合并存储、O_DIRECT 或多盘实现**。当前仍从原 checkpoint 两个张量读取。

## 5. 待确认事项的静态结论

| 需求第 8 节事项 | 静态结论 | 运行状态 |
|---|---|---|
| 完整 token、prefix 与位置 | 入队保存 `origin_input_ids + output_ids` 快照；首次筛选或 raw 提前返回边界按当前 prefix 切段，不自行修改 KV 缓存；不同 priority 策略不保证已匹配前缀 | CPU／CUDA hash 一致性测试未运行 |
| waiting 队列别名 | raw 内有 grammar 追加、原地排序和 list 重绑定；以 priority Hook＋外层恢复处理 | CPU 替身测试已写，真实 scheduler 未运行 |
| 结果处理类及 overlap | 目标是 frozen／slots 的 `SchedulerBatchResultProcessor`；状态放 runtime，`batch.copy` 保留 prefix_lens、extend_lens、forward_iter | span／generation 测试已写，真实 overlap 未运行 |
| pinned GPU 读取性能 | 通过 CUDA 映射指针调用原 `engram_gather`，不另写反量化 kernel | 构建、映射、数值和性能均待目标机验证 |
| 线程与进程 | 普通 NVIDIA 路径每 rank scheduler 在进程内直接调用 TP worker／model runner，GPU 工作异步执行；插件另有唯一后台线程 | 实际锁争用、调度延迟未测 |
| CPU 集合通信组 | 使用已存在的 `scheduler.tp_group.cpu_group`，不是在后台新建或调用集合通信 | 一次 MIN 已静态核对，真实 TP8 调度未验证 |

prefix 缩小时，旧段 READY 不代表早于原覆盖起点的数据已就绪；此时按未覆盖请求的入队超时规则放行、同步补读，不误报完整命中。

池满的 acquire 不等待，跨层占槽失败会 cancel 尚未读的自建 PENDING。取消只影响本句柄新建行；共享失败行显式报错，最后引用归还后回收，不留下永久 PENDING。

## 6. 验收状态与目标命令

| 编号 | 已交付测试 | 运行结果 |
|---|---|---|
| A0 | 合成表与 Torch 参考、TP1／模拟 owner8、真实 torchrun TP8 | 未运行 |
| A1 | 独立进程 Hook 门控、配置关闭与非法值校验 | 未运行 |
| A2 | 随机长度、非零起点、段边界、官方 Torch／CUDA hash | 未运行 |
| A3 | 原生引用、LRU、池满、并发、取消；CPU 调度生命周期替身 | 未运行 |
| A4 | 全／部分／无命中、PENDING、空 owner、CUDA lease、真实 TP8 | 未运行 |
| A5 | 原顺序恢复、跨 chunk 重复、owner 去重 | 未运行 |
| A6 | 基线记录与优化服务实际输出 token ID 对比 | 未运行 |
| A7 | 并发 abort、长 prompt、真实 retract、全部 rank 空闲引用归零 | 未运行 |
| A8 | 客户端 TTFT／吞吐、原生日志和插件逐 rank 计数 | 未运行，无性能结论 |

本地已做：Python AST 解析、行长、Hook 参数签名静态对照、引用／同步／关闭路径审阅、`git diff --check`。没有导入执行项目来代替测试，也没有编译原生扩展。

目标环境先按 [query 构建说明](src/sglang_engram_ssd/query/README.md#构建) 重建，再执行 README 的 pytest／torchrun 命令。服务验收按 [SERVICE_ACCEPTANCE.md](tests/SERVICE_ACCEPTANCE.md) 操作，先记录同步基线，再重启为优化服务，避免同时加载两套模型。

服务工具不自动启动服务，不下载模型或 tokenizer。测试依赖缺失时显式 skip；原生扩展存在但 ABI／初始化错误时直接失败。真实 A7 缺少 retract 或全部 rank 的新 idle 归零快照时，不报告完整通过。

## 7. 指标口径与限制

- `submitted_rows` 只计实际交付 reader 的预取缺失行；取消占槽不是读取提交。
- `dedup_saved_rows` 计成功登记复用，包括同段重复与跨请求复用；不是物理字节节省的直接测量。
- `miss_read_ms` 是 forward 同步补读 wall time，包含等待 reader、H2D 与 gather，不是纯 SSD 时间。
- 原生统计逐 rank 保留，不把时间求和当作整机关键路径延迟。
- 行池预算是压缩 payload，元数据、对齐余量和其他工作区另计；READY 缓存占槽不等于引用泄漏。
- CPU hash 成本、页缓存条件、GPU 访问 pinned 内存性能和实际 waiting 深度仍需报告，不能从源码推断优化倍数。

## 8. 代码审查后的修正

针对工作区 `ENGRAM_SSD_CODE_ISSUES.md` 的逐项判断、处理和测试入口见
[CODE_REVIEW_RESOLUTION.md](CODE_REVIEW_RESOLUTION.md)。P1～P7、P9、P10 已修改实现及相关测试；P8 保留分配策略，等待目标环境确认分配器的实际开销。

N1 删除 forward 与调度中的本地后台异常抛出，在有 prefill 需求的 raw 调度中统一归约失败。正常筛选附带失败标志；有 waiting／chunked 请求但提前返回时只归约该标志，每次至多一次。仅有 chunked 请求、候选为空时仍需失败归约，此处按用户 N1 建议调整原规范第 8.2 节及需求 R4.3 的空候选通信约束，不增加第二次通信或新配置。

N4 在原生 waiting 为空且没有 `chunked_req` 时跳过归约和第二次对账，消除纯 decode 的额外 CPU collective；原生 raw 与首次对账保留，避免漏掉内部 grammar-ready 入队及 GPU 引用回收。不能用本地 `states` 决定跳过，因为后台失败后它不保证跨 rank 一致。积压后台错误保留至下一次 prefill 需求统一传播；前台 I/O／CUDA 错误仍直接传播。

N2 在 raw 提前返回后为未切段 waiting 请求提交预取，恢复 batch 满时的排队重叠窗口；仍可能重复读取当前尚未匹配的缓存前缀。N3 保持无 chunk 长 prompt 的入队 deadline 补读，并明确 `wait_complete` 在超过两段容量时不保证等待全量完成。

按更新后的规范第 10 节补全生产代码的中文类／函数 docstring、张量约定、状态转换及关键调用注释。没有新增配置或 Hook，也没有改变默认关闭的约束。修正仍仅完成静态检查；新增的原生读取、调度和 CUDA 回归测试均未运行，不据此声明性能提升或多 rank 验收通过。
