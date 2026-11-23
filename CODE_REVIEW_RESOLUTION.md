# 调度级预取审查问题处理记录

针对工作区根目录 `ENGRAM_SSD_CODE_ISSUES.md` 核对源码并修正。只修改插件，保留工作区已有的 C++ 中文注释修改；未增加配置或 Hook，未修改官方 SGLang。

## 1. 问题判断与处理

| 编号 | 判断 | 本轮处理 |
|---|---|---|
| P1 | 成立；整段读取共用 reader 锁会挡住 forward 补读，具体阻塞毫秒数待测 | 开启调度预取时创建独立 `prefetch_reader`，各自持有 ring／锁，共用 runtime 的只读 FD；关闭预取不创建第二实例 |
| P2 | 成立；整批收齐才补交会出现队列空档，但不能据此认定是此前吞吐差距的唯一原因 | 改成滑动窗口；顺带收回已到达 CQE 后补交，不等待同批未完成请求；短读／EINTR 重提剩余部分，错误后先收齐在途请求再抛错 |
| P3 | 成立，但 `calc_priority` 并非所有策略都会刷新 prefix | 首次筛选按可见 prefix 切段；raw 提前返回时也为未切段 waiting 请求启动预取，不丢掉排队窗口 |
| P4 | 成立；仅检查覆盖起点的一段不能证明整个 chunk 就绪 | 检查下一 chunk token 上界涉及的全部段及完整覆盖；保留两段上限，无法完整预取时按既有入队 deadline 放行并补读 |
| P5 | 成立；仅按调度次数重试会反复计算 hash | 池满段缓存 CPU 行号，引用释放后推进 `release_epoch` 才能重试；包括 GPU event 完成后的临时引用释放；自身失败回滚不触发自身重试 |
| P6 | 成立；原语句同步当前 CUDA 流，并非全设备同步 | 删除 lookup 末尾额外流同步；保留保护中转区的 chunk event；输出恢复及下游消费仍在同一流上有序执行 |
| P7 | 成立；只改 filter 内的检查还不够 | 删除本地后台异常抛出，候选归约附带失败标志；有 prefill 需求的 raw 早退仍归约，无 waiting／chunked 时跳过，每次 raw 至多一次 MIN |
| P8 | 分配器开销待验证；“4095B 导致默认额外约 60MB”的归因不成立 | 保留现有显式对齐及 mapped pinned 分配；补充注释，暂不替换为自管 `cudaHostAlloc` |
| P9 | 成立；输出在当前流分配和使用，两个标记冗余 | 删除两处 `out.record_stream`；保留池槽的临时引用、索引张量与 CUDA event 生命周期保护 |
| P10 | `_queue_entered` 的风险成立；`states` 原本已持有请求引用 | 时间记录改为包含请求强引用的 `QueueEntry`，对账时清理；继续用对象身份＋generation 隔离请求，不改用可能重复的 rid |
| N4 | 纯 decode 的额外归约成立；本地 `states` 不保证跨 rank 一致 | 只按原生 waiting／chunked 判断，无 prefill 需求时跳过归约和第二次对账，有新需求时恢复失败传播 |

## 2. 前后台读取与关闭顺序

forward 的 `reader` 和后台的 `prefetch_reader` 各有一个 ring，每个 ring 的在途上限均为 `IO_URING_QUEUE_DEPTH`，不是两个实例合计共用这一上限。两个实例消除了软件互斥等待，仍会竞争 SSD、页缓存和主机资源，吞吐改善需实测。

关闭顺序为：停止后台提交并等待线程及在途 I/O → 收齐 GPU 临时引用 → 释放行池 → 关闭两个 reader → 关闭 runtime 的 FD。reader 只借用 FD，不关闭它们。

## 3. 分段与调度的边界

原生 raw 调度在 batch 已满、请求槽不足等情况下，可能在调用 `calc_priority` 之前返回。本轮 N2 修正会在该返回边界，按当前 prefix 为尚未切段的 waiting 请求启动预取；正常筛选则使用 priority 阶段可见的 prefix。这样恢复排队期间的读取窗口，调度线程仍只提交任务，CPU hash 和 I/O 由唯一后台线程完成。

某些 priority 策略不会在此时匹配 prefix；原生后续 `init_next_round_input` 也可能再次匹配。插件仅使用当前快照，不能保证完全消除 prefix 重复读取。最终 forward 始终按真实行号消费，缺失行同步补齐。

P4 使用 `chunked_prefill_size` 作为下一 chunk 的长度上界；原生调度还可按剩余 token 预算缩短它。显式段长过小、一个 chunk 涉及三段以上时，仍不突破每请求两段上限，沿用入队超时补读，避免 `wait_complete` 永久等待。未启用 chunked prefill 时，上界为整个剩余 prompt；超过两段容量时同样按入队 deadline 放行，`wait_complete` 不保证等到全量预取完成（N3）。

N1 删除 forward、raw 提前返回及空候选边界的单 rank 后台异常抛出。正常筛选把失败标志与候选向量合并；有 waiting／chunked 请求但提前返回时只传失败标志。归约后各 rank 统一报错，不按本地失败状态决定是否参加通信。

N4 修正纯 decode 的额外开销：waiting 队列为空且没有 `chunked_req` 时，不做失败归约，也不做早退分支的第二次对账；首次对账仍保留，以收集 GPU event 和处理生命周期。原生 raw 必须照常调用，避免漏掉它内部加入的 grammar-ready 请求。

没有直接采用 `not self.states` 作为跳过条件：本地后台失败后，登记可能提前返回、对账也可能暂停，因此失败 rank 的 `states` 与健康 rank 不保证一致。通信分支只使用原生 waiting／chunked 状态，不查看本地后台任务、失败或池状态。

这保留每次 raw 至多一次归约，不新增周期配置。有 chunked 请求但无 waiting 候选时仍会通信；纯 decode／空闲期间保留积压错误，直到下一次原生 prefill 需求出现再统一传播。失败若发生在本轮归约之后，当前 forward 仍可消费 READY 行并同步补齐 miss。

生产路径中只有一个后台线程 acquire PENDING 槽，下一段占槽前当前段 I/O 已完成；forward 的 resolve 仅为 READY 加引用。因此后台失败先收齐在途 I/O 再 cancel 当前段，不会使其他生产句柄持有失败的 PENDING 槽。原生池对外仍保留共享失败槽显式报错的契约，不据此放宽 pool 接口。

前台补读本身的 I/O 错误、CUDA 错误、任意进程退出或各 rank 调度流程分歧不属于后台失败标志修复的保证范围。真实 TP8 的调用顺序与无挂起退出仍待目标环境验证。

## 4. pinned 分配的待验证结论

两张表显式多申请的对齐余量合计为 `2×4095=8190B`，分配器的取整、缓存和元数据开销另计。默认 2GiB payload、D=256 时，两张表的有效数据约为 1985.94MiB 与 62.06MiB。即使假设分配器按二次幂取整，删除 4095B 后也仍可能分别取到 2048MiB 与 64MiB，不能认定删除余量即可解决。

实际 Torch pinned 分配器的行为、驻留内存及 GPU 映射需要在目标环境确认。当前保持已定义的 4096 对齐契约，不依赖尚未确认的分配器对齐行为，也不引入另一套内存所有权实现。

## 5. 回归测试与验证状态

| 测试文件 | 补充覆盖 | 状态 |
|---|---|---|
| `tests/test_io_uring_reader.py` | 超过 QD 的读取、重复行、非对齐偏移、短读到 EOF、CQE 错误、错误后复用与旧输出保护 | 未运行 |
| `tests/test_pool_lookup.py` | 后台持锁时前台补读、第二 reader 门控及关闭、实际 GPU 引用释放后通知、后台失败后 forward 不抢先抛错 | 未运行 |
| `tests/test_runtime.py` | owner 与空 owner 路径不额外同步当前流，保留 chunk event 保护 | 未运行 |
| `tests/test_sched_prefetch.py` | prefix 快照、满 batch／grammar-ready 预取、跨段就绪、无 chunk deadline、池满重试、活跃需求一次失败归约、纯 decode 零归约及积压失败恢复、请求身份清理 | 未运行 |
| `tests/test_distributed.py` | 真实 TP8 Gloo：有／无候选时，仅一个 rank 持有后台错误，各 rank 经同一次 MIN 报错 | 未运行 |

本地仅做 Python AST、行长、差异格式和调用／资源生命周期静态检查，没有编译扩展、运行 pytest、模型或服务。静态检查不能证明 CUDA／io_uring 运行正确或性能改善。

目标环境需重建扩展，再按 [README](README.md#验证状态) 执行测试。EINTR、submit／wait 故障注入、真实 TP8 后台失败传播、allocator 开销及吞吐收益仍待验证。

## 6. 中文注释补全

按更新后的代码规范第 10 节，补齐 `src/` 中类与函数的中文职责、输入输出、生命周期和线程约束，张量标明设备、dtype、形状、连续性及行号含义。调度状态机、放行条件、mapped gather、reader 调用、CUDA event 与资源回收分别补充关键行说明。

C++ 实现文件补首行职责说明，匿名辅助函数、滑动窗口 lambda 及私有类型补充输入输出、锁前提与副作用。文档改动与行为修复分开核对，避免以“有 docstring”代替对实际契约的审阅。

静态检查确认 `src/` 的 20 个类及 99 个函数／方法均有 docstring，函数均包含输入与输出说明；28 个生产／测试 Python 文件 AST 可解析，代码行长、C++ 文件首注释、Markdown 链接及差异格式检查通过。上述检查不运行项目或测试。
