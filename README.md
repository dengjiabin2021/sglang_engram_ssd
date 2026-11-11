# SGLang Engram SSD 插件

基于官方 SGLang `baa9befa381511cd4d56f83980951e5a45fffc6d`，通过 HookRegistry 接入，所有修改均位于插件仓库。

## 固定读取路径

每个 TP rank 负责 `[N*r//P,N*(r+1)//P)` 的全局行区间。模型到达 Engram 层后，同步 io_uring 读取所属行，单套 pinned／GPU 工作区串行分块完成 H2D 和原生 `engram_gather`，再将局部 BF16 结果放回完整零张量。模型线程在原 Engram 调用点执行一次 TP all-reduce；空 owner 也参加，TP=1 不通信。

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
| `SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS` | `4096` | 单套同步中转区的行容量，必须为正 |
| `SGLANG_DSV41_ENGRAM_SSD_IO_URING_QUEUE_DEPTH` | `32` | 在途文件请求数，必须为正；weight、scale 各占一个请求 |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP` | `0` | 对整次本 rank 查询去重，读取唯一行后恢复重复位置 |
| `SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH` | `0` | 调度预取总开关；捕获 CPU hash 参数并创建独立 pinned 行池 |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POOL_BYTES` | `2147483648` | 每 rank 行池 payload 预算，必须为正且至少容纳一行 |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_SEGMENT_TOKENS` | `0` | 段大小；0 随 scheduler 的 chunk 大小，未启用 chunk 时为 8192 |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POLICY` | `timeout` | `best_effort`、`wait_complete`、`timeout` |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_TIMEOUT_MS` | `500` | timeout 策略的非负等待毫秒数；其他策略显式设置会警告并忽略 |
| `SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_LOG_INTERVAL` | `0` | 每 N 次 prefill 打印统计，0 关闭 |

去重在同步内部读取入口完成，保留原顺序、形状和 BF16 数值；关闭时不分配 unique／inverse 张量。

CPU 行号计算直接调用官方 `compute_engram_hash_ids`，参数从模型已构造的 Hasher 复制，不重建 tokenizer，也不提前更新请求 history。其 Hook 仅随调度预取总开关登记。

行池保存两张连续 CPU pinned 表，起点均 4096 对齐，容量为 `POOL_BYTES//(D+D/32)`。同一层／行只占一个槽；零引用 READY 槽按 LRU 复用，PENDING 和有引用槽不得覆盖。该预算不包含元数据与对齐余量，不是整个进程 RSS 上限。

调度预取关闭时不解析其数值／策略参数，显式设置只警告并忽略，不创建行池、后台线程或 CPU hash 副本。4096 对齐仅预留后续读取接口，本轮仍从原 checkpoint 正常读取，不包含 O_DIRECT 或合并格式。

旧版读取选择、查询模式、按层预取、流水和 reader 缓存配置均已退役；显式设置只在启动时警告一次并忽略。配置只由 `config.py` 解析，非法值在原模型加载前拒绝。

## 启动

### 调度预取流程

请求成功进入 waiting 队列后，插件保存完整纯文本 token 快照，从匹配 prefix 之后分段；每 rank 只有一个持久后台线程，计算 CPU 行号并向同一个 io_uring reader 提交本 rank 的缺失行。

优先顺序为 chunked 请求下一段、waiting 请求各自首段，再补后续段；每个请求最多占有当前段与下一段。池满立即停止占槽，在后续调度中重试，不让 scheduler 等待 I/O。

| 策略 | 放行规则 |
|---|---|
| `best_effort` | 全部放行，未就绪数据在 forward 补读 |
| `wait_complete` | 下一段在全部 TP rank READY 才放行 |
| `timeout` | 全部 rank 各自满足 READY 或提交后超时即可放行 |

整批候选只通过一次 CPU gloo `all_reduce(MIN)` 得到一致结果，空候选不通信。未登记、池满或实际 prefix 比原计划更早的未覆盖请求，按入队 deadline 防止饿死；非 timeout 策略使用默认 500ms，不解析显式 TIMEOUT_MS。chunked continuation 不经过 waiting 筛选。

原生 raw 调度会在内部加入 grammar-ready 请求并排序，因此新增 priority AFTER Hook 在该阶段结束后统一过滤；外层 AROUND 在返回时恢复暂缓请求及期间新入队项。run_batch AFTER 保存不可变 forward_iter／token 区间／generation，结果处理 AFTER 按此对账，避免 overlap 与 retract 的可变 Req 字段造成错误释放。

后台异常在下一次调度或 forward 主路径抛出；后台不发 TP collective。abort、完成和 retract 释放请求引用；已提交 I/O 完成后才回收槽位。关闭顺序为停止提交、收齐后台、收齐 GPU 临时引用、关闭行池、reader、FD。

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

## 验证状态

本地 macOS 仅做源码静态检查，未编译、未运行 pytest 或模型。测试位于 `tests/`，原生和 CUDA 测试在目标环境执行；实际 TP8 数值、端到端和性能结果仍待验证。
