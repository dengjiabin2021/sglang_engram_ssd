# A6–A8 远端服务验收

当前状态：**未运行**。此工具只在目标 NVIDIA 测试机手动执行，本地 macOS 不启动服务、不执行验收。
脚本使用 Python 标准库访问已启动的服务，不负责启动或停止模型。

## 1. 准备固定输入与结果目录

在容器中执行。`/model` 使用已在本地的模型目录，无网络下载。

```bash
export PLUGIN_DIR=/workspace/sglang_engram_ssd
export ACCEPT_DIR=/workspace/engram_service_acceptance/run_01
mkdir -p "$ACCEPT_DIR"
```

手动将本仓库的 `tests/service_acceptance.py` 上传至对应目录。结果文件已存在时脚本拒绝覆盖，下一轮改用
`run_02`。输入 JSONL 每行包含唯一 `id`，以及 `text` 或 `input_ids` 二选一，不接受多模态。

下面只用本地 tokenizer 生成固定 token 长度的输入，不加载模型权重：

```bash
python - <<'PY'
import json
import os
from pathlib import Path
from transformers import AutoTokenizer

root = Path(os.environ["ACCEPT_DIR"])
tokenizer = AutoTokenizer.from_pretrained("/model", trust_remote_code=True, local_files_only=True)
for filename, lengths in (
    ("correctness.jsonl", [16, 128, 512, 1024, 4096, 8192]),
    ("performance.jsonl", [1024] * 64),
    ("stress.jsonl", [1024, 4096, 8192, 1024] * 8),
):
    with (root / filename).open("x") as out:
        for index, length in enumerate(lengths):
            text = f"编号{index}。请分析这段文字并给出结论。" + "研究存储系统中的数据访问与推理计算。" * length
            ids = tokenizer.encode(text, add_special_tokens=False)[:length]
            assert len(ids) == length
            out.write(json.dumps({"id": f"case-{index}", "input_ids": ids}) + "\n")
PY
```

## 2. 原同步 SSD 服务记录基线

使用项目已有部署命令启动服务：TP8、io_uring、单套同步工作区，调度预取与去重均关闭。
两次服务使用相同模型、revision、dtype、TP、chunked-prefill、batch/token 预算及其他模型参数；关闭 CUDA graph，
不启用 Attention DP/CP、PP、PD、多模态或投机解码。

建议固定 `--chunked-prefill-size 1024 --max-running-requests 8 --disable-radix-cache`。
部署命令末尾用 `2>&1 | tee "$ACCEPT_DIR/baseline.log"` 保留日志。先等待权重加载、算子编译和预热完成。

```bash
python "$PLUGIN_DIR/tests/service_acceptance.py" record \
  --workload "$ACCEPT_DIR/correctness.jsonl" --concurrency 1 --max-new-tokens 64 \
  --server-log "$ACCEPT_DIR/baseline.log" --output "$ACCEPT_DIR/baseline_tokens.json"

python "$PLUGIN_DIR/tests/service_acceptance.py" perf \
  --workload "$ACCEPT_DIR/performance.jsonl" --concurrency 32 --max-new-tokens 1 \
  --server-log "$ACCEPT_DIR/baseline.log" --output "$ACCEPT_DIR/baseline_perf.json"
```

`record` 使用 `/generate`、`temperature=0`、`return_logprob=true`，保存实际输出 token ID，不对文本重新分词。
流式日志概率列表支持累计／增量两种格式。HTTP、token 元数据或请求失败会使验收退出码非零。

## 3. 优化服务逐位比较与性能记录

停止基线服务，确认其已释放 GPU 后启动优化服务。开启：

```bash
export SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH=1
export SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP=1
export SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_LOG_INTERVAL=1
```

沿用相同部署参数，将日志保存为 `"$ACCEPT_DIR/optimized.log"`。等待加载与预热完成后执行：

```bash
python "$PLUGIN_DIR/tests/service_acceptance.py" compare \
  --baseline "$ACCEPT_DIR/baseline_tokens.json" \
  --workload "$ACCEPT_DIR/correctness.jsonl" --concurrency 1 --max-new-tokens 64 \
  --server-log "$ACCEPT_DIR/optimized.log" --output "$ACCEPT_DIR/optimized_tokens.json"

python "$PLUGIN_DIR/tests/service_acceptance.py" perf \
  --workload "$ACCEPT_DIR/performance.jsonl" --concurrency 32 --max-new-tokens 1 \
  --server-log "$ACCEPT_DIR/optimized.log" --output "$ACCEPT_DIR/optimized_perf.json"
```

A6 通过条件：`token_mismatch_ids=[]`、`failed_request_ids=[]`，并且退出码为 0。
脚本校验输入、最大输出长度及并发数与基线相同。再测并发一致性时，基线和优化两边都以相同并发分别记录。
两套服务分开启动，基线文件保持不变。

## 4. 并发中止、长 prompt 与真实 retract

在优化服务上执行：

```bash
python "$PLUGIN_DIR/tests/service_acceptance.py" stress \
  --workload "$ACCEPT_DIR/stress.jsonl" --concurrency 32 --max-new-tokens 256 \
  --abort-delay-ms 500 --require-retract --timeout 1200 \
  --server-log "$ACCEPT_DIR/optimized.log" --output "$ACCEPT_DIR/stress.json"
```

前 32 个并发任务中，每四个选一个请求，延迟后按其真实 `rid` 调用 `/abort_request`；其他请求正常完成。
长输入超过 1024 token，服务按部署配置执行 chunked prefill。`ignore_eos=true` 只用于压力阶段，延长请求存活时间。
只收到 abort HTTP 200 不算中止成功：需要相应请求最终返回 `finish_reason.type=abort`。目标请求已过早完成、未及时
入队或网络失败都不会伪装成成功；按实际情况调整延迟和输出长度，使用新结果文件重测。

真正的 retract 必须至少有一种观测：

- HTTP `meta_info.num_retractions > 0`；脚本累计到 `http_retractions_observed`。
- 本轮日志有 `Testing retraction.` 或 `KV cache pool is full. Retract requests.`，且 `#retracted_reqs > 0`。

`--require-retract` 下没有观测会返回非零退出码，不能声称覆盖 retract。普通长 prompt／HTTP abort 也不等于 retract。
若正常内存压力未触发，可额外启动一次专用优化服务，使用 SGLang 已有测试能力：

```bash
export SGLANG_TEST_RETRACT=1
export SGLANG_TEST_RETRACT_INTERVAL=8
```

这些变量在服务启动时读取，需要重启。专用日志保存为 `retract.log`，重跑 `stress` 并另存 `retract_stress.json`；
不能将强制 retract 下的吞吐混入第 3 节性能对照。完成后取消两个变量。

A7 的 HTTP 部分要求：无失败、所有目标请求实际 abort、其他请求完成、服务 `/health` 正常、retract 有观测。
**完整 A7 还需核对全部请求结束且 GPU event 已回收之后的空闲行池快照：`referenced_slots=0`。**
READY 缓存可以继续占槽，`occupied_slots` 无需为 0。forward 中间打印的引用数不能代替空闲快照。
脚本默认按 TP8 等待最多 10 秒，只有八个 rank 均出现本轮新增的 `phase=idle, referenced_slots=0` 快照，
并满足中止、完成、健康与 retract 检查，才将 `full_A7_passed=true`。日志没有全部 rank 的明确身份、缺失空闲快照
或只有上一轮空闲记录时均不判通过。可设置 `--idle-wait-seconds 30` 延长等待；`--tp-size` 必须等于真实 TP 数。
没有快照时保留待验证，不能将单纯 HTTP 压力成功当成无泄漏证明。`--require-retract` 同时要求完整 A7 检查通过。

## 5. 性能指标与日志读取

| 数据 | 来源与口径 |
| --- | --- |
| TTFT mean／p50／p95 | 客户端发起 HTTP 到首个实际输出 token 事件；含网络、服务排队、prefill、流式发送。 |
| `client_prompt_tokens_per_second` | 完成请求的 prompt token 总数／整轮 wall time；`perf` 只输出一个 token，是端到端 prefill 服务吞吐，不是纯 GPU 算子吞吐。 |
| 服务 `input throughput (token/s)` | 原生 `Prefill batch` 日志；是相邻 prefill 日志之间的服务统计，包含该间隔中的其他工作，不能当成单层计算速度。 |
| `forward_hits`／`forward_misses` | 插件 JSON 统计；可用于计算行池命中率。 |
| `miss_read_ms` | forward 同步未命中补读的 wall time，含 H2D／gather 等；不是纯 SSD I/O 时间。 |
| `submitted_rows`／`dedup_saved_rows` | 实际向 reader 提交的预取行／成功登记复用行；占槽取消不算读取提交。 |
| `occupied_slots`／`referenced_slots` | 日志快照，不是本轮累计次数；引用泄漏需要空闲时核对。 |

脚本收集 `Engram SSD sched prefetch stats: {JSON}` 的逐 rank 最新快照。若运行前后都有同 rank 快照，
还输出对应计数差值与 `forward_pool_hit_rate`；没有起始快照时不虚构差值。完整日志中的快照、吞吐样本及队列峰值
可能包含先前验收请求，分析单轮性能时应使用计数差值，或从该轮对应的日志范围提取。

```bash
rg 'Engram SSD sched prefetch stats:|Prefill batch|Testing retraction|Retract requests' \
  "$ACCEPT_DIR/optimized.log"
```

A8 报告同时列出 baseline／optimized 的 TTFT、端到端 prefill 服务吞吐，以及插件行池命中／未命中同步等待。
在 `Prefill batch` 日志中确认等待深度不少于两批，例如 `--max-running-requests 8` 时观察到至少 16 个 waiting 请求；
若负载不足，增加输入条数和客户端并发后对两套服务重新测量。不从 HTTP 推导未提供的 SSD、H2D 或 gather 时间。

本轮工具未在目标环境执行，A6／A7／A8 均保持待验证。

两套服务应使用相同预热过程，并记录文件页缓存初态。本轮仍为 buffered 文件读取，尚无 O_DIRECT；不能把不同页缓存条件造成的变化全部归因于调度预取。
