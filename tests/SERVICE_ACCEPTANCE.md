# Engram SSD 目标机验收

状态为**未运行**，下列构建、测试和服务命令在目标 Linux NVIDIA 机器执行。

## 1. 重编扩展与运行 pytest

按部署第 1～4 节准备模型、宿主机目录和容器，再同步插件更新。进入容器后，按部署第 5 节的命令重新安装 editable 包并编译 `query/_C*.so`。 [部署流程](../../SGLANG_ENGRAM_SSD_DEPLOY.md)

```bash
sudo docker exec -it sglang-engram-ssd bash
```

```bash
python -m pip install --no-deps --no-build-isolation -e /workspace/sglang_engram_ssd
cd /workspace/sglang_engram_ssd
cmake -S src/sglang_engram_ssd/query/csrc -B build/query \
  -DCMAKE_BUILD_TYPE=Release \
  -DENGRAM_SSD_BUILD_PYTHON=ON \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/query --parallel 4
```

构建成功后运行常规套件和真实 TP8 的 NCCL／Gloo 用例。

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m pytest -ra tests --ignore=tests/test_distributed.py
torchrun --standalone --nproc-per-node=8 -m pytest -ra tests/test_distributed.py
```

检查 `-ra` 的跳过原因，确认 GPU、原生 io_uring 和真实分布式用例都已执行。

## 2. 生成固定负载

开两个容器 Bash 终端，一个启动服务，一个发起验收。两个终端使用同一结果目录，每套服务依次启动。重做整套验收时使用新的结果目录。

```bash
export PLUGIN_DIR=/workspace/sglang_engram_ssd
export ACCEPT_DIR=/workspace/engram_service_acceptance/run_01
export SERVICE_URL=http://127.0.0.1:30001
mkdir -p "$ACCEPT_DIR"
```

在验收终端执行一次。沿用 `/model` 的本地 tokenizer，直接记录固定长度的 `input_ids`。共享前缀组的种子与各用例复用同一段 4096 token。

```bash
python - <<'PY'
import json
import os
from pathlib import Path

from transformers import AutoTokenizer

root = Path(os.environ["ACCEPT_DIR"])
tokenizer = AutoTokenizer.from_pretrained("/model", trust_remote_code=True, local_files_only=True)

def encode(label, index, length):
    """将 label 和 index 编入提示，返回恰好 length 个本地 tokenizer 生成的 token。"""
    text = f"{label}编号{index}。请分析这段文字并给出结论。" + "研究存储系统中的数据访问与推理计算。" * length
    ids = tokenizer.encode(text, add_special_tokens=False)[:length]
    assert len(ids) == length
    return ids

shared = encode("共享前缀", 0, 4096)
for name, lengths in {
    "correctness": [16, 128, 512, 1024, 4096, 8192],
    "radix-seed": [4608],
    "correctness-radix": [5120, 6144, 7168, 8192],
    "prefill": [4096, 5120, 6144, 8192] * 32,
    "decode": [1024] * 128,
    "mixed": [1024, 2048, 4096, 8192] * 32,
    "stress": [1024, 4096, 8192, 8192] * 32,
}.items():
    prefix = shared if name in ("radix-seed", "correctness-radix") else []
    with (root / f"{name}.jsonl").open("x", encoding="utf-8") as out:
        for index, length in enumerate(lengths):
            ids = prefix + encode(name, index, length - len(prefix))
            out.write(json.dumps({"id": f"case-{index}", "input_ids": ids}) + "\n")
PY
```

## 3. 按模式启动并检查服务

在服务终端定义部署第 6 节的 `launch_engram`，再定义 `start_round`。各模式使用同一条 launch 命令，同轮只切换环境。模型、TP8、Marlin、`--chunked-prefill-size 8192` 和 `--max-running-requests 32` 固定，`--max-prefill-tokens` 用默认值。 [公用启动命令](../../SGLANG_ENGRAM_SSD_DEPLOY.md#6-新容器启动-engram-服务)

```bash
start_round() (
  local mode=$1 round=$2
  unset SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_{WINDOW_BATCHES,POOL_BATCHES,POOL_BYTES,LOG_INTERVAL}
  unset SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEBUG_CHECK_ROWS SGLANG_TEST_RETRACT SGLANG_TEST_RETRACT_INTERVAL
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export SGLANG_PLUGINS=engram_ssd
  export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0
  export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=shared
  export SGLANG_ENABLE_DSV41_ENGRAM_SSD=0
  export SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH=0
  export SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS=1024
  case "$mode" in
    ddr) export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 ;;
    sync) export SGLANG_ENABLE_DSV41_ENGRAM_SSD=1 ;;
    prefetch)
      export SGLANG_ENABLE_DSV41_ENGRAM_SSD=1
      export SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH=1
      export SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_WINDOW_BATCHES=2
      export SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POOL_BATCHES=3
      export SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_LOG_INTERVAL=1
      export SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEBUG_CHECK_ROWS=0
      if [[ $round == correctness* ]]; then
        export SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEBUG_CHECK_ROWS=1
      fi
      ;;
    *) return 2 ;;
  esac
  if [[ $round == stress ]]; then
    export SGLANG_TEST_RETRACT=1 SGLANG_TEST_RETRACT_INTERVAL=8
  fi
  if [[ $round == correctness-radix ]]; then
    launch_engram "$ACCEPT_DIR/$mode-$round.log"
  else
    launch_engram "$ACCEPT_DIR/$mode-$round.log" --disable-radix-cache
  fi
)
```

在验收终端定义 `check_ready` 和 `accept`。每轮检查 `/health`、服务启动日志，调度预取还需看到所有 rank 的起始 idle 快照，用于计算该轮差值。

```bash
check_ready() {
  local mode=$1 round=$2 rank
  local log="$ACCEPT_DIR/$mode-$round.log"
  curl --noproxy '*' -fsS "$SERVICE_URL/health" >/dev/null || return
  if [[ $mode == ddr ]]; then
    grep -m 1 'engram host table .*layout=shared' "$log"
    return
  fi
  for rank in {0..7}; do
    grep -m 1 "Engram SSD enabled: TP rank=$rank," "$log" || return
  done
  if grep -n 'stream memory operations unavailable' "$log"; then
    return 1
  fi
  if [[ $mode == prefetch ]]; then
    for rank in {0..7}; do
      grep -q 'Engram SSD sched prefetch stats: .*"phase": "idle".*"referenced_slots": 0.*"tp_rank": '"$rank"'[,}]' "$log" || return
    done
  fi
}

accept() {
  local mode=$1 round=$2 command=$3 workload=$4 concurrency=$5 tokens=$6
  shift 6
  python "$PLUGIN_DIR/tests/service_acceptance.py" "$command" \
    --url "$SERVICE_URL" --tp-size 8 --timeout 1200 --idle-wait-seconds 30 \
    --workload "$ACCEPT_DIR/$workload.jsonl" --concurrency "$concurrency" --max-new-tokens "$tokens" \
    --server-log "$ACCEPT_DIR/$mode-$round.log" --output "$ACCEPT_DIR/$mode-$workload.json" "$@"
}
```

服务终端每次运行一个 `start_round`。等待加载和预热完成后，在验收终端运行对应命令。每轮结束后用 Ctrl+C 停止服务，确认 GPU 已释放再启动下一轮。

## 4. A6 逐位核对与共享前缀

两组都用并发 1 和最多 64 个输出 token。先记录 DDR 基线，再逐请求核对同步 SSD 和调度预取的输出 token ID 与结束原因。调度预取正确性轮还开行号逐位核对。任一 `compare` 退出非零、`failed_request_ids` 或 `token_mismatch_ids` 非空，或服务出现 `Engram layer … row mismatch`，该轮即失败。

| 服务终端 | 验收终端，先执行同模式的 `check_ready MODE correctness` |
| --- | --- |
| `start_round ddr correctness` | `accept ddr correctness record correctness 1 64` |
| `start_round sync correctness` | `accept sync correctness compare correctness 1 64 --baseline "$ACCEPT_DIR/ddr-correctness.json"` |
| `start_round prefetch correctness` | `accept prefetch correctness compare correctness 1 64 --baseline "$ACCEPT_DIR/ddr-correctness.json"` |

共享前缀组对每种模式重新启动 `start_round MODE correctness-radix`，省略 `--disable-radix-cache`。执行 `check_ready MODE correctness-radix` 后，在验收终端将下列 `mode` 依次设为 `ddr`、`sync`、`prefetch`，每次使用同模式的服务。

```bash
mode=ddr
accept "$mode" correctness-radix record radix-seed 1 1 && (
  if [[ $mode == ddr ]]; then
    accept "$mode" correctness-radix record correctness-radix 1 64
  else
    accept "$mode" correctness-radix compare correctness-radix 1 64 \
      --baseline "$ACCEPT_DIR/ddr-correctness-radix.json"
  fi
)
```

逐组确认报告中共享前缀用例都实际命中 KV cache。

```bash
python - "$ACCEPT_DIR" <<'PY'
import json
import sys
from pathlib import Path

for mode in ("ddr", "sync", "prefetch"):
    report = json.loads((Path(sys.argv[1]) / f"{mode}-correctness-radix.json").read_text())
    assert report["acceptance_passed"], mode
    assert all((row["cached_tokens"] or 0) > 0 for row in report["records"]), mode
PY
```

## 5. 清页缓存并记录 SSD I/O

在宿主机用 `findmnt -T "$ENGRAM_MODEL_DIR"` 与 `lsblk` 确认模型所在的块设备和底层 SSD。每次性能或压力轮先完成服务预热，再在宿主机或 privileged 容器执行 `sync` 和 `echo 3 > /proc/sys/vm/drop_caches`，紧接着运行负载。这个操作会清理整机文件页缓存，当前 reader 使用 buffered 读，长时间运行时页缓存会回暖。

下列函数在部署的 privileged 容器验收终端执行，需要 `iostat`（sysstat）。如果用宿主机记录，先用 `sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'` 清缓存，在宿主机启动 `iostat -x -y 1` 并把输出保存到同一轮结果目录，再在验收终端直接执行对应 `accept`。结束时停止 `iostat`。

```bash
measure() (
  local mode=$1 round=$2 monitor
  shift 2
  check_ready "$mode" "$round" || return
  command -v iostat >/dev/null || return
  sync && echo 3 > /proc/sys/vm/drop_caches || return
  iostat -x -y 1 > "$ACCEPT_DIR/$mode-$round.iostat" &
  monitor=$!
  trap 'kill "$monitor" 2>/dev/null; wait "$monitor" 2>/dev/null' EXIT
  accept "$mode" "$round" "$@"
)
```

## 6. A8 固定性能轮

按 `ddr`、`sync`、`prefetch` 分别执行下表。每个单元使用新服务和独立日志，保持相同预热过程与冷页缓存起点。表中 `mode` 替换为三个模式之一。`perf` 对多 token 轮忽略 EOS，并核对实际输出长度。

| 负载 | 服务终端 | 验收终端 |
| --- | --- | --- |
| Prefill 为主，4096～8192 输入／1 输出 | `start_round mode prefill` | `measure mode prefill perf prefill 128 1` |
| Decode 为主，1024 输入／256 输出 | `start_round mode decode` | `measure mode decode perf decode 128 256` |
| 混合，1024～8192 输入／128 输出 | `start_round mode mixed` | `measure mode mixed perf mixed 128 128` |

性能轮和压力轮固定使用 128 并发，给 32 个运行槽和至少 64 个 waiting 请求留出余量。查该轮 `Prefill batch` 的 `#queue-req` 或报告的 `server_log.prefill_queue_peak_logged`，确认至少出现 64 个 waiting，覆盖两个最多各 32 请求的预测批。两个 SSD 模式都用 `.iostat` 确认块设备实际发生读取，记录初始冷读和后续回暖区间。调度预取报告的 `idle_zero_references_observed_all_ranks` 应为 `true`，表示轮末快照已收齐。缺少相应观测时，性能结论保持待验证。

## 7. A7 中止、撤回与空闲引用

在服务终端运行 `start_round prefetch stress`。该轮启用原生测试撤回，每 8 个 forward 尝试一次。验收终端运行下列命令，结束后停止专用服务。

```bash
measure prefetch stress stress stress 128 256 --abort-delay-ms 500 --require-retract
```

脚本按真实 `rid` 中止首批并发任务中的每四个请求之一。完整 A7 要求所有目标请求都返回 `finish_reason.type=abort`，其余请求正常结束，`/health` 通过，并且 HTTP 的 `num_retractions` 或服务日志的 `#retracted_reqs` 证明确有撤回。八个 rank 还须各有本轮新增的 `phase=idle, referenced_slots=0` 快照。READY 缓存可留在池中。以报告的 `full_A7_passed=true` 和命令退出码 0 为准。

## 8. 汇总时延、吞吐与 SSD 统计

客户端报告中 `plugin_counter_deltas_by_rank` 使用同一进程前后快照做差，`window_hit_rate` 使用轮末累计快照。分母为零的派生指标为空。用相同负载下的 TTFT、TPOT 和吞吐比较三套服务，结合各 rank 的读块时间与放行等待判断读盘隐藏程度。

| 字段 | 来源与口径 |
| --- | --- |
| `ttft_ms`、`client_ttft_{mean,p50,p95}_ms` | 客户端从发出 HTTP 到收到首个实际输出 token 的 wall time，包含排队、计算和流式传输。 |
| `tpot_ms`、`client_tpot_{mean,p50,p95}_ms` | 每请求 `(latency_ms - ttft_ms) / (输出 token 数 - 1)`。输出少于 2 个时为空。汇总使用有 TPOT 的请求。 |
| `client_prompt_tokens_per_second`、`client_output_tokens_per_second` | 无客户端错误的请求的输入／实际输出 token 总数，各除以整轮客户端 wall time。 |
| `input throughput (token/s)`、`#queue-req` | SGLang `Prefill batch` 日志的 prefill 输入吞吐与采样时 waiting 数。 |
| `prefill_wait_ms`、`prefill_waits`、`avg_prefill_wait_ms` | `wait_urgent` 从进入到返回的累计毫秒数、调用数与两者之比，含后台等待和前台同步补读。每个 prefill forward 的每个 Engram 层各调用一次，只等该层的行。 |
| `decode_wait_ms`、`decode_waits`、`avg_decode_wait_ms` | hash D2H 事件同步返回至 `publish_decode` 放行 H2D 的累计毫秒数、批次数与两者之比。这段主机耗时近似 Engram 额外等待。 |
| `urgent_rows` | Prefill 和 decode 组批发现的唯一 owner 缺行数之和，按层去重。 |
| `predicted_spans`、`matched_spans`、`span_match_rate` | 参与预取组装的 prefill 实际非空区间数、与同请求有效窗口段完全相等的个数，以及 `matched_spans / predicted_spans`。 |
| `window_hit_rate` | Prefill 组批时 READY 的唯一 owner 行数除以唯一 owner 查询行数，自进程启动累计。 |
| `read_ms`、`read_rows` | 成功返回的后台 `read_rows_to_slots` 调用累计毫秒数与行数，包含普通和紧急块。`read_ms / read_rows` 为每行平均读取耗时。 |
| `evicted_slots`、`evicted_unused_slots`、`wasted_read_ratio` | LRU 淘汰的 READY 槽数，其中发布后从未被 `acquire` 或 `resolve` 命中的槽数，以及 `evicted_unused_slots / read_rows`。淘汰可涉及轮前已读入的槽。 |
| `referenced_slots` | 该快照中有引用的槽数。A7 使用全部 rank 新发出的空闲快照。 |
| `r/s`、`rkB/s`、`r_await`、`%util` | `iostat -x -y 1` 的块设备读次数／秒、读取 KiB／秒、读请求延迟毫秒和设备繁忙时间占比。检查模型所在 SSD 的读活动。 |
