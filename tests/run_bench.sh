#!/usr/bin/env bash
# 一次完成 Engram SSD 对照测试：生成负载 → 依次测同步 SSD、调度预取、DDR 三组 → 汇总并打包结果。
# 前提：已按执行手册安装 liburing、iostat 并编译好扩展。用法（容器内）：
#   bash /workspace/sglang_engram_ssd/tests/run_bench.sh --window 2 --pool 3 --max-running 8 --concurrency 64 \
#     --prompt-len 500-1500 --num-prompts 128 --output-len 64-192 --ep 8 --modes "sync prefetch ddr"
# 每个参数都可省略，省略时用下面的默认值；--out 指定结果目录，默认 /workspace/engram_bench/<时间>。
# --prompt-len 和 --output-len 写成 "最小-最大" 时每条请求在范围内随机取值，写成单个数则固定长度。
set -euo pipefail

PLUGIN_DIR=/workspace/sglang_engram_ssd
OUT=/workspace/engram_bench/$(date +%Y%m%d-%H%M)
EP=8                     # MoE 专家并行数，TP8 下 EP8 时每卡持有完整专家
MAX_RUNNING=8            # 服务同时运行的请求数上限（--max-running-requests）
CONCURRENCY=64           # 客户端并发，多出的请求留在 waiting 队列供预取
WINDOW=2                 # 预取尚未开始组批的前几个 prefill 批
POOL=3                   # 行池容量，单位为批，须大于 WINDOW
PROMPT_LEN=500-1500      # 每条请求的输入 token 数（范围内随机）
NUM_PROMPTS=128          # 请求条数
OUTPUT_LEN=64-192        # 每条请求的输出 token 数（范围内随机，让请求错开结束）
MODES="sync prefetch ddr"  # 依次测试的组
while (( $# )); do
  case "$1" in
    --window) WINDOW=$2 ;;
    --pool) POOL=$2 ;;
    --max-running) MAX_RUNNING=$2 ;;
    --concurrency) CONCURRENCY=$2 ;;
    --prompt-len) PROMPT_LEN=$2 ;;
    --num-prompts) NUM_PROMPTS=$2 ;;
    --output-len) OUTPUT_LEN=$2 ;;
    --ep) EP=$2 ;;
    --modes) MODES=$2 ;;
    --out) OUT=$2 ;;
    *) echo "未知参数 $1"; exit 1 ;;
  esac
  shift 2
done
PORT=30001
(( POOL > WINDOW )) || { echo "POOL 必须大于 WINDOW"; exit 1; }
SERVER_PID=
IOSTAT_PID=
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
cd /workspace  # 避免 /workspace 下的同名目录遮蔽已安装的包
command -v iostat >/dev/null || { echo "缺少 iostat，先执行 apt-get install -y sysstat"; exit 1; }

# 退出时停掉仍在运行的服务和 iostat。
cleanup() {
  [[ -n $IOSTAT_PID ]] && kill "$IOSTAT_PID" 2>/dev/null || true
  [[ -n $SERVER_PID ]] && stop_server || true
}
trap cleanup EXIT

# 用本地 tokenizer 的词表生成 NUM_PROMPTS 条互不相同的随机 token 请求：输入长度取自 PROMPT_LEN，
# 输出长度取自 OUTPUT_LEN 并写进每条请求；固定种子，各组输入完全相同。
# 随机 token 让 n-gram 基本不重复，Engram 行的访问接近真实文本的分散程度，SSD 读量不会被重复内容掩盖。
make_workload() {
  OUT="$OUT" PROMPT_LEN="$PROMPT_LEN" OUTPUT_LEN="$OUTPUT_LEN" NUM_PROMPTS="$NUM_PROMPTS" python - <<'PY'
import json
import os
import random
from pathlib import Path

from transformers import AutoTokenizer


def bounds(text):
    """把 "a-b" 或 "n" 解析成 (最小, 最大)。"""
    low, _, high = text.partition("-")
    return int(low), int(high or low)


tokenizer = AutoTokenizer.from_pretrained("/model", trust_remote_code=True, local_files_only=True)
rng = random.Random(0)
# 只取基础词表，跳过前 100 个和追加的特殊 token。
vocab = range(100, tokenizer.vocab_size)
prompt, output = bounds(os.environ["PROMPT_LEN"]), bounds(os.environ["OUTPUT_LEN"])
with (Path(os.environ["OUT"]) / "mixed.jsonl").open("x", encoding="utf-8") as out:
    for index in range(int(os.environ["NUM_PROMPTS"])):
        ids = [rng.choice(vocab) for _ in range(rng.randint(*prompt))]
        case = {"id": f"case-{index}", "input_ids": ids, "max_new_tokens": rng.randint(*output)}
        out.write(json.dumps(case) + "\n")
PY
}

# 按模式设置环境变量：ddr 为原生 DDR 主机表；sync 为 SSD 同步查询；prefetch 为 SSD + 调度预取。
# DDR 用 per_rank 布局：每 rank 只存自己的行，查询后与 SSD 组一样做一次 TP all-reduce。
# 两个 SSD 组都开启同步查询去重，同步路径每行只读一次，与预取组按唯一行读取的口径一致。
# 三组都开启 Engram 计时：每个 prefill forward 输出各 Engram 层（含 all-reduce）的 GPU 耗时。
set_mode() {
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SGLANG_PLUGINS=engram_ssd SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
  export SGLANG_DSV41_ENGRAM_SSD_CHUNK_ROWS=1024 SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEBUG_CHECK_ROWS=0
  export SGLANG_ENABLE_DSV41_ENGRAM_SSD_DEDUP=1 SGLANG_ENABLE_DSV41_ENGRAM_TIMING=1
  export SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH=0
  case "$1" in
    ddr) export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1 SGLANG_ENABLE_DSV41_ENGRAM_SSD=0 ;;
    sync) export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0 SGLANG_ENABLE_DSV41_ENGRAM_SSD=1 ;;
    prefetch)
      export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=0 SGLANG_ENABLE_DSV41_ENGRAM_SSD=1
      export SGLANG_ENABLE_DSV41_ENGRAM_SSD_SCHED_PREFETCH=1
      export SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_WINDOW_BATCHES="$WINDOW"
      export SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_POOL_BATCHES="$POOL"
      export SGLANG_DSV41_ENGRAM_SSD_SCHED_PREFETCH_LOG_INTERVAL=1
      ;;
    *) echo "未知模式 $1"; exit 1 ;;
  esac
}

# 后台启动 TP8/EP 服务，日志写到 $1；setsid 让服务及其子进程同属一个进程组，便于整体停止。
start_server() {
  setsid bash -c 'cd /workspace/sglang && exec python -m sglang.launch_server \
    --model-path /model --trust-remote-code --language-model-only --device cuda --tp-size 8 --ep-size '"$EP"' \
    --moe-runner-backend marlin --load-format safetensors --startup-weight-load-mode serial \
    --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
    --chunked-prefill-size 8192 --max-running-requests '"$MAX_RUNNING"' --disable-radix-cache \
    --host 127.0.0.1 --port '"$PORT" > "$1" 2>&1 &
  SERVER_PID=$!
}

# 停止服务进程组，等到 GPU 上没有计算进程；超时后强杀。
stop_server() {
  local waited=0
  kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
  while [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader) ]]; do
    sleep 5; waited=$((waited + 5))
    if (( waited == 120 )); then kill -KILL -- "-$SERVER_PID" 2>/dev/null || true; fi
    if (( waited > 300 )); then echo "GPU 进程未退出"; exit 1; fi
  done
  SERVER_PID=
}

# 等服务就绪；prefetch 还要求 8 个 rank 启用 SSD、流内存操作可用，并已输出起始 idle 统计快照。
wait_ready() {
  local mode=$1 log=$2 rank waited=0
  until grep -q 'The server is fired up and ready to roll' "$log"; do
    # 只按进程退出或调度进程异常判失败；插件加载失败等非致命 Traceback 由后面的 SSD 启用检查兜住。
    if ! kill -0 "$SERVER_PID" 2>/dev/null || grep -q 'Scheduler hit an exception' "$log"; then
      echo "服务启动失败，见 $log"; exit 1
    fi
    sleep 10
  done
  if grep -q 'Failed to load plugin engram_ssd' "$log"; then
    echo "插件 engram_ssd 加载失败，见 $log"; exit 1
  fi
  [[ $mode == ddr ]] && return 0
  for rank in {0..7}; do
    grep -q "Engram SSD enabled: TP rank=$rank," "$log" || { echo "rank $rank 未启用 SSD"; exit 1; }
  done
  [[ $mode == sync ]] && return 0
  if grep -q 'stream memory operations unavailable' "$log"; then echo "decode 退回同步查询"; exit 1; fi
  for rank in {0..7}; do
    until grep -q '"phase": "idle".*"tp_rank": '"$rank"'[,}]' "$log"; do
      sleep 1; waited=$((waited + 1))
      (( waited < 300 )) || { echo "rank $rank 没有 idle 快照"; exit 1; }
    done
  done
}

# 跑一组：启动服务 → 就绪 → 清页缓存 → 记录 iostat → 按 CONCURRENCY 并发发送负载（每条按负载里的输出长度生成）→ 停服务。
run_mode() {
  local mode=$1
  echo "== $mode 开始 $(date +%T) =="
  set_mode "$mode"
  start_server "$OUT/$mode.log"
  wait_ready "$mode" "$OUT/$mode.log"
  sync
  echo 3 > /proc/sys/vm/drop_caches || { echo "清页缓存失败，容器需要 privileged"; exit 1; }
  stdbuf -oL iostat -x -y -t 1 > "$OUT/$mode.iostat" &  # 行缓冲，避免被 kill 时丢掉缓冲区里的记录
  IOSTAT_PID=$!
  date +%s.%N > "$OUT/$mode.start"  # 负载开始时间，报告据此剔除就绪前的预热 forward
  python "$PLUGIN_DIR/tests/service_acceptance.py" perf \
    --url "http://127.0.0.1:$PORT" --tp-size 8 --timeout 1800 --idle-wait-seconds 60 \
    --workload "$OUT/mixed.jsonl" --concurrency "$CONCURRENCY" --max-new-tokens "${OUTPUT_LEN##*-}" \
    --server-log "$OUT/$mode.log" --output "$OUT/$mode.json"
  kill "$IOSTAT_PID" 2>/dev/null || true; IOSTAT_PID=
  stop_server
  echo "== $mode 结束 $(date +%T) =="
}

echo "配置：EP=$EP MAX_RUNNING=$MAX_RUNNING CONCURRENCY=$CONCURRENCY WINDOW=$WINDOW POOL=$POOL" \
  "PROMPT_LEN=$PROMPT_LEN NUM_PROMPTS=$NUM_PROMPTS OUTPUT_LEN=$OUTPUT_LEN MODES=$MODES" | tee "$OUT/config.txt"
make_workload
for mode in $MODES; do
  run_mode "$mode"
done
python "$PLUGIN_DIR/tests/bench_report.py" "$OUT" | tee "$OUT/report.txt"
tar -czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "完成：结果 $OUT，压缩包 $OUT.tar.gz"