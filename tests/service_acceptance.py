"""在已启动的目标服务上手动验收 A6–A8，记录 token 一致性、HTTP 时延及压力结果。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics
import threading
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid


_STATS_MARKER = "Engram SSD sched prefetch stats: "
_COUNTERS = (
    "submitted_rows", "dedup_saved_rows", "forward_hits", "forward_misses",
    "miss_read_ms", "prefill_wait_ms", "prefill_waits", "decode_wait_ms", "decode_waits",
    "urgent_rows", "predicted_spans", "matched_spans", "read_ms", "read_rows",
    "evicted_slots", "evicted_unused_slots",
)


def _load_workload(path: Path) -> tuple[list[dict[str, Any]], str]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if not isinstance(case.get("id"), str):
            raise ValueError("Each workload row requires a string id")
        if ("text" in case) == ("input_ids" in case):
            raise ValueError(f"{case['id']}: provide exactly one of text or input_ids")
        if "text" in case and (not isinstance(case["text"], str) or not case["text"]):
            raise ValueError(f"{case['id']}: text must be nonempty")
        if "input_ids" in case and (
            not case["input_ids"] or not all(isinstance(i, int) and i >= 0 for i in case["input_ids"])
        ):
            raise ValueError(f"{case['id']}: input_ids must be a nonempty list of nonnegative integers")
        # 可选的逐条输出长度，覆盖命令行 --max-new-tokens，用于让请求错开结束。
        if "max_new_tokens" in case and not (isinstance(case["max_new_tokens"], int) and case["max_new_tokens"] > 0):
            raise ValueError(f"{case['id']}: max_new_tokens must be a positive integer")
        cases.append({key: case[key] for key in ("id", "text", "input_ids", "max_new_tokens") if key in case})
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Workload must contain nonempty cases with distinct ids")
    encoded = json.dumps(cases, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return cases, hashlib.sha256(encoded).hexdigest()


def _request_json(url: str, body: dict[str, Any] | None, timeout: float) -> Any:
    data = None if body is None else json.dumps(body).encode()
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        content = response.read()
        return json.loads(content) if content else None


def _generate(
    *, url: str, case: dict[str, Any], rid: str, max_new_tokens: int,
    timeout: float, command: str, started: threading.Event,
) -> dict[str, Any]:
    """发起一条流式生成请求并记录实际 token 和客户端时延。

    输入：url 为服务地址，case 为输入用例，rid 为请求标识；max_new_tokens 和 timeout
        分别限制输出长度和 HTTP 等待秒数，用例自带 max_new_tokens 时以用例为准。
        command 选择采样及中止检查，started 在发出 HTTP 前置位。
    输出：含 token ID、完成原因、TTFT、TPOT 和总延迟的记录；请求错误写入 error。
    """
    max_new_tokens = case.get("max_new_tokens", max_new_tokens)
    body = {key: case[key] for key in ("text", "input_ids") if key in case}
    # 多 token 性能轮固定输出长度；压力轮也禁用 EOS，给 abort 和 retract 留出时间。
    ignore_eos = command == "stress" or (command == "perf" and max_new_tokens > 1)
    body.update({
        "rid": rid, "stream": True, "return_logprob": True,
        "logprob_start_len": -1, "top_logprobs_num": 0, "return_text_in_logprobs": False,
        "sampling_params": {
            "temperature": 0, "max_new_tokens": max_new_tokens, "ignore_eos": ignore_eos,
        },
    })
    request = Request(
        url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
    )
    output_ids: list[int] = []
    metadata: dict[str, Any] = {}
    first_token_ms = None
    began = time.perf_counter()
    started.set()
    try:
        with urlopen(request, timeout=timeout) as response:
            event_lines = []
            for raw in response:
                line = raw.decode("utf-8").rstrip("\r\n")
                if line.startswith("data:"):
                    event_lines.append(line[5:].lstrip())
                    continue
                if line or not event_lines:
                    continue
                data = "\n".join(event_lines)
                event_lines.clear()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if "error" in event:
                    raise RuntimeError(f"Generate error: {event['error']}")
                metadata.update(event.get("meta_info", {}))
                logprobs = metadata.get("output_token_logprobs", [])
                token_ids = [int(item[1]) for item in logprobs]
                total = metadata.get("output_token_logprobs_length", len(token_ids))
                offset = int(total) - len(token_ids)
                if offset < 0 or offset > len(output_ids):
                    raise RuntimeError("Missing or inconsistent streaming token/logprob positions")
                overlap = min(len(output_ids) - offset, len(token_ids))
                if output_ids[offset : offset + overlap] != token_ids[:overlap]:
                    raise RuntimeError("Streaming token IDs changed after they were emitted")
                output_ids.extend(token_ids[overlap:])
                if token_ids and first_token_ms is None:
                    first_token_ms = (time.perf_counter() - began) * 1000
                if event.get("output_ids") is not None:
                    reported = event["output_ids"]
                    if reported != output_ids and reported != token_ids:
                        raise RuntimeError("output_ids disagrees with actual output logprob token IDs")
        finish = metadata.get("finish_reason")
        if finish is None:
            raise RuntimeError("Stream ended without a final finish_reason")
        if finish.get("type") == "abort" and command != "stress":
            raise RuntimeError("A non-stress generation was unexpectedly aborted")
        if not output_ids and finish.get("type") != "abort":
            raise RuntimeError("No output token IDs returned; return_logprob is required")
        if command == "perf" and len(output_ids) != max_new_tokens:
            raise RuntimeError(f"Expected {max_new_tokens} output tokens, got {len(output_ids)}")
        error = None
    except (HTTPError, URLError, OSError, ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
        finish = metadata.get("finish_reason")
    latency_ms = (time.perf_counter() - began) * 1000
    # 首个 token 已计入 TTFT，余下输出间隔共 len(output_ids) - 1 个。
    tpot_ms = (
        (latency_ms - first_token_ms) / (len(output_ids) - 1)
        if first_token_ms is not None and len(output_ids) > 1 else None
    )
    return {
        "id": case["id"], "rid": rid, "output_token_ids": output_ids,
        "finish_reason": finish, "error": error,
        "ttft_ms": first_token_ms, "tpot_ms": tpot_ms, "latency_ms": latency_ms,
        "prompt_tokens": metadata.get("prompt_tokens"), "cached_tokens": metadata.get("cached_tokens"),
        "num_retractions": metadata.get("num_retractions", 0),
    }


def _read_server_log(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"provided": False}
    snapshots = {}
    idle_counts: dict[str, int] = {}
    prefill_rates = []
    retract_count = 0
    queue_peak = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if _STATS_MARKER in line:
            stats = json.loads(line.split(_STATS_MARKER, 1)[1])
            prefix_rank = re.search(r"\bTP(\d+)\b", line)
            rank = stats.get("tp_rank", stats.get("rank", prefix_rank.group(1) if prefix_rank else "unknown"))
            snapshots[str(rank)] = stats
            if stats.get("phase") == "idle" and stats.get("referenced_slots") == 0:
                idle_counts[str(rank)] = idle_counts.get(str(rank), 0) + 1
        if "Prefill batch" in line:
            rate = re.search(r"input throughput \(token/s\):\s*([\d.]+)", line)
            queue = re.search(r"#queue-req:\s*(\d+)", line)
            if rate:
                prefill_rates.append(float(rate.group(1)))
            if queue:
                queue_peak = max(queue_peak, int(queue.group(1)))
        if "Testing retraction." in line or "Retract requests." in line:
            count = re.search(r"#retracted_reqs:\s*(\d+)", line)
            if count:
                retract_count += int(count.group(1))
    return {
        "provided": True, "path": str(path), "latest_pool_snapshots_by_rank": snapshots,
        "idle_zero_reference_snapshots_by_rank": idle_counts,
        "server_prefill_input_throughput_samples": prefill_rates,
        "retracted_requests_logged": retract_count, "prefill_queue_peak_logged": queue_peak,
    }


def _log_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """计算单轮服务日志的逐 rank 统计。

    输入：before、after 为同一服务在请求前后的日志快照。
    输出：两次均有快照的 rank 的累计计数差值和派生指标，分母为零时相应比率为空。
    """
    deltas = {}
    old = before.get("latest_pool_snapshots_by_rank", {})
    for rank, current in after.get("latest_pool_snapshots_by_rank", {}).items():
        previous = old.get(rank)
        if previous is None:
            continue
        delta = {
            name: current[name] - previous[name]
            for name in _COUNTERS if name in current and name in previous
        }
        if any(value < 0 for value in delta.values()):
            raise RuntimeError("Server counters decreased; use one unchanged server per acceptance run")
        if "forward_hits" in delta and "forward_misses" in delta:
            count = delta["forward_hits"] + delta["forward_misses"]
            delta["forward_pool_hit_rate"] = delta["forward_hits"] / count if count else None
        # 窗口命中率已经按唯一 owner 行累计，直接保留结束时的控制器口径。
        if "window_hit_rate" in current:
            delta["window_hit_rate"] = current["window_hit_rate"]
        for metric, amount, count in (
            ("avg_prefill_wait_ms", "prefill_wait_ms", "prefill_waits"),
            ("avg_decode_wait_ms", "decode_wait_ms", "decode_waits"),
            ("span_match_rate", "matched_spans", "predicted_spans"),
            ("wasted_read_ratio", "evicted_unused_slots", "read_rows"),
        ):
            if amount in delta and count in delta:
                delta[metric] = delta[amount] / delta[count] if delta[count] else None
        deltas[rank] = delta
    return deltas


def _new_idle_snapshots(before: dict[str, Any], after: dict[str, Any], tp_size: int) -> bool:
    old = before.get("idle_zero_reference_snapshots_by_rank", {})
    new = after.get("idle_zero_reference_snapshots_by_rank", {})
    latest = after.get("latest_pool_snapshots_by_rank", {})
    return all(
        new.get(str(rank), 0) > old.get(str(rank), 0)
        and latest.get(str(rank), {}).get("phase") == "idle"
        and latest[str(rank)].get("referenced_slots") == 0
        for rank in range(tp_size)
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    """按固定负载并发发起请求并汇总一轮验收结果。

    输入：args 提供负载路径、服务地址、并发、输出长度、日志和可选的中止参数。
    输出：可写入 JSON 的逐条记录、客户端统计和逐 rank 日志计数差值。
    """
    cases, workload_hash = _load_workload(path=args.workload)
    before_log = _read_server_log(path=args.server_log)
    records = []
    abort_attempts = []
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        targets = []
        for index, case in enumerate(cases):
            rid = f"engram-accept-{uuid.uuid4().hex}"
            started = threading.Event()
            future = executor.submit(
                _generate, url=args.url.rstrip("/"), case=case, rid=rid,
                max_new_tokens=args.max_new_tokens, timeout=args.timeout,
                command=args.command, started=started,
            )
            futures[future] = case["id"]
            if args.command == "stress" and index < args.concurrency and index % 4 == 0:
                targets.append((rid, started))
        if targets:
            time.sleep(args.abort_delay_ms / 1000)
        for rid, started in targets:
            if not started.wait(timeout=args.timeout):
                abort_attempts.append({"rid": rid, "error": "Request worker did not start"})
                continue
            try:
                _request_json(url=args.url.rstrip("/") + "/abort_request", body={"rid": rid}, timeout=args.timeout)
                abort_attempts.append({"rid": rid, "error": None})
            except (HTTPError, URLError, OSError) as exc:
                abort_attempts.append({"rid": rid, "error": str(exc)})
        for future in as_completed(futures):
            records.append(future.result())
    elapsed = time.perf_counter() - began
    after_log = _read_server_log(path=args.server_log)
    if args.server_log is not None and (
        args.command == "stress" or (args.command == "perf" and before_log.get("latest_pool_snapshots_by_rank"))
    ):
        # 性能轮也收齐末尾 decode 的计数。等待日志的时间不计入客户端吞吐。
        deadline = time.monotonic() + args.idle_wait_seconds
        while not _new_idle_snapshots(before=before_log, after=after_log, tp_size=args.tp_size):
            if time.monotonic() >= deadline:
                break
            time.sleep(0.25)
            after_log = _read_server_log(path=args.server_log)
    ttfts = sorted(row["ttft_ms"] for row in records if row["ttft_ms"] is not None)
    tpots = sorted(row["tpot_ms"] for row in records if row["tpot_ms"] is not None)
    prompt_tokens = sum(int(row["prompt_tokens"] or 0) for row in records if row["error"] is None)
    output_tokens = sum(len(row["output_token_ids"]) for row in records if row["error"] is None)
    errors = [row["id"] for row in records if row["error"]]
    abort_rids = {row["rid"] for row in abort_attempts}
    aborted = [row["rid"] for row in records if (row["finish_reason"] or {}).get("type") == "abort"]
    retracts = sum(int(row["num_retractions"] or 0) for row in records)
    logged_retracts = after_log.get("retracted_requests_logged", 0) - before_log.get("retracted_requests_logged", 0)
    return {
        "schema_version": 1, "command": args.command, "url": args.url,
        "created_utc": datetime.now(timezone.utc).isoformat(), "workload_sha256": workload_hash,
        "concurrency": args.concurrency, "max_new_tokens": args.max_new_tokens,
        "expected_tp_size": args.tp_size,
        "summary": {
            "requests": len(records), "failed_request_ids": errors, "wall_seconds": elapsed,
            "client_ttft_mean_ms": statistics.mean(ttfts) if ttfts else None,
            "client_ttft_p50_ms": statistics.median(ttfts) if ttfts else None,
            "client_ttft_p95_ms": ttfts[max(0, (len(ttfts) * 95 + 99) // 100 - 1)] if ttfts else None,
            "client_tpot_mean_ms": statistics.mean(tpots) if tpots else None,
            "client_tpot_p50_ms": statistics.median(tpots) if tpots else None,
            "client_tpot_p95_ms": tpots[max(0, (len(tpots) * 95 + 99) // 100 - 1)] if tpots else None,
            "client_prompt_tokens_per_second": prompt_tokens / elapsed,
            "client_output_tokens_per_second": output_tokens / elapsed,
            "prefill_only_workload": args.max_new_tokens == 1,
            "abort_requests_sent": len(abort_attempts), "abort_finish_reasons_observed": len(aborted),
            "abort_targets_not_observed": sorted(abort_rids.difference(aborted)),
            "unexpected_abort_rids": sorted(set(aborted).difference(abort_rids)),
            "http_retractions_observed": retracts, "log_retractions_observed": logged_retracts,
            "idle_zero_references_observed_all_ranks": _new_idle_snapshots(
                before=before_log, after=after_log, tp_size=args.tp_size,
            ),
        },
        "records": sorted(records, key=lambda row: row["id"]), "abort_attempts": abort_attempts,
        "server_log": after_log, "plugin_counter_deltas_by_rank": _log_delta(before=before_log, after=after_log),
    }


def record_baseline(args: argparse.Namespace) -> dict[str, Any]:
    """A6：按 args 运行基线服务的固定负载，返回 greedy 输出 token ID 和结束原因供对照。"""
    report = _run(args=args)
    report["acceptance_passed"] = not report["summary"]["failed_request_ids"]
    return report


def compare_outputs(args: argparse.Namespace) -> dict[str, Any]:
    """A6：优化服务逐请求逐 token 对照先前记录，不同时加载两套模型。"""
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if not baseline.get("acceptance_passed") or baseline.get("command") != "record":
        raise ValueError("Baseline must be a successful record run")
    cases, workload_hash = _load_workload(path=args.workload)
    del cases
    if baseline["workload_sha256"] != workload_hash or baseline["max_new_tokens"] != args.max_new_tokens:
        raise ValueError("Workload or max_new_tokens differs from the baseline")
    if baseline["concurrency"] != args.concurrency:
        raise ValueError("A6 baseline and comparison must use the same concurrency")
    report = _run(args=args)
    reference = {row["id"]: row for row in baseline["records"]}
    mismatches = []
    for row in report["records"]:
        expected = reference[row["id"]]
        if row["output_token_ids"] != expected["output_token_ids"] or row["finish_reason"] != expected["finish_reason"]:
            mismatches.append(row["id"])
    report["summary"]["token_mismatch_ids"] = mismatches
    report["acceptance_passed"] = not mismatches and not report["summary"]["failed_request_ids"]
    return report


def run_stress(args: argparse.Namespace) -> dict[str, Any]:
    """A7：并发长 prompt、按 rid 中止及真实 retract 观测；不把无观测当作通过。"""
    report = _run(args=args)
    summary = report["summary"]
    observed = summary["http_retractions_observed"] > 0 or summary["log_retractions_observed"] > 0
    try:
        _request_json(url=args.url.rstrip("/") + "/health", body=None, timeout=10)
        summary["service_healthy_after_stress"] = True
    except (HTTPError, URLError, OSError):
        summary["service_healthy_after_stress"] = False
    summary["retract_coverage"] = "observed" if observed else "not_observed"
    idle_confirmed = summary["idle_zero_references_observed_all_ranks"]
    summary["idle_pool_reference_cleanup"] = "confirmed" if idle_confirmed else "not_observed"
    http_passed = (
        not summary["failed_request_ids"] and not summary["abort_targets_not_observed"]
        and not summary["unexpected_abort_rids"]
        and not any(row["error"] for row in report["abort_attempts"])
        and summary["service_healthy_after_stress"]
    )
    report["http_stress_passed"] = http_passed
    report["full_A7_passed"] = http_passed and observed and idle_confirmed
    summary["full_A7_passed"] = report["full_A7_passed"]
    report["acceptance_passed"] = report["full_A7_passed"] if args.require_retract else http_passed
    return report


def measure_perf(args: argparse.Namespace) -> dict[str, Any]:
    """A8：按 args 的并发和固定输出长度运行性能负载，返回时延、吞吐及日志报告。"""
    report = _run(args=args)
    report["acceptance_passed"] = not report["summary"]["failed_request_ids"]
    return report


def main() -> int:
    """从命令行读取验收参数，写报告并返回该轮验收的退出码。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "compare", "perf", "stress"))
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--abort-delay-ms", type=float, default=500)
    parser.add_argument("--idle-wait-seconds", type=float, default=10)
    parser.add_argument("--require-retract", action="store_true")
    args = parser.parse_args()
    if min(args.concurrency, args.max_new_tokens, args.tp_size, args.timeout) <= 0:
        parser.error("concurrency, max-new-tokens and timeout must be positive; abort-delay-ms must be nonnegative")
    if args.abort_delay_ms < 0 or args.idle_wait_seconds < 0:
        parser.error("abort-delay-ms and idle-wait-seconds must be nonnegative")
    if args.command == "compare" and args.baseline is None:
        parser.error("compare requires --baseline")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    functions = {"record": record_baseline, "compare": compare_outputs, "perf": measure_perf, "stress": run_stress}
    report = functions[args.command](args=args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), **report["summary"]}, ensure_ascii=False, indent=2))
    if args.command == "stress" and not report["full_A7_passed"]:
        print("A7 remains incomplete: require observed retraction and new idle zero-reference snapshots on all ranks.")
    return 0 if report["acceptance_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
