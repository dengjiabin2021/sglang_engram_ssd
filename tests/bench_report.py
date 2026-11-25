"""从 run_bench.sh 的结果目录提取对照指标：python bench_report.py <结果目录>。"""

import json
import sys
from pathlib import Path

CLIENT = (
    ("TTFT p50 (ms)", "client_ttft_p50_ms"), ("TTFT p95 (ms)", "client_ttft_p95_ms"),
    ("TPOT p50 (ms)", "client_tpot_p50_ms"), ("TPOT p95 (ms)", "client_tpot_p95_ms"),
    ("输入吞吐 (tok/s)", "client_prompt_tokens_per_second"), ("输出吞吐 (tok/s)", "client_output_tokens_per_second"),
)
PLUGIN = (
    ("窗口命中率", "window_hit_rate"), ("区间预测命中率", "span_match_rate"),
    ("prefill 平均等待 (ms)", "avg_prefill_wait_ms"), ("decode 平均等待 (ms)", "avg_decode_wait_ms"),
    ("紧急读行数", "urgent_rows"), ("后台读行数", "read_rows"), ("后台读耗时 (ms)", "read_ms"),
    ("淘汰行数", "evicted_slots"), ("未用即淘汰行数", "evicted_unused_slots"),
)


def fmt(value):
    """数值保留两位小数，空值显示为 -。"""
    return "-" if value is None else f"{value:.2f}"


def iostat(path):
    """返回读次数最多的块设备名、平均 r/s 和平均读带宽 MB/s；文件不存在时返回 None。"""
    if not path.exists():
        return None
    columns, totals = None, {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if fields and fields[0].startswith("Device"):
            columns = fields
        elif columns and len(fields) == len(columns):
            sums = totals.setdefault(fields[0], [0.0, 0.0, 0])
            sums[0] += float(fields[columns.index("r/s")])
            sums[1] += float(fields[columns.index("rkB/s")])
            sums[2] += 1
    if not totals:
        return None
    device, (reads, kbytes, count) = max(totals.items(), key=lambda item: item[1][0])
    return device, reads / count, kbytes / count / 1024


def timing(log, start):
    """按 forward 序号合并各 rank 的 Engram 计时日志，只保留负载开始后、8 个 rank 都有记录的 forward。

    返回 [(各层耗时取各 rank 最大值, 各 rank 各层紧急缺行数)]；没有计时日志时返回空列表。
    """
    if not log.exists():
        return []
    forwards = {}
    for line in log.read_text(errors="replace").splitlines():
        _, sep, body = line.partition("Engram timing: ")
        if sep:
            record = json.loads(body)
            forwards.setdefault(record["forward"], []).append(record)
    result = []
    for records in forwards.values():
        if len(records) != 8 or min(r["t"] for r in records) < start:
            continue
        layers = len(records[0]["ms"])
        ms = [max(r["ms"][i] for r in records) for i in range(layers)]
        result.append((ms, [r["urgent"] for r in records]))
    return result


def print_timing(root, names):
    """打印各组 prefill forward 的 Engram 每层 GPU 耗时；预取组按全命中 / 有紧急读 / 未用预取缓冲分开。"""
    groups = {}
    for mode in ("ddr", "sync", "prefetch"):
        start_file = root / f"{mode}.start"
        if not start_file.exists():
            continue
        for ms, urgent in timing(root / f"{mode}.log", float(start_file.read_text())):
            if mode != "prefetch":
                name = names[mode]
            elif any(u is None for rank in urgent for u in rank):
                name = "预取-未用预取缓冲"
            elif any(u for rank in urgent for u in rank):
                name = "预取-有紧急读"  # 至少一个 rank 的某层有紧急缺行
            else:
                name = "预取-全命中"  # 8 个 rank 各层都不需要紧急读
            groups.setdefault(name, []).append(ms)
    if not groups:
        return
    print("\nprefill Engram 每层 GPU 耗时（含 TP all-reduce；每个 forward 取 8 个 rank 的最大值，再求平均）")
    print(f"{'组':<20}{'forward 数':>10}{'层1 (ms)':>12}{'层14 (ms)':>12}")
    for name, rows in groups.items():
        means = [sum(row[i] for row in rows) / len(rows) for i in range(2)]
        print(f"{name:<20}{len(rows):>10}" + "".join(f"{fmt(value):>12}" for value in means))


def main():
    """打印各组服务的客户端指标、预取隐藏比例、预取组 8 个 rank 平均的插件统计和 SSD 读取量。"""
    root = Path(sys.argv[1])
    reports = {
        mode: json.loads((root / f"{mode}.json").read_text())
        for mode in ("ddr", "sync", "prefetch") if (root / f"{mode}.json").exists()
    }
    names = {"ddr": "DDR", "sync": "同步SSD", "prefetch": "预取"}
    print(f"{'指标':<24}" + "".join(f"{names[mode]:>14}" for mode in reports))
    for label, key in CLIENT:
        print(f"{label:<24}" + "".join(f"{fmt(reports[mode]['summary'][key]):>14}" for mode in reports))
    print("失败请求数".ljust(24) + "".join(f"{len(reports[m]['summary']['failed_request_ids']):>14}" for m in reports))

    if {"ddr", "sync", "prefetch"} <= reports.keys():
        # 隐藏比例 = 预取比同步 SSD 少花的时间 / 同步 SSD 比 DDR 多花的时间；分母过小时不计算。
        print("\n预取隐藏比例（(同步SSD - 预取) / (同步SSD - DDR)）")
        for label, key in (("TPOT p50", "client_tpot_p50_ms"), ("TTFT p50", "client_ttft_p50_ms")):
            ddr, sync, prefetch = (reports[m]["summary"][key] for m in ("ddr", "sync", "prefetch"))
            if None in (ddr, sync, prefetch):
                continue  # 只输出 1 个 token 的负载没有 TPOT
            gap = sync - ddr
            print(f"{label:<24}{fmt((sync - prefetch) / gap * 100 if abs(gap) > 1e-6 else None):>13}%")

    if "prefetch" in reports:
        ranks = list(reports["prefetch"]["plugin_counter_deltas_by_rank"].values())
        print(f"\n预取插件统计（{len(ranks)} 个 rank 平均，本轮差值）")
        for label, key in PLUGIN:
            values = [rank[key] for rank in ranks if rank.get(key) is not None]
            print(f"{label:<24}{fmt(sum(values) / len(values) if values else None):>14}")

    print_timing(root, names)

    print("\nSSD 读取（iostat，读次数最多的设备）")
    for mode in reports:
        stat = iostat(root / f"{mode}.iostat")
        print(f"{names[mode]:<10}" + ("无数据" if stat is None else f"{stat[0]}  r/s={stat[1]:.0f}  读带宽={stat[2]:.1f} MB/s"))


if __name__ == "__main__":
    main()
