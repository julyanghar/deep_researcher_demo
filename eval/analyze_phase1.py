# SPDX-License-Identifier: Apache-2.0
"""Phase-1 efficiency analysis: decompose where A (CacheBlend) vs B (native)
spend time.

Inputs per mode:
  - reports.jsonl                      (per-sample end-to-end latency)
  - <calls>.jsonl                      (per-LLM-call client timing, by role tag)
  - metrics_before.txt / after.txt     (vLLM /metrics snapshots; histogram diff)
  - server log (mode A only)           ("Blend (KV load + fuse) took X ms")
  - gpu_monitor.csv                    (10s GPU samples; averaged per window)

Usage:
  python -m eval.analyze_phase1 \
    --a-dir eval/results/ab_blend_10 --b-dir eval/results/ab_native_10 \
    --a-calls eval/results/ab_meta/p1A_calls.jsonl \
    --b-calls eval/results/ab_meta/p1B_calls.jsonl \
    --a-metrics eval/results/ab_meta/p1A_metrics \
    --b-metrics eval/results/ab_meta/p1B_metrics \
    --a-serverlog /home/yilin/tmp/vllm_p1A_serve.log \
    --gpu-csv eval/results/ab_meta/gpu_monitor.csv
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

HIST_METRICS = [
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:time_to_first_token_seconds",
    "vllm:time_per_output_token_seconds",
]


def parse_prom(path: Path) -> dict[str, float]:
    """Sum up _sum and _count for each histogram metric (across engines)."""
    out: dict[str, float] = defaultdict(float)
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            continue
        for m in HIST_METRICS:
            for suffix in ("_sum", "_count"):
                if line.startswith(m + suffix):
                    try:
                        out[m + suffix] += float(line.rsplit(" ", 1)[-1])
                    except ValueError:
                        pass
    return out


def metric_means(before: Path, after: Path) -> dict[str, dict]:
    b, a = parse_prom(before), parse_prom(after)
    res = {}
    for m in HIST_METRICS:
        dsum = a.get(m + "_sum", 0) - b.get(m + "_sum", 0)
        dcnt = a.get(m + "_count", 0) - b.get(m + "_count", 0)
        if dcnt > 0:
            res[m.removeprefix("vllm:")] = {
                "count": int(dcnt),
                "mean_s": round(dsum / dcnt, 4),
                "total_s": round(dsum, 1),
            }
    return res


def call_stats(path: Path) -> tuple[dict, tuple[float, float] | None]:
    by_tag: dict[str, list[dict]] = defaultdict(list)
    t_min, t_max = float("inf"), 0.0
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            by_tag[r["tag"]].append(r)
            t_min = min(t_min, r["start_ts"])
            t_max = max(t_max, r["end_ts"])
    table = {}
    for tag, rows in sorted(by_tag.items()):
        el = [r["elapsed_s"] for r in rows]
        ct = [r.get("completion_tokens") or 0 for r in rows]
        pt = [r.get("prompt_tokens") or 0 for r in rows]
        table[tag] = {
            "n": len(rows),
            "mean_s": round(statistics.mean(el), 2),
            "total_s": round(sum(el), 1),
            "avg_prompt_tok": int(statistics.mean(pt)) if pt else 0,
            "avg_completion_tok": int(statistics.mean(ct)) if ct else 0,
        }
    window = (t_min, t_max) if t_max > 0 else None
    return table, window


def report_latency(d: Path) -> dict:
    path = d / "reports.jsonl"
    # Reruns may append; keep only the latest record per sample.
    latest: dict = {}
    for line in path.open():
        r = json.loads(line)
        latest[r.get("sample_id")] = r
    ok = []
    err = 0
    for r in latest.values():
        if r.get("error") or r.get("traceback"):
            err += 1
        else:
            ok.append(r["latency_seconds"])
    if not ok:
        return {"n_ok": 0, "n_err": err}
    return {
        "n_ok": len(ok),
        "n_err": err,
        "mean_s": round(statistics.mean(ok), 1),
        "median_s": round(statistics.median(ok), 1),
        "min_s": round(min(ok), 1),
        "max_s": round(max(ok), 1),
        "total_s": round(sum(ok), 1),
    }


def blend_stats(server_log: Path) -> dict:
    if not server_log or not server_log.exists():
        return {}
    ms, toks = [], []
    load_ms, compute_ms = [], []
    pat = re.compile(r"Blend \(KV load \+ fuse\) took ([0-9.]+) ms for (\d+) hit")
    pat_split = re.compile(r"Blend timing: load=([0-9.]+) ms, compute=([0-9.]+) ms")
    for line in server_log.open(errors="ignore"):
        # Each TP rank logs the same blend; count one rank only.
        if "Worker_TP" in line and "Worker_TP0" not in line:
            continue
        m = pat.search(line)
        if m:
            ms.append(float(m.group(1)))
            toks.append(int(m.group(2)))
            continue
        m = pat_split.search(line)
        if m:
            load_ms.append(float(m.group(1)))
            compute_ms.append(float(m.group(2)))
    if not ms and not load_ms:
        return {}
    out = {}
    if ms:
        out.update(
            {
                "n_blends": len(ms),
                "mean_ms": round(statistics.mean(ms), 1),
                "max_ms": round(max(ms), 1),
                "total_s": round(sum(ms) / 1000, 1),
                "mean_hit_tokens": int(statistics.mean(toks)),
            }
        )
    if load_ms:
        out.update(
            {
                "mean_load_ms": round(statistics.mean(load_ms), 1),
                "mean_compute_ms": round(statistics.mean(compute_ms), 1),
                "load_share_pct": round(
                    100 * sum(load_ms) / max(sum(load_ms) + sum(compute_ms), 1e-9), 1
                ),
            }
        )
    return out


def gpu_window_stats(csv_path: Path, window: tuple[float, float] | None, gpus=(1, 3, 4, 5)) -> dict:
    if not csv_path.exists() or window is None:
        return {}
    import datetime

    t0, t1 = window
    ours, theirs = [], []
    for line in csv_path.open(errors="ignore"):
        if line.startswith("#") or line.startswith("timestamp"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            ts = datetime.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            idx = int(parts[1])
            util = float(parts[2].split()[0])
        except (ValueError, IndexError):
            continue
        if t0 <= ts <= t1:
            (ours if idx in gpus else theirs).append(util)
    out = {}
    if ours:
        out["our_gpus_mean_util_pct"] = round(statistics.mean(ours), 1)
    if theirs:
        out["other_gpus_mean_util_pct"] = round(statistics.mean(theirs), 1)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a-dir", required=True)
    p.add_argument("--b-dir", required=True)
    p.add_argument("--a-calls", required=True)
    p.add_argument("--b-calls", required=True)
    p.add_argument("--a-metrics", required=True, help="prefix; expects _before.txt/_after.txt")
    p.add_argument("--b-metrics", required=True)
    p.add_argument("--a-serverlog", default="")
    p.add_argument("--gpu-csv", default="")
    args = p.parse_args()

    for label, d, calls, metrics, serverlog in [
        ("A (CacheBlend)", args.a_dir, args.a_calls, args.a_metrics, args.a_serverlog),
        ("B (native prefix cache)", args.b_dir, args.b_calls, args.b_metrics, ""),
    ]:
        print(f"\n{'=' * 20} Mode {label} {'=' * 20}")
        print("[end-to-end per-sample]", json.dumps(report_latency(Path(d))))
        table, window = call_stats(Path(calls))
        print("[per-LLM-call by role]")
        for tag, s in table.items():
            print(f"  {tag:<36} {json.dumps(s)}")
        means = metric_means(Path(metrics + "_before.txt"), Path(metrics + "_after.txt"))
        print("[server-side phase times (vLLM /metrics diff)]")
        for mname, s in means.items():
            print(f"  {mname:<34} {json.dumps(s)}")
        if serverlog:
            bs = blend_stats(Path(serverlog))
            if bs:
                print("[blend = KV load + fuse (LMCache)]", json.dumps(bs))
        if args.gpu_csv:
            gw = gpu_window_stats(Path(args.gpu_csv), window)
            if gw:
                print("[GPU util in run window]", json.dumps(gw))


if __name__ == "__main__":
    main()
