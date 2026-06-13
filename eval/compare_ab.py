# SPDX-License-Identifier: Apache-2.0
"""Compare two DeepSearchQA eval runs (efficiency + scores).

Usage:
    python -m eval.compare_ab eval/results/ab_blend_50 eval/results/ab_native_50
"""

import json
import statistics
import sys
from pathlib import Path


def load_reports(output_dir: Path) -> list[dict]:
    path = output_dir / "reports.jsonl"
    return [json.loads(line) for line in path.open()] if path.exists() else []


def load_metrics(output_dir: Path) -> dict:
    path = output_dir / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def latency_stats(reports: list[dict]) -> dict:
    ok = [r["latency_seconds"] for r in reports if not (r.get("error") or r.get("traceback"))]
    err = sum(1 for r in reports if r.get("error") or r.get("traceback"))
    if not ok:
        return {"n_ok": 0, "n_err": err}
    ok_sorted = sorted(ok)
    return {
        "n_ok": len(ok),
        "n_err": err,
        "mean_s": round(statistics.mean(ok), 1),
        "median_s": round(statistics.median(ok), 1),
        "p90_s": round(ok_sorted[int(len(ok_sorted) * 0.9) - 1], 1),
        "min_s": round(min(ok), 1),
        "max_s": round(max(ok), 1),
        "total_s": round(sum(ok), 1),
    }


def main() -> None:
    dirs = [Path(p) for p in sys.argv[1:3]]
    if len(dirs) != 2:
        raise SystemExit(__doc__)
    for label, d in zip(("A", "B"), dirs):
        reports = load_reports(d)
        metrics = load_metrics(d)
        print(f"\n=== {label}: {d} ===")
        print("latency:", json.dumps(latency_stats(reports), ensure_ascii=False))
        if metrics:
            flat = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, str))
            }
            print("metrics:", json.dumps(flat, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
