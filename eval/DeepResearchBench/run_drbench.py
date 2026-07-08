"""在 DeepResearchBench 题库上跑 demo + 用其 scorer(RACE 报告质量 / FACT 引用核验)打分。

两段:
  --mode generate : 跑 demo(REPORT_MODE=detailed_cited)→ 写 DRBench raw_data/<tag>.jsonl(id/prompt/article)(需 vLLM 服务器)
  --mode score    : 子进程调 DRBench 的 RACE(deepresearch_bench_race.py)+ FACT(utils.extract→dedup→scrape→validate→stat)
  --mode all      : 两段都跑

评委经 scorer_env 指到 DashScope(OPENAI_BASE_URL + RACE_MODEL/FACT_MODEL=kimi)。
RACE 不抓网页(只比参考报告);FACT 的 scrape 用 DRBench 自带抓取(见 README 的缓存复用 TODO)。
"""
import os
import sys
import json
import asyncio
import argparse
import subprocess
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[2]
# 自包含:默认指向本目录(vendored race脚本+utils+prompt+data),不再跳外部 repo
DRBENCH = Path(os.getenv("DRBENCH_DIR", str(Path(__file__).resolve().parent)))
sys.path.insert(0, str(DEMO_ROOT))
from eval.benchmarks import judge_adapter as JA  # noqa: E402
from eval.benchmarks import harvest_gen as HG    # noqa: E402

QUERY_FILE = DRBENCH / "data/prompt_data/query.jsonl"
RAW_DATA_DIR = DRBENCH / "data/test_data/raw_data"
# 跑的 eval 结果统一放 eval/results/(不再放 eval/benchmarks/results)。可 DRBENCH_RESULTS_DIR 覆盖。
RESULTS_DIR = Path(os.getenv("DRBENCH_RESULTS_DIR", str(DEMO_ROOT / "eval/results/drbench")))


# --------------------- generate + 每题 harvest(子进程)---------------------
async def generate_harvest(tag, n, only_lang, concurrency, model, base_url):
    """子进程跑 demo CLI:每题留 harvest.jsonl + llm_calls.jsonl + report.md,再汇总成 raw_data/<tag>.jsonl。"""
    out_dir = RESULTS_DIR / tag
    rows = [json.loads(l) for l in open(QUERY_FILE) if l.strip()]
    if only_lang:
        rows = [r for r in rows if r.get("language") == only_lang]
    rows = rows[:n]
    items = [(r["prompt"], str(r["id"])) for r in rows]
    await HG.run_batch(items, out_dir, model, base_url, concurrency)
    # 汇总成 DRBench scorer 要的 raw_data/<tag>.jsonl(id/prompt/article)
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(RAW_DATA_DIR / f"{tag}.jsonl", "w") as f:
        for r in rows:
            rep = out_dir / f"q{r['id']}" / "report.md"
            article = rep.read_text(encoding="utf-8") if (rep.exists() and rep.stat().st_size > 0) else ""
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "article": article},
                               ensure_ascii=False) + "\n")
    print(f"[generate_harvest] raw_data/{tag}.jsonl 已汇总", flush=True)


# --------------------------- generate ---------------------------
async def generate(tag, n, only_lang, concurrency):
    os.environ["REPORT_MODE"] = "detailed_cited"
    # 详细长报告(max_tokens 1万)生成耗时长,默认 120s 会 ReadTimeout → 放宽
    os.environ.setdefault("LLM_TIMEOUT", "900")
    from eval.deepresearchqa.run_deepsearchqa import build_parser, build_config, EvalRunner
    from deep_researcher_demo.progress import NullProgressReporter
    config = build_config(build_parser().parse_args([]))
    runner = EvalRunner(config=config, quiet=True)

    rows = [json.loads(l) for l in open(QUERY_FILE) if l.strip()]
    if only_lang:
        rows = [r for r in rows if r.get("language") == only_lang]
    rows = rows[:n]
    out_path = RAW_DATA_DIR / f"{tag}.jsonl"
    done = {}
    if out_path.exists():                              # 断点续
        done = {json.loads(l)["id"]: json.loads(l) for l in open(out_path) if l.strip()}
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        qid, prompt = row["id"], row["prompt"]
        if qid in done and done[qid].get("article"):
            return done[qid]
        async with sem:
            wf = runner.build_workflow(reporter=NullProgressReporter(), cache_key=f"drbench_{qid}")
            try:
                result = await wf.run(prompt)          # 原始 prompt
                return {"id": qid, "prompt": prompt, "article": result.final_report or ""}
            except Exception as e:                     # noqa: BLE001
                return {"id": qid, "prompt": prompt, "article": "", "error": str(e)[:120]}

    results = await asyncio.gather(*[one(r) for r in rows])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ok = sum(1 for r in results if r.get("article"))
    print(f"[generate] {ok}/{len(results)} 报告 -> {out_path}", flush=True)
    return out_path


# --------------------------- score ---------------------------
def score(tag, metrics, workers):
    env = JA.scorer_env()
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"tag": tag, "outputs": {}}

    if "race" in metrics:
        race_out = out_dir / "race"
        race_out.mkdir(exist_ok=True)
        cmd = [sys.executable, "-u", "deepresearch_bench_race.py", tag,
               "--raw_data_dir", str(RAW_DATA_DIR), "--max_workers", str(workers),
               "--query_file", str(QUERY_FILE), "--output_dir", str(race_out)]
        print("[race]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(DRBENCH), env=env, check=False)
        summary["outputs"]["race"] = str(race_out)
        summary["race"] = _collect_race(race_out)

    if "fact" in metrics:
        fact_out = out_dir / "fact"
        fact_out.mkdir(exist_ok=True)
        raw = str(RAW_DATA_DIR / f"{tag}.jsonl")
        steps = [
            ["-m", "utils.extract", "--raw_data_path", raw,
             "--output_path", f"{fact_out}/extracted.jsonl", "--query_data_path", str(QUERY_FILE),
             "--n_total_process", str(workers)],
            ["-m", "utils.deduplicate", "--raw_data_path", f"{fact_out}/extracted.jsonl",
             "--output_path", f"{fact_out}/deduplicated.jsonl", "--query_data_path", str(QUERY_FILE),
             "--n_total_process", str(workers)],
            ["-m", "utils.scrape", "--raw_data_path", f"{fact_out}/deduplicated.jsonl",
             "--output_path", f"{fact_out}/scraped.jsonl", "--n_total_process", str(workers)],
            ["-m", "utils.validate", "--raw_data_path", f"{fact_out}/scraped.jsonl",
             "--output_path", f"{fact_out}/validated.jsonl", "--query_data_path", str(QUERY_FILE),
             "--n_total_process", str(workers)],
            ["-m", "utils.stat", "--input_path", f"{fact_out}/validated.jsonl",
             "--output_path", f"{fact_out}/fact_result.txt"],
        ]
        for st in steps:
            print("[fact]", st[1], flush=True)
            subprocess.run([sys.executable, "-u"] + st, cwd=str(DRBENCH), env=env, check=False)
        summary["outputs"]["fact"] = str(fact_out)
        fr = fact_out / "fact_result.txt"
        summary["fact"] = fr.read_text()[:500] if fr.exists() else None

    sp = out_dir / "summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] -> {sp}\n{json.dumps(summary, ensure_ascii=False, indent=2)[:600]}", flush=True)
    return summary


def _collect_race(race_out: Path):
    """从 race_result.txt 读 Overall + 4 维(RACE 自带的聚合);兜底用 raw_results.jsonl 的 overall_score 均值。"""
    txt = race_out / "race_result.txt"
    if txt.exists():
        out = {}
        for line in txt.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                try:
                    out[k.strip()] = round(float(v.strip()), 4)
                except ValueError:
                    pass
        if out:
            return out
    raw = race_out / "raw_results.jsonl"
    if raw.exists():
        scores = [json.loads(l).get("overall_score") for l in open(raw) if l.strip()]
        scores = [s for s in scores if isinstance(s, (int, float))]
        if scores:
            return {"Overall Score": round(sum(scores) / len(scores), 4), "n": len(scores)}
    return {"note": "见 race 输出目录,未解析到分数字段"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--mode", choices=["generate", "score", "all"], default="all")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--only-lang", choices=["zh", "en"], default=None)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--metrics", default="race,fact")
    ap.add_argument("--harvest", action="store_true", help="子进程跑、每题留 harvest+计时(需 vLLM server)")
    ap.add_argument("--model", default="Qwen3-32B")
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    args = ap.parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if args.mode in ("generate", "all"):
        if args.harvest:
            asyncio.run(generate_harvest(args.tag, args.n, args.only_lang,
                                         args.concurrency, args.model, args.base_url))
        else:
            asyncio.run(generate(args.tag, args.n, args.only_lang, args.concurrency))
    if args.mode in ("score", "all"):
        score(args.tag, metrics, args.workers)


if __name__ == "__main__":
    main()
