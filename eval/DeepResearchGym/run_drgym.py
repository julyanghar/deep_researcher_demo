"""在 DeepResearchGym 题库上跑 demo + 用其 scorer(KPR/Quality/Citation 精召)打分。

两段:
  --mode generate : 跑 demo(REPORT_MODE=detailed_cited)→ 写 reports/<tag>/<id>.{a,q}(需 vLLM 服务器)
  --mode score    : 调 DRGym scorer 给 reports/<tag> 打分 → summary(只需 DashScope 评委)
  --mode all      : 两段都跑

DRGym scorer 的 evaluate_folder_async 都把 path_to_reports 作参数 → 直接 import 调用,
传我们的 reports 根目录即可,无需改 DRGym 源码(硬编码 path 只在它的 __main__)。
citation 抓网页用 demo 的 search 缓存替代 crawl4ai(monkey-patch crawl_urls)。
"""
import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[2]
# 自包含:默认指向本目录(vendored eval_*_async + queries + key_point),不再跳外部 repo
DRGYM = Path(os.getenv("DRGYM_DIR", str(Path(__file__).resolve().parent)))
sys.path.insert(0, str(DEMO_ROOT))
sys.path.insert(0, str(DRGYM))   # 让 import eval_kpr_async 等走本目录 vendored 副本

from eval.benchmarks import judge_adapter as JA  # noqa: E402
from eval.benchmarks import harvest_gen as HG    # noqa: E402

DEFAULT_QUERIES = DRGYM / "queries/researchy_queries_sample_doc_click_100.jsonl"
KEY_POINT_DIR = DRGYM / "key_point"


# --------------------- generate + 每题 harvest(子进程)---------------------
async def generate_harvest(tag, reports_root, queries, n, concurrency, model, base_url):
    """子进程跑 demo CLI:每题留 harvest.jsonl + llm_calls.jsonl + report.md,再落 scorer 格式 <id>.a/.q。"""
    out_dir = Path(reports_root) / tag
    rows = [json.loads(l) for l in open(queries) if l.strip()][:n]
    items = [(r["query"], str(r["id"])) for r in rows]
    await HG.run_batch(items, out_dir, model, base_url, concurrency)
    # 落 scorer 格式(供以后 --mode score):<id>.a = report.md;<id>.q = query
    for r in rows:
        sid = str(r["id"])
        rep = out_dir / f"q{sid}" / "report.md"
        if rep.exists() and rep.stat().st_size > 0:
            (out_dir / f"{sid}.a").write_text(rep.read_text(encoding="utf-8"), encoding="utf-8")
            (out_dir / f"{sid}.q").write_text(r["query"], encoding="utf-8")
    print(f"[generate_harvest] scorer 格式已落 -> {out_dir}/<id>.a,<id>.q", flush=True)


# --------------------------- generate ---------------------------
async def generate(tag, reports_root, queries, n, concurrency):
    os.environ["REPORT_MODE"] = "detailed_cited"
    # 详细长报告(max_tokens 1万)生成耗时长,默认 120s 会 ReadTimeout → 放宽
    os.environ.setdefault("LLM_TIMEOUT", "900")
    from eval.deepresearchqa.run_deepsearchqa import build_parser, build_config, EvalRunner
    from deep_researcher_demo.progress import NullProgressReporter
    config = build_config(build_parser().parse_args([]))
    runner = EvalRunner(config=config, quiet=True)

    out_dir = Path(reports_root) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in open(queries) if l.strip()][:n]
    sem = asyncio.Semaphore(concurrency)

    async def one(row):
        qid, query = str(row["id"]), row["query"]
        if (out_dir / f"{qid}.a").exists():
            return qid, "skip"
        async with sem:
            wf = runner.build_workflow(reporter=NullProgressReporter(), cache_key=f"drgym_{qid}")
            try:
                result = await wf.run(query)          # 原始 query,不套 deepsearchqa wrapper
                (out_dir / f"{qid}.a").write_text(result.final_report or "", encoding="utf-8")
                (out_dir / f"{qid}.q").write_text(query, encoding="utf-8")
                return qid, "ok"
            except Exception as e:                    # noqa: BLE001
                return qid, f"err:{str(e)[:80]}"

    res = await asyncio.gather(*[one(r) for r in rows])
    ok = sum(1 for _, s in res if s in ("ok", "skip"))
    print(f"[generate] {ok}/{len(rows)} 报告就绪 -> {out_dir}", flush=True)
    return out_dir


# --------------------------- score ---------------------------
async def score(tag, reports_root, metrics, model):
    JA.apply_dashscope_env()                         # 必须在 import DRGym scorer 之前
    summary = {}

    if "quality" in metrics:
        import eval_quality_async as Q
        qres = await Q.evaluate_folder_async(tag, model, str(reports_root))
        summary["quality"] = _mean_quality(qres)
        print(f"[quality] {summary['quality']}", flush=True)

    if "kpr" in metrics:
        import eval_kpr_async as K
        results = await K.evaluate_folder_async(tag, model, str(reports_root), str(KEY_POINT_DIR))
        summary["kpr"] = _mean_kpr(results)
        print(f"[kpr] {summary['kpr']}", flush=True)

    if "citation_recall" in metrics:
        import eval_citation_recall_async as CR
        out = await CR.evaluate_folder_async(tag, model, str(reports_root))
        summary["citation_recall"] = out[1] if isinstance(out, tuple) else out
        print(f"[citation_recall] {summary['citation_recall']}", flush=True)

    if "citation_precision" in metrics:
        # 用 demo 缓存核引用,不需要 crawl4ai → 注入 stub 让模块顶部 import 通过
        import types
        if "crawl4ai" not in sys.modules:
            stub = types.ModuleType("crawl4ai")
            class _Dummy:                                 # 接受任意参数;实际抓取已被 cache_crawl 替换
                def __init__(self, *a, **k): pass
                async def __aenter__(self): return self
                async def __aexit__(self, *a): pass
            for _n in ("AsyncWebCrawler", "CrawlerRunConfig", "CacheMode", "BrowserConfig"):
                setattr(stub, _n, _Dummy)
            sys.modules["crawl4ai"] = stub
        import eval_citation_async as CP
        # 用 demo 缓存核引用,替代 crawl4ai
        url2path = JA.build_cache_url_index()
        async def cache_crawl(urls):
            return [JA.cache_fetch(u, url2path) or "" for u in urls]
        CP.crawl_urls = cache_crawl                  # monkey-patch
        out = await CP.evaluate_folder_async(tag, model, str(reports_root))
        summary["citation_precision"] = out[1] if isinstance(out, tuple) else out
        print(f"[citation_precision] {summary['citation_precision']}", flush=True)

    sp = Path(reports_root) / tag / "summary.json"
    sp.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[done] -> {sp}", flush=True)
    return summary


def _mean_quality(results):
    """Quality = 5 维各 0-10 的均值;返回逐维 + overall。results: {qid:{"scores":{crit:(rating,just)}}}"""
    by_crit = {}
    for _qid, rec in (results or {}).items():
        for crit, val in ((rec or {}).get("scores") or {}).items():
            rating = val[0] if isinstance(val, (list, tuple)) else val
            if isinstance(rating, (int, float)):
                by_crit.setdefault(crit, []).append(rating)
    per_crit = {c: round(sum(v) / len(v), 3) for c, v in by_crit.items()}
    overall = round(sum(per_crit.values()) / len(per_crit), 3) if per_crit else None
    return {"per_criterion": per_crit, "overall": overall}


def _mean_kpr(results):
    """KPR = 各题 Supported 占比 的均值。results: {qid: {"labels": {pt_num: (label, just)}}}"""
    rates = []
    for _qid, rec in (results or {}).items():
        labels_map = (rec or {}).get("labels") or {}
        labels = [v[0] if isinstance(v, (list, tuple)) else v for v in labels_map.values()]
        if labels:
            rates.append(sum(1 for l in labels if l == "Supported") / len(labels))
    return round(sum(rates) / len(rates), 4) if rates else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="demo")
    ap.add_argument("--mode", choices=["generate", "score", "all"], default="all")
    ap.add_argument("--reports-root", default=str(DEMO_ROOT / "eval/benchmarks/results/drgym"))
    ap.add_argument("--queries", default=str(DEFAULT_QUERIES))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--metrics", default="quality,kpr,citation_recall,citation_precision")
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--harvest", action="store_true", help="子进程跑、每题留 harvest+计时(需 vLLM server)")
    ap.add_argument("--model", default="Qwen3-32B", help="harvest 子进程用的 demo LLM 名")
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    args = ap.parse_args()
    model = args.judge_model or JA.judge_settings()["model"]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    Path(args.reports_root).mkdir(parents=True, exist_ok=True)

    if args.mode in ("generate", "all"):
        if args.harvest:
            asyncio.run(generate_harvest(args.tag, args.reports_root, args.queries,
                                         args.n, args.concurrency, args.model, args.base_url))
        else:
            asyncio.run(generate(args.tag, args.reports_root, args.queries, args.n, args.concurrency))
    if args.mode in ("score", "all"):
        asyncio.run(score(args.tag, args.reports_root, metrics, model))


if __name__ == "__main__":
    main()
