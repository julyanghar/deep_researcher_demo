"""DRQA(DeepSearchQA)的 harvest 生成驱动:每题起 demo CLI 子进程(角色级分隔符生效),
产出与 DRBench/DRGym 相同的 per-题 {report.md, llm_calls.jsonl, harvest.jsonl} 结构。

用法:
  python eval/deepresearchqa/run_drqa.py --tag a_van200_qa --n 200 --offset 0 --concurrency 1
搜索模式由调用方 env 决定:
  record 预跑: RESEARCH_MODE=online SEARCH_BENCHMARK=drqa SEARCH_PROVIDER=tavily TAVILY_API_KEY=...
  对比两臂:   RESEARCH_MODE=local  SEARCH_BENCHMARK=drqa
"""
import argparse
import asyncio
import json
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(DEMO_ROOT))
from eval.benchmarks import harvest_gen as HG  # noqa: E402

DATA = Path(__file__).resolve().parent / "data/deepsearchqa.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--model", default="Qwen3-32B")
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--reports-root", default=str(DEMO_ROOT / "eval/benchmarks/results/drqa"))
    args = ap.parse_args()

    # DRQA 默认配置(2026-07-24 用户拍板):embedding 统一 gemini-embedding-001
    # (走 Google OpenAI 兼容层;key 由调用方 env 提供,如 EMBED_API_KEY/JUDGE_API_KEY)。
    # drbench/drgym 缓存仍是 DashScope 向量空间,不受此默认影响。
    import os
    os.environ.setdefault("EMBED_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    os.environ.setdefault("EMBED_MODEL", "gemini-embedding-001")

    rows = [json.loads(l) for l in open(DATA) if l.strip()]
    sel = rows[args.offset:args.offset + args.n]
    # sample_id = 数据集全局行号(q0..q899),offset 切块跨块不冲突
    items = [(r["problem"], str(args.offset + i)) for i, r in enumerate(sel)]
    out_dir = Path(args.reports_root) / args.tag
    asyncio.run(HG.run_batch(items, out_dir, args.model, args.base_url, args.concurrency))
    # 落 gold 对照(打分用):qid -> answer/answer_type/category
    gold = {str(args.offset + i): {k: r.get(k) for k in ("answer", "answer_type", "problem_category", "problem")}
            for i, r in enumerate(sel)}
    gp = out_dir / "gold.json"
    old = json.loads(gp.read_text()) if gp.exists() else {}
    old.update(gold)
    gp.write_text(json.dumps(old, ensure_ascii=False, indent=1))
    print(f"[run_drqa] gold 已并入 -> {gp} (累计 {len(old)} 题)", flush=True)


if __name__ == "__main__":
    main()
