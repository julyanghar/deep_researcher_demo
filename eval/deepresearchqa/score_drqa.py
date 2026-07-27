"""DRQA(DeepSearchQA)harvest 结果打分:官方 autorater 口径(judge 从报告中评答案),
逐题落盘可续跑,聚合出 f1/precision/recall/全对率。

用法:
  source /home/yilin/tmp/blend_e2e/judge.env
  python eval/deepresearchqa/score_drqa.py --tag a_van200_qa [--out ratings.jsonl] [--workers 4]
第二遍自一致性:--out ratings_r2.jsonl 再跑一遍即可。
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(DEMO_ROOT))

from deep_researcher_demo.llm import OpenAICompatibleClient  # noqa: E402
from eval.deepresearchqa.judge import rate_report  # noqa: E402
from eval.deepresearchqa.scoring import (  # noqa: E402
    aggregate_ratings, build_item_rating_from_report, reduce_autorater_response, score_answer,
)


async def rate_one(llm, model, qid, gold, report, sem):
    async with sem:
        local_scores = score_answer("", gold.get("answer", ""), gold.get("answer_type", ""))
        shell = build_item_rating_from_report({
            "sample_id": qid, "problem": gold.get("problem", ""),
            "problem_category": gold.get("problem_category", ""),
            "answer": gold.get("answer", ""), "answer_type": gold.get("answer_type", ""),
            "final_report": report, "final_answer": "", "local_scores": local_scores,
        })
        jr = await rate_report(
            llm=llm, model=model,
            problem=gold.get("problem", ""), answer_type=gold.get("answer_type", ""),
            answer=gold.get("answer", ""), response=report,
        )
        item = reduce_autorater_response(
            shell, grader_llm_response_text=jr["rating_response"],
            grader_llm_prompt_text=jr["rating_prompt"], rating_error=jr.get("rating_error"),
        )
        rec = item.to_dict()
        rec.update({"sample_id": qid, "problem_category": gold.get("problem_category", ""),
                    "autorater_error": item.error_message})
        return rec


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--reports-root", default=str(DEMO_ROOT / "eval/benchmarks/results/drqa"))
    args = ap.parse_args()

    tag_dir = Path(args.reports_root) / args.tag
    gold = json.loads((tag_dir / "gold.json").read_text())
    out = Path(args.out) if args.out else tag_dir / "ratings.jsonl"
    done = set()
    if out.exists():
        for l in open(out):
            try:
                done.add(str(json.loads(l).get("sample_id")))
            except Exception:  # noqa: BLE001
                pass

    model = os.environ["JUDGE_MODEL"]
    llm = OpenAICompatibleClient(base_url=os.environ["JUDGE_BASE_URL"],
                                 api_key=os.environ["JUDGE_API_KEY"])
    sem = asyncio.Semaphore(args.workers)
    todo = []
    for qid, g in sorted(gold.items(), key=lambda kv: int(kv[0])):
        if qid in done:
            continue
        rep = tag_dir / f"q{qid}" / "report.md"
        if not (rep.exists() and rep.stat().st_size > 0):
            print(f"q{qid}: 缺 report,跳过", flush=True)
            continue
        todo.append((qid, g, rep.read_text(errors="replace")))
    print(f"待打分 {len(todo)} 题(已完成 {len(done)}),judge={model}", flush=True)

    # 配额耗尽早停(2026-07-26):原实现会把 200 题 × 3 次重试全打在 429 上,
    # 既污染结果文件(续跑会跳过),又把剩余配额烧光。
    fail_streak = 0
    with open(out, "a") as w:
        futs = [rate_one(llm, model, q, g, r, sem) for q, g, r in todo]
        for fut in asyncio.as_completed(futs):
            rec = await fut
            err = str(rec.get("autorater_error") or "")
            if "429" in err or "quota" in err.lower():
                fail_streak += 1
                if fail_streak >= 25:
                    print(f"[score_drqa] 连续 {fail_streak} 次 429(配额耗尽),提前停止;已写入 {w.tell() and '若干'} 条,"
                          f"该批未完成的题不落盘以便续跑", flush=True)
                    break
                continue          # 不落盘,留待续跑
            fail_streak = 0
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
            w.flush()

    ratings = [json.loads(l) for l in open(out)]
    agg = aggregate_ratings(ratings)
    metrics = agg.to_dict() if hasattr(agg, "to_dict") else dict(agg.__dict__)
    mp = out.with_name(out.stem + "_metrics.json")
    mp.write_text(json.dumps(metrics, ensure_ascii=False, indent=1, default=str))
    print(f"[score_drqa] {len(ratings)} 题聚合 -> {mp}", flush=True)
    for k in ("f1_score", "precision", "recall", "pct_w_ci_all_answers_correct"):
        if k in metrics:
            print(f"  {k}: {metrics[k]}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
