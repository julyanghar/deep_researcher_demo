"""每题起 demo CLI 子进程生成报告 + 留 per-题 harvest 调用链 + per-调用计时。

为什么子进程:harvest 路径 DRIFT_HARVEST 是全局(llm.py import 期读),进程内并发跑多题会串;
每题一个子进程、各设独立 env,天然分题不串(replicate exp_writer_run.py 的 run_demo)。

每题产物(<out_dir>/q<id>/):
  report.md       —— detailed_cited 报告
  harvest.jsonl   —— DRIFT_HARVEST:5 种调用的完整 messages+content(调用链)
  llm_calls.jsonl —— LLM_CALL_LOG:逐调用 timing(tag + prefill_s/decode_s/...),需 server 带 VLLM_RESP_TIMING=1
"""
import os
import sys
import asyncio
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[2]

HARVEST_TAGS = ("INITIAL_RESEARCH_QUESTIONS_JSON,QUERY_PLAN_JSON,RESEARCH_SUMMARY_TEXT,"
                "SUPERVISOR_DECISION_JSON,FINAL_REPORT_MARKDOWN")  # 全链 5 项


def _child_env(sample_id: str, qdir: Path, model: str, base_url: str) -> dict:
    env = dict(os.environ)
    env.update({
        "REPORT_MODE": "detailed_cited",
        "DRIFT_HARVEST": str(qdir / "harvest.jsonl"),
        "DRIFT_HARVEST_TAGS": HARVEST_TAGS,
        "LLM_CALL_LOG": str(qdir / "llm_calls.jsonl"),     # 逐调用 timing(prefill/decode)
        "MODEL": model,
        "SUPERVISOR_MODEL": model, "RESEARCHER_MODEL": model,
        "SUMMARY_MODEL": model, "FINAL_MODEL": model,
        "OPENAI_BASE_URL": base_url,
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "dummy"),
        "SEARCH_CACHE": "record",                          # 边搜边存,可复现 + 后续 citation 能用
        "SEARCH_CACHE_DIR": os.getenv(
            "SEARCH_CACHE_DIR", str(DEMO_ROOT / "eval/results/search_cache")),
        "SEARCH_CACHE_SAMPLE_ID": sample_id,
        "SUPERVISOR_REASONING": "0",
        "LLM_TIMEOUT": os.getenv("LLM_TIMEOUT", "900"),    # 长报告生成慢,默认 120s 会超时
    })
    return env


async def gen_one(question: str, sample_id: str, out_dir: Path, model: str,
                  base_url: str, sem: asyncio.Semaphore) -> tuple[str, str]:
    qdir = out_dir / f"q{sample_id}"
    report = qdir / "report.md"
    if report.exists() and report.stat().st_size > 0:      # 断点续
        return sample_id, "skip"
    qdir.mkdir(parents=True, exist_ok=True)
    async with sem:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "deep_researcher_demo", "--quiet",
            "--output", str(report), question,
            cwd=str(DEMO_ROOT), env=_child_env(sample_id, qdir, model, base_url),
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, err = await proc.communicate()
        if proc.returncode != 0 or not (report.exists() and report.stat().st_size > 0):
            (qdir / "error.txt").write_bytes(err[-2000:] if err else b"(no stderr)")
            return sample_id, f"err(rc={proc.returncode})"
        return sample_id, "ok"


async def run_batch(items: list[tuple[str, str]], out_dir: Path, model: str,
                    base_url: str, concurrency: int) -> dict:
    """items: [(question, sample_id)]. 返回 {sample_id: status}。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    done = 0
    total = len(items)
    results = {}

    async def one(q, sid):
        nonlocal done
        sid_, status = await gen_one(q, sid, out_dir, model, base_url, sem)
        results[sid_] = status
        done += 1
        print(f"[harvest_gen] {done}/{total} q{sid_}: {status}", flush=True)
        return sid_, status

    await asyncio.gather(*[one(q, sid) for q, sid in items])
    ok = sum(1 for s in results.values() if s in ("ok", "skip"))
    print(f"[harvest_gen] 完成 {ok}/{total} (ok/skip) -> {out_dir}", flush=True)
    return results
