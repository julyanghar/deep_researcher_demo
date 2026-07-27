#!/usr/bin/env python3
"""blend-router-e2e 三口径分析:TTFT(supervisor decide 配对) / 逐题 e2e / B 臂 blend 核实。

用法:
  单臂过程量: python analyze_blend_e2e.py --a-dir <dir>
  双臂对比:   python analyze_blend_e2e.py --a-dir <A> --b-dir <B> [--server-b-log <log>] [--out out.json]

口径(方案 §8):排 warmup(max_tokens==8);supervisor decide 按 (题,轮次序) 配对;
逐题 wallclock = 该题 llm_calls 的 max(end_ts)-min(start_ts);bootstrap 10k 配对 CI。
"""
import argparse
import glob
import json
import os
import random
import re
import time


def load_question(qdir):
    path = os.path.join(qdir, "llm_calls.jsonl")
    if not os.path.exists(path):
        return None
    calls = [json.loads(l) for l in open(path) if l.strip()]
    if not calls:
        return None
    return calls


def is_warmup(c):
    return c.get("max_tokens") == 8


def interval_union(calls):
    """LLM 调用区间并集长度(s):≥1 个调用在飞的总时长。wallclock − 它 = 非 LLM 时间
    (搜索缓存回放/检索/编排/调用间空隙)。"""
    ivs = sorted((c["start_ts"], c["end_ts"]) for c in calls)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    return total + (cur_e - cur_s)


def per_question(calls):
    real = [c for c in calls if not is_warmup(c)]
    bytag = {}
    for c in real:
        bytag.setdefault(c["tag"], []).append(c)
    for v in bytag.values():
        v.sort(key=lambda c: c["start_ts"])
    return {
        "wallclock": max(c["end_ts"] for c in calls) - min(c["start_ts"] for c in calls),
        "llm_union": interval_union(calls),
        "n_calls": len(calls),
        "n_warmup": sum(1 for c in calls if is_warmup(c)),
        "bytag": bytag,
    }


def load_arm(d):
    arm = {}
    for qdir in sorted(glob.glob(os.path.join(d, "q*")), key=lambda p: int(re.search(r"q(\d+)$", p).group(1))):
        calls = load_question(qdir)
        if calls:
            arm[int(re.search(r"q(\d+)$", qdir).group(1))] = per_question(calls)
    return arm


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci(diffs, n=10000, seed=0):
    if not diffs:
        return (float("nan"),) * 2
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        s = [rng.choice(diffs) for _ in diffs]
        means.append(mean(s))
    means.sort()
    return means[int(0.025 * n)], means[int(0.975 * n)]


def tag_stat(arm, tag, field):
    out = []
    for q in arm.values():
        out += [c.get(field) for c in q["bytag"].get(tag, []) if c.get(field) is not None]
    return out


VLLM_TS = re.compile(r"(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})")
BLEND_LINE = re.compile(r"BLEND_PATH=(\S+) total=(\d+) tok .*?recompute=(\d+).*?reuse=(\d+)")


def parse_server_blend(log_path, year=None):
    """返回 [(epoch_ts, path_label, total, recompute, reuse)];TP4 同一 blend 打多行,按 (ts,total) 粗去重。"""
    if not log_path or not os.path.exists(log_path):
        return []
    year = year or time.localtime().tm_year
    rows, seen = [], set()
    for line in open(log_path, errors="replace"):
        m = BLEND_LINE.search(line)
        if not m:
            continue
        t = VLLM_TS.search(line)
        ts = time.mktime((year, int(t.group(1)), int(t.group(2)), int(t.group(3)),
                          int(t.group(4)), int(t.group(5)), 0, 0, -1)) if t else None
        key = (ts, m.group(2), m.group(4))
        if key in seen:
            continue
        seen.add(key)
        rows.append((ts, m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return rows


def window_hits(blends, calls, slack_start=1.0, slack_end=0.0):
    """把 blend 行按时间戳归到调用窗口内。blend 行发生在 prefill 起点(靠近
    start_ts),所以起点留 1s 余量、终点不留——题间只隔零点几秒,终点余量会把
    下一题 warmup 的命中误归到上一题最后一个调用(实测 12/370 全是此形态)。"""
    out = []
    for c in calls:
        hits = [b for b in blends if b[0] is not None and c["start_ts"] - slack_start <= b[0] <= c["end_ts"] + slack_end]
        out.append((c, hits))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True)
    ap.add_argument("--b-dir")
    ap.add_argument("--server-b-log")
    ap.add_argument("--out")
    args = ap.parse_args()

    A = load_arm(args.a_dir)
    print(f"== A 臂 {args.a_dir}: {len(A)} 题 ==")
    print(f"  wallclock/题: mean={mean([q['wallclock'] for q in A.values()]):.1f}s")
    for tag in ("SUPERVISOR_DECISION_JSON", "RESEARCH_SUMMARY_TEXT", "FINAL_REPORT_MARKDOWN",
                "FINAL_REPORT_OUTLINE_JSON", "FINAL_REPORT_REVIEW_JSON", "FINAL_REPORT_ADDITION_MARKDOWN"):
        tt = tag_stat(A, tag, "ttft_s")
        if tt:
            print(f"  {tag}: n={len(tt)} ttft mean={mean(tt)*1000:.0f}ms "
                  f"prefill mean={mean(tag_stat(A, tag, 'prefill_s'))*1000:.0f}ms "
                  f"ptok mean={mean(tag_stat(A, tag, 'prompt_tokens')):.0f}")
    result = {"A": {"n": len(A), "wallclock_mean": mean([q["wallclock"] for q in A.values()])}}

    if args.b_dir:
        B = load_arm(args.b_dir)
        print(f"\n== B 臂 {args.b_dir}: {len(B)} 题 ==")
        print(f"  wallclock/题: mean={mean([q['wallclock'] for q in B.values()]):.1f}s")
        for tag in ("SUPERVISOR_DECISION_JSON", "RESEARCH_SUMMARY_TEXT", "FINAL_REPORT_REVIEW_JSON",
                    "FINAL_REPORT_ADDITION_MARKDOWN"):
            tt = tag_stat(B, tag, "ttft_s")
            if tt:
                print(f"  {tag}: n={len(tt)} ttft mean={mean(tt)*1000:.0f}ms "
                      f"ptok mean={mean(tag_stat(B, tag, 'prompt_tokens')):.0f}")

        common = sorted(set(A) & set(B))
        wd = [B[q]["wallclock"] - A[q]["wallclock"] for q in common]
        lo, hi = bootstrap_ci(wd)
        print(f"\n== 配对对比({len(common)} 题) ==")
        print(f"  e2e[含非LLM] Δ(B-A)/题: mean={mean(wd):.1f}s ({100*mean(wd)/mean([A[q]['wallclock'] for q in common]):+.1f}%) CI95=[{lo:.1f},{hi:.1f}]s")
        ud = [B[q]["llm_union"] - A[q]["llm_union"] for q in common]
        ulo, uhi = bootstrap_ci(ud)
        ua = mean([A[q]["llm_union"] for q in common]); ub = mean([B[q]["llm_union"] for q in common])
        print(f"  e2e[仅LLM并集] Δ(B-A)/题: A={ua:.1f}s B={ub:.1f}s mean={mean(ud):.1f}s ({100*mean(ud)/ua:+.1f}%) CI95=[{ulo:.1f},{uhi:.1f}]s")
        print(f"  非LLM时间/题: A={mean([A[q]['wallclock']-A[q]['llm_union'] for q in common]):.1f}s B={mean([B[q]['wallclock']-B[q]['llm_union'] for q in common]):.1f}s")
        td = []
        for q in common:
            a_dec = A[q]["bytag"].get("SUPERVISOR_DECISION_JSON", [])
            b_dec = B[q]["bytag"].get("SUPERVISOR_DECISION_JSON", [])
            for i in range(min(len(a_dec), len(b_dec))):
                if a_dec[i].get("ttft_s") and b_dec[i].get("ttft_s"):
                    td.append(b_dec[i]["ttft_s"] - a_dec[i]["ttft_s"])
        if td:
            lo, hi = bootstrap_ci(td)
            print(f"  supervisor decide ttft Δ(B-A): n={len(td)} mean={mean(td)*1000:+.0f}ms CI95=[{lo*1000:.0f},{hi*1000:.0f}]ms")
        result["B"] = {"n": len(B), "wallclock_mean": mean([q["wallclock"] for q in B.values()])}
        result["paired"] = {"n": len(common), "e2e_delta_mean": mean(wd), "decide_ttft_delta_mean": mean(td) if td else None}

        if args.server_b_log:
            blends = parse_server_blend(args.server_b_log)
            print(f"\n== B server blend 核实({args.server_b_log}) ==")
            print(f"  BLEND_PATH 行(去重): {len(blends)};标签: {sorted(set(b[1] for b in blends))}")
            print(f"  reuse tok 总量: {sum(b[4] for b in blends)}; recompute 总量: {sum(b[3] for b in blends)}")
            dec_calls = [c for q in B.values() for c in q["bytag"].get("SUPERVISOR_DECISION_JSON", [])]
            wr_calls = [c for q in B.values() for t in ("FINAL_REPORT_MARKDOWN", "FINAL_REPORT_OUTLINE_JSON",
                        "FINAL_REPORT_REVIEW_JSON", "FINAL_REPORT_ADDITION_MARKDOWN") for c in q["bytag"].get(t, [])]
            dec_hit = sum(1 for _, h in window_hits(blends, dec_calls) if h)
            wr_hit = sum(1 for _, h in window_hits(blends, wr_calls) if h)
            print(f"  supervisor decide 窗口含 blend: {dec_hit}/{len(dec_calls)} (期望≈全部)")
            print(f"  writer/report 窗口含 blend: {wr_hit}/{len(wr_calls)} (期望=0)")
            fails = sum(1 for l in open(args.server_b_log, errors="replace") if "ATTN_GUIDED job failed" in l)
            print(f"  ATTN_GUIDED job failed: {fails} (期望=0)")
            result["blend"] = {"rows": len(blends), "reuse_tok": sum(b[4] for b in blends),
                               "decide_windows_hit": [dec_hit, len(dec_calls)],
                               "writer_windows_hit": [wr_hit, len(wr_calls)], "ag_failed": fails}

    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
        print(f"\nJSON -> {args.out}")


if __name__ == "__main__":
    main()
