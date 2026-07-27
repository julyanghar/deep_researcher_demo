#!/usr/bin/env python3
"""逐 tag 比 A(baseline) vs Ceager(suffix+eager)的 decode 速度,查有没有哪个 tag 亏。
口径:两臂独立 e2e,内容会漂 → 用聚合 tok/s(Σcompletion/Σdecode_s)比速率,公平;另给逐条中位。
tok/s 越高越快;ratio=Ceager/A >1 = Ceager 赢该 tag。"""
import glob, json, statistics as st
from collections import defaultdict

def load(d):
    byt = defaultdict(list)  # tag -> list of (tokens, decode_s)
    for f in glob.glob(d+'/q*/llm_calls.jsonl'):
        for l in open(f):
            c = json.loads(l)
            t = c.get('tag','?'); ct = c.get('completion_tokens',0); ds = c.get('decode_s',0)
            if ct > 1 and ds > 0.01:
                byt[t].append((ct, ds))
    return byt

A = load('/home/yilin/deep_researcher_demo/eval/results/drbench/census_A')
C = load('/home/yilin/deep_researcher_demo/eval/results/drbench/census_Ceager')

def agg_tps(rows): return sum(t for t,_ in rows)/sum(d for _,d in rows)
def med_tps(rows): return st.median(t/d for t,d in rows)

tags = sorted(set(A)|set(C), key=lambda t: -sum(d for _,d in C.get(t,[])))
print(f"{'tag':<26}{'nA/nC':>9}{'A tok/s':>9}{'C tok/s':>9}{'聚合比':>8}{'中位比':>8}  判定")
anylose = False
for t in tags:
    if t not in A or t not in C:
        print(f"{t[:26]:<26} 单臂缺失,跳过"); continue
    aT, cT = agg_tps(A[t]), agg_tps(C[t])
    aM, cM = med_tps(A[t]), med_tps(C[t])
    r = cT/aT; rm = cM/aM
    verdict = "✓Ceager快" if r>=1.0 else "✗亏!"
    if r < 1.0: anylose = True
    print(f"{t[:26]:<26}{len(A[t]):>4}/{len(C[t]):<4}{aT:>9.1f}{cT:>9.1f}{r:>8.2f}{rm:>8.2f}  {verdict}")

# 全局
allA=[x for v in A.values() for x in v]; allC=[x for v in C.values() for x in v]
print(f"\n{'全局':<26}{len(allA):>4}/{len(allC):<4}{agg_tps(allA):>9.1f}{agg_tps(allC):>9.1f}{agg_tps(allC)/agg_tps(allA):>8.2f}")
print(f"\n结论:{'有 tag 亏(见 ✗)→ Ceager 并非永远不亏' if anylose else '所有 tag 聚合口径 Ceager ≥ A → 未见任何 tag 亏'}")

# 逐条:每个 tag 有多少条 Ceager 比 A 的中位还慢(看有没有个别慢条,注意50%=同分布)
print("\n逐条分布(Ceager 条速度 vs A 该tag中位;<A中位 的占比,~50%=无差别,>50%=普遍变慢):")
for t in tags:
    if t not in A or t not in C: continue
    amed = med_tps(A[t])
    below = sum(1 for tok,d in C[t] if tok/d < amed)/len(C[t])
    print(f"  {t[:26]:<26} Ceager 条 < A中位 占 {below:.0%}  (n={len(C[t])})")
