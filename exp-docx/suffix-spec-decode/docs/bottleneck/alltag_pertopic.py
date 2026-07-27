#!/usr/bin/env python3
"""逐题、逐 tag 比较 tok/s:A vs Ceager,按题ID配对。每 tag 出 赢/平/亏 + 亏损题清单。"""
import glob, json, os, statistics as st
from collections import defaultdict

TAGS = ['RESEARCH_SUMMARY_TEXT','FINAL_REPORT_MARKDOWN','QUERY_PLAN_JSON',
        'SUPERVISOR_DECISION_JSON','INITIAL_RESEARCH_QUESTIONS_JSON']

def per_q_tag(d):
    # {tag: {q: (tok/s, n)}}
    out = defaultdict(dict)
    for qd in glob.glob(d+'/q*/'):
        q = os.path.basename(qd.rstrip('/'))
        f = qd+'llm_calls.jsonl'
        if not os.path.exists(f): continue
        agg = defaultdict(lambda:[0,0,0])  # tag->[tok,dec,n]
        for l in open(f):
            c = json.loads(l)
            t=c.get('tag','?'); ct=c.get('completion_tokens',0); ds=c.get('decode_s',0)
            if ct>1 and ds>0.01:
                agg[t][0]+=ct; agg[t][1]+=ds; agg[t][2]+=1
        for t,(tok,dec,n) in agg.items():
            if n: out[t][q]=(tok/dec,n)
    return out

A=per_q_tag('/home/yilin/deep_researcher_demo/eval/results/drbench/census_A')
C=per_q_tag('/home/yilin/deep_researcher_demo/eval/results/drbench/census_Ceager')

print(f"{'tag':<26}{'配对题':>6}{'赢':>4}{'平':>4}{'亏':>4}{'中位比':>8}{'最差':>7}  亏损题")
for t in TAGS:
    common = sorted(set(A.get(t,{}))&set(C.get(t,{})), key=lambda q:int(q[1:]))
    if not common:
        print(f"{t[:26]:<26} 无配对"); continue
    ratios=[]; losers=[]
    for q in common:
        r = C[t][q][0]/A[t][q][0]
        ratios.append(r)
        if r<0.98: losers.append(f"{q}({r:.2f})")
    win=sum(1 for r in ratios if r>=1.02); flat=sum(1 for r in ratios if 0.98<=r<1.02); lose=sum(1 for r in ratios if r<0.98)
    print(f"{t[:26]:<26}{len(common):>6}{win:>4}{flat:>4}{lose:>4}{st.median(ratios):>8.3f}{min(ratios):>7.2f}  {', '.join(losers) or '无'}")

# 汇总:全 tag 逐(题,tag)对
print("\n=== 全部 (题×tag) 单元格汇总 ===")
allr=[]; alllose=[]
for t in TAGS:
    for q in set(A.get(t,{}))&set(C.get(t,{})):
        r=C[t][q][0]/A[t][q][0]; allr.append(r)
        if r<0.98: alllose.append((t.split('_')[0],q,round(r,2)))
w=sum(1 for r in allr if r>=1.02); f=sum(1 for r in allr if 0.98<=r<1.02); l=sum(1 for r in allr if r<0.98)
print(f"共 {len(allr)} 个(题×tag)单元格:赢 {w} / 平 {f} / 亏 {l}  中位 {st.median(allr):.3f}")
print(f"所有亏损单元格({l}个):{alllose}")
