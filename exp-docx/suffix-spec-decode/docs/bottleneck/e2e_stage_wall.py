import json, glob
from collections import defaultdict

D='/home/yilin/deep_researcher_demo/eval/results/drbench/census_Ceager/'
def merge(ivs):
    if not ivs: return 0.0
    ivs=sorted(ivs); tot=0.0; cs,ce=ivs[0]
    for s,e in ivs[1:]:
        if s<=ce: ce=max(ce,e)
        else: tot+=ce-cs; cs,ce=s,e
    tot+=ce-cs; return tot

TAGS=['INITIAL_RESEARCH_QUESTIONS_JSON','QUERY_PLAN_JSON','RESEARCH_SUMMARY_TEXT','SUPERVISOR_DECISION_JSON','FINAL_REPORT_MARKDOWN']
wall=defaultdict(float); naive=defaultdict(float); ncalls=defaultdict(int)
total_span=0.0; all_active=0.0; nq=0
for qd in glob.glob(D+'q*/llm_calls.jsonl'):
    calls=[json.loads(l) for l in open(qd)]
    if not calls: continue
    nq+=1
    byt=defaultdict(list)
    for c in calls:
        byt[c['tag']].append((c['start_ts'],c['end_ts'])); naive[c['tag']]+=c['elapsed_s']; ncalls[c['tag']]+=1
    for t,ivs in byt.items(): wall[t]+=merge(ivs)
    allivs=[(c['start_ts'],c['end_ts']) for c in calls]
    total_span+=max(e for _,e in allivs)-min(s for s,_ in allivs)
    all_active+=merge(allivs)

print(f"=== census_Ceager e2e 阶段耗时(40题合计,题间串行)===")
print(f"{'阶段':<24}{'墙钟s':>8}{'占跨度':>7}{'朴素求和s':>10}{'并发压缩':>8}{'调用':>5}")
for t in TAGS:
    w=wall[t]; n=naive[t]; comp=n/w if w else 0
    print(f"{t[:24]:<24}{w:>8.0f}{100*w/total_span:>6.1f}%{n:>10.0f}{comp:>7.1f}x{ncalls[t]:>5}")
print(f"{'LLM阶段合计(并集)':<24}{all_active:>8.0f}{100*all_active/total_span:>6.1f}%")
print(f"{'e2e总跨度(首→尾)':<24}{total_span:>8.0f}{100.0:>6.1f}%")
print(f"{'非LLM(检索/编排/空隙)':<24}{total_span-all_active:>8.0f}{100*(total_span-all_active)/total_span:>6.1f}%")
print(f"\n注:墙钟=同tag区间并集(并发算一次);朴素求和=elapsed相加;并发压缩=两者比。检索是local缓存(近0)。{nq}题。")
