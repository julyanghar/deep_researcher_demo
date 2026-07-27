import json, glob
from collections import defaultdict

D='/home/yilin/deep_researcher_demo/eval/results/drbench/census_Ceager/'
short=lambda t: t.replace('_JSON','').replace('_TEXT','').replace('_MARKDOWN','').replace('RESEARCH_','')[:14]

gap_by_transition=defaultdict(float); gap_count=defaultdict(int)
total_gap=0.0; big_gaps=[]
for qd in glob.glob(D+'q*/llm_calls.jsonl'):
    calls=sorted((json.loads(l) for l in open(qd)), key=lambda c:c['start_ts'])
    if len(calls)<2: continue
    # 扫时间线:维护"当前已覆盖到的最晚end",出现空隙就记
    cover_end=calls[0]['end_ts']; prev_tag=calls[0]['tag']
    for c in calls[1:]:
        s,e,t=c['start_ts'],c['end_ts'],c['tag']
        if s>cover_end:              # 空隙:cover_end → s 之间没调用
            g=s-cover_end
            gap_by_transition[(short(prev_tag),short(t))]+=g
            gap_count[(short(prev_tag),short(t))]+=1
            total_gap+=g
            if g>3: big_gaps.append((g, short(prev_tag), short(t)))
        if e>cover_end: cover_end=e; prev_tag=t   # 谁把覆盖推到最晚,算它
        elif e==cover_end: prev_tag=t

print(f"=== 非LLM空隙 按'前阶段→后阶段'归类(40题合计 {total_gap:.0f}s)===")
print(f"{'空隙位置(前→后)':<32}{'总秒':>8}{'次数':>6}{'均秒':>7}")
for k,v in sorted(gap_by_transition.items(), key=lambda x:-x[1])[:12]:
    print(f"{k[0]+' → '+k[1]:<32}{v:>8.0f}{gap_count[k]:>6}{v/gap_count[k]:>7.1f}")
print(f"\n=== 最大的10个单次空隙 ===")
for g,a,b in sorted(big_gaps,reverse=True)[:10]:
    print(f"  {g:6.1f}s   {a} → {b}")
