"""逐题接受率:每题 summary/report 各自成批差分 /metrics,并算逐题中文占比。
验"越中文的题,eagle3 接受率越高吗"。用法: python x.py <PORT> <LABEL>"""
import asyncio, json, glob, re, urllib.request, sys
sys.path.insert(0, "/home/yilin/deep_researcher_demo")
from deep_researcher_demo.llm import OpenAICompatibleClient
PORT=int(sys.argv[1]); LABEL=sys.argv[2]; KSUM=6
M="Qwen3-32B"; cli=OpenAICompatibleClient(base_url=f"http://localhost:{PORT}/v1",timeout=600)
D="/home/yilin/deep_researcher_demo/eval/results/drbench/c2_spec_nonpar"

def zh(s):
    if not s:return 0.0
    c=len(re.findall(r'[一-鿿]',s));l=len(re.findall(r'[A-Za-z]',s));return c/(c+l) if c+l else 0.0
def metrics():
    t=urllib.request.urlopen(f"http://localhost:{PORT}/metrics",timeout=5).read().decode()
    g=lambda n:sum(float(m.group(1)) for m in re.finditer(r'vllm:'+n+r'\{[^}]*\}\s+([\d.eE+-]+)',t))
    return g("spec_decode_num_draft_tokens_total"),g("spec_decode_num_accepted_tokens_total")
def load(qd):
    sm=[];rp=[]
    for ln in open(qd,encoding="utf-8"):
        try:d=json.loads(ln)
        except:continue
        if d.get("tag")=="RESEARCH_SUMMARY_TEXT" and "messages" in d: sm.append((d["messages"],d.get("max_tokens") or 1500,zh(d.get("content",""))))
        if d.get("tag")=="FINAL_REPORT_MARKDOWN" and "messages" in d: rp.append((d["messages"],d.get("max_tokens") or 2000,zh(d.get("content",""))))
    return sm,rp
async def one(m,mt,sem):
    async with sem:
        try:await cli.chat(m,model=M,temperature=0.0,max_tokens=min(mt,2500),tag="X");return 1
        except Exception as e:print("fail",str(e)[:80]);return 0
async def run(items):
    sem=asyncio.Semaphore(6);await asyncio.gather(*[one(m,mt,sem) for m,mt,_ in items])
async def main():
    print(f"{LABEL} 逐题接受率")
    print("题 | zhSum | zhRep | sumAcc | repAcc")
    rows=[]
    for qd in sorted(glob.glob(D+"/q*/harvest.jsonl"),key=lambda p:int(re.search(r'q(\d+)',p).group(1))):
        q=int(re.search(r'q(\d+)',qd).group(1)); sm,rp=load(qd)
        sm=sm[:KSUM]
        zs=sum(x[2] for x in sm)/len(sm) if sm else 0
        zr=sum(x[2] for x in rp)/len(rp) if rp else 0
        d0,a0=metrics(); await run(sm); d1,a1=metrics()
        sacc=(a1-a0)/(d1-d0) if d1>d0 else 0
        await run(rp); d2,a2=metrics()
        racc=(a2-a1)/(d2-d1) if d2>d1 else 0
        rows.append((q,zs,zr,sacc,racc))
        print(f"q{q:<3}| {zs:.0%}   | {zr:.0%}   | {sacc:.1%}  | {racc:.1%}",flush=True)
    import statistics as st
    def corr(xs,ys):
        n=len(xs);mx=sum(xs)/n;my=sum(ys)/n
        num=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        den=(sum((x-mx)**2 for x in xs)*sum((y-my)**2 for y in ys))**.5
        return num/den if den else 0
    zs=[r[1] for r in rows];zr=[r[2] for r in rows];sa=[r[3] for r in rows];ra=[r[4] for r in rows]
    print(f"\n相关 corr(zhSum,sumAcc)={corr(zs,sa):+.2f}  corr(zhRep,repAcc)={corr(zr,ra):+.2f}  (>0=越中文越高)")
    print(f"summary接受率 中位 {st.median(sa):.1%} 区间[{min(sa):.1%},{max(sa):.1%}]")
    print(f"report 接受率 中位 {st.median(ra):.1%} 区间[{min(ra):.1%},{max(ra):.1%}]")
    json.dump([{"q":r[0],"zhSum":r[1],"zhRep":r[2],"sumAcc":r[3],"repAcc":r[4]} for r in rows],
              open(f"/home/yilin/modify-code-runs/eagle3-test/perq_{LABEL}.json","w"))
asyncio.run(main())
