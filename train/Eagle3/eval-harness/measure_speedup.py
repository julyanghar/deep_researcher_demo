"""受控回放:同批 prompt 串行发到某 server,测逐 tag decode token/s + 接受率(/metrics),
存 completion 供跨 server 输出保持抽查。用法: python measure_speedup.py <PORT> <LABEL> <spec:0|1>"""
import urllib.request, json, glob, re, time, sys, hashlib
from collections import defaultdict

PORT=int(sys.argv[1]); LABEL=sys.argv[2]; SPEC=(len(sys.argv)>3 and sys.argv[3]=="1")
NS, NR = 20, 10
HARVEST=sorted(glob.glob("/home/yilin/deep_researcher_demo/eval/results/drbench/c2_spec40/q*/harvest.jsonl"))
OUT=f"/home/yilin/modify-code-runs/eagle3-test/speedup_{LABEL}.json"

def metrics():
    if not SPEC: return (0.0,0.0)
    t=urllib.request.urlopen(f"http://localhost:{PORT}/metrics",timeout=5).read().decode()
    g=lambda n:sum(float(m.group(1)) for m in re.finditer(r'vllm:'+n+r'\{[^}]*\}\s+([\d.eE+-]+)',t))
    return g("spec_decode_num_draft_tokens_total"), g("spec_decode_num_accepted_tokens_total")

def load(tag,n):
    out=[]
    for f in HARVEST:
        q=re.search(r'q(\d+)',f).group(1)
        for ln in open(f,encoding="utf-8"):
            try:d=json.loads(ln)
            except:continue
            if d.get("tag")==tag and "messages" in d:
                out.append((f"{tag}:{q}:{len(out)}", d["messages"], min(d.get("max_tokens") or 1500,2500)))
                if len(out)>=n: return out
    return out

def call(msgs,mt):
    body=json.dumps({"model":"Qwen3-32B","messages":msgs,"temperature":0.0,"max_tokens":mt}).encode()
    req=urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions",data=body,
        headers={"Content-Type":"application/json","Authorization":"Bearer dummy"})
    t0=time.perf_counter()
    r=json.loads(urllib.request.urlopen(req,timeout=600).read().decode())
    wall=time.perf_counter()-t0
    tm=r.get("timing",{}) or {}
    content=r["choices"][0]["message"]["content"]
    return dict(gen=r.get("usage",{}).get("completion_tokens",0), decode_s=tm.get("decode_s",0),
                prefill_s=tm.get("prefill_s",0), ttft_s=tm.get("ttft_s",0), wall=wall,
                chash=hashlib.md5(content.encode()).hexdigest(), clen=len(content))

def run_batch(items, tagname):
    d0,a0=metrics(); recs=[]
    for pid,msgs,mt in items:
        try:
            r=call(msgs,mt); r["pid"]=pid; r["tag"]=tagname; recs.append(r)
            print(f"  {pid}  gen={r['gen']} decode={r['decode_s']:.2f}s tps={r['gen']/r['decode_s'] if r['decode_s'] else 0:.1f}",flush=True)
        except Exception as e:
            print("  fail",pid,str(e)[:80],flush=True)
    d1,a1=metrics()
    return recs,(d1-d0,a1-a0)

def summ(name, recs, acc):
    n=len(recs);
    if not n: print(f"[{name}] 空"); return {}
    gen=sum(r["gen"] for r in recs); dec=sum(r["decode_s"] for r in recs); wall=sum(r["wall"] for r in recs)
    tps_decode=gen/dec if dec else 0
    tps_wall=gen/wall if wall else 0
    dd,da=acc; ar=da/dd if dd else 0
    print(f"\n[{name}] n={n} totGen={gen}")
    print(f"   decode tok/s = {tps_decode:.1f}   (client-wall tok/s = {tps_wall:.1f})")
    print(f"   均 decode={dec/n:.2f}s 均 ttft={sum(r['ttft_s'] for r in recs)/n:.3f}s 均 wall={wall/n:.2f}s")
    if SPEC: print(f"   接受率 accepted/draft = {ar:.1%} (draft={dd:.0f} acc={da:.0f})")
    return dict(n=n,gen=gen,tps_decode=tps_decode,tps_wall=tps_wall,accept=ar if SPEC else None)

print(f"=== {LABEL} (port {PORT}, spec={SPEC}) 受控回放 ===")
S=load("RESEARCH_SUMMARY_TEXT",NS); R=load("FINAL_REPORT_MARKDOWN",NR)
print(f"载入 summary {len(S)} / report {len(R)}")
print("-- summary --"); sr,sa=run_batch(S,"summary")
print("-- report --");  rr,ra=run_batch(R,"report")
res={"label":LABEL,"summary":summ("summary 分析",sr,sa),"report":summ("report 照抄",rr,ra),
     "calls":[{k:r[k] for k in ("pid","tag","gen","decode_s","ttft_s","wall","chash","clen")} for r in sr+rr]}
json.dump(res,open(OUT,"w"),indent=1)
print(f"\n存 {OUT}")
