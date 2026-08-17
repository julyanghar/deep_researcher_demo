"""修正版:vllm: 前缀 + 无锚点。回放真实 summary/report prompt 到 eagle3,分批差分。
用法: python measure_eagle3_v2.py <PORT> <HEAD_LABEL> [n_summary] [n_report]"""
import asyncio, json, glob, re, urllib.request, time, sys
sys.path.insert(0, "/home/yilin/deep_researcher_demo")
from deep_researcher_demo.llm import OpenAICompatibleClient

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 30001
LABEL = sys.argv[2] if len(sys.argv) > 2 else "eagle3"
NS = int(sys.argv[3]) if len(sys.argv) > 3 else 20
NR = int(sys.argv[4]) if len(sys.argv) > 4 else 10
M = "Qwen3-32B"
cli = OpenAICompatibleClient(base_url=f"http://localhost:{PORT}/v1", timeout=600)
HARVEST = sorted(glob.glob("/home/yilin/deep_researcher_demo/eval/results/drbench/c2_spec_nonpar/q*/harvest.jsonl"))

def metrics():
    t = urllib.request.urlopen(f"http://localhost:{PORT}/metrics", timeout=5).read().decode()
    def g(name):
        s = 0.0
        for m in re.finditer(r'vllm:'+name+r'\{[^}]*\}\s+([\d.eE+-]+)', t):
            s += float(m.group(1))
        return s
    def gpos(name, p):
        m = re.search(r'vllm:'+name+r'\{[^}]*position="'+str(p)+r'"[^}]*\}\s+([\d.eE+-]+)', t)
        return float(m.group(1)) if m else 0.0
    return {
        "draft": g("spec_decode_num_draft_tokens_total"),
        "acc": g("spec_decode_num_accepted_tokens_total"),
        "ndrafts": g("spec_decode_num_drafts_total"),
        "p0": gpos("spec_decode_num_accepted_tokens_per_pos_total", 0),
        "p1": gpos("spec_decode_num_accepted_tokens_per_pos_total", 1),
        "p2": gpos("spec_decode_num_accepted_tokens_per_pos_total", 2),
    }

def load(tag, n):
    items = []
    for f in HARVEST:
        for ln in open(f, encoding="utf-8"):
            try: d = json.loads(ln)
            except: continue
            if d.get("tag") == tag and "messages" in d:
                items.append((d["messages"], d.get("max_tokens") or 1500))
                if len(items) >= n: return items
    return items

async def one(msgs, mt, sem):
    async with sem:
        try:
            await cli.chat(msgs, model=M, temperature=0.0, max_tokens=min(mt, 2500), tag="X")
            return True
        except Exception as e:
            print("  fail:", str(e)[:100]); return False

async def batch(items, conc=6):
    sem = asyncio.Semaphore(conc)
    return sum(await asyncio.gather(*[one(m, mt, sem) for m, mt in items]))

def report(name, b, a, dt, nok, nreq):
    dd = a["draft"]-b["draft"]; da = a["acc"]-b["acc"]; dn = a["ndrafts"]-b["ndrafts"]
    r = da/dd if dd else 0
    al = 1 + da/dn if dn else 1
    p = [(a[f"p{i}"]-b[f"p{i}"])/dn if dn else 0 for i in range(3)]
    print(f"\n[{name}] {LABEL}  成功 {nok}/{nreq}  用时 {dt:.0f}s")
    print(f"   draft={dd:.0f} accepted={da:.0f} steps={dn:.0f}")
    print(f"   接受率 accepted/draft = {r:.1%}   mean acceptance length = {al:.2f} tok/前向")
    print(f"   逐位接受率 pos0={p[0]:.1%} pos1={p[1]:.1%} pos2={p[2]:.1%}")
    return {"rate": r, "AL": al, "pos": p}

async def main():
    summ = load("RESEARCH_SUMMARY_TEXT", NS)
    rep  = load("FINAL_REPORT_MARKDOWN", NR)
    print(f"{LABEL}: 载入 summary {len(summ)} / report {len(rep)}")
    m0 = metrics(); t0 = time.time()
    n1 = await batch(summ); m1 = metrics(); t1 = time.time()
    n2 = await batch(rep);  m2 = metrics(); t2 = time.time()
    rs = report("summary 分析类", m0, m1, t1-t0, n1, len(summ))
    rr = report("report 照抄类",  m1, m2, t2-t1, n2, len(rep))
    print("\n" + "="*60)
    print(f"对照 suffix: summary 32% / report 57%")
    print(f"{LABEL}:      summary {rs['rate']:.1%} (AL {rs['AL']:.2f}) / report {rr['rate']:.1%} (AL {rr['AL']:.2f})")
    print(f"→ summary 差 {rs['rate']-0.32:+.1%} , report 差 {rr['rate']-0.57:+.1%}")
    json.dump({"label":LABEL,"summary":rs,"report":rr},
              open(f"/home/yilin/modify-code-runs/eagle3-test/accept_{LABEL}.json","w"))

asyncio.run(main())
