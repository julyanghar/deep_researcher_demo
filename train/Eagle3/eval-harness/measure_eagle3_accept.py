"""回放 c2 那 20 题 suffix 处理过的真实 prompt 到 eagle3(30001),/metrics 差分测接受率。
summary(RESEARCH_SUMMARY_TEXT=分析类) vs report(FINAL_REPORT_MARKDOWN=照抄类),
对照 suffix 的 summary 32% / report 57%。"""
import asyncio, json, glob, re, urllib.request, time, sys
sys.path.insert(0, "/home/yilin/deep_researcher_demo")
from deep_researcher_demo.llm import OpenAICompatibleClient

PORT = 30001
M = "Qwen3-32B"
cli = OpenAICompatibleClient(base_url=f"http://localhost:{PORT}/v1", timeout=600)
HARVEST = sorted(glob.glob("/home/yilin/deep_researcher_demo/eval/results/drbench/c2_spec_nonpar/q*/harvest.jsonl"))

def raw_metrics():
    t = urllib.request.urlopen(f"http://localhost:{PORT}/metrics", timeout=5).read().decode()
    out = {}
    for m in re.finditer(r'^(spec_decode_[a-z_]+)\{[^}]*\}\s+([\d.eE+-]+)', t, re.M):
        out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(2))
    return out

def snap():
    d = raw_metrics()
    return d.get("spec_decode_num_draft_tokens_total", 0.0), d.get("spec_decode_num_accepted_tokens_total", 0.0)

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
            print("  call fail:", str(e)[:120]); return False

async def run_batch(items, conc=6):
    sem = asyncio.Semaphore(conc)
    ok = await asyncio.gather(*[one(m, mt, sem) for m, mt in items])
    return sum(ok)

async def main():
    print("=== eagle3 /metrics 现有 spec_decode_* 指标 ===")
    for k, v in sorted(raw_metrics().items()): print(f"  {k} = {v}")
    summ = load("RESEARCH_SUMMARY_TEXT", 40)
    rep  = load("FINAL_REPORT_MARKDOWN", 20)
    print(f"\n载入 summary {len(summ)} 条, report {len(rep)} 条")

    d0, a0 = snap(); t0 = time.time()
    nsucc = await run_batch(summ)
    d1, a1 = snap(); t1 = time.time()
    nsucc2 = await run_batch(rep)
    d2, a2 = snap(); t2 = time.time()

    def line(name, d, a, nreq, dt, nsucc):
        r = a/d if d else 0
        print(f"\n[{name}]  成功 {nsucc}/{nreq}  用时 {dt:.0f}s")
        print(f"   draft={d:.0f}  accepted={a:.0f}  接受率 accepted/draft = {r:.1%}")
        print(f"   (num_spec_tokens=3 → 每步均接受 ≈ {3*r:.2f} tok, 每次前向出 tok ≈ {1+3*r:.2f})")
        return r
    print("\n" + "="*60)
    rs = line("summary 分析类", d1-d0, a1-a0, len(summ), t1-t0, nsucc)
    rr = line("report 照抄类",  d2-d1, a2-a1, len(rep),  t2-t1, nsucc2)
    print("\n" + "="*60)
    print("对照 suffix(20题记录): summary 32% / report 57%")
    print(f"eagle3:                 summary {rs:.1%} / report {rr:.1%}")
    print(f"→ summary 差: {rs-0.32:+.1%}   report 差: {rr-0.57:+.1%}")
    json.dump({"summary_ratio":rs,"report_ratio":rr,"suffix_summary":0.32,"suffix_report":0.57},
              open("/home/yilin/modify-code-runs/eagle3-test/accept_result.json","w"))

asyncio.run(main())
