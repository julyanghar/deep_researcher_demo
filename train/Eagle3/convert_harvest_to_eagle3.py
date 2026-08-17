#!/usr/bin/env python
"""harvest(RESEARCH_SUMMARY_TEXT)-> EAGLE-3 域内训练数据。
去重(按 user 内容)+ 过滤(finish_reason=stop 且 output>=50字符)+ 题面隔离 held-out
+ 长度分桶(<=8192 主集 / >8192 长尾)。assistant = Qwen3-32B temp=0 自产(on-policy)。
确定性(held-out 用 md5 排序,不依赖随机种子)-> 重跑结果一致。
用法: python convert_harvest_to_eagle3.py [OUTDIR]
"""
import json, glob, hashlib, re, sys, os

DEMO = "/home/yilin/deep_researcher_demo/eval/results/drbench"
# 数据源(2026-07-11 第三版):local 模式合并集 = 新自产轨迹(q41-100) + TRASH 里全部旧 local run(q1-40)。
# online100(v1, online 模式)在循环里按名排除;改过 summary prompt 的实验 run(l3v2/l0v2 等)由
# 下面的"标准 system 串"过滤自动挡掉——不靠名单。
RUNS = ["eagletraj_local_r1", "TRASH/*"]
# 第四版(2026-07-13)增广:DRGym 232 题 **local 重放**轨迹(1迭代×3researcher×3subquery,
# SKIP_FINAL_REPORT 只产 summary)。根目录与 drbench 不同,单列绝对路径。
# 注意:drgym_online400_r1(online 轨迹)只用于**建缓存池**,prompt 是整页粘贴分布≠部署(local
# 块检索),照 online100 v1 先例不进训练集——所以这里只挂 local 重放的 tag。
EXTRA_ROOTS = ["/home/yilin/deep_researcher_demo/eval/benchmarks/results/drgym/drgym_local232_r1"]
OUT = sys.argv[1] if len(sys.argv) > 1 else "/home/yilin/deep_researcher_demo/train/Eagle3/data"
MAXLEN = 12288  # 第六版(2026-07-13):二期双裁剪落地解锁 TP4@12288;7681条 zh/en均衡,有效监督3.98M(一期×2.08),longtail仅71
STD_SYS = "Compress the provided search results"  # 标准 summary system prompt 前缀(自校验)
N_HELDOUT_TOPICS = 15
os.makedirs(OUT, exist_ok=True)

from transformers import AutoTokenizer
_TOK = AutoTokenizer.from_pretrained("/data/yilin/huggingface/Qwen3-32B")
def real_tok(system, user, out_ids, content):
    """真分词计数(Qwen3 tokenizer)+ ~30 token 模板开销;输出优先用 harvest 的 token_ids 真值。"""
    n = len(_TOK(system, add_special_tokens=False)["input_ids"])         + len(_TOK(user, add_special_tokens=False)["input_ids"])         + (len(out_ids) if out_ids else len(_TOK(content, add_special_tokens=False)["input_ids"]))
    return n + 30
def qkey(user):
    m = re.search(r'<research_question>\s*(.*?)\s*</research_question>', user, re.S)
    return (m.group(1).strip()[:200] if m else user[:200])
def lang(c):
    cjk = len(re.findall(r'[一-鿿]', c)); lat = len(re.findall(r'[A-Za-z]', c)); t = cjk+lat
    r = cjk/t if t else 0
    return "zh" if r > 0.6 else "en" if r < 0.2 else "mix"

seen = set(); recs = []
files = [f for run in RUNS for f in sorted(glob.glob(f"{DEMO}/{run}/q*/harvest.jsonl"))] \
      + [f for r in EXTRA_ROOTS for f in sorted(glob.glob(f"{r}/q*/harvest.jsonl"))]
for f in files:
    if any(x in f for x in (".bad", ".old", "/online100")): continue  # online 模式 v1 排除;TRASH 本身要收
    # run 名:取 /results/ 之后、q目录之前的路径段(drbench 与 drgym 两种根都适用)
    run = f.split("/results/")[1].rsplit("/q", 1)[0].replace("/", "_"); q = re.search(r'q(\d+)', f).group(1)
    for i, ln in enumerate(open(f)):
        try: d = json.loads(ln)
        except: continue
        if d.get("tag") != "RESEARCH_SUMMARY_TEXT" or "messages" not in d: continue
        # 只要非并行(整段)summary:排除 SUMMARY_PARALLEL 的分节 section 调用
        # (prompt 含 section_to_write / <outline>,和部署的非并行路径分布不同)
        alltxt = " ".join(m.get("content", "") for m in d["messages"])
        if "section_to_write" in alltxt or "<outline>" in alltxt: continue
        sysm = d["messages"][0]["content"]; usr = d["messages"][-1]["content"]; out = d.get("content", "")
        if not sysm.startswith(STD_SYS): continue  # 只收标准 summary prompt(挡掉 l3v2/l0v2 等 prompt 实验)
        key = hashlib.md5(usr.encode()).hexdigest()
        if key in seen: continue
        seen.add(key)
        if d.get("finish_reason") != "stop" or len(out) < 50: continue
        recs.append(dict(id=f"{run}_q{q}_{i}", sys=sysm, user=usr, out=out,
                         qk=qkey(usr), tok=real_tok(sysm, usr, d.get("token_ids"), out), lang=lang(out)))

from collections import defaultdict
byq = defaultdict(list)
for r in recs: byq[r["qk"]].append(r)
qs = sorted(byq.keys(), key=lambda k: hashlib.md5(k.encode()).hexdigest())
# heldout 锁定(2026-07-13):一期评测用的 15 个 drbench topic 固定不变,新增数据(DRGym 等)
# 全部进训练——否则 md5 重排会换掉 heldout,一期/二期评测失去可比性。锁文件由一期 manifest 生成。
LOCK = f"{OUT}/heldout_topics_locked.json"
if os.path.exists(LOCK):
    held_qs = set(json.load(open(LOCK)))
    _missing = held_qs - set(qs)
    assert not _missing, f"锁定 heldout topic 在数据里缺失(检查数据源没丢 run): {sorted(_missing)[:2]}"
else:
    held_qs = set(qs[:N_HELDOUT_TOPICS])
    json.dump(sorted(held_qs), open(LOCK, "w"), ensure_ascii=False, indent=1)

def to_sf(r):
    return {"id": r["id"], "language": r["lang"], "conversations": [
        {"role": "system", "content": r["sys"]},
        {"role": "user", "content": r["user"]},
        {"role": "assistant", "content": r["out"]}]}

fmain = open(f"{OUT}/summary_train_main.jsonl", "w")
flong = open(f"{OUT}/summary_train_longtail.jsonl", "w")
fheld = open(f"{OUT}/summary_heldout.jsonl", "w")
fraw = open(f"{OUT}/summary_all_raw.jsonl", "w")
n_main = n_long = n_held = 0; lang_main = defaultdict(int)
for r in recs:
    fraw.write(json.dumps({"id": r["id"], "qk": r["qk"], "tok": r["tok"], "lang": r["lang"],
        "messages": [{"role": "system", "content": r["sys"]}, {"role": "user", "content": r["user"]}],
        "content": r["out"]}, ensure_ascii=False) + "\n")
    sf = to_sf(r)
    if r["qk"] in held_qs:
        fheld.write(json.dumps(sf, ensure_ascii=False) + "\n"); n_held += 1
    elif r["tok"] <= MAXLEN:
        fmain.write(json.dumps(sf, ensure_ascii=False) + "\n"); n_main += 1; lang_main[r["lang"]] += 1
    else:
        flong.write(json.dumps(sf, ensure_ascii=False) + "\n"); n_long += 1
for fh in (fmain, flong, fheld, fraw): fh.close()

json.dump({"maxlen": MAXLEN, "n_dedup_filtered": len(recs), "n_train_main": n_main,
           "n_train_longtail": n_long, "n_heldout": n_held, "n_qtopics": len(qs),
           "lang_main": dict(lang_main), "heldout_topics": sorted(held_qs)},
          open(f"{OUT}/manifest.json", "w"), ensure_ascii=False, indent=1)
print(f"dedup+filter {len(recs)} | topics {len(qs)}")
print(f"train_main {n_main}({dict(lang_main)}) | longtail {n_long} | heldout {n_held}")
print(f"-> {OUT}/")
