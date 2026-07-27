#!/usr/bin/env python3
"""把 DRQA 已有缓存题的 chunks.jsonl 向量从 DashScope 空间重嵌为 gemini-embedding-001。
(页面文本已在盘上,只重算 emb 字段,不花 Tavily。)
用法: source /home/yilin/tmp/blend_e2e/drqa_embed.env && python reembed_drqa_google.py
幂等:已是目标维度(3072)的题跳过。
"""
import json
import os
import sys
import glob
import time

sys.path.insert(0, "/home/yilin/deep_researcher_demo")
from deep_researcher_demo.relevance import embed_texts  # 复用生产同款批量+退避

CACHE = "/home/yilin/deep_researcher_demo/eval/results/search_cache/drqa"
TARGET_DIM = int(os.getenv("REEMBED_TARGET_DIM", "3072"))  # gemini-embedding-001 默认维度

for qdir in sorted(glob.glob(f"{CACHE}/q*")):
    f = os.path.join(qdir, "chunks.jsonl")
    if not os.path.exists(f):
        continue
    rows = [json.loads(l) for l in open(f, errors="replace")]
    if not rows:
        continue
    if len(rows[0].get("emb", [])) == TARGET_DIM:
        print(f"{qdir}: 已是目标维度,跳过", flush=True)
        continue
    texts = [r.get("text", "") for r in rows]
    t0 = time.time()
    embs = embed_texts(texts)
    assert len(embs) == len(rows), f"{qdir}: embed 数量不齐 {len(embs)}/{len(rows)}"
    for r, e in zip(rows, embs):
        r["emb"] = e
    tmp = f + ".tmp"
    with open(tmp, "w") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, f)
    print(f"{qdir}: {len(rows)} 块重嵌完成 ({time.time()-t0:.0f}s)", flush=True)
print("REEMBED_ALL_DONE", flush=True)
