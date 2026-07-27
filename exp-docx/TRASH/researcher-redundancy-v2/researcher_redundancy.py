#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析1：online100_v2 同一 question 内部，多个 summary(sub-query 单元)之间的输入/输出重复。

单元 = 每条 RESEARCH_SUMMARY_TEXT 调用。每题内两两配对，度量：
  输入-URL       : url_jac (Jaccard) + url_shared (共享URL数)   —— 检索冗余
  输入-snippet   : snip_contain8 (短方 snippet 正文被长方 >=8 逐字覆盖占比)
  输出-逐字      : out_contain8  (短方 summary 被长方 >=8 逐字覆盖占比)
  输出-Jaccard   : out_jac5 (字符5-gram, headline) + out_jac3 (字符3-gram)
每对打标：same_rq(同 research_question=同 researcher)/diff_rq；same_round/cross_round。

URL 集合取结构化 `URL:` 行(检索页面)；snippet 正文取 `Snippet:` 体(去脚手架)。
round 由 harvest 中 SUPERVISOR_DECISION_JSON 出现次数推断(该轮 summary 都在其 supervisor 之前)。
产物落 OUT_DIR。纯读，不改原始数据。
"""
import sys, json, glob, os, re, csv, statistics
from itertools import combinations
from collections import Counter, defaultdict

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/home/yilin/deep_researcher_demo/eval/results/drbench/online100_v2"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "/home/yilin/tmp/researcher-redundancy-v2"
K = 8                 # 逐字 containment 的最短连续字符阈值
NGRAMS = (5, 3)       # 输出 Jaccard 的字符 n-gram

os.makedirs(OUT_DIR, exist_ok=True)


# ---------------- 后缀自动机（拆成 build + scan，SAM 每单元建一次复用） ----------------
class SAM:
    def __init__(s):
        s.nx = [{}]; s.link = [-1]; s.length = [0]; s.last = 0

    def extend(s, c):
        cur = len(s.nx); s.nx.append({}); s.link.append(-1); s.length.append(s.length[s.last] + 1)
        p = s.last
        while p != -1 and c not in s.nx[p]:
            s.nx[p][c] = cur; p = s.link[p]
        if p == -1:
            s.link[cur] = 0
        else:
            q = s.nx[p][c]
            if s.length[p] + 1 == s.length[q]:
                s.link[cur] = q
            else:
                clone = len(s.nx); s.nx.append(dict(s.nx[q])); s.link.append(s.link[q]); s.length.append(s.length[p] + 1)
                while p != -1 and s.nx[p].get(c) == q:
                    s.nx[p][c] = clone; p = s.link[p]
                s.link[q] = clone; s.link[cur] = clone
        s.last = cur


def build_sam(text):
    sam = SAM()
    for ch in text:
        sam.extend(ch)
    return sam


def scan_matchlen(sam, text):
    """text 中每个位置结尾、且是 sam 所建串子串的最长逐字长度。"""
    v = 0; l = 0; ml = [0] * len(text)
    for i, ch in enumerate(text):
        while v and ch not in sam.nx[v]:
            v = sam.link[v]; l = sam.length[v]
        if ch in sam.nx[v]:
            v = sam.nx[v][ch]; l += 1
        else:
            v = 0; l = 0
        ml[i] = l
    return ml


def containment(short_text, short_sam, long_text, long_sam, k=K):
    """短方被长方 >=k 逐字覆盖的字符占比。传入双方 text+已建 SAM，内部挑短的一方做被扫方。"""
    if len(short_text) <= len(long_text):
        s_text, l_sam = short_text, long_sam
    else:
        s_text, l_sam = long_text, short_sam
    if not s_text:
        return 0.0
    ml = scan_matchlen(l_sam, s_text)
    covered = [False] * len(s_text)
    for i in range(len(s_text)):
        if ml[i] >= k:
            for j in range(i - ml[i] + 1, i + 1):
                covered[j] = True
    return sum(covered) / len(s_text)


# ---------------- n-gram / Jaccard ----------------
def ngram_set(text, n):
    return {text[i:i + n] for i in range(len(text) - n + 1)} if len(text) >= n else set()


def jaccard(a, b):
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# ---------------- 解析 harvest 里的 search_results ----------------
URL_LINE_RE = re.compile(r"(?m)^URL:\s*(\S+)\s*$")
# 每条结果块： [N] Query:.. \n Title:.. \n URL:.. \n Snippet:<body 直到下一个 [N] Query 或结尾>
BLOCK_RE = re.compile(r"\[\d+\]\s*Query:.*?\nTitle:.*?\nURL:\s*(\S*)\s*\nSnippet:(.*?)(?=\n\n\[\d+\]\s*Query:|\Z)", re.S)
RQ_RE = re.compile(r"<research_question>\s*(.*?)\s*</research_question>", re.S)
SR_RE = re.compile(r"<search_results>\s*(.*?)\s*</search_results>", re.S)


def parse_summary_input(user_content):
    rq = RQ_RE.search(user_content)
    rq = rq.group(1).strip() if rq else ""
    sr = SR_RE.search(user_content)
    sr_text = sr.group(1) if sr else user_content
    urls = set()
    snippets = []
    for m in BLOCK_RE.finditer(sr_text):
        u = m.group(1).strip()
        if u:
            urls.add(u)
        snippets.append(m.group(2).strip())
    if not urls:  # 兜底：直接抓 URL: 行
        urls = set(URL_LINE_RE.findall(sr_text))
    snippet_text = "\n".join(snippets)
    return rq, urls, snippet_text


# ---------------- 主流程 ----------------
def q_sort_key(d):
    b = os.path.basename(d)
    m = re.match(r"q(\d+)$", b)
    return (0, int(m.group(1))) if m else (1, b)


def main():
    dirs = sorted([d for d in glob.glob(os.path.join(ROOT, "q*")) if os.path.isdir(d)], key=q_sort_key)
    meta_rows = []
    pair_rows = []
    per_q_url_redundancy = []  # (q, 被>=2单元共享的URL占比, 单元数)

    for d in dirs:
        q = os.path.basename(d)
        rounds_seen = 0
        units = []  # dict: idx, rq, rq_id, round, urls, snip, out
        rq_ids = {}
        for line in open(os.path.join(d, "harvest.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            tag = r.get("tag")
            if tag == "SUPERVISOR_DECISION_JSON":
                rounds_seen += 1
                continue
            if tag != "RESEARCH_SUMMARY_TEXT":
                continue
            user = [m["content"] for m in r["messages"] if m["role"] == "user"][0]
            rq, urls, snip = parse_summary_input(user)
            rq_id = rq_ids.setdefault(rq, len(rq_ids))
            units.append({
                "idx": len(units), "rq": rq, "rq_id": rq_id, "round": rounds_seen,
                "urls": urls, "snip": snip, "out": r["content"],
            })

        # 预建 SAM（snippet 与 output 各一次）
        for u in units:
            u["snip_sam"] = build_sam(u["snip"]) if u["snip"] else build_sam("")
            u["out_sam"] = build_sam(u["out"]) if u["out"] else build_sam("")
            meta_rows.append({
                "q": q, "idx": u["idx"], "round": u["round"], "rq_id": u["rq_id"],
                "n_urls": len(u["urls"]), "snippet_len": len(u["snip"]), "gen_len": len(u["out"]),
            })

        # 每题 URL 冗余度：被 >=2 个单元共享的 URL 占比
        urlcnt = Counter(uu for u in units for uu in u["urls"])
        if urlcnt:
            shared = sum(1 for _, c in urlcnt.items() if c >= 2)
            per_q_url_redundancy.append((q, shared / len(urlcnt), len(units)))

        # 两两配对
        for a, b in combinations(units, 2):
            ua, ub = a["urls"], b["urls"]
            url_jac = jaccard(ua, ub)
            url_shared = len(ua & ub)
            snip_c = containment(a["snip"], a["snip_sam"], b["snip"], b["snip_sam"])
            out_c = containment(a["out"], a["out_sam"], b["out"], b["out_sam"])
            row = {
                "q": q, "i": a["idx"], "j": b["idx"],
                "same_rq": int(a["rq_id"] == b["rq_id"]),
                "same_round": int(a["round"] == b["round"]),
                "url_jac": url_jac, "url_shared": url_shared,
                "snip_contain8": snip_c, "out_contain8": out_c,
            }
            for n in NGRAMS:
                row[f"out_jac{n}"] = jaccard(ngram_set(a["out"], n), ngram_set(b["out"], n))
            pair_rows.append(row)

    write_meta(meta_rows)
    write_pairs(pair_rows)
    write_summary(pair_rows, meta_rows, per_q_url_redundancy)
    write_examples(pair_rows, dirs)

    # 断言：pair 数 = Σ C(n_q,2)
    from math import comb
    q_units = Counter(m["q"] for m in meta_rows)
    expect = sum(comb(n, 2) for n in q_units.values())
    print(f"单元(summary)数={len(meta_rows)}  配对数={len(pair_rows)} (期望 {expect})  题数={len(q_units)}")
    assert len(pair_rows) == expect, "配对数与 ΣC(n,2) 不符!"
    console_report(pair_rows, per_q_url_redundancy)


def mean(xs): return statistics.fmean(xs) if xs else 0.0
def med(xs): return statistics.median(xs) if xs else 0.0
def pct(xs, p):
    if not xs: return 0.0
    s = sorted(xs); return s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))]


def write_meta(rows):
    cols = ["q", "idx", "round", "rq_id", "n_urls", "snippet_len", "gen_len"]
    with open(os.path.join(OUT_DIR, "per_summary_meta.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows: w.writerow(r)


def write_pairs(rows):
    cols = ["q", "i", "j", "same_rq", "same_round", "url_jac", "url_shared",
            "snip_contain8", "out_contain8", "out_jac5", "out_jac3"]
    with open(os.path.join(OUT_DIR, "per_pair_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in cols})


def _dist_table(rows, key):
    xs = [r[key] for r in rows]
    return f"| {key} | {mean(xs):.1%} | {med(xs):.1%} | {pct(xs,90):.1%} | {max(xs) if xs else 0:.1%} |" \
        if key != "url_shared" else \
        f"| {key} | {mean(xs):.2f} | {med(xs):.1f} | {pct(xs,90):.1f} | {max(xs) if xs else 0} |"


def write_summary(pairs, meta, urlred):
    L = []
    L.append("# online100_v2 分析1：同题内 summary(researcher) 之间输入/输出重复\n")
    L.append(f"- 单元 = 每条 RESEARCH_SUMMARY_TEXT（{len(meta)} 条）；同题两两配对，共 {len(pairs)} 对。")
    L.append("- 单元 = 单个 sub-query 的摘要；同 research_question 的多个单元 = 同一 researcher。\n")

    L.append("## 逐字覆盖 vs n-gram Jaccard（两种输出重复口径的区别）\n")
    L.append("- **逐字覆盖 out_contain8**：短的一方 summary 里，有多少字落在'与另一方完全相同的 ≥8 连续字符'块内（占比）。要求**连续、完全一致**，抓**硬照抄/复制粘贴**，保守精确。")
    L.append("- **n-gram Jaccard out_jac5/3**：把两条 summary 各切成字符 n-gram **集合**，算 |交|/|并|。**无序、看集合**，容忍改写/语序，抓**'讲同一件事'**；n 越小越宽松（也越易被套话抬高）。")
    L.append("- 互补：逐字漏掉的'换了说法但同义'由 Jaccard 补上。\n")

    L.append("## 全体配对分布\n")
    L.append("| 指标 | mean | median | p90 | max |")
    L.append("|---|---|---|---|---|")
    for k in ["url_jac", "url_shared", "snip_contain8", "out_contain8", "out_jac5", "out_jac3"]:
        L.append(_dist_table(pairs, k))
    L.append("")

    # 分组：same_rq vs diff_rq
    for label, sel in [("same_rq=同一researcher(不同sub-query)", lambda r: r["same_rq"] == 1),
                       ("diff_rq=不同researcher", lambda r: r["same_rq"] == 0)]:
        sub = [r for r in pairs if sel(r)]
        L.append(f"### 分组：{label}  (n={len(sub)})\n")
        L.append("| 指标 | mean | median | p90 |")
        L.append("|---|---|---|---|")
        for k in ["url_jac", "url_shared", "snip_contain8", "out_contain8", "out_jac5", "out_jac3"]:
            xs = [r[k] for r in sub]
            if k == "url_shared":
                L.append(f"| {k} | {mean(xs):.2f} | {med(xs):.1f} | {pct(xs,90):.1f} |")
            else:
                L.append(f"| {k} | {mean(xs):.1%} | {med(xs):.1%} | {pct(xs,90):.1%} |")
        L.append("")

    # 分组：same_round vs cross_round
    for label, sel in [("same_round=同一轮", lambda r: r["same_round"] == 1),
                       ("cross_round=跨轮", lambda r: r["same_round"] == 0)]:
        sub = [r for r in pairs if sel(r)]
        L.append(f"### 分组：{label}  (n={len(sub)})\n")
        L.append("| 指标 | mean | median | p90 |")
        L.append("|---|---|---|---|")
        for k in ["url_jac", "out_contain8", "out_jac5"]:
            xs = [r[k] for r in sub]
            L.append(f"| {k} | {mean(xs):.1%} | {med(xs):.1%} | {pct(xs,90):.1%} |")
        L.append("")

    # 高重复对占比
    L.append("## 高重复配对占比\n")
    n = len(pairs)
    L.append("| 阈值 | 占比 |")
    L.append("|---|---|")
    L.append(f"| out_contain8 > 30% (硬照抄) | {sum(1 for r in pairs if r['out_contain8']>0.3)/n:.1%} |")
    L.append(f"| out_jac5 > 30% | {sum(1 for r in pairs if r['out_jac5']>0.3)/n:.1%} |")
    L.append(f"| out_jac5 > 50% | {sum(1 for r in pairs if r['out_jac5']>0.5)/n:.1%} |")
    L.append(f"| url_shared >= 1 (搜到同一页面) | {sum(1 for r in pairs if r['url_shared']>=1)/n:.1%} |")
    L.append(f"| url_jac > 20% | {sum(1 for r in pairs if r['url_jac']>0.2)/n:.1%} |")
    L.append("")

    # 每题 URL 冗余
    if urlred:
        fr = [x[1] for x in urlred]
        L.append("## 每题检索冗余（被 ≥2 个单元共享的 URL 占比）\n")
        L.append(f"- 跨题 mean={mean(fr):.1%}, median={med(fr):.1%}, p90={pct(fr,90):.1%}, max={max(fr):.1%}")
        L.append("\n> 明细见 per_pair_metrics.csv / per_summary_meta.csv；典型冗余对见 redundancy_examples.md")

    with open(os.path.join(OUT_DIR, "redundancy_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def write_examples(pairs, dirs):
    # 需要回读文本以并列展示
    cache = {}
    def load_units(q):
        if q in cache: return cache[q]
        units = []; rounds_seen = 0
        for line in open(os.path.join(ROOT, q, "harvest.jsonl"), encoding="utf-8"):
            r = json.loads(line)
            if r.get("tag") == "SUPERVISOR_DECISION_JSON": rounds_seen += 1; continue
            if r.get("tag") != "RESEARCH_SUMMARY_TEXT": continue
            units.append(r["content"])
        cache[q] = units; return units

    L = ["# 分析1 典型冗余对（并列展示当证据）\n"]
    for title, key in [("输出逐字重复最高 (out_contain8)", "out_contain8"),
                       ("输出 n-gram(5) 相似最高 (out_jac5)", "out_jac5"),
                       ("输入 URL 重叠最高 (url_jac)", "url_jac")]:
        top = sorted(pairs, key=lambda r: r[key], reverse=True)[:4]
        L.append(f"\n## {title}\n")
        for r in top:
            units = load_units(r["q"])
            si, sj = units[r["i"]], units[r["j"]]
            L.append(f"### {r['q']}  单元#{r['i']} ↔ #{r['j']}  {key}={r[key]:.1%}  "
                     f"(same_rq={r['same_rq']}, same_round={r['same_round']}, url_shared={r['url_shared']})\n")
            L.append("**summary #%d：**\n" % r["i"]); L.append("```\n" + si[:500] + "\n```\n")
            L.append("**summary #%d：**\n" % r["j"]); L.append("```\n" + sj[:500] + "\n```\n")
    with open(os.path.join(OUT_DIR, "redundancy_examples.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def console_report(pairs, urlred):
    def m(k): return mean([r[k] for r in pairs])
    def md(k): return med([r[k] for r in pairs])
    diff = [r for r in pairs if r["same_rq"] == 0]
    same = [r for r in pairs if r["same_rq"] == 1]
    print(f"[全体] url_jac med={md('url_jac'):.1%} | out_contain8 med={md('out_contain8'):.1%} | out_jac5 med={md('out_jac5'):.1%}")
    print(f"[diff_rq 不同researcher] n={len(diff)} url_jac med={med([r['url_jac'] for r in diff]):.1%} "
          f"out_contain8 med={med([r['out_contain8'] for r in diff]):.1%} out_jac5 med={med([r['out_jac5'] for r in diff]):.1%}")
    print(f"[same_rq 同researcher]  n={len(same)} url_jac med={med([r['url_jac'] for r in same]):.1%} "
          f"out_contain8 med={med([r['out_contain8'] for r in same]):.1%} out_jac5 med={med([r['out_jac5'] for r in same]):.1%}")
    if urlred:
        fr = [x[1] for x in urlred]
        print(f"[每题检索冗余] 被>=2单元共享URL占比 median={med(fr):.1%} mean={mean(fr):.1%}")


if __name__ == "__main__":
    main()
