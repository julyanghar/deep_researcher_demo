# 非LLM墙钟(检索空隙)诊断:645s 到底花在哪

> **缘起**:20 题 e2e 里"非LLM检索"占 645-706s(~15-20% e2e)。曾一度归因"DashScope API 限流",本文**实测推翻**并钉死真凶。
> **一句话**:645s **100% 在搜索阶段**(QUERY_PLAN→SUMMARY);**不是限流**;是两块——① DashScope `embed_query`(~1-3s/子查询,网络、并发)② **纯 Python cosine(`select_topk`)遍历全部块、随块数涨、并发下被 GIL 串行**(大题主凶)。**修复首选:把 cosine 换成 numpy(一处小改、最狠)+ 本地 embedding。**

## 一、定位:645s = 100% 搜索阶段

按"前阶段→后阶段"给非LLM空隙归类(c2,20 题):

| 前 → 后 | 总秒 | 次数 | 均秒 |
|---|---|---|---|
| **QUERY_PLAN → SUMMARY** | **644** | 36 | **17.9** |
| 其它所有转换 | ~0 | — | 0.0 |

**全在 QUERY_PLAN→SUMMARY = 搜索阶段**;且**双峰**——多数近 0,少数巨大(最大单次 **166s**、57s、46s、42s…)。

## 二、逐块实测(推翻限流 + 排除 re-embed)

**① 不是限流**:8 个 `embed_query` 并发 → 总墙钟 **2.3s**(各 ~1.9-2.3s),**没被 429 拖慢**。单次 embed 0.9-3.1s(网络波动,非限流)。

**② 不是现场建索引**:8 题的块索引 **全完整**(未索引页 = 0),`_ensure_chunk_index` 不 re-embed。块索引已持久化在 `eval/results/search_cache/drbench/q*/chunks.jsonl`。

**③ chunk 加载**:0.2-3.8s(JSON 解析,q2 最大 271MB/10874 块),memoize 后每题 subprocess 首次付一遍——中等。

**④ ★真凶:纯 Python cosine`select_topk`,随块数涨**([relevance.py:67-71](../../../deep_researcher_demo/relevance.py#L67-L71) 的 `_cosine` = `sum(x*y for ...)`;[:122-135](../../../deep_researcher_demo/relevance.py#L122-L135) 遍历全部块):

| 题 | 块数 | embed_query | cosine(纯Python) | 合计/子查询 |
|---|---|---|---|---|
| q1 | 1311 | 0.93s | 0.15s | 1.08s |
| q8 | 846 | 3.09s | 0.12s | 3.21s |
| **q2** | **10874** | 2.96s | **1.28s** | **4.24s** |

## 三、根因:大题的 cosine 被 GIL 串行

`_search_replay_embed`([search.py:635](../../../deep_researcher_demo/search.py#L635))对每个子查询做 `select_topk(embed_query(q), records)`:
- **embed_query**:网络,放 GIL → 子查询间**真并发**(8 并发才 2.3s);
- **cosine**:纯 Python、CPU-bound、**持 GIL** → 并发子查询间**串行**。

一题 ~16 个子查询:
- **小题(q1 1311 块)**:cosine 0.15s × 16 串行 ≈ 2.4s + embed 并发 ≈ 3s → 搜索快(gap 近 0);
- **大题(q2 10874 块)**:cosine **1.28s × 16 串行 ≈ 20s** + embed 并发 → 搜索 ~20s+ → **就是 17.9s 均、166s 峰的来源**。

**结论**:645s = **DashScope embed(~1-3s/子查询,网络)+ 纯 Python cosine(随块数涨、GIL 串行,大题主凶)**。**"API 不是大头"成立**——大题上 cosine 的聚合墙钟不输甚至超过 embed;小题上 embed 略大。**都不是限流,不是 re-embed。**

## 四、修复(按性价比)

| 修复 | 治哪块 | 力度 | 代价 |
|---|---|---|---|
| **① `select_topk` 换 numpy**(把 `_cosine` 循环换成 `records_matrix @ query_vec`) | cosine 1.28s→~5ms,**并去掉 GIL 串行** | **最狠**(大题 20s→~0) | 改 [relevance.py](../../../deep_researcher_demo/relevance.py) 一个函数 |
| ② 本地 embedding 模型 | embed_query ~1-3s 网络→<10ms | 大(尤其小题) | 加本地编码分支 + 维度对齐 |
| ③ 块索引存二进制(.npy) | chunk 加载 0.2-3.8s→~0 | 中 | 改存/读格式 |

**先做 ①**——一处小改,直接把大题那 20s/166s 的 cosine 干到毫秒级,是 645s 的主体。②本地 embedding 再补上 embed 的网络那块。

## 五、澄清历史(记忆对账)

- **已优化过的**:chunk 索引 **memoize**(git `a0783e0`),避免每次 search 重解析 23-271MB JSON——这是之前"检索 local 近0"那句的来源(**加载**这块确实优化过)。
- **一直没动的**:`select_topk` 的**纯 Python cosine** + `embed_query` 的**网络**——这两块才是 645s。
- **我中途的错**:先归因"限流退避"→ 8 并发 embed 实测 2.3s **推翻**;再实测定位到纯 Python cosine + embed 网络。**以本文实测为准。**

## 物证

- 空隙归类:本会话脚本(QUERY_PLAN→SUMMARY 100%);
- 组件实测:embed 8并发 2.3s、cosine q1/q2/q8 见 §二表;
- 块索引:`eval/results/search_cache/drbench/q*/chunks.jsonl`(完整、0 未索引页);
- 代码:纯 Python cosine [relevance.py:67-71](../../../deep_researcher_demo/relevance.py#L67-L71) / [select_topk:122-135](../../../deep_researcher_demo/relevance.py#L122-L135);search 路径 [search.py:635](../../../deep_researcher_demo/search.py#L635)。
