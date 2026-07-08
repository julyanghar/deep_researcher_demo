# e2e "搜索阶段" 为什么占 32%:不是限流,是 23MB 索引被反复全解析

## 缘起

e2e 每阶段墙钟拆解(census_Ceager 40 题,题间串行,interval-merge 口径)显示:
LLM 阶段合计 ~68%(summary 32.8% + report 27.8% 为大头),而 **"非 LLM"(检索/编排/空隙)占 31.8%
= 2718s**,全部落在 `QUERY_PLAN → SUMMARY` 这段 LLM 空转窗口里——即**搜索/检索**。
本文钉死这 2718s 的真因,并记录一次已落地的优化(索引缓存,窗口 34s→~11s)。

> 数据口径:题间**确实串行**(各题时间跨度 0 重叠、求和/并集=1.00),所以窗口内 LLM 空转
> = 本题真在做 CPU 侧检索,不是跨题排队。

## TL;DR

- **限流被排除**:DashScope embedding API 并发爬坡 10/30/60 路 **0 次 429**,100 路才冒 10 次、
  150 路 35 次,且都被内置 6 次指数退避接住(成功率 100%)。真实 workload 峰值 ~9 路并发,离门槛远。
- **真因**:replay+embed 检索里,**23MB 的 `chunks.jsonl` 每次 `search()` 被完整 JSON 解析两遍**
  (`_indexed_urls` 一遍 + `_load_chunk_index` 一遍),每题 ~18 次 search() = **~36 遍全解析**,
  全是 CPU/事件循环绑定 → 3 个并发 researcher 退化成 **3× 串行**。
- **修复**:按 (path, mtime, size) 缓存解析结果,一次 run 内索引不变则复用。
  单 search() 12→7.5s,3-researcher 窗口 **35→~11s(3.5×)**,行为逐值一致。见
  [review-report](../../../../../modify-code-runs/search-index-cache/review-report.md)(run 目录)。

## 一、怎么排除"限流"

担心点:embedding 用 DashScope API、key 还和 kimi 评委共用,会不会被限流退避拖慢?

**测法**:直接 import `relevance.embed_query`,并发爆发探天花板(包住 `_embed_one_batch` 数 429)。

| 并发 | 墙钟 | 成功 | 429 退避 |
|---|---|---|---|
| 10 | 2.0s | 10/10 | 0 |
| 30 | 1.1s | 30/30 | 0 |
| 60 | 2.0s | 60/60 | 0 |
| 100 | 4.5s | 100/100 | 10 |
| 150 | 6.3s | 150/150 | 35 |

**门槛在 ~60→100 并发之间**。真实 workload:题间串行,题内 ~3 researcher × 3 子查询 = **~9 路并发峰值**,
远低于门槛。且 key 与评委不同时用(评委在 research 跑完后才跑)。→ **限流不是主因**。

## 二、真因定位:一次窗口 = 3 researcher × 一次 search(),且零并行

按真实形态复现(q3 的真实子查询,replay+embed):

- 单 researcher `search()`(3 子查询)= **12.2s**
- **3 researcher 并发**(9 子查询)= **35.1s** ≈ 正好 3× 单个 → **一点没并行**,和实测空隙窗口 ~34s 严丝合缝。

并发=3×串行 ⇒ 瓶颈是 **CPU/GIL 绑定**,不是网络 I/O(否则该重叠)。给单 search() 逐组件插桩(q3=3455 块):

| 组件 | 耗时 | 性质 |
|---|---|---|
| `_ensure_chunk_index` 校验索引(内含 `_indexed_urls` 全解析 23MB) | 2.54s | CPU / GIL |
| `_load_chunk_index` 再全解析 23MB | 2.17s | **直接同步调用、阻塞事件循环** |
| 3× `embed_query` 网络 | 5.67s | I/O,本可并发但被前两项闸住 |
| 3× `select_topk` 纯 Python 余弦 | 1.26s | CPU / GIL |
| 合计 | 11.7s | (实测 12.2s) |

**为什么并发救不了**:embedding API(占 ~49%)本是 I/O、能重叠,但被卡在那个**阻塞事件循环**的
`_load_chunk_index` 后面;加上 `_ensure_chunk_index`(2.5s)和纯 Python 余弦占死线程池(GIL)
→ 整条链退化成串行,3 个 researcher 排队 3×。**同一份 23MB 索引一题内被解析 ~36 遍**是核心浪费。

源码钉子:`_indexed_urls` [search.py:529](../../../../deep_researcher_demo/search.py#L529)、
`_load_chunk_index` [search.py:533](../../../../deep_researcher_demo/search.py#L533)、
热路径 [search.py:614](../../../../deep_researcher_demo/search.py#L614)(`records = self._load_chunk_index()`)、
`_ensure_chunk_index` [search.py:579](../../../../deep_researcher_demo/search.py#L579)。

## 三、修复:进程级索引缓存(行为逐值一致)

按 (path, `st_mtime_ns`, `st_size`) 签名缓存 `_load_chunk_index` 的解析结果
([search.py:533](../../../../deep_researcher_demo/search.py#L533));`_indexed_urls` 改为从缓存派生
([search.py:529](../../../../deep_researcher_demo/search.py#L529)),把 `_ensure_chunk_index` 那遍 2.5s 也一并干掉。
append 写文件后签名变 → 自动失效重解析,正确性保住。独立锁 `_CHUNK_INDEX_LOCK`
([search.py:316](../../../../deep_researcher_demo/search.py#L316))避免与 append 的 `_CACHE_LOCK` 重入。
缓存带 **内存 LRU 上限**(默认 16GB,env `CHUNK_CACHE_MAX_GB` 可调):实测每题索引 ~118MB 常驻,
超上限踢最久没用的题(至少保留当前题)。逐题串行的 workload 只需 1~2 题常驻,16GB 是防失控 OOM 的安全天花板、正常不触发。

**效果**(改后复跑):单 search() **12→7.5s**;3-researcher 窗口 **35→~11s(3.5×)**;
一题内索引解析 **36 遍→1 遍**;检索 top-k 逐值一致(A1 IDENTICAL)。

**没做的**:`select_topk` 的 numpy 化(仅省 ~11%)。它靠 `sorted((score, idx), reverse=True)`
的精确 tie-break,numpy 浮点微差会翻转近似并列的 top-k → **动检索结果**,风险不配收益。缓存落地后,
余弦(GIL 串行)成了窗口里可见的次大头,若日后要再压,需对 numpy 版做逐位 top-k 校验后再上。

## 相关

- 优化的完整证据/审查:[review-report.md](../../../../../modify-code-runs/search-index-cache/review-report.md)
- decode 侧算力拆解:[decode-step-compute-anatomy.md](decode-step-compute-anatomy.md)
