# 搜索重做:online/local 两模式 + crawl4ai 抓取 + embedding 相关性过滤

> 2026-06-30 落地。替掉原来"DDG + httpx 抓正文(常空)+ 前几名整页照搬"的做法。

## 解决了什么
1. **正文常空**:原 httpx GET 不渲染 JS、不绕反爬 → 学术站/SPA 抓回空壳。改用 **crawl4ai**(无头 Chromium 渲染)抓干净 Markdown。实测真实 DDG 8 结果:crawl4ai 7/8、httpx 5/8(sciencedirect/pubs.rsc.org 这类只有 crawl4ai 抓得到)。
2. **检索离题**:原来前几名整页直接喂 LLM。按 gpt-researcher 做法加 **embedding 相关性过滤**:**按子查询把所有页的块汇到一起→全局算余弦→取全局 top_k(≥阈值)→ 再按来源页拼回**,没块入选的页清空。是**跨页全局 top_k**(100 个页的块在同一池里竞争),不是每页各留 top_k。某子查询内容总量很小(<`RELEVANCE_MIN_TOTAL_CHARS`,默认 8000)时整体不过滤(gpt-researcher 同款,内容本就少不修剪)。

## 两个模式
| 模式 | 干什么 | 等价 |
|---|---|---|
| **online** | 远程搜索(DDG默认/Tavily)拿 URL → crawl4ai 抓整页 → 存本地 → embedding 过滤 → 返回 | 旧 SEARCH_CACHE=record + 新抓取/过滤 |
| **local** | 把子查询 embed → 在该题块向量索引(chunks.jsonl)里 cosine 取**全局 top-k** → 按 url 拼回返回(**纯离线、语义检索**,与原查询抓了哪些 url 解耦) | 旧 replay 的"query→url 精确匹配"已换成 embedding 检索 |

`RESEARCH_MODE=online` → 缓存 record;`=local` → replay;`=off`(默认)→ 沿用旧 `SEARCH_CACHE`。

## 缓存按 benchmark名+题目index 分类
```
<SEARCH_CACHE_DIR>/<SEARCH_BENCHMARK>/q<SEARCH_CACHE_SAMPLE_ID>/
    chunks.jsonl        新:块级 embedding 索引,每行 {url,ci,text,emb(1024)} —— local 语义检索读它
    search_cache.json   旧:query→urls(保留作 provenance,local 不依赖)
    pages_index.json    旧:url→{file,title}
    pages/<hash>.txt     crawl4ai 抓的整页正文(完整页;chunks.jsonl 缺失时据此懒构建)
```
`SEARCH_BENCHMARK` 空时退回旧 `<dir>/q<index>`(向后兼容)。
块向量**只在 online 存页时算一次**(`chunks.jsonl`),online return + 所有 local 查询复用;chunks.jsonl 缺失(老缓存)→ local 首次跑时从 pages 懒构建。
注意:chunks.jsonl(含 1024 维向量)通常比 pages/ 大数倍(向量占大头)。

## 环境变量
| env | 默认 | 作用 |
|---|---|---|
| `RESEARCH_MODE` | off | online / local / off |
| `SEARCH_BENCHMARK` | (空) | benchmark 名(drbench/drgym/deepsearchqa) |
| `SEARCH_CACHE_SAMPLE_ID` | (题目hash) | 题目 index |
| `SEARCH_CACHE_DIR` | eval/results/search_cache | 缓存根 |
| `SEARCH_FETCHER` | crawl4ai | 抓取后端:crawl4ai / httpx(兜底) |
| `RELEVANCE_FILTER` | online/local 时 on | embedding 过滤开关 |
| `RELEVANCE_THRESHOLD` | 0.35 | 余弦阈值(DashScope text-embedding-v3 上 REL≈0.85 / 离题≈0.26,0.35 分得开) |
| `RELEVANCE_TOP_K` | 10 | **每个子查询全局**最多留几块(跨该子查询所有页竞争,非每页) |
| `RELEVANCE_CHUNK_TOKENS` | 1000 | 切块大小(token,tiktoken cl100k;缺则按 ~4000 字符回退) |
| `RELEVANCE_MIN_TOTAL_CHARS` | 8000 | 子查询所有页内容总量 < 此值则整体不过滤(设 0 则总是过滤) |
| `EMBED_MODEL` | text-embedding-v3 | DashScope embedding 模型 |
| `EMBED_API_KEY`/`EMBED_BASE_URL` | 取 JUDGE_* | embedding 端点(默认复用 .env 的评委 key)|
| `CRAWL4AI_LIBS_DIR` | /home/yilin/crawl4ai-libs | 自带 libasound.so.2 目录(见下)|

## crawl4ai 运行时(libasound)
Playwright Chromium 缺系统库 `libasound.so.2`,本机无 sudo → 自带一份放 `/home/yilin/crawl4ai-libs/`,代码在 import crawl4ai 前把它前插进 `os.environ["LD_LIBRARY_PATH"]`(Chromium 是子进程,继承即可加载)。安装详情见 [crawl4ai-setup.md](crawl4ai-setup.md)。crawl4ai 装在 conda `gpt-deep` 环境(进程内调用)。

## 代码位置
- `deep_researcher_demo/scrape_crawl4ai.py`:crawl4ai 抓取 + LD_LIBRARY_PATH 注入。
- `deep_researcher_demo/relevance.py`:DashScope embedding 切块过滤。
- `deep_researcher_demo/search.py`:`DuckDuckGoSearchProvider`(fetcher 开关)、`CachingSearchProvider`(benchmark 参数)、`RelevanceFilteringProvider` / `wrap_with_relevance`。
- `deep_researcher_demo/config.py`:research_mode/search_benchmark/search_fetcher/relevance_enabled + 映射。
- `deep_researcher_demo/cli.py`:接线(fetcher + benchmark + relevance 包装)。
