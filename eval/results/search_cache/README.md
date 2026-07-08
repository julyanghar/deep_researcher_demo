# eval/results/search_cache/ 检索缓存

~3.1G,是**整个仓库最大的结果目录**,也是检索/提速工作的数据底座。一题一个子目录
(`drbench/q<编号>/`),把该题的联网检索结果固化下来,让 replay 模式零联网、可复现。

## 一题目录的四件套

| 文件/目录 | 内容 |
|---|---|
| `search_cache.json` | `{归一化查询: [url, ...]}` —— 查询 → 命中网址 |
| `pages_index.json` | `{url: {file, title}}` —— 网址 → 本地页文件 |
| `pages/` | 每个网址的正文(一 url 一文件,题内去重) |
| `chunks.jsonl` | **块级 embedding 索引**:每行 `{url, ci, text, emb}`,即页切块(~1000 token)+ 每块 1024 维向量。检索就是拿查询向量在这里取 top-k |

## 两种模式怎么用它(见 `deep_researcher_demo/search.py`)

- **record / online**:联网抓页 → 切块 embed → 写 `chunks.jsonl`(块向量只算这一次)。
- **replay / local**:零联网,直接读 `chunks.jsonl` + `embed_query` + 取 top-k。

## 性能注意(本会话的优化对象)

`chunks.jsonl` 一题可达 23MB(几千块)。旧实现**每次 `search()` 都把它完整解析两遍**、一题 ~36 遍,
是 e2e "搜索阶段占 32%" 的主因。已加**进程级缓存 + 16GB 内存 LRU**(解析降到 1 遍/题、窗口 34s→~10s)。
详见 [e2e-search-bottleneck.md](../../../exp-docx/suffix-spec-decode/docs/examine-spec-tax/e2e-search-bottleneck.md)、
[search-index-cache-change.md](../../../exp-docx/suffix-spec-decode/docs/examine-spec-tax/search-index-cache-change.md)。
