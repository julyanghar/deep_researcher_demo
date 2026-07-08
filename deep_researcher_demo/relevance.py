"""gpt-researcher 式相关性:把页切块、按与子查询的 embedding 余弦取全局 top-k。

两个用法共享同一套 chunk+embed+topk(保证 online/local 一致):
- `filter_results`:外层过滤(老 SEARCH_CACHE 直连 / off 路径用)——对返回的整页现切现 embed。
- `chunk_by_tokens` / `embed_texts` / `embed_query` / `select_topk`:可复用件,给
  CachingSearchProvider 的 cache 层用(online 存块向量、local 读块向量做检索)。

embedding 走 DashScope(text-embedding-v3,与 kimi 评委同一套 key)。
"""
import math
import os
import time
from collections import defaultdict

from deep_researcher_demo.schemas import SearchResult

_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.35"))   # 抄 gpt-researcher 默认
_TOP_K = int(os.getenv("RELEVANCE_TOP_K", "10"))
_CHUNK_TOKENS = int(os.getenv("RELEVANCE_CHUNK_TOKENS", "1000"))  # 块大小(token)
_CHUNK_CHARS = int(os.getenv("RELEVANCE_CHUNK_CHARS", "4000"))    # tiktoken 不可用时的回退(~1000tok)
# 全局判据:某子查询**内容总量**小于此值才整体不过滤(gpt-researcher 同款,避免内容本就少
# 时过度修剪)。仅 filter_results(外层)用;cache 层 select_topk 不设(检索总要 top-k)。
_MIN_TOTAL = int(os.getenv("RELEVANCE_MIN_TOTAL_CHARS", "8000"))
_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
_BATCH = int(os.getenv("EMBED_BATCH", "10"))  # 单次 embedding 最多几个 chunk(分批;DashScope v3 上限)
_EMBED_RETRIES = int(os.getenv("EMBED_RETRIES", "6"))            # 遇限流(429)重试次数
_EMBED_BACKOFF = float(os.getenv("EMBED_RETRY_BACKOFF", "2.0"))  # 退避起始秒(每次翻倍)


def _client():
    from openai import OpenAI
    return OpenAI(
        api_key=os.getenv("EMBED_API_KEY") or os.getenv("JUDGE_API_KEY"),
        base_url=(os.getenv("EMBED_BASE_URL") or os.getenv("JUDGE_BASE_URL")
                  or "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    )


# -- 切块 ----------------------------------------------------------------
_ENC = None


def _encoder():
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:  # noqa: BLE001 - tiktoken 缺失则回退字符切块
            _ENC = False
    return _ENC


def chunk_by_tokens(text: str, max_tokens: int = _CHUNK_TOKENS) -> list[str]:
    """按 ~max_tokens 个 token 切块(tiktoken cl100k);tiktoken 不可用则按字符回退。"""
    text = text or ""
    enc = _encoder()
    if not enc:
        size = max_tokens * 4  # 回退:英文约 4 字符/token
        return [text[i:i + size] for i in range(0, len(text), size)] or [""]
    toks = enc.encode(text)
    if not toks:
        return [""]
    return [enc.decode(toks[i:i + max_tokens]) for i in range(0, len(toks), max_tokens)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


cosine = _cosine  # 公开别名


def _is_rate_limit(exc: Exception) -> bool:
    """判定是否限流/配额类错误(429)——值得退避重试,而非真错。"""
    if exc.__class__.__name__ == "RateLimitError":
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("429", "rate limit", "quota", "too many requests"))


def _embed_one_batch(client, batch: list[str]) -> list[list[float]]:
    """单批 embedding;遇 429/限流 → 指数退避重试 _EMBED_RETRIES 次,末次才抛。"""
    delay = _EMBED_BACKOFF
    for attempt in range(_EMBED_RETRIES):
        try:
            resp = client.embeddings.create(model=_MODEL, input=batch)
            return [d.embedding for d in resp.data]
        except Exception as exc:  # noqa: BLE001
            if attempt == _EMBED_RETRIES - 1 or not _is_rate_limit(exc):
                raise
            time.sleep(delay)
            delay *= 2
    return []  # 不可达


def _embed(client, texts: list[str]) -> list[list[float]]:
    """批量 embedding:分批(每批 ≤ _BATCH 个 chunk)+ 每批遇限流退避重试,保序返回。
    "限制单次 chunk 数 + 分多次给" 由 _BATCH 控制;限流不致命由 _embed_one_batch 兜底。"""
    out: list[list[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = [t if t.strip() else " " for t in texts[i:i + _BATCH]]
        out.extend(_embed_one_batch(client, batch))
    return out


def embed_texts(texts: list[str]) -> list[list[float]]:
    """批量 embed 一组文本(自建 client)。"""
    if not texts:
        return []
    return _embed(_client(), texts)


def embed_query(text: str) -> list[float]:
    """embed 单条查询。"""
    return _embed(_client(), [text])[0]


def select_topk(
    query_vec: list[float],
    chunk_records: list[dict],
    *,
    threshold: float = _THRESHOLD,
    top_k: int = _TOP_K,
) -> dict[str, list[str]]:
    """给一组块记录({"url","ci","text","emb"})和查询向量,取**全局** top_k(≥threshold)块,
    按 url 归组返回 {url: [块文本...]}(url 内按 ci 原序)。块向量复用传入的 emb,不重算。"""
    scored = sorted(
        ((_cosine(query_vec, r["emb"]), idx) for idx, r in enumerate(chunk_records)),
        key=lambda x: x[0], reverse=True,
    )
    keep_idx = [idx for s, idx in scored if s >= threshold][:top_k]
    by_url: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for idx in keep_idx:
        r = chunk_records[idx]
        by_url[r["url"]].append((r.get("ci", 0), r.get("text", "")))
    return {url: [t for _, t in sorted(items)] for url, items in by_url.items()}


def filter_results(
    results: list[SearchResult],
    *,
    threshold: float = _THRESHOLD,
    top_k: int = _TOP_K,
    min_total: int = _MIN_TOTAL,
) -> list[SearchResult]:
    """外层过滤(off / 老 SEARCH_CACHE 直连路径用):按子查询全局 top_k,小总量直通。
    cache 层(online/local)走 select_topk,不经此函数。"""
    by_query: dict[str, list[SearchResult]] = defaultdict(list)
    for r in results:
        if (r.raw_content or "").strip():
            by_query[r.query].append(r)
    client = None
    for query, group in by_query.items():
        total = sum(len(r.raw_content) for r in group)
        if total < min_total:
            continue  # 内容总量小 → 整体不过滤
        per_page: list[list[str]] = []
        indexed: list[tuple[int, int, str]] = []
        for gi, r in enumerate(group):
            cs = chunk_by_tokens(r.raw_content)
            per_page.append(cs)
            for ci, c in enumerate(cs):
                indexed.append((gi, ci, c))
        if client is None:
            client = _client()
        qv = _embed(client, [query])[0]
        embs = _embed(client, [c for _, _, c in indexed])
        scored = sorted(
            ((_cosine(qv, e), gi, ci) for (gi, ci, _), e in zip(indexed, embs)),
            key=lambda x: x[0], reverse=True,
        )
        keep = {(gi, ci) for _, gi, ci in
                [(s, gi, ci) for s, gi, ci in scored if s >= threshold][:top_k]}
        for gi, r in enumerate(group):
            kept = [per_page[gi][ci] for ci in range(len(per_page[gi])) if (gi, ci) in keep]
            r.raw_content = "\n\n".join(kept)
    return results
