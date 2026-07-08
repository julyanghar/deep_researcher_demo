"""Search provider abstractions for the simplified deep researcher."""

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import httpx
from bs4 import BeautifulSoup

from deep_researcher_demo.schemas import SearchResult

# 用 warning 级别(像 LMCache 的 BLEND_PATH 一样保证可见):每次 search 打一条 SEARCH_PATH=...,
# grep 即可判定到底走了缓存(CACHE)还是 live 联网(LIVE),不靠猜。
logger = logging.getLogger(__name__)

# Process-wide pacing for DuckDuckGo/ddgs calls (see _search_one).
_DDG_THROTTLE_LOCK = threading.Lock()
_DDG_MIN_INTERVAL = float(os.getenv("DDG_MIN_INTERVAL", "1.5"))
_DDG_LAST_CALL = 0.0


class SearchProvider(Protocol):
    """Minimal async search interface."""

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        """Search for multiple queries and return normalized results."""


class DuckDuckGoSearchProvider:
    """DuckDuckGo search provider using the `ddgs` package."""

    def __init__(
        self,
        *,
        fetch_webpages: bool = True,
        max_content_chars: int = 12000,
        fetch_timeout: float = 10.0,
        fetch_concurrency: int = 8,
        fetcher: str = "crawl4ai",
    ) -> None:
        self.fetch_webpages = fetch_webpages
        self.max_content_chars = max_content_chars
        self.fetch_timeout = fetch_timeout
        self.fetch_concurrency = fetch_concurrency
        # 正文抓取后端:crawl4ai(无头浏览器,默认,正文命中率高)或 httpx(轻量兜底)
        self.fetcher = (fetcher or "crawl4ai").strip().lower()

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        tasks = [asyncio.to_thread(self._search_one, query, max_results) for query in queries]
        batches = await asyncio.gather(*tasks)
        results: list[SearchResult] = []
        seen_urls: set[str] = set()

        for batch in batches:
            for result in batch:
                if result.url and result.url in seen_urls:
                    continue
                if result.url:
                    seen_urls.add(result.url)
                results.append(result)

        if self.fetch_webpages:
            results = await self._fetch_raw_content(results)

        return results

    def _search_one(self, query: str, max_results: int) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as exc:
            raise RuntimeError(
                "DuckDuckGo search requires the `ddgs` package. Install with `pip install -e .`."
            ) from exc

        # Sustained bursts (several researchers searching concurrently) get
        # this host rate-limited/blocked by the search backends, which then
        # fail whole benchmark samples. Serialize all ddgs calls process-wide
        # with a minimum interval, retry with backoff on transient errors,
        # and treat "no results" as an empty result set instead of a failure.
        raw_results: list = []
        attempts = 4
        for attempt in range(attempts):
            try:
                with _DDG_THROTTLE_LOCK:
                    global _DDG_LAST_CALL
                    wait = _DDG_MIN_INTERVAL - (time.monotonic() - _DDG_LAST_CALL)
                    if wait > 0:
                        time.sleep(wait)
                    try:
                        with DDGS() as ddgs:
                            raw_results = list(ddgs.text(query, max_results=max_results))
                    finally:
                        _DDG_LAST_CALL = time.monotonic()
                break
            except Exception as exc:  # noqa: BLE001 - backend errors are diverse
                if "no results" in str(exc).lower():
                    raw_results = []
                    break
                if attempt == attempts - 1:
                    raise
                time.sleep(5.0 * (attempt + 1))

        normalized: list[SearchResult] = []
        for item in raw_results:
            url = str(item.get("href") or item.get("url") or "")
            normalized.append(
                SearchResult(
                    query=query,
                    title=str(item.get("title") or ""),
                    url=url,
                    snippet=str(item.get("body") or item.get("snippet") or ""),
                )
            )
        return normalized

    async def _fetch_raw_content(self, results: list[SearchResult]) -> list[SearchResult]:
        if self.fetcher == "crawl4ai":
            return await self._fetch_with_crawl4ai(results)
        return await self._fetch_with_httpx(results)

    async def _fetch_with_crawl4ai(self, results: list[SearchResult]) -> list[SearchResult]:
        """用 crawl4ai(无头浏览器渲染)抓正文,按 url 回填 raw_content。"""
        from deep_researcher_demo import scrape_crawl4ai

        urls = [r.url for r in results if r.url]
        try:
            pages = await scrape_crawl4ai.fetch_pages(
                urls, max_content_chars=self.max_content_chars
            )
        except Exception as exc:  # noqa: BLE001 - 浏览器/依赖问题不应整批失败
            logger.warning("crawl4ai fetch 失败,本批留空正文: %s", str(exc)[:200])
            return results
        for result in results:
            text = pages.get(result.url)
            if text:
                result.raw_content = text
        return results

    async def _fetch_with_httpx(self, results: list[SearchResult]) -> list[SearchResult]:
        semaphore = asyncio.Semaphore(max(1, self.fetch_concurrency))
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        }
        async with httpx.AsyncClient(
            timeout=self.fetch_timeout,
            follow_redirects=True,
            headers=headers,
        ) as client:
            tasks = [self._fetch_one(client, result, semaphore) for result in results]
            return await asyncio.gather(*tasks)

    async def _fetch_one(
        self,
        client: httpx.AsyncClient,
        result: SearchResult,
        semaphore: asyncio.Semaphore,
    ) -> SearchResult:
        if not result.url:
            return result
        async with semaphore:
            try:
                response = await client.get(result.url)
                response.raise_for_status()
            except Exception:
                return result

        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(kind in content_type for kind in ["text/html", "text/plain", "application/xhtml+xml"]):
            return result

        text = extract_text(response.text)
        if text:
            result.raw_content = text[: self.max_content_chars]
        return result


class TavilySearchProvider:
    """Tavily Search API provider."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = "https://api.tavily.com/search",
        search_depth: str = "basic",
        topic: str = "general",
        include_raw_content: bool = True,
        timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY")
        self.base_url = base_url
        self.search_depth = search_depth
        self.topic = topic
        self.include_raw_content = include_raw_content
        self.timeout = timeout

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        if not self.api_key:
            raise RuntimeError("Tavily search requires TAVILY_API_KEY.")

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            tasks = [self._search_one(client, query, max_results) for query in queries]
            batches = await asyncio.gather(*tasks)

        results: list[SearchResult] = []
        seen_urls: set[str] = set()
        for batch in batches:
            for result in batch:
                if result.url and result.url in seen_urls:
                    continue
                if result.url:
                    seen_urls.add(result.url)
                results.append(result)
        return results

    async def _search_one(
        self,
        client: httpx.AsyncClient,
        query: str,
        max_results: int,
    ) -> list[SearchResult]:
        payload = {
            "query": query,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "include_answer": False,
            "include_raw_content": self.include_raw_content,
            "max_results": max_results,
            "api_key": self.api_key,
        }
        try:
            response = await client.post(
                self.base_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        normalized: list[SearchResult] = []
        for item in data.get("results", []):
            url = str(item.get("url") or item.get("href") or "")
            content = str(item.get("content") or item.get("body") or "")
            raw_content = item.get("raw_content")
            normalized.append(
                SearchResult(
                    query=str(data.get("query") or query),
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=content,
                    raw_content=str(raw_content) if raw_content else None,
                )
            )
        return normalized


def extract_text(html: str) -> str:
    """Extract readable text from an HTML/text response."""
    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text("\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def create_search_provider(
    name: str,
    *,
    fetch_webpages: bool = True,
    max_content_chars: int = 12000,
    fetch_timeout: float = 10.0,
    fetch_concurrency: int = 8,
    fetcher: str = "crawl4ai",
) -> SearchProvider:
    """Create a search provider by name."""
    normalized = name.lower().strip()
    if normalized == "duckduckgo":
        return DuckDuckGoSearchProvider(
            fetch_webpages=fetch_webpages,
            max_content_chars=max_content_chars,
            fetch_timeout=fetch_timeout,
            fetch_concurrency=fetch_concurrency,
            fetcher=fetcher,
        )
    if normalized == "tavily":
        return TavilySearchProvider()
    raise ValueError(f"Unknown search provider: {name}. Expected duckduckgo or tavily.")


# Process-wide guard for the global query-keyed cache files.
_CACHE_LOCK = threading.Lock()

# Process-wide memoization of parsed chunks.jsonl. replay 模式下同一题的 23MB 块索引
# 一次 run 内不变,却被每次 search() 完整 JSON 解析两遍(_indexed_urls + _load_chunk_index),
# 每题 ~36 遍。这里按 (path, mtime_ns, size) 签名缓存解析结果:签名不变直接复用,
# append 写文件后签名变 → 自动失效重解析(正确性保住)。独立锁,避免与 _CACHE_LOCK 重入。
# LRU + 内存上限:缓存的 records 常驻内存(每题 ~百 MB,主要是 emb 浮点),总量超上限就踢
# 最久没用的那道题(至少保留刚加载的当前题)。上限默认 16GB,env CHUNK_CACHE_MAX_GB 可调。
_CHUNK_INDEX_CACHE: "OrderedDict[str, tuple]" = OrderedDict()  # path -> (sig, records, nbytes)
_CHUNK_INDEX_LOCK = threading.Lock()
_CHUNK_CACHE_MAX_BYTES = int(float(os.getenv("CHUNK_CACHE_MAX_GB", "16")) * (1024 ** 3))


def _estimate_records_bytes(records: list[dict]) -> int:
    """估一份块索引常驻内存的字节数。大头是每块 emb 的浮点表:list 容器 + 每个 float ~24B;
    另加 text/url 字符串与每条 dict 的固定开销。用于 LRU 按内存(而非题数)驱逐。"""
    total = 0
    for r in records:
        emb = r.get("emb") or ()
        total += sys.getsizeof(emb) + len(emb) * 24  # 浮点对象各 ~24B(CPython 不缓存 float)
        total += sys.getsizeof(r.get("text") or "") + sys.getsizeof(r.get("url") or "")
        total += 200  # dict + ci 等杂项每条固定开销
    return total


def _norm_query(query: str) -> str:
    """Normalize a query for cache keying (collapse whitespace, strip)."""
    return " ".join((query or "").split())


def _url_filename(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".txt"


class CachingSearchProvider:
    """Record/replay decorator keyed by query, with deduplicated page content.

    One directory **per question** (sample): each question keeps its own
    two-level cache, so re-running a question replays exactly its own searches
    and questions never interfere even if they issue the same query string.
    Within a question, the same query always resolves to the same URLs and each
    URL to the same content -- preserving the query->url->content relationship,
    so a different query (e.g. after a decision flips) retrieves its OWN recorded
    sources rather than a shared pool that would mask trajectory divergence.

      <cache_dir>/q<sample_id>/
        search_cache.json : {normalized_query: [url, ...]}
        pages_index.json  : {url: {"file": <name>, "title": <title>}}
        pages/<name>      : page content (one file per URL, deduped within the question)

    Modes:
      - "record": run the wrapped provider live (one search per query so the
        per-query URL list is faithful) and persist query->urls + url->content.
        First-write-wins: existing entries are never overwritten, so recording
        is idempotent and reproducible. Returns the live results.
      - "replay": resolve each query from search_cache.json -> urls ->
        pages_index -> content with zero network. A query absent from the cache
        (cold) falls back to a single live search, is recorded, and is flagged
        in `misses` so the sample can be excluded from analysis.
    """

    def __init__(
        self,
        base: SearchProvider,
        *,
        mode: str,
        cache_dir: str,
        fix_n: int = 0,
        sample_id: object = None,
        benchmark: str | None = None,
        relevance: bool = False,
    ) -> None:
        self.base = base
        self.mode = mode
        self.fix_n = fix_n
        self.sample_id = sample_id
        self.benchmark = (benchmark or "").strip() or None
        # relevance=True:在 cache 层做块级 embedding 检索(online 存块向量、local 读块向量
        # 做 top-k),不再依赖 query→url 精确匹配;False 走旧 query→url 行为(向后兼容)。
        self.relevance = bool(relevance)
        self.root = Path(cache_dir)
        # 按 benchmark名+题目index 一等公民分类:有 benchmark 时落
        # <root>/<benchmark>/q<index>/,否则退回 <root>/q<index>(向后兼容)。
        base_dir = self.root / self.benchmark if self.benchmark else self.root
        self.qdir = base_dir / f"q{sample_id}"
        self.pages_dir = self.qdir / "pages"
        self.query_map_path = self.qdir / "search_cache.json"
        self.pages_index_path = self.qdir / "pages_index.json"
        # 块嵌入索引(每行 {"url","ci","text","emb"}):online 建、local 读。
        self.chunks_path = self.qdir / "chunks.jsonl"
        # Queries that missed the cache in replay and triggered a live
        # fallback; non-empty means this question's timing/sources are
        # contaminated and the sample should be excluded.
        self.misses: list[str] = []

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        if self.mode == "replay":
            if self.relevance:
                return await self._search_replay_embed(queries, max_results)
            return await self._search_replay(queries, max_results)
        # record mode (and any unknown mode degrades to live + persist). One
        # live search per query keeps the per-query URL list faithful.
        logger.warning("SEARCH_PATH=LIVE(record%s) q%s | %d queries 全部联网(不读缓存)",
                       "+embed" if self.relevance else "", self.sample_id, len(queries))
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for query in queries:
            results = await self.base.search([query], max_results=max_results)
            new_pages = self._record_query(query, results)
            if self.relevance:
                # 新页切块+embed存索引(只此一次),return 用块向量做该查询 top-k
                if new_pages:
                    await asyncio.to_thread(self._append_chunk_index, new_pages)
                self._extend(merged, seen, await asyncio.to_thread(
                    self._embed_select_for_query, query, [r.url for r in results]))
            else:
                for result in results:
                    url = (result.url or "").strip()
                    if url and url in seen:
                        continue
                    if url:
                        seen.add(url)
                    merged.append(result)
        return merged

    async def _search_replay(self, queries: list[str], max_results: int) -> list[SearchResult]:
        query_map = self._load_json(self.query_map_path)
        pages_index = self._load_json(self.pages_index_path)
        merged: list[SearchResult] = []
        seen: set[str] = set()
        cold: list[str] = []
        for query in queries:
            urls = query_map.get(_norm_query(query))
            if urls is None:
                cold.append(query)
                continue
            self._extend(merged, seen, self._results_for(query, urls, pages_index))
        logger.warning("SEARCH_PATH=CACHE(replay) q%s | 命中=%d cold_live=%d",
                       self.sample_id, len(queries) - len(cold), len(cold))
        if cold:
            # Cold queries: one live search each, record, flag, then re-resolve
            # from the freshly written cache so the returned content is the
            # persisted (reproducible) copy.
            for query in cold:
                results = await self.base.search([query], max_results=max_results)
                self._record_query(query, results)
            self.misses.extend(cold)
            self._log_miss(cold)
            query_map = self._load_json(self.query_map_path)
            pages_index = self._load_json(self.pages_index_path)
            for query in cold:
                urls = query_map.get(_norm_query(query), [])
                self._extend(merged, seen, self._results_for(query, urls, pages_index))
        return merged

    # -- resolution ------------------------------------------------------

    @staticmethod
    def _extend(
        merged: list[SearchResult],
        seen: set[str],
        results: list[SearchResult],
    ) -> None:
        for result in results:
            if result.url and result.url in seen:
                continue
            if result.url:
                seen.add(result.url)
            merged.append(result)

    def _results_for(
        self,
        query: str,
        urls: list[str],
        pages_index: dict,
    ) -> list[SearchResult]:
        n = self.fix_n if self.fix_n and self.fix_n > 0 else len(urls)
        out: list[SearchResult] = []
        for url in urls[:n]:
            entry = pages_index.get(url)
            if not entry:
                continue
            path = self.pages_dir / entry["file"]
            if not path.exists():
                continue
            out.append(
                SearchResult(
                    query=query,
                    title=entry.get("title", ""),
                    url=url,
                    snippet="",
                    raw_content=path.read_text(encoding="utf-8"),
                )
            )
        return out

    # -- persistence -----------------------------------------------------

    def _record_query(self, query: str, results: list[SearchResult]) -> list[tuple[str, str]]:
        """存 query->urls(provenance) + url->content;返回本次**新存的页** [(url, content)]
        (供 relevance 模式建块嵌入索引,只对新页 embed)。"""
        key = _norm_query(query)
        urls: list[str] = []
        seen: set[str] = set()
        pages: list[tuple[str, str, str]] = []  # (url, title, content)
        for result in results:
            url = (result.url or "").strip()
            if not url or url in seen:
                continue
            content = result.raw_content or result.snippet or ""
            if not content.strip():
                continue  # failed fetch with no snippet -> nothing to reuse
            seen.add(url)
            urls.append(url)
            pages.append((url, result.title or "", content))
        if not urls:
            return []
        new_pages: list[tuple[str, str]] = []
        with _CACHE_LOCK:
            self.pages_dir.mkdir(parents=True, exist_ok=True)
            query_map = self._load_json(self.query_map_path)
            if key not in query_map:  # first-write-wins keeps query->urls stable
                query_map[key] = urls
                self._write_json(self.query_map_path, query_map)
            pages_index = self._load_json(self.pages_index_path)
            changed = False
            for url, title, content in pages:
                if url in pages_index:  # first-write-wins keeps url->content stable
                    continue
                fname = _url_filename(url)
                self._write_text(self.pages_dir / fname, content)
                pages_index[url] = {"file": fname, "title": title}
                new_pages.append((url, content))  # 走到这就是"新存的页"
                changed = True
            if changed:
                self._write_json(self.pages_index_path, pages_index)
        return new_pages

    # -- 块嵌入索引(relevance 模式)----------------------------------------
    def _indexed_urls(self) -> set[str]:
        # 从(已缓存的)块索引派生 url 集合,不再独立全解析 → 省掉 _ensure_chunk_index 里那遍 2.5s。
        return {r["url"] for r in self._load_chunk_index() if "url" in r}

    def _load_chunk_index(self) -> list[dict]:
        if not self.chunks_path.exists():
            return []
        st = self.chunks_path.stat()
        key = str(self.chunks_path)
        sig = (st.st_mtime_ns, st.st_size)
        with _CHUNK_INDEX_LOCK:
            hit = _CHUNK_INDEX_CACHE.get(key)
            if hit is not None and hit[0] == sig:
                _CHUNK_INDEX_CACHE.move_to_end(key)  # 标记最近使用(LRU)
                return hit[1]
        # 缓存未命中(首次 / 文件变了):真正解析一遍。放锁外做慢活,解析完再存。
        if os.getenv("MCDBG"):
            logger.warning("MCDBG_parse %s", key)
        records: list[dict] = []
        for line in self.chunks_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        nbytes = _estimate_records_bytes(records)
        with _CHUNK_INDEX_LOCK:
            _CHUNK_INDEX_CACHE[key] = (sig, records, nbytes)
            _CHUNK_INDEX_CACHE.move_to_end(key)
            # 超内存上限就踢最久没用的;至少保留当前刚加载这份(len>1 才踢)。
            while (len(_CHUNK_INDEX_CACHE) > 1 and
                   sum(v[2] for v in _CHUNK_INDEX_CACHE.values()) > _CHUNK_CACHE_MAX_BYTES):
                _CHUNK_INDEX_CACHE.popitem(last=False)
        return records

    def _append_chunk_index(self, pages: list[tuple[str, str]]) -> None:
        """对新页切块(~1000token)+embed,追加 chunks.jsonl(url first-write-wins 幂等)。
        block 向量只在这里算一次,online return + 所有 local 查询都复用它。"""
        from deep_researcher_demo import relevance
        with _CACHE_LOCK:
            existing = self._indexed_urls()
        todo = [(u, c) for u, c in pages if u and u not in existing and (c or "").strip()]
        if not todo:
            return
        lines: list[str] = []
        for url, content in todo:
            chunks = relevance.chunk_by_tokens(content)
            embs = relevance.embed_texts(chunks)
            for ci, (text, emb) in enumerate(zip(chunks, embs)):
                lines.append(json.dumps(
                    {"url": url, "ci": ci, "text": text, "emb": emb}, ensure_ascii=False))
        with _CACHE_LOCK:
            self.qdir.mkdir(parents=True, exist_ok=True)
            with self.chunks_path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")

    def _ensure_chunk_index(self) -> None:
        """local:缺索引/有未索引的页 → 从 pages 现切现 embed 补上(老缓存懒升级)。"""
        pages_index = self._load_json(self.pages_index_path)
        indexed = self._indexed_urls()
        missing: list[tuple[str, str]] = []
        for url, entry in pages_index.items():
            if url in indexed:
                continue
            path = self.pages_dir / entry.get("file", "")
            if path.exists():
                missing.append((url, path.read_text(encoding="utf-8")))
        if missing:
            self._append_chunk_index(missing)

    def _title_for(self, url: str, pages_index: dict) -> str:
        return (pages_index.get(url) or {}).get("title", "")

    def _embed_select_for_query(self, query: str, urls: list[str]) -> list[SearchResult]:
        """online return:用已存块向量对该查询取 top-k(只 embed query,不重算块)。"""
        from deep_researcher_demo import relevance
        wanted = {u for u in urls if u}
        records = [r for r in self._load_chunk_index() if r.get("url") in wanted]
        if not records:
            return []
        by_url = relevance.select_topk(relevance.embed_query(query), records)
        pages_index = self._load_json(self.pages_index_path)
        return [
            SearchResult(query=query, title=self._title_for(url, pages_index),
                         url=url, snippet="", raw_content="\n\n".join(texts))
            for url, texts in by_url.items()
        ]

    async def _search_replay_embed(self, queries: list[str], max_results: int) -> list[SearchResult]:
        """local:embedding 块级语义检索(全题页池)。无索引则懒构建;无任何页则退老 replay。"""
        await asyncio.to_thread(self._ensure_chunk_index)
        records = self._load_chunk_index()
        if not records:
            return await self._search_replay(queries, max_results)
        pages_index = self._load_json(self.pages_index_path)
        merged: list[SearchResult] = []
        seen: set[str] = set()
        hit = 0

        def _select(q: str):
            from deep_researcher_demo import relevance
            return relevance.select_topk(relevance.embed_query(q), records)

        for query in queries:
            by_url = await asyncio.to_thread(_select, query)
            hit += sum(len(v) for v in by_url.values())
            self._extend(merged, seen, [
                SearchResult(query=query, title=self._title_for(url, pages_index),
                             url=url, snippet="", raw_content="\n\n".join(texts))
                for url, texts in by_url.items()
            ])
        logger.warning("SEARCH_PATH=CACHE(embed) q%s | 池块=%d 命中块=%d",
                       self.sample_id, len(records), hit)
        return merged

    @staticmethod
    def _load_json(path: Path) -> dict:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return {}
        return {}

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)

    def _log_miss(self, queries: list[str]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "cache_misses.jsonl"
        ts = datetime.now(timezone.utc).isoformat()
        with path.open("a", encoding="utf-8") as file:
            for query in queries:
                file.write(
                    json.dumps(
                        {"sample_id": self.sample_id, "query": query, "ts": ts},
                        ensure_ascii=False,
                    )
                    + "\n"
                )


def wrap_with_cache(
    base: SearchProvider,
    *,
    mode: str,
    cache_dir: str,
    fix_n: int = 0,
    sample_id: object = None,
    benchmark: str | None = None,
    relevance: bool = False,
) -> SearchProvider:
    """Wrap a provider with record/replay caching; `off`/unknown -> passthrough.
    relevance=True 时 cache 层做块级 embedding 检索(online 存块向量、local 读块向量 top-k)。"""
    if mode not in {"record", "replay"}:
        return base
    return CachingSearchProvider(
        base, mode=mode, cache_dir=cache_dir, fix_n=fix_n,
        sample_id=sample_id, benchmark=benchmark, relevance=relevance,
    )


class RelevanceFilteringProvider:
    """在任意 provider 外再包一层:返回前按子查询做 embedding 相关性过滤。

    gpt-researcher 式提质——把"搜得到/抓得到"与"喂给 LLM 的内容"解耦:每个结果的
    raw_content 被裁剪成只剩与其 .query 相关的块。online/local 两模式都套这一层
    (online 抓存后过滤,local 读盘后过滤),所以相关性逻辑只此一处、与后端无关。
    """

    def __init__(self, base: SearchProvider) -> None:
        self.base = base

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        results = await self.base.search(queries, max_results=max_results)
        if not results:
            return results
        from deep_researcher_demo import relevance
        try:
            # 同步、可能多次调 embedding API → 丢线程,别阻塞事件循环
            return await asyncio.to_thread(relevance.filter_results, results)
        except Exception as exc:  # noqa: BLE001 - 过滤失败不该毁掉整轮检索
            logger.warning("relevance 过滤失败,返回未过滤结果: %s", str(exc)[:200])
            return results

    # 透传 record/replay 的 misses(供上层判定缓存命中情况)
    @property
    def misses(self) -> list[str]:
        return getattr(self.base, "misses", [])


def wrap_with_relevance(base: SearchProvider, *, enabled: bool) -> SearchProvider:
    """enabled 时套上 embedding 相关性过滤层,否则原样返回。"""
    return RelevanceFilteringProvider(base) if enabled else base
