"""Search provider abstractions for the simplified deep researcher."""

import asyncio
import hashlib
import json
import logging
import os
import re
import threading
import time
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
    ) -> None:
        self.fetch_webpages = fetch_webpages
        self.max_content_chars = max_content_chars
        self.fetch_timeout = fetch_timeout
        self.fetch_concurrency = fetch_concurrency

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
) -> SearchProvider:
    """Create a search provider by name."""
    normalized = name.lower().strip()
    if normalized == "duckduckgo":
        return DuckDuckGoSearchProvider(
            fetch_webpages=fetch_webpages,
            max_content_chars=max_content_chars,
            fetch_timeout=fetch_timeout,
            fetch_concurrency=fetch_concurrency,
        )
    if normalized == "tavily":
        return TavilySearchProvider()
    raise ValueError(f"Unknown search provider: {name}. Expected duckduckgo or tavily.")


# Process-wide guard for the global query-keyed cache files.
_CACHE_LOCK = threading.Lock()


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
    ) -> None:
        self.base = base
        self.mode = mode
        self.fix_n = fix_n
        self.sample_id = sample_id
        self.root = Path(cache_dir)
        # Per-question directory: each question saves its own two-level cache.
        self.qdir = self.root / f"q{sample_id}"
        self.pages_dir = self.qdir / "pages"
        self.query_map_path = self.qdir / "search_cache.json"
        self.pages_index_path = self.qdir / "pages_index.json"
        # Queries that missed the cache in replay and triggered a live
        # fallback; non-empty means this question's timing/sources are
        # contaminated and the sample should be excluded.
        self.misses: list[str] = []

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        if self.mode == "replay":
            return await self._search_replay(queries, max_results)
        # record mode (and any unknown mode degrades to live + persist). One
        # live search per query keeps the per-query URL list faithful.
        logger.warning("SEARCH_PATH=LIVE(record) q%s | %d queries 全部联网(不读缓存)",
                       self.sample_id, len(queries))
        merged: list[SearchResult] = []
        seen: set[str] = set()
        for query in queries:
            results = await self.base.search([query], max_results=max_results)
            self._record_query(query, results)
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

    def _record_query(self, query: str, results: list[SearchResult]) -> None:
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
            return
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
                changed = True
            if changed:
                self._write_json(self.pages_index_path, pages_index)

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
) -> SearchProvider:
    """Wrap a provider with record/replay caching; `off`/unknown -> passthrough."""
    if mode not in {"record", "replay"}:
        return base
    return CachingSearchProvider(
        base, mode=mode, cache_dir=cache_dir, fix_n=fix_n, sample_id=sample_id
    )
