"""Search provider abstractions for the simplified deep researcher."""

import asyncio
import hashlib
import json
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


def _strip_front_matter(text: str) -> str:
    """Drop the leading `---\\n...\\n---\\n` YAML block written by _write_md."""
    if text.startswith("---"):
        parts = text.split("\n---\n", 1)
        if len(parts) == 2:
            return parts[1].lstrip("\n")
    return text


class CachingSearchProvider:
    """Record/replay decorator over a real SearchProvider.

    Makes end-to-end timing hermetic: record the live web search once into a
    per-question document pool (markdown per URL), then replay from the pool
    on every subsequent run with zero network so latency/results are fixed.

    Modes:
      - "record": run the wrapped provider live and persist every result
        (one markdown file per URL + a manifest) into the per-question dir.
        Returns the live results unchanged.
      - "replay": serve from the per-question pool, returning the first
        `fix_n` docs (manifest order; 0 = all) for every query -- a shared
        pool, so the documents are identical no matter which queries the LLM
        generates (A/B fairness). On a cold pool (never recorded) it falls
        back to a live search, records it, and flags the question via
        `misses` so the sample can be excluded from timing analysis.
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
        self.qdir = self.root / f"q{sample_id}"
        # Queries that missed the cache in replay and triggered a live
        # fallback; non-empty means this question's timing is contaminated.
        self.misses: list[str] = []
        self._lock = asyncio.Lock()

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        if self.mode == "replay":
            pool = self._load_pool()
            if pool:
                return self._select(pool, queries)
            # Cold pool: fall back to a live search, persist it, flag the miss.
            results = await self.base.search(queries, max_results=max_results)
            await self._persist(results)
            self.misses.extend(queries)
            self._log_miss(queries)
            return self._select(self._load_pool(), queries)
        # record mode (and any unknown mode degrades to live + persist)
        results = await self.base.search(queries, max_results=max_results)
        await self._persist(results)
        return results

    # -- persistence -----------------------------------------------------

    async def _persist(self, results: list[SearchResult]) -> None:
        self.qdir.mkdir(parents=True, exist_ok=True)
        async with self._lock:
            manifest = self._read_manifest()
            seen = {entry["url"] for entry in manifest}
            for result in results:
                url = (result.url or "").strip()
                if not url or url in seen:
                    continue
                content = result.raw_content or result.snippet or ""
                if not content.strip():
                    continue  # failed fetch with no snippet -> nothing to reuse
                fname = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16] + ".md"
                self._write_md(self.qdir / fname, result, content)
                manifest.append(
                    {"url": url, "file": fname, "title": result.title, "query": result.query}
                )
                seen.add(url)
            self._write_manifest(manifest)

    @staticmethod
    def _write_md(path: Path, result: SearchResult, content: str) -> None:
        front_matter = (
            "---\n"
            f"url: {json.dumps(result.url, ensure_ascii=False)}\n"
            f"title: {json.dumps(result.title, ensure_ascii=False)}\n"
            f"query: {json.dumps(result.query, ensure_ascii=False)}\n"
            f"fetched_at: {datetime.now(timezone.utc).isoformat()}\n"
            "---\n\n"
        )
        tmp = path.with_suffix(".md.tmp")
        tmp.write_text(front_matter + content, encoding="utf-8")
        tmp.replace(path)

    def _read_manifest(self) -> list[dict]:
        path = self.qdir / "manifest.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []

    def _write_manifest(self, manifest: list[dict]) -> None:
        path = self.qdir / "manifest.json"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load_pool(self) -> list[tuple[dict, str]]:
        pool: list[tuple[dict, str]] = []
        for entry in self._read_manifest():
            path = self.qdir / entry["file"]
            if not path.exists():
                continue
            body = _strip_front_matter(path.read_text(encoding="utf-8"))
            pool.append((entry, body))
        return pool

    def _select(self, pool: list[tuple[dict, str]], queries: list[str]) -> list[SearchResult]:
        n = self.fix_n if self.fix_n and self.fix_n > 0 else len(pool)
        query = queries[0] if queries else ""
        return [
            SearchResult(
                query=query,
                title=entry.get("title", ""),
                url=entry.get("url", ""),
                snippet="",
                raw_content=body,
            )
            for entry, body in pool[:n]
        ]

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
