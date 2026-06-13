"""Search provider abstractions for the simplified deep researcher."""

import asyncio
import os
import re
import threading
import time
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
