import asyncio

import pytest

from deep_researcher_demo.schemas import SearchResult
from deep_researcher_demo.search import DuckDuckGoSearchProvider, TavilySearchProvider, create_search_provider, extract_text


def test_search_provider_factory_duckduckgo():
    provider = create_search_provider(
        "duckduckgo",
        fetch_webpages=True,
        max_content_chars=123,
        fetch_timeout=4.5,
        fetch_concurrency=2,
    )

    assert isinstance(provider, DuckDuckGoSearchProvider)
    assert provider.fetch_webpages is True
    assert provider.max_content_chars == 123
    assert provider.fetch_timeout == 4.5
    assert provider.fetch_concurrency == 2


def test_search_provider_factory_tavily():
    provider = create_search_provider("tavily")
    assert isinstance(provider, TavilySearchProvider)


def test_tavily_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    provider = TavilySearchProvider(api_key="")

    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        asyncio.run(provider.search(["test"], max_results=1))


def test_tavily_maps_response_to_search_results():
    provider = TavilySearchProvider(api_key="test-key")
    client = FakeTavilyClient(
        {
            "query": "query",
            "results": [
                {
                    "title": "Result title",
                    "url": "https://example.com/a",
                    "content": "Short content.",
                    "raw_content": "Full page content.",
                }
            ],
        }
    )

    results = asyncio.run(provider._search_one(client, "query", max_results=2))

    assert len(results) == 1
    assert results[0].query == "query"
    assert results[0].title == "Result title"
    assert results[0].url == "https://example.com/a"
    assert results[0].snippet == "Short content."
    assert results[0].raw_content == "Full page content."
    assert client.requests[0]["json"]["max_results"] == 2
    assert client.requests[0]["json"]["include_raw_content"] is True


def test_tavily_search_deduplicates_urls():
    provider = TavilySearchProvider(api_key="test-key")

    async def fake_search_one(client, query, max_results):
        return [
            SearchResult(query=query, title=f"{query} duplicate", url="https://example.com/shared", snippet=query),
            SearchResult(query=query, title=f"{query} unique", url=f"https://example.com/{query}", snippet=query),
        ]

    provider._search_one = fake_search_one  # type: ignore[method-assign]
    results = asyncio.run(provider.search(["q1", "q2"], max_results=1))

    assert [result.url for result in results] == [
        "https://example.com/shared",
        "https://example.com/q1",
        "https://example.com/q2",
    ]


def test_tavily_single_query_failure_returns_empty_batch():
    provider = TavilySearchProvider(api_key="test-key")
    results = asyncio.run(provider._search_one(FailingTavilyClient(), "query", max_results=1))

    assert results == []


def test_extract_text_removes_noise_tags():
    html = """
    <html>
      <head><style>.x { color: red; }</style><script>alert("x")</script></head>
      <body><h1>Title</h1><noscript>hidden</noscript><p>Hello   world</p></body>
    </html>
    """

    text = extract_text(html)

    assert "Title" in text
    assert "Hello world" in text
    assert "alert" not in text
    assert "hidden" not in text
    assert "color" not in text


def test_duckduckgo_fetch_webpages_false_leaves_raw_content_empty(monkeypatch):
    provider = DuckDuckGoSearchProvider(fetch_webpages=False)

    monkeypatch.setattr(
        provider,
        "_search_one",
        lambda query, max_results: [
            SearchResult(
                query=query,
                title="Example",
                url="https://example.com",
                snippet="Snippet only.",
            )
        ],
    )

    results = asyncio.run(provider.search(["query"], max_results=1))

    assert len(results) == 1
    assert results[0].snippet == "Snippet only."
    assert results[0].raw_content is None


def test_fetch_one_sets_truncated_raw_content():
    provider = DuckDuckGoSearchProvider(max_content_chars=12)
    result = SearchResult(query="q", url="https://example.com", snippet="snippet")

    fetched = asyncio.run(provider._fetch_one(FakeClient(FakeResponse()), result, asyncio.Semaphore(1)))

    assert fetched.raw_content == "Page Title\nH"


def test_fetch_one_falls_back_on_fetch_error():
    provider = DuckDuckGoSearchProvider()
    result = SearchResult(query="q", url="https://example.com", snippet="snippet")

    fetched = asyncio.run(provider._fetch_one(FailingClient(), result, asyncio.Semaphore(1)))

    assert fetched.raw_content is None
    assert fetched.snippet == "snippet"


def test_fetch_one_ignores_non_html_content():
    provider = DuckDuckGoSearchProvider()
    result = SearchResult(query="q", url="https://example.com/file.pdf", snippet="snippet")

    fetched = asyncio.run(
        provider._fetch_one(
            FakeClient(FakeResponse(content_type="application/pdf", text="%PDF")),
            result,
            asyncio.Semaphore(1),
        )
    )

    assert fetched.raw_content is None


class FakeResponse:
    def __init__(self, *, content_type: str = "text/html", text: str | None = None) -> None:
        self.headers = {"content-type": content_type}
        self.text = text or """
        <html>
          <body>
            <h1>Page Title</h1>
            <script>bad()</script>
            <p>Hello from the page body.</p>
          </body>
        </html>
        """

    def raise_for_status(self) -> None:
        return None


class FakeClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    async def get(self, url: str) -> FakeResponse:
        return self.response


class FailingClient:
    async def get(self, url: str) -> FakeResponse:
        raise RuntimeError("network failed")


class FakeTavilyResponse:
    def __init__(self, data: dict) -> None:
        self.data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self.data


class FakeTavilyClient:
    def __init__(self, data: dict) -> None:
        self.data = data
        self.requests = []

    async def post(self, url: str, **kwargs) -> FakeTavilyResponse:
        self.requests.append({"url": url, **kwargs})
        return FakeTavilyResponse(self.data)


class FailingTavilyClient:
    async def post(self, url: str, **kwargs) -> FakeTavilyResponse:
        raise RuntimeError("tavily failed")
