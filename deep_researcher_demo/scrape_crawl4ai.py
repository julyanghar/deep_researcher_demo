"""crawl4ai-based webpage fetcher — replaces the fragile httpx fetch.

httpx GET 抓不动 JS 渲染/反爬的站点(正文常空)。crawl4ai 用无头 Chromium 渲染后
抽干净 Markdown,正文命中率高得多。

运行时坑:Playwright 的 Chromium 依赖系统库 libasound.so.2,本机没装(无 sudo),
所以我们自带一份并用 LD_LIBRARY_PATH 指过去。Chromium 是 Playwright 起的**子进程**,
只要在它启动前把 LD_LIBRARY_PATH 写进本进程 os.environ,子进程就能继承、加载到。
"""
import os
from pathlib import Path

# 自带 libasound.so.2 的永久目录(出 /home/yilin/tmp,可被 CRAWL4AI_LIBS_DIR 覆盖)。
_LIBS_DIR = os.getenv("CRAWL4AI_LIBS_DIR", "/home/yilin/crawl4ai-libs")


def _ensure_ld_library_path() -> None:
    """把自带 libs 目录前插进 LD_LIBRARY_PATH(幂等),供 Chromium 子进程继承。"""
    if not Path(_LIBS_DIR).is_dir():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if _LIBS_DIR not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = _LIBS_DIR + ((":" + current) if current else "")


# import 期就设好,确保任何后续 crawl4ai 浏览器启动都带上。
_ensure_ld_library_path()


async def fetch_pages(
    urls: list[str],
    *,
    max_content_chars: int = 12000,
    timeout_s: float = 30.0,
) -> dict[str, str]:
    """抓一批 URL → {输入url: 干净Markdown正文}。失败/空的 URL 不在返回里。

    按输入顺序与结果对齐(crawl4ai 保序),用**输入 url** 作键,这样调用方的
    SearchResult.url 能直接对上(即使发生重定向)。
    """
    urls = [u for u in (urls or []) if u]
    if not urls:
        return {}
    _ensure_ld_library_path()
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode

    browser = BrowserConfig(
        headless=True,
        # 服务器/容器里跑无头 Chromium 的常规两件套
        extra_args=["--no-sandbox", "--disable-dev-shm-usage"],
        verbose=False,
    )
    run_cfg = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,   # 我们自己管缓存(SEARCH_CACHE),不用 crawl4ai 的
        page_timeout=int(timeout_s * 1000),
        verbose=False,
        stream=False,
    )
    out: dict[str, str] = {}
    async with AsyncWebCrawler(config=browser) as crawler:
        results = await crawler.arun_many(urls=urls, config=run_cfg)
        for url, r in zip(urls, results):
            if not getattr(r, "success", False):
                continue
            text = _markdown_text(r)
            if text.strip():
                out[url] = text[:max_content_chars]
    return out


def _markdown_text(result) -> str:
    """crawl4ai 0.9 的 result.markdown 可能是 str 或 MarkdownGenerationResult,统一取文本。"""
    md = getattr(result, "markdown", "") or ""
    if isinstance(md, str):
        return md
    # 对象形态:优先 raw_markdown / fit_markdown
    return getattr(md, "raw_markdown", "") or getattr(md, "fit_markdown", "") or str(md)
