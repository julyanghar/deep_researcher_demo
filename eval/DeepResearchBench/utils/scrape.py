"""FACT pipeline 的 citation 抓取步:本地 crawl4ai(替代 Jina)。

一个 AsyncWebCrawler 起一次浏览器,用 asyncio 并发抓每篇报告缺内容的引用 URL,
回填 citations_deduped[url]['url_content']。按 id 断点续跑,输出形状与原版一致:
成功 -> url_content = f"{title}\n\n{description}\n\n{content}"
失败 -> url_content = "scrape failed: <err>"(validate 按不可核验处理,与 Jina 失败等价)。

并发上限沿用 --n_total_process(用一个 asyncio.Semaphore 卡住,不再起多进程)。
"""
import json
import os
import argparse
import asyncio
from pathlib import Path
from tqdm import tqdm
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, BrowserConfig, CacheMode
from .io_utils import load_jsonl


PAGE_TIMEOUT_MS = int(os.environ.get("SCRAPE_PAGE_TIMEOUT_MS", "60000"))

# crawl4ai 的 Chromium 缺系统库 libasound.so.2(本机无 sudo);自带一份,import 期就把
# 目录塞进 LD_LIBRARY_PATH,Chromium 是子进程会继承到。见 deep_researcher_demo/scrape_crawl4ai.py。
_LIBS_DIR = os.getenv("CRAWL4AI_LIBS_DIR", "/home/yilin/crawl4ai-libs")


def _ensure_ld_library_path() -> None:
    if not Path(_LIBS_DIR).is_dir():
        return
    cur = os.environ.get("LD_LIBRARY_PATH", "")
    if _LIBS_DIR not in cur.split(":"):
        os.environ["LD_LIBRARY_PATH"] = _LIBS_DIR + ((":" + cur) if cur else "")


_ensure_ld_library_path()


def _markdown_text(result) -> str:
    """result.markdown 可能是 str 或 MarkdownGenerationResult(有 .raw_markdown)。"""
    md = getattr(result, "markdown", None)
    if md is None:
        return ""
    if isinstance(md, str):
        return md
    raw = getattr(md, "raw_markdown", None)
    return raw if isinstance(raw, str) else str(md)


def _content_from_result(result) -> str:
    """把 crawl4ai 结果拼成原版 scrape 的 url_content 形状,或返回失败串。"""
    if result is None or not getattr(result, "success", False):
        err = getattr(result, "error_message", "") if result is not None else "no result"
        return f"scrape failed: {err or 'unknown error'}"
    meta = getattr(result, "metadata", None) or {}
    title = meta.get("title", "") or ""
    description = meta.get("description", "") or ""
    content = _markdown_text(result)
    if not content.strip():
        return "scrape failed: empty content"
    return f"{title}\n\n{description}\n\n{content}"


async def _scrape_one(crawler, url, config, sem):
    async with sem:
        try:
            result = await crawler.arun(url, config=config)
            return url, _content_from_result(result)
        except Exception as e:  # noqa: BLE001
            return url, f"scrape failed: {e}"


async def _scrape_urls(crawler, urls, config, sem) -> dict:
    if not urls:
        return {}
    pairs = await asyncio.gather(*[_scrape_one(crawler, u, config, sem) for u in urls])
    return dict(pairs)


async def _run(args):
    try:
        raw_data = load_jsonl(args.raw_data_path)
    except Exception:
        import sys
        print(f"cannot process file {args.raw_data_path}")
        sys.exit(f'{args.raw_data_path} has not been processed yet...')

    processed = set()
    if os.path.exists(args.output_path):
        processed = {d['id'] for d in load_jsonl(args.output_path)}
    data_to_process = [d for d in raw_data if d['id'] not in processed]
    print(f"processing {len(data_to_process)} instances...")

    config = CrawlerRunConfig(word_count_threshold=0, page_timeout=PAGE_TIMEOUT_MS,
                              cache_mode=CacheMode.BYPASS, verbose=False)
    browser = BrowserConfig(headless=True, verbose=False,
                            extra_args=["--no-sandbox", "--disable-dev-shm-usage"])
    sem = asyncio.Semaphore(max(1, args.n_total_process))

    crawler = AsyncWebCrawler(config=browser)
    await crawler.start()
    try:
        for d in tqdm(data_to_process):
            # 只抓还没有 url_content 的引用
            urls = [k for k, v in d['citations_deduped'].items()
                    if 'url_content' not in v or not v['url_content']]
            url2content = await _scrape_urls(crawler, urls, config, sem)
            for u, content in url2content.items():
                d['citations_deduped'][u]['url_content'] = content
            with open(args.output_path, 'a+', encoding='utf-8') as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    finally:
        await crawler.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--raw_data_path", type=str, required=True)
    parser.add_argument("--n_total_process", type=int, default=1)
    args = parser.parse_args()

    asyncio.run(_run(args))
