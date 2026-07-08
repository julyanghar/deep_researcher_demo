# crawl4ai 在本机的安装与使用

> 验证日期：2026-06-30 · 机器：Ubuntu 22.04 / x86_64 / Python 3.11 · 状态：✅ 已跑通

本机**能装、能用** crawl4ai。下面是现成可复用的用法，以及安装时踩的唯一一个坑的说明。

## 一句话用法

```bash
source /home/yilin/tmp/crawl4ai-venv/bin/activate   # activate 已自动带上缺的系统库路径
python 你的脚本.py
```

最小示例（已验证，文件在 `/home/yilin/tmp/test_crawl.py`）：

```python
import asyncio
from crawl4ai import AsyncWebCrawler

async def main():
    async with AsyncWebCrawler() as crawler:
        r = await crawler.arun(url="https://example.com")
        print(r.success, r.status_code)   # True 200
        print(r.markdown)                  # 提取出的 Markdown 正文

asyncio.run(main())
```

## 装在哪 / 装了啥

| 项 | 位置 / 版本 |
|---|---|
| Python 虚拟环境 | `/home/yilin/tmp/crawl4ai-venv/`（独立 venv，**没动 conda 的 gpt-deep 环境**）|
| crawl4ai | 0.9.0 |
| 浏览器内核 | Playwright Chromium，下到 `~/.cache/ms-playwright/` |
| 手动补的系统库 | `/home/yilin/tmp/crawl4ai-libs/libasound.so.2`（见下文）|

验证过的站点：example.com、quotes.toscrape.com、en.wikipedia.org/wiki/Web_scraping —— 全部 HTTP 200，Markdown 正常提取（Wikipedia 抓出 ~68K 文本 + 319 链接）。

## 安装时的唯一一个坑：缺 libasound.so.2

**现象**：直接装完后第一次抓取失败，Chromium 起不来，报
`error while loading shared libraries: libasound.so.2: cannot open shared object file`（exitCode=127）。

**原因**：Playwright 的 Chromium 依赖系统库 `libasound.so.2`（ALSA 音频库），本机没装。
`ldd` 检查过，Chromium **只缺这一个**库。

**为什么不用标准办法装**：标准做法是 `sudo apt install libasound2`（或官方 `playwright install-deps`），
但本机**没有免密 sudo**，apt 装不了。

**采用的绕过办法**：系统里别人的 conda 环境已有一份标准 ALSA 库
（`/data/zheyu/anaconda3/pkgs/alsa-lib-1.2.14-.../lib/libasound.so.2`）。
只把这**一个 .so 文件**拷到独立目录 `/home/yilin/tmp/crawl4ai-libs/`，
再用 `LD_LIBRARY_PATH` 喂给 Chromium。好处：不需要 root，不污染任何环境，
拷来的是独立副本（对方删环境也不影响）。

这条 `LD_LIBRARY_PATH` 已经追加进 venv 的 `activate` 脚本，所以**平时 `source` 一下就自动生效，不用手动设**。activate 末尾那行：

```bash
export LD_LIBRARY_PATH="/home/yilin/tmp/crawl4ai-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

## 等以后拿到 sudo 了，怎么收尾（可选）

这套是「没 sudo 时的权宜之计」。哪天有 root 权限，干净做法是：

```bash
sudo apt install -y libasound2          # 或 python -m playwright install-deps chromium
```

然后就可以删掉 `/home/yilin/tmp/crawl4ai-libs/`，并把 activate 脚本末尾那行
`export LD_LIBRARY_PATH=...` 删掉，回到标准状态。

## 注意事项

- 当前是**无头（headless）**模式，纯抓取/正文提取没问题；截图、跑重 JS 的站点也支持。
- venv 放在 `/home/yilin/tmp/` 下，属临时区。若要长期保留/纳入项目，建议挪到项目内或重建一个固定位置的 venv（重建后记得重新执行 `python -m playwright install chromium`，并把 libasound 那行补回 activate）。
- 进阶能力（LLM 结构化抽取、批量并发抓取、截图）尚未在本机逐一验证，需要时可再测。
