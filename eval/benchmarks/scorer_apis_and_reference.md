# Scorer 外部 API + RACE 参考报告(DeepResearchBench / DeepResearchGym 打分内部机制)

> 两个 benchmark 的 scorer 各用**两类外部 API**:一个 **LLM(打分引擎)**、一个 **网页抓取(给引用核验取证据)**。本文说清各在哪、起什么作用,并定位 RACE 的"高质量参考报告"。
> 代码已 vendored 进自包含目录(见 [README](README.md));LLM 统一经 [judge_adapter.py](judge_adapter.py) 重接到 demo 的 **DashScope JUDGE(kimi-k2.5)**。

---

## 一、两类外部 API 一览
| 类型 | 谁用 | 作用 | 哪些指标需要 |
|---|---|---|---|
| **LLM(打分引擎)** | 两个 repo 全部 scorer | 做所有"语义判断"(报告好不好 / 要点覆盖没 / 引用支不支持)| 全部指标 |
| **网页抓取** | DRBench=Jina,DRGym=crawl4ai | 把报告**引用的网页正文**抓回来,作"参考资料"给 LLM 判 claim 真假 | 只有**引用/事实类**(FACT、Citation);RACE/Quality/KPR **不抓网页** |

LLM 没有就没法打分;抓取只为引用核验取证据。

---

## 二、DeepResearchBench
| API | 在哪 | 作用 |
|---|---|---|
| **LLM**(OpenAI 兼容 chat completions)| `DeepResearchBench/utils/api.py` 的 `AIClient`/`call_model`(`requests.post .../chat/completions`)| **RACE**(`deepresearch_bench_race.py`)按 criteria+参考报告打 4 维;**FACT** 的 `extract.py`(抽 claim)/`deduplicate.py`(去重)/`validate.py`(判 claim 是否被引用页支持);`generate_criteria.py`(动态生成 criteria,已预生成在 `data/criteria_data/criteria.jsonl`,打分时一般不重跑)|
| **Jina Reader**(`https://r.jina.ai/<url>`,`JINA_API_KEY` Bearer)| `utils/api.py` 的 `scrape_url`(约 L213)| **FACT** 的 `scrape.py`:抓取报告引用的网页正文,作参考资料喂给 validate 判真假。非 LLM,纯网页抓取 |

默认 backend 是 openrouter/openai,`judge_adapter` 把 `OPENAI_BASE_URL`/`RACE_MODEL`/`FACT_MODEL` 重接到 DashScope kimi。

## 三、DeepResearchGym
| API | 在哪 | 作用 |
|---|---|---|
| **LLM**(`AsyncOpenAI`,`beta.chat.completions.parse` 结构化输出)| 4 个 `eval_*_async.py` 模块级 `client = AsyncOpenAI(...)` | Quality 5 维、KPR 判 key point 是否被覆盖、Citation 判 claim 是否被引用文档支持 |
| **crawl4ai**(本地无头浏览器抓取)| `eval_citation_async.py`(`AsyncWebCrawler.arun_many`)| **citation_precision** 抓引用 URL 正文。⚠️ 在 [run_drgym.py](../DeepResearchGym/run_drgym.py) 里被 **monkey-patch 成用 demo 的 search 缓存替代**(免装 crawl4ai + 复用 `SEARCH_CACHE=record` 已存页)|

LLM 同样经 `judge_adapter.apply_dashscope_env()` 重接 DashScope kimi。

---

## 四、crawl4ai 与 Jina Reader 是什么
两者目标一样:**把网页抓下来、清洗成干净 Markdown 喂给 LLM**;形态相反——一个云服务、一个本地库。

| | **Jina Reader** | **crawl4ai** |
|---|---|---|
| 形态 | Jina AI 的**云端 HTTP 服务** | **开源 Python 库** |
| 怎么跑 | URL 前拼 `https://r.jina.ai/`,**远端**帮你抓+渲染(含JS)+去噪→返回正文 Markdown | **本机**起无头浏览器(Playwright/Chromium)自己抓+渲染→Markdown(`AsyncWebCrawler` / `arun_many` 并发)|
| 认证/费用 | 要 `JINA_API_KEY`、有额度/速率限制 | 无 key,免费 |
| 安装负担 | 几乎为零 | 重(浏览器+Playwright 依赖)|
| 数据流向 | URL 发给 Jina 服务器 | 全在本地 |
| 谁用 | DeepResearchBench(FACT scrape)| DeepResearchGym(citation precision)|

**我们为何替换 crawl4ai**:它要装浏览器依赖、又得现抓网页 → 在 `run_drgym` 里换成 **demo 的 search 缓存**(复用已存引用页),既免装又可复现。DRBench 的 Jina(简单 HTTP)保留。

---

## 五、RACE 的"高质量参考报告"在哪
**`DeepResearchBench/data/test_data/cleaned_data/reference.jsonl`**(已 vendored;原 repo 同路径)。

- **100 行**,每行 `{id, prompt, article}`——**每个 query 一篇参考报告**(按 `prompt` 匹配),article 是完整长报告(~20K+ 字)。
- 同目录 `claude-3-7-sonnet-latest.jsonl` 是某被测模型的报告(作 target,不是参考)。

### RACE 怎么用它(`deepresearch_bench_race.py`)
```
L30   REFERENCE_FILE = "data/test_data/cleaned_data/reference.jsonl"   # 硬编码相对路径
L258  all_reference_articles = load_jsonl(REFERENCE_FILE)             # 加载
L70   reference_articles_map[prompt]                                  # 按 prompt 取
L102  article_1 = target(你的报告), article_2 = reference_article     # 喂给评委
L160  overall_score = target_total / (target_total + reference_total) # 相对分!
```
**RACE 是"相对分",参考报告是分母锚点**:评委对 target 与 reference 各按 criterion 打分,最终 `Overall = target / (target + reference)`(每维同理 L170)。
- =0.5:与参考打平;<0.5:不如参考;>0.5:超过参考。

### 跟实测对应
小样实测 **Overall 0.3047**(见 [harvest_run_report 同目录的 mini2 结果])→ 我们 32B 报告明显弱于参考报告(参考是强 DR 系统产出的高质量长报告),符合预期。

**一句话**:参考报告 = benchmark **自带的、每题一篇的高质量对照报告**,不是"逐字匹配的标准答案",而是 RACE **相对打分的基准线**;已随 `data/` vendored 进 `eval/DeepResearchBench/`,自包含。

---

## 附:这些是 scorer(打分)侧的外部 API
报告**生成**侧用的是本地 vLLM(OpenAI 兼容),不算外部。三类外部依赖小结:
- **DashScope kimi**(评委 LLM,`.env` 的 `JUDGE_*`)——所有打分必需。
- **Jina**(`JINA_API_KEY`)——仅 DRBench FACT 抓引用页需要。
- **crawl4ai**——DRGym citation 原需要,已用 demo 缓存替代(实际不再调用)。
