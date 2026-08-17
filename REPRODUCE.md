# REPRODUCE.md — 从零复现指南

> 面向拿到本仓库的零上下文新人:按顺序走完 环境重建 → 资源就位 → 起服务 → 冒烟 → 完整实验。
> 本文只做导航和骨架,细节以各节链接的原文档为准。

## 0. 这个仓库是什么

`deep_researcher_demo` 是一个纯 Python(无 LangChain/LangGraph)的 deep researcher agent:
supervisor 拆题 → researcher 搜索+摘要 → supervisor 决定追问 → final writer 出 Markdown 报告
(见 [README.md](README.md) 与 [claude-docx/workflow.md](claude-docx/workflow.md))。
在它之上做了三件事:

1. **三个 benchmark 评测**:DeepSearchQA / DeepResearchBench(RACE+FACT)/ DeepResearchGym(KPR+Quality+Citation);
2. **search_cache 机制**:把联网检索固化成本地缓存,replay 模式零联网、可复现(推理优化实验的数据底座);
3. **EAGLE-3 域内训练**:用 agent 自产轨迹训 Qwen3-32B 的投机解码 draft head(训练本体在 SpecForge fork)。

复现按此分层,做到哪层看你的需求。

## 1. 资源位置(代码 / 权重 / 数据)

| 资源 | 位置 | 说明 |
|---|---|---|
| 本仓库代码 | https://github.com/julyanghar/deep_researcher_demo | 公开 |
| EAGLE-3 draft head 权重(二期)+ 二期训练数据 | https://huggingface.co/julyanghar/Efficient-DRAgent | 公开;`model.safetensors`(1.4G)+ `config_deploy.json` + `data/train_main.jsonl`(7,681 条)+ heldout |
| 大数据备份(私有) | https://huggingface.co/datasets/julyanghar/Efficient-DRAgent-data | `tars/` 12 个 tar.zst(search_cache 三份、benchmark 结果、TRASH/saved 存档、benchmark_gold、实验物证等)+ 一期训练数据 `train-Eagle3-data/`;布局与逐 tar 说明见该 repo 的 README,取用前按 `tars/SHA256SUMS` 校验 |
| KV 复用 / vLLM 服务端代码 | https://github.com/julyanghar/LMCache-yilin(私有,分支 `branch`) | vLLM server 配置在其 `server/vllm/<model>/`,补丁说明在其 `vllm_patch_backup/README.md` |
| EAGLE-3 训练框架 | https://github.com/julyanghar/SpecForge(sgl-project/SpecForge 的 fork) | 备份分支 `backup-env`、训练补丁分支 `pr-ropebuf*` / `pr-trim-a*` 等均已推到该 fork |
| DeepSearchQA 题集 | https://huggingface.co/datasets/google/deepsearchqa | 本仓库已 vendor 一份 `eval/deepresearchqa/data/deepsearchqa.jsonl`(在 git 里,开箱即用) |
| DeepResearchBench 上游 | https://github.com/Ayanami0730/deep_research_bench | 提供 `data/`(题集/criteria/参考报告),见 §5.2 |
| DeepResearchGym 上游 | https://github.com/cxcscmu/deepresearch_benchmarking | 提供 `key_point/` 与 `queries/`,见 §5.3 |
| 底座模型 | https://huggingface.co/Qwen/Qwen3-32B 、 https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507 | agent 主力 32B;example.env 默认写的是 30B-A3B |

> 本机约定:HF 权重统一放 `HF_HOME=/data/yilin/huggingface`,不放根盘。

## 2. conda 环境重建(gpt-deep)

细节见 [env/README.md](env/README.md)。Python 3.11,torch 2.9.1+cu128,pip 真实版本以
[env/requirements-gpt-deep.txt](env/requirements-gpt-deep.txt) 为准:

```bash
conda create -n gpt-deep python=3.11 -y
conda activate gpt-deep
# 先手动删掉 requirements 里两行:deep-researcher-demo(本仓库的 editable 安装)
# 和 flash_attn(agent/评测主链路用不到;真要装见 env/README.md 提的社区预编译轮)
pip install -r env/requirements-gpt-deep.txt
pip install -e .          # 在仓库根执行
pip install -e ".[eval]"  # 跑 benchmark 需要(datasets/pandas)
```

## 3. 配置 .env

Python CLI 默认加载仓库根的 `.env`(也可 `--env-file` 指定)。从模板复制:

```bash
cp example.env .env
```

**注意**:example.env 的默认 `MODEL` 是 30B-A3B;如果 §4 起的是 Qwen3-32B,把 `.env` 的
`MODEL` 改成 `Qwen3-32B`(必须与 served-model-name 一致,否则请求 404)。

[example.env](example.env) 里的全部变量(凭据一律占位符,别提交真 key):

| 变量 | 作用 |
|---|---|
| `OPENAI_BASE_URL` / `OPENAI_API_KEY` / `MODEL` | agent 打的 OpenAI 兼容端点(本地 vLLM 就 `http://localhost:30000/v1` + `dummy`);`MODEL` 必须 = server 的 served-model-name |
| `SUPERVISOR_MODEL` / `RESEARCHER_MODEL` / `SUMMARY_MODEL` / `FINAL_MODEL` | 可选,按角色覆盖模型,缺省都用 `MODEL` |
| `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` | 判分评委端点(默认 DashScope compatible-mode),见 §7 |
| `SEARCH_PROVIDER` | `duckduckgo`(默认,免 key)或 `tavily` |
| `TAVILY_API_KEY` | `SEARCH_PROVIDER=tavily` 时必需:`export TAVILY_API_KEY=<your-tavily-key>` |
| `MAX_ITERATIONS` / `MAX_FOLLOWUPS` / `MAX_QUERIES_PER_RESEARCHER` / `MAX_CONCURRENCY` / `MAX_RESULTS` | 研究广度/并发旋钮 |
| `FETCH_WEBPAGES` / `MAX_CONTENT_CHARS` / `FETCH_TIMEOUT` / `FETCH_CONCURRENCY` | 网页抓取参数 |
| `OUTPUT` | 报告落盘路径 |

完整默认值与解析逻辑见 [deep_researcher_demo/config.py](deep_researcher_demo/config.py)。

## 4. 起模型服务(OpenAI 兼容端点)

agent 只认 `OPENAI_BASE_URL`,任何 vLLM/SGLang 起的 OpenAI 兼容服务都行。最简起服(实验约定端口 30000、served-model-name=`Qwen3-32B`,eval 脚本的默认值都按这个写):

```bash
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 \
  --port 30000 --served-model-name Qwen3-32B
```

显存紧张可换 `Qwen/Qwen3-30B-A3B-Instruct-2507`(MoE,example.env 的默认 `MODEL`)。
就绪探测:日志出现 `Application startup complete`(32B/TP4 加载约 2-3 分钟)。

> KV-reuse(CacheBlend)/ 逐请求计时(`VLLM_RESP_TIMING=1`)等**实验版服务端**不在本仓库:
> 用 LMCache-yilin 私有仓的 `server/vllm/Qwen3-32B/start.sh`(`lmcache`/`no_lmcache` 两模式),
> 起法与 server env 全表见 [eval/EVAL_GUIDE.md](eval/EVAL_GUIDE.md) §1。

## 5. 数据/权重就位

### 5.1 DeepSearchQA(开箱即用)

题集已 vendor 在 git 里:`eval/deepresearchqa/data/deepsearchqa.jsonl`(本地优先;缺失时
`eval/deepresearchqa/run_deepsearchqa.py` 会从 HF `google/deepsearchqa` 自动下载落盘)。无需动作。

### 5.2 DeepResearchBench 数据(.gitignore 排除,需自取)

`eval/DeepResearchBench/data/`(题集 `prompt_data/query.jsonl`、`criteria_data/`、`test_data/`)不入 git,从上游取:

```bash
git clone https://github.com/Ayanami0730/deep_research_bench /tmp/drb
cp -r /tmp/drb/data eval/DeepResearchBench/data
```

### 5.3 DeepResearchGym 数据(.gitignore 排除,需自取)

`eval/DeepResearchGym/key_point/*.json` 与 `queries/*.jsonl` 不入 git(目录里只有 .py 脚本入库),从上游取:

```bash
git clone https://github.com/cxcscmu/deepresearch_benchmarking /tmp/drgym
cp /tmp/drgym/key_point/*.json eval/DeepResearchGym/key_point/
cp /tmp/drgym/queries/*.jsonl eval/DeepResearchGym/queries/
```

默认题集是 `queries/researchy_queries_sample_doc_click_100.jsonl`(`run_drgym.py` 的 `DEFAULT_QUERIES`)。

### 5.4 search_cache 与历史实验结果(HF 私有备份)

`eval/results/` 与 `eval/benchmarks/results/` 整体不入 git。要 hermetic 复跑(replay 零联网)或对账历史结果,从私有备份仓取(需该 repo 的访问权限):

```bash
pip install -U huggingface_hub   # 提供 hf 命令
hf download julyanghar/Efficient-DRAgent-data --repo-type dataset \
  --local-dir /tmp/dragent-data --token <your-hf-token>
# tar 内路径为仓库根相对路径(eval/...),须在仓库根(含 pyproject.toml 那层)解包:
cd /path/to/deep_researcher_demo
tar -I zstd -xf /tmp/dragent-data/tars/search_cache_drbench.tar.zst
tar -I zstd -xf /tmp/dragent-data/tars/search_cache_drgym.tar.zst
tar -I zstd -xf /tmp/dragent-data/tars/search_cache_drqa.tar.zst
```

完整清单(12 个 tar)与逐 tar 说明见备份仓 README;解包前对 `tars/SHA256SUMS` 逐一
`sha256sum -c` 校验(HF 上传限流期的查询结果不可信,只认哈希)。

没有备份权限也能自建缓存:用 record 模式实跑一遍(见 §6)。

### 5.5 EAGLE-3 权重(可选,只为投机解码实验)

```bash
hf download julyanghar/Efficient-DRAgent --local-dir phase2-12k-3ep
```

## 6. search_cache / RESEARCH_MODE 怎么生效(源码核实过)

缓存布局与用法见 [eval/results/search_cache/README.md](eval/results/search_cache/README.md)
(一题一目录:`search_cache.json` + `pages_index.json` + `pages/` + `chunks.jsonl` 块级 embedding 索引)。
env 的解析都在 [deep_researcher_demo/config.py](deep_researcher_demo/config.py)(`_resolve_cache_mode` 等),消费在
[deep_researcher_demo/search.py](deep_researcher_demo/search.py) / [deep_researcher_demo/relevance.py](deep_researcher_demo/relevance.py):

| env | 取值 | 作用 |
|---|---|---|
| `RESEARCH_MODE` | `online` / `local` / `off`(默认) | **优先级最高**:`online`→缓存 record + 块级 embedding 检索;`local`→replay(零联网)+ 块检索;`off`→退回老 `SEARCH_CACHE` 行为 |
| `SEARCH_CACHE` | `off` / `record` / `replay` | 老开关(`RESEARCH_MODE` 未设时才生效) |
| `SEARCH_BENCHMARK` | `drbench` / `drgym` / `drqa` | 缓存按 `<SEARCH_CACHE_DIR>/<benchmark>/q<id>/` 分类 |
| `SEARCH_CACHE_DIR` | 默认 `eval/results/search_cache` | 持久缓存根(别放进 per-run OUTPUT_DIR) |
| `SEARCH_CACHE_SAMPLE_ID` | 题号 | 题目录 id(eval runner 自动传) |
| `SEARCH_CACHE_FIX_N` | 整数 | replay 时每 query 固定取前 N 篇(A/B 公平) |
| `EMBED_API_KEY` / `EMBED_BASE_URL` / `EMBED_MODEL` | 默认落到 `JUDGE_*` + `text-embedding-v3`(DashScope) | online/local 的块级 embedding;drqa 线默认改用 `gemini-embedding-001`(`run_drqa.py` 里 setdefault) |

**铁律**:计时/对比实验一律先 record 建库一次,之后全部 replay;验证零联网看产物里
`search_cache_miss` 是否全 0(见 [eval/EVAL_GUIDE.md](eval/EVAL_GUIDE.md) §4)。

## 7. 判分评委(JUDGE_*)

- 三处评测都走 OpenAI 兼容评委端点,配置从 `.env` 的 `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL` 读;
- 默认模型两套(代码里核实):**DeepSearchQA** 默认 `deepseek-v3.2`(`deep_researcher_demo/config.py` 的 `DEFAULT_JUDGE_MODEL`);**DRBench/DRGym** 默认 `kimi-k2.5`([eval/benchmarks/judge_adapter.py](eval/benchmarks/judge_adapter.py) 的 `judge_settings`,并自动把 `OPENAI_*`/`RACE_MODEL`/`FACT_MODEL` 指到评委端点,无需改 benchmark 源码);
- **`.env` 里的 `JUDGE_MODEL` 会覆盖以上代码兜底值**——跑 DRBench/DRGym score 前确认它是
  `kimi-k2.5`(或临时 `export JUDGE_MODEL=kimi-k2.5`),别让 example.env 抄来的默认值顶掉;
- `export JUDGE_API_KEY=<your-dashscope-key>`,只在 score 阶段必需(generate 不用);
- DRBench 的 FACT 抓网页走 Jina:可选 `export JINA_API_KEY=<your-jina-key>`,不配则 scrape 取不到正文、validate 偏低(见 [eval/benchmarks/README.md](eval/benchmarks/README.md) 的说明)。

## 8. 最小冒烟

```bash
# ① 单测:无网络、无 GPU,应 73 passed
python -m pytest -q

# ② agent 单题端到端(需 §4 的端点 + 联网 DuckDuckGo 搜索)
./run_example.sh "What are the main tradeoffs of local LLM inference?"
# 报告默认落 outputs/example_report.md(.env 的 OUTPUT)

# ③ eval 链路冒烟(2 题,只生成不判分,不需要 JUDGE key)
MODE=generate LIMIT=2 OUTPUT_DIR=eval/results/smoke ./eval_deepsearchqa.sh
```

## 9. 完整实验

### 9.1 DeepSearchQA

入口 [eval_deepsearchqa.sh](eval_deepsearchqa.sh)(头部注释列了每个参数),
Python 本体 [eval/deepresearchqa/run_deepsearchqa.py](eval/deepresearchqa/run_deepsearchqa.py):

```bash
export JUDGE_API_KEY=<your-dashscope-key>
MODE=all LIMIT=50 OUTPUT_DIR=eval/results/deepsearchqa_50 ./eval_deepsearchqa.sh
# 断点续(完整写法):
MODE=all LIMIT=50 OUTPUT_DIR=eval/results/deepsearchqa_50 RESUME=1 OVERWRITE=0 ./eval_deepsearchqa.sh
# 全量:LIMIT=""
```

产物:`OUTPUT_DIR/{reports.jsonl,predictions.jsonl,metrics.json,failures.jsonl}`。
另有 harvest 版驱动 [eval/deepresearchqa/run_drqa.py](eval/deepresearchqa/run_drqa.py)(per-题 report/harvest/计时)+
[eval/deepresearchqa/score_drqa.py](eval/deepresearchqa/score_drqa.py)(官方 autorater 口径,可续跑)。

### 9.2 DeepResearchBench(RACE + FACT)

入口 [eval/DeepResearchBench/run_drbench.py](eval/DeepResearchBench/run_drbench.py)(自包含 vendored scorer,需先做 §5.2):

```bash
python eval/DeepResearchBench/run_drbench.py --tag demo --mode all --n 100 \
    --metrics race,fact --concurrency 2 --workers 4
# generate 需 §4 服务;score 只需评委。--harvest 子进程跑法留 per-题 harvest/计时
# 汇总: eval/results/drbench/demo/summary.json(RACE Overall + 4 维)
```

### 9.3 DeepResearchGym(Quality / KPR / Citation 精·召)

入口 [eval/DeepResearchGym/run_drgym.py](eval/DeepResearchGym/run_drgym.py)(需先做 §5.3):

```bash
python eval/DeepResearchGym/run_drgym.py --tag demo --mode all --n 100 \
    --metrics quality,kpr,citation_recall,citation_precision --concurrency 2
# 汇总: eval/benchmarks/results/drgym/demo/summary.json
```

注意:citation_precision 用 demo 自己的 search 缓存核引用 → generate 时要处于 record/online 模式,否则该指标偏 0(原理见 [eval/benchmarks/README.md](eval/benchmarks/README.md))。

### 9.4 online 实跑(Tavily 建缓存)

[eval/run/README.md](eval/run/README.md) 有完整说明与踩坑记录:

```bash
export TAVILY_API_KEY=<your-tavily-key>
bash eval/run/run_online_tavily.sh 3 1                       # DRBench 前 3 题试跑
bash eval/run/run_drbench_online_100.sh 100 3 online100      # 全量 100 题,断点续
```

### 9.5 KV-reuse A/B、逐请求计时等推理优化实验

档位/模式/缓存/RUN_ID 纪律全在 [eval/EVAL_GUIDE.md](eval/EVAL_GUIDE.md)(§7 有一轮完整可复制示例);服务端在 LMCache-yilin 私有仓。

## 10. EAGLE-3 训练与部署(可选层)

- **精确复现指针先看 [PROVENANCE.md](PROVENANCE.md)**:外部权重 revision、训练代码版本、数据谱系、已知损失都在那里落账;
- **线索文档**:[train/Eagle3/README.md](train/Eagle3/README.md)(数据谱系/三道闸验证/训练命令)、转换脚本 [train/Eagle3/convert_harvest_to_eagle3.py](train/Eagle3/convert_harvest_to_eagle3.py)(harvest → SpecForge conversations 格式)、真实启动脚本与训练日志 [train/Eagle3/run-artifacts/](train/Eagle3/run-artifacts/)、验收 harness [train/Eagle3/eval-harness/](train/Eagle3/eval-harness/)——这些**已 force-add 进 git**(`train/` 目录整体仍被 .gitignore,大文件不进);数据从 HF 取:一期在数据仓 `train-Eagle3-data/`(`summary_train_main.jsonl` 等),二期随权重放在 model 仓 `julyanghar/Efficient-DRAgent` 的 `data/`;
- **训练本体**:SpecForge fork(https://github.com/julyanghar/SpecForge )。**训练用的是 commit `77f4a0f`(分支 `yilin-trim-mem-probes`)**——PR 贡献线 pr-trim-a-v3 的 scripts/ 里没有 train_eagle3.py,复现训练必须 checkout 77f4a0f(详见 PROVENANCE.md);二期完整超参记录在权重仓 README + `training_state.pt` 的 args(79 项);
- **部署**(vLLM 零转换,权重按 §5.5 就位):

```bash
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 \
  --speculative-config '{"method":"eagle3","model":"./phase2-12k-3ep","num_speculative_tokens":3}'
```

(部署用 `config_deploy.json` 替换 `config.json`,权重仓 README 有说明。)

## 11. 文档导航

| 想了解 | 看 |
|---|---|
| agent 用法/参数 | [README.md](README.md) |
| 复现资源指针(revision/代码版本/数据谱系/评委真值) | [PROVENANCE.md](PROVENANCE.md) |
| conda 环境 | [env/README.md](env/README.md) |
| workflow/搜索机制源码解读 | [claude-docx/](claude-docx/)(workflow.md、search-call-chain.md、research-modes-and-search.md、research-walkthrough.md) |
| eval 传参 runbook(A/B + blend) | [eval/EVAL_GUIDE.md](eval/EVAL_GUIDE.md) |
| 三 benchmark 集成与指标口径 | [eval/benchmarks/README.md](eval/benchmarks/README.md) |
| online 实跑脚本 | [eval/run/README.md](eval/run/README.md) |
| 实验结果目录地图(不入 git) | [eval/results/README.md](eval/results/README.md)、[eval/results/search_cache/README.md](eval/results/search_cache/README.md) |
| 实验设计/结论(投机解码、EAGLE-3、论文) | [exp-docx/](exp-docx/) |
| vLLM 服务端补丁 | LMCache-yilin 私有仓的 `vllm_patch_backup/README.md` |
