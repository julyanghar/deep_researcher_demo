# Deep Researcher Demo

A small pure-Python implementation of a simplified deep researcher workflow.

It intentionally does not use LangChain or LangGraph. The workflow is a normal
Python async loop:

```text
user question
  -> supervisor decomposes it into standalone research sub-questions
  -> researcher decomposes each sub-question into web search queries
  -> search provider returns results
  -> researcher summarizes key findings
  -> supervisor decides whether follow-up questions are needed
  -> final writer creates a Markdown report
```

## Quickstart

```bash
python -m deep_researcher_demo "What are the main tradeoffs of local LLM inference?"
```

You can also use the example launcher. By default, the Python CLI loads
`/home/yilin/deep_researcher_demo/.env`:

```bash
./run_example.sh
./run_example.sh "Research recent trends in local LLM inference"
```

If you want a local editable env file, copy the example first:

```bash
cp example.env .env
./run_example.sh "Research recent trends in local LLM inference"
```

To load a different env file explicitly:

```bash
python -m deep_researcher_demo --env-file example.env "What are local LLM inference tradeoffs?"
```

For a real OpenAI-compatible endpoint:

```bash
export OPENAI_BASE_URL=http://localhost:30000/v1
export OPENAI_API_KEY=dummy
export MODEL=your-model-name

python -m deep_researcher_demo "Research recent trends in local LLM inference"
```

DuckDuckGo is the default search provider. It fetches each search result page
and passes extracted page text to the researcher by default; if a page cannot be
fetched or parsed, the workflow falls back to the search snippet.

Tavily is also available:

```bash
export SEARCH_PROVIDER=tavily
export TAVILY_API_KEY=your-tavily-api-key
```

DuckDuckGo uses free search results and this demo fetches page text itself with
`httpx` and BeautifulSoup. Tavily uses the Tavily API and can return
`content/raw_content` directly, so the demo does not do an extra webpage fetch
for Tavily. For both providers, `MAX_RESULTS` means results per search query.

The main research breadth controls are:

```env
MAX_FOLLOWUPS=3
MAX_QUERIES_PER_RESEARCHER=3
MAX_RESULTS=5
```

`MAX_FOLLOWUPS` limits how many researcher tasks the supervisor can create in
one round. `MAX_QUERIES_PER_RESEARCHER` limits how many search queries each
researcher can derive from one sub-question. `MAX_RESULTS` limits results per
search query.

## DeepSearchQA Evaluation

Install eval dependencies when you want to run the benchmark:

```bash
pip install -e ".[eval]"
```

Run a small benchmark job:

```bash
MODE=all LIMIT=2 OUTPUT_DIR=eval/results/smoke ./eval_deepsearchqa.sh
MODE=all LIMIT=2 OUTPUT_DIR=eval/results/smoke RESUME=1 OVERWRITE=0 ./eval_deepsearchqa.sh
```

Run the Hugging Face `google/deepsearchqa` eval split:

```bash
export JUDGE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export JUDGE_API_KEY=your-dashscope-api-key
export JUDGE_MODEL=deepseek-v3.2
./eval_deepsearchqa.sh
```

The evaluator supports three modes:

```bash
MODE=generate LIMIT=50 ./eval_deepsearchqa.sh
MODE=score LIMIT=50 OVERWRITE=1 ./eval_deepsearchqa.sh
MODE=all LIMIT=50 ./eval_deepsearchqa.sh
```

`generate` only writes complete reports to `reports.jsonl`. `score` reuses
existing reports and writes `predictions.jsonl` plus `metrics.json` with the
DashScope OpenAI-compatible LLM autorater. `all` does both in one command and
is the default mode. Report generation uses `OPENAI_BASE_URL` / `OPENAI_API_KEY`;
autorater scoring uses `JUDGE_BASE_URL` / `JUDGE_API_KEY` / `JUDGE_MODEL`.
The judge model resolution is `--judge-model`, then `JUDGE_MODEL`, then
`deepseek-v3.2`. Omit `LIMIT` to run the full split, and use `--resume` to skip
reports that were already generated successfully. `JUDGE_API_KEY` is only
required for `MODE=score` and `MODE=all`.
See comments in `eval_deepsearchqa.sh` for every parameter.
