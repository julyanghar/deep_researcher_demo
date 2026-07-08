"""把 DeepResearchBench / DeepResearchGym 的 scorer 接到 demo 的 DashScope 评委。

两套 benchmark 的 LLM client 都是 OpenAI 兼容:
- DRGym:`AsyncOpenAI()` 读 env `OPENAI_API_KEY` / `OPENAI_BASE_URL`(SDK 默认读这俩 env)。
- DRBench(`utils/api.py`):读 env `OPENAI_BASE_URL` + `RACE_MODEL`/`FACT_MODEL` + `OPENAI_API_KEY`。
所以只要把这些 env 指到 demo 的 DashScope JUDGE(kimi),无需改 benchmark 源码。

另外提供"用 demo 的 search 缓存核引用"的抓取器,替代 DRGym/DRBench 的 crawl4ai
(demo 报告引的 URL 多半就在它自己的 search 缓存里 → 快、确定、免联网)。
"""
import os
import json
import glob
from pathlib import Path

DEMO_ROOT = Path(__file__).resolve().parents[2]   # /home/yilin/deep_researcher_demo


def _read_env_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def judge_settings() -> dict:
    """从 demo .env 读 JUDGE_*,返回 {api_key, base_url, model}。"""
    env = _read_env_file(DEMO_ROOT / ".env")
    return {
        "api_key": os.getenv("JUDGE_API_KEY") or env.get("JUDGE_API_KEY") or env.get("DASHSCOPE_API_KEY"),
        "base_url": os.getenv("JUDGE_BASE_URL") or env.get("JUDGE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "model": os.getenv("JUDGE_MODEL") or env.get("JUDGE_MODEL") or "kimi-k2.5",
    }


def scorer_env(extra: dict | None = None) -> dict:
    """给 scorer 子进程用的完整 env:把 OpenAI 兼容变量指到 DashScope JUDGE。"""
    j = judge_settings()
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = j["api_key"]
    env["OPENAI_BASE_URL"] = j["base_url"]
    # DRBench utils/api.py 的模型 env(用 openai backend,base_url 走 DashScope)
    env["LLM_BACKEND"] = "openai"
    env["RACE_MODEL"] = j["model"]
    env["FACT_MODEL"] = j["model"]
    if extra:
        env.update(extra)
    return env


def apply_dashscope_env() -> dict:
    """在**导入** benchmark scorer 之前调用:把 DashScope 设进当前进程 os.environ
    (DRGym 模块级 `client=AsyncOpenAI()` 在 import 时读 env)。返回 judge_settings。"""
    j = judge_settings()
    os.environ["OPENAI_API_KEY"] = j["api_key"]
    os.environ["OPENAI_BASE_URL"] = j["base_url"]
    return j


# ---------- 用 demo 的 search 缓存核引用(替代 crawl4ai) ----------
def build_cache_url_index(cache_dir: str | None = None) -> dict:
    """扫 demo search 缓存的所有 pages_index,建 url -> 正文.txt 路径 映射。"""
    cache_dir = cache_dir or os.getenv(
        "SEARCH_CACHE_DIR", str(DEMO_ROOT / "eval/results/search_cache"))
    url2path = {}
    for idx in glob.glob(f"{cache_dir}/qq*/pages_index.json"):
        base = os.path.dirname(idx)
        try:
            d = json.load(open(idx))
        except Exception:
            continue
        for url, meta in d.items():
            fn = meta.get("file") if isinstance(meta, dict) else None
            if not fn:
                continue
            for cand in (os.path.join(base, fn), os.path.join(base, "pages", fn)):
                if os.path.exists(cand):
                    # 取较长的那份(更可能是真正文)
                    if url not in url2path or os.path.getsize(cand) > os.path.getsize(url2path[url]):
                        url2path[url] = cand
                    break
    return url2path


def cache_fetch(url: str, url2path: dict) -> str | None:
    """从缓存取某 url 的网页正文;没有则 None(调用方可退 crawl4ai)。"""
    p = url2path.get(url)
    if p and os.path.exists(p):
        try:
            return open(p, encoding="utf-8").read()
        except Exception:
            return None
    return None
