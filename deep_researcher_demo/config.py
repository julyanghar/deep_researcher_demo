"""Runtime configuration for the CLI workflow."""

import os
from dataclasses import dataclass


DEFAULT_JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_JUDGE_MODEL = "deepseek-v3.2"


@dataclass(slots=True)
class AppConfig:
    """Configuration shared by agents and workflow."""

    openai_base_url: str = "http://localhost:30000/v1"
    openai_api_key: str = "dummy"
    model: str = "gpt-4.1"
    supervisor_model: str | None = None
    researcher_model: str | None = None
    summary_model: str | None = None
    final_model: str | None = None
    judge_model: str | None = None
    judge_base_url: str = DEFAULT_JUDGE_BASE_URL
    judge_api_key: str | None = None
    max_iterations: int = 3
    max_followups: int = 3
    min_rounds: int = 0  # Exp A: force >= this many research rounds (0 = off)
    max_queries_per_researcher: int = 3
    max_concurrency: int = 3
    max_results: int = 5
    fetch_webpages: bool = True
    max_content_chars: int = 12000
    fetch_timeout: float = 10.0
    fetch_concurrency: int = 8
    search_provider: str = "duckduckgo"
    output: str | None = None
    # When non-empty, researcher summaries are embedded between this separator
    # in downstream prompts (supervisor decisions / final report) so an
    # LMCache-blend backend can reuse the decode-generated KV of each summary.
    # Must equal the server's LMCACHE_BLEND_SPECIAL_STR and should be an
    # atomic special token of the served model (e.g. "<|fim_pad|>" for Qwen),
    # which tokenizes identically in any surrounding context.
    kv_reuse_separator: str = ""
    # Per-role override of kv_reuse_separator (default inherits the global one).
    # Lets a single run mix reuse/prefill per role — e.g. exp-test-supervisor sets
    # final_kv_reuse_separator="" so the Writer runs full prefill while the
    # Supervisor (+Researcher, to store the summary KV) still reuse.
    supervisor_kv_reuse_separator: str = ""
    researcher_kv_reuse_separator: str = ""
    final_kv_reuse_separator: str = ""
    # Query-keyed web-search cache for reproducible multi-round search. One
    # directory per question (<dir>/q<sample_id>/), each with its own two-level
    # maps (search_cache.json {query:[urls]} + pages_index.json/pages {url:content}).
    #   off    : passthrough, live search (default, unchanged behavior)
    #   record : live search + persist query->urls and url->content
    #            (per question, deduped within it, first-write-wins)
    #   replay : resolve query->urls->content from that question's cache (no
    #            network) so re-running the question is deterministic
    # search_cache_fix_n caps replay to the first N URLs per query (0 = all);
    # search_cache_dir is the persistent root (NOT a per-run OUTPUT_DIR).
    search_cache_mode: str = "off"
    search_cache_fix_n: int = 0
    search_cache_dir: str = "eval/results/search_cache"
    # research 模式:online(联网:搜+crawl4ai抓+存本地+过滤)/ local(纯离线:读本地+过滤)/
    # off(沿用 SEARCH_CACHE 旧行为)。online→search_cache_mode=record、local→replay。
    research_mode: str = "off"
    # 缓存按 benchmark名+题目index 分类:<search_cache_dir>/<benchmark>/q<index>/。
    search_benchmark: str = ""
    # 正文抓取后端:crawl4ai(无头浏览器,默认)/ httpx(轻量兜底)。
    search_fetcher: str = "crawl4ai"
    # cache 层块级 embedding 检索(online 存块向量、local 读块向量做 top-k)。
    # online/local 默认开;它一开,relevance 的 top-k 在 cache 层做。
    cache_relevance: bool = False
    # 外层 RelevanceFilteringProvider 开关:仅 RELEVANCE_FILTER 显式开(给老 SEARCH_CACHE
    # 直连/off 路径用)。online/local 走 cache_relevance,**不**用外层(免重复 embed)。
    relevance_enabled: bool = False

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load defaults from environment variables."""
        model = os.getenv("MODEL", "gpt-4.1")
        sep = os.getenv("KV_REUSE_SEPARATOR", "")  # 全局;下面三角色默认继承、各自可 env 覆盖
        return cls(
            openai_base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:30000/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY", "dummy"),
            model=model,
            supervisor_model=os.getenv("SUPERVISOR_MODEL") or model,
            researcher_model=os.getenv("RESEARCHER_MODEL") or model,
            summary_model=os.getenv("SUMMARY_MODEL") or model,
            final_model=os.getenv("FINAL_MODEL") or model,
            judge_model=os.getenv("JUDGE_MODEL") or DEFAULT_JUDGE_MODEL,
            judge_base_url=os.getenv("JUDGE_BASE_URL", DEFAULT_JUDGE_BASE_URL),
            judge_api_key=os.getenv("JUDGE_API_KEY") or None,
            max_iterations=_env_int("MAX_ITERATIONS", 3),
            max_followups=_env_int("MAX_FOLLOWUPS", 3),
            min_rounds=_env_int("MIN_ROUNDS", 0),
            max_queries_per_researcher=_env_int("MAX_QUERIES_PER_RESEARCHER", 3),
            max_concurrency=_env_int("MAX_CONCURRENCY", 3),
            max_results=_env_int("MAX_RESULTS", 5),
            fetch_webpages=_env_bool("FETCH_WEBPAGES", True),
            max_content_chars=_env_int("MAX_CONTENT_CHARS", 12000),
            fetch_timeout=_env_float("FETCH_TIMEOUT", 10.0),
            fetch_concurrency=_env_int("FETCH_CONCURRENCY", 8),
            search_provider=os.getenv("SEARCH_PROVIDER", "duckduckgo"),
            output=os.getenv("OUTPUT") or None,
            kv_reuse_separator=sep,
            supervisor_kv_reuse_separator=os.getenv("SUPERVISOR_KV_REUSE_SEPARATOR", sep),
            researcher_kv_reuse_separator=os.getenv("RESEARCHER_KV_REUSE_SEPARATOR", sep),
            final_kv_reuse_separator=os.getenv("FINAL_KV_REUSE_SEPARATOR", sep),
            search_cache_mode=_resolve_cache_mode(),
            search_cache_fix_n=_env_int("SEARCH_CACHE_FIX_N", 0),
            search_cache_dir=os.getenv("SEARCH_CACHE_DIR", "eval/results/search_cache"),
            research_mode=(os.getenv("RESEARCH_MODE", "off") or "off").strip().lower(),
            search_benchmark=(os.getenv("SEARCH_BENCHMARK", "") or "").strip(),
            search_fetcher=(os.getenv("SEARCH_FETCHER", "crawl4ai") or "crawl4ai").strip().lower(),
            cache_relevance=_resolve_cache_relevance(),
            relevance_enabled=_resolve_relevance_enabled(),
        )

    def apply_model_override(self, model: str | None) -> None:
        """Apply a CLI model override to every role-specific model."""
        if not model:
            return
        self.model = model
        self.supervisor_model = model
        self.researcher_model = model
        self.summary_model = model
        self.final_model = model


def _resolve_cache_mode() -> str:
    """RESEARCH_MODE 优先:online→record、local→replay;否则沿用旧 SEARCH_CACHE。"""
    rm = (os.getenv("RESEARCH_MODE", "") or "").strip().lower()
    if rm == "online":
        return "record"
    if rm == "local":
        return "replay"
    return (os.getenv("SEARCH_CACHE", "off") or "off").strip().lower()


def _resolve_cache_relevance() -> bool:
    """online/local → cache 层做块级 embedding 检索;CACHE_RELEVANCE 可显式覆盖。"""
    rm = (os.getenv("RESEARCH_MODE", "") or "").strip().lower()
    return _env_bool("CACHE_RELEVANCE", rm in {"online", "local"})


def _resolve_relevance_enabled() -> bool:
    """外层过滤:online/local 走 cache 层(故默认关外层);仅 RELEVANCE_FILTER 显式开
    (给老 SEARCH_CACHE 直连/off 路径)。"""
    return _env_bool("RELEVANCE_FILTER", False)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {value!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a float, got {value!r}") from exc
