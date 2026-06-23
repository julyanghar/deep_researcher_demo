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

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Load defaults from environment variables."""
        model = os.getenv("MODEL", "gpt-4.1")
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
            kv_reuse_separator=os.getenv("KV_REUSE_SEPARATOR", ""),
            search_cache_mode=(os.getenv("SEARCH_CACHE", "off") or "off").strip().lower(),
            search_cache_fix_n=_env_int("SEARCH_CACHE_FIX_N", 0),
            search_cache_dir=os.getenv("SEARCH_CACHE_DIR", "eval/results/search_cache"),
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
