"""Command line interface for the simplified deep researcher."""

import argparse
import asyncio
import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv

from deep_researcher_demo.agents import FinalWriter, Researcher, Supervisor
from deep_researcher_demo.config import AppConfig
from deep_researcher_demo.llm import OpenAICompatibleClient
from deep_researcher_demo.progress import ConsoleProgressReporter, NullProgressReporter
from deep_researcher_demo.search import create_search_provider, wrap_with_cache
from deep_researcher_demo.workflow import DeepResearchWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simplified deep researcher workflow.")
    parser.add_argument("question", nargs="+", help="Research question to investigate.")
    parser.add_argument("--env-file", help="Environment file to load before reading configuration.")
    parser.add_argument("--output", help="Optional Markdown output file.")
    parser.add_argument("--model", help="Override all model roles with one model name.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--max-followups", type=int)
    parser.add_argument("--max-queries-per-researcher", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-results", type=int)
    parser.add_argument(
        "--search-provider",
        choices=["duckduckgo", "tavily"],
        default=None,
        help="Search provider to use. Tavily is reserved but not implemented yet.",
    )
    parser.add_argument("--quiet", action="store_true", help="Only print the final report.")
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env(args.env_file)
    question = " ".join(args.question).strip()

    config = AppConfig.from_env()
    config.apply_model_override(args.model)
    if args.base_url:
        config.openai_base_url = args.base_url
    if args.output:
        config.output = args.output
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.max_followups is not None:
        config.max_followups = args.max_followups
    if args.max_queries_per_researcher is not None:
        config.max_queries_per_researcher = args.max_queries_per_researcher
    if args.max_concurrency is not None:
        config.max_concurrency = args.max_concurrency
    if args.max_results is not None:
        config.max_results = args.max_results
    if args.search_provider:
        config.search_provider = args.search_provider

    llm = OpenAICompatibleClient(
        base_url=config.openai_base_url,
        api_key=config.openai_api_key,
    )
    search_provider = create_search_provider(
        config.search_provider,
        fetch_webpages=config.fetch_webpages,
        max_content_chars=config.max_content_chars,
        fetch_timeout=config.fetch_timeout,
        fetch_concurrency=config.fetch_concurrency,
    )
    # Wrap with the two-level (query->urls, url->content) per-question cache so a
    # trajectory can be recorded once and then replayed frozen — Exp A needs the
    # exact same retrieval across the truth/swap teacher-forcing passes. Off by
    # default (SEARCH_CACHE=off). The per-question dir id is SEARCH_CACHE_SAMPLE_ID
    # if given, else a short hash of the question so distinct questions don't mix.
    if config.search_cache_mode != "off":
        sample_id = os.getenv("SEARCH_CACHE_SAMPLE_ID") or hashlib.sha1(
            question.encode("utf-8")
        ).hexdigest()[:12]
        search_provider = wrap_with_cache(
            search_provider,
            mode=config.search_cache_mode,
            cache_dir=config.search_cache_dir,
            fix_n=config.search_cache_fix_n,
            sample_id=sample_id,
        )

    supervisor_model = config.supervisor_model or config.model
    researcher_model = config.researcher_model or config.model
    summary_model = config.summary_model or config.model
    final_model = config.final_model or config.model
    reporter = NullProgressReporter() if args.quiet else ConsoleProgressReporter()

    workflow = DeepResearchWorkflow(
        supervisor=Supervisor(llm, supervisor_model, config.kv_reuse_separator),
        researcher=Researcher(
            llm, researcher_model, summary_model,
            kv_reuse_separator=config.kv_reuse_separator,
        ),
        final_writer=FinalWriter(llm, final_model, config.kv_reuse_separator),
        search_provider=search_provider,
        max_iterations=config.max_iterations,
        max_followups=config.max_followups,
        min_rounds=config.min_rounds,
        max_queries_per_researcher=config.max_queries_per_researcher,
        max_concurrency=config.max_concurrency,
        max_results=config.max_results,
        reporter=reporter,
        output_path=config.output,
    )
    result = await workflow.run(question)

    if config.output:
        output_path = Path(config.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.final_report, encoding="utf-8")
    print(result.final_report)
    return 0


def load_env(env_file: str | None) -> None:
    """Load env vars in code, defaulting to the repo-level `.env` file."""
    path = Path(env_file) if env_file else default_env_path()
    if not path.exists():
        if env_file:
            raise SystemExit(f"Env file not found: {env_file}")
        return
    load_dotenv(path)


def default_env_path() -> Path:
    """Return the default env file path for this repository."""
    return Path(__file__).resolve().parents[1] / ".env"


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))
