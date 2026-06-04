"""Run DeepSearchQA evaluation for deep_researcher_demo."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from deep_researcher_demo.agents import FinalWriter, Researcher, Supervisor
from deep_researcher_demo.cli import load_env
from deep_researcher_demo.config import AppConfig
from deep_researcher_demo.llm import OpenAICompatibleClient
from deep_researcher_demo.progress import ConsoleProgressReporter, NullProgressReporter
from deep_researcher_demo.search import create_search_provider
from deep_researcher_demo.workflow import DeepResearchWorkflow
from eval.judge import rate_report
from eval.scoring import (
    aggregate_metrics,
    aggregate_starter_metrics,
    build_item_rating_from_report,
    extract_final_answer,
    reduce_autorater_response,
    score_answer,
)


DATASET_NAME = "google/deepsearchqa"
DATASET_SPLIT = "eval"
DEFAULT_OUTPUT_DIR = Path("eval/results")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate deep_researcher_demo on DeepSearchQA.")
    parser.add_argument("--env-file", help="Environment file to load before reading configuration.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, help="Number of examples to evaluate. Defaults to all.")
    parser.add_argument("--start", type=int, default=0, help="Start index in the selected dataset.")
    parser.add_argument("--ids", help="Comma-separated dataset indices to evaluate. Overrides start/limit.")
    parser.add_argument("--category", help="Filter by problem_category.")
    parser.add_argument(
        "--mode",
        choices=["generate", "score", "all"],
        default="all",
        help="generate only reports, score existing reports, or run both phases.",
    )
    parser.add_argument("--sample-concurrency", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Skip successful rows in reports.jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Delete previous output before running.")
    parser.add_argument("--quiet", action="store_true", help="Hide researcher progress; benchmark progress still prints.")
    parser.add_argument("--judge-model", help="OpenAI-compatible judge model. Defaults to JUDGE_MODEL, then deepseek-v3.2.")
    parser.add_argument("--judge-base-url", help="OpenAI-compatible base URL for the judge model.")
    parser.add_argument("--judge-api-key", help="API key for the judge model endpoint.")
    parser.add_argument("--model", help="Override all researcher model roles.")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL.")
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--max-followups", type=int)
    parser.add_argument("--max-queries-per-researcher", type=int)
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--max-results", type=int)
    parser.add_argument("--search-provider", choices=["duckduckgo", "tavily"], default=None)
    return parser


async def async_main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite are mutually exclusive.")

    load_env(args.env_file)
    config = build_config(args)
    validate_judge_config(config, mode=args.mode)
    output_dir = Path(args.output_dir)
    reports_path = output_dir / "reports.jsonl"
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    failures_path = output_dir / "failures.jsonl"

    prepare_outputs(
        output_dir=output_dir,
        reports_path=reports_path,
        predictions_path=predictions_path,
        metrics_path=metrics_path,
        failures_path=failures_path,
        resume=args.resume,
        overwrite=args.overwrite,
        mode=args.mode,
    )

    examples = load_examples(args)
    print(f"[dataset] loaded {len(examples)} examples", flush=True)

    runner = EvalRunner(config=config, quiet=args.quiet)
    if args.mode in {"generate", "all"}:
        completed_ids = read_completed_ids(reports_path) if args.resume else set()
        if args.resume:
            print(f"[resume] skipping {len(completed_ids)} completed reports", flush=True)
        examples_to_generate = [example for example in examples if example["sample_id"] not in completed_ids]
        await run_generation_phase(
            examples_to_generate,
            runner=runner,
            reports_path=reports_path,
            sample_concurrency=args.sample_concurrency,
            total_count=len(examples),
        )
        if args.mode == "generate":
            print(f"[completed] generated reports in {reports_path}", flush=True)
            return 0

    selected_ids = [example["sample_id"] for example in examples]
    report_records = load_selected_report_records(reports_path, selected_ids=selected_ids)
    prediction_records = await run_scoring_phase(
        report_records,
        runner=runner,
        predictions_path=predictions_path,
        failures_path=failures_path,
        sample_concurrency=args.sample_concurrency,
    )

    metrics = aggregate_starter_metrics(prediction_records)
    metrics["local_metrics"] = aggregate_metrics(prediction_records)
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        "[metrics] "
        f"precision={metrics['precision'] or '0.00%'} "
        f"recall={metrics['recall'] or '0.00%'} "
        f"f1={metrics['f1_score'] or '0.00%'} "
        f"invalid={metrics['num_invalid_auto_rater_response']} "
        f"empty={metrics['num_empty_auto_rater_response']}",
        flush=True,
    )
    return 0


def build_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.from_env()
    config.apply_model_override(args.model)
    if args.base_url:
        config.openai_base_url = args.base_url
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
    if args.judge_base_url:
        config.judge_base_url = args.judge_base_url
    if args.judge_api_key:
        config.judge_api_key = args.judge_api_key
    config.judge_model = args.judge_model or config.judge_model
    return config


def validate_judge_config(config: AppConfig, *, mode: str) -> None:
    """Require a judge API key only for modes that call the autorater."""
    if mode == "generate":
        return
    if not config.judge_api_key:
        raise SystemExit(
            "JUDGE_API_KEY is required for DeepSearchQA scoring. "
            "Set JUDGE_API_KEY in .env/env or pass --judge-api-key. "
            "Use --mode generate if you only want to create reports."
        )


def load_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    dataset = load_dataset_rows()
    examples = [
        {
            "sample_id": index,
            "problem": str(row["problem"]),
            "problem_category": str(row.get("problem_category", "")),
            "answer": str(row["answer"]),
            "answer_type": str(row["answer_type"]),
        }
        for index, row in enumerate(dataset)
    ]
    if args.category:
        examples = [example for example in examples if example["problem_category"] == args.category]
    return select_examples(examples, ids=args.ids, start=args.start, limit=args.limit)


def load_dataset_rows():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The `datasets` package is required for DeepSearchQA. Install with `pip install -e '.[eval]'."
        ) from exc
    return load_dataset(DATASET_NAME, split=DATASET_SPLIT)


def select_examples(
    examples: list[dict[str, Any]],
    *,
    ids: str | None,
    start: int,
    limit: int | None,
) -> list[dict[str, Any]]:
    if ids:
        selected_ids = [int(value.strip()) for value in ids.split(",") if value.strip()]
        examples_by_id = {example["sample_id"]: example for example in examples}
        return [examples_by_id[sample_id] for sample_id in selected_ids if sample_id in examples_by_id]
    selected = examples[max(0, start) :]
    if limit is not None:
        selected = selected[: max(0, limit)]
    return selected


class EvalRunner:
    def __init__(self, *, config: AppConfig, quiet: bool) -> None:
        self.config = config
        self.quiet = quiet

    async def generate_report(self, example: dict[str, Any]) -> dict[str, Any]:
        start_time = time.perf_counter()
        started_at = utc_now_iso()
        try:
            workflow = self.build_workflow()
            eval_prompt = build_eval_prompt(example["problem"])
            result = await workflow.run(eval_prompt)
            final_report = result.final_report
            final_answer = extract_final_answer(final_report)
            completed_at = utc_now_iso()
            latency_seconds = round(time.perf_counter() - start_time, 3)
            return {
                **example,
                "input_problem": eval_prompt,
                "final_report": final_report,
                "final_answer": final_answer,
                "report_generation_started_at": started_at,
                "report_generation_completed_at": completed_at,
                "report_generation_latency_seconds": latency_seconds,
                "latency_seconds": latency_seconds,
                "error": None,
            }
        except Exception as exc:
            completed_at = utc_now_iso()
            latency_seconds = round(time.perf_counter() - start_time, 3)
            return {
                **example,
                "input_problem": build_eval_prompt(example["problem"]),
                "final_report": "",
                "final_answer": "",
                "report_generation_started_at": started_at,
                "report_generation_completed_at": completed_at,
                "report_generation_latency_seconds": latency_seconds,
                "latency_seconds": latency_seconds,
                "error": str(exc),
                "traceback": traceback.format_exc(limit=5),
            }

    async def rate_report_record(self, report_record: dict[str, Any]) -> dict[str, Any]:
        local_scores = score_answer(
            report_record.get("final_answer", ""),
            report_record.get("answer", ""),
            report_record.get("answer_type", ""),
        )
        rating_shell = build_item_rating_from_report({**report_record, "local_scores": local_scores})
        judge_result = await rate_report(
            llm=self.build_judge_llm(),
            model=self.config.judge_model or self.config.model,
            problem=report_record.get("problem", ""),
            answer_type=report_record.get("answer_type", ""),
            answer=report_record.get("answer", ""),
            response=report_record.get("final_report", ""),
        )
        item_rating = reduce_autorater_response(
            rating_shell,
            grader_llm_response_text=judge_result["rating_response"],
            grader_llm_prompt_text=judge_result["rating_prompt"],
            rating_error=judge_result.get("rating_error"),
        )
        record = item_rating.to_dict()
        record.update(
            {
                "sample_id": report_record.get("sample_id"),
                "problem": report_record.get("problem", ""),
                "problem_category": report_record.get("problem_category", ""),
                "answer": report_record.get("answer", ""),
                "answer_type": report_record.get("answer_type", ""),
                "final_report": report_record.get("final_report", ""),
                "final_answer": report_record.get("final_answer", ""),
                "report_generation_started_at": report_record.get("report_generation_started_at"),
                "report_generation_completed_at": report_record.get("report_generation_completed_at"),
                "report_generation_latency_seconds": report_record.get("report_generation_latency_seconds"),
                "latency_seconds": report_record.get("latency_seconds"),
                "local_scores": local_scores,
                "autorater_error": item_rating.error_message,
                "error": None,
            }
        )
        return record

    def build_workflow(self) -> DeepResearchWorkflow:
        llm = self.build_llm()
        search_provider = create_search_provider(
            self.config.search_provider,
            fetch_webpages=self.config.fetch_webpages,
            max_content_chars=self.config.max_content_chars,
            fetch_timeout=self.config.fetch_timeout,
            fetch_concurrency=self.config.fetch_concurrency,
        )
        reporter = NullProgressReporter() if self.quiet else ConsoleProgressReporter()
        supervisor_model = self.config.supervisor_model or self.config.model
        researcher_model = self.config.researcher_model or self.config.model
        summary_model = self.config.summary_model or self.config.model
        final_model = self.config.final_model or self.config.model
        return DeepResearchWorkflow(
            supervisor=Supervisor(llm, supervisor_model),
            researcher=Researcher(llm, researcher_model, summary_model),
            final_writer=FinalWriter(llm, final_model),
            search_provider=search_provider,
            max_iterations=self.config.max_iterations,
            max_followups=self.config.max_followups,
            max_queries_per_researcher=self.config.max_queries_per_researcher,
            max_concurrency=self.config.max_concurrency,
            max_results=self.config.max_results,
            reporter=reporter,
            output_path=None,
        )

    def build_llm(self):
        return OpenAICompatibleClient(
            base_url=self.config.openai_base_url,
            api_key=self.config.openai_api_key,
        )

    def build_judge_llm(self):
        return OpenAICompatibleClient(
            base_url=self.config.judge_base_url,
            api_key=self.config.judge_api_key,
        )


def build_eval_prompt(problem: str) -> str:
    return (
        f"{problem}\n\n"
        "Evaluation output requirement: End your report with a Markdown section named exactly "
        "`## Final Answer`. In that section, provide only the concise final answer. "
        "Do not mention benchmark metadata."
    )


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp for eval records."""
    return datetime.now(timezone.utc).isoformat()


async def run_generation_phase(
    examples: list[dict[str, Any]],
    *,
    runner: EvalRunner,
    reports_path: Path,
    sample_concurrency: int,
    total_count: int,
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(max(1, sample_concurrency))
    completed_count = 0
    records: list[dict[str, Any]] = []
    if not examples:
        print("[generation] no new reports to generate", flush=True)
        return records
    print(f"[generation] generating {len(examples)} reports", flush=True)

    async def run_with_semaphore(position: int, example: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            print(
                f"[sample] {position}/{total_count} "
                f"sample_id={example['sample_id']} category={example['problem_category']}",
                flush=True,
            )
            return await runner.generate_report(example)

    tasks = [
        asyncio.create_task(run_with_semaphore(index + 1, example))
        for index, example in enumerate(examples)
    ]
    for task in asyncio.as_completed(tasks):
        record = await task
        append_jsonl(reports_path, record)
        records.append(record)
        completed_count += 1
        print(
            f"[report_complete] {completed_count}/{len(examples)} "
            f"sample_id={record['sample_id']} "
            f"latency={record.get('latency_seconds')}s "
            f"error={bool(record.get('error'))}",
            flush=True,
        )
    return records


async def run_scoring_phase(
    report_records: list[dict[str, Any]],
    *,
    runner: EvalRunner,
    predictions_path: Path,
    failures_path: Path,
    sample_concurrency: int,
) -> list[dict[str, Any]]:
    reset_file(predictions_path)
    reset_file(failures_path)

    generation_failures = [record for record in report_records if record.get("error")]
    for record in generation_failures:
        append_jsonl(failures_path, {"phase": "generation", **record})

    successful_reports = [record for record in report_records if not record.get("error")]
    if not successful_reports:
        print("[scoring] no successful reports to score", flush=True)
        return []

    print(f"[scoring] rating {len(successful_reports)} reports", flush=True)
    semaphore = asyncio.Semaphore(max(1, sample_concurrency))
    completed_count = 0
    prediction_records: list[dict[str, Any]] = []

    async def rate_with_semaphore(position: int, report_record: dict[str, Any]) -> dict[str, Any]:
        async with semaphore:
            print(
                f"[rating] {position}/{len(successful_reports)} "
                f"sample_id={report_record['sample_id']}",
                flush=True,
            )
            return await runner.rate_report_record(report_record)

    tasks = [
        asyncio.create_task(rate_with_semaphore(index + 1, report_record))
        for index, report_record in enumerate(successful_reports)
    ]
    for task in asyncio.as_completed(tasks):
        record = await task
        append_jsonl(predictions_path, record)
        if record.get("autorater_error"):
            append_jsonl(failures_path, {"phase": "autorater", **record})
        prediction_records.append(record)
        completed_count += 1
        print(
            f"[rating_complete] {completed_count}/{len(successful_reports)} "
            f"sample_id={record['sample_id']} "
            f"invalid={record.get('invalid_auto_rater_response')} "
            f"empty={record.get('empty_auto_rater_response')}",
            flush=True,
        )
    return prediction_records


def prepare_outputs(
    *,
    output_dir: Path,
    predictions_path: Path,
    metrics_path: Path,
    failures_path: Path,
    resume: bool,
    overwrite: bool,
    mode: str = "all",
    reports_path: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if mode == "generate":
        output_paths = [path for path in [reports_path] if path is not None]
    elif mode == "score":
        output_paths = [predictions_path, metrics_path, failures_path]
    else:
        output_paths = [path for path in [reports_path, predictions_path, metrics_path, failures_path] if path is not None]
    existing = [path for path in output_paths if path.exists()]
    if overwrite:
        for path in existing:
            path.unlink()
        return
    if existing and not resume:
        raise SystemExit(
            f"Output files already exist in {output_dir}. Use --resume or --overwrite."
        )


def read_completed_ids(path: Path) -> set[int]:
    completed = set()
    for record in read_jsonl(path):
        if not record.get("error"):
            completed.add(int(record["sample_id"]))
    return completed


def latest_records_for_ids(records: list[dict[str, Any]], *, selected_ids: list[int]) -> list[dict[str, Any]]:
    latest: dict[int, dict[str, Any]] = {}
    selected_set = set(selected_ids)
    for record in records:
        sample_id = int(record["sample_id"])
        if sample_id in selected_set:
            latest[sample_id] = record
    return [latest[sample_id] for sample_id in selected_ids if sample_id in latest]


def load_selected_report_records(reports_path: Path, *, selected_ids: list[int]) -> list[dict[str, Any]]:
    """Load latest report records for selected ids, failing if any are missing."""
    if not reports_path.exists():
        raise SystemExit(f"No reports.jsonl found at {reports_path}. Run with --mode generate first.")
    report_records = latest_records_for_ids(read_jsonl(reports_path), selected_ids=selected_ids)
    present_ids = {int(record["sample_id"]) for record in report_records}
    missing_ids = [sample_id for sample_id in selected_ids if sample_id not in present_ids]
    if missing_ids:
        preview = ", ".join(str(sample_id) for sample_id in missing_ids[:10])
        suffix = "..." if len(missing_ids) > 10 else ""
        raise SystemExit(
            f"Missing reports for sample ids: {preview}{suffix}. Run with --mode generate first."
        )
    return report_records


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_file(path: Path) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def main(argv: list[str] | None = None) -> None:
    raise SystemExit(asyncio.run(async_main(argv)))


if __name__ == "__main__":
    main()
