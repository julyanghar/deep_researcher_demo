import json
import asyncio
from argparse import Namespace

import pytest

import eval.run_deepsearchqa as eval_module
from deep_researcher_demo.config import AppConfig, DEFAULT_JUDGE_BASE_URL, DEFAULT_JUDGE_MODEL
from eval.run_deepsearchqa import (
    EvalRunner,
    async_main,
    build_config,
    build_eval_prompt,
    build_workflow_stats,
    load_selected_report_records,
    prepare_outputs,
    read_completed_ids,
    read_jsonl,
    select_examples,
    validate_judge_config,
)


EXAMPLES = [{"sample_id": index, "problem": str(index)} for index in range(5)]
DATASET_ROWS = [
    {
        "problem": "Name the stub answer.",
        "problem_category": "Stub",
        "answer": "Stub Answer",
        "answer_type": "Single Answer",
    }
]


def test_limit_selects_requested_count():
    assert [item["sample_id"] for item in select_examples(EXAMPLES, ids=None, start=0, limit=2)] == [0, 1]


def test_start_and_limit_select_window():
    assert [item["sample_id"] for item in select_examples(EXAMPLES, ids=None, start=2, limit=2)] == [2, 3]


def test_ids_override_start_and_limit():
    assert [item["sample_id"] for item in select_examples(EXAMPLES, ids="4,1", start=0, limit=2)] == [4, 1]


def test_resume_reads_only_successful_ids(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps({"sample_id": 1, "error": None}) + "\n"
        + json.dumps({"sample_id": 2, "error": "boom"}) + "\n",
        encoding="utf-8",
    )
    assert read_completed_ids(predictions) == {1}


def test_existing_outputs_require_resume_or_overwrite(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    failures = tmp_path / "failures.jsonl"
    predictions.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        prepare_outputs(
            output_dir=tmp_path,
            predictions_path=predictions,
            metrics_path=metrics,
            failures_path=failures,
            resume=False,
            overwrite=False,
        )


def test_overwrite_removes_existing_outputs(tmp_path):
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    failures = tmp_path / "failures.jsonl"
    predictions.write_text("{}", encoding="utf-8")

    prepare_outputs(
        output_dir=tmp_path,
        predictions_path=predictions,
        metrics_path=metrics,
        failures_path=failures,
        resume=False,
        overwrite=True,
    )

    assert not predictions.exists()


def test_score_overwrite_keeps_reports_file(tmp_path):
    reports = tmp_path / "reports.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    failures = tmp_path / "failures.jsonl"
    for path in [reports, predictions, metrics, failures]:
        path.write_text("{}", encoding="utf-8")

    prepare_outputs(
        output_dir=tmp_path,
        reports_path=reports,
        predictions_path=predictions,
        metrics_path=metrics,
        failures_path=failures,
        resume=False,
        overwrite=True,
        mode="score",
    )

    assert reports.exists()
    assert not predictions.exists()
    assert not metrics.exists()
    assert not failures.exists()


def test_generate_overwrite_keeps_scoring_outputs(tmp_path):
    reports = tmp_path / "reports.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    failures = tmp_path / "failures.jsonl"
    for path in [reports, predictions, metrics, failures]:
        path.write_text("{}", encoding="utf-8")

    prepare_outputs(
        output_dir=tmp_path,
        reports_path=reports,
        predictions_path=predictions,
        metrics_path=metrics,
        failures_path=failures,
        resume=False,
        overwrite=True,
        mode="generate",
    )

    assert not reports.exists()
    assert predictions.exists()
    assert metrics.exists()
    assert failures.exists()


def test_load_selected_report_records_requires_existing_reports(tmp_path):
    with pytest.raises(SystemExit, match="No reports.jsonl"):
        load_selected_report_records(tmp_path / "reports.jsonl", selected_ids=[1])


def test_load_selected_report_records_requires_selected_ids(tmp_path):
    reports = tmp_path / "reports.jsonl"
    reports.write_text(json.dumps({"sample_id": 1, "error": None}) + "\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Missing reports"):
        load_selected_report_records(reports, selected_ids=[1, 2])


def test_mode_generate_only_writes_reports(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    install_eval_stubs(monkeypatch)

    asyncio.run(
        async_main(
            [
                "--env-file",
                str(env_file),
                "--mode",
                "generate",
                "--limit",
                "1",
                "--output-dir",
                str(tmp_path),
                "--overwrite",
                "--quiet",
            ]
        )
    )

    assert (tmp_path / "reports.jsonl").exists()
    assert (tmp_path / "workflow_traces.jsonl").exists()
    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "metrics.json").exists()
    report_record = read_jsonl(tmp_path / "reports.jsonl")[0]
    trace_record = read_jsonl(tmp_path / "workflow_traces.jsonl")[0]
    assert "workflow_events" not in report_record
    assert "workflow_stats" in report_record
    assert "workflow_events" in trace_record
    assert "workflow_stats" in trace_record


def test_mode_score_uses_existing_reports(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    install_eval_stubs(monkeypatch)

    asyncio.run(
        async_main(
            [
                "--env-file",
                str(env_file),
                "--mode",
                "generate",
                "--limit",
                "1",
                "--output-dir",
                str(tmp_path),
                "--overwrite",
                "--quiet",
            ]
        )
    )
    asyncio.run(
        async_main(
            [
                "--env-file",
                str(env_file),
                "--mode",
                "score",
                "--limit",
                "1",
                "--output-dir",
                str(tmp_path),
                "--overwrite",
                "--quiet",
            ]
        )
    )

    assert (tmp_path / "reports.jsonl").exists()
    assert (tmp_path / "workflow_traces.jsonl").exists()
    assert (tmp_path / "predictions.jsonl").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "failures.jsonl").exists()


def test_mode_all_writes_reports_and_metrics(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("", encoding="utf-8")
    install_eval_stubs(monkeypatch)

    asyncio.run(
        async_main(
            [
                "--env-file",
                str(env_file),
                "--mode",
                "all",
                "--limit",
                "1",
                "--output-dir",
                str(tmp_path),
                "--overwrite",
                "--quiet",
            ]
        )
    )

    assert (tmp_path / "reports.jsonl").exists()
    assert (tmp_path / "predictions.jsonl").exists()
    assert (tmp_path / "metrics.json").exists()


def test_eval_prompt_does_not_include_gold_metadata():
    prompt = build_eval_prompt("Question text")
    assert "Question text" in prompt
    assert "answer_type" not in prompt
    assert "gold" not in prompt.lower()


def test_workflow_stats_preserve_queries_from_events():
    events = [
        {
            "step": "initial_questions",
            "message": "2 questions",
            "data": {"research_questions": ["q1", "q2"]},
        },
        {
            "step": "iteration_start",
            "message": "iteration 1",
            "data": {"question_count": 2},
        },
        {"step": "researcher_start", "message": "r1", "data": {"question": "q1"}},
        {
            "step": "queries_planned",
            "message": "2 queries",
            "data": {"question": "q1", "queries": ["query a", "query b"]},
        },
        {
            "step": "subquery_search_complete",
            "message": "5 results",
            "data": {"query": "query a", "result_count": 5, "backup_url_count": 5},
        },
        {
            "step": "supervisor_decision",
            "message": "complete",
            "data": {"status": "complete", "reason": "done", "followup_questions": []},
        },
        {"step": "completed", "message": "done", "data": {}},
    ]

    stats = build_workflow_stats(events)

    assert stats["num_events"] == len(events)
    assert stats["num_iterations"] == 1
    assert stats["initial_question_count"] == 2
    assert stats["researcher_count"] == 1
    assert stats["queries_per_researcher"] == [2]
    assert stats["queries_by_researcher"] == [["query a", "query b"]]
    assert stats["branches_by_iteration"] == [2]
    assert stats["search_result_count"] == 5
    assert stats["backup_url_count"] == 5
    assert stats["completed"] is True


def test_judge_model_priority(monkeypatch):
    monkeypatch.setenv("MODEL", "base-model")
    monkeypatch.setenv("JUDGE_MODEL", "env-judge")
    args = Namespace(
        model=None,
        base_url=None,
        judge_base_url=None,
        judge_api_key=None,
        max_iterations=None,
        max_followups=None,
        max_queries_per_researcher=None,
        max_concurrency=None,
        max_results=None,
        search_provider=None,
        judge_model="cli-judge",
    )
    assert build_config(args).judge_model == "cli-judge"

    args.judge_model = None
    assert build_config(args).judge_model == "env-judge"

    monkeypatch.delenv("JUDGE_MODEL")
    assert build_config(args).judge_model == DEFAULT_JUDGE_MODEL


def test_judge_base_url_priority(monkeypatch):
    monkeypatch.setenv("JUDGE_BASE_URL", "https://env-judge.example/v1")
    args = Namespace(
        model=None,
        base_url=None,
        judge_base_url="https://cli-judge.example/v1",
        judge_api_key=None,
        judge_model=None,
        max_iterations=None,
        max_followups=None,
        max_queries_per_researcher=None,
        max_concurrency=None,
        max_results=None,
        search_provider=None,
    )

    assert build_config(args).judge_base_url == "https://cli-judge.example/v1"

    args.judge_base_url = None
    assert build_config(args).judge_base_url == "https://env-judge.example/v1"

    monkeypatch.delenv("JUDGE_BASE_URL")
    assert build_config(args).judge_base_url == DEFAULT_JUDGE_BASE_URL


def test_eval_build_config_overrides_max_queries_per_researcher(monkeypatch):
    monkeypatch.setenv("MAX_QUERIES_PER_RESEARCHER", "3")
    args = Namespace(
        model=None,
        base_url=None,
        judge_base_url=None,
        judge_api_key=None,
        judge_model=None,
        max_iterations=None,
        max_followups=None,
        max_queries_per_researcher=7,
        max_concurrency=None,
        max_results=None,
        search_provider=None,
    )

    assert build_config(args).max_queries_per_researcher == 7


def test_judge_key_required_only_for_scoring_modes():
    config = AppConfig(judge_api_key=None)

    validate_judge_config(config, mode="generate")
    with pytest.raises(SystemExit, match="JUDGE_API_KEY is required"):
        validate_judge_config(config, mode="score")
    with pytest.raises(SystemExit, match="JUDGE_API_KEY is required"):
        validate_judge_config(config, mode="all")


def test_eval_runner_uses_separate_research_and_judge_clients(monkeypatch):
    created_clients = []

    class RecordingClient:
        def __init__(self, *, base_url, api_key):
            self.base_url = base_url
            self.api_key = api_key
            created_clients.append(self)

    monkeypatch.setattr(eval_module, "OpenAICompatibleClient", RecordingClient)
    config = AppConfig(
        openai_base_url="http://research.local/v1",
        openai_api_key="research-key",
        judge_base_url="https://judge.example/v1",
        judge_api_key="judge-key",
    )
    runner = EvalRunner(config=config, quiet=True)

    research_client = runner.build_llm()
    judge_client = runner.build_judge_llm()

    assert research_client.base_url == "http://research.local/v1"
    assert research_client.api_key == "research-key"
    assert judge_client.base_url == "https://judge.example/v1"
    assert judge_client.api_key == "judge-key"
    assert created_clients == [research_client, judge_client]


def install_eval_stubs(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "test-judge-key")
    monkeypatch.setattr(eval_module, "load_dataset_rows", lambda: DATASET_ROWS)
    monkeypatch.setattr(eval_module.EvalRunner, "generate_report", fake_generate_report)
    monkeypatch.setattr(eval_module.EvalRunner, "rate_report_record", fake_rate_report_record)


async def fake_generate_report(self, example):
    return {
        **example,
        "input_problem": eval_module.build_eval_prompt(example["problem"]),
        "final_report": "# Stub Report\n\n## Final Answer\nStub Answer",
        "final_answer": "Stub Answer",
        "latency_seconds": 0.001,
        "error": None,
    }


async def fake_rate_report_record(self, report_record):
    return {
        "original_index": report_record["sample_id"],
        "example_id": str(report_record["sample_id"]),
        "query": report_record["problem"],
        "response": report_record["final_report"],
        "category_type": report_record["problem_category"],
        "expected_correct_answer": report_record["answer"],
        "sample_id": report_record["sample_id"],
        "answer_type": report_record["answer_type"],
        "final_answer": report_record["final_answer"],
        "local_scores": {
            "answer_type": report_record["answer_type"],
            "exact": True,
            "gold_substring": True,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
        },
        "answer_correctness_explanation": "Stub rating is correct.",
        "expected_correct_answer_list": [report_record["answer"]],
        "response_wrong_answers_list": None,
        "grader_ratings_list": [True],
        "empty_model_response": False,
        "empty_auto_rater_response": False,
        "invalid_auto_rater_response": False,
        "rating_response": "{}",
        "rating_prompt": "",
        "error_message": None,
        "problem": report_record["problem"],
        "problem_category": report_record["problem_category"],
        "answer": report_record["answer"],
        "final_report": report_record["final_report"],
        "autorater_error": None,
        "error": None,
    }
