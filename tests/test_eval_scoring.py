from eval.deepresearchqa.judge import build_grader_prompt
from eval.deepresearchqa.scoring import (
    ItemRating,
    aggregate_ratings,
    calculate_metric,
    extract_final_answer,
    get_excessive_answers,
    parse_json_response,
    reduce_autorater_response,
    score_answer,
    split_set_answer,
)


def test_extract_final_answer_section():
    report = "# Report\n\n## Findings\nText\n\n## Final Answer\nAustria, Switzerland\n\n## Sources\nx"
    assert extract_final_answer(report) == "Austria, Switzerland"


def test_score_single_answer_exact_and_substring():
    score = score_answer("The answer is New Zealand.", "New Zealand", "Single Answer")
    assert score["exact"] is False
    assert score["gold_substring"] is True


def test_score_set_answer_f1():
    score = score_answer("Austria, Switzerland, Singapore", "Austria, Switzerland", "Set Answer")
    assert score["precision"] == 2 / 3
    assert score["recall"] == 1.0
    assert round(score["f1"], 3) == 0.8


def test_split_set_answer_removes_duplicates():
    assert split_set_answer("France, France; Italy\n- Romania") == ["france", "italy", "romania"]


def test_starter_prompt_includes_full_report_fields():
    prompt = build_grader_prompt(
        problem="User question?",
        answer_type="Set Answer",
        answer="A, B",
        response="# Report\n\n## Final Answer\nA",
    )

    assert "<prompt>\nUser question?\n</prompt>" in prompt
    assert "Prompt Type: Set Answer" in prompt
    assert "<answer>\nA, B\n</answer>" in prompt
    assert "<response>\n# Report\n\n## Final Answer\nA\n</response>" in prompt


def test_parse_fenced_autorater_json():
    parsed = parse_json_response(
        """```json
        {"Answer Correctness": {"Explanation": "ok", "Correctness Details": {"A": true}}}
        ```"""
    )

    assert parsed["Answer Correctness"]["Correctness Details"] == {"A": True}


def test_missing_excessive_answers_defaults_to_empty_list():
    parsed = {"Answer Correctness": {"Correctness Details": {"A": True}}}
    assert get_excessive_answers(parsed) == []


def test_reduce_autorater_response_marks_missing_explanation_invalid():
    item = reduce_autorater_response(
        ItemRating(example_id="1", query="q", response="answer"),
        grader_llm_response_text='{"Answer Correctness": {"Correctness Details": {"A": true}, "Excessive Answers": []}}',
        grader_llm_prompt_text="prompt",
    )

    assert item.invalid_auto_rater_response is True
    assert "Explanation" in item.error_message


def test_starter_metric_formula():
    assert calculate_metric(true_positives=2, false_positives=1, false_negatives=0) == {
        "precision": 2 / 3,
        "recall": 1.0,
        "f1_score": 0.8,
    }


def test_project_rating_aggregation_matches_starter_code_shape():
    ratings = [
        ItemRating(
            example_id="1",
            query="q1",
            response="r1",
            category_type="Mock",
            grader_ratings_list=[True, True],
            response_wrong_answers_list=None,
        ),
        ItemRating(
            example_id="2",
            query="q2",
            response="r2",
            category_type="Mock",
            grader_ratings_list=[True, True],
            response_wrong_answers_list=["extra"],
        ),
        ItemRating(
            example_id="3",
            query="q3",
            response="r3",
            category_type="Mock",
            grader_ratings_list=[False, False],
            response_wrong_answers_list=None,
        ),
    ]

    aggregate = aggregate_ratings(ratings)

    assert aggregate.num_total_ratings == 3
    assert aggregate.num_valid_ratings == 3
    assert aggregate.num_answer_correctness_evaluated == 3
    assert aggregate.precision == "55.56%"
    assert aggregate.recall == "66.67%"
    assert aggregate.f1_score == "60.00%"
    assert aggregate.pct_w_ci_all_answers_correct.startswith("33.33")
    assert aggregate.pct_w_ci_fully_incorrect_items.startswith("33.33")
    assert aggregate.pct_w_ci_correct_with_excessive_answers.startswith("33.33")
