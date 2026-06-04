"""Scoring helpers for DeepSearchQA."""

import dataclasses
import json
import math
import re
import string
from collections import defaultdict
from typing import Any


FINAL_ANSWER_RE = re.compile(
    r"^##+\s*Final Answer\s*$\s*(.*?)(?=^##+\s|\Z)",
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


def extract_final_answer(report: str) -> str:
    """Extract the `## Final Answer` section from a report."""
    match = FINAL_ANSWER_RE.search(report or "")
    if not match:
        return (report or "").strip()
    return match.group(1).strip()


def normalize_answer(value: str) -> str:
    """Normalize an answer string for approximate exact matching."""
    text = (value or "").lower()
    text = text.replace("&", " and ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def split_set_answer(value: str) -> list[str]:
    """Split set-style answers into normalized answer items."""
    text = value or ""
    text = re.sub(r"^\s*[-*]\s+", "", text, flags=re.MULTILINE)
    parts = re.split(r"\n|;|,", text)
    normalized = []
    for part in parts:
        item = normalize_answer(part)
        if item:
            normalized.append(item)
    return list(dict.fromkeys(normalized))


def score_answer(prediction: str, gold: str, answer_type: str) -> dict[str, Any]:
    """Score one prediction against gold."""
    if answer_type == "Set Answer":
        return score_set_answer(prediction, gold)
    return score_single_answer(prediction, gold)


def score_single_answer(prediction: str, gold: str) -> dict[str, Any]:
    """Score a single-answer task."""
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)
    exact = bool(pred_norm) and pred_norm == gold_norm
    gold_substring = bool(gold_norm) and gold_norm in pred_norm
    return {
        "answer_type": "Single Answer",
        "exact": exact,
        "gold_substring": gold_substring,
        "precision": 1.0 if exact else 0.0,
        "recall": 1.0 if exact else 0.0,
        "f1": 1.0 if exact else 0.0,
        "pred_items": [pred_norm] if pred_norm else [],
        "gold_items": [gold_norm] if gold_norm else [],
    }


def score_set_answer(prediction: str, gold: str) -> dict[str, Any]:
    """Score a set-answer task with item-level precision, recall, and F1."""
    pred_items = set(split_set_answer(prediction))
    gold_items = set(split_set_answer(gold))
    correct = pred_items & gold_items
    precision = len(correct) / len(pred_items) if pred_items else 0.0
    recall = len(correct) / len(gold_items) if gold_items else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    exact_set_match = bool(gold_items) and pred_items == gold_items
    return {
        "answer_type": "Set Answer",
        "exact": exact_set_match,
        "exact_set_match": exact_set_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct_count": len(correct),
        "pred_count": len(pred_items),
        "gold_count": len(gold_items),
        "pred_items": sorted(pred_items),
        "gold_items": sorted(gold_items),
    }


@dataclasses.dataclass
class ItemRatingBase:
    """Base item rating compatible with the starter-code shape."""

    original_index: int | None = dataclasses.field(default=None, kw_only=True, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ItemRating(ItemRatingBase):
    """DeepSearchQA item-level autorater result."""

    example_id: str = ""
    query: str = ""
    response: str = ""
    category_type: str | None = None
    expected_correct_answer: str | None = None
    sample_id: int | None = None
    answer_type: str | None = None
    final_answer: str | None = None
    local_scores: dict[str, Any] | None = None

    answer_correctness_explanation: str | None = None
    expected_correct_answer_list: list[str] | None = None
    response_wrong_answers_list: list[str] | None = None
    grader_ratings_list: list[bool] | None = None

    empty_model_response: bool = False
    empty_auto_rater_response: bool = False
    invalid_auto_rater_response: bool = False
    rating_response: str = ""
    rating_prompt: str = ""
    error_message: str | None = None


@dataclasses.dataclass
class ProjectRatingBase:
    """Base project rating compatible with the starter-code shape."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ProjectRating(ProjectRatingBase):
    """DeepSearchQA project-level autorater aggregate."""

    num_total_ratings: int = 0
    num_empty_model_response: int = 0
    num_invalid_auto_rater_response: int = 0
    num_empty_auto_rater_response: int = 0
    num_valid_ratings: int = 0
    num_answer_correctness_evaluated: int = 0

    pct_w_ci_all_answers_correct: str = ""
    pct_w_ci_fully_incorrect_items: str = ""
    pct_w_ci_correct_with_excessive_answers: str = ""

    pct_empty_model_response: float = 0.0
    pct_invalid_auto_rater_response: float = 0.0
    pct_empty_auto_rater_response: float = 0.0

    precision: str = ""
    recall: str = ""
    f1_score: str = ""


def parse_json_response(ori_json_response: str) -> Any:
    """Parse JSON from a raw autorater response, including fenced JSON."""
    json_str = (ori_json_response or "").strip()
    if not json_str or json_str.upper() == "NULL":
        return None
    start_marker = "```json"
    start_idx = json_str.find(start_marker)
    if start_idx != -1:
        json_str = json_str[start_idx + len(start_marker) :].strip()
        end_idx = json_str.rfind("```")
        if end_idx != -1:
            json_str = json_str[:end_idx].strip()
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None


def get_answer_correctness_details(json_response: Any) -> dict[str, bool] | None:
    """Extract valid Answer Correctness / Correctness Details."""
    try:
        details = json_response["Answer Correctness"]["Correctness Details"]
    except (KeyError, TypeError):
        return None
    if not isinstance(details, dict):
        return None
    if not all(isinstance(key, str) for key in details):
        return None
    if not all(isinstance(value, bool) for value in details.values()):
        return None
    return details


def get_excessive_answers(json_response: Any) -> list[str] | None:
    """Extract Excessive Answers, matching starter-code missing-key behavior."""
    try:
        excessive_answers = json_response["Answer Correctness"]["Excessive Answers"]
    except (KeyError, TypeError):
        return []
    if not isinstance(excessive_answers, list):
        return None
    if not all(isinstance(item, str) for item in excessive_answers):
        return None
    return excessive_answers


def build_item_rating_from_report(record: dict[str, Any]) -> ItemRating:
    """Create an ItemRating shell from a generated report record."""
    return ItemRating(
        original_index=int(record["sample_id"]) if record.get("sample_id") is not None else None,
        example_id=str(record.get("example_id", record.get("sample_id", ""))).strip(),
        query=str(record.get("problem", "")).strip(),
        response=str(record.get("final_report", "")).strip(),
        category_type=str(record.get("problem_category", "")).strip(),
        expected_correct_answer=str(record.get("answer", "")).strip(),
        sample_id=int(record["sample_id"]) if record.get("sample_id") is not None else None,
        answer_type=str(record.get("answer_type", "")).strip(),
        final_answer=str(record.get("final_answer", "")).strip(),
        local_scores=record.get("local_scores"),
    )


def item_rating_from_dict(record: dict[str, Any]) -> ItemRating:
    """Hydrate an ItemRating from a JSONL dictionary."""
    valid_fields = {field.name for field in dataclasses.fields(ItemRating)}
    values = {key: value for key, value in record.items() if key in valid_fields}
    return ItemRating(**values)


def reduce_autorater_response(
    item_rating: ItemRating,
    *,
    grader_llm_response_text: str,
    grader_llm_prompt_text: str,
    rating_error: str | None = None,
) -> ItemRating:
    """Parse the autorater response into starter-code item rating fields."""
    item_rating.rating_prompt = grader_llm_prompt_text
    item_rating.rating_response = grader_llm_response_text or ""

    if not item_rating.response:
        item_rating.empty_model_response = True
        item_rating.error_message = "AI response was empty."
        return item_rating

    if rating_error:
        item_rating.empty_auto_rater_response = True
        item_rating.error_message = rating_error
        return item_rating

    if not grader_llm_response_text:
        item_rating.empty_auto_rater_response = True
        item_rating.error_message = "Auto-rater response was empty."
        return item_rating

    parsed_json_response = parse_json_response(grader_llm_response_text)
    if not parsed_json_response:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid JSON response from auto-rater."
        return item_rating

    answer_correctness_node = parsed_json_response.get("Answer Correctness")
    if not isinstance(answer_correctness_node, dict):
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Missing or malformed 'Answer Correctness' node."
        return item_rating

    grader_explanation = answer_correctness_node.get("Explanation")
    if not isinstance(grader_explanation, str):
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Missing or malformed 'Explanation' in Answer Correctness."
        return item_rating
    item_rating.answer_correctness_explanation = grader_explanation

    details = get_answer_correctness_details(parsed_json_response)
    if details is None:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid 'Correctness Details' in Answer Correctness."
        return item_rating
    item_rating.expected_correct_answer_list = list(details.keys())
    item_rating.grader_ratings_list = list(details.values())

    excessive_answers = get_excessive_answers(parsed_json_response)
    if excessive_answers is None:
        item_rating.invalid_auto_rater_response = True
        item_rating.error_message = "Invalid 'Excessive Answers' in Answer Correctness."
        return item_rating
    if excessive_answers:
        item_rating.response_wrong_answers_list = excessive_answers

    return item_rating


def calculate_metric(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> dict[str, float]:
    """Calculate starter-code precision, recall, and F1."""
    precision_val = 0.0
    if true_positives + false_positives > 0:
        precision_val = true_positives / (true_positives + false_positives)

    recall_val = 0.0
    if true_positives + false_negatives > 0:
        recall_val = true_positives / (true_positives + false_negatives)

    f1_score_val = 0.0
    if precision_val + recall_val > 0:
        f1_score_val = 2 * (precision_val * recall_val) / (precision_val + recall_val)

    return {
        "precision": precision_val,
        "recall": recall_val,
        "f1_score": f1_score_val,
    }


def aggregate_ratings(item_ratings: list[ItemRating | dict[str, Any]]) -> ProjectRating:
    """Aggregate item-level starter-code ratings into project-level metrics."""
    ratings = [
        item_rating_from_dict(item) if isinstance(item, dict) else item
        for item in item_ratings
    ]
    total_items = len(ratings)
    project_rating = ProjectRating(num_total_ratings=total_items)
    if not ratings:
        return project_rating

    num_answer_correctness_evaluated = 0
    num_answer_correctness_all_correct = 0
    num_fully_incorrect_items = 0
    num_items_correct_with_excessive_answers = 0
    per_item_metrics = {
        "precision": [],
        "recall": [],
        "f1_score": [],
    }

    for item_rating in ratings:
        if item_rating.invalid_auto_rater_response:
            project_rating.num_invalid_auto_rater_response += 1
            continue
        if item_rating.empty_auto_rater_response:
            project_rating.num_empty_auto_rater_response += 1
            continue
        if item_rating.empty_model_response:
            project_rating.num_empty_model_response += 1
            continue

        project_rating.num_valid_ratings += 1

        if item_rating.grader_ratings_list is not None:
            num_answer_correctness_evaluated += 1
            grader_ratings = item_rating.grader_ratings_list
            num_correct = sum(1 for rating in grader_ratings if rating)

            true_positives = num_correct
            false_negatives = len(grader_ratings) - num_correct

            has_expected_answers = bool(grader_ratings)
            all_expected_answers_correct = False
            if has_expected_answers:
                all_expected_answers_correct = num_correct == len(grader_ratings)
                if num_correct == 0:
                    num_fully_incorrect_items += 1

            excessive_answers = item_rating.response_wrong_answers_list
            has_excessive_answers = bool(excessive_answers)
            false_positives = len(excessive_answers) if has_excessive_answers else 0
            if has_excessive_answers and (all_expected_answers_correct or not has_expected_answers):
                num_items_correct_with_excessive_answers += 1

            is_all_correct = (all_expected_answers_correct or not has_expected_answers) and not has_excessive_answers
            if is_all_correct:
                num_answer_correctness_all_correct += 1

            per_item_metric = calculate_metric(true_positives, false_positives, false_negatives)
            for key, value in per_item_metric.items():
                per_item_metrics[key].append(value)

    if total_items > 0:
        project_rating.pct_empty_model_response = round(project_rating.num_empty_model_response * 100.0 / total_items, 2)
        project_rating.pct_invalid_auto_rater_response = round(
            project_rating.num_invalid_auto_rater_response * 100.0 / total_items,
            2,
        )
        project_rating.pct_empty_auto_rater_response = round(
            project_rating.num_empty_auto_rater_response * 100.0 / total_items,
            2,
        )

    if num_answer_correctness_evaluated > 0:
        project_rating.num_answer_correctness_evaluated = num_answer_correctness_evaluated
        project_rating.pct_w_ci_all_answers_correct = calculate_ci_str(
            num_answer_correctness_all_correct,
            num_answer_correctness_evaluated,
        )
        project_rating.pct_w_ci_fully_incorrect_items = calculate_ci_str(
            num_fully_incorrect_items,
            num_answer_correctness_evaluated,
        )
        project_rating.pct_w_ci_correct_with_excessive_answers = calculate_ci_str(
            num_items_correct_with_excessive_answers,
            num_answer_correctness_evaluated,
        )
        project_rating.precision = format_percentage(mean(per_item_metrics["precision"]))
        project_rating.recall = format_percentage(mean(per_item_metrics["recall"]))
        project_rating.f1_score = format_percentage(mean(per_item_metrics["f1_score"]))

    return project_rating


def aggregate_starter_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate starter-code autorater metrics overall and by useful keys."""
    ratings = [item_rating_from_dict(record) for record in records]
    return {
        **aggregate_ratings(ratings).to_dict(),
        "by_category": aggregate_ratings_by_key(ratings, "category_type"),
        "by_answer_type": aggregate_ratings_by_key(ratings, "answer_type"),
    }


def aggregate_ratings_by_key(item_ratings: list[ItemRating], key: str) -> dict[str, Any]:
    """Aggregate starter-code autorater metrics by an ItemRating attribute."""
    groups: dict[str, list[ItemRating]] = defaultdict(list)
    for item_rating in item_ratings:
        groups[str(getattr(item_rating, key) or "")].append(item_rating)
    return {
        group_key: aggregate_ratings(group_records).to_dict()
        for group_key, group_records in sorted(groups.items())
    }


def calculate_ci_str(count: int, total: int, z: float = 1.96) -> str:
    """Return a normal-approximation confidence interval string like the notebook."""
    if total == 0:
        return f"N/A ({count}/{total})"
    count = max(0, min(count, total))
    p = count / total
    p_percent = p * 100.0
    try:
        variance = p * (1.0 - p)
        margin_of_error = z * math.sqrt(variance / total)
    except (ValueError, ZeroDivisionError):
        return "N/A"
    moe_percent = margin_of_error * 100.0
    result = f"{round(p_percent, 2):.2f} +/- {round(moe_percent, 2):.2f} ({count}/{total})"
    if total <= 5:
        result += " (CI not robust for n<=5)"
    return result


def format_percentage(value: float) -> str:
    """Format a float as a two-decimal percentage."""
    return f"{value:.2%}"


def aggregate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate scoring metrics overall and by category / answer type."""
    successful = [record for record in records if not record.get("error")]
    return {
        "total": len(records),
        "successful": len(successful),
        "failed": len(records) - len(successful),
        "overall": summarize_group(successful),
        "by_category": summarize_by_key(successful, "problem_category"),
        "by_answer_type": summarize_by_key(successful, "answer_type"),
    }


def summarize_by_key(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get(key, ""))].append(record)
    return {group_key: summarize_group(group_records) for group_key, group_records in sorted(groups.items())}


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "accuracy": 0.0,
            "gold_substring_rate": 0.0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
        }
    scores = [record.get("local_scores") or record.get("scores", {}) for record in records]
    return {
        "count": len(records),
        "accuracy": mean(1.0 if score.get("exact") else 0.0 for score in scores),
        "gold_substring_rate": mean(1.0 if score.get("gold_substring") else 0.0 for score in scores),
        "avg_precision": mean(float(score.get("precision", 0.0)) for score in scores),
        "avg_recall": mean(float(score.get("recall", 0.0)) for score in scores),
        "avg_f1": mean(float(score.get("f1", 0.0)) for score in scores),
    }


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
