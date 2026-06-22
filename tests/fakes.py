import json
import re
import string

from deep_researcher_demo.schemas import SearchResult


class StubChatClient:
    def __init__(self) -> None:
        self.supervisor_decision_calls = 0
        self.research_summary_calls = 0

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        store_generated_kv: bool = False,
        tag: str | None = None,
    ) -> str:
        joined = "\n".join(message.get("content", "") for message in messages)

        if tag == "INITIAL_RESEARCH_QUESTIONS_JSON":
            question = _between(joined, "<original_question>", "</original_question>") or "the user question"
            return json.dumps(
                {
                    "research_questions": [
                        f"Research the main evidence for: {question}",
                        f"Research the key tradeoffs and limitations for: {question}",
                    ]
                }
            )

        if tag == "QUERY_PLAN_JSON":
            question = _between(joined, "<research_question>", "</research_question>") or "topic"
            return json.dumps({"queries": [f"{question} overview", f"{question} evidence"]})

        if tag == "RESEARCH_SUMMARY_TEXT":
            self.research_summary_calls += 1
            question = _between(joined, "<research_question>", "</research_question>") or "topic"
            query = _between(joined, "Query:", "\n") or "unknown query"
            return (
                f"Stub compressed context for {query} within {question}. "
                "The search results suggest several relevant facts."
            )

        if tag == "SUPERVISOR_DECISION_JSON":
            self.supervisor_decision_calls += 1
            if self.supervisor_decision_calls == 1:
                return json.dumps(
                    {
                        "status": "continue",
                        "followup_questions": [
                            "What evidence is strongest?",
                            "What risks or limitations remain?",
                        ],
                        "reason": "More targeted follow-up research would improve coverage.",
                    }
                )
            return json.dumps(
                {
                    "status": "complete",
                    "followup_questions": [],
                    "reason": "The available summaries are sufficient for a concise report.",
                }
            )

        if tag == "FINAL_REPORT_MARKDOWN":
            original = _between(joined, "<original_question>", "</original_question>") or "the topic"
            return (
                "# Stub Deep Research Report\n\n"
                f"## Question\n{original}\n\n"
                "## Findings\nStub findings compiled from researcher summaries.\n\n"
                "## Final Answer\nStub Answer\n"
            )

        # Judge/autorater calls carry no tag; route them by prompt content.
        if "Answer Correctness Task" in joined and "<response>" in joined:
            answer = _between_last(joined, "<answer>", "</answer>")
            answer_type = _between(joined, "Prompt Type:", "<answer>").strip()
            response = _between_last(joined, "<response>", "</response>")
            response_norm = _normalize(response)
            details = {
                expected_answer: _normalize(expected_answer) in response_norm
                for expected_answer in _expected_answers(answer, answer_type)
            }
            return json.dumps(
                {
                    "Answer Correctness": {
                        "Explanation": "Stub autorater compared expected answer parts against the response text.",
                        "Correctness Details": details,
                        "Excessive Answers": [],
                    }
                }
            )

        if tag == "REPAIR_JSON":
            return "{}"

        return "{}"


class StubSearchProvider:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], int]] = []

    async def search(self, queries: list[str], max_results: int = 5) -> list[SearchResult]:
        self.calls.append((queries, max_results))
        results: list[SearchResult] = []
        for query in queries:
            for index in range(min(max_results, 2)):
                results.append(
                    SearchResult(
                        query=query,
                        title=f"Stub result {index + 1} for {query}",
                        url=f"https://example.com/{abs(hash((query, index))) % 100000}",
                        snippet=f"Stub snippet {index + 1} about {query}.",
                    )
                )
        return results


def _between(text: str, start: str, end: str) -> str:
    try:
        return text.split(start, 1)[1].split(end, 1)[0].strip()
    except IndexError:
        return ""


def _between_last(text: str, start: str, end: str) -> str:
    try:
        return text.rsplit(start, 1)[1].split(end, 1)[0].strip()
    except IndexError:
        return ""


def _expected_answers(answer: str, answer_type: str) -> list[str]:
    if answer_type == "Set Answer":
        parts = re.split(r"\n|;|,", answer or "")
    else:
        parts = [answer or ""]
    return [part.strip() for part in parts if part.strip()]


def _normalize(value: str) -> str:
    text = (value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())
