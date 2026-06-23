"""Agent roles implemented with plain prompt + JSON protocols."""

import asyncio

from pydantic import BaseModel

from deep_researcher_demo.json_utils import JSONParseError, parse_model_json
from deep_researcher_demo.llm import ChatClient, Message
from deep_researcher_demo.progress import NullProgressReporter, ProgressEvent, ProgressReporter, format_list
from deep_researcher_demo.schemas import (
    InitialResearchQuestions,
    QueryPlan,
    SearchResult,
    SupervisorDecision,
)

import os

INITIAL_QUESTIONS_MAX_TOKENS = 1000
SUPERVISOR_DECISION_MAX_TOKENS = 1000
QUERY_PLAN_MAX_TOKENS = 1000
RESEARCH_SUMMARY_MAX_TOKENS = int(os.getenv("RESEARCH_SUMMARY_MAX_TOKENS", "5000"))
JSON_REPAIR_MAX_TOKENS = 2000
FINAL_REPORT_MAX_TOKENS = 10000

# When set (e.g. SUMMARY_DETAILED=1), researchers write long, information-dense
# digests instead of tight compressions. Used to scale up the reusable-context
# size for KV-reuse experiments; applies identically to all serving backends.
SUMMARY_DETAILED = os.getenv("SUMMARY_DETAILED", "").strip().lower() in {"1", "true", "yes"}

# When set (SUPERVISOR_REASONING=1), the supervisor's per-round decision JSON is
# stored as a reusable KV segment (r_t) and interleaved into later decide()
# contexts: [sys] SEP out_1 SEP r_1 SEP out_2 SEP r_2 ... This turns the
# supervisor into a recurrent reasoning trace (reuse error can compound across
# rounds). Default off -> behavior identical to the summaries-only baseline.
SUPERVISOR_REASONING = os.getenv("SUPERVISOR_REASONING", "").strip().lower() in {"1", "true", "yes"}


def join_reusable_segments(parts: list[str], separator: str) -> str:
    """Join generated texts so each one becomes a reusable KV segment.

    With an empty separator this is the original "\\n\\n---\\n\\n" join. With a
    non-empty separator (the server's LMCACHE_BLEND_SPECIAL_STR, ideally an
    atomic special token such as "<|fim_pad|>"), every part is delimited on
    BOTH sides, so the LMCache-blend backend splits each part into its own
    content-hashed segment and reuses the KV that blend_store_generated saved
    when the part was originally generated. Parts must be embedded verbatim:
    any added or stripped character changes the token sequence and the
    content hash misses entirely.
    """
    if not separator:
        return "\n\n---\n\n".join(parts)
    return separator + separator.join(parts) + separator


def interleave_segments(summaries: list[str], reasonings: list[str]) -> list[str]:
    """Interleave worker outputs and supervisor reasonings: out_1, r_1, out_2, ...

    `reasonings` is typically one shorter than `summaries` (the current round's
    r_t is not generated yet), so the returned flat list ends with the latest
    worker output out_t. With an empty `reasonings` this returns `summaries`
    unchanged (the summaries-only baseline).
    """
    parts: list[str] = []
    for index, summary in enumerate(summaries):
        parts.append(summary)
        if index < len(reasonings):
            parts.append(reasonings[index])
    return parts


class Agent:
    """Base class with JSON repair support."""

    def __init__(self, llm: ChatClient, model: str, kv_reuse_separator: str = "") -> None:
        self.llm = llm
        self.model = model
        self.kv_reuse_separator = kv_reuse_separator

    async def call_json(
        self,
        messages: list[Message],
        schema: type[BaseModel],
        *,
        max_tokens: int = JSON_REPAIR_MAX_TOKENS,
        tag: str | None = None,
        store_generated_kv: bool = False,
    ) -> tuple[BaseModel, str]:
        """Call an LLM and validate JSON output, with one repair attempt.

        Returns (parsed_model, raw_content) where raw_content is the FIRST
        decode's text. When store_generated_kv is set, that first decode's KV is
        stored as a reusable segment; raw_content is exactly what callers must
        embed verbatim downstream to hit it. A repair re-generation does NOT
        overwrite raw_content, so the stored segment and the embedded text stay
        in sync even when the first decode was malformed JSON.
        """
        content = await self.llm.chat(
            messages,
            model=self.model,
            temperature=0.0,
            max_tokens=max_tokens,
            store_generated_kv=store_generated_kv,
            tag=tag,
        )
        try:
            return parse_model_json(content, schema), content
        except Exception as first_error:
            repair_messages = [
                {
                    "role": "system",
                    "content": (
                        "Convert the user's text into one valid JSON object that matches the requested schema. "
                        "Return only JSON.\n\n"
                        f"Schema name: {schema.__name__}\n"
                        f"JSON schema: {schema.model_json_schema()}"
                    ),
                },
                {"role": "user", "content": content},
            ]
            repaired = await self.llm.chat(
                repair_messages,
                model=self.model,
                temperature=0.0,
                max_tokens=JSON_REPAIR_MAX_TOKENS,
                tag="REPAIR_JSON",
            )
            try:
                return parse_model_json(repaired, schema), content
            except Exception as second_error:
                raise JSONParseError(
                    f"Could not parse model output as {schema.__name__}: {first_error}; repair failed: {second_error}"
                ) from second_error


class Supervisor(Agent):
    """Creates initial research sub-questions and decides follow-up questions."""

    async def initialize_question(self, original_question: str, max_questions: int) -> InitialResearchQuestions:
        messages = [
            {
                "role": "system",
                "content": (
                    "Decompose the user's original question into standalone research sub-questions. "
                    "Each sub-question must be complete enough to hand directly to a researcher. "
                    f"Return at most {max_questions} sub-questions. "
                    "Do not ask for clarification. Return only JSON with exactly this shape: "
                    '{"research_questions": ["...", "..."]}'
                ),
            },
            {
                "role": "user",
                "content": f"<original_question>\n{original_question}\n</original_question>",
            },
        ]
        initial, _ = await self.call_json(
            messages,
            InitialResearchQuestions,
            max_tokens=INITIAL_QUESTIONS_MAX_TOKENS,
            tag="INITIAL_RESEARCH_QUESTIONS_JSON",
        )
        typed_initial = initial  # type: ignore[assignment]
        typed_initial.research_questions = [
            question.strip()
            for question in typed_initial.research_questions[:max_questions]
            if question.strip()
        ]
        return typed_initial

    @staticmethod
    def _decide_system(max_followups: int) -> str:
        return (
            "You are the research supervisor. Decide only whether more research is needed. "
            "If more research is needed, produce concrete follow-up questions for researchers. "
            f"Return at most {max_followups} follow-up questions. Return only JSON with this shape: "
            '{"status": "continue|complete", "followup_questions": ["..."], "reason": "..."}'
        )

    def _decide_user(self, original_question: str, findings: str) -> str:
        if self.kv_reuse_separator:
            # KV-reuse layout: summaries first, question after. The blend
            # lookup only counts segments that hit contiguously from token 0,
            # so everything before the first separator must be a constant,
            # cacheable prefix (chat template + system + this fixed header).
            return (
                f"<research_summaries>\n{findings}\n</research_summaries>\n\n"
                f"<original_question>\n{original_question}\n</original_question>"
            )
        return (
            f"<original_question>\n{original_question}\n</original_question>\n\n"
            f"<research_summaries>\n{findings}\n</research_summaries>"
        )

    async def warmup_kv_prefix(self, max_followups: int) -> None:
        """Prime the constant prompt prefix (segment 0) in the KV cache.

        The first segment of the decide() prompt (chat template + system +
        fixed user header) must already be cached for the blend lookup to
        reach the summary segments. A tiny request whose final save stores
        exactly that prefix makes even the first decide() of a session hit.
        """
        if not self.kv_reuse_separator:
            return
        findings = join_reusable_segments(["warmup"], self.kv_reuse_separator)
        messages = [
            {"role": "system", "content": self._decide_system(max_followups)},
            {"role": "user", "content": self._decide_user("warmup", findings)},
        ]
        try:
            # Warmup primes the constant prefix segment; store it so the blend
            # lookup (contiguous-from-token-0) can reach the summary segments.
            await self.llm.chat(
                messages, model=self.model, temperature=0.0, max_tokens=8,
                store_generated_kv=True, tag="SUPERVISOR_DECISION_JSON",
            )
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass

    async def decide(
        self,
        *,
        original_question: str,
        summaries: list[str],
        reasonings: list[str] | None = None,
        max_followups: int,
        store_generated_kv: bool = False,
    ) -> tuple[SupervisorDecision, str]:
        # Interleave worker outputs and prior supervisor reasonings:
        #   SEP out_1 SEP r_1 SEP ... SEP out_{t-1} SEP r_{t-1} SEP out_t
        # `reasonings` is one behind `summaries` (r_t not generated yet), so the
        # context ends with the latest out_t. With reasonings empty this reduces
        # to the summaries-only baseline (identical bytes -> identical behavior).
        parts = interleave_segments(summaries, reasonings or [])
        findings = join_reusable_segments(parts, self.kv_reuse_separator)
        messages = [
            {
                "role": "system",
                "content": self._decide_system(max_followups),
            },
            {
                "role": "user",
                "content": self._decide_user(original_question, findings),
            },
        ]
        # Store this round's decision JSON as the reusable r_t segment only when
        # asked AND in KV-reuse mode. raw_content is the exact stored text, which
        # the caller embeds verbatim next round to hit this segment.
        store = store_generated_kv and bool(self.kv_reuse_separator)
        decision, raw_content = await self.call_json(
            messages,
            SupervisorDecision,
            max_tokens=SUPERVISOR_DECISION_MAX_TOKENS,
            tag="SUPERVISOR_DECISION_JSON",
            store_generated_kv=store,
        )
        typed_decision = decision  # type: ignore[assignment]
        typed_decision.followup_questions = [
            question.strip()
            for question in typed_decision.followup_questions[:max_followups]
            if question.strip()
        ]
        if typed_decision.status == "complete":
            typed_decision.followup_questions = []
        if typed_decision.status == "continue" and not typed_decision.followup_questions:
            typed_decision.status = "complete"
            typed_decision.reason = typed_decision.reason or "No follow-up questions were provided."
        return typed_decision, raw_content


class Researcher:
    """Turns a question into search queries and summarizes search results."""

    def __init__(
        self,
        llm: ChatClient,
        planner_model: str,
        summary_model: str,
        kv_reuse_separator: str = "",
    ) -> None:
        self.planner = Agent(llm, planner_model)
        self.summarizer = Agent(llm, summary_model)
        self.kv_reuse_separator = kv_reuse_separator

    async def plan_queries(self, question: str, max_queries: int) -> QueryPlan:
        messages = [
            {
                "role": "system",
                "content": (
                    "Break the research question into concise web search queries. "
                    f"Return at most {max_queries} search queries. "
                    "Return only JSON with this shape: "
                    '{"queries": ["...", "..."]}'
                ),
            },
            {
                "role": "user",
                "content": f"<research_question>\n{question}\n</research_question>",
            },
        ]
        plan, _ = await self.planner.call_json(
            messages,
            QueryPlan,
            max_tokens=QUERY_PLAN_MAX_TOKENS,
            tag="QUERY_PLAN_JSON",
        )
        typed_plan = plan  # type: ignore[assignment]
        typed_plan.queries = [
            query.strip()
            for query in typed_plan.queries[:max_queries]
            if query.strip()
        ]
        if not typed_plan.queries:
            typed_plan.queries = [question]
        return typed_plan

    async def summarize_results(
        self,
        *,
        question: str,
        queries: list[str],
        results: list[SearchResult],
    ) -> str:
        if SUMMARY_DETAILED:
            summary_instruction = (
                "Write a detailed, information-dense research digest of the provided "
                "search results for the current sub-query. Preserve all facts, "
                "figures, dates, names, definitions, and source attributions that "
                "could be relevant to the overall research question. Organize the "
                "digest by source. Be thorough rather than brief. "
                "Write plain text only."
            )
        else:
            summary_instruction = (
                "Compress the provided search results for the current sub-query. "
                "Only Extract key information that is relevant to the overall research question. "
                "Write plain text only."
            )
        messages = [
            {
                "role": "system",
                "content": summary_instruction,
            },
            {
                "role": "user",
                "content": (
                    f"<research_question>\n{question}\n</research_question>\n\n"
                    f"<search_results>\n{format_search_results(results)}\n</search_results>"
                ),
            },
        ]
        summary = await self.summarizer.llm.chat(
            messages,
            model=self.summarizer.model,
            temperature=0.0,
            max_tokens=RESEARCH_SUMMARY_MAX_TOKENS,
            # The researcher summary is the reusable segment: opt in to storing
            # its decode-generated KV (only meaningful in KV-reuse mode).
            store_generated_kv=bool(self.kv_reuse_separator),
            tag="RESEARCH_SUMMARY_TEXT",
        )
        if self.kv_reuse_separator:
            # Keep the text byte-exact: downstream prompts embed it verbatim
            # between separators, and the serving side must re-tokenize it to
            # the same token ids that were stored when it was generated.
            # Stripping even one whitespace character breaks the KV reuse.
            return summary if summary.strip() else "No relevant information was found."
        return summary.strip() or "No relevant information was found."

    async def _process_sub_query(
        self,
        *,
        question: str,
        query: str,
        search_provider,
        max_results: int,
        reporter: ProgressReporter,
    ) -> tuple[str, list[str]]:
        reporter.emit(
            ProgressEvent(
                "subquery_start",
                f"Sub-query started: {query}",
                {"question": question, "query": query},
            )
        )
        results = await search_provider.search([query], max_results=max_results)
        urls = extract_urls(results)
        reporter.emit(
            ProgressEvent(
                "subquery_search_complete",
                f"{len(results)} results for: {query}",
                {
                    "question": question,
                    "query": query,
                    "result_count": len(results),
                    "backup_url_count": len(urls),
                },
            )
        )
        summary = await self.summarize_results(
            question=question,
            queries=[query],
            results=results,
        )
        reporter.emit(
            ProgressEvent(
                "subquery_summary_complete",
                f"Compressed context ready for: {query}",
                {
                    "question": question,
                    "query": query,
                    "summary_chars": len(summary),
                    "backup_url_count": len(urls),
                },
            )
        )
        return summary, urls

    async def research(
        self,
        question: str,
        search_provider,
        max_results: int,
        max_queries: int,
        reporter: ProgressReporter | None = None,
    ) -> tuple[str, list[str]]:
        reporter = reporter or NullProgressReporter()
        plan = await self.plan_queries(question, max_queries=max_queries)
        reporter.emit(
            ProgressEvent(
                "queries_planned",
                f"{len(plan.queries)} queries: {format_list(plan.queries)}",
                {"question": question, "queries": plan.queries},
            )
        )
        reporter.emit(
            ProgressEvent(
                "search_start",
                f"Searching {len(plan.queries)} queries, max {max_results} results per query",
                {"question": question, "query_count": len(plan.queries), "max_results_per_query": max_results},
            )
        )
        # max_results 表示一个query最多返回多少个搜索结果
        sub_results = list(
            await asyncio.gather(
                *[
                    self._process_sub_query(
                        question=question,
                        query=query,
                        search_provider=search_provider,
                        max_results=max_results,
                        reporter=reporter,
                    )
                    for query in plan.queries
                ]
            )
        )
        sub_summaries = [summary for summary, _ in sub_results]
        backup_urls = dedupe_urls([url for _, urls in sub_results for url in urls])
        if self.kv_reuse_separator:
            # Each sub-summary is a separate generation (its own stored KV
            # segment), so they must stay individually delimited. The
            # supervisor/final-writer add the outer separators via
            # join_reusable_segments, giving: SEP s1 SEP s2 ... SEP.
            combined_summary = (
                self.kv_reuse_separator.join(sub_summaries)
                or "No relevant information was found."
            )
        else:
            combined_summary = "\n\n".join(sub_summaries) or "No relevant information was found."
        reporter.emit(
            ProgressEvent(
                "search_complete",
                f"Processed {len(sub_summaries)} sub-query contexts",
                {
                    "question": question,
                    "subquery_count": len(sub_summaries),
                    "backup_url_count": len(backup_urls),
                },
            )
        )
        reporter.emit(
            ProgressEvent(
                "summary_complete",
                f"Summary ready for: {question}",
                {
                    "question": question,
                    "summary_chars": len(combined_summary),
                    "backup_url_count": len(backup_urls),
                },
            )
        )
        return combined_summary, backup_urls


class FinalWriter(Agent):
    """Writes the final report from accumulated researcher summaries."""

    _SYSTEM = (
        "Write a clear, well-structured Markdown research report. "
        "Use the same language as the user's original question. Include source links where useful."
    )

    def _write_user(self, original_question: str, findings: str) -> str:
        if self.kv_reuse_separator:
            # See Supervisor._decide_user: constant prefix before the first
            # separator so the blend lookup reaches the summary segments.
            return (
                f"<findings>\n{findings}\n</findings>\n\n"
                f"<original_question>\n{original_question}\n</original_question>"
            )
        return (
            f"<original_question>\n{original_question}\n</original_question>\n\n"
            f"<findings>\n{findings}\n</findings>"
        )

    async def warmup_kv_prefix(self) -> None:
        """Prime the constant prompt prefix (segment 0) in the KV cache."""
        if not self.kv_reuse_separator:
            return
        findings = join_reusable_segments(["warmup"], self.kv_reuse_separator)
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": self._write_user("warmup", findings)},
        ]
        try:
            # Warmup primes the constant prefix segment; store it so the blend
            # lookup (contiguous-from-token-0) can reach the summary segments.
            await self.llm.chat(
                messages, model=self.model, temperature=0.0, max_tokens=8,
                store_generated_kv=True, tag="FINAL_REPORT_MARKDOWN",
            )
        except Exception:  # noqa: BLE001 - warmup is best-effort
            pass

    async def write(self, *, original_question: str, summaries: list[str]) -> str:
        findings = join_reusable_segments(summaries, self.kv_reuse_separator)
        messages = [
            {
                "role": "system",
                "content": self._SYSTEM,
            },
            {
                "role": "user",
                "content": self._write_user(original_question, findings),
            },
        ]
        return await self.llm.chat(
            messages,
            model=self.model,
            temperature=0.0,
            max_tokens=FINAL_REPORT_MAX_TOKENS,
            tag="FINAL_REPORT_MARKDOWN",
        )


def format_search_results(results: list[SearchResult]) -> str:
    """Format search results for the summarizer prompt."""
    if not results:
        return "No search results."
    lines = []
    for index, result in enumerate(results, start=1):
        content = result.raw_content or result.snippet
        lines.append(
            f"[{index}] Query: {result.query}\n"
            f"Title: {result.title}\n"
            f"URL: {result.url}\n"
            f"Snippet: {content}"
        )
    return "\n\n".join(lines)


def extract_urls(results: list[SearchResult]) -> list[str]:
    """Extract non-empty URLs from normalized search results."""
    return [result.url.strip() for result in results if result.url.strip()]


def dedupe_urls(urls: list[str]) -> list[str]:
    """Deduplicate URLs while preserving first-seen order."""
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped
