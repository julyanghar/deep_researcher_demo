"""Agent roles implemented with plain prompt + JSON protocols."""

import asyncio

from pydantic import BaseModel

from deep_researcher_demo.json_utils import JSONParseError, parse_model_json
from deep_researcher_demo.llm import ChatClient, Message
from deep_researcher_demo.progress import NullProgressReporter, ProgressEvent, ProgressReporter, format_list
from deep_researcher_demo.schemas import (
    InitialResearchQuestions,
    Outline,
    QueryPlan,
    ReportReview,
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
# REPORT_MODE: "answer"(默认,deepsearchqa 短答案式,KV 复用路径完全不变)
#            | "detailed_cited"(benchmark 评测用:分章节长报告 + inline [URL] 引用 + References)
REPORT_MODE = os.getenv("REPORT_MODE", "answer")

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


def _env_flag(name: str) -> bool:
    """Read a boolean env flag at call time (so a runner setting it in-process
    takes effect), matching the SUMMARY_DETAILED / SUPERVISOR_REASONING style."""
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


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


async def outline_and_sections(
    agent: "Agent",
    *,
    question: str,
    source_pool: list[str],
    sources_tag: str,
    outline_system: str,
    section_system: str,
    outline_tag: str,
    section_tag: str,
    section_max_tokens: int,
    max_sections: int = 8,
) -> list[tuple[str, str]]:
    """Steps 1-2 of outline_then_parallel: plan the outline, generate every
    section in parallel, and return [(title, raw_body), ...]. The outline-failed
    fallback (single generation over all sources) is returned as [("", body)] —
    an empty title is the unambiguous fallback marker since outline titles are
    filtered to be non-empty.
    """
    def _numbered(items: list[str]) -> str:
        return "\n\n".join(f"[{i}] {s}" for i, s in enumerate(items, 1))

    async def _section(title: str, source_ids: list[int], all_titles: list[str]) -> str:
        chosen = [source_pool[i - 1] for i in source_ids if 1 <= i <= len(source_pool)]
        if not chosen:  # empty / out-of-range ids -> give the section everything
            chosen = source_pool
        if title:
            # Sibling-aware focus: show the full outline + name this section, and
            # append the "only this section, no preamble/overlap" directive so the
            # section stays in its lane (fights the section-split output bloat).
            sys_content = section_system + _SECTION_FOCUS
            outline_list = "\n".join(f"- {t}" for t in all_titles)
            focus = (
                f"\n\n<outline>\n{outline_list}\n</outline>\n\n"
                f"<section_to_write>\n{title}\n</section_to_write>"
            )
        else:  # fallback single generation over everything: no outline/focus
            sys_content = section_system
            focus = ""
        messages = [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": (
                f"<question>\n{question}\n</question>{focus}\n\n"
                f"<{sources_tag}>\n{_numbered(chosen)}\n</{sources_tag}>"
            )},
        ]
        return await agent.llm.chat(
            messages, model=agent.model, temperature=0.0,
            max_tokens=section_max_tokens, tag=section_tag,
        )

    # 1) Outline (one serial call).
    outline_messages = [
        {"role": "system", "content": outline_system},
        {"role": "user", "content": (
            f"<question>\n{question}\n</question>\n\n"
            f"<{sources_tag}>\n{_numbered(source_pool)}\n</{sources_tag}>"
        )},
    ]
    try:
        outline, _ = await agent.call_json(
            outline_messages, Outline, max_tokens=QUERY_PLAN_MAX_TOKENS, tag=outline_tag,
        )
        sections = [s for s in outline.sections if (s.title or "").strip()][:max_sections]
    except Exception:  # noqa: BLE001 - fall back to single generation
        sections = []

    # 2) Fallback: no usable outline -> single generation over all sources.
    if not sections:
        body = await _section("", [], [])
        return [("", body)]

    # 3) Sections in parallel.
    all_titles = [s.title for s in sections]
    bodies = await asyncio.gather(
        *(_section(s.title, s.source_ids, all_titles) for s in sections)
    )
    return list(zip(all_titles, bodies))


async def outline_then_parallel(
    agent: "Agent",
    *,
    question: str,
    source_pool: list[str],
    sources_tag: str,
    outline_system: str,
    section_system: str,
    outline_tag: str,
    section_tag: str,
    section_max_tokens: int,
    heading_prefix: str = "## ",
    max_sections: int = 8,
) -> str:
    """Generate text by (1) planning an outline (sections + the 1-based ids of
    the sources each uses), (2) generating every section in PARALLEL from ONLY
    its assigned sources, then (3) concatenating them under their titles.

    `source_pool[i-1]` is the already-formatted text of source i. Shared by
    SUMMARY_PARALLEL (sources = search results) and REPORT_PARALLEL (sources =
    findings). Falls back to a single generation over all sources when the
    outline is empty/unparseable, so it never crashes or returns empty.
    """
    sections = await outline_and_sections(
        agent,
        question=question,
        source_pool=source_pool,
        sources_tag=sources_tag,
        outline_system=outline_system,
        section_system=section_system,
        outline_tag=outline_tag,
        section_tag=section_tag,
        section_max_tokens=section_max_tokens,
        max_sections=max_sections,
    )
    if len(sections) == 1 and not sections[0][0]:  # outline-failed fallback
        return sections[0][1].strip() or "No relevant information was found."
    parts: list[str] = []
    for title, body in sections:
        body = (body or "").strip()
        if not body:
            continue
        parts.append(f"{heading_prefix}{title}\n\n{body}" if heading_prefix else body)
    return "\n\n".join(parts) or "No relevant information was found."


# Outline planner system prompt for SUMMARY_PARALLEL (report has its own on
# FinalWriter). Kept module-level so summarize_results can pass it directly.
_SUMMARY_OUTLINE_SYSTEM = (
    "Plan a short outline for compressing the search results into a digest for the sub-query. "
    "Split the answer into 2-5 short, non-overlapping sections. For each section give a concise "
    "title and the ids of the search results it should use (from the <search_results> block). "
    'Return only JSON with this shape: {"sections": [{"title": "...", "source_ids": [1, 2]}]}'
)

# Appended (by outline_then_parallel) to every section's system prompt: keeps each
# parallel section in its own lane so the split output does not balloon (measured
# ~1.59x summary output bloat came from sections repeating context / overlapping).
_SECTION_FOCUS = (
    " You are writing ONLY the section named in <section_to_write>, which is one of the sections "
    "listed in <outline>. Cover ONLY this section's topic; do NOT repeat content that belongs to "
    "the other sections, do NOT restate the question, and write no separate intro/preamble or "
    "conclusion. Be concise and non-repetitive."
)

# Section-generation system prompt for SUMMARY_PARALLEL (terser than the single-call
# `summary_instruction`, which tells the model to answer the whole sub-query).
_SUMMARY_SECTION_SYSTEM = (
    "Extract ONLY the facts for this section's aspect from the provided search results. "
    "Be terse and factual: no preamble, no restating the question. Write plain text only."
)


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
        # The real original_question is already available here, but we pass a
        # throwaway ("warmup") on purpose. Only segment 0 -- everything before
        # the first separator (chat template + system + summaries header) -- is
        # what we prime; the dummy question and dummy summary sit after it and
        # are never matched. Keeping the question OUT of the prefix is the whole
        # point: that prefix is then byte-identical for every question and every
        # session, so it is primed once and stays hit across the entire run.
        # Folding the real question into the prefix would only cache its handful
        # of tokens (re-prefilled cheaply each round anyway) while making the
        # prefix per-question and unshareable -- a zero-gain trade.
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
        if os.getenv("SUMMARY_QUOTE") == "1":
            # L3 实验:逐字引用让 summary 输出"可收割"(suffix 投机解码在长连续照搬处才有加速),
            # 同时逐字引用本身可能减少改写失真。默认关,不影响任何现有路径。
            summary_instruction += (
                " When a source contains directly relevant sentences, QUOTE them "
                "VERBATIM (copy the exact wording, do not paraphrase or translate) as "
                "bullet points, and put the source URL right after each quote. Only "
                "paraphrase when no source sentence is directly usable. Be selective: "
                "at most 2-3 short quotes per source, and keep the whole digest no "
                "longer than a normal concise digest (do not exceed about 600 words)."
            )
        # SUMMARY_PARALLEL: DEPRECATED (2026-07-20 用户拍板:分节只用于 report,
        # summary 不再分节;保留代码但任何实验不得设此 env)。原语义:outline ->
        # assign sources -> parallel sections. Lossy, gated off by default and
        # disabled in KV-reuse mode (parallel split breaks the stored-segment hash).
        if _env_flag("SUMMARY_PARALLEL") and not self.kv_reuse_separator and results:
            source_pool = [
                f"Query: {r.query}\nTitle: {r.title}\nURL: {r.url}\n"
                f"Snippet: {r.raw_content or r.snippet}"
                for r in results
            ]
            return await outline_then_parallel(
                self.summarizer,
                question=question,
                source_pool=source_pool,
                sources_tag="search_results",
                outline_system=_SUMMARY_OUTLINE_SYSTEM,
                section_system=_SUMMARY_SECTION_SYSTEM,
                outline_tag="RESEARCH_SUMMARY_OUTLINE_JSON",
                section_tag="RESEARCH_SUMMARY_TEXT",
                section_max_tokens=RESEARCH_SUMMARY_MAX_TOKENS,
                heading_prefix="### ",
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

    # detailed_cited 模式(benchmark 评测):长报告 + inline [URL] 引用 + References。
    _SYSTEM_DETAILED = (
        "You are an expert research report writer. Using ONLY the provided findings, write a "
        "comprehensive, well-structured research report in Markdown that thoroughly answers the question.\n"
        "Requirements:\n"
        "1. Organize into clear sections with `##` headings; be detailed and analytical "
        "(cover multiple relevant angles, explain causes/implications, not just list facts).\n"
        "2. Cite sources INLINE: right after each factual claim, put the supporting source URL(s) in "
        "square brackets, e.g. [https://example.com]. ONLY cite URLs that appear in the findings' "
        "`sources:` lines below. Do not invent URLs.\n"
        "3. End with a `## References` section listing every URL you cited.\n"
        "Use the same language as the user's original question."
    )

    # REPORT_PARALLEL prompts: one to plan the outline, one to write a single
    # section body (no heading / no references; those are added by the stitcher).
    _OUTLINE_SYSTEM = (
        "Plan an outline for a research report that answers the question using ONLY the findings. "
        "Split it into clear, non-overlapping sections. For each section give a title and the ids "
        "of the findings it should draw from (from the <findings> block). "
        'Return only JSON with this shape: {"sections": [{"title": "...", "source_ids": [1, 3]}]}'
    )
    _SECTION_SYSTEM = (
        "You are writing ONE section of a research report. Using ONLY the provided findings, write "
        "the body of the section named in <section_to_write> in Markdown. Be informative and "
        "analytical but concise — no filler, no repetition. "
        "Do NOT repeat the section title as a heading and do NOT add a references list. Cite sources "
        "inline: right after each factual claim put the supporting URL(s) from the findings' "
        "`sources:` lines in square brackets, e.g. [https://example.com]. Do not invent URLs. "
        "Use the same language as the question."
    )

    # REPORT_REVIEW prompts: one reviewer pass over the full drafted sections
    # (decides the final order and which sections are missing), one to write the
    # body of a single added section. The reviewer output IS the final outline —
    # there is no second outline call.
    _REVIEW_SYSTEM = (
        "You are reviewing a sectioned draft of a research report. You get the question and the "
        "draft's numbered sections. Produce the COMPLETE final outline: reorder the existing "
        "sections into a logical flow, and insert new sections (e.g. an introduction, a conclusion, "
        "or an important uncovered aspect) only where they genuinely improve the report. "
        'Return only JSON: {"sections": [{"id": 2}, {"id": 0, "title": "..."}, ...]} listing every '
        "slot of the final report in order — {\"id\": n} places existing section n, "
        "{\"id\": 0, \"title\": \"...\"} inserts a new section there. Include EVERY existing "
        "section exactly once; add at most 3 new sections. Use the same language as the question "
        "for new titles."
    )
    _ADDITION_SYSTEM = (
        "You are writing ONE additional section for a research report. Using the question and the "
        "existing sections provided, write the body of the section named in <section_to_write> in "
        "Markdown. It must ADD what the existing sections lack (introduce, conclude, or cover the "
        "named gap); do NOT repeat their content. Do NOT repeat the section title as a heading and "
        "do NOT add a references list. Cite sources inline only with URLs that already appear in "
        "the existing sections. Use the same language as the question."
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
        # Throwaway "warmup" question on purpose, not original_question -- see
        # Supervisor.warmup_kv_prefix: keeping the question out of segment 0 keeps
        # the primed prefix a question-agnostic constant shared across all runs.
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

    async def write(self, *, original_question: str, summaries: list[str],
                    summary_sources: list[list[str]] | None = None) -> str:
        # REPORT_PARALLEL: outline -> assign source summaries -> parallel sections.
        # Lossy (output changes); off by default; disabled in KV-reuse mode. Takes
        # precedence over REPORT_MODE (produces its own sectioned + cited markdown).
        if _env_flag("REPORT_PARALLEL") and not self.kv_reuse_separator:
            return await self._write_parallel(original_question, summaries, summary_sources)
        # 实时读 env(而非 import 期常量),runner 进程内设置也能生效
        if os.getenv("REPORT_MODE", REPORT_MODE) == "detailed_cited":
            return await self._write_detailed_cited(original_question, summaries, summary_sources)
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

    async def _write_detailed_cited(
        self, original_question: str, summaries: list[str],
        summary_sources: list[list[str]] | None,
    ) -> str:
        """benchmark 评测用:分章节长报告 + inline [URL] 引用。不走 KV 复用拼接
        (复用路径只在默认 answer 模式),每条 finding 带上它的来源 URL 供 writer 引用。"""
        srcs = summary_sources or [[] for _ in summaries]
        blocks = []
        for i, (summ, urls) in enumerate(zip(summaries, srcs), 1):
            uniq = list(dict.fromkeys(u for u in (urls or []) if u))
            src_line = "sources: " + (", ".join(uniq) if uniq else "(none)")
            blocks.append(f"[Finding {i}] {src_line}\n{summ}")
        findings_block = "\n\n".join(blocks)
        user = (
            f"<original_question>\n{original_question}\n</original_question>\n\n"
            f"<findings>\n{findings_block}\n</findings>"
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_DETAILED},
            {"role": "user", "content": user},
        ]
        return await self.llm.chat(
            messages,
            model=self.model,
            temperature=0.0,
            max_tokens=FINAL_REPORT_MAX_TOKENS,
            tag="FINAL_REPORT_MARKDOWN",
        )

    async def _write_parallel(
        self, original_question: str, summaries: list[str],
        summary_sources: list[list[str]] | None,
    ) -> str:
        """REPORT_PARALLEL: outline the report, write sections in parallel from
        their assigned findings, then append a deterministic References list."""
        srcs = summary_sources or [[] for _ in summaries]
        source_pool: list[str] = []
        for summ, urls in zip(summaries, srcs):
            uniq = list(dict.fromkeys(u for u in (urls or []) if u))
            src_line = "sources: " + (", ".join(uniq) if uniq else "(none)")
            source_pool.append(f"{src_line}\n{summ}")
        if not source_pool:
            source_pool = [""]
        section_max_tokens = max(1000, FINAL_REPORT_MAX_TOKENS // 2)
        # REPORT_REVIEW 默认开(2026-07-20 终审拍板);显式 REPORT_REVIEW=0 才关。
        if os.getenv("REPORT_REVIEW", "1").strip().lower() in {"1", "true", "yes"}:
            body = await self._write_reviewed(original_question, source_pool, section_max_tokens)
        else:
            body = await outline_then_parallel(
                self,
                question=original_question,
                source_pool=source_pool,
                sources_tag="findings",
                outline_system=self._OUTLINE_SYSTEM,
                section_system=self._SECTION_SYSTEM,
                outline_tag="FINAL_REPORT_OUTLINE_JSON",
                section_tag="FINAL_REPORT_MARKDOWN",
                section_max_tokens=section_max_tokens,
                heading_prefix="## ",
            )
        all_urls = list(dict.fromkeys(u for urls in srcs for u in (urls or []) if u))
        if all_urls:
            refs = "\n".join(f"- {u}" for u in all_urls)
            body = f"{body}\n\n## References\n\n{refs}"
        return body

    async def _write_reviewed(
        self, original_question: str, source_pool: list[str], section_max_tokens: int,
    ) -> str:
        """REPORT_REVIEW: outline -> sections -> one reviewer pass over the full
        section texts (final order + missing sections) -> added sections in
        parallel -> deterministic assembly. Every degradation path (reviewer
        JSON fails, bad ids, an addition call fails) falls back toward the
        plain outline_then_parallel result — existing content is never lost."""
        raw = await outline_and_sections(
            self,
            question=original_question,
            source_pool=source_pool,
            sources_tag="findings",
            outline_system=self._OUTLINE_SYSTEM,
            section_system=self._SECTION_SYSTEM,
            outline_tag="FINAL_REPORT_OUTLINE_JSON",
            section_tag="FINAL_REPORT_MARKDOWN",
            section_max_tokens=section_max_tokens,
        )
        if len(raw) == 1 and not raw[0][0]:  # outline-failed fallback: nothing to review
            return raw[0][1].strip() or "No relevant information was found."
        secs = [(t, (b or "").strip()) for t, b in raw]
        secs = [(t, b) for t, b in secs if b]
        if len(secs) < 2:  # a single section: no order to fix, skip the review pass
            return "\n\n".join(f"## {t}\n\n{b}" for t, b in secs) or "No relevant information was found."

        numbered = "\n\n".join(f"[{i}] ## {t}\n\n{b}" for i, (t, b) in enumerate(secs, 1))
        review_messages = [
            {"role": "system", "content": self._REVIEW_SYSTEM},
            {"role": "user", "content": (
                f"<question>\n{original_question}\n</question>\n\n"
                f"<sections>\n{numbered}\n</sections>"
            )},
        ]
        try:
            review, _ = await self.call_json(
                review_messages, ReportReview, tag="FINAL_REPORT_REVIEW_JSON",
            )
            items = review.sections
        except Exception:  # noqa: BLE001 - reviewer is an enhancement, never fatal
            items = []

        # Sanitize into final slots: dedupe ids, drop out-of-range, cap and
        # title-dedupe additions; existing sections the reviewer forgot are
        # appended at the end in their original relative order (zero loss).
        slots: list[tuple[str, object]] = []  # ("old", 1-based idx) | ("new", title)
        seen: set[int] = set()
        used_titles = {t.strip().lower() for t, _ in secs}
        new_count = 0
        for item in items:
            if item.id and 1 <= item.id <= len(secs) and item.id not in seen:
                seen.add(item.id)
                slots.append(("old", item.id))
            elif not item.id:
                title = (item.title or "").strip()
                if title and title.lower() not in used_titles and new_count < 3:
                    used_titles.add(title.lower())
                    slots.append(("new", title))
                    new_count += 1
        for i in range(1, len(secs) + 1):
            if i not in seen:
                slots.append(("old", i))

        async def _addition(title: str) -> str:
            messages = [
                {"role": "system", "content": self._ADDITION_SYSTEM},
                {"role": "user", "content": (
                    f"<question>\n{original_question}\n</question>\n\n"
                    f"<report_sections>\n{numbered}\n</report_sections>\n\n"
                    f"<section_to_write>\n{title}\n</section_to_write>"
                )},
            ]
            try:
                return await self.llm.chat(
                    messages, model=self.model, temperature=0.0,
                    max_tokens=section_max_tokens, tag="FINAL_REPORT_ADDITION_MARKDOWN",
                )
            except Exception:  # noqa: BLE001 - a failed addition is skipped, never fatal
                return ""

        new_titles = [t for kind, t in slots if kind == "new"]
        added_bodies = await asyncio.gather(*(_addition(t) for t in new_titles))
        parts: list[str] = []
        add_i = 0
        for kind, val in slots:
            if kind == "old":
                title, body = secs[val - 1]
            else:
                title, body = val, (added_bodies[add_i] or "").strip()
                add_i += 1
            if body:
                parts.append(f"## {title}\n\n{body}")
        return "\n\n".join(parts) or "No relevant information was found."


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
