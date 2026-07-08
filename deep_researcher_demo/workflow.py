"""Workflow loop for the simplified deep researcher."""

import asyncio

from deep_researcher_demo.agents import (
    SUPERVISOR_REASONING,
    FinalWriter,
    Researcher,
    Supervisor,
)
from deep_researcher_demo.progress import (
    NullProgressReporter,
    ProgressEvent,
    ProgressReporter,
    format_list,
)
from deep_researcher_demo.schemas import WorkflowResult


class DeepResearchWorkflow:
    """Plain Python implementation of the simplified deep research workflow."""

    def __init__(
        self,
        *,
        supervisor: Supervisor,
        researcher: Researcher,
        final_writer: FinalWriter,
        search_provider,
        max_iterations: int = 3,
        max_followups: int = 3,
        max_queries_per_researcher: int = 3,
        max_concurrency: int = 3,
        max_results: int = 5,
        min_rounds: int = 0,
        reporter: ProgressReporter | None = None,
        output_path: str | None = None,
    ) -> None:
        self.supervisor = supervisor
        self.researcher = researcher
        self.final_writer = final_writer
        self.search_provider = search_provider
        self.max_iterations = max_iterations
        self.max_followups = max_followups
        # Exp A: force at least this many research rounds (override an early
        # "complete") so trajectories accumulate multiple summary segments. 0 = off.
        self.min_rounds = min_rounds
        self.max_queries_per_researcher = max_queries_per_researcher
        self.max_concurrency = max_concurrency
        self.max_results = max_results
        self.reporter = reporter or NullProgressReporter()
        self.output_path = output_path

    async def run(self, original_question: str) -> WorkflowResult:
        self.reporter.emit(
            ProgressEvent(
                "starting_research",
                f"Starting research: {original_question}",
                {"original_question": original_question},
            )
        )
        if (getattr(self.supervisor, "kv_reuse_separator", "")
                or getattr(self.final_writer, "kv_reuse_separator", "")):
            # KV-reuse mode: prime the constant prompt prefixes of the
            # supervisor-decision and final-report roles so that the blend
            # lookup (which requires contiguous hits from token 0) can reach
            # the researcher-summary segments on their very first use.
            # Gate on EITHER role's separator (each warmup_kv_prefix self-guards
            # on its own separator): otherwise a writer-reuse-only config (e.g.
            # supervisor prefill + writer reuse) would skip the writer warmup and
            # silently break writer reuse.
            await asyncio.gather(
                self.supervisor.warmup_kv_prefix(self.max_followups),
                self.final_writer.warmup_kv_prefix(),
            )
        initial = await self.supervisor.initialize_question(
            original_question,
            max_questions=self.max_followups,
        )
        initial_questions = initial.research_questions or [original_question]
        self.reporter.emit(
            ProgressEvent(
                "initial_questions",
                f"{len(initial_questions)} questions: {format_list(initial_questions)}",
                {"research_questions": initial_questions},
            )
        )
        pending_questions = initial_questions
        seen_questions = set(initial_questions)
        summaries: list[str] = []
        summary_sources: list[list[str]] = []  # 与 summaries 对齐:每条的来源 URL
        supervisor_reasons: list[str] = []
        # r_t trace: each round's decision JSON, stored as a reusable KV segment
        # and interleaved into later decide() contexts when SUPERVISOR_REASONING
        # is on. Off -> stays empty -> summaries-only baseline.
        reasonings: list[str] = []

        for iteration in range(1, self.max_iterations + 1):
            if not pending_questions:
                break

            current_questions = pending_questions[: self.max_followups]
            self.reporter.emit(
                ProgressEvent(
                    "iteration_start",
                    f"iteration {iteration}/{self.max_iterations}, questions {len(current_questions)}",
                    {"iteration": iteration, "question_count": len(current_questions)},
                )
            )
            round_results = await self._run_researchers(current_questions)
            round_summaries = [summary for summary, _ in round_results]
            round_backup_urls = [url for _, urls in round_results for url in urls]
            summaries.extend(round_summaries)
            # 与 summaries 一一对齐:每条 summary 的来源 URL,供 detailed_cited 报告模式引用
            summary_sources.extend([list(urls) for _, urls in round_results])

            # max_followups 是：supervisor 每一轮最多可以提出多少个后续研究问题。
            decision, round_reasoning = await self.supervisor.decide(
                original_question=original_question,
                summaries=summaries,
                reasonings=reasonings,
                max_followups=self.max_followups,
                store_generated_kv=SUPERVISOR_REASONING,
            )
            if SUPERVISOR_REASONING:
                # Append the stored r_t so the next round interleaves it. Off ->
                # reasonings stays empty -> identical to the summaries-only path.
                reasonings.append(round_reasoning)
            supervisor_reasons.append(decision.reason)
            if decision.status == "continue":
                decision_message = (
                    f"continue: {decision.reason}; followups: {format_list(decision.followup_questions)}"
                )
            else:
                decision_message = f"complete: {decision.reason}"
            self.reporter.emit(
                ProgressEvent(
                    "supervisor_decision",
                    decision_message,
                    {
                        "status": decision.status,
                        "reason": decision.reason,
                        "followup_questions": decision.followup_questions,
                        "backup_url_count": len(round_backup_urls),
                    },
                )
            )

            # Exp A: force multi-round. Don't stop on "complete" until we've done
            # min_rounds; if forced past a complete with no follow-ups, deepen with
            # a generic probe so the extra round still yields a real new summary.
            forced = iteration < self.min_rounds
            if decision.status == "complete" and not forced:
                break

            pending_questions = []
            for question in decision.followup_questions:
                if question not in seen_questions:
                    seen_questions.add(question)
                    pending_questions.append(question)
            if forced and not pending_questions:
                probe = self._forced_followup(original_question, seen_questions)
                if probe:
                    seen_questions.add(probe)
                    pending_questions.append(probe)

        self.reporter.emit(
            ProgressEvent(
                "final_report",
                f"Writing final report from {len(summaries)} summaries",
                {"summary_count": len(summaries)},
            )
        )
        final_report = await self.final_writer.write(
            original_question=original_question,
            summaries=summaries,
            summary_sources=summary_sources,
        )
        self.reporter.emit(
            ProgressEvent(
                "completed",
                f"Completed with {len(summaries)} summaries; output: {self.output_path or 'stdout'}",
                {"summary_count": len(summaries), "output_path": self.output_path},
            )
        )
        return WorkflowResult(
            original_question=original_question,
            initial_research_questions=initial_questions,
            summaries=summaries,
            supervisor_reasons=supervisor_reasons,
            supervisor_reasonings=reasonings,
            final_report=final_report,
        )

    # Generic deepening probes, used only when min_rounds forces another round but
    # the supervisor offered no follow-ups. Content-bearing so the extra round
    # produces a real (non-degenerate) summary segment rather than an empty one.
    _FORCED_PROBES = (
        "What are the main limitations, caveats, or sources of uncertainty regarding: {q}",
        "What are notable counterarguments, alternative views, or contradicting evidence on: {q}",
        "What recent developments, data, or concrete examples are most relevant to: {q}",
    )

    def _forced_followup(self, original_question: str, seen: set[str]) -> str | None:
        for tmpl in self._FORCED_PROBES:
            cand = tmpl.format(q=original_question)
            if cand not in seen:
                return cand
        return None

    async def _run_researchers(self, questions: list[str]) -> list[tuple[str, list[str]]]:
        semaphore = asyncio.Semaphore(max(1, self.max_concurrency))

        async def run_one(question: str) -> tuple[str, list[str]]:
            async with semaphore:
                self.reporter.emit(
                    ProgressEvent(
                        "researcher_start",
                        f"Researcher started: {question}",
                        {"question": question},
                    )
                )
                return await self.researcher.research(
                    question,
                    self.search_provider,
                    self.max_results,
                    max_queries=self.max_queries_per_researcher,
                    reporter=self.reporter,
                )

        return list(await asyncio.gather(*(run_one(question) for question in questions)))
