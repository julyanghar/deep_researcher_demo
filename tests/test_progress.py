import asyncio

from deep_researcher_demo.agents import FinalWriter, Researcher, Supervisor
from deep_researcher_demo.progress import MemoryProgressReporter, NullProgressReporter
from deep_researcher_demo.workflow import DeepResearchWorkflow
from fakes import StubChatClient, StubSearchProvider


def test_researcher_emits_progress_events():
    llm = StubChatClient()
    reporter = MemoryProgressReporter()
    researcher = Researcher(llm, "stub-model", "stub-model")

    asyncio.run(
        researcher.research(
            "What evidence is strongest?",
            StubSearchProvider(),
            max_results=2,
            max_queries=2,
            reporter=reporter,
        )
    )

    steps = [event.step for event in reporter.events]
    assert "queries_planned" in steps
    assert "subquery_start" in steps
    assert "subquery_search_complete" in steps
    assert "subquery_summary_complete" in steps
    assert "search_complete" in steps
    assert "summary_complete" in steps


def test_workflow_emits_expected_progress_sequence():
    llm = StubChatClient()
    reporter = MemoryProgressReporter()
    workflow = DeepResearchWorkflow(
        supervisor=Supervisor(llm, "stub-model"),
        researcher=Researcher(llm, "stub-model", "stub-model"),
        final_writer=FinalWriter(llm, "stub-model"),
        search_provider=StubSearchProvider(),
        max_iterations=1,
        max_followups=1,
        max_concurrency=1,
        max_results=1,
        reporter=reporter,
        output_path="outputs/example_report.md",
    )

    asyncio.run(workflow.run("What are local LLM inference tradeoffs?"))

    steps = [event.step for event in reporter.events]
    assert steps[0] == "starting_research"
    assert "initial_questions" in steps
    assert "iteration_start" in steps
    assert "supervisor_decision" in steps
    assert steps[-1] == "completed"


def test_null_progress_reporter_is_quiet():
    reporter = NullProgressReporter()
    assert reporter.emit is not None
