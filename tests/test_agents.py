import asyncio

from deep_researcher_demo.agents import Researcher, Supervisor, dedupe_urls, extract_urls
from deep_researcher_demo.schemas import SearchResult
from fakes import StubChatClient, StubSearchProvider


def test_supervisor_initializes_research_questions():
    llm = StubChatClient()
    supervisor = Supervisor(llm, "stub-model")
    result = asyncio.run(
        supervisor.initialize_question("What is local LLM inference?", max_questions=2)
    )
    assert len(result.research_questions) == 2
    assert all("local LLM inference" in question for question in result.research_questions)


def test_supervisor_initial_question_decomposition_respects_limit():
    llm = StubChatClient()
    supervisor = Supervisor(llm, "stub-model")
    result = asyncio.run(
        supervisor.initialize_question("What is local LLM inference?", max_questions=1)
    )
    assert len(result.research_questions) == 1


def test_supervisor_decides_continue_then_complete():
    llm = StubChatClient()
    supervisor = Supervisor(llm, "stub-model")
    first = asyncio.run(
        supervisor.decide(original_question="topic", summaries=[], max_followups=2)
    )
    second = asyncio.run(
        supervisor.decide(original_question="topic", summaries=[], max_followups=2)
    )
    assert first.status == "continue"
    assert len(first.followup_questions) == 2
    assert second.status == "complete"


def test_researcher_plans_search_and_summarizes():
    llm = StubChatClient()
    search_provider = StubSearchProvider()
    researcher = Researcher(llm, "stub-model", "stub-model")
    summary, backup_urls = asyncio.run(
        researcher.research(
            "What evidence is strongest?",
            search_provider,
            max_results=2,
            max_queries=2,
        )
    )
    assert isinstance(summary, str)
    assert "Stub compressed context" in summary
    assert backup_urls
    assert len(backup_urls) == len(set(backup_urls))
    assert llm.research_summary_calls == 2
    assert len(search_provider.calls) == 2
    assert all(len(queries) == 1 for queries, _ in search_provider.calls)
    assert all(max_results == 2 for _, max_results in search_provider.calls)


def test_researcher_combines_all_sub_query_summaries():
    llm = StubChatClient()
    researcher = Researcher(llm, "stub-model", "stub-model")
    summary, backup_urls = asyncio.run(
        researcher.research(
            "What evidence is strongest?",
            StubSearchProvider(),
            max_results=1,
            max_queries=2,
        )
    )

    assert isinstance(summary, str)
    assert "Stub compressed context for What evidence is strongest? overview" in summary
    assert "Stub compressed context for What evidence is strongest? evidence" in summary
    assert backup_urls


def test_researcher_summarize_results_returns_plain_text():
    llm = StubChatClient()
    researcher = Researcher(llm, "stub-model", "stub-model")

    summary = asyncio.run(
        researcher.summarize_results(
            question="What evidence is strongest?",
            queries=["What evidence is strongest? overview"],
            results=[],
        )
    )

    assert isinstance(summary, str)
    assert "Stub compressed context" in summary
    assert "{" not in summary


def test_researcher_plan_queries_respects_limit():
    llm = StubChatClient()
    researcher = Researcher(llm, "stub-model", "stub-model")

    plan = asyncio.run(researcher.plan_queries("What evidence is strongest?", max_queries=1))

    assert len(plan.queries) == 1


def test_search_result_url_helpers_filter_and_dedupe():
    results = [
        SearchResult(query="q", url="https://example.com/a"),
        SearchResult(query="q", url=""),
        SearchResult(query="q", url=" https://example.com/a "),
        SearchResult(query="q", url="https://example.com/b"),
    ]

    assert extract_urls(results) == [
        "https://example.com/a",
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert dedupe_urls(extract_urls(results)) == [
        "https://example.com/a",
        "https://example.com/b",
    ]
