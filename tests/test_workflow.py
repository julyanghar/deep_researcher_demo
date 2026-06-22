import asyncio

from deep_researcher_demo.agents import FinalWriter, Researcher, Supervisor
from deep_researcher_demo.workflow import DeepResearchWorkflow
from fakes import StubChatClient, StubSearchProvider


def test_workflow_runs_multiple_rounds_with_stubs():
    llm = StubChatClient()
    workflow = DeepResearchWorkflow(
        supervisor=Supervisor(llm, "stub-model"),
        researcher=Researcher(llm, "stub-model", "stub-model"),
        final_writer=FinalWriter(llm, "stub-model"),
        search_provider=StubSearchProvider(),
        max_iterations=2,
        max_followups=2,
        max_concurrency=1,
        max_results=2,
    )
    result = asyncio.run(workflow.run("What are local LLM inference tradeoffs?"))
    assert len(result.summaries) == 4
    assert all(isinstance(summary, str) for summary in result.summaries)
    assert "Stub Deep Research Report" in result.final_report
    assert len(result.initial_research_questions) == 2


def test_workflow_falls_back_to_original_question_when_initial_decomposition_is_empty():
    llm = EmptyInitialQuestionsChatClient()
    workflow = DeepResearchWorkflow(
        supervisor=Supervisor(llm, "stub-model"),
        researcher=Researcher(llm, "stub-model", "stub-model"),
        final_writer=FinalWriter(llm, "stub-model"),
        search_provider=StubSearchProvider(),
        max_iterations=1,
        max_followups=2,
        max_concurrency=1,
        max_results=1,
    )

    result = asyncio.run(workflow.run("What are local LLM inference tradeoffs?"))

    assert result.initial_research_questions == ["What are local LLM inference tradeoffs?"]
    assert isinstance(result.summaries[0], str)
    assert "What are local LLM inference tradeoffs?" in result.summaries[0]


class EmptyInitialQuestionsChatClient(StubChatClient):
    async def chat(
        self, messages, *, model, temperature=0.0, max_tokens=None,
        store_generated_kv=False, tag=None,
    ):
        if tag == "INITIAL_RESEARCH_QUESTIONS_JSON":
            return '{"research_questions": []}'
        return await super().chat(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            store_generated_kv=store_generated_kv,
            tag=tag,
        )
