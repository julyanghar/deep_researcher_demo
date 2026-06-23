"""Shared schemas for the simplified deep researcher workflow."""

from typing import Literal

from pydantic import BaseModel, Field


class InitialResearchQuestions(BaseModel):
    """Supervisor output for initial research sub-questions."""

    research_questions: list[str] = Field(default_factory=list)


class QueryPlan(BaseModel):
    """Researcher output for web search queries."""

    queries: list[str] = Field(default_factory=list)


class SupervisorDecision(BaseModel):
    """Supervisor decision after reviewing accumulated research summaries."""

    status: Literal["continue", "complete"]
    followup_questions: list[str] = Field(default_factory=list)
    reason: str = ""


class SearchResult(BaseModel):
    """Normalized search result from a web search provider."""

    query: str
    title: str = ""
    url: str = ""
    snippet: str = ""
    raw_content: str | None = None


class WorkflowResult(BaseModel):
    """Final result returned by the workflow."""

    original_question: str
    initial_research_questions: list[str] = Field(default_factory=list)
    summaries: list[str] = Field(default_factory=list)
    supervisor_reasons: list[str] = Field(default_factory=list)
    # Per-round supervisor decision JSON reused as the r_t KV trace (populated
    # only when SUPERVISOR_REASONING is on); kept for downstream trace analysis.
    supervisor_reasonings: list[str] = Field(default_factory=list)
    final_report: str
