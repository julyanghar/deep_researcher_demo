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


class OutlineSection(BaseModel):
    """One section of a parallel-generation outline: a title plus the 1-based
    ids of the sources this section should draw from (SUMMARY_PARALLEL /
    REPORT_PARALLEL)."""

    title: str = ""
    source_ids: list[int] = Field(default_factory=list)


class Outline(BaseModel):
    """Outline for parallel section generation (summary or report)."""

    sections: list[OutlineSection] = Field(default_factory=list)


class ReviewItem(BaseModel):
    """One slot in the reviewed final outline (REPORT_REVIEW): id>0 places the
    existing section with that 1-based number; id==0 inserts a NEW section with
    the given title at this position."""

    id: int = 0
    title: str = ""


class ReportReview(BaseModel):
    """Reviewer output for REPORT_REVIEW: the complete final outline, in order,
    over existing (by id) and new (by title) sections."""

    sections: list[ReviewItem] = Field(default_factory=list)


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
