"""What the API accepts and returns."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class SearchRequest(BaseModel):
    query: Text
    k: int = Field(default=5, ge=1, le=20)


class HitOut(BaseModel):
    path: str
    start_line: int
    end_line: int
    score: float
    text: str


class SearchResponse(BaseModel):
    hits: list[HitOut]


class AskRequest(BaseModel):
    question: Text
    stream: bool = False


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    abstain_reason: str | None
    truncated: bool
    citations: list[int]
    sources: list[str]
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    cached: str | None = None  # "exact" or "semantic" when it came from a cache
    degraded: bool = False  # the model failed: `chunks` holds what was retrieved instead
    error: str | None = None
    chunks: list[HitOut] = []


def answer_out(answer):
    """An `Answer` from the Phase 7 pipeline as the API's response."""
    return AskResponse(
        answer=answer.text,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason,
        truncated=answer.truncated,
        citations=list(answer.citations),
        sources=list(answer.sources),
        model=answer.model,
        input_tokens=answer.usage.input_tokens,
        output_tokens=answer.usage.output_tokens,
        cost_usd=answer.cost_usd,
        latency_ms=answer.latency_ms,
    )
