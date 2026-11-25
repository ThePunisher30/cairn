"""What the API accepts and returns."""

from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from cairn.generation.citations import source_url

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


class SourceRef(BaseModel):
    number: int
    path: str
    start_line: int
    end_line: int
    url: str  # the same lines on GitHub, at the pinned tag


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
    references: list[SourceRef] = []  # one per citation (or per retrieved chunk if degraded)


def references_for(answer):
    """The cited chunks of an `Answer` as structured references (the UI makes chips of them)."""
    by_number = {chunk.number: chunk.hit for chunk in answer.context.chunks}
    refs = []
    for number in answer.citations:
        hit = by_number.get(number)
        if hit is not None:
            refs.append(make_ref(number, hit))
    return refs


def make_ref(number, hit):
    return SourceRef(
        number=number,
        path=hit.path,
        start_line=hit.start_line,
        end_line=hit.end_line,
        url=source_url(hit.path, hit.start_line, hit.end_line),
    )


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
        references=references_for(answer),
    )
