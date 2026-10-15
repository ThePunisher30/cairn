from dataclasses import dataclass
from typing import Protocol

CATEGORIES = ("lexical", "semantic", "multi_chunk", "code")
SPLITS = ("dev", "test")


@dataclass(frozen=True)
class Span:
    path: str
    start_line: int
    end_line: int

    def __post_init__(self):
        if self.start_line < 1:
            raise ValueError(f"start_line must be >= 1, it is currently {self.start_line}")
        if self.end_line < self.start_line:
            raise ValueError(f"end_line must be >= start_line, it is currently {self.end_line}")


@dataclass(frozen=True)
class Hit:
    path: str
    start_line: int
    end_line: int
    score: float = 0.0
    text: str = ""


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


@dataclass(frozen=True)
class Question:
    id: str
    category: str
    split: str
    text: str
    gold: tuple[Span, ...]


@dataclass(frozen=True)
class Document:
    path: str  # relative to the corpus root, forward slashes
    text: str  # the whole file
    kind: str  # "prose" (markdown) or "code" (python)

    @property
    def lines(self) -> list[str]:
        if not self.text:
            return []
        parts = self.text.split("\n")
        if parts[-1] == "":
            parts.pop()
        return parts


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    path: str
    start_line: int
    end_line: int
    kind: str
    content_hash: str
    token_count: int
    heading_path: tuple[str, ...] = ()

    @property
    def index_text(self) -> str:
        if self.heading_path:
            return " > ".join(self.heading_path) + "\n" + self.text
        return self.text


class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...
