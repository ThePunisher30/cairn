"""The things a request needs, built once at startup from the Phase 7 parts."""

from dataclasses import dataclass
from pathlib import Path

from cairn import config
from cairn.generation.factory import build_llm, build_retrievers, load_prices
from cairn.generation.packer import ContextPacker
from cairn.generation.pipeline import Answerer
from cairn.generation.prompts import load_prompt


@dataclass(frozen=True)
class EngineConfig:
    data_dir: Path
    retriever: str = "hybrid"
    embedding_model: str = "bge-small"
    llm: str = "ollama"
    model: str | None = "gemma4:e4b"
    k: int = 5
    max_context_tokens: int = 1500
    max_tokens: int = 1024


@dataclass(frozen=True)
class Engine:
    retriever: object
    llm: object
    model: str
    config: EngineConfig
    prices: dict | None

    def make_answerer(self):
        """A new Answerer per request: it keeps `last_answer`, so one must not be shared."""
        cfg = self.config
        costs = self.prices if self.prices and self.model in self.prices else None
        return Answerer(
            self.retriever,
            self.llm,
            packer=ContextPacker(cfg.max_context_tokens),
            template=load_prompt("answer"),
            k=cfg.k,
            max_tokens=cfg.max_tokens,
            costs=costs,
        )


def build_engine(cfg):
    retriever, _dense = build_retrievers(
        cfg.retriever,
        chunks_path=cfg.data_dir / "chunks.jsonl",
        bm25_path=cfg.data_dir / "bm25.pkl",
        dense_path=cfg.data_dir / f"dense-{cfg.embedding_model}.npz",
        model=cfg.embedding_model,
    )
    llm, model = build_llm(cfg.llm, model=cfg.model)
    prices_path = config.PROJECT_ROOT / "eval" / "llm_prices.json"
    prices = load_prices(prices_path) if prices_path.is_file() else None
    return Engine(retriever, llm, model, cfg, prices)


def warm_up(engine):
    """One search, so lazy loading (the embedder, the first multiply) happens before a user."""
    engine.retriever.search("how do I declare a path parameter", 1)
