"""The things a request needs, built once at startup from the Phase 7 parts."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cairn import config
from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import load_embedder
from cairn.generation.factory import build_llm, load_prices
from cairn.generation.packer import ContextPacker
from cairn.generation.pipeline import Answerer
from cairn.generation.prompts import load_prompt
from cairn.ingest.pipeline import read_chunks
from cairn.retrieval.hybrid import HybridRetriever
from cairn.serve.cache import LRUCache, SemanticCache
from cairn.serve.indexer import Indexer
from cairn.serve.jobs import JobRunner
from cairn.serve.live import LiveBM25, LiveDense, LiveIndex
from cairn.serve.manifest import Manifest
from cairn.serve.observe import RequestStats, TimedLLM, TimedRetriever
from cairn.serve.protect import CircuitBreaker, ConcurrencyLimit, GuardedLLM

log = logging.getLogger("cairn.serve.engine")

RETRIEVERS = ("hybrid", "bm25")


@dataclass(frozen=True)
class EngineConfig:
    data_dir: Path
    corpus_root: Path | None = None  # None: data_dir/raw/fastapi
    embedding_model: str = "bge-small"
    llm: str = "ollama"  # "none": retrieval only, no model attached
    retriever: str = "hybrid"  # or "bm25": no embedding model at all
    llm_base_url: str = "http://localhost:11434"
    model: str | None = "gemma4:e4b"
    llm_timeout: float = 120.0  # seconds; a service must not wait for a model for 15 minutes
    k: int = 5
    max_context_tokens: int = 1500
    max_tokens: int = 1024
    exact_cache_size: int = 500
    semantic_cache: bool = False  # off by default: see docs/results.md for why
    semantic_threshold: float = 0.92
    max_concurrent_retrievals: int = 4
    torch_threads: int = 1  # see docs/results.md: more threads per request means fewer requests/s
    retrieval_wait_s: float = 2.0
    breaker_failures: int = 3
    breaker_cooldown_s: float = 30.0


@dataclass(frozen=True)
class Engine:
    retriever: object
    llm: object
    model: str
    config: EngineConfig
    prices: dict | None
    live: LiveIndex | None = None
    indexer: Indexer | None = None
    jobs: JobRunner | None = None
    exact_cache: LRUCache | None = None
    semantic_cache: SemanticCache | None = None
    breaker: CircuitBreaker | None = None
    limit: ConcurrencyLimit | None = None
    source_root: Path | None = None  # where /source reads the files cited in answers
    extras: dict = field(default_factory=dict)

    @property
    def retrieval_only(self):
        return self.config.llm == "none"

    @property
    def generation(self):
        return self.live.generation if self.live is not None else 0

    def scope(self, k):
        """What a cached answer depends on: the index version, the number of chunks, the model."""
        return (self.generation, k, self.model)

    def make_answerer(self, stats=None):
        """A new Answerer per request: it keeps `last_answer`, so one must not be shared. The
        model client is forked for the same reason (its `last_usage` is per request)."""
        stats = stats if stats is not None else RequestStats()
        cfg = self.config
        fork = getattr(self.llm, "fork", None)
        llm = TimedLLM(fork() if fork else self.llm, stats)
        retriever = TimedRetriever(LimitedRetriever(self.retriever, self.limit), stats)
        costs = self.prices if self.prices and self.model in self.prices else None
        return Answerer(
            retriever,
            llm,
            packer=ContextPacker(cfg.max_context_tokens),
            template=load_prompt("answer"),
            k=cfg.k,
            max_tokens=cfg.max_tokens,
            costs=costs,
        )


class LimitedRetriever:
    """Takes a slot of the concurrency limit around each search (the CPU-heavy part)."""

    def __init__(self, retriever, limit):
        self._retriever = retriever
        self._limit = limit

    def search(self, query, k):
        if self._limit is None:
            return self._retriever.search(query, k)
        with self._limit:
            return self._retriever.search(query, k)


def set_torch_threads(n):
    """Torch uses every core for one small matrix product by default. With several requests at
    once the threads of different requests fight for the same cores, and total throughput falls
    (measured: 15 requests/s with 12 threads against 27 with 1). Does nothing without torch."""
    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(n)


class NoEmbedder:
    """Stands in for the embedding model when only BM25 is used: vectors of length 0."""

    dim = 0

    def embed_documents(self, texts):
        return np.zeros((len(texts), 0), dtype=np.float32)

    def embed_query(self, text):
        return np.zeros(0, dtype=np.float32)


def build_engine(cfg):
    if cfg.retriever not in RETRIEVERS:
        raise ValueError(f"retriever must be one of {RETRIEVERS}, it is {cfg.retriever!r}")
    if cfg.semantic_cache and cfg.retriever == "bm25":
        raise ValueError("the semantic cache needs the embedding model: use --retriever hybrid")
    data = Path(cfg.data_dir)
    chunks = read_chunks(data / "chunks.jsonl")
    dense_path = data / f"dense-{cfg.embedding_model}.npz"
    if cfg.retriever == "bm25":
        # no torch, no embedding model, no dense index: fits a host with 512 MB of memory
        embedder, dense_path = NoEmbedder(), None
        live = LiveIndex.build(chunks, np.zeros((len(chunks), 0), dtype=np.float32))
        retriever = LiveBM25(live)
    else:
        flat = FlatIndex.load(dense_path)
        if [c.id for c in chunks] != flat.doc_ids:
            raise ValueError("chunks.jsonl and the dense index do not match: rebuild the index")
        live = LiveIndex.build(chunks, flat.vectors)
        embedder = load_embedder(cfg.embedding_model)
        set_torch_threads(cfg.torch_threads)
        retriever = HybridRetriever(
            [LiveBM25(live), LiveDense(embedder, live)],
            depth=50,
            method="weighted",
            normalize="minmax",
        )

    exact = LRUCache(cfg.exact_cache_size)
    semantic = SemanticCache(embedder, cfg.semantic_threshold) if cfg.semantic_cache else None
    manifest = indexer = jobs = None
    root = Path(cfg.corpus_root) if cfg.corpus_root else data / "raw" / "fastapi"
    if root.is_dir():
        manifest = Manifest(data / "manifest.sqlite")
        indexer = Indexer(
            live,
            manifest,
            embedder,
            root,
            chunks_path=data / "chunks.jsonl",
            dense_path=dense_path,
        )
        if len(manifest) == 0:
            indexer.bootstrap()

        def reindex():
            report = indexer.reindex()
            exact.clear()  # entries of the old generation could not be found anyway; free them
            if semantic is not None:
                semantic.purge_except(live.generation)
            return report.as_dict()

        jobs = JobRunner(reindex)
    else:
        root = None
        log.warning("no corpus at the configured place: /ingest and /source are disabled")

    breaker = CircuitBreaker(cfg.breaker_failures, cfg.breaker_cooldown_s)
    if cfg.llm == "none":
        llm, model = None, "none"  # retrieval only: /ask answers with the retrieved passages
    else:
        llm, model = build_llm(
            cfg.llm, model=cfg.model, timeout=cfg.llm_timeout, base_url=cfg.llm_base_url
        )
        if hasattr(llm, "max_calls"):
            llm.max_calls = 10**12  # the fake's loop guard is for tests; a service runs on
        llm = GuardedLLM(llm, breaker)
    prices_path = config.PROJECT_ROOT / "eval" / "llm_prices.json"
    prices = load_prices(prices_path) if prices_path.is_file() else None
    return Engine(
        retriever,
        llm,
        model,
        cfg,
        prices,
        live=live,
        indexer=indexer,
        jobs=jobs,
        exact_cache=exact,
        semantic_cache=semantic,
        breaker=breaker,
        limit=ConcurrencyLimit(cfg.max_concurrent_retrievals, cfg.retrieval_wait_s),
        source_root=root,
    )


def warm_up(engine):
    """One search, so lazy loading (the embedder, the first multiply) happens before a user."""
    engine.retriever.search("how do I declare a path parameter", 1)
