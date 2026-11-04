"""Builds the parts the Phase 7 scripts share. Provided, not part of the exercise.

It uses your retry, client and pipeline modules, so those have to work first.
"""

import json
from pathlib import Path

from cairn.dense.flat_index import FlatIndex
from cairn.dense.models import load_embedder
from cairn.dense.retriever import DenseRetriever
from cairn.generation.abstain import top_score
from cairn.generation.fake import FakeLLM
from cairn.ingest.pipeline import read_chunks
from cairn.retrieval.bm25 import BM25Retriever
from cairn.retrieval.hybrid import HybridRetriever
from cairn.retrieval.index import InvertedIndex
from cairn.retrieval.rerank import RerankingRetriever
from cairn.retrieval.rerank_models import RERANKERS, load_model, load_reranker

RETRIEVERS = ("bm25", "dense", "hybrid", "rerank")
LLMS = ("anthropic", "ollama", "fake")
CONFIDENCES = ("top-score", "dense")


def demo_responder(system, messages):
    """What `--llm fake` answers: a judge gets a fixed verdict, anything else a cited sentence."""
    if "strict grader" in system.lower():
        return '{"correct": 1, "faithful": 1, "reason": "demo"}'
    return "This is a demo answer [1]."


def obedient_responder(system, messages):
    """A model that follows any instruction it finds in the prompt (for the injection script)."""
    text = messages[-1]["content"]
    if "PWNED-7431" in text:
        return "PWNED-7431"
    return demo_responder(system, messages)


def build_retrievers(name, *, chunks_path, bm25_path, dense_path, model="bge-small",
                     reranker="minilm-ce", depth=50, rerank_depth=10):  # fmt: skip
    """Returns (retriever, dense_retriever): the one named, and the plain dense one."""
    if name not in RETRIEVERS:
        raise ValueError(f"retriever must be one of {RETRIEVERS}, it is currently {name!r}")
    for path, script in ((chunks_path, "ingest"), (bm25_path, "build_index"),
                         (dense_path, "embed_chunks")):  # fmt: skip
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} not found (run scripts/{script}.py)")
    chunks = read_chunks(chunks_path)
    bm25 = BM25Retriever(InvertedIndex.load(bm25_path), chunks)
    dense = DenseRetriever(load_embedder(model), FlatIndex.load(dense_path), chunks)
    hybrid = HybridRetriever([bm25, dense], depth=depth, method="weighted", normalize="minmax")
    if name == "rerank":
        scorer = load_reranker(reranker, batch_size=4, model=load_model(RERANKERS[reranker]))
        return RerankingRetriever(hybrid, scorer, depth=rerank_depth), dense
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}[name], dense


def build_confidence(kind, dense):
    """`top-score`: the retriever's own best score. `dense`: the best cosine of the plain dense
    retriever, a score that means the same thing for every question whatever retriever answers."""
    if kind == "dense":
        return lambda question, hits: top_score(dense.search(question, 1))
    return None  # the Answerer's default: the best hit score


def build_llm(kind, *, model=None, effort="low", timeout=None, base_url="http://localhost:11434",
              responder=None):  # fmt: skip
    """Returns (client, model name). Real clients come wrapped in your RetryingClient.

    `timeout` None means 60 s for the Anthropic API and 900 s for Ollama: a local model on a CPU
    needs minutes just to read a 3,000-token prompt (about 135 s measured), and a timeout would
    only make `RetryingClient` start the same slow call again."""
    from cairn.generation.clients import AnthropicClient, OllamaClient
    from cairn.generation.retry import RetryingClient

    if timeout is None:
        timeout = 900.0 if kind == "ollama" else 60.0
    if kind == "fake":
        llm = FakeLLM(responder=responder or demo_responder)
        return llm, llm.model
    if kind == "ollama":
        model = model or "llama3.2"
        # thinking off: a thinking model spends max_tokens on hidden reasoning and can return an
        # empty answer (seen with gemma4:e4b, 40 tokens, done_reason "length")
        client = OllamaClient(model, base_url=base_url, timeout=timeout, think=False)
        return RetryingClient(client), model
    model = model or "claude-opus-5-5"
    return RetryingClient(AnthropicClient(model, timeout=timeout, effort=effort)), model


def load_prices(path):
    """{model: (dollars per million input tokens, dollars per million output tokens)}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        model: (float(p[0]), float(p[1])) for model, p in data.items() if not model.startswith("_")
    }
