"""Run one row of the comparison on the dev questions and write its result file (Phase 11).

    python bench/frameworks/run_row.py --row li-default
    python bench/frameworks/run_row.py --row cairn-hybrid --limit-files 30     # a quick smoke test

Each row runs in its own process so that its peak memory is its own. The test split is refused
here: it has a script of its own (`run_test.py`) that checks the freeze first.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))  # bench/ is not a package

from common import (  # noqa: E402
    BGE,
    CORPUS,
    QUESTIONS,
    RESULTS,
    K,
    peak_memory_mb,
    rss_mb,
    say,
    write_json,
)
from rows import ROWS  # noqa: E402

from cairn.evaluation.adapters import CharSpanRetriever  # noqa: E402
from cairn.evaluation.harness import run_retriever, summarize  # noqa: E402
from cairn.evaluation.questions import load_questions  # noqa: E402
from cairn.tokens import count_tokens  # noqa: E402

WARMUP = 3


def chunk_stats(texts):
    """How big the text that was embedded is, in three units, and how much the model cannot read."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BGE)
    bge = [len(ids) for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]]
    cairn = [count_tokens(t) for t in texts]
    chars = [len(t) for t in texts]
    return {
        "chunks": len(texts),
        "chars_mean": round(statistics.fmean(chars), 1),
        "chars_max": max(chars),
        "cairn_tokens_mean": round(statistics.fmean(cairn), 1),
        "cairn_tokens_max": max(cairn),
        "bge_tokens_mean": round(statistics.fmean(bge), 1),
        "bge_tokens_max": max(bge),
        "share_over_512_bge_tokens": round(sum(n > 512 for n in bge) / len(bge), 4),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run one comparison row on the dev questions.")
    parser.add_argument("--row", required=True, choices=sorted(ROWS))
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--limit-files", type=int, default=None, help="smoke test on N files")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    args = parser.parse_args(argv)
    if args.split == "test":
        print("error: the test split is run by run_test.py, after the freeze", file=sys.stderr)
        return 1

    cfg = ROWS[args.row]
    say(f"[{args.row}] importing")
    began = time.perf_counter()
    if cfg["framework"] == "llamaindex":
        import faiss  # noqa: F401
        import llama_index.core  # noqa: F401
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: F401
        from llama_index.vector_stores.faiss import FaissVectorStore  # noqa: F401
    import sentence_transformers  # noqa: F401

    import_seconds = time.perf_counter() - began
    rss_after_import = rss_mb()

    say(f"[{args.row}] building")
    if cfg["framework"] == "cairn":
        from cairn_build import build_cairn

        retriever, info = build_cairn(cfg["kind"], args.corpus, args.limit_files)
        located = 0
    else:
        from llamaindex_build import build_llamaindex

        retrieve, texts, info = build_llamaindex(cfg, args.corpus, args.limit_files)
        retriever = CharSpanRetriever(retrieve, texts, locate_missing=True)
    rss_after_build = rss_mb()
    texts_embedded = info.pop("embed_texts")
    info.pop("embedder", None)

    questions = load_questions(QUESTIONS, split=args.split)
    for question in questions[:WARMUP]:  # lazy initialisation must not count as query time
        retriever.search(question.text, K)
    say(f"[{args.row}] asking {len(questions)} questions")
    results = run_retriever(retriever, questions, K)
    summary = summarize(results, ks=(1, 5, K))
    located = getattr(retriever, "located", 0)

    top5 = [sum(count_tokens(h.text) for h in r.hits[:5]) for r in results]
    per_question = [
        {
            "id": r.question.id,
            "category": r.question.category,
            "gold": [[s.path, s.start_line, s.end_line] for s in r.question.gold],
            "hits": [[h.path, h.start_line, h.end_line] for h in r.hits],
        }
        for r in results
    ]
    record = {
        "row": args.row,
        "split": args.split,
        "config": cfg,
        "limit_files": args.limit_files,
        "build": info,
        "chunk_stats": chunk_stats(texts_embedded),
        "top5_cairn_tokens_mean": round(statistics.fmean(top5), 1),
        "nodes_placed_by_text_search": located,
        "import_seconds": round(import_seconds, 2),
        "rss_after_import_mb": round(rss_after_import, 1),
        "rss_after_build_mb": round(rss_after_build, 1),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "summary": summary,
        "per_question": per_question,
        "versions": {"python": platform.python_version(), "platform": platform.platform()},
    }
    out = (
        args.out or RESULTS / f"{args.split}-{args.row}{'-smoke' if args.limit_files else ''}.json"
    )
    write_json(out, record)
    o = summary["overall"]
    say(
        f"[{args.row}] chunks {info['chunks']}, build {info['build_seconds']:.1f}s, "
        f"hit@5 {o['hit@5']:.3f} mrr {o['mrr']:.3f} p50 {o['latency_p50_ms']:.1f}ms "
        f"peak {record['peak_memory_mb']:.0f}MB -> {out.name}"
    )
    print(json.dumps({"row": args.row, "hit@5": o["hit@5"], "mrr": o["mrr"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
