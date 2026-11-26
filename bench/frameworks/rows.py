"""Every row of the comparison, by name (Phase 11). The LlamaIndex candidates come from
`candidates.json`, which was written before any of them was run."""

import json
from pathlib import Path

CANDIDATES = json.loads((Path(__file__).parent / "candidates.json").read_text(encoding="utf-8"))

DEFAULT_CHUNKING = {
    "kind": "sentence",
    "chunk_size": 1024,
    "chunk_overlap": 200,
}  # LlamaIndex's own


def li(chunking=None, exclude_metadata=False, query_instruction=True, retriever="vector",
       faiss="flat_l2", lf=False):  # fmt: skip
    return {
        "framework": "llamaindex",
        "chunking": chunking or DEFAULT_CHUNKING,
        "exclude_metadata": exclude_metadata,
        "query_instruction": query_instruction,
        "retriever": retriever,
        "faiss": faiss,
        "normalize_newlines": lf,  # False: the reader's own behaviour (keeps CRLF)
    }


ROWS = {
    "cairn-bm25": {"framework": "cairn", "kind": "bm25"},
    "cairn-dense": {"framework": "cairn", "kind": "dense"},
    "cairn-hybrid": {"framework": "cairn", "kind": "hybrid"},
    "li-default": li(),
    "li-default-noprefix": li(query_instruction=False),
    "li-default-hnsw": li(faiss="hnsw"),
    # a control added after the first dev runs, not a tuning candidate: does a Windows checkout's
    # CRLF line endings (which Cairn's reader normalizes and LlamaIndex's does not) change anything?
    "li-default-lf": li(lf=True),
    "li-bm25-default": li(retriever="bm25"),
    "li-fusion-default": li(retriever="fusion"),
}
for candidate in CANDIDATES["chunking_candidates"]:
    ROWS["li-" + candidate["name"]] = li(candidate["chunking"], candidate["exclude_metadata"])


def with_selected(selected_name):
    """The rows that reuse the chunking chosen on dev (added after the selection)."""
    base = ROWS["li-" + selected_name]
    return {
        "li-bm25-selected": {**base, "retriever": "bm25"},
        "li-fusion-selected": {**base, "retriever": "fusion"},
    }
