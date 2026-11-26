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


SELECTED = json.loads((Path(__file__).parent / "selected.json").read_text(encoding="utf-8"))[
    "selected"
]
ROWS["li-bm25-selected"] = {**ROWS["li-" + SELECTED], "retriever": "bm25"}
ROWS["li-fusion-selected"] = {**ROWS["li-" + SELECTED], "retriever": "fusion"}

# Added after the first fusion rows, to diagnose them (not tuning candidates). With the FAISS L2
# index of LlamaIndex's own documentation, the vector retriever returns nodes best first but with
# the raw L2 distance as the score (smaller is better); QueryFusionRetriever re-sorts by score,
# largest first, and so reverses the dense ranking. An inner-product index has larger-is-better
# scores. Same vectors (they are unit length), same chunks, only the score direction differs.
ROWS["li-fusion-default-ip"] = {**ROWS["li-fusion-default"], "faiss": "flat_ip"}
ROWS["li-fusion-selected-ip"] = {**ROWS["li-fusion-selected"], "faiss": "flat_ip"}
