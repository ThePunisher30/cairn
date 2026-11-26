"""How many gold answers does each chunker keep whole? (Phase 11)

For every chunking, the share of the dev questions' gold spans that lie entirely inside one chunk
(the Phase 2 measure, `cairn.evaluation.integrity`). A span cut in two can still be found by
retrieving both halves, but a single retrieved chunk can no longer contain the whole answer.
No embedding is done, only the reading and the splitting, so this is quick.

    python bench/frameworks/integrity_compare.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from common import CORPUS, QUESTIONS, write_json  # noqa: E402
from llamaindex_build import _documents, _parse  # noqa: E402
from rows import ROWS  # noqa: E402

from cairn.evaluation.adapters import span_lines  # noqa: E402
from cairn.evaluation.integrity import gold_integrity  # noqa: E402
from cairn.evaluation.questions import load_questions  # noqa: E402
from cairn.ingest.documents import load_documents  # noqa: E402
from cairn.ingest.pipeline import default_chunkers  # noqa: E402
from cairn.types import Hit  # noqa: E402

CONFIGS = ["li-default", "li-sent-512-50", "li-sent-256-32-nometa", "li-md-code-30"]


def main():
    questions = load_questions(QUESTIONS, split="dev")
    report = {}

    chunkers = default_chunkers()
    chunks = []
    for doc in load_documents(CORPUS):
        chunks.extend(chunkers[doc.kind].split(doc))
    report["cairn"] = {"chunks": len(chunks), "integrity": gold_integrity(chunks, questions)}

    documents, path_of, _ = _documents(CORPUS, None)
    texts = {path_of[d.doc_id]: d.text for d in documents}
    for name in CONFIGS:
        cfg = ROWS[name]
        nodes = _parse(documents, path_of, cfg["chunking"], cfg["exclude_metadata"])
        hits, missing = [], 0
        for node in nodes:
            if node.start_char_idx is None:
                missing += 1
                continue
            path = path_of[node.ref_doc_id]
            first, last = span_lines(texts[path], node.start_char_idx, node.end_char_idx)
            hits.append(Hit(path, first, last))
        report[name] = {
            "chunks": len(nodes),
            "without_offsets": missing,
            "integrity": gold_integrity(hits, questions),
        }
    for name, row in report.items():
        overall = row["integrity"]["overall"]
        cats = {c: round(v["span_rate"], 2) for c, v in row["integrity"]["by_category"].items()}
        print(
            f"{name:24} chunks {row['chunks']:>5}  spans whole {overall['span_rate']:.3f}  {cats}"
        )
    write_json(Path(__file__).parent / "results" / "integrity.json", report)


if __name__ == "__main__":
    main()
