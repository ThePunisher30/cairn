import dataclasses
from dataclasses import dataclass

from cairn.ingest.chunkers import MarkdownChunker, PythonChunker
from cairn.ingest.documents import load_documents
from cairn.jsonl import read_jsonl, write_jsonl
from cairn.types import Chunk


@dataclass(frozen=True)
class IngestStats:
    documents: int
    chunks: int
    by_kind: dict
    mean_tokens: float
    max_tokens: int
    over_limit: int
    token_limit: int
    histogram: dict


def default_chunkers():
    return {"prose": MarkdownChunker(300), "code": PythonChunker()}


def token_histogram(chunks, edges=(50, 100, 200, 300, 400, 512)):
    labels = [f"<={edges[0]}"]
    labels += [f"{low + 1}-{high}" for low, high in zip(edges, edges[1:], strict=False)]
    labels.append(f">{edges[-1]}")
    counts = dict.fromkeys(labels, 0)
    for chunk in chunks:
        for edge, label in zip(edges, labels, strict=False):
            if chunk.token_count <= edge:
                counts[label] += 1
                break
        else:  # bigger than every edge
            counts[labels[-1]] += 1
    return counts


def ingest(root, out_path, chunkers=None, token_limit=512):
    chunkers = chunkers or default_chunkers()
    documents = load_documents(root)
    chunks = []
    for doc in documents:
        chunks.extend(chunkers[doc.kind].split(doc))
    chunks.sort(key=lambda c: (c.path, c.start_line))
    write_jsonl(out_path, (dataclasses.asdict(c) for c in chunks))

    sizes = [c.token_count for c in chunks]
    return IngestStats(
        documents=len(documents),
        chunks=len(chunks),
        by_kind={
            "prose": sum(c.kind == "prose" for c in chunks),
            "code": sum(c.kind == "code" for c in chunks),
        },
        mean_tokens=sum(sizes) / len(sizes) if sizes else 0.0,
        max_tokens=max(sizes, default=0),
        over_limit=sum(n > token_limit for n in sizes),
        token_limit=token_limit,
        histogram=token_histogram(chunks),
    )


def read_chunks(path):
    chunks = []
    for row in read_jsonl(path):
        row["heading_path"] = tuple(row["heading_path"])
        chunks.append(Chunk(**row))
    return chunks


def format_stats(stats):
    kinds = ", ".join(f"{n} {kind}" for kind, n in stats.by_kind.items())
    lines = [
        f"documents: {stats.documents}",
        f"chunks:    {stats.chunks}  ({kinds})",
        f"tokens:    mean {stats.mean_tokens:.1f}, max {stats.max_tokens}",
        f"over the {stats.token_limit}-token limit: {stats.over_limit}",
        "histogram:",
    ]
    for label, count in stats.histogram.items():
        lines.append(f"  {label:>8}  {count}")
    return "\n".join(lines)
