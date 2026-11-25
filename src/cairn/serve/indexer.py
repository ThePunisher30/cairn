"""Incremental re-indexing: find the files that changed, redo only those (Phase 9)."""

import hashlib
import logging
import threading
import time
from dataclasses import asdict, dataclass

import numpy as np

from cairn.ingest.documents import load_documents
from cairn.ingest.pipeline import default_chunkers
from cairn.serve.manifest import diff, hash_text

log = logging.getLogger("cairn.serve.indexer")


@dataclass(frozen=True)
class ReindexReport:
    added: int = 0
    changed: int = 0
    removed: int = 0
    unchanged: int = 0
    chunks_added: int = 0
    chunks_removed: int = 0
    embedded: int = 0  # chunks that went through the embedding model
    reused: int = 0  # chunks whose vector was taken from an old chunk with the same text
    compacted: bool = False
    seconds: float = 0.0
    generation: int = 0

    def as_dict(self):
        return asdict(self)


def text_key(chunk):
    return hashlib.sha256(chunk.index_text.encode("utf-8")).hexdigest()


class Indexer:
    def __init__(
        self, live, manifest, embedder, root, *, chunkers=None, chunks_path=None, dense_path=None
    ):
        self.live = live
        self.manifest = manifest
        self.embedder = embedder
        self.root = root
        self.chunkers = chunkers or default_chunkers()
        self.chunks_path = chunks_path
        self.dense_path = dense_path
        self._lock = threading.Lock()  # one re-index at a time

    def bootstrap(self):
        """Fills an empty manifest from an index that already exists. A file is recorded only if
        chunking it now gives exactly the chunks in the index (the ids contain a hash of the
        text), so the manifest never claims more than is true. A file that does not match stays
        out, and the next `reindex` redoes it."""
        with self._lock:
            verified, mismatched = [], 0
            for doc in load_documents(self.root):
                in_index = {c.id for c in self.live.chunks_of(doc.path)}
                chunks = self.chunkers[doc.kind].split(doc)
                if {c.id for c in chunks} == in_index:
                    verified.append((doc.path, hash_text(doc.text), len(chunks)))
                else:
                    mismatched += 1
            self.manifest.apply(upserts=verified)
            log.info("bootstrap: %d files match the index, %d do not", len(verified), mismatched)
            return len(verified), mismatched

    def reindex(self):
        with self._lock:
            began = time.perf_counter()
            documents = load_documents(self.root)
            found = diff(self.manifest.hashes(), documents)
            on_disk = {doc.path for doc in documents}
            # chunks in the index whose file the manifest does not know and the disk no longer has
            orphans = sorted(self.live.paths() - on_disk - set(found.removed))
            removed = [*found.removed, *orphans]
            if found.empty and not orphans:
                return ReindexReport(unchanged=found.unchanged, generation=self.live.generation)

            by_path = {doc.path: doc for doc in documents}
            known = self.live.paths()
            # every file whose old chunks must go: changed, deleted, and "added" ones the index
            # already holds (a manifest that was missing them)
            drop = [*found.changed, *removed, *(p for p in found.added if p in known)]

            reusable = {}  # text key -> vector, from the chunks about to be dropped
            for path in drop:
                vectors = self.live.vectors_of(path)
                for chunk in self.live.chunks_of(path):
                    reusable.setdefault(text_key(chunk), vectors[chunk.id])

            new = {}
            for path in [*found.added, *found.changed]:
                doc = by_path[path]
                new[path] = self.chunkers[doc.kind].split(doc)

            to_embed = {}
            for chunks in new.values():
                for chunk in chunks:
                    key = text_key(chunk)
                    if key not in reusable and key not in to_embed:
                        to_embed[key] = chunk.index_text
            if to_embed:
                vectors = self.embedder.embed_documents(list(to_embed.values()))
                reusable.update(zip(to_embed, vectors, strict=True))

            batches = []
            for chunks in new.values():
                vectors = [reusable[text_key(c)] for c in chunks]
                batches.append(
                    (chunks, np.array(vectors, dtype=np.float32).reshape(len(chunks), -1))
                )
            chunks_removed = sum(len(self.live.chunks_of(p)) for p in drop)
            result = self.live.apply(remove_paths=drop, add=batches)

            # the files first, the manifest last: a crash in between leaves a manifest that is
            # behind the index, and the next re-index simply redoes those files
            if self.chunks_path is not None and self.dense_path is not None:
                self.live.save(self.chunks_path, self.dense_path)
            self.manifest.apply(
                upserts=[(p, hash_text(by_path[p].text), len(new[p])) for p in new],
                deletes=removed,
            )
            n_new = sum(len(c) for c in new.values())
            report = ReindexReport(
                added=len(found.added),
                changed=len(found.changed),
                removed=len(removed),
                unchanged=found.unchanged,
                chunks_added=n_new,
                chunks_removed=chunks_removed,
                embedded=len(to_embed),
                reused=n_new - len(to_embed),
                compacted=result["compacted"],
                seconds=round(time.perf_counter() - began, 3),
                generation=self.live.generation,
            )
            log.info(
                "reindex: %d added, %d changed, %d removed, %d unchanged; "
                "%d chunks embedded, %d reused; %d tombstones%s; %.2fs",
                report.added,
                report.changed,
                report.removed,
                report.unchanged,
                report.embedded,
                report.reused,
                self.live.n_dead,
                " (compacted)" if report.compacted else "",
                report.seconds,
            )
            return report
