"""What is in the index, file by file: a SQLite manifest and the diff against the corpus."""

import hashlib
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


def hash_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Diff:
    added: tuple = ()
    changed: tuple = ()
    removed: tuple = ()
    unchanged: int = 0

    @property
    def empty(self):
        return not (self.added or self.changed or self.removed)


def diff(indexed, documents):
    """`indexed` is {path: content hash} (the manifest); `documents` the files now on disk.
    A pure function: it decides what has to be re-done, nothing else."""
    now = {doc.path: hash_text(doc.text) for doc in documents}
    added = sorted(p for p in now if p not in indexed)
    changed = sorted(p for p in now if p in indexed and indexed[p] != now[p])
    removed = sorted(p for p in indexed if p not in now)
    unchanged = sum(1 for p in now if p in indexed and indexed[p] == now[p])
    return Diff(tuple(added), tuple(changed), tuple(removed), unchanged)


@dataclass
class Manifest:
    """path -> hash of the file's text when it was indexed. SQLite gives the one property that
    matters here: a batch of rows changes together or not at all (a transaction), so a crash
    cannot leave the manifest half-updated."""

    path: Path | str
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self):
        if str(self.path) != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        with self._lock, self._db:
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS files ("
                "path TEXT PRIMARY KEY, content_hash TEXT NOT NULL, "
                "n_chunks INTEGER NOT NULL, indexed_at REAL NOT NULL)"
            )

    def hashes(self):
        with self._lock:
            return dict(self._db.execute("SELECT path, content_hash FROM files"))

    def __len__(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM files").fetchone()[0]

    def apply(self, upserts=(), deletes=()):
        """`upserts`: (path, content_hash, n_chunks). One transaction for the whole batch."""
        now = time.time()
        with self._lock, self._db:
            self._db.executemany("DELETE FROM files WHERE path = ?", [(p,) for p in deletes])
            self._db.executemany(
                "INSERT OR REPLACE INTO files VALUES (?, ?, ?, ?)",
                [(path, h, n, now) for path, h, n in upserts],
            )

    def close(self):
        self._db.close()
