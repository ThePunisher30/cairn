from collections import defaultdict

from cairn.types import Hit

MODES = ("section", "parent")


class ContextExpander:
    def __init__(self, chunks, max_tokens=800, mode="parent"):
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, it is currently {max_tokens}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, it is currently {mode!r}")
        self.max_tokens = max_tokens
        self.mode = mode
        self.chunks = list(chunks)
        self._row = {}
        self._in_file = defaultdict(list)
        for row, chunk in enumerate(self.chunks):
            self._row[(chunk.path, chunk.start_line, chunk.end_line)] = row
            self._in_file[chunk.path].append(row)

    def _family(self, chunk):
        """The rows that count as the chunk's surroundings, in file order."""
        path = chunk.heading_path
        if not path:  # nothing says what its parent is
            return [self._row[(chunk.path, chunk.start_line, chunk.end_line)]]
        if self.mode == "parent" and len(path) >= 2:
            parent = path[:-1]
            rows = [
                r
                for r in self._in_file[chunk.path]
                if self.chunks[r].heading_path[: len(parent)] == parent
            ]
        else:
            rows = [r for r in self._in_file[chunk.path] if self.chunks[r].heading_path == path]
        return sorted(rows, key=lambda r: (self.chunks[r].start_line, self.chunks[r].end_line))

    def expand_one(self, hit):
        row = self._row.get((hit.path, hit.start_line, hit.end_line))
        if row is None:
            return hit
        rows = self._family(self.chunks[row])
        low = high = rows.index(row)
        total = self.chunks[row].token_count
        grew = True
        while grew:
            grew = False
            if low > 0:
                cost = self.chunks[rows[low - 1]].token_count
                if total + cost <= self.max_tokens:
                    low -= 1
                    total += cost
                    grew = True
            if high < len(rows) - 1:
                cost = self.chunks[rows[high + 1]].token_count
                if total + cost <= self.max_tokens:
                    high += 1
                    total += cost
                    grew = True
        members = [self.chunks[r] for r in rows[low : high + 1]]
        return Hit(
            hit.path,
            members[0].start_line,
            max(m.end_line for m in members),
            hit.score,
            "\n".join(m.text for m in members),
        )

    def expand(self, hits):
        result = []
        for hit in hits:
            bigger = self.expand_one(hit)
            if any(
                old.path == bigger.path
                and old.start_line <= bigger.start_line
                and old.end_line >= bigger.end_line
                for old in result
            ):
                continue
            result.append(bigger)
        return result
