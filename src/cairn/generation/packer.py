from dataclasses import dataclass

from cairn.tokens import count_tokens, truncate_tokens
from cairn.types import Hit

ORDERS = ("best_first", "edges", "reverse")
OVERFLOWS = ("truncate", "skip")


@dataclass(frozen=True)
class PackedChunk:
    number: int
    hit: Hit
    text: str
    truncated: bool


@dataclass(frozen=True)
class PackedContext:
    chunks: tuple
    tokens: int
    dropped: int


def _inside(hit, other):
    """True when `hit` lies within the lines of `other`, in the same file."""
    return (
        other.path == hit.path
        and other.start_line <= hit.start_line
        and other.end_line >= hit.end_line
    )


class ContextPacker:
    def __init__(
        self, max_tokens=3000, *, order="best_first", overflow="truncate", min_chunk_tokens=40
    ):
        if max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, it is currently {max_tokens}")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}, it is currently {order!r}")
        if overflow not in OVERFLOWS:
            raise ValueError(f"overflow must be one of {OVERFLOWS}, it is currently {overflow!r}")
        if min_chunk_tokens < 1:
            raise ValueError(f"min_chunk_tokens must be >= 1, it is currently {min_chunk_tokens}")
        self.max_tokens = max_tokens
        self.order = order
        self.overflow = overflow
        self.min_chunk_tokens = min_chunk_tokens

    def _unique(self, hits):
        kept = []
        for hit in hits:
            if not any(_inside(hit, earlier) for earlier in kept):
                kept.append(hit)
        return kept

    def _ordered(self, items):
        if self.order == "reverse":
            return items[::-1]
        if self.order == "edges":
            return items[0::2] + items[1::2][::-1]
        return items

    def pack(self, hits):
        hits = list(hits)
        chosen, total = [], 0
        for hit in self._unique(hits):
            size = count_tokens(hit.text)
            if total + size <= self.max_tokens:
                chosen.append((hit, hit.text, False))
                total += size
                continue
            if self.overflow == "skip":
                continue
            room = self.max_tokens - total
            if room >= self.min_chunk_tokens or not chosen:
                text = truncate_tokens(hit.text, room)
                chosen.append((hit, text, True))
                total += count_tokens(text)
            break
        chunks = tuple(
            PackedChunk(number, hit, text, truncated)
            for number, (hit, text, truncated) in enumerate(self._ordered(chosen), start=1)
        )
        return PackedContext(chunks, total, len(hits) - len(chunks))
