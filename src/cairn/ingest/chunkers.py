"""The chunkers, and make_chunk, the one way a Chunk gets built."""

import hashlib

from cairn.tokens import count_tokens
from cairn.types import Chunk


def make_chunk(path, start_line, end_line, text, kind, heading_path=()) -> Chunk:
    if start_line < 1 or end_line < start_line:
        raise ValueError(f"Invalid line range: {start_line}-{end_line}")
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    place_and_content = f"{path}:{start_line}:{end_line}:{content_hash}"
    chunk_id = hashlib.sha256(place_and_content.encode("utf-8")).hexdigest()[:16]

    return Chunk(
        id=chunk_id,
        text=text,
        path=path,
        start_line=start_line,
        end_line=end_line,
        kind=kind,
        content_hash=content_hash,
        token_count=count_tokens(text),
        heading_path=tuple(heading_path),
    )
