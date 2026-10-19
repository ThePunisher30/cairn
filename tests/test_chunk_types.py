"""Phase 2, step 2: Document, Chunk and make_chunk (stable, content-derived identities)."""

import dataclasses
import re

import pytest

from cairn.ingest.chunkers import make_chunk
from cairn.tokens import count_tokens
from cairn.types import Chunk, Document


# ------------------------------------------------------------------ Document
@pytest.mark.parametrize(
    "text,expected",
    [
        ("", []),
        ("\n", [""]),
        ("a", ["a"]),
        ("a\n", ["a"]),
        ("a\nb", ["a", "b"]),
        ("a\nb\n", ["a", "b"]),
        ("a\n\nb\n", ["a", "", "b"]),
        ("a\x0cb\nc", ["a\x0cb", "c"]),  # a form feed is NOT a line break
    ],
)
def test_document_lines(text, expected):
    assert Document("f.md", text, "prose").lines == expected


def test_document_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        Document("f.md", "x", "prose").text = "y"


# --------------------------------------------------------------------- Chunk
def test_make_chunk_fills_in_everything():
    c = make_chunk("docs/a.md", 3, 5, "hello world", "prose", ("Title", "Sub"))
    assert isinstance(c, Chunk)
    assert (c.path, c.start_line, c.end_line, c.text, c.kind) == (
        "docs/a.md",
        3,
        5,
        "hello world",
        "prose",
    )
    assert c.heading_path == ("Title", "Sub")
    assert c.token_count == count_tokens("hello world")
    assert re.fullmatch(r"[0-9a-f]{16}", c.id)
    assert re.fullmatch(r"[0-9a-f]{16}", c.content_hash)


def test_make_chunk_heading_path_defaults_to_empty_and_is_a_tuple():
    assert make_chunk("a.md", 1, 1, "x", "prose").heading_path == ()
    assert make_chunk("a.md", 1, 1, "x", "prose", ["A", "B"]).heading_path == ("A", "B")


def test_chunk_is_frozen_and_hashable():
    c = make_chunk("a.md", 1, 1, "x", "prose")
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "y"
    assert hash(c) == hash(make_chunk("a.md", 1, 1, "x", "prose"))


def test_same_inputs_give_the_same_id_and_hash():
    a = make_chunk("a.md", 1, 4, "same text", "prose")
    b = make_chunk("a.md", 1, 4, "same text", "prose")
    assert a == b and a.id == b.id and a.content_hash == b.content_hash


@pytest.mark.parametrize(
    "changed",
    [
        dict(path="b.md"),
        dict(start_line=2),
        dict(end_line=5),
        dict(text="other text"),
    ],
)
def test_changing_any_ingredient_changes_the_id(changed):
    base = dict(path="a.md", start_line=1, end_line=4, text="same text", kind="prose")
    original = make_chunk(**base)
    modified = make_chunk(**{**base, **changed})
    assert modified.id != original.id


def test_content_hash_depends_only_on_the_text():
    a = make_chunk("a.md", 1, 2, "identical", "prose")
    b = make_chunk("other.md", 50, 51, "identical", "prose")
    assert a.content_hash == b.content_hash
    assert a.id != b.id
    assert a.content_hash != make_chunk("a.md", 1, 2, "different", "prose").content_hash


@pytest.mark.parametrize("start,end", [(0, 3), (-1, 3), (5, 4)])
def test_make_chunk_rejects_impossible_line_numbers(start, end):
    with pytest.raises(ValueError):
        make_chunk("a.md", start, end, "x", "prose")


def test_index_text_puts_the_heading_path_in_front():
    c = make_chunk("a.md", 3, 4, "the body", "prose", ("Billing", "Refunds"))
    assert c.index_text == "Billing > Refunds\nthe body"
    assert c.text == "the body"  # the stored text is never changed


def test_index_text_without_a_heading_path_is_just_the_text():
    assert make_chunk("a.md", 1, 1, "the body", "prose").index_text == "the body"
