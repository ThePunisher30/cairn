"""Phase 6, step 4: small-to-big expansion (`src/cairn/retrieval/context.py`).

Retrieval works best on SMALL chunks (a precise match), the reader (the LLM, in Phase 7) works best
with BIG ones (the surrounding explanation). Small-to-big retrieves small and hands over the parent.

    ContextExpander(chunks, max_tokens=800, mode="parent")
    expander.expand_one(hit) -> Hit
    expander.expand(hits) -> [Hit, ...]

`chunks` is the list of every Chunk. A hit is matched to its chunk by its place: (path, start_line,
end_line). A hit that matches no chunk is returned unchanged.

The chunk's FAMILY is the chunks of the SAME FILE (never another file) that it may be joined with:
  - mode "section": the chunks with exactly its heading_path (the same section of the page, or the
    same class or function);
  - mode "parent": for a heading_path of two or more headings, every chunk whose heading_path
    STARTS WITH the heading_path without its last heading (its section, the sibling sections
    and everything below them); for one heading, the same as "section";
  - a chunk with an empty heading_path is never expanded: nothing says what its parent is.
The family is read in file order (by start_line), whatever the order of `chunks`.

The expansion starts from the hit's chunk and grows through its family, while the total
`token_count` stays <= `max_tokens`. Each round it tries the previous chunk of the family first,
then the next one, and it stops when neither fits. The chunk that was hit is always kept, even if
it alone is bigger than `max_tokens`.

The result is a Hit with the SAME path and score, the lines from the first kept chunk's start_line
to the last kept chunk's end_line (the largest end_line), and `text` = the kept chunks' texts, in
file order, joined with "\\n".

`expand(hits)` expands every hit, keeps the order of `hits`, and drops a hit whose expanded lines
lie inside the expanded lines of a hit before it in the same file (so the reader is not given the
same passage twice). A later hit that is BIGGER than an earlier one is kept.

ValueError for `max_tokens < 1` or a mode that is not "section" or "parent".
"""

import pytest

from cairn.ingest.chunkers import make_chunk
from cairn.retrieval.context import ContextExpander
from cairn.types import Hit


def words(n):
    return " ".join(["w"] * n)  # n words = n tokens


def chunk(path, start, end, heading, tokens=100):
    return make_chunk(path, start, end, words(tokens), "prose", heading)


def hit_for(c, score=0.5):
    return Hit(c.path, c.start_line, c.end_line, score, c.text)


def span(h):
    return (h.start_line, h.end_line)


@pytest.fixture
def page():
    c = {
        "intro": chunk("a.md", 1, 2, ("T",)),
        "a1": chunk("a.md", 3, 4, ("T", "A")),
        "a2": chunk("a.md", 5, 6, ("T", "A")),
        "b1": chunk("a.md", 7, 8, ("T", "B")),
        "b2": chunk("a.md", 9, 10, ("T", "B")),
        "other": chunk("b.md", 1, 2, ("T", "A")),  # another file with the same headings
    }
    return c


def expander(page, **options):
    return ContextExpander(list(page.values()), **options)


# --------------------------------------------------------------------- section mode
def test_section_mode_joins_the_chunks_of_the_same_section(page):
    result = expander(page, mode="section", max_tokens=800).expand_one(hit_for(page["a1"]))
    assert (result.path, span(result)) == ("a.md", (3, 6))
    assert result.text == page["a1"].text + "\n" + page["a2"].text


def test_the_expansion_is_the_same_whichever_chunk_of_the_section_was_hit(page):
    ex = expander(page, mode="section", max_tokens=800)
    assert span(ex.expand_one(hit_for(page["a2"]))) == (3, 6)
    assert span(ex.expand_one(hit_for(page["b1"]))) == (7, 10)


def test_a_section_of_one_chunk_is_returned_as_it_was(page):
    result = expander(page, mode="section").expand_one(hit_for(page["intro"], score=0.9))
    assert (span(result), result.text, result.score) == ((1, 2), page["intro"].text, 0.9)


def test_chunks_of_another_file_are_never_joined(page):
    result = expander(page, mode="section").expand_one(hit_for(page["other"]))
    assert (result.path, span(result)) == ("b.md", (1, 2))


def test_the_order_of_the_chunk_list_does_not_matter(page):
    shuffled = [page[name] for name in ("b2", "a2", "other", "intro", "b1", "a1")]
    ex = ContextExpander(shuffled, mode="section", max_tokens=800)
    result = ex.expand_one(hit_for(page["a2"]))
    assert span(result) == (3, 6)
    assert result.text == page["a1"].text + "\n" + page["a2"].text  # file order, not list order


# ---------------------------------------------------------------------- parent mode
def test_parent_mode_takes_the_whole_parent_section(page):
    # the parent of ("T", "A") is ("T",): the intro, A and B
    result = expander(page, mode="parent", max_tokens=800).expand_one(hit_for(page["a1"]))
    assert span(result) == (1, 10)
    assert result.text.count("\n") == 4  # five chunks joined


def test_parent_is_the_default_mode(page):
    assert span(ContextExpander(list(page.values())).expand_one(hit_for(page["a1"]))) == (1, 10)


def test_parent_means_one_heading_up_not_the_top_of_the_page():
    deep = {
        "a0": chunk("d.md", 1, 2, ("T", "A")),
        "ax": chunk("d.md", 3, 4, ("T", "A", "X")),
        "ay": chunk("d.md", 5, 6, ("T", "A", "Y")),
        "bz": chunk("d.md", 7, 8, ("T", "B", "Z")),
    }
    ex = ContextExpander(list(deep.values()), mode="parent", max_tokens=800)
    # the parent of ("T", "A", "X") is ("T", "A"): the section, the sibling Y, but not B
    assert span(ex.expand_one(hit_for(deep["ax"]))) == (1, 6)
    assert span(ex.expand_one(hit_for(deep["bz"]))) == (7, 8)
    # section mode keeps to the exact heading
    assert span(
        ContextExpander(list(deep.values()), mode="section").expand_one(hit_for(deep["ax"]))
    ) == (3, 4)


def test_a_chunk_with_a_single_heading_is_not_widened_by_parent_mode(page):
    assert span(expander(page, mode="parent").expand_one(hit_for(page["intro"]))) == (1, 2)


def test_a_chunk_with_no_heading_is_never_expanded():
    first = chunk("m.py", 1, 2, ())
    second = chunk("m.py", 5, 6, ())
    ex = ContextExpander([first, second], mode="parent", max_tokens=800)
    assert span(ex.expand_one(hit_for(first))) == (1, 2)
    assert span(ContextExpander([first, second], mode="section").expand_one(hit_for(second))) == (
        5,
        6,
    )


# ---------------------------------------------------------------------- the budget
def test_growth_stops_at_the_token_budget_and_tries_the_previous_chunk_first(page):
    ex = expander(page, mode="parent", max_tokens=250)
    # a1 (100) + previous (intro) = 200; the next chunk would make 300
    assert span(ex.expand_one(hit_for(page["a1"]))) == (1, 4)
    # a2 (100) + previous (a1) = 200; the next chunk (b1) would make 300
    assert span(ex.expand_one(hit_for(page["a2"]))) == (3, 6)


def test_growth_can_continue_on_one_side_when_the_other_is_full(page):
    ex = expander(page, mode="parent", max_tokens=350)
    # b2 is the last chunk: it can only grow backwards, three chunks in all
    assert span(ex.expand_one(hit_for(page["b2"]))) == (5, 10)


def test_the_chunk_that_was_hit_is_kept_even_if_it_exceeds_the_budget(page):
    result = expander(page, mode="parent", max_tokens=50).expand_one(hit_for(page["a1"]))
    assert (span(result), result.text) == ((3, 4), page["a1"].text)


def test_a_budget_that_fits_exactly_is_used_in_full(page):
    assert span(expander(page, mode="parent", max_tokens=300).expand_one(hit_for(page["a1"]))) == (
        1,
        6,
    )


def test_a_budget_that_fits_exactly_is_used_in_full_when_growing_backwards_too(page):
    # a2 (100) + the previous chunk a1 (100) = 200 exactly; the next chunk would make 300
    assert span(expander(page, mode="parent", max_tokens=200).expand_one(hit_for(page["a2"]))) == (
        3,
        6,
    )


# --------------------------------------------------------------- unknown hits, validation
def test_a_hit_that_matches_no_chunk_is_returned_unchanged(page):
    stranger = Hit("a.md", 3, 5, 0.1, "not a chunk")
    assert expander(page).expand_one(stranger) == stranger
    assert expander(page).expand([stranger]) == [stranger]


def test_bad_arguments_are_refused(page):
    with pytest.raises(ValueError):
        ContextExpander(list(page.values()), max_tokens=0)
    with pytest.raises(ValueError):
        ContextExpander(list(page.values()), mode="document")


# -------------------------------------------------------------------------- expand
def test_expand_keeps_the_order_and_the_scores(page):
    hits = [hit_for(page["b1"], 0.9), hit_for(page["other"], 0.8)]
    result = expander(page, mode="section").expand(hits)
    assert [(h.path, span(h), h.score) for h in result] == [
        ("a.md", (7, 10), 0.9),
        ("b.md", (1, 2), 0.8),
    ]


def test_two_hits_in_the_same_section_give_one_context(page):
    hits = [hit_for(page["a1"], 0.9), hit_for(page["a2"], 0.8)]
    result = expander(page, mode="section").expand(hits)
    assert [span(h) for h in result] == [(3, 6)]
    assert result[0].score == 0.9  # the better hit's


def test_a_later_hit_inside_an_earlier_context_is_dropped(page):
    hits = [hit_for(page["a1"]), hit_for(page["intro"])]  # intro (1-2) lies inside 1-10
    result = expander(page, mode="parent", max_tokens=800).expand(hits)
    assert [span(h) for h in result] == [(1, 10)]


def test_a_later_hit_that_is_bigger_than_an_earlier_one_is_kept(page):
    hits = [hit_for(page["intro"]), hit_for(page["a1"])]
    result = expander(page, mode="parent", max_tokens=800).expand(hits)
    assert [span(h) for h in result] == [(1, 2), (1, 10)]


def test_expanding_nothing_gives_nothing(page):
    assert expander(page).expand([]) == []


def test_the_chunks_and_hits_are_not_changed(page):
    ex = expander(page)
    original = hit_for(page["a1"])
    ex.expand([original])
    assert original == hit_for(page["a1"])
