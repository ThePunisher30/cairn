"""Phase 3, step 1: the tokenizer, which decides what a search "word" is.

Rules (from docs/briefs/phase3.md):
- a word is a run of letters, digits and underscores, lowercased; punctuation is dropped;
- code-aware: an identifier such as `get_request_handler` or `BackgroundTasks` is kept WHOLE and
  ALSO split into its parts (`get`, `request`, `handler`), whole token first, then the parts;
- a word that has no parts of its own (`request`, `FOO`, `404`) is emitted once, not twice;
- repeats are kept (term frequency needs them) and text order is preserved;
- stopwords, if given, are removed from the output (the whole token and the parts alike).
"""

import dataclasses

import pytest

from cairn.retrieval.tokenizer import ENGLISH_STOPWORDS, Tokenizer


def tokens(text, **options):
    return Tokenizer(**options).tokenize(text)


# ------------------------------------------------------------------ plain text
def test_words_are_lowercased_and_punctuation_is_dropped():
    assert tokens("The Quick, brown FOX!") == ["the", "quick", "brown", "fox"]


def test_numbers_are_words():
    assert tokens("status 404 and 200") == ["status", "404", "and", "200"]


def test_nothing_in_nothing_out():
    assert tokens("") == []
    assert tokens("   \n\t ") == []
    assert tokens("!!! ??? --- ()") == []
    assert tokens("_ __ ___") == []


def test_repeats_are_kept_because_term_frequency_needs_them():
    assert tokens("task task Task") == ["task", "task", "task"]


def test_markdown_noise_does_not_leave_symbols_behind():
    assert tokens("## Title { #anchor }") == ["title", "anchor"]


# ------------------------------------------------------------- code identifiers
def test_snake_case_is_kept_whole_and_split():
    assert tokens("get_request_handler") == ["get_request_handler", "get", "request", "handler"]


def test_camel_case_is_kept_whole_and_split():
    assert tokens("BackgroundTasks") == ["backgroundtasks", "background", "tasks"]


def test_an_acronym_stays_together_in_the_split():
    assert tokens("HTTPException") == ["httpexception", "http", "exception"]
    assert tokens("APIRouter") == ["apirouter", "api", "router"]


def test_a_dunder_name_is_split_into_its_word():
    assert tokens("__init__") == ["__init__", "init"]


def test_letters_and_digits_are_split_apart():
    assert tokens("py310") == ["py310", "py", "310"]


def test_a_dotted_path_is_several_words():
    assert tokens("fastapi.responses.JSONResponse") == [
        "fastapi",
        "responses",
        "jsonresponse",
        "json",
        "response",
    ]


def test_a_word_without_parts_is_emitted_only_once():
    assert tokens("request") == ["request"]
    assert tokens("FOO") == ["foo"]
    assert tokens("404") == ["404"]


def test_the_whole_token_comes_first_then_the_parts_then_the_next_word_in_text_order():
    assert tokens("call add_task now") == ["call", "add_task", "add", "task", "now"]


def test_a_call_in_code_gives_the_function_and_its_argument():
    assert tokens("Depends(get_db)") == ["depends", "get_db", "get", "db"]


def test_splitting_can_be_switched_off():
    assert tokens("get_request_handler BackgroundTasks", split_identifiers=False) == [
        "get_request_handler",
        "backgroundtasks",
    ]


def test_splitting_off_does_not_change_plain_words():
    text = "The quick brown fox"
    assert tokens(text, split_identifiers=False) == tokens(text)


# ------------------------------------------------------------------- stopwords
def test_stopwords_are_removed():
    stop = frozenset({"the", "a"})
    assert tokens("The cat and a hat", stopwords=stop) == ["cat", "and", "hat"]


def test_stopwords_are_compared_in_lowercase():
    assert tokens("THE Cat", stopwords=frozenset({"the"})) == ["cat"]


def test_stopwords_also_remove_parts_of_identifiers():
    assert tokens("get_user", stopwords=frozenset({"get"})) == ["get_user", "user"]


def test_no_stopwords_by_default():
    assert tokens("the and of") == ["the", "and", "of"]


def test_the_english_stopword_list_is_a_sensible_frozenset():
    assert isinstance(ENGLISH_STOPWORDS, frozenset)
    assert len(ENGLISH_STOPWORDS) >= 20
    for word in (
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "in",
        "is",
        "are",
        "for",
        "how",
        "what",
    ):
        assert word in ENGLISH_STOPWORDS, word
    assert all(word == word.lower() for word in ENGLISH_STOPWORDS)


def test_the_stopword_list_keeps_words_that_matter_in_code():
    # these are Python keywords or common API words: removing them would hurt code search
    for word in ("not", "none", "true", "false", "get", "set", "from", "import", "class", "def"):
        assert word not in ENGLISH_STOPWORDS, word


# ------------------------------------------------------------ the tokenizer itself
def test_it_is_deterministic():
    text = "BackgroundTasks.add_task(send_email, to='a@b.c')"
    assert tokens(text) == tokens(text)


def test_a_tokenizer_is_an_immutable_value_that_can_be_compared():
    assert Tokenizer() == Tokenizer()
    assert Tokenizer(split_identifiers=False) != Tokenizer()
    assert Tokenizer(stopwords=frozenset({"x"})) != Tokenizer()
    assert hash(Tokenizer()) == hash(Tokenizer())
    with pytest.raises(dataclasses.FrozenInstanceError):
        Tokenizer().split_identifiers = False
