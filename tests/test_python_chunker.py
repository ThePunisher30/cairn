"""Phase 2, step 4d: PythonChunker, which uses the standard `ast` module to split source code.

Rules:
- Only functions (def / async def) and classes sitting directly in the module body or in a class
  body
  are split out. Anything inside a function (nested defs) stays part of that function. A def
  inside an
  `if` or `try` block is just part of that block.
- Consecutive other statements (imports, constants, docstring, an `if` block...) form ONE chunk.
- Every function is one chunk. A class becomes a HEADER chunk (the `class` line, its docstring
  and any
  attributes before the first method) followed by one chunk per method (nested classes follow the
  same rule). A class with no methods is therefore one chunk.
- heading_path is the qualified name: () for module-level glue, ("helper",), ("Greeter",),
  ("Greeter", "name"), ("Outer", "Inner", "m").
- Coverage: a chunk STARTS at the first non-blank line after the previous chunk's end (so
  comments and
  decorators above a def belong to it) and ENDS at the end of its last statement. After the last
  chunk,
  any trailing non-blank lines (a final comment) are added to that chunk.
- A class header's end is the end of its leading non-def statements; if the body starts with a
  def,
  it is the line before that def's first decorator/def line, with trailing blank lines trimmed.
- A file that does not parse falls back to FixedChunker(fallback_max_tokens).
"""

import pytest

from cairn.ingest.chunkers import PythonChunker
from cairn.types import Document

SOURCE = '''\
"""Module docstring."""
import os

CONST = 1

# helper function
def helper(x):
    return x + 1

@decorator
class Greeter:
    """Says hello."""
    greeting = "hi"

    def __init__(self):
        self.n = 0

    # explain
    @property
    def name(self):
        return "x"

x = Greeter()
'''


def code(text, path="m.py"):
    return Document(path, text, "code")


def summary(chunks):
    return [(c.start_line, c.end_line, c.heading_path) for c in chunks]


def test_the_sample_module_is_split_into_the_expected_chunks():
    assert summary(PythonChunker().split(code(SOURCE))) == [
        (1, 4, ()),  # docstring, import, constant
        (6, 8, ("helper",)),  # the comment above belongs to the function
        (10, 13, ("Greeter",)),  # decorator, class line, docstring, attribute
        (15, 16, ("Greeter", "__init__")),
        (18, 21, ("Greeter", "name")),  # comment and decorator above the method
        (23, 23, ()),
    ]


def test_chunk_text_is_the_source_lines():
    chunks = PythonChunker().split(code(SOURCE))
    assert chunks[1].text == "# helper function\ndef helper(x):\n    return x + 1"
    assert chunks[0].kind == "code" and chunks[0].path == "m.py"


def test_class_whose_body_starts_with_a_method():
    source = "class A:\n    def f(self):\n        pass\n    def g(self):\n        pass\n"
    assert summary(PythonChunker().split(code(source))) == [
        (1, 1, ("A",)),
        (2, 3, ("A", "f")),
        (4, 5, ("A", "g")),
    ]


def test_class_without_methods_is_one_chunk():
    source = 'class E(Exception):\n    """Boom."""\n'
    assert summary(PythonChunker().split(code(source))) == [(1, 2, ("E",))]


def test_async_functions_are_split_like_functions():
    assert summary(PythonChunker().split(code("async def f():\n    pass\n"))) == [(1, 2, ("f",))]


def test_nested_classes_follow_the_same_rule():
    source = (
        "class Outer:\n"
        "    class Inner:\n"
        "        def m(self):\n"
        "            pass\n"
        "    def o(self):\n"
        "        pass\n"
    )
    assert summary(PythonChunker().split(code(source))) == [
        (1, 1, ("Outer",)),
        (2, 2, ("Outer", "Inner")),
        (3, 4, ("Outer", "Inner", "m")),
        (5, 6, ("Outer", "o")),
    ]


def test_a_nested_function_stays_inside_its_function():
    source = "def outer():\n    def inner():\n        pass\n    return inner\n"
    assert summary(PythonChunker().split(code(source))) == [(1, 4, ("outer",))]


def test_a_long_function_is_never_split():
    body = "".join(f"    y{i} = {i}\n" for i in range(60))
    chunks = PythonChunker().split(code("def big():\n" + body))
    assert summary(chunks) == [(1, 61, ("big",))]


def test_a_def_inside_an_if_block_is_just_part_of_the_block():
    source = "if True:\n    def f():\n        pass\n"
    assert summary(PythonChunker().split(code(source))) == [(1, 3, ())]


def test_a_trailing_comment_is_added_to_the_last_chunk():
    assert summary(PythonChunker().split(code("def f():\n    pass\n# trailing\n"))) == [
        (1, 3, ("f",))
    ]


def test_module_with_only_imports_is_one_chunk():
    assert summary(PythonChunker().split(code("import os\nimport sys\n"))) == [(1, 2, ())]


def test_a_file_that_does_not_parse_still_gets_chunks():
    broken = "def broken(:\n    pass\nx = 1\n"
    chunks = PythonChunker(fallback_max_tokens=200).split(code(broken))
    assert chunks and all(c.heading_path == () for c in chunks)
    assert summary(chunks) == [(1, 3, ())]  # one fixed chunk is enough for 3 short lines


def test_the_fallback_budget_is_used_for_unparsable_files():
    broken = "def broken(:\n" + "a b c\n" * 10
    chunks = PythonChunker(fallback_max_tokens=6).split(code(broken))
    assert len(chunks) > 1


@pytest.mark.parametrize("bad", [0, -1])
def test_fallback_budget_must_be_positive(bad):
    with pytest.raises(ValueError):
        PythonChunker(fallback_max_tokens=bad)


def test_empty_and_blank_files_give_no_chunks():
    assert PythonChunker().split(code("")) == []
    assert PythonChunker().split(code("\n\n")) == []
