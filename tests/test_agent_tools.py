"""Phase 8, step 4: tools (`agent/tools.py`).

The model never runs anything: it sends a request, our code runs it and sends back text. The
model's output is untrusted input, so every request is checked, and every failure comes back as a
message the model can read, never as a crash.

ToolRegistry: the generic part
    class ToolError(Exception)         a tool raises it with a message meant for the model
    Tool(name, description, parameters, func)     frozen. `parameters` is a JSON schema; `func`
        is called as func(**arguments) and returns the text for the model.
    ToolResult(text, error=None, truncated=False)     frozen
    ToolRegistry(tools, *, max_result_chars=6000)
        ValueError when two tools share a name.
        .names                     the tool names, in the order given (a tuple)
        .specs() -> list[ToolSpec] (name, description, parameters), same order
        .execute(call: ToolCall) -> ToolResult     NEVER raises an Exception (KeyboardInterrupt
            and SystemExit pass through). In this order:
            1. call.error is set          -> error = call.error (the tool is not run)
            2. unknown tool name          -> error mentions the name and lists the available tools
            3. arguments are checked against the schema (a subset of JSON schema):
                 a required argument missing          -> error names it
                 an argument the schema does not list -> error names it
                 wrong type -> error names the argument. "string", "integer", "number" (an int or
                     a float), "boolean". A bool is NOT an integer or a number.
                 "minimum" / "maximum" on integer and number -> error names the argument
               Missing optional arguments are not filled in: `func` has its own defaults.
            4. func raises ToolError(msg)  -> error = msg
               func raises any other Exception -> error = "tool '<name>' failed: <ExcType>: <msg>"
            5. otherwise the text is the result. A text over max_result_chars is cut to its first
               max_result_chars characters followed by "\\n[truncated: N more characters]" (N =
               the number of characters cut) and `truncated` is True.
        For every failure `text` is "Error: " + error (that is what the model reads).

The four corpus tools
    build_corpus_tools(retriever, root, *, max_k=10, snippet_chars=400, max_lines=120,
                       max_result_chars=6000) -> CorpusTools
        `retriever` has search(query, k) -> list[Hit]; `root` is the corpus directory.
    CorpusTools has .registry (a ToolRegistry with the four tools below, in this order), .seen (a
        list of the Hits search has returned in this session) and .reset() (empties .seen and
        the ids; a new question starts clean).

    search(query: string, k: integer 1..max_k (default 5), path_filter: string (optional))
        Returns one block per hit, blocks separated by a blank line:
            [3] docs/en/docs/tutorial/cors.md:10-30
            the first snippet_chars characters of hit.text (" ..." added when cut)
        The number in brackets is the chunk's id in this session (what the model cites as [3]):
        1, 2, 3 ... in order of first appearance, the SAME id when the same path and line range is
        returned again, and `.seen[id - 1]` is that Hit. With path_filter only hits whose path
        starts with it are kept (the retriever is asked for more than k so the filter has
        something to keep); at most k come back. No hits at all -> the text "No results." (not an
        error). An empty or whitespace query -> ToolError.
    read_chunk(id: integer >= 1)
        "[id] path:start-end" and then the FULL text of that hit. An id search has not returned ->
        ToolError saying there is no chunk with that id.
    read_file(path: string, start: integer >= 1, end: integer >= 1)
        Lines start..end (1-based, inclusive) of a file under `root`, each as "<n>: <line>", lines
        counted as `Document.lines` does. `end` past the end of the file is cut to the last line;
        `start` past it -> ToolError saying how many lines the file has; end < start -> ToolError.
        More than max_lines lines are not returned: the first max_lines come back, followed by
        a final line "[stopped after <max_lines> lines; continue from line <next>]".
    list_symbols(path: string)
        For a .py file, one line per class, function and method (top level, and directly inside a
        class; not functions nested in functions), in file order, two spaces of indent per level:
            class Router (lines 3-20)
              def add (lines 5-9)
            async def fetch (lines 22-30)
        No symbols -> "No symbols." A path that is not .py, or does not parse -> ToolError.

    PATH SAFETY (read_file and list_symbols). A path is refused with a ToolError (so the model
    sees "Error: ..."), and the file is never opened, when it:
        is absolute (starts with "/" or "\\", or has a drive such as "C:"), contains a ".."
        component (after turning "\\" into "/"; "docs/../docs/a.md" too), contains a NUL
        character, or after resolving symbolic links is not inside `root`.
    A path that does not exist, or is a directory -> ToolError. The message for a path outside the
    corpus contains the word "outside". "docs\\a.md" is the same file as "docs/a.md".
"""

import os

import pytest

from cairn.agent.tools import (
    Tool,
    ToolError,
    ToolRegistry,
    ToolResult,
    build_corpus_tools,
)
from cairn.generation.types import ToolCall, ToolSpec
from cairn.types import Hit

SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string"},
        "k": {"type": "integer", "minimum": 1, "maximum": 10},
        "weight": {"type": "number"},
        "flag": {"type": "boolean"},
    },
    "required": ["query"],
}


def echo(query, k=5, weight=1.0, flag=False):
    return f"{query}|{k}|{weight}|{flag}"


def registry(**kw):
    return ToolRegistry([Tool("echo", "Echoes.", SCHEMA, echo)], **kw)


def run(reg, name="echo", **arguments):
    return reg.execute(ToolCall("c1", name, arguments))


# ------------------------------------------------------------------ ToolRegistry
def test_the_specs_and_names_keep_the_order():
    other = Tool("other", "Other.", {"type": "object", "properties": {}}, lambda: "x")
    reg = ToolRegistry([Tool("echo", "Echoes.", SCHEMA, echo), other])
    assert reg.names == ("echo", "other")
    assert reg.specs() == [
        ToolSpec("echo", "Echoes.", SCHEMA),
        ToolSpec("other", "Other.", {"type": "object", "properties": {}}),
    ]


def test_two_tools_with_one_name_are_refused():
    t = Tool("echo", "Echoes.", SCHEMA, echo)
    with pytest.raises(ValueError):
        ToolRegistry([t, t])


def test_a_valid_call_runs_the_function():
    result = run(registry(), query="hi", k=3, weight=2, flag=True)
    assert result == ToolResult("hi|3|2|True")
    assert run(registry(), query="hi").text == "hi|5|1.0|False"  # func's own defaults


def test_a_call_with_an_error_is_not_run():
    ran = []
    reg = ToolRegistry([Tool("echo", "E.", SCHEMA, lambda **kw: ran.append(kw) or "x")])
    result = reg.execute(ToolCall("c1", "echo", {}, error="arguments are not a JSON object: [1]"))
    assert ran == []
    assert result.error == "arguments are not a JSON object: [1]"
    assert result.text == "Error: " + result.error


def test_an_unknown_tool_names_itself_and_the_choices():
    result = run(registry(), "serach", query="x")
    assert result.error and "serach" in result.error and "echo" in result.error
    assert result.text == "Error: " + result.error


@pytest.mark.parametrize(
    "arguments, word",
    [
        ({}, "query"),  # required argument missing
        ({"query": "x", "colour": "red"}, "colour"),  # not in the schema
        ({"query": 5}, "query"),  # wrong type
        ({"query": "x", "k": "3"}, "k"),
        ({"query": "x", "k": 2.5}, "k"),
        ({"query": "x", "k": True}, "k"),  # a bool is not an integer
        ({"query": "x", "weight": True}, "weight"),  # nor a number
        ({"query": "x", "weight": "heavy"}, "weight"),
        ({"query": "x", "flag": 1}, "flag"),
        ({"query": "x", "k": 0}, "k"),  # below minimum
        ({"query": "x", "k": 11}, "k"),  # above maximum
        ({"query": None}, "query"),
    ],
)
def test_bad_arguments_are_refused_by_name_and_the_function_is_not_run(arguments, word):
    ran = []
    reg = ToolRegistry([Tool("echo", "E.", SCHEMA, lambda **kw: ran.append(kw) or "x")])
    result = reg.execute(ToolCall("c1", "echo", arguments))
    assert result.error is not None and word in result.error
    assert result.text == "Error: " + result.error
    assert ran == []


def test_a_float_is_a_number_and_the_limits_are_inclusive():
    assert run(registry(), query="x", weight=0.5, k=1).error is None
    assert run(registry(), query="x", k=10).error is None


def test_a_tool_error_message_reaches_the_model_as_it_is():
    def fail(query):
        raise ToolError("no such page")

    reg = ToolRegistry([Tool("echo", "E.", SCHEMA, fail)])
    result = run(reg, query="x")
    assert result.error == "no such page" and result.text == "Error: no such page"


def test_any_other_exception_becomes_an_error_and_does_not_escape():
    def crash(query):
        raise KeyError("boom")

    result = run(ToolRegistry([Tool("echo", "E.", SCHEMA, crash)]), query="x")
    assert result.error.startswith("tool 'echo' failed: KeyError")


def test_keyboard_interrupt_is_not_swallowed():
    def stop(query):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(ToolRegistry([Tool("echo", "E.", SCHEMA, stop)]), query="x")


def test_a_long_result_is_cut_and_says_how_much():
    reg = ToolRegistry([Tool("echo", "E.", SCHEMA, lambda query: "a" * 100)], max_result_chars=40)
    result = run(reg, query="x")
    assert result.truncated is True and result.error is None
    assert result.text == "a" * 40 + "\n[truncated: 60 more characters]"


def test_a_result_at_the_limit_is_not_cut():
    reg = ToolRegistry([Tool("echo", "E.", SCHEMA, lambda query: "a" * 40)], max_result_chars=40)
    result = run(reg, query="x")
    assert result.text == "a" * 40 and result.truncated is False


# ------------------------------------------------------------------ the corpus tools
class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits
        self.asked = []

    def search(self, query, k):
        self.asked.append((query, k))
        return self.hits[:k]


def hit(path, start, end, text="body"):
    return Hit(path, start, end, 1.0, text)


HITS = [
    hit("docs/en/docs/a.md", 1, 5, "alpha text"),
    hit("fastapi/b.py", 10, 20, "beta text"),
    hit("docs/en/docs/c.md", 7, 9, "gamma text"),
]
FILE_LINES = ["first", "second", "third", "fourth", "fifth"]
PY_SOURCE = '''"""module"""
import os


class Router:
    """doc"""

    def add(self, x):
        def inner():
            return x

        return inner

    async def fetch(self):
        return 1


@decorator
def helper(a):
    return a


async def go():
    pass
'''


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    (root / "docs").mkdir(parents=True)
    (root / "fastapi").mkdir()
    (root / "docs" / "a.md").write_text("\n".join(FILE_LINES) + "\n", encoding="utf-8")
    (root / "fastapi" / "routing.py").write_text(PY_SOURCE, encoding="utf-8")
    (root / "fastapi" / "empty.py").write_text("# only a comment\n", encoding="utf-8")
    (root / "fastapi" / "broken.py").write_text("def (:\n", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret\n", encoding="utf-8")
    return root


def tools_for(corpus, hits=HITS, **kw):
    retriever = FakeRetriever(hits)
    return build_corpus_tools(retriever, corpus, **kw), retriever


def call(tools, name, **arguments):
    return tools.registry.execute(ToolCall("c1", name, arguments))


def test_there_are_four_tools_in_order_with_the_documented_schemas(corpus):
    tools, _ = tools_for(corpus, max_k=7)
    assert tools.registry.names == ("search", "read_chunk", "read_file", "list_symbols")
    specs = {s.name: s for s in tools.registry.specs()}
    search = specs["search"].parameters
    assert search["required"] == ["query"]
    assert search["properties"]["k"]["minimum"] == 1 and search["properties"]["k"]["maximum"] == 7
    assert set(search["properties"]) == {"query", "k", "path_filter"}
    assert specs["read_file"].parameters["required"] == ["path", "start", "end"]
    assert all(s.description for s in specs.values())


def test_search_numbers_the_hits_and_shows_snippets(corpus):
    tools, retriever = tools_for(corpus)
    result = call(tools, "search", query="cors")
    blocks = result.text.split("\n\n")
    assert blocks[0] == "[1] docs/en/docs/a.md:1-5\nalpha text"
    assert blocks[1] == "[2] fastapi/b.py:10-20\nbeta text"
    assert len(blocks) == 3
    assert retriever.asked[0] == ("cors", 5)  # the default k
    assert tools.seen == HITS


def test_search_passes_k_on_and_cuts_long_snippets(corpus):
    long_hit = hit("docs/en/docs/a.md", 1, 5, "x" * 50)
    tools, retriever = tools_for(corpus, hits=[long_hit], snippet_chars=10)
    result = call(tools, "search", query="q", k=2)
    assert result.text == "[1] docs/en/docs/a.md:1-5\n" + "x" * 10 + " ..."
    assert retriever.asked[0][1] == 2


def test_ids_continue_across_searches_and_repeat_for_the_same_chunk(corpus):
    tools, retriever = tools_for(corpus)
    call(tools, "search", query="one", k=2)  # ids 1 and 2
    retriever.hits = [HITS[1], HITS[2]]  # b again, and a new one
    result = call(tools, "search", query="two", k=2)
    assert result.text.startswith("[2] fastapi/b.py:10-20")  # the same chunk keeps its id
    assert "[3] docs/en/docs/c.md:7-9" in result.text
    assert tools.seen == [HITS[0], HITS[1], HITS[2]]


def test_search_path_filter_keeps_only_matching_paths(corpus):
    tools, _ = tools_for(corpus)
    result = call(tools, "search", query="q", path_filter="docs/en/docs/")
    assert "fastapi/b.py" not in result.text
    assert "docs/en/docs/a.md" in result.text and "docs/en/docs/c.md" in result.text


def test_search_path_filter_looks_past_the_first_k_hits(corpus):
    many = [hit(f"other/f{i}.md", 1, 2) for i in range(12)] + [hit("docs/en/docs/z.md", 1, 2)]
    tools, _ = tools_for(corpus, hits=many)
    result = call(tools, "search", query="q", k=3, path_filter="docs/")
    assert "docs/en/docs/z.md" in result.text


def test_search_returns_at_most_k_after_filtering(corpus):
    many = [hit(f"docs/f{i}.md", 1, 2) for i in range(9)]
    tools, _ = tools_for(corpus, hits=many)
    result = call(tools, "search", query="q", k=2, path_filter="docs/")
    assert result.text.count("docs/f") == 2


def test_search_with_no_hits_is_not_an_error(corpus):
    tools, _ = tools_for(corpus, hits=[])
    result = call(tools, "search", query="nothing")
    assert result.text == "No results." and result.error is None


@pytest.mark.parametrize("query", ["", "   "])
def test_search_refuses_an_empty_query(corpus, query):
    tools, retriever = tools_for(corpus)
    assert call(tools, "search", query=query).error
    assert retriever.asked == []


def test_search_k_over_the_limit_is_refused(corpus):
    tools, retriever = tools_for(corpus, max_k=4)
    assert "k" in call(tools, "search", query="q", k=5).error
    assert retriever.asked == []


def test_read_chunk_gives_the_full_text(corpus):
    full = hit("docs/en/docs/a.md", 1, 5, "y" * 900)
    tools, _ = tools_for(corpus, hits=[full], snippet_chars=20)
    call(tools, "search", query="q")
    result = call(tools, "read_chunk", id=1)
    assert result.text == "[1] docs/en/docs/a.md:1-5\n" + "y" * 900


def test_read_chunk_with_an_unknown_id_is_an_error(corpus):
    tools, _ = tools_for(corpus)
    result = call(tools, "read_chunk", id=3)
    assert result.error and "3" in result.error
    call(tools, "search", query="q")
    assert call(tools, "read_chunk", id=3).error is None
    assert call(tools, "read_chunk", id=4).error


def test_reset_starts_a_new_session(corpus):
    tools, _ = tools_for(corpus)
    call(tools, "search", query="q")
    tools.reset()
    assert tools.seen == []
    assert call(tools, "read_chunk", id=1).error
    assert call(tools, "search", query="q").text.startswith("[1] docs/en/docs/a.md:1-5")


def test_read_file_numbers_the_lines(corpus):
    tools, _ = tools_for(corpus)
    result = call(tools, "read_file", path="docs/a.md", start=2, end=4)
    assert result.text == "2: second\n3: third\n4: fourth"


def test_read_file_cuts_an_end_past_the_file_and_refuses_a_start_past_it(corpus):
    tools, _ = tools_for(corpus)
    assert call(tools, "read_file", path="docs/a.md", start=4, end=99).text == "4: fourth\n5: fifth"
    result = call(tools, "read_file", path="docs/a.md", start=6, end=8)
    assert result.error and "5" in result.error  # the file has 5 lines
    assert call(tools, "read_file", path="docs/a.md", start=3, end=2).error


def test_read_file_stops_after_max_lines_and_says_where_to_continue(corpus):
    tools, _ = tools_for(corpus, max_lines=3)
    lines = call(tools, "read_file", path="docs/a.md", start=1, end=5).text.split("\n")
    assert lines[:3] == ["1: first", "2: second", "3: third"]
    assert len(lines) == 4
    assert "stopped after 3 lines" in lines[3] and "line 4" in lines[3]


def test_read_file_accepts_backslashes(corpus):
    tools, _ = tools_for(corpus)
    assert call(tools, "read_file", path="docs\\a.md", start=1, end=1).text == "1: first"


def test_read_file_missing_file_and_directory(corpus):
    tools, _ = tools_for(corpus)
    assert call(tools, "read_file", path="docs/nope.md", start=1, end=2).error
    assert call(tools, "read_file", path="docs", start=1, end=2).error


@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "..\\secret.txt",
        "docs/../../secret.txt",
        "docs/../docs/a.md",  # any ".." is refused, even one that stays inside
        "/etc/passwd",
        "\\Windows\\win.ini",
        "C:\\Windows\\win.ini",
        "C:/Windows/win.ini",
        "docs/a.md\0.txt",
    ],
)
@pytest.mark.parametrize("tool", ["read_file", "list_symbols"])
def test_paths_that_could_leave_the_corpus_are_refused(corpus, path, tool):
    tools, _ = tools_for(corpus)
    arguments = {"path": path, "start": 1, "end": 5} if tool == "read_file" else {"path": path}
    result = call(tools, tool, **arguments)
    assert result.error is not None
    assert "top secret" not in result.text


def test_an_absolute_path_to_a_real_file_outside_the_corpus_is_refused(corpus):
    tools, _ = tools_for(corpus)
    secret = str(corpus.parent / "secret.txt")
    result = call(tools, "read_file", path=secret, start=1, end=2)
    assert result.error and "outside" in result.error
    assert "top secret" not in result.text


def test_a_symlink_that_points_outside_is_refused(corpus):
    link = corpus / "docs" / "link.md"
    try:
        os.symlink(corpus.parent / "secret.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow symbolic links")
    tools, _ = tools_for(corpus)
    result = call(tools, "read_file", path="docs/link.md", start=1, end=2)
    assert result.error and "outside" in result.error
    assert "top secret" not in result.text


def test_list_symbols_lists_classes_functions_and_methods(corpus):
    tools, _ = tools_for(corpus)
    result = call(tools, "list_symbols", path="fastapi/routing.py")
    assert result.text.split("\n") == [
        "class Router (lines 5-15)",
        "  def add (lines 8-12)",
        "  async def fetch (lines 14-15)",
        "def helper (lines 19-20)",
        "async def go (lines 23-24)",
    ]


def test_list_symbols_edge_cases(corpus):
    tools, _ = tools_for(corpus)
    assert call(tools, "list_symbols", path="fastapi/empty.py").text == "No symbols."
    assert call(tools, "list_symbols", path="fastapi/broken.py").error
    assert call(tools, "list_symbols", path="docs/a.md").error  # not a .py file
    assert call(tools, "list_symbols", path="fastapi/missing.py").error
