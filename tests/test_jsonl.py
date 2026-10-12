"""Phase 1, step 1: your Lesson 2 JSONL helpers, promoted into the package."""

import pytest

from cairn.jsonl import JsonlError, read_jsonl, write_jsonl


def test_jsonl_error_is_an_exception():
    assert issubclass(JsonlError, Exception)


def test_roundtrip_with_unicode_and_new_folder(tmp_path):
    path = tmp_path / "deep" / "dir" / "rows.jsonl"
    rows = [{"id": 1, "text": "café ✓"}, {"id": 2, "text": "plain"}]
    assert write_jsonl(path, rows) == 2
    assert list(read_jsonl(path)) == rows


def test_write_accepts_a_generator(tmp_path):
    path = tmp_path / "g.jsonl"
    assert write_jsonl(path, ({"i": i} for i in range(5))) == 5
    assert len(path.read_text(encoding="utf-8").splitlines()) == 5


def test_read_is_lazy(tmp_path):
    path = tmp_path / "x.jsonl"
    write_jsonl(path, [{"a": 1}, {"a": 2}])
    it = read_jsonl(path)
    assert iter(it) is it  # a generator is its own iterator
    assert next(it) == {"a": 1}


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "blank.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert list(read_jsonl(path)) == [{"a": 1}, {"a": 2}]


def test_bad_json_names_the_file_and_the_line(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(JsonlError, match=r"bad\.jsonl: line 2"):
        list(read_jsonl(path))
