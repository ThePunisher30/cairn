"""Phase 2, step 3: finding the files that belong to the corpus and reading them."""

import pytest

from cairn.ingest.documents import load_documents


def write(root, rel, text="x\n", newline=None):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline=newline) as f:
        f.write(text)


@pytest.fixture
def repo(tmp_path):
    write(tmp_path, "docs/en/docs/index.md", "# Home\n")
    write(tmp_path, "docs/en/docs/tutorial/first-steps.md", "# First\n")
    write(tmp_path, "docs/en/docs/release-notes.md", "# Changelog\n")  # out: changelog noise
    write(tmp_path, "docs/de/docs/index.md", "# Startseite\n")  # out: a translation
    write(tmp_path, "docs/en/docs/img/logo.png", "not text")  # out: not markdown
    write(tmp_path, "fastapi/applications.py", "app = 1\n")
    write(tmp_path, "fastapi/security/oauth2.py", "x = 1\n")
    write(tmp_path, "fastapi/__pycache__/applications.cpython-312.pyc", "junk")  # out
    write(tmp_path, "docs_src/background_tasks/tutorial001_py310.py", "y = 2\n")
    write(tmp_path, "tests/test_app.py", "def test(): pass\n")  # out: the repo's own tests
    write(tmp_path, "README.md", "# Readme\n")  # out: not in a corpus folder
    return tmp_path


def test_only_the_corpus_folders_are_loaded(repo):
    paths = [d.path for d in load_documents(repo)]
    assert paths == [
        "docs/en/docs/index.md",
        "docs/en/docs/tutorial/first-steps.md",
        "docs_src/background_tasks/tutorial001_py310.py",
        "fastapi/applications.py",
        "fastapi/security/oauth2.py",
    ]


def test_documents_come_back_sorted_by_path(repo):
    paths = [d.path for d in load_documents(repo)]
    assert paths == sorted(paths)


def test_kind_follows_the_file_type(repo):
    kinds = {d.path: d.kind for d in load_documents(repo)}
    assert kinds["docs/en/docs/index.md"] == "prose"
    assert kinds["fastapi/applications.py"] == "code"
    assert kinds["docs_src/background_tasks/tutorial001_py310.py"] == "code"


def test_text_is_read_as_utf8(repo):
    write(repo, "docs/en/docs/unicode.md", "café ✓\n")
    docs = {d.path: d for d in load_documents(repo)}
    assert docs["docs/en/docs/unicode.md"].text == "café ✓\n"


def test_windows_line_endings_become_plain_newlines(repo):
    write(repo, "docs/en/docs/crlf.md", "a\r\nb\r\n", newline="")
    docs = {d.path: d for d in load_documents(repo)}
    assert docs["docs/en/docs/crlf.md"].text == "a\nb\n"
    assert docs["docs/en/docs/crlf.md"].lines == ["a", "b"]


def test_a_missing_corpus_folder_is_skipped_not_an_error(tmp_path):
    write(tmp_path, "fastapi/only.py", "x = 1\n")
    assert [d.path for d in load_documents(tmp_path)] == ["fastapi/only.py"]


def test_an_empty_folder_gives_no_documents(tmp_path):
    assert load_documents(tmp_path) == []
