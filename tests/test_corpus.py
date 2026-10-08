"""Phase 0: downloading the corpus must be repeatable, safe, and testable without the internet.

These tests build a tiny git repository on disk (with tags 1.0.0 and 2.0.0) and fetch from that
instead of GitHub.
"""
import subprocess
import sys

import pytest

from cairn import config
from cairn.corpus import CorpusError, fetch_corpus

SCRIPT = config.PROJECT_ROOT / "scripts" / "fetch_corpus.py"


def git(*args, cwd):
    cmd = ["git", "-c", "user.name=t", "-c", "user.email=t@example.com",
           "-c", "commit.gpgsign=false", *args]
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def origin(tmp_path):
    repo = tmp_path / "origin"
    (repo / "docs").mkdir(parents=True)
    git("init", "-b", "main", cwd=repo)
    (repo / "docs" / "intro.md").write_text("# Intro\nversion one\n", encoding="utf-8")
    git("add", ".", cwd=repo)
    git("commit", "-m", "v1", cwd=repo)
    git("tag", "1.0.0", cwd=repo)
    (repo / "docs" / "intro.md").write_text("# Intro\nversion two\n", encoding="utf-8")
    git("commit", "-am", "v2", cwd=repo)
    git("tag", "2.0.0", cwd=repo)
    return repo


def text_of(dest):
    return (dest / "docs" / "intro.md").read_text(encoding="utf-8")


def test_fetch_checks_out_the_requested_tag(origin, tmp_path):
    dest = tmp_path / "raw" / "fastapi"            # the parent folder "raw" does not exist yet
    result = fetch_corpus(origin.as_uri(), "1.0.0", dest)
    assert result == dest
    assert "version one" in text_of(dest)
    assert git("rev-parse", "HEAD", cwd=dest) == git("rev-list", "-n", "1", "1.0.0", cwd=origin)


def test_second_call_is_a_no_op_that_keeps_existing_files(origin, tmp_path):
    dest = tmp_path / "fastapi"
    fetch_corpus(origin.as_uri(), "1.0.0", dest)
    marker = dest / "marker.txt"
    marker.write_text("keep me", encoding="utf-8")
    fetch_corpus(origin.as_uri(), "1.0.0", dest)   # must not fail and must not re-clone
    assert marker.exists()


def test_existing_checkout_of_a_different_tag_is_refused_and_left_alone(origin, tmp_path):
    dest = tmp_path / "fastapi"
    fetch_corpus(origin.as_uri(), "1.0.0", dest)
    with pytest.raises(CorpusError, match="2.0.0"):
        fetch_corpus(origin.as_uri(), "2.0.0", dest)
    assert "version one" in text_of(dest)           # nothing was deleted or changed


def test_existing_folder_that_is_not_a_checkout_is_refused_and_left_alone(origin, tmp_path):
    dest = tmp_path / "fastapi"
    dest.mkdir()
    (dest / "junk.txt").write_text("x", encoding="utf-8")
    with pytest.raises(CorpusError):
        fetch_corpus(origin.as_uri(), "1.0.0", dest)
    assert (dest / "junk.txt").exists()


def test_unknown_tag_raises_corpus_error_and_leaves_no_folder_behind(origin, tmp_path):
    dest = tmp_path / "fastapi"
    with pytest.raises(CorpusError, match="9.9.9"):
        fetch_corpus(origin.as_uri(), "9.9.9", dest)
    assert not dest.exists()


def test_unreachable_repository_raises_corpus_error(tmp_path):
    with pytest.raises(CorpusError):
        fetch_corpus((tmp_path / "does-not-exist").as_uri(), "1.0.0", tmp_path / "dest")


def run_script(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True)


def test_script_fetches_and_exits_zero(origin, tmp_path):
    dest = tmp_path / "fastapi"
    result = run_script("--repo", origin.as_uri(), "--tag", "1.0.0", "--dest", str(dest))
    assert result.returncode == 0, result.stderr
    assert "version one" in text_of(dest)


def test_script_failure_is_a_clean_one_line_error_not_a_traceback(origin, tmp_path):
    result = run_script("--repo", origin.as_uri(), "--tag", "9.9.9", "--dest", str(tmp_path / "d"))
    assert result.returncode == 1
    assert "9.9.9" in result.stderr
    assert "Traceback" not in result.stderr
