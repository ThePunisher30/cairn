import subprocess
from pathlib import Path


class CorpusError(Exception):
    pass


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as e:
        raise CorpusError(f"git {' '.join(args)} failed: {e.stderr.strip()}") from e
    return result.stdout.strip()


def fetch_corpus(repo_url: str, tag: str, dest: Path) -> Path:
    dest = Path(dest)

    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            _git("clone", "--depth", "1", "--branch", tag, repo_url, str(dest))
        except CorpusError as e:
            raise CorpusError(f"could not fetch tag {tag} from {repo_url}: {e}") from e
    else:
        if not (dest / ".git").exists():
            raise CorpusError(f"destination {dest} exists but is not a git checkout")

        wrong_version = (
            f"destination {dest} is not checked out at tag {tag}; delete it and run again"
        )
        try:
            head = _git("rev-parse", "HEAD", cwd=dest)
            tagged_commit = _git(
                "rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}", cwd=dest
            )
        except CorpusError as e:
            raise CorpusError(wrong_version) from e
        if head != tagged_commit:
            raise CorpusError(wrong_version)
    return dest

