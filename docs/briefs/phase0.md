# Phase 0 brief: Foundations

**Goal:** anyone (including you in three months) can clone this repo, run two commands, and have a working project with passing tests, a linter, and the corpus downloaded the same way every time.

**Time:** 2-3 days. **You write everything below. The tests in `tests/` are already written and currently fail.**

**Done when**
- `C:\mrv\Scripts\python -m pytest -q` is green (34 tests in `tests/`)
- `C:\mrv\Scripts\python -m ruff check .` reports no problems
- `C:\mrv\Scripts\python scripts\fetch_corpus.py` downloads FastAPI 0.143.0 into `data\raw\fastapi`, and running it a second time does nothing and does not fail
- the work is on a branch, in small commits, pushed to a public GitHub repo, merged through a pull request

---

## Step 1. Git repo, `.gitignore`, `.env.example`
```powershell
cd path\to\cairn
git init -b main
```
- **`.gitignore`** must contain these lines (tests check them): `.venv/`, `__pycache__/`, `.env`, `data/raw/`. Also add `*.pyc`, `.pytest_cache/`, `.ruff_cache/`, `*.egg-info/`, and `learn/` (your lesson exercises are private study notes, not part of the project).
- **`.env.example`**: a template listing the environment variables, **with no real values**. It must mention `ANTHROPIC_API_KEY` (empty after the `=`).
  Why: the real key goes in a file called `.env` (ignored by git) or your system environment. The example file shows teammates what to set, without leaking anything.

## Step 2. Folders
Create: `src/cairn/`, `eval/`, `bench/`, `docs/` (exists), `scripts/`, `data/`. Git doesn't track empty folders, so put an empty file called `.gitkeep` in the ones with nothing else yet (`eval`, `bench`, `data`).

## Step 3. `pyproject.toml` and the package
This one file describes your project. Required content (the test reads it with `tomllib`):
- `[build-system]`: use `setuptools` (`requires = ["setuptools>=68"]`, `build-backend = "setuptools.build_meta"`)
- `[project]`: `name = "cairn"`, a `version`, `requires-python = ">=3.11"`, `dependencies = ["numpy>=2.0"]`
  (Add other dependencies **only when the code needs them**. That keeps the install honest.)
- `[project.optional-dependencies]` with a `dev` list containing `pytest` and `ruff`
- `[tool.setuptools.packages.find]` with `where = ["src"]`
- `[tool.pytest.ini_options]` with `testpaths = ["tests"]`
- `[tool.ruff]` with `line-length = 100` and `extend-exclude = ["learn"]` (so the linter ignores your study folder). Optionally add a `[tool.ruff.lint]` section with `select = ["E", "F", "I", "B"]` (style errors, bugs, import sorting, bug-prone patterns).

Then create `src/cairn/__init__.py` containing `__version__ = "0.1.0"`, and install the project **in editable mode**:
```powershell
C:\mrv\Scripts\python -m pip install -e ".[dev]"
```
*Editable* means Python imports your code straight from `src/`, so edits take effect immediately without reinstalling.

> **Why a `src/` folder?** With `src/cairn/`, Python can only import `cairn` if it was properly installed. That means your tests exercise the installed package, the way a user would, and not whatever happens to be in the current folder.

## Step 4. `src/cairn/config.py`
Settings come from **environment variables**. Build:
- `PROJECT_ROOT`: a `Path` to the repo root, computed from `__file__` (hint: `Path(__file__).resolve().parents[...]`; count how many levels up `src/cairn/config.py` is)
- `class ConfigError(Exception)`
- a **frozen dataclass** `Settings` with these fields:

| Field | Env variable | Default | Rule |
|---|---|---|---|
| `data_dir: Path` | `CAIRN_DATA_DIR` | `PROJECT_ROOT / "data"` | |
| `embedding_model: str` | `CAIRN_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | |
| `top_k: int` | `CAIRN_TOP_K` | `5` | must be an integer >= 1 |
| `log_level: str` | `CAIRN_LOG_LEVEL` | `INFO` | one of DEBUG/INFO/WARNING/ERROR, case-insensitive, stored upper-case |
| `corpus_repo_url: str` | `CAIRN_CORPUS_REPO_URL` | `https://github.com/fastapi/fastapi.git` | |
| `corpus_tag: str` | `CAIRN_CORPUS_TAG` | `0.143.0` | |
| `anthropic_api_key: str \| None` | `ANTHROPIC_API_KEY` | `None` | **must not appear in `repr`/`str`** |

- `load_settings(env: Mapping[str, str] | None = None) -> Settings`
  - if `env` is `None`, use `os.environ`; otherwise use the dict you were given
  - a variable that is missing, empty, or only whitespace counts as **unset** (use the default)
  - invalid values raise `ConfigError` with a message that **names the variable** (for example `"CAIRN_TOP_K must be an integer, got 'abc'"`)

Things to look up: `dataclasses.field(repr=False)`, `int("2.5")` (what does it raise?), `str.strip()`, `str.upper()`.

> **Why does `load_settings` take a dictionary?** So tests can pass in any environment they like without touching your real one. Passing a dependency in as an argument, instead of reaching out and grabbing it, is called *dependency injection*, and you'll use it everywhere in this project.

## Step 5. `src/cairn/corpus.py`
```python
class CorpusError(Exception): ...

def fetch_corpus(repo_url: str, tag: str, dest: Path) -> Path: ...
```
Behaviour (each line is tested):
1. If `dest` doesn't exist: create its parent folders, then clone `repo_url` at `tag` into `dest`. Use a **shallow clone** (`git clone --depth 1 --branch <tag> <url> <dest>`), which downloads one snapshot instead of the full history. Return `dest`.
2. If `dest` already exists and is a git checkout **at that tag**: do nothing and return `dest`. (To compare, ask git for `HEAD`'s commit and for the tag's commit.)
3. If `dest` exists but is checked out at a **different** tag: raise `CorpusError` mentioning the tag, **without deleting or changing anything**.
4. If `dest` exists but is **not** a git checkout: raise `CorpusError`, and leave it alone.
5. If the tag doesn't exist or the URL is unreachable: raise `CorpusError` that mentions the tag, and **leave no half-made `dest` folder behind**.

Use `subprocess.run([...], check=True, capture_output=True, text=True)`, catch `subprocess.CalledProcessError`, and raise your `CorpusError` **`from` the original** so the cause isn't lost (Lesson 2).

> **Why refuse instead of just deleting and re-cloning?** A function that deletes folders when it's "pretty sure" is how people lose work. It's safer to stop and tell the human what to do.

## Step 6. `scripts/fetch_corpus.py`
A thin command-line wrapper around `fetch_corpus`, using `argparse`:
- options `--repo`, `--tag`, `--dest`, defaulting to the values from `load_settings()` (`dest` defaults to `<data_dir>/raw/fastapi`)
- on success: print the destination path, exit code 0
- on `CorpusError`: print **one line** like `error: ...` to **stderr** (not a traceback), and exit with code 1
- use the standard pattern: `def main(argv=None) -> int:` and `if __name__ == "__main__": sys.exit(main())`

## Step 7. Run all the checks
```powershell
C:\mrv\Scripts\python -m pytest -q
C:\mrv\Scripts\python -m ruff check .
C:\mrv\Scripts\python scripts\fetch_corpus.py
```
Then **explore what you downloaded**: how many `.md` files are under `data\raw\fastapi\docs\en\docs`? How many `.py` files under `data\raw\fastapi\fastapi`? Write the two counts in `docs/results.md` (create it). That file starts life as your lab notebook.

## Step 8. Git workflow (this is what recruiters see)
A pull request needs a `main` branch that already has a commit to merge into. So:
1. After Steps 1-2, make the **first commit on `main`** containing only the safety files (`.gitignore`, `.env.example`), message `Add .gitignore and env template`.
2. **Then** create your working branch and do everything else there:
```powershell
git switch -c phase-0-foundations
```
Make several **small commits** with clear messages as you go (for example `Add project folders`, `Add pyproject and package skeleton`, `Add settings loader`, `Add corpus fetcher`).
Before the first push, **tell me and I'll check what git is about to publish** (`git status`, `git log`). The repo will be public, so I want to confirm no secrets or private files are included. Then:
```powershell
gh repo create cairn --public --source . --remote origin   # creates the empty GitHub repo and links it
git push -u origin main                                       # publish main first (the root commit)
git push -u origin phase-0-foundations                        # then your branch
```
Open a pull request from `phase-0-foundations` into `main` with a short description, and merge it.

---

## Checkpoint questions (answer in your own words in `docs/notes/phase0.md`)
1. Why does the project use a `src/` layout and an editable install?
2. Why does `load_settings` accept a dictionary instead of always reading `os.environ`?
3. Why is the API key excluded from `repr`, and what else in a program might accidentally print it?
4. Why does `fetch_corpus` refuse to touch an existing folder at the wrong tag, instead of fixing it?
5. Why pin to a tag (`0.143.0`) rather than download "latest"? What would go wrong for your eval set if the corpus changed?

## Hint ladder reminder
Stuck? Tell me which step and what the test or error says. I'll start with a nudge, then pseudocode, then a short snippet.
