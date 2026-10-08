# Phase 0: checkpoint notes

## 1. Why a `src/` layout and an editable install?

With the code in `src/cairn/`, Python can only import `cairn` if the package was actually installed.
Without `src/`, running from the repo root would let Python import the loose folder directly, so
tests could pass even when the installed package is broken (a missing file, a wrong `pyproject.toml`).
The `src/` layout makes the tests exercise the package the way a real user gets it.

An editable install (`pip install -e .`) installs `cairn` as a link to my `src/` folder instead of a
frozen copy. When I edit a file the change is live straight away, with no reinstall. (Dependency
versions are a separate thing, handled by the `dependencies` list in `pyproject.toml`.)

## 2. Why does `load_settings` take a dictionary instead of always reading `os.environ`?

It is about testing, not speed (`os.environ` is already dictionary-like and instant to look up).
If the function always read the real environment, every test would depend on whatever is set on my
machine (including my real `ANTHROPIC_API_KEY`), and a test would have to change the real environment
to try an input, which could leak into other tests. Passing the dictionary in lets a test say "pretend
the environment is exactly this" with no side effects. Handing a function what it needs, instead of
letting it grab it, is called dependency injection.

## 3. Why is the API key excluded from `repr`, and what else could print it?

A leaked key can be misused, and someone could run up costs on my account. `repr=False` hides it at
the type level, so it cannot leak by accident however careless the debugging line is.

Things that could print the whole settings object: a `print(settings)` or a log line while debugging;
an error message or traceback that includes the object; a debugger or IDE variable view; pytest
showing the object when an assertion fails; error-reporting tools that capture crashes; and me
pasting terminal output into a chat or a GitHub issue. Hiding the field stops all of these at once.

## 4. Why does `fetch_corpus` refuse an existing folder at the wrong tag instead of fixing it?

The function cannot know what is in that folder. It might be a large download I want to keep, or
files someone edited by hand. Deleting is irreversible, and silently "fixing" it could change the
corpus without me noticing. Stopping with a clear message ("delete it and run again") lets a human
decide, and nothing is lost.

## 5. Why pin a tag (`0.143.0`) instead of downloading "latest"?

For reproducibility. The eval questions will have their gold answers saved as file and line
numbers. If the FastAPI docs changed between downloads, those lines would shift, the labels would
silently become wrong, and results from different days could not be compared. A pinned tag means
everyone gets the identical corpus, so the numbers mean the same thing.
