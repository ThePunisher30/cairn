"""Finding the corpus files and reading them into Documents."""

from pathlib import Path

from cairn.types import Document

# (folder, file suffix, kind): the whole definition of "what is in the corpus"
SCOPE = (
    ("docs/en/docs", ".md", "prose"),
    ("fastapi", ".py", "code"),
    ("docs_src", ".py", "code"),
)
EXCLUDED = ("docs/en/docs/release-notes.md",)


def load_documents(root) -> list[Document]:
    root = Path(root)
    documents = []
    for folder, suffix, kind in SCOPE:
        base = root / folder
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or not path.suffix == suffix:
                continue
            relative = path.relative_to(root).as_posix()
            if relative in EXCLUDED:
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            documents.append(Document(relative, text, kind))
    return sorted(documents, key=lambda doc: doc.path)
