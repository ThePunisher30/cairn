"""Prepare the folder you push to a Hugging Face Space (Docker SDK) to host the public demo.

    python scripts/make_space.py --out ../cairn-space
    python scripts/make_space.py --out ../cairn-space --dense     # hybrid search (needs more RAM)

A Space is a git repository whose README.md starts with a small block of settings, and whose
Dockerfile it builds. This script copies the files the image needs (not the data: the Dockerfile
downloads and indexes the corpus itself) and writes that README. Then, in the new folder:

    git init -b main && git add . && git commit -m "Cairn demo"
    git remote add space https://huggingface.co/spaces/<you>/cairn
    git push space main              # asks for your Hugging Face username and an access token

See docs/deploy.md. Nothing here talks to the network.
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

from cairn import config

ROOT = config.PROJECT_ROOT

FRONT_MATTER = """---
title: Cairn
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Ask the FastAPI docs and code, with passages you can check
---

# Cairn

A retrieval engine over the FastAPI 0.143.0 docs and source, built from scratch.
This Space runs it without a language model: you get the passages that match your question,
with the exact file and lines, and a viewer for them.

Source, measurements and write-ups: {repo}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face Space folder.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dense", action="store_true", help="build with the embedding model")
    parser.add_argument("--repo", default="https://github.com/ThePunisher30/cairn")
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        print(f"error: {args.out} exists and is not empty", file=sys.stderr)
        return 1
    args.out.mkdir(parents=True, exist_ok=True)

    for name in ("Dockerfile", ".dockerignore", "pyproject.toml"):
        shutil.copyfile(ROOT / name, args.out / name)
    shutil.copytree(
        ROOT / "src", args.out / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(
        ROOT / "scripts",
        args.out / "scripts",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (args.out / "eval").mkdir()
    shutil.copyfile(ROOT / "eval" / "llm_prices.json", args.out / "eval" / "llm_prices.json")

    if args.dense:
        dockerfile = args.out / "Dockerfile"
        text = dockerfile.read_text(encoding="utf-8")
        dockerfile.write_text(
            re.sub(r"ARG WITH_DENSE=0", "ARG WITH_DENSE=1", text), encoding="utf-8"
        )

    (args.out / "README.md").write_text(FRONT_MATTER.format(repo=args.repo), encoding="utf-8")
    print(f"wrote {args.out} ({'hybrid' if args.dense else 'BM25 only'}); now push it to the Space")
    return 0


if __name__ == "__main__":
    sys.exit(main())
