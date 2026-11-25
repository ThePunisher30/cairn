"""How well does a semantic cache tell "the same question" from "a similar question"? (Phase 9)

For each dev question there is a paraphrase (the same question in other words: a hit is right) and
a near-miss (almost the same words but another meaning: a hit is WRONG, it would serve a stale or
unrelated answer). For each threshold the script prints how many paraphrases and how many
near-misses would have been answered from the cache. It also reports the closest pair among the
different dev questions themselves.

    python scripts/semantic_cache_eval.py

The 15 paraphrases and 18 near-misses are written by hand (eval/semantic_cache_pairs.json) and use
only dev questions, never the test split. Small sample: 15 and 18 pairs, one embedding model.
"""

import json
import sys

import numpy as np

from cairn import config
from cairn.dense.models import load_embedder

PAIRS = config.PROJECT_ROOT / "eval" / "semantic_cache_pairs.json"

THRESHOLDS = (0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98)


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [
        json.loads(line)
        for line in (config.PROJECT_ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")
    ]
    dev = {r["id"]: r["question"] for r in rows if r["split"] == "dev"}
    pairs = json.loads(PAIRS.read_text(encoding="utf-8"))
    PARAPHRASES, NEAR_MISSES = pairs["paraphrases"], pairs["near_misses"]
    for qid, _ in PARAPHRASES + NEAR_MISSES:
        if qid not in dev:
            print(f"error: {qid} is not a dev question", file=sys.stderr)
            return 1
    embedder = load_embedder("bge-small")

    def similarity(pairs):
        a = np.array([embedder.embed_query(dev[qid]) for qid, _ in pairs])
        b = np.array([embedder.embed_query(text) for _, text in pairs])
        return (a * b).sum(axis=1)

    para = similarity(PARAPHRASES)
    miss = similarity(NEAR_MISSES)
    all_dev = np.array([embedder.embed_query(q) for q in dev.values()])
    gram = all_dev @ all_dev.T
    np.fill_diagonal(gram, -1)
    ids = list(dev)
    i, j = np.unravel_index(np.argmax(gram), gram.shape)

    print(f"paraphrases: {len(para)}  similarity min {para.min():.3f} median {np.median(para):.3f}")
    print(f"near-misses: {len(miss)}  similarity min {miss.min():.3f} median {np.median(miss):.3f}")
    print(f"closest two different dev questions: {gram[i, j]:.3f} ({ids[i]}, {ids[j]})\n")
    print("threshold | paraphrases hit | near-misses hit (false hits)")
    for t in THRESHOLDS:
        print(f"{t:>9.2f} | {int((para >= t).sum()):>6}/{len(para)} {(para >= t).mean():>6.0%}"
              f" | {int((miss >= t).sum()):>6}/{len(miss)} {(miss >= t).mean():>6.0%}")  # fmt: skip
    print("\nnear-misses, most similar first (a false hit at any threshold below the number):")
    for k in np.argsort(-miss)[:8]:
        qid, text = NEAR_MISSES[k]
        print(f"  {miss[k]:.3f}  {qid}: {dev[qid][:70]}\n          vs: {text[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
