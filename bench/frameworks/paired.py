"""Paired comparisons between rows on the same questions (Phase 11).

For each pair of rows and each split: hit@5 and reciprocal rank (k = 10) per question, the
questions where only one row hit, an exact two-sided sign test on those, and a paired bootstrap
95% interval for the difference in mean hit@5 and in MRR (10,000 resamples, seed 0).

    python bench/frameworks/paired.py

With 40 dev questions one question is 0.025, with 20 test questions 0.05: most differences below
are within the noise, and the output says so with the p value and the interval.
"""

import json
import math
import random
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
PAIRS = [
    ("cairn-hybrid", "cairn-dense"),
    ("cairn-dense", "li-md-code-30"),
    ("cairn-dense", "li-default"),
    ("cairn-hybrid", "li-md-code-30"),
    ("cairn-hybrid", "li-default"),
    ("li-md-code-30", "li-default"),
    ("cairn-bm25", "li-bm25-default"),
    ("cairn-hybrid", "li-fusion-selected-ip"),
]


def overlap(hit, gold):
    return hit[0] == gold[0] and hit[1] <= gold[2] and hit[2] >= gold[1]


def per_question(row, split):
    data = json.loads((RESULTS / f"{split}-{row}.json").read_text(encoding="utf-8"))
    out = {}
    for q in data["per_question"]:
        first = next(
            (i for i, h in enumerate(q["hits"][:10], 1) if any(overlap(h, g) for g in q["gold"])),
            None,
        )
        out[q["id"]] = (first is not None and first <= 5, 1 / first if first else 0.0)
    return out


def sign_test(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def bootstrap(diffs, reps=10_000, seed=0):
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=len(diffs))) / len(diffs) for _ in range(reps))
    return means[int(0.025 * reps)], means[int(0.975 * reps)]


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    report = []
    for split in ("dev", "test"):
        print(f"\n== {split}")
        print(
            f"{'A vs B':44} {'hit@5 A,B':>11} {'A only':>6} {'B only':>6} {'p':>5}  "
            f"{'hit@5 diff [95%]':>22}  {'MRR diff [95%]':>22}"
        )
        for a, b in PAIRS:
            qa, qb = per_question(a, split), per_question(b, split)
            ids = sorted(qa)
            h = [(int(qa[i][0]), int(qb[i][0])) for i in ids]
            r = [qa[i][1] - qb[i][1] for i in ids]
            hd = [x - y for x, y in h]
            only_a = sum(1 for x, y in h if x and not y)
            only_b = sum(1 for x, y in h if y and not x)
            p = sign_test(only_a, only_b)
            hl, hh = bootstrap(hd)
            rl, rh = bootstrap(r)
            n = len(ids)
            ha, hb = sum(x for x, _ in h) / n, sum(y for _, y in h) / n
            hdiff, rdiff = sum(hd) / n, sum(r) / n
            name = f"{a} vs {b}"
            print(
                f"{name:44} {ha:>5.3f},{hb:.3f} {only_a:>6} {only_b:>6} {p:>5.2f}  "
                f"{hdiff:>+6.3f} [{hl:+.2f},{hh:+.2f}]  {rdiff:>+6.3f} [{rl:+.2f},{rh:+.2f}]"
            )
            report.append(
                {
                    "split": split,
                    "a": a,
                    "b": b,
                    "n": n,
                    "a_only": only_a,
                    "b_only": only_b,
                    "sign_test_p": round(p, 4),
                    "hit5_diff": round(hdiff, 4),
                    "hit5_ci": [round(hl, 3), round(hh, 3)],
                    "mrr_diff": round(rdiff, 4),
                    "mrr_ci": [round(rl, 3), round(rh, 3)],
                }
            )
    (RESULTS / "paired.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
