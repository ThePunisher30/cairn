"""How far can the LLM judge be trusted? Grade a sample by hand and measure the agreement (Phase 7).

    python scripts/judge_agreement.py --run data/gen_runs/NAME.jsonl --export-sample 15 --seed 0 \
        --out learn/grading-sample.md
    ... grade each answer in the markdown file yourself, fill in the labels file next to it ...
    python scripts/judge_agreement.py --run data/gen_runs/NAME.jsonl \
        --labels learn/grading-sample.labels.json

The sample file shows the question, the reference answer, the answer and the chunks, and NOT the
judge's verdict, so that you grade blind. `correct` and `faithful` are 1 or 0 (null = skip).
Provided, not part of the exercise; it uses `agreement_rate` and `cohens_kappa`.
"""

import argparse
import json
import random
import sys
from pathlib import Path

from cairn.evaluation.generation import agreement_rate, cohens_kappa
from cairn.jsonl import read_jsonl


def gradable(rows):
    return [r for r in rows if r["answerable"] and not r["abstained"] and r["correct"] is not None]


def export_sample(rows, count, seed, out):
    pool = gradable(rows)
    if count > len(pool):
        raise ValueError(
            f"only {len(pool)} answers can be graded, --export-sample asks for {count}"
        )
    sample = random.Random(seed).sample(pool, count)
    lines = ["# Grade these answers (blind)\n",
             "correct: does the answer state the same facts as the reference? (1 or 0)\n",
             "faithful: is every claim in the answer supported by the chunks? (1, 0, or null if "
             "there are no claims)\n"]  # fmt: skip
    for number, r in enumerate(sample, start=1):
        lines += [f"\n---\n\n## {number}. {r['id']} [{r['category']}]\n",
                  f"**Question:** {r['question']}\n", f"**Reference:** {r['reference']}\n",
                  f"**Answer:** {r['answer']}\n", "**Chunks the model saw:**\n"]  # fmt: skip
        for c in r["context"]:
            lines.append(f"\n[{c['number']}] {c['path']}:{c['start_line']}-{c['end_line']}\n")
            lines.append("```\n" + c["text"] + "\n```\n")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(lines), encoding="utf-8")
    labels = out.with_suffix(".labels.json")
    labels.write_text(
        json.dumps({r["id"]: {"correct": None, "faithful": None} for r in sample}, indent=2),
        encoding="utf-8",
    )
    return out, labels


def compare(rows, labels):
    by_id = {r["id"]: r for r in rows}
    report = {}
    for field in ("correct", "faithful"):
        human, judge, ids = [], [], []
        for qid, label in labels.items():
            if qid not in by_id:
                raise ValueError(f"{qid} is not in the run file")
            h, j = label.get(field), by_id[qid][field]
            if h is None or j is None:
                continue
            human.append(bool(h))
            judge.append(bool(j))
            ids.append(qid)
        if human:
            report[field] = {
                "n": len(human),
                "agreement": agreement_rate(human, judge),
                "kappa": cohens_kappa(human, judge),
                "disagreements": [
                    (i, h, j) for i, h, j in zip(ids, human, judge, strict=True) if h != j
                ],
            }
    return report


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Judge agreement with a hand-graded sample.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--export-sample", type=int, default=None, metavar="N")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("grading-sample.md"))
    parser.add_argument("--labels", type=Path, default=None)
    args = parser.parse_args(argv)
    if (args.export_sample is None) == (args.labels is None):
        parser.error("use exactly one of --export-sample and --labels")
    try:
        rows = list(read_jsonl(args.run))
        if args.export_sample is not None:
            out, labels = export_sample(rows, args.export_sample, args.seed, args.out)
            print(f"wrote {out} and {labels}")
            return 0
        report = compare(rows, json.loads(args.labels.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if not report:
        print("no label pair to compare (fill in the labels file)")
        return 1
    for field, r in report.items():
        print(f"{field}: {r['n']} answers, agreement {r['agreement']:.3f}, kappa {r['kappa']:.3f}")
        for qid, human, judge in r["disagreements"]:
            print(f"  {qid}: you said {int(human)}, the judge said {int(judge)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
