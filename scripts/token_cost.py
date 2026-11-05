"""Token counts and cost estimates from a gen_eval run file (Phase 7, no LLM).

    python scripts/token_cost.py data/gen_runs/gemma4-hybrid.jsonl

1. Our token proxy (`cairn.tokens.count_tokens`, a regex) against what the model reported for the
   same prompt (`prompt_eval_count`, stored as usage[0] in each row). The prompt is rebuilt from
   the stored chunks with the same template the run used.
2. What the same 50 questions would have cost with the paid models in eval/llm_prices.json, taking
   the token counts of this run as given. The paid models use other tokenizers, so this is an
   estimate to within tens of percent, not a bill. The local run itself cost $0.
"""

import json
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

from cairn import config
from cairn.generation.factory import load_prices
from cairn.generation.prompts import format_chunks, load_prompt, neutralize
from cairn.tokens import count_tokens


def rebuild_prompt(row, template):
    chunks = tuple(
        SimpleNamespace(
            number=c["number"],
            text=c["text"],
            hit=SimpleNamespace(path=c["path"], start_line=c["start_line"], end_line=c["end_line"]),
        )
        for c in row["context"]
    )
    system, user = template.render(
        chunks=format_chunks(SimpleNamespace(chunks=chunks)), question=neutralize(row["question"])
    )
    return system + "\n" + user


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = (argv if argv is not None else sys.argv)[1:]
    if len(args) != 1:
        print("usage: token_cost.py <run file>", file=sys.stderr)
        return 1
    rows = [json.loads(line) for line in Path(args[0]).read_text("utf-8").splitlines() if line]
    template = load_prompt("answer")  # the run used the latest version
    pairs = []
    for row in rows:
        reported = row["usage"][0]
        if reported > 0:
            pairs.append((row["id"], count_tokens(rebuild_prompt(row, template)), reported))
    if not pairs:
        print("no row reports input tokens")
        return 1

    ratios = [reported / ours for _, ours, reported in pairs]
    print(f"1. prompt tokens: ours vs reported by the model ({len(pairs)} prompts)")
    print(
        f"   reported / ours   median {statistics.median(ratios):.2f}   "
        f"min {min(ratios):.2f}   max {max(ratios):.2f}"
    )
    mean_abs = statistics.mean(abs(r - 1) for r in ratios)
    print(f"   mean absolute error of our count: {mean_abs:.1%}")
    odd = [(i, o, r) for i, o, r in pairs if r / o < 0.5]
    if odd:
        print("   far below ours (probably a cached prompt, only the new tokens are counted):")
        for i, o, r in odd:
            print(f"     {i}: ours {o}, reported {r}")

    prices = load_prices(config.PROJECT_ROOT / "eval" / "llm_prices.json")
    n_in = sum(r["usage"][0] for r in rows)
    n_out = sum(r["usage"][1] for r in rows)
    # use our count where the model reported nothing or a cached count
    print(f"\n2. {len(rows)} questions: {n_in} input tokens, {n_out} output tokens (as reported)")
    print(f"   {'model':20} {'total $':>9} {'per question $':>15}")
    for model, (p_in, p_out) in prices.items():
        total = (n_in * p_in + n_out * p_out) / 1_000_000
        print(f"   {model:20} {total:9.4f} {total / len(rows):15.5f}")
    print("   (answering only; the judge would add about the same again. The paid models use other")
    print("    tokenizers, so these are estimates.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
