# Phase 1: checkpoint notes

The numbers quoted here come from running the real code on the real questions (see `docs/results.md`).

## 1. Why is gold defined as file and line spans instead of chunk IDs?

Because in Phase 2 I will try many ways of cutting files into chunks, and every way produces different
chunks with different IDs. If the answer key said "the answer is chunk 17", it would be wrong the moment
the chunker changed. A file and a line range describes where the answer is in the original text, which
never changes, so one answer key works for every chunker.

A retrieved chunk counts as correct when it shares at least one line with a gold span in the same file.
For question q002 (gold: `response-model.md` lines 306-312) this works for any window size: with 20-line
windows the matching chunk is 301-320, with 40-line windows it is 281-320, with 100-line windows it is
301-344. The chunks and their boundaries differ every time; the gold span does not.

## 2. What is the difference between hit@k and recall@k? Give an example where they differ.

hit@k asks "did I find the answer at all in the top k?" and is 1 or 0. recall@k asks "what fraction of
the gold spans did I find in the top k?". They are the same when a question has one gold span, and they
differ when it has two or more.

Example, question q046 (needs `background-tasks.md` 36-46 and `fastapi/background.py` 41-62): a retriever
that finds only the docs passage gets hit@5 = 1.00 but recall@5 = 0.50. One that finds both gets 1.00 for
both. The oracle baseline shows the same thing: its recall@1 is 0.875, not 1.0, because the 10
`multi_chunk` questions have two gold spans and only one result fits at k = 1.

## 3. Give two result lists with the same hit@5 but different MRR.

The answer is in the top 5 in both lists, but at a different position:
- list A has the answer at rank 1: hit@5 = 1.00, reciprocal rank = 1.000 (nDCG@5 = 1.000)
- list B has the answer at rank 5: hit@5 = 1.00, reciprocal rank = 0.200 (nDCG@5 = 0.387)

hit@k does not care where inside the top k the answer is. MRR (the average of 1 / rank of the first
correct result) and nDCG reward putting it higher. A user reads from the top, so list A is much better,
and hit@5 alone cannot see that.

## 4. What is the credit rule, and what would go wrong without it?

Walking down the ranked results, a result earns credit only if it overlaps a gold span that no earlier
result has already covered. Repeats of an already-found span earn nothing.

Without it, a retriever could return the same passage twice (for example lines 10-20 and 21-30 inside one
gold span of 10-40). Both would count as relevant, and nDCG would come out as 1.631, which is above 1 and
impossible for a score that is supposed to be between 0 and 1. With the rule the flags are
`[True, False]` and nDCG is exactly 1.000. It also stops a retriever from gaming recall by returning many
slightly different pieces of the same passage.

## 5. What does the oracle baseline prove about my harness? What does the random baseline prove?

The oracle returns exactly the gold spans, so a correct harness must give it a perfect score. It does
(1.000 on hit@10, MRR and nDCG@10), so the question loader, the metrics and the harness agree with each
other. As a check that this test has teeth, I ran an oracle with a deliberate bug (every span moved 200
lines down) and the harness scored it 0.000: it can tell right from wrong.

The random baseline picks 10 windows out of 10,608, a roughly 0.1-0.3% chance per question. It scored
0.000 on three different seeds, so the questions cannot be answered by luck. First-N (the same few
windows for every question, a German translation page) also scored 0.000, so position alone is no help
either. Together they give me a floor of 0 and a ceiling of 1 to measure real methods against.

## 6. Why is the test set frozen, and what goes wrong if I tune on it?

The 20 test questions are the final exam; the 40 dev questions are the practice problems. If I keep
looking at test results while I change things, I start choosing the settings that happen to suit those
exact 20 questions, and the score stops measuring how well the method works in general.

I simulated it: 40 methods that were all equally good (a 30% true success rate), each tried on 20
questions. The best one scored 55% on those questions, but on fresh questions it scored 29%. It only
looked good because it was picked after seeing the answers. So I tune only on dev, and the test set is
run exactly twice in the whole project (after Phase 6 and at the end). `scripts/run_eval.py` refuses
`--split test` unless I pass `--use-test-set`, and it writes each such run to a log, so the count can be
checked.
