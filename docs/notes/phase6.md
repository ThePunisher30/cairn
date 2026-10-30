# Phase 6: checkpoint notes

The numbers quoted here come from running the real code (see `docs/results.md`, section "Phase 6"). "Dev" is
the 40 development questions (one question is 0.025), "test" the 20 held-out ones (one question is 0.05, run
once).

## 1. Why can't BM25 scores and cosine similarities be added directly? Show it with numbers from your own data.

They are measured in different units, and BM25's is not even fixed. Over the 40 dev questions the best BM25
score has a median of 30.4 (smallest 16.0, largest 66.0), because it grows with the number of matching rare
words in the question. The best cosine similarity has a median of 0.789 (0.678 to 0.875). Across the top 50 of
a list, BM25 falls by 15.4 from first to fiftieth (median) and cosine by 0.115, so BM25 moves a sum about 130
times more than dense does.

The raw-sum row shows the result. Its MRR is 0.672 against 0.660 for BM25 and 0.711 for dense, its hit@10 is
exactly BM25's 0.800, and its top 10 is identical to BM25's for 13 of the 40 questions. Dense only reorders
chunks whose BM25 scores are nearly equal (on q001, BM25 ranks 2 to 5 score 31.7, 31.3, 31.2 and 31.1, while
dense's gaps are 0.003 to 0.03). On q016 dense has the gold at rank 1, but the raw sum returns BM25's favourite,
which is not the gold; min-max returns the gold. A chunk that only dense finds scores below 1 and BM25's 50th
result is typically near 15, so it can never reach the top 10 (an argument from the ranges, not a direct check).

## 2. Why does RRF need no score normalisation? What does `k` do, and what would `k = 0` and `k = 1000` do to the result?

RRF never looks at a score: a chunk at rank `r` in a list earns `1 / (k + r)`, and the points of the lists are
added. Ranks are the same kind of number in every list, so the two scales never meet.

`k` sets how steeply the points fall. At `k` = 0 first place is worth twice second place (1 against 1/2); at
`k` = 1, 1.5 times; at `k` = 60, 1.6% more (1/61 against 1/62); at `k` = 1000 about 0.1%, so the points are
almost flat and a chunk that appears in both lists beats a chunk that is first in only one. A small `k` lets one
list's first place count for a lot, a large one rewards agreement between the lists. On dev, `k` = 1 gave MRR
0.761 and hit@5 0.850, `k` = 60 gave 0.732 and 0.800, and 60, 100 and 1000 gave identical results (the order
stopped changing). I did not run `k` = 0. Every constant was within 0.05 of the best RRF cell, so I would not
read the order of the small values as a real difference.

## 3. What is the weakness of min-max normalisation on a top-50 list? What did z-score change in your table?

Min-max rescales to the 50 chunks that were returned, not to the whole corpus. So "the worst" is only the 50th
chunk, not a bad one, and "the best" gets 1 even if it is a poor match. One outlier compresses everything else
(scores 100, 10, 9, 8 become 1.00, 0.02, 0.01, 0.00), and a chunk's fused score depends on which other chunks
happened to be in the list. If all scores are equal there is nothing to separate them, and the code gives every
chunk 1.0.

Z-score uses the mean and the standard deviation, so it does not hang on the two end points, but one outlier
still inflates the deviation, and scores can be negative. In my dev table z-score is 0.016 MRR below min-max
(0.750 against 0.766), two questions lower at hit@1 (0.650 against 0.700) and equal at hit@5 (0.850). By
category: semantic 0.700 against 0.775, multi_chunk 0.725 against 0.717, code 0.625 against 0.624. I did not
test whether the difference is real; it is less than one question of MRR.

## 4. Why can a reranker never find a chunk that the first stage missed? What does that make the first stage's `depth` responsible for, and what did your depth experiment show?

The reranker only scores the candidates it is given and returns the best of them in a new order. A chunk that is
not among the candidates is never scored. So the first stage's recall at the pool size is a ceiling: for hybrid
RRF, hit@k of the first stage alone was 0.900 at 10, 0.925 at 20, 0.975 at 30 and 0.975 at 50. The `depth` is
responsible for putting the answer in the pool at all.

The experiment showed more than that: a deeper pool did not help. Depth 10 was as good as or better than 20, 30
and 50 on every dev column (hit@1 0.750, hit@5 0.900, MRR 0.806, against 0.725, 0.875 and 0.789 at depth 30).
hit@10 stayed at 0.900 at every depth although the ceiling rose from 0.900 to 0.975: the three golds that RRF
finds at ranks 11 to 30 end at rank 16 after reranking. The cost grew linearly, about 1.2 s at depth 10 and 4.9 s
at depth 50 (roughly 100 to 125 ms per candidate). The gaps between depths are one question, so I say only that
going beyond 10 bought nothing I could measure.

## 5. Why is a cross-encoder more accurate than comparing two vectors, and why is it too slow to run over the whole corpus? What did it cost in latency here?

Comparing two vectors squeezes the question into one vector and each chunk into another, separately, and the
chunk vector was fixed long before the question existed. A cross-encoder reads the question and one chunk in the
same pass, so every word of the question can interact with every word of the chunk and it can tell that this
chunk answers this question. That is more work per pair and nothing can be precomputed: each (question, chunk)
pair needs a model run, about 70 to 130 ms on this CPU. For 3,267 chunks that would be several minutes per
question, so the cross-encoder only re-scores a short list.

Measured: embedding the question takes 42 ms, BM25 5 ms, the flat search 0.5 ms and the fusion 0.3 ms (48 ms for
the whole first stage); reranking costs 0.7 to 1.2 s at depth 10 and 3.7 to 3.9 s at depth 30, which is 94 to 96%
and about 99% of the total time. Nothing I tried got under 300 ms; the fastest row (463 ms) lost four questions
at hit@1. "More accurate" held on dev for the top rank (hybrid RRF hit@1 26 to 29 questions) but not on test,
where dense alone (MRR 0.659) beat dense + rerank (0.615).

## 6. Which question category improved most from reranking, and which got worse? Why?

`semantic` improved most: BM25 MRR 0.432 to 0.800 and hybrid RRF 0.667 to 0.850 on dev. `multi_chunk` also gained
for dense and RRF (0.700 to 0.803 and 0.695 to 0.808). `code` got worse or stayed flat: 0.548 for all three
reranked rows against 0.550, 0.602 and 0.617 before, and on test 0.200 to 0.267 against 0.319 to 0.433.

What I saw when I read the chunks: 4 of the 5 code questions that fail have a tutorial page as the top hit while
the gold is a `.py` file, and in two examples the gold code chunk gets a negative reranker score (-2.43 and
-1.18) while the docs page gets 6.39 and 4.06. My explanation is that the model was trained on web passages and a
code chunk has few natural-language words, but I did not test it, and in one more example (q026) the gold is
prose and also scores low (-8.8). I also checked that the identical code MRR in three rows is arithmetic, not a
bug: seven of the 10 code questions get the same rank in all three rows, ranks beyond 10 add nothing at k = 10,
and the rest sum to 5.476 over 10.

## 7. What is small-to-big for, and why is `hit@5` of an expanded context not comparable with `hit@5` of chunks? What did expansion cost in tokens for what it gained?

Small chunks match precisely, but the reader of Phase 7 (the LLM) often needs the text around the answer. Small-to-big
retrieves the small chunk and hands over its parent section, grown outward within a token budget.

An expanded context spans the lines of many chunks, so a gold passage next to the hit is counted as found although
the retriever never ranked it. It measures a bigger window, not a better ranking. That is why the table reports
the share of the gold lines contained, and the tokens, next to the hit rate. On hybrid RRF + rerank, top 5
contexts: plain chunks cover 0.802 of the gold lines at 1,351 tokens; `section` mode adds nothing (0.807);
`parent` up to 800 tokens covers 0.832 (+0.030) at 2,889 tokens (2.1 times) and gains one question at hit@5
(0.875 to 0.900); up to 1,600 tokens covers 0.841 at 4,002 tokens (3.0 times). Even then 16% of the gold lines
are not in the context. I kept it available and off by default.

## 8. What was your dev-against-test gap? Which rows gapped most, and how does that relate to how you chose them? Why is it a mistake to change anything after the test run?

MRR, dev to test: dense 0.711 to 0.659 (-0.052), BM25 0.660 to 0.452 (-0.208), hybrid min-max 0.766 to 0.533
(-0.233), RRF 0.732 to 0.495 (-0.237), z-score -0.226, raw sum -0.214, the reranked rows -0.148 to -0.183. The
chosen row (min-max) had hit@1 0.350 against a predicted 0.65 (range 0.50 to 0.80) and MRR 0.533 against 0.70
(range 0.58 to 0.80); hit@5 and hit@10 (0.800 and 0.900) were as predicted. The order of the rows did not survive:
dense, which was below the fused rows on dev, was the best row without a reranker on test.

The fused rows gapped most, and they are the rows whose method I picked on dev (fusion method, normalisation,
with the constant and the weight looked at in a sweep); dense gapped least. That fits selection on 40 questions,
but BM25 also dropped by 0.208, and it has no fusion setting, so selection is not the only explanation. With 20
test questions a gap of 0.20 is four questions, and I did not read the test questions to find out more.

Changing the configuration after seeing the test run would be choosing on the test split. It would then
measure how well I fitted it, not how the system does on questions it has not seen, and I would report an
optimistic number. So the configuration stays as written before the run, the report says that dense did better,
and the second test run, in Phase 11, measures the whole system.
