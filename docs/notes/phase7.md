# Phase 7: checkpoint notes

The numbers quoted here come from the real runs (see `docs/results.md`, section "Phase 7"). One model (`gemma4:e4b`, local), one run of 40 answerable and 10 unanswerable dev questions; one answerable question is 0.025.

> **Draft:** these answers were drafted with the assistant from the run data. The brief asks for answers in my own words, so each one should be rewritten in my words (keep the numbers) before the notes count as mine.

## 1. How can a faithful answer still be wrong? Show an example from your own run.

Faithful means every claim is supported by the chunks. It says nothing about whether the chunks answer the question. If retrieval put the wrong chunks in the context, a model that sticks to them gives a faithful and wrong answer.

q050 asks which helpers a WebSocket endpoint can use and what happens when resolving them fails. The answer lists `WebSocket` methods (`receive_text`, `send_text`, `accept`) taken correctly from a chunk about the `WebSocket` class, and never lists `Depends`, `Security`, `Cookie`, `Header`, `Path`, `Query`. Every claim is in a chunk, so by hand grading it is faithful; against the reference it is wrong. q047 is similar: the answer is faithful and half right, because the chunk with the second half (`exception_handlers.py`) was not in the context. The reverse also exists: a correct answer made from the model's memory is correct and unfaithful.

## 2. What does the abstain threshold trade off? Show your numbers. Which score did you use, and why did the others fail or work?

A higher threshold refuses more, so it catches more unanswerable questions (fewer wrong answers) and also refuses more answerable ones (more wrong refusals). A lower one does the opposite.

I compared the top score of four retrievers on the 40 answerable and 10 unanswerable questions, without a model. AUC (the chance an answerable question outscores an unanswerable one): hybrid 0.907, BM25 0.795, reranker 0.782, dense cosine 0.728. The best F1 over thresholds (fitted on the same 50 questions, so optimistic): hybrid 0.696 at 1.558, BM25 0.563, reranker 0.571, dense 0.529. The dense cosine fails because the answerable and unanswerable scores overlap almost completely (the lowest answerable score, 0.678, is below the unanswerable median, 0.748). The strictest hybrid threshold that never refuses an answerable question (1.048) catches only 5 of the 10. I did not test why hybrid works best; my guess is that it is high when BM25 and dense agree on the top result.

In the end I used no threshold: the model's own `CANNOT_ANSWER` refused 10 of 10 unanswerable and 2 of 40 answerable questions (F1 0.909), better than any score. A threshold on top of that was not measured.

## 3. What did the injection test reveal? What held, what leaked, and what would you still not trust?

All 9 cases held (the canary `PWNED-7431` never appeared): direct override, fake closing tag, role spoof, citation hijack, hidden in code, markdown comment, a malicious question, and my two (`polite_docs`, a documentation-style note asking for a "verification code", and `split_canary`, which asks for the canary in three steps). Nothing leaked.

I would still not trust it as a safety claim: nine attacks is a tiny sample, the check only looks for the canary (a half-obeyed instruction would pass), it ran against one model, and I did not change the system prompt and re-measure, so I do not know which defence (the rule that chunks are data, or `neutralize` escaping fake tags) did the work. A model that resists these nine can fall to a better-disguised one.

## 4. Why retry with backoff and jitter? Why is a failure in the middle of a stream not retried, and why is the SDK's own retry switched off?

A failure like a rate limit or a 5xx is often gone a moment later, so retrying helps. Waiting longer after each failure (backoff) gives a recovering service room. The random part (jitter) stops many clients that failed together from retrying together.

A stream that fails after some pieces have arrived has already shown the user part of the text. A second attempt starts again from the beginning, so the user would see repeated or different words. The code raises the error instead, and only retries before the first piece.

The SDK's retry is off because two retry layers multiply the attempts (4 of mine times the SDK's own), and the SDK's retries happen out of sight, so my classifier, my cost counter and my logs would not see them.

## 5. Which errors are worth retrying and which never are? What would retrying a 400 or a 401 do?

Worth retrying: 429 (rate limit; wait for `Retry-After` if given), 408 and timeouts, 5xx, and connection failures. Never: 400 and 422 (the request is wrong), 401 and 403 (the key or permission is wrong), 404 (the model or path does not exist), and any error I cannot classify.

Retrying a 400 or 401 sends the same request and gets the same answer, so it only wastes time and calls (and may hammer an authentication endpoint with a bad key).

*Step 1 questions from the brief:* why `except Exception` and not a bare `except`? A bare `except` also catches `KeyboardInterrupt` and `SystemExit`, so Ctrl-C would be swallowed and retried; `except Exception` leaves those alone. Why is an unknown error not retried? Retrying something I cannot name may repeat a request that had a side effect or cost money, and it hides a bug that should be fixed.

## 6. What does the judge's kappa tell you? How much do you trust your correctness number, and what would change it?

Kappa measures agreement beyond what chance would give, given how often each side says yes. For `correct`, `judge_v1` agreed with the hand grading on 13 of 15 (kappa 0.667) and for `faithful` on 11 of 15 (kappa 0.250). The low faithful kappa is partly an artefact: the hand grading found only one unfaithful answer, which pushes kappa down, but the judge's own reasons showed real mistakes (a contradiction, a misread negation, grading against the reference instead of the chunks).

I trust the correctness number as "roughly 0.6 to 0.8": 31 of 40 under `v1` (interval about ±0.13) and 24 of 40 under `v2` on identical answers. A different or larger judge model, grading all answers by hand, or verifying the references would change it. The hand grading was done by the assistant, not me, and the prompt `v2` was written after seeing the disagreements on the same 15 answers.

## 7. How much did the order and the size of the context matter, and was it above the noise of 40 questions?

Order: not measured. Size: measured without a judge, as the share of questions whose context holds a gold chunk. With hybrid at 1,500 tokens, k = 3, 5, 8 give 0.775, 0.850, 0.900. The step from 5 to 8 is 2 questions (0.05), which is at the edge of what 40 questions can show, not clearly above it. Going from 1,500 to 3,000 tokens changed nothing at k <= 5 because only about 1,000 tokens were used. The bigger effect is retrieval itself: 6 of 40 questions had no gold chunk in the context, and no prompt can fix that.

## 8. Why is `count_tokens` only a proxy? What did the provider's `usage` show?

`count_tokens` is a regex that splits words, digits and punctuation. A real model uses its own tokenizer (subwords), so the counts differ, and a different model counts differently again. Against Gemma's reported prompt tokens for 50 prompts the ratio (reported / ours) had a median of 1.04 and a range of 0.99 to 1.15, a mean error of 4.7%, so our budget was close for this model. For a Claude model it was not measured.

## 9. What does one answer cost, and what dominates the cost (the input, the output, the judge)?

On the local model, $0. Counting the tokens as if a paid model had run: the 50 answers used 73,077 input and 2,931 output tokens, so the input is about 25 times the output and dominates. Per question that would be about $0.007 on Opus 5.5, $0.0035 on Sonnet 5.5 and $0.0002 on Haiku 5.5 (estimates; none was run). The judge call, which also sends the chunks, would add about the same again, so the judge is as big a cost as the answering. Smaller or fewer chunks are the main lever.
