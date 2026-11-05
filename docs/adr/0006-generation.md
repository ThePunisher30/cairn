# ADR 0006: Local model, the model's own refusal, retry only what can succeed

Status: accepted, 2026-11-05. Evidence: `docs/results.md`, section "Phase 7"; `eval/failures/phase7.md`.
Every number below comes from one model (`gemma4:e4b` through Ollama), one run of 50 questions and a judge that is the same model, so the claims are narrow. Section "What I would not trust" says where.

## Context

Phase 7 turns the retrieved chunks into a cited answer that refuses when the chunks do not hold the answer, and measures it. I wrote `retry.py`, `packer.py`, `prompts.py`, `citations.py`, `abstain.py`, `pipeline.py` and `evaluation/generation.py` myself (`src/cairn/generation/`, `src/cairn/evaluation/`); the HTTP clients (`clients.py`) take their models from the standard `anthropic` SDK and from `httpx` for Ollama, each only inside its own client. No `langchain` or similar. The project had a hard constraint of spending no money, which decided the model.

## Decision

- **The model:** `gemma4:e4b`, run locally with Ollama, **thinking off** (`think=false`). A thinking model spent a 40-token limit on hidden reasoning and returned an empty answer (`done_reason: "length"`); with thinking off the answer comes back. The Anthropic client exists and is tested but was never run against the real API.
- **The Ollama timeout is 900 s** (60 s for the Anthropic API). Reading a 3,357-token prompt took 135 s on this CPU, and a timeout would only make the retry layer start the same slow call again.
- **Context:** the hybrid min-max retriever of ADR 0005, `k = 5`, a 1,500-token budget, order `best_first`, overflow `truncate`. At k = 5 the budget is not reached (about 1,000 tokens used), so 1,500 against 3,000 changes nothing. The order was chosen by default and not measured.
- **Prompts:** `answer_v1.txt` and `judge_v1.txt`, files with a version in the name. The judge is the same model as the answerer.
- **Abstention:** the model's own. The system prompt tells it to reply exactly `CANNOT_ANSWER` when the chunks do not hold the answer, and `says_cannot_answer` detects it. **No retrieval-score threshold is used by default.** If one is wanted, the best score is the hybrid top score (AUC 0.907; best in-sample F1 0.696 at 1.558), but a threshold was never run end to end.
- **Citations:** chunks are numbered in the prompt, `[n]` markers are parsed, markers that point to no chunk are stripped, and the sources are rendered as links to the pinned tag.
- **Retry policy** (`RetryPolicy`): 4 attempts, a delay cap that doubles from 0.5 s up to 30 s, full jitter (a random delay between 0 and the cap), and a `Retry-After` header can lengthen a wait but never beyond 30 s. **Retried:** rate limits (429), timeouts (408 and timeout exceptions), server errors (5xx) and connection failures. **Never retried:** 400 and 422 (bad request), 401 and 403 (authentication), 404, and any error the classifier does not recognise. A **stream is retried only before its first piece arrives**. The SDK's own retries are switched off (`max_retries=0`).
- **Evaluation:** `judge_v1` is the headline judge because it was fixed before any result; `judge_v2` is reported as a sensitivity check.

## Why

- **Local model:** zero cost, no key, repeatable. The cost is a 64 s median latency and a weak judge. The same 50 answering calls would cost about $0.35 on Opus 5.5, $0.18 on Sonnet 5.5 and $0.01 on Haiku 5.5 (an estimate from token counts, none was run).
- **The model's refusal over a threshold:** it refused all 10 unanswerable questions and 2 of 40 answerable ones (F1 0.909). The best retrieval score reached 0.696 in sample, and the strictest threshold that never refuses an answerable question catches 5 of 10 unanswerable ones. The 10 unanswerable questions are about topics completely absent from the corpus, which flatters the model.
- **Retry only what can succeed:** a rate limit, a timeout or a 5xx may be gone a moment later. A 400, 401, 403 or 404 gets the same answer however often it is sent, and repeating a rejected key or request only wastes calls and time. An error the classifier cannot name is treated as non-retryable because retrying something unknown can repeat a harmful or paid action.
- **Backoff with jitter:** waiting longer after each failure gives a recovering service room, and the random part stops many clients from retrying at the same moment.
- **No retry in the middle of a stream:** the caller has already received part of the text, and a second attempt would restart it, so the user would see repeated or different words. The failure is raised instead.
- **SDK retries off:** two layers of retries multiply the attempts, and the SDK's retries hide the errors from the classifier, the cost counter and the logs.

## Alternatives measured

- **Dense or hybrid as the retriever:** a tie on the context they supply (hit rate 0.825 against 0.850 at k = 5, one question). A generation-level run could not have told them apart, so ADR 0005's choice stands. The reranker is two questions ahead (0.900) and is worth re-checking in Phase 11.
- **Judge prompt `v1` against `v2`:** correctness 0.775 against 0.600 and faithfulness 0.763 against 0.947 on identical answers. Neither fixed the judge (kappa on faithfulness 0.250 and -0.098 against 15 hand-graded answers; on correctness 0.667 and 0.727).
- **Score thresholds against the model's refusal:** see above.

## Trade-offs

- **Wrong refusals against wrong answers:** every threshold or prompt that refuses more catches more unanswerable questions and refuses more answerable ones. Here the model refused 2 of 40 answerable questions (q026, where the gold passage was not retrieved, so the refusal was right for the context, and q055, where it was retrieved) and answered no unanswerable question.
- **Quality against cost and time:** k = 8 adds about 5 points of hit rate (hybrid 0.900) for about 330 more input tokens, and input is about 25 times the output in this run. On a paid model that is where the money goes; on this CPU it is also where the time goes.

## What I would not trust

- **Faithfulness numbers.** The judge disagreed with a hand grading on 4 of 15 answers, always in the strict direction, and changing its prompt moved the score by 18 points.
- **Correctness to better than about ±0.13** (40 questions), and to less than the 0.60 to 0.78 spread between two judge prompts.
- **Anything about the order of the context** (not measured), a threshold on top of the model's refusal (not measured), or any comparison between this model and a paid one (never run).
- **The injection result as proof of safety.** 9 of 9 hand-written attacks held on this model; that is all it says.
- **The references.** Drafted with an assistant; 15 were checked against answers and none was wrong, which is weak evidence about the rest.

## Consequences

- Phase 8's agent loop can use the same `LLMClient`, `RetryingClient` and `CostTracker`. A model that follows tool-use formats may need a larger one than `gemma4:e4b`; that has not been tried.
- A better judge (a different or larger model, or a mix of two) is the first thing to change before trusting a faithfulness number. The hand-graded sample and `scripts/rejudge.py` make a re-measurement cheap.
- A second test run belongs to Phase 11; nothing in this phase used the test split.
