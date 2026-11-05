ABSTAIN_MARKER = "CANNOT_ANSWER"
ABSTAIN_MESSAGE = (
    "I could not find the answer to that in the FastAPI documentation, source code and "
    "examples I searched."
)
LEADING_JUNK = " \t\r\n\"'`*_"


def says_cannot_answer(text):
    return text.lstrip(LEADING_JUNK).startswith(ABSTAIN_MARKER)


def top_score(hits):
    return max((hit.score for hit in hits), default=None)


def is_low_confidence(confidence, threshold):
    return confidence is None or confidence < threshold


def abstain_metrics(predicted, should):
    if len(predicted) != len(should):
        raise ValueError(f"{len(predicted)} predictions but {len(should)} expected values")
    pairs = list(zip(predicted, should, strict=True))
    tp = sum(1 for p, s in pairs if p and s)
    fp = sum(1 for p, s in pairs if p and not s)
    fn = sum(1 for p, s in pairs if not p and s)
    tn = sum(1 for p, s in pairs if not p and not s)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def calibrate_threshold(answerable_scores, unanswerable_scores):
    answerable, unanswerable = list(answerable_scores), list(unanswerable_scores)
    if not answerable or not unanswerable:
        raise ValueError("calibration needs scores of answerable and of unanswerable questions")
    scores = answerable + unanswerable
    should = [False] * len(answerable) + [True] * len(unanswerable)
    candidates = sorted(set(scores)) + [max(scores) + 1.0]  # the last one abstains on everything
    best = None
    for threshold in candidates:
        metrics = abstain_metrics([score < threshold for score in scores], should)
        if best is None or metrics["f1"] > best[1]["f1"]:  # a tie keeps the lower threshold
            best = (threshold, metrics)
    return best
