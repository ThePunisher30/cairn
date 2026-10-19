from collections import defaultdict

from cairn.types import CATEGORIES


def _by_path(chunks):
    index = defaultdict(list)
    for chunk in chunks:
        index[chunk.path].append(chunk)
    return index


def _is_intact(span, chunks_by_path):
    return any(
        c.start_line <= span.start_line and c.end_line >= span.end_line
        for c in chunks_by_path.get(span.path, [])
    )


def _summary(questions, chunks_by_path):
    spans = 0
    spans_intact = 0
    questions_intact = 0
    for question in questions:
        results = [_is_intact(span, chunks_by_path) for span in question.gold]
        spans += len(results)
        spans_intact += sum(results)
        questions_intact += all(results)
    return {
        "questions": len(questions),
        "spans": spans,
        "spans_intact": spans_intact,
        "span_rate": spans_intact / spans,
        "question_rate": questions_intact / len(questions),
    }


def gold_integrity(chunks, questions):
    if not questions:
        raise ValueError("gold_integrity needs at least one question")
    chunks_by_path = _by_path(chunks)
    by_category = {}
    for category in CATEGORIES:
        members = [q for q in questions if q.category == category]
        if members:
            by_category[category] = _summary(members, chunks_by_path)
    return {"overall": _summary(questions, chunks_by_path), "by_category": by_category}


def broken_spans(chunks, questions):
    chunks_by_path = _by_path(chunks)
    return [
        (question, span)
        for question in questions
        for span in question.gold
        if not _is_intact(span, chunks_by_path)
    ]
