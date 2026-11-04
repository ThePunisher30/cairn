import re

from cairn.evaluation.metrics import overlaps

DEFAULT_REPO = "https://github.com/fastapi/fastapi"
DEFAULT_TAG = "0.143.0"

# One pass over the text. A code span (a ``` fence or `inline code`) is matched first so that
# whatever sits inside it is skipped. A marker is [3], [1, 2] or [1,2]; it does not count right
# after a letter, digit, underscore or ")" (x[1], [a](b)[1]). The optional space in front of it
# is captured so that strip_invalid can remove the marker together with that one space.

TOKEN = re.compile(
    r"(?P<code>```.*?```|`[^`\n]*`)"
    r"|(?P<space> ?)(?<![\w)])\[\s*(?P<numbers>\d+(?:\s*,\s*\d+)*)\s*\]",
    re.DOTALL,
)


def citation_precision_recall(cited_hits, gold):
    gold = list(gold)
    if not gold:
        raise ValueError("gold must hold at least one span")
    cited_hits = list(cited_hits)
    if not cited_hits:
        return 0.0, 0.0
    good = sum(any(overlaps(hit, span) for span in gold) for hit in cited_hits)
    covered = sum(any(overlaps(hit, span) for hit in cited_hits) for span in gold)
    return good / len(cited_hits), covered / len(gold)


def extract_citations(text):
    found = []
    for match in TOKEN.finditer(text):
        if match.group("code") is not None:
            continue
        for number in re.findall(r"\d+", match.group("numbers")):
            if int(number) not in found:
                found.append(int(number))
    return found


def render_sources(context, cited, *, repo=DEFAULT_REPO, tag=DEFAULT_TAG):
    by_number = {chunk.number: chunk for chunk in context.chunks}
    sources = []
    for number in cited:
        chunk = by_number.get(number)
        if chunk is None:
            continue
        hit = chunk.hit
        if hit.start_line == hit.end_line:
            where = f"{hit.start_line}"
        else:
            where = f"{hit.start_line}-{hit.end_line}"
        url = source_url(hit.path, hit.start_line, hit.end_line, repo=repo, tag=tag)
        sources.append(f"[{number}] [{hit.path}:{where}]({url})")
    return sources


def source_url(path, start_line, end_line, *, repo=DEFAULT_REPO, tag=DEFAULT_TAG):
    lines = f"L{start_line}" if start_line == end_line else f"L{start_line}-L{end_line}"
    return f"{repo}/blob/{tag}/{path}#{lines}"


def strip_invalid(text, n_chunks):
    def fix(match):
        if match.group("code") is not None:
            return match.group(0)
        numbers = [int(n) for n in re.findall(r"\d+", match.group("numbers"))]
        keep = [n for n in numbers if 1 <= n <= n_chunks]
        if len(keep) == len(numbers):
            return match.group(0)
        if not keep:
            return ""
        return match.group("space") + "[" + ", ".join(str(n) for n in keep) + "]"

    return TOKEN.sub(fix, text)


def validate_citations(cited, n_chunks):
    valid = [n for n in cited if 1 <= n <= n_chunks]
    invalid = [n for n in cited if not 1 <= n <= n_chunks]
    return valid, invalid
