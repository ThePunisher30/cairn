import re

_TOKEN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+|[^\sA-Za-z\d]")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def truncate_tokens(text: str, max_tokens: int) -> str:
    """The start of `text` that holds at most `max_tokens` tokens (as `count_tokens` counts them).

    The cut falls right after the last kept token, so the result never ends in the middle of a
    word. A text that is already short enough is returned unchanged.
    """
    if max_tokens <= 0:
        return ""
    for number, match in enumerate(_TOKEN.finditer(text), start=1):
        if number == max_tokens:
            end = match.end()
            return text if count_tokens(text[end:]) == 0 else text[:end]
    return text
