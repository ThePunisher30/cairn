import re

_TOKEN = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+|[^\sA-Za-z\d]")


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))
