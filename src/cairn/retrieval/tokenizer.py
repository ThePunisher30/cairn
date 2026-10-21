import re
from dataclasses import dataclass

_WORD = re.compile(r"_*[A-Za-z][A-Za-z0-9_]*|\d+")
_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
ENGLISH_STOPWORDS = frozenset(
    "a an and are as at be by do does for how i in is it of on or that the this to "
    "was what when where which who why with you your".split()
)


@dataclass(frozen=True)
class Tokenizer:
    split_identifiers: bool = True
    stopwords: frozenset = frozenset()

    def tokenize(self, text: str) -> list[str]:
        tokens = []
        for word in _WORD.findall(text):
            whole = word.lower()
            candidates = [whole]
            if self.split_identifiers:
                parts = [part.lower() for part in _PART.findall(word)]
                if parts != [whole]:
                    candidates.extend(parts)
            tokens.extend([t for t in candidates if t not in self.stopwords])
        return tokens
