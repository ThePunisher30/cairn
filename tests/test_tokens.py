"""Phase 2, step 1: a token counter, so chunk sizes are measured in tokens and not characters."""

import pytest

from cairn.tokens import count_tokens


@pytest.mark.parametrize(
    "text,expected",
    [
        ("", 0),
        ("   \n\t ", 0),
        ("Hello, world!", 4),  # Hello , world !
        ("x = 42", 3),  # x = 42
        ("snake_case_name", 5),  # snake _ case _ name
        ("response_model_exclude_unset", 7),
        ("camelCaseName", 3),  # camel Case Name
        ("HTTPException", 2),  # HTTP Exception
        ("HELLO", 1),
        ("`Depends`", 3),  # ` Depends `
        ("a.b(c)", 6),  # a . b ( c )
        ("v2beta", 3),  # v 2 beta
        ("Hello\nworld", 2),
    ],
)
def test_count_tokens(text, expected):
    assert count_tokens(text) == expected


def test_whitespace_never_changes_the_count():
    assert count_tokens("a b c") == count_tokens("a    b\n\n c")


def test_counts_add_up_across_lines():
    lines = ["def f(x):", "    return x + 1", "", "# done"]
    assert count_tokens("\n".join(lines)) == sum(count_tokens(line) for line in lines)
