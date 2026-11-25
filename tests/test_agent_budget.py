"""Phase 8, step 3: budgets (`agent/budget.py`).

A loop that can run forever is a bill and a hang waiting to happen. Three independent limits, and
the loop checks all of them before every model call.

    Budget(max_steps=8, max_tokens=30_000, timeout_s=600.0, max_calls_per_step=4)
        a frozen dataclass. ValueError when max_steps < 1, max_tokens < 1, timeout_s <= 0 or
        max_calls_per_step < 1.

    BudgetTracker(budget, clock=time.monotonic)
        the clock starts when the tracker is built.
        .steps   model calls recorded so far          .tokens  input + output tokens recorded
        .elapsed()  seconds since the start
        .record(usage)   one model call finished: steps += 1, tokens += input + output
        .exceeded() -> str | None
            "max_steps"   when steps >= budget.max_steps
            "max_tokens"  when tokens >= budget.max_tokens
            "timeout"     when elapsed() >= budget.timeout_s
            None          otherwise
            When several are true the first of those three, in that order, is returned.
        .remaining_tokens() -> int    max_tokens - tokens, never below 0
"""

import pytest

from cairn.agent.budget import Budget, BudgetTracker
from cairn.generation.types import Usage


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def tracker(**kw):
    clock = Clock()
    return BudgetTracker(Budget(**kw), clock=clock), clock


def test_the_defaults():
    b = Budget()
    assert (b.max_steps, b.max_tokens, b.timeout_s, b.max_calls_per_step) == (8, 30_000, 600.0, 4)


@pytest.mark.parametrize(
    "kw",
    [
        {"max_steps": 0},
        {"max_tokens": 0},
        {"timeout_s": 0},
        {"timeout_s": -1.0},
        {"max_calls_per_step": 0},
    ],
)
def test_a_bad_budget_is_refused(kw):
    with pytest.raises(ValueError):
        Budget(**kw)


def test_a_new_tracker_is_within_budget():
    t, _ = tracker()
    assert (t.steps, t.tokens, t.exceeded()) == (0, 0, None)


def test_steps_run_out_at_the_limit():
    t, _ = tracker(max_steps=2)
    t.record(Usage(1, 1))
    assert t.exceeded() is None
    t.record(Usage(1, 1))
    assert t.steps == 2 and t.exceeded() == "max_steps"


def test_tokens_add_input_and_output_over_all_steps():
    t, _ = tracker(max_tokens=100)
    t.record(Usage(40, 10))
    assert t.tokens == 50 and t.exceeded() is None and t.remaining_tokens() == 50
    t.record(Usage(30, 20))
    assert t.tokens == 100 and t.exceeded() == "max_tokens" and t.remaining_tokens() == 0


def test_a_call_may_overshoot_and_remaining_tokens_stays_at_zero():
    t, _ = tracker(max_tokens=100)
    t.record(Usage(500, 0))
    assert t.exceeded() == "max_tokens" and t.remaining_tokens() == 0


def test_the_timeout_uses_the_clock():
    t, clock = tracker(timeout_s=30.0)
    clock.now += 29.9
    assert t.exceeded() is None and t.elapsed() == pytest.approx(29.9)
    clock.now += 0.1
    assert t.exceeded() == "timeout"


def test_when_several_limits_are_hit_the_order_is_steps_tokens_timeout():
    t, clock = tracker(max_steps=1, max_tokens=10, timeout_s=5.0)
    t.record(Usage(50, 50))
    clock.now += 60
    assert t.exceeded() == "max_steps"
    t2, clock2 = tracker(max_steps=9, max_tokens=10, timeout_s=5.0)
    t2.record(Usage(50, 50))
    clock2.now += 60
    assert t2.exceeded() == "max_tokens"
