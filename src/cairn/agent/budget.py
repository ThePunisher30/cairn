"""Budgets for the agent loop: three independent limits, checked before every model call."""

import time
from dataclasses import dataclass

from cairn.generation.types import Usage


@dataclass(frozen=True)
class Budget:
    """What one agent run may spend.

    `max_tokens` counts input + output tokens summed over all model calls, as the provider
    reports them. A call can overshoot it (we only know a call's cost after it is made), so the
    loop stops at the next check instead of cutting a call short.
    """

    max_steps: int = 8
    max_tokens: int = 30_000
    timeout_s: float = 600.0
    max_calls_per_step: int = 4

    def __post_init__(self):
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, it is currently {self.max_steps}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, it is currently {self.max_tokens}")
        if self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be > 0, it is currently {self.timeout_s}")
        if self.max_calls_per_step < 1:
            raise ValueError(
                f"max_calls_per_step must be >= 1, it is currently {self.max_calls_per_step}"
            )


class BudgetTracker:
    """Counts what a run has used so far. The clock starts when the tracker is built."""

    def __init__(self, budget, clock=time.monotonic):
        self.budget = budget
        self._clock = clock
        self._start = clock()
        self.steps = 0
        self.tokens = 0

    def elapsed(self):
        return self._clock() - self._start

    def record(self, usage: Usage):
        """One model call finished."""
        self.steps += 1
        self.tokens += usage.input_tokens + usage.output_tokens

    def exceeded(self):
        """The name of the first limit that is used up (steps, tokens, timeout), else None."""
        if self.steps >= self.budget.max_steps:
            return "max_steps"
        if self.tokens >= self.budget.max_tokens:
            return "max_tokens"
        if self.elapsed() >= self.budget.timeout_s:
            return "timeout"
        return None

    def remaining_tokens(self):
        return max(0, self.budget.max_tokens - self.tokens)
