"""Phase 1, step 1: your Lesson 2 Timer, promoted into the package."""

import time

import pytest

from cairn.timing import Timer


def test_timer_measures_elapsed_seconds():
    with Timer() as t:
        time.sleep(0.05)
    assert 0.04 <= t.seconds < 1.0


def test_timer_sets_seconds_and_does_not_swallow_errors():
    t = Timer()
    with pytest.raises(RuntimeError):
        with t:
            raise RuntimeError("boom")
    assert t.seconds >= 0
