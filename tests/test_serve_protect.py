import threading
import time

import pytest

from cairn.generation.errors import LLMError, ServerError
from cairn.generation.fake import FakeLLM
from cairn.serve.protect import (
    CLOSED,
    HALF_OPEN,
    OPEN,
    CircuitBreaker,
    CircuitOpen,
    ConcurrencyLimit,
    GuardedLLM,
    Overloaded,
    RateLimiter,
    TokenBucket,
)


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


# --- token bucket --------------------------------------------------------------------------


def test_a_bucket_allows_a_burst_then_holds_the_client_to_the_rate():
    clock = FakeClock()
    bucket = TokenBucket(rate=2, burst=3, clock=clock)
    assert [bucket.take()[0] for _ in range(3)] == [True, True, True]
    allowed, retry_after = bucket.take()
    assert not allowed and retry_after == pytest.approx(0.5)  # one token at 2 per second
    clock.now = 0.5
    assert bucket.take()[0]
    assert not bucket.take()[0]


def test_a_bucket_never_holds_more_than_its_burst():
    clock = FakeClock()
    bucket = TokenBucket(rate=1, burst=2, clock=clock)
    clock.now = 1000  # a long idle time must not store up 1000 tokens
    assert [bucket.take()[0] for _ in range(3)] == [True, True, False]


def test_the_retry_after_is_the_time_until_a_token_exists():
    clock = FakeClock()
    bucket = TokenBucket(rate=1, burst=1, clock=clock)
    bucket.take()
    clock.now = 0.25
    allowed, retry_after = bucket.take()
    assert not allowed and retry_after == pytest.approx(0.75)
    clock.now += retry_after
    assert bucket.take()[0]


def test_bucket_arguments_are_checked():
    with pytest.raises(ValueError):
        TokenBucket(rate=0, burst=1)
    with pytest.raises(ValueError):
        TokenBucket(rate=1, burst=0)


def test_each_client_has_its_own_bucket():
    clock = FakeClock()
    limiter = RateLimiter(rate=1, burst=1, clock=clock)
    assert limiter.check("alice")[0]
    assert not limiter.check("alice")[0]
    assert limiter.check("bob")[0]  # bob is not punished for alice
    assert limiter.rejected == 1


def test_the_limiter_forgets_the_least_recent_clients_to_bound_memory():
    clock = FakeClock()
    limiter = RateLimiter(rate=1, burst=1, clock=clock, max_keys=2)
    limiter.check("a")
    limiter.check("b")
    limiter.check("c")  # a is dropped
    assert limiter.check("a")[0]  # so a starts with a full bucket again
    assert len(limiter._buckets) == 2


def test_concurrent_requests_never_get_more_than_the_burst():
    limiter = RateLimiter(rate=0.001, burst=5)
    allowed = []

    def hit():
        allowed.append(limiter.check("k")[0])

    threads = [threading.Thread(target=hit) for _ in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(allowed) == 5


# --- concurrency limit ---------------------------------------------------------------------


def test_requests_past_the_limit_wait_and_then_are_refused():
    limit = ConcurrencyLimit(max_in_flight=1, max_wait_s=0.05)
    with limit:
        assert limit.in_flight == 1
        started = time.monotonic()
        with pytest.raises(Overloaded):
            with limit:
                pass
        assert 0.04 <= time.monotonic() - started < 1
    assert limit.in_flight == 0 and limit.rejected == 1
    with limit:  # the slot came back
        pass


def test_a_waiting_request_gets_the_slot_when_it_frees_up():
    limit = ConcurrencyLimit(max_in_flight=1, max_wait_s=5)
    got = []
    gate = threading.Event()

    def first():
        with limit:
            gate.wait(5)

    def second():
        with limit:
            got.append(True)

    t1 = threading.Thread(target=first)
    t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=second)
    t2.start()
    time.sleep(0.05)
    assert limit.waiting == 1 and not got
    gate.set()
    t1.join(5)
    t2.join(5)
    assert got == [True] and limit.rejected == 0


def test_the_slot_is_released_when_the_work_raises():
    limit = ConcurrencyLimit(max_in_flight=1)
    with pytest.raises(RuntimeError):
        with limit:
            raise RuntimeError("boom")
    assert limit.in_flight == 0
    with limit:
        pass


# --- circuit breaker -----------------------------------------------------------------------


def test_the_circuit_opens_after_consecutive_failures_and_closes_after_a_good_probe():
    clock = FakeClock()
    breaker = CircuitBreaker(failures=3, cooldown_s=30, clock=clock)
    assert breaker.state == CLOSED
    for _ in range(2):
        breaker.failure()
    assert breaker.state == CLOSED and breaker.allow()
    breaker.failure()
    assert breaker.state == OPEN and not breaker.allow()
    clock.now = 29
    assert not breaker.allow()
    clock.now = 31
    assert breaker.state == HALF_OPEN
    assert breaker.allow()  # the probe
    assert not breaker.allow()  # only one probe at a time
    breaker.success()
    assert breaker.state == CLOSED and breaker.allow()


def test_a_failed_probe_reopens_the_circuit_for_another_cooldown():
    clock = FakeClock()
    breaker = CircuitBreaker(failures=1, cooldown_s=10, clock=clock)
    breaker.failure()
    clock.now = 11
    assert breaker.allow()
    breaker.failure()
    assert breaker.state == OPEN and not breaker.allow()
    clock.now = 22
    assert breaker.allow()
    assert breaker.opened_total == 2  # opened, then opened again after the failed probe


def test_a_success_resets_the_failure_count():
    breaker = CircuitBreaker(failures=3)
    breaker.failure()
    breaker.failure()
    breaker.success()
    breaker.failure()
    breaker.failure()
    assert breaker.state == CLOSED


def test_the_guard_stops_calling_a_dead_model_and_recovers_when_it_is_back():
    clock = FakeClock()
    breaker = CircuitBreaker(failures=2, cooldown_s=30, clock=clock)
    inner = FakeLLM([ServerError("down"), ServerError("down"), "back again"])
    llm = GuardedLLM(inner, breaker)
    for _ in range(2):
        with pytest.raises(ServerError):
            llm.complete("s", [{"role": "user", "content": "q"}])
    calls = len(inner.calls)
    with pytest.raises(CircuitOpen) as info:
        llm.complete("s", [{"role": "user", "content": "q"}])
    assert isinstance(info.value, LLMError) and len(inner.calls) == calls  # not even called
    clock.now = 31
    assert llm.complete("s", [{"role": "user", "content": "q"}]).text == "back again"
    assert breaker.state == CLOSED


def test_the_guard_counts_a_streaming_failure_and_a_streaming_success():
    breaker = CircuitBreaker(failures=1)
    llm = GuardedLLM(FakeLLM([ServerError("down"), "fine"]), breaker)
    with pytest.raises(ServerError):
        list(llm.stream("s", [{"role": "user", "content": "q"}]))
    assert breaker.state == OPEN
    breaker2 = CircuitBreaker(failures=1)
    ok = GuardedLLM(FakeLLM(["fine words"]), breaker2)
    assert "".join(ok.stream("s", [{"role": "user", "content": "q"}])).strip() == "fine words"
    assert breaker2.state == CLOSED


def test_the_guard_does_not_count_other_errors_as_the_models_fault():
    breaker = CircuitBreaker(failures=1)

    class Buggy:
        model = "m"
        last_usage = None

        def complete(self, system, messages, **kwargs):
            raise KeyError("a bug in our own code")

    with pytest.raises(KeyError):
        GuardedLLM(Buggy(), breaker).complete("s", [])
    assert breaker.state == CLOSED
