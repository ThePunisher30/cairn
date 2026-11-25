"""Keeping the service up under abuse and failure (Phase 9): a rate limiter, a concurrency limit
(backpressure) and a circuit breaker around the language model."""

import threading
import time
from collections import OrderedDict

from cairn.generation.errors import LLMError


class TokenBucket:
    """Holds up to `burst` tokens and refills `rate` per second. A request takes one. A client
    can burst up to `burst` requests at once, then is held to `rate` per second."""

    def __init__(self, rate, burst, clock=time.monotonic):
        if rate <= 0 or burst < 1:
            raise ValueError("rate must be positive and burst at least 1")
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self._tokens = float(burst)
        self._updated = clock()

    def take(self, n=1):
        """Returns (allowed, seconds until `n` tokens would be there)."""
        now = self._clock()
        self._tokens = min(self.burst, self._tokens + (now - self._updated) * self.rate)
        self._updated = now
        if self._tokens >= n:
            self._tokens -= n
            return True, 0.0
        return False, (n - self._tokens) / self.rate


class RateLimiter:
    """One bucket per client key (an API key, or an address). At most `max_keys` buckets are
    kept, the least recently seen are dropped, so memory cannot grow without limit."""

    def __init__(self, rate, burst, clock=time.monotonic, max_keys=10_000):
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self._max_keys = max_keys
        self._lock = threading.Lock()
        self._buckets = OrderedDict()
        self.rejected = 0

    def check(self, key):
        with self._lock:
            bucket = self._buckets.pop(key, None)
            if bucket is None:
                bucket = TokenBucket(self.rate, self.burst, self._clock)
            self._buckets[key] = bucket  # most recently used goes last
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
            allowed, retry_after = bucket.take()
            if not allowed:
                self.rejected += 1
            return allowed, retry_after


class Overloaded(Exception):
    """Too many requests are being worked on and none finished within the wait."""


class ConcurrencyLimit:
    """At most `max_in_flight` requests do the heavy work at once; others wait up to `max_wait_s`
    for a slot, then are refused. This is backpressure: past the machine's capacity, extra work
    only makes every request slower, so it is better to say "busy" quickly."""

    def __init__(self, max_in_flight, max_wait_s=2.0):
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be at least 1")
        self.max_in_flight = max_in_flight
        self.max_wait_s = max_wait_s
        self._slots = threading.BoundedSemaphore(max_in_flight)
        self._lock = threading.Lock()
        self.in_flight = 0
        self.waiting = 0
        self.rejected = 0

    def __enter__(self):
        with self._lock:
            self.waiting += 1
        got = self._slots.acquire(timeout=self.max_wait_s)
        with self._lock:
            self.waiting -= 1
            if got:
                self.in_flight += 1
            else:
                self.rejected += 1
        if not got:
            raise Overloaded(f"more than {self.max_in_flight} requests in progress")
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.in_flight -= 1
        self._slots.release()


class CircuitOpen(LLMError):
    """The breaker is open: the model is not being called at all for now."""


CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


class CircuitBreaker:
    """After `failures` model failures in a row the circuit opens: calls fail at once (no waiting
    for a timeout) for `cooldown_s`. Then one probe call is let through (half open). If it works
    the circuit closes; if it fails the cooldown starts again."""

    def __init__(self, failures=3, cooldown_s=30.0, clock=time.monotonic):
        self.failures = failures
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CLOSED
        self._count = 0
        self._opened_at = 0.0
        self._probing = False
        self.opened_total = 0

    @property
    def state(self):
        with self._lock:
            if self._state == OPEN and self._clock() - self._opened_at >= self.cooldown_s:
                return HALF_OPEN
            return self._state

    def allow(self):
        with self._lock:
            if self._state == CLOSED:
                return True
            if self._clock() - self._opened_at < self.cooldown_s:
                return False
            if self._probing:
                return False  # one probe at a time
            self._state = HALF_OPEN
            self._probing = True
            return True

    def success(self):
        with self._lock:
            self._state, self._count, self._probing = CLOSED, 0, False

    def failure(self):
        with self._lock:
            self._probing = False
            self._count += 1
            if self._state == HALF_OPEN or self._count >= self.failures:
                if self._state != OPEN:
                    self.opened_total += 1
                self._state = OPEN
                self._opened_at = self._clock()


class GuardedLLM:
    """The language model behind a circuit breaker. Same interface as the clients."""

    def __init__(self, llm, breaker):
        self._llm = llm
        self.breaker = breaker

    @property
    def model(self):
        return getattr(self._llm, "model", "")

    @property
    def last_usage(self):
        return self._llm.last_usage

    def fork(self):
        fork = getattr(self._llm, "fork", None)
        return GuardedLLM(fork() if fork else self._llm, self.breaker)

    def _check(self):
        if not self.breaker.allow():
            raise CircuitOpen("the language model is unavailable (circuit open)")

    def complete(self, system, messages, **kwargs):
        self._check()
        try:
            result = self._llm.complete(system, messages, **kwargs)
        except LLMError:
            self.breaker.failure()
            raise
        self.breaker.success()
        return result

    def stream(self, system, messages, **kwargs):
        self._check()
        try:
            yield from self._llm.stream(system, messages, **kwargs)
        except LLMError:
            self.breaker.failure()
            raise
        self.breaker.success()
