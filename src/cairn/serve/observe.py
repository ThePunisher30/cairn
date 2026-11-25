"""Seeing what the service does (Phase 9): request ids, JSON logs, per-stage timing, metrics."""

import json
import logging
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

request_id_var = ContextVar("request_id", default="-")
_VALID_ID = re.compile(r"[A-Za-z0-9._-]{1,64}")


def new_request_id(incoming=None):
    """An id the caller sent, if it is harmless (it ends up in logs and headers), else a new one."""
    if incoming and _VALID_ID.fullmatch(incoming):
        return incoming
    return uuid.uuid4().hex[:16]


class JsonFormatter(logging.Formatter):
    """One JSON object per line. Extra fields passed as `extra={"fields": {...}}` are merged in."""

    def format(self, record):
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level="INFO", stream=None):
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("cairn")
    root.handlers[:] = [handler]
    root.setLevel(level)
    root.propagate = False
    return handler


# --- metrics -------------------------------------------------------------------------------

SECONDS_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120)


def _labels(labels):
    if not labels:
        return ""
    body = ",".join(
        f'{name}="{str(value).replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for name, value in sorted(labels)
    )
    return "{" + body + "}"


class Metrics:
    """Counters, histograms and gauges, rendered in the Prometheus text format."""

    def __init__(self):
        self._lock = threading.Lock()
        self._counters = {}  # name -> {labels: value}
        self._histograms = {}  # name -> {labels: [bucket counts..., sum, count]}
        self._gauges = {}  # name -> function returning a number
        self._help = {}

    def describe(self, name, text):
        self._help[name] = text

    def inc(self, name, value=1, **labels):
        key = tuple(sorted(labels.items()))
        with self._lock:
            series = self._counters.setdefault(name, {})
            series[key] = series.get(key, 0) + value

    def observe(self, name, value, **labels):
        key = tuple(sorted(labels.items()))
        with self._lock:
            series = self._histograms.setdefault(name, {})
            row = series.setdefault(key, [0] * len(SECONDS_BUCKETS) + [0.0, 0])
            for i, edge in enumerate(SECONDS_BUCKETS):
                if value <= edge:
                    row[i] += 1
            row[-2] += value
            row[-1] += 1

    def gauge(self, name, fn):
        self._gauges[name] = fn

    def counter_value(self, name, **labels):
        with self._lock:
            return self._counters.get(name, {}).get(tuple(sorted(labels.items())), 0)

    def histogram_count(self, name, **labels):
        with self._lock:
            row = self._histograms.get(name, {}).get(tuple(sorted(labels.items())))
            return row[-1] if row else 0

    def render(self):
        lines = []

        def header(name, kind):
            if name in self._help:
                lines.append(f"# HELP {name} {self._help[name]}")
            lines.append(f"# TYPE {name} {kind}")

        with self._lock:
            for name, series in sorted(self._counters.items()):
                header(name, "counter")
                for key, value in sorted(series.items()):
                    lines.append(f"{name}{_labels(key)} {value}")
            for name, series in sorted(self._histograms.items()):
                header(name, "histogram")
                for key, row in sorted(series.items()):
                    for edge, count in zip(SECONDS_BUCKETS, row, strict=False):
                        lines.append(f"{name}_bucket{_labels((*key, ('le', edge)))} {count}")
                    lines.append(f"{name}_bucket{_labels((*key, ('le', '+Inf')))} {row[-1]}")
                    lines.append(f"{name}_sum{_labels(key)} {row[-2]:.6f}")
                    lines.append(f"{name}_count{_labels(key)} {row[-1]}")
        for name, fn in sorted(self._gauges.items()):
            header(name, "gauge")
            try:
                lines.append(f"{name} {float(fn())}")
            except Exception:  # a broken gauge must not take /metrics down
                lines.append(f"{name} NaN")
        return "\n".join(lines) + "\n"


# --- per-request stage timing --------------------------------------------------------------


class RequestStats:
    """What one request spent time on. Created by the endpoint and passed down explicitly: a
    streaming response runs its steps on different threads, so a thread-local would lose it."""

    def __init__(self):
        self.stages = {}
        self.cache = None  # "exact", "semantic" or None
        self.fallback = False

    @contextmanager
    def timed(self, stage):
        began = time.perf_counter()
        try:
            yield
        finally:
            self.stages[stage] = self.stages.get(stage, 0.0) + time.perf_counter() - began


class TimedRetriever:
    def __init__(self, retriever, stats):
        self._retriever = retriever
        self._stats = stats

    def search(self, query, k):
        with self._stats.timed("retrieve"):
            return self._retriever.search(query, k)


class TimedLLM:
    """Times the model: `llm_first_token` (streaming only) and `llm_total`."""

    def __init__(self, llm, stats):
        self._llm = llm
        self._stats = stats

    @property
    def model(self):
        return getattr(self._llm, "model", "")

    @property
    def last_usage(self):
        return self._llm.last_usage

    def complete(self, system, messages, **kwargs):
        with self._stats.timed("llm_total"):
            return self._llm.complete(system, messages, **kwargs)

    def stream(self, system, messages, **kwargs):
        began = time.perf_counter()
        first = True
        try:
            for piece in self._llm.stream(system, messages, **kwargs):
                if first:
                    self._stats.stages["llm_first_token"] = time.perf_counter() - began
                    first = False
                yield piece
        finally:
            self._stats.stages["llm_total"] = time.perf_counter() - began


# --- the ASGI middleware ---------------------------------------------------------------------

access_log = logging.getLogger("cairn.serve.access")


class ObservabilityMiddleware:
    """Gives every request an id (header `X-Request-ID`, in and out), and records one access-log
    line and the request metrics when it ends. Pure ASGI, so streamed responses pass through."""

    def __init__(self, app, metrics):
        self.app = app
        self.metrics = metrics

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode().lower(): v.decode() for k, v in scope["headers"]}
        request_id = new_request_id(headers.get("x-request-id"))
        token = request_id_var.set(request_id)
        began = time.perf_counter()
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
                message.setdefault("headers", []).append((b"x-request-id", request_id.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            seconds = time.perf_counter() - began
            route = scope.get("route")
            path = getattr(route, "path", None)  # bounded label values only
            if path is None:
                path = "/ui" if scope["path"].startswith("/ui") else "unmatched"
            self.metrics.inc("cairn_requests_total", route=path, status=status["code"])
            self.metrics.observe("cairn_request_seconds", seconds, route=path)
            access_log.info(
                "%s %s -> %s",
                scope["method"],
                scope["path"],
                status["code"],
                extra={
                    "fields": {
                        "route": path,
                        "status": status["code"],
                        "seconds": round(seconds, 4),
                    }
                },
            )
            request_id_var.reset(token)
