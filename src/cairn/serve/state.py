"""The service's life cycle: starting, loading, warming, ready (or failed)."""

import threading
import time

STARTING = "starting"
LOADING = "loading"
WARMING = "warming"
READY = "ready"
FAILED = "failed"


class Service:
    """Loads the engine (indexes, embedder, LLM client) in a background thread.

    `loader()` returns the engine. `warm_up(engine)`, if given, runs once on it so the first real
    request does not pay for lazy loading. Any exception in either one makes the service `failed`
    with the error text kept; the thread never dies silently.
    """

    def __init__(self, loader, warm_up=None):
        self._loader = loader
        self._warm_up = warm_up
        self._lock = threading.Lock()
        self._thread = None
        self._finished = threading.Event()
        self.status = STARTING
        self.error = None
        self.engine = None
        self.load_seconds = None
        self.warm_seconds = None

    def start(self):
        """Begins loading. Calling it again does nothing (the loader runs once)."""
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(target=self._run, name="cairn-loader", daemon=True)
            self._thread.start()

    def _run(self):
        try:
            self.status = LOADING
            began = time.monotonic()
            engine = self._loader()
            self.load_seconds = time.monotonic() - began
            if self._warm_up is not None:
                self.status = WARMING
                began = time.monotonic()
                self._warm_up(engine)
                self.warm_seconds = time.monotonic() - began
            self.engine = engine
            self.status = READY
        except Exception as exc:  # report it through /health instead of killing the thread
            self.error = f"{type(exc).__name__}: {exc}"
            self.status = FAILED
        finally:
            self._finished.set()

    @property
    def ready(self):
        return self.status == READY

    def wait(self, timeout=None):
        """Blocks until loading finished (ready or failed). False if the timeout passed first."""
        return self._finished.wait(timeout)

    def describe(self):
        """What /health shows."""
        return {
            "status": self.status,
            "ready": self.ready,
            "error": self.error,
            "load_seconds": self.load_seconds,
            "warm_seconds": self.warm_seconds,
        }
