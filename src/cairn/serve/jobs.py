"""Background jobs for `POST /ingest`: it answers at once, the work goes on in a thread."""

import threading
import time
import uuid
from dataclasses import dataclass, field

QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"


@dataclass
class Job:
    id: str
    status: str = QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None

    def describe(self):
        return {
            "id": self.id,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "result": self.result,
            "error": self.error,
        }


class JobRunner:
    """Runs `work()` in a thread, one job at a time. Asking for a job while one is queued or
    running returns that one instead of starting a second (two re-indexes in a row would only
    repeat the work). Only the last `keep` finished jobs are remembered."""

    def __init__(self, work, keep=20):
        self._work = work
        self._keep = keep
        self._lock = threading.Lock()
        self._jobs = {}  # insertion order = creation order

    def submit(self):
        """Returns (job, created)."""
        with self._lock:
            for job in self._jobs.values():
                if job.status in (QUEUED, RUNNING):
                    return job, False
            job = Job(id=uuid.uuid4().hex[:12])
            self._jobs[job.id] = job
            while len(self._jobs) > self._keep:
                del self._jobs[next(iter(self._jobs))]
        threading.Thread(
            target=self._run, args=(job,), name=f"cairn-job-{job.id}", daemon=True
        ).start()
        return job, True

    def _run(self, job):
        job.started_at = time.time()
        job.status = RUNNING
        try:
            job.result = self._work()
            job.status = DONE
        except Exception as exc:  # the job's error is its result; the thread must not die quietly
            job.error = f"{type(exc).__name__}: {exc}"
            job.status = FAILED
        finally:
            job.finished_at = time.time()

    def get(self, job_id):
        with self._lock:
            return self._jobs.get(job_id)

    def wait(self, job_id, timeout=10):
        """For tests and scripts: polls until the job finished. Returns the job."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.get(job_id)
            if job is not None and job.status in (DONE, FAILED):
                return job
            time.sleep(0.01)
        raise TimeoutError(f"job {job_id} did not finish in {timeout}s")
