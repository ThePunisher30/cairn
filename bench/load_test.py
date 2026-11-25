"""Load test for the Cairn service (Phase 9).

Start the service first, with a fake model so that only the service itself is measured (a real
local model needs about 40 s per answer on this CPU, which would measure the model instead):

    python scripts/serve.py --llm fake --no-auth --port 8766 --debug-blocking-route
    python bench/load_test.py --endpoint search --users 1 5 20 --duration 20

`--endpoint` is search (retrieval only), ask (retrieval, packing, a fake model) or blocking (the
deliberately wrong route that runs retrieval on the event loop). While each level runs, one more
client calls GET /health every 100 ms: if the event loop stalls, that is where it shows.
Questions are the dev questions (never the test split). The load generator runs on the same
machine as the server, so both compete for the same CPU cores; this is stated in the results.
"""

import argparse
import json
import re
import statistics
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PATHS = {"search": "/search", "ask": "/ask", "blocking": "/debug/search-blocking"}


def dev_questions():
    rows = [json.loads(line) for line in (ROOT / "eval" / "questions.jsonl").open(encoding="utf-8")]
    return [r["question"] for r in rows if r["split"] == "dev"]


def percentile(values, p):
    """Nearest-rank percentile."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, int(-(-p * len(ordered) // 100)) - 1))]


def stage_totals(client):
    """{stage: (sum of seconds, count)} from /metrics, to see where the server spent time."""
    text = client.get("/metrics").text
    out = {}
    for stage in ("retrieve", "llm_total"):
        s = re.search(rf'cairn_stage_seconds_sum{{stage="{stage}"}} ([0-9.eE+-]+)', text)
        c = re.search(rf'cairn_stage_seconds_count{{stage="{stage}"}} (\d+)', text)
        out[stage] = (float(s.group(1)), int(c.group(1))) if s and c else (0.0, 0)
    return out


def run_level(base_url, endpoint, users, duration, questions, key=None):
    headers = {"X-API-Key": key} if key else {}
    path = PATHS[endpoint]
    deadline = time.monotonic() + duration
    lock = threading.Lock()
    latencies, statuses, probes = [], {}, []
    counter = iter(range(10**9))

    def body():
        n = next(counter)
        q = questions[n % len(questions)]
        if endpoint == "ask":
            return {"question": f"{q} (ref {n})"}  # unique, so the answer cache cannot help
        return {"query": q, "k": 5}

    def user():
        with httpx.Client(base_url=base_url, timeout=120, headers=headers) as client:
            while time.monotonic() < deadline:
                payload = body()
                began = time.perf_counter()
                try:
                    status = client.post(path, json=payload).status_code
                except httpx.HTTPError as exc:
                    status = type(exc).__name__
                took = time.perf_counter() - began
                with lock:
                    latencies.append(took)
                    statuses[status] = statuses.get(status, 0) + 1

    def probe():
        with httpx.Client(base_url=base_url, timeout=120) as client:
            while time.monotonic() < deadline:
                began = time.perf_counter()
                try:
                    client.get("/health")
                    probes.append(time.perf_counter() - began)
                except httpx.HTTPError:
                    probes.append(float("inf"))
                time.sleep(0.1)

    with httpx.Client(base_url=base_url, headers=headers, timeout=30) as admin:
        before = stage_totals(admin)
        started = time.monotonic()
        threads = [threading.Thread(target=user) for _ in range(users)]
        threads.append(threading.Thread(target=probe))
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        elapsed = time.monotonic() - started
        after = stage_totals(admin)

    def mean_stage(stage):
        seconds = after[stage][0] - before[stage][0]
        count = after[stage][1] - before[stage][1]
        return seconds / count if count else None

    ok = statuses.get(200, 0)
    return {
        "endpoint": endpoint,
        "users": users,
        "seconds": round(elapsed, 1),
        "requests": len(latencies),
        "ok": ok,
        "statuses": {str(k): v for k, v in sorted(statuses.items(), key=lambda kv: str(kv[0]))},
        "rps": round(ok / elapsed, 1),
        "p50_ms": round(1000 * percentile(latencies, 50), 1),
        "p95_ms": round(1000 * percentile(latencies, 95), 1),
        "p99_ms": round(1000 * percentile(latencies, 99), 1),
        "max_ms": round(1000 * max(latencies), 1) if latencies else None,
        "health_p50_ms": round(1000 * statistics.median(probes), 1) if probes else None,
        "health_max_ms": round(1000 * max(probes), 1) if probes else None,
        "server_retrieve_mean_ms": None
        if mean_stage("retrieve") is None
        else round(1000 * mean_stage("retrieve"), 1),
    }


def print_table(rows):
    cols = [
        ("endpoint", "endpoint"),
        ("users", "users"),
        ("ok/req", None),
        ("rps", "rps"),
        ("p50 ms", "p50_ms"),
        ("p95 ms", "p95_ms"),
        ("p99 ms", "p99_ms"),
        ("max ms", "max_ms"),
        ("health p50", "health_p50_ms"),
        ("health max", "health_max_ms"),
        ("retrieve ms", "server_retrieve_mean_ms"),
    ]
    print(" | ".join(f"{name:>11}" for name, _ in cols))
    for row in rows:
        cells = [
            f"{row['ok']}/{row['requests']}" if key is None else str(row[key]) for _, key in cols
        ]
        print(" | ".join(f"{c:>11}" for c in cells))
        if set(row["statuses"]) - {"200"}:
            print(f"{'':>11}   statuses: {row['statuses']}")


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Load test the Cairn service.")
    parser.add_argument("--url", default="http://127.0.0.1:8766")
    parser.add_argument("--endpoint", choices=list(PATHS), default="search")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--duration", type=float, default=20.0, help="seconds per level")
    parser.add_argument("--warmup", type=float, default=3.0, help="seconds, not measured")
    parser.add_argument("--key", default=None, help="API key, if the service needs one")
    parser.add_argument("--json", type=Path, default=None, help="also write the rows here")
    args = parser.parse_args(argv)

    questions = dev_questions()
    with httpx.Client(base_url=args.url, timeout=10) as client:
        if client.get("/health").status_code != 200:
            print("error: the service is not ready (GET /health is not 200)", file=sys.stderr)
            return 1
    if args.warmup > 0:
        run_level(args.url, args.endpoint, 1, args.warmup, questions, args.key)
    rows = [run_level(args.url, args.endpoint, n, args.duration, questions, args.key)
            for n in args.users]  # fmt: skip
    print_table(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
