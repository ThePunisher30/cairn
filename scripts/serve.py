"""Run the Cairn HTTP service (Phase 9).

    CAIRN_API_KEY=some-long-secret python scripts/serve.py
    python scripts/serve.py --llm none --retriever bm25 --no-auth   # the small public demo
    python scripts/serve.py --llm fake --no-auth          # no model, no key: for trying it out
    python scripts/serve.py --llm anthropic --model claude-opus-5-5   # needs ANTHROPIC_API_KEY

The API key is read from the CAIRN_API_KEY environment variable and never from the command line
(a command line is visible to every user of the machine and ends up in shell history).
The server starts listening at once; GET /health says 503 until the indexes are loaded.
"""

import argparse
import os
import sys
from pathlib import Path

from cairn.config import load_settings
from cairn.generation.factory import LLMS
from cairn.serve.app import create_app
from cairn.serve.engine import RETRIEVERS, EngineConfig, build_engine, warm_up
from cairn.serve.observe import configure_logging
from cairn.serve.state import Service


def main(argv=None):
    settings = load_settings()
    parser = argparse.ArgumentParser(description="Serve Cairn over HTTP.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--data-dir", type=Path, default=settings.data_dir)
    parser.add_argument("--corpus-root", type=Path, default=None)
    parser.add_argument("--llm", choices=[*LLMS, "none"], default="ollama",
                        help="none: retrieval only, /ask returns the passages")  # fmt: skip
    parser.add_argument("--retriever", choices=RETRIEVERS, default="hybrid",
                        help="bm25 needs no embedding model and little memory")  # fmt: skip
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--model", default=None)
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--no-auth", action="store_true", help="no API key (local use only)")
    parser.add_argument("--rate", type=float, default=None, help="requests per second per client")
    parser.add_argument("--burst", type=int, default=10)
    parser.add_argument("--max-concurrent", type=int, default=4, help="searches at once")
    parser.add_argument("--torch-threads", type=int, default=1, help="threads per embedding call")
    parser.add_argument("--semantic-cache", action="store_true")
    parser.add_argument("--semantic-threshold", type=float, default=0.92)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--debug-blocking-route", action="store_true", help="for the load test")
    args = parser.parse_args(argv)

    key = os.environ.get("CAIRN_API_KEY")
    if key is None and not args.no_auth:
        print("error: set CAIRN_API_KEY, or pass --no-auth for local use", file=sys.stderr)
        return 1
    if args.no_auth:
        key = None
    if key is not None and len(key) < 16:
        print("error: CAIRN_API_KEY must be at least 16 characters", file=sys.stderr)
        return 1

    import uvicorn

    configure_logging(args.log_level)
    cfg = EngineConfig(
        data_dir=args.data_dir,
        corpus_root=args.corpus_root,
        llm=args.llm,
        retriever=args.retriever,
        llm_base_url=args.ollama_url,
        model=args.model or ("gemma4:e4b" if args.llm == "ollama" else None),
        llm_timeout=args.llm_timeout,
        semantic_cache=args.semantic_cache,
        semantic_threshold=args.semantic_threshold,
        max_concurrent_retrievals=args.max_concurrent,
        torch_threads=args.torch_threads,
    )
    service = Service(lambda: build_engine(cfg), warm_up=warm_up)
    app = create_app(
        service,
        api_key=key,
        rate_limit=(args.rate, args.burst) if args.rate else None,
        blocking_demo=args.debug_blocking_route,
    )

    service.start()  # loads in a background thread; the server below listens straight away
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
