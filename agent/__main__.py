"""CLI entry point: run with `python -m agent "your question"`."""

import os
import sys
import time
import uuid

from .agent_loop import run_agent
from .config import ROOT, load_environment
from .event_log import log_event


def main() -> int:
    # A separate log for every invocation, including failed ones.
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    log_path = log_dir / f"{run_id}.jsonl"
    started = time.perf_counter()
    load_environment()
    model = os.getenv("MODEL", "")

    try:
        if len(sys.argv) < 2:
            raise ValueError(
                'Specify a question, e.g. python -m agent "Was steht am 29.10.2026 im Kalender?"'
            )
        answer = run_agent(" ".join(sys.argv[1:]), log_path, run_id, model)
        print(answer)  # Final answer only on standard output.
        return 0
    except Exception as exc:
        log_event(
            log_path, run_id, "error", os.getenv("MODEL", ""),
            ok=False,
            duration_ms=(time.perf_counter() - started) * 1000,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
