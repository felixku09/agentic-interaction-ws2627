"""Write append-only JSONL events for each agent run."""

import json
from datetime import datetime, timezone
from pathlib import Path

def now_iso() -> str:
    """An ISO-8601 timestamp including its UTC offset."""
    return datetime.now(timezone.utc).isoformat()


def log_event(log_path: Path, run_id: str, event: str, model: str,
              *, tool=None, args=None, ok=True, duration_ms=0, **extra) -> None:
    """Append one JSON object to this run's private JSONL file."""
    record = {
        "ts": now_iso(),
        "run_id": run_id,
        "event": event,
        "tool": tool,
        "args": args,
        "ok": bool(ok),
        "model": model,
        "duration_ms": int(duration_ms),
    }
    record.update(extra)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
