"""First Tool Call: one model -> calendar tool -> model exchange."""

import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from dotenv import load_dotenv
from openai import OpenAI

# The folder containing README.md and .env is one level above agent/.
ROOT = Path(__file__).resolve().parent.parent


# Tool description: the language model sees this, NOT our Python function itself.
CALENDAR_TOOL = {
    "type": "function",
    "function": {
        "name": "get_calendar",
        "description": (
            "Retrieve the appointments for exactly one date from the mock "
            "calendar. Use this tool for questions about calendar events, "
            "even if you think you already know the answer."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": "Calendar day in ISO format YYYY-MM-DD, e.g. 2026-10-29",
                }
            },
            "required": ["date"],
        },
    },
}

# current time in ISO-8601 format with UTC offset, e.g. 2026-10-29T12:34:56.789012+00:00
def now_iso() -> str:
    """An ISO-8601 timestamp including its UTC offset."""
    return datetime.now(timezone.utc).isoformat()

#information from event stored in a  dictionary and appended to a JSONL file for logging purposes
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

#Parameter date and adress of the mock calendar are passed to the function,
# which returns a dictionary with the calendar data for that date.
# If the date is invalid or there is no data for that date, an appropriate error message is returned.
def get_calendar(date: str, base_url: str) -> dict:
    """Return a day's calendar JSON. Treat HTTP 404 as 'no data'."""
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return {"date": date, "status": "error", "error": "Use YYYY-MM-DD."}
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {"date": date, "status": "error", "error": "Invalid date."}

    url = f"{base_url.rstrip('/')}/calendar/{date}.json"
    try:
        with urlopen(url, timeout=15) as response:
            return json.load(response)
    except HTTPError as exc:
        if exc.code == 404:
            return {
                "date": date,
                "status": "no_data",
                "events": [],
                "message": "No calendar data exists for this date (HTTP 404).",
            }
        return {"date": date, "status": "error", "error": f"HTTP {exc.code}"}
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "date": date,
            "status": "error",
            "error": f"Could not read the calendar: {type(exc).__name__}",
        }


def run_agent(question: str, log_path: Path, run_id: str, model: str) -> str:
    """Ask model to use the tool, execute it, then obtain the final answer."""
    # os.environ takes precedence over values in the file.
    load_dotenv(ROOT / ".env", override=False)
    required = ("OPENAI_BASE_URL", "MODEL", "OPENAI_API_KEY", "DATA_BASE_URL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing configuration: " + ", ".join(missing))

    model = os.environ["MODEL"]
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a calendar assistant. The mock calendar covers "
                "2026-10-26 through 2026-11-08. For calendar questions, "
                "use the get_calendar tool. Convert the requested date into "
                "YYYY-MM-DD. Only report appointments in the tool result; "
                "never invent appointments. Distinguish a date with no events "
                "from a date with no calendar data (status=no_data). "
                "If status=error, explain that the information could not "
                "be retrieved. Respond in the language of the user."
            ),
        },
        {"role": "user", "content": question},
    ]
    log_event(log_path, run_id, "user_task", model, args={"question": question})

    # Step 1: the model requests a tool call.
    start = time.perf_counter()
    first = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=[CALENDAR_TOOL],
        tool_choice="required",
    )
    model_duration_ms = (time.perf_counter() - start) * 1000
    assistant_message = first.choices[0].message
    calls = assistant_message.tool_calls or []
    if not calls:
        raise RuntimeError("The model did not request a calendar tool call.")
    messages.append(assistant_message)

    # Step 2: Python executes the tool and adds its result to messages.
    for call in calls:
        if call.function.name != "get_calendar":
            raise RuntimeError("Unknown tool requested by the model.")
        arguments = json.loads(call.function.arguments)
        date = arguments.get("date")
        if not isinstance(date, str):
            raise RuntimeError("The model's tool date must be a string.")
        log_event(log_path, run_id, "tool_call", model, tool="get_calendar",
                  args={"date": date}, duration_ms=model_duration_ms)
        start = time.perf_counter()
        result = get_calendar(date, os.environ["DATA_BASE_URL"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        log_event(log_path, run_id, "tool_result", model, tool="get_calendar",
                  args={"date": date}, ok=result.get("status") != "error",
                  duration_ms=elapsed_ms, result=result)
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": json.dumps(result, ensure_ascii=False),
        })

    # Step 3: the model uses the tool result to produce the answer.
    start = time.perf_counter()
    second = client.chat.completions.create(model=model, messages=messages)
    elapsed_ms = (time.perf_counter() - start) * 1000
    answer = second.choices[0].message.content
    if not answer:
        raise RuntimeError("The model did not return a final answer.")
    log_event(log_path, run_id, "final", model, duration_ms=elapsed_ms,
              answer=answer)
    return answer


def main() -> int:
    # A separate log for every invocation, including failed ones.
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    run_id = uuid.uuid4().hex
    log_path = log_dir / f"{run_id}.jsonl"
    started = time.perf_counter()
    load_dotenv(ROOT / ".env", override=False)
    model = os.getenv("MODEL", "")
    try:
        if len(sys.argv) < 2:
            raise ValueError('Specify a question, e.g. python -m agent "Was steht am 29.10.2026 im Kalender?"')
        answer = run_agent(" ".join(sys.argv[1:]), log_path, run_id, model)
        print(answer)  # Final answer only on standard output.
        return 0
    except Exception as exc:
        log_event(log_path, run_id, "error", os.getenv("MODEL", ""), ok=False,
                  duration_ms=(time.perf_counter() - started) * 1000,
                  error_type=type(exc).__name__, error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
