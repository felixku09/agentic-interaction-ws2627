"""First Tool Call: one model -> calendar tool -> model exchange."""

import json
import os
import re
import sys
import time
import uuid
import math
from openai import OpenAI, RateLimitError, BadRequestError
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from dotenv import load_dotenv

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

def call_model_with_rate_limit(
    client, model, messages, tools, tool_choice, log_path, run_id
):
    """Call the model and wait if a rate limit is reached."""

    max_retries = 6
    max_total_wait = 300  # Maximum waiting time: 5 minutes
    started = time.monotonic()

    for attempt in range(max_retries + 1):
        try:
            # Request the response including HTTP headers
            raw = client.chat.completions.with_raw_response.create(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.2,
            )

            response = raw.parse()

            # Read Groq's current token limits
            limit = raw.headers.get("x-ratelimit-limit-tokens")
            remaining = raw.headers.get("x-ratelimit-remaining-tokens")
            reset = raw.headers.get("x-ratelimit-reset-tokens")

            # Tokens used by this individual request
            usage = response.usage
            request_tokens = (
                usage.total_tokens if usage is not None else None
            )

            if limit is not None and remaining is not None:
                limit = int(limit)
                remaining = int(remaining)
                used = max(0, limit - remaining)

                print(
                    f"\nTokenfenster: {used}/{limit} Tokens "
                    f"(noch {remaining} verfuegbar)",
                    file=sys.stderr
                )

            if request_tokens is not None:
                print(
                    f"Dieser Modellaufruf: {request_tokens} Tokens",
                    file=sys.stderr
                )

            if reset is not None:
                print(
                    f"Tokenfenster-Reset: {reset}",
                    file=sys.stderr
                )

            return response

        except RateLimitError as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    "Rate-Limit: Maximum retry count reached."
                ) from exc

            # Groq provides the waiting time on HTTP 429
            retry_after = exc.response.headers.get("retry-after")

            try:
                wait_seconds = max(
                    1, math.ceil(float(retry_after))
                )
            except (TypeError, ValueError):
                # Fallback if the server gives no valid delay
                wait_seconds = min(60, 2 ** (attempt + 1))

            elapsed = time.monotonic() - started

            if elapsed + wait_seconds > max_total_wait:
                raise RuntimeError(
                    "Rate-Limit: Maximum total waiting time reached."
                ) from exc

            # Record that a rate limit was reached
            log_event(
                log_path, run_id, "error", model,
                ok=False,
                error_type="RateLimitError",
                retry_after_seconds=wait_seconds,
                will_retry=True,
            )

            # Countdown in the terminal
            for seconds in range(wait_seconds, 0, -1):
                print(
                    f"\rRate-Limit erreicht. "
                    f"Neuer Versuch in {seconds} Sekunden ...",
                    end="",
                    flush=True,
                    file=sys.stderr,
                )
                time.sleep(1)

            print(
                "\rWartezeit beendet. Wiederhole Modellaufruf.     ",
                file=sys.stderr,
            )


        except BadRequestError as exc:
            error_message = str(exc).lower()

            # Only retry invalid tool calls
            if (
                "tool call validation failed" not in error_message
                and "tool_use_failed" not in error_message
            ):
                raise

            # Stop after at most 3 attempts
            if attempt >= 2:
                raise

            log_event(
                log_path,
                run_id,
                "error",
                model,
                ok=False,
                error_type="ToolCallValidationError",
                will_retry=True,
            )

            print(
                f"Ungueltiger Tool-Aufruf. "
                f"Wiederhole Modellanfrage (Versuch {attempt + 2}).",
                file=sys.stderr,
            )

            continue


def run_agent(question: str, log_path: Path, run_id: str, model: str) -> str:
    """Run the agent until it finishes or reaches the step limit."""

    # 1. Load configuration
    load_dotenv(ROOT / ".env", override=False)

    required = ("OPENAI_BASE_URL", "MODEL", "OPENAI_API_KEY", "DATA_BASE_URL")
    missing = [name for name in required if not os.getenv(name)]

    if missing:
        raise RuntimeError("Missing configuration: " + ", ".join(missing))

    model = os.environ["MODEL"]
    max_steps = int(os.getenv("AGENT_MAX_STEPS", "10"))

    if max_steps < 1:
        raise ValueError("AGENT_MAX_STEPS must be at least 1.")

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        max_retries=0,
    )

    # 2. Prepare messages
    messages = [
    {
        "role": "system",
        "content": (
            "You are a calendar assistant. The mock calendar covers "
            "2026-10-26 through 2026-11-08. "
            "For calendar questions, use the get_calendar tool. "
            "Convert dates into YYYY-MM-DD. "
            "If the user asks about multiple dates, check every "
            "requested date before giving your final answer. "
            "Never invent appointments. "
            "Distinguish days without events from days without data. "
            "If one tool call fails, continue checking other dates "
            "and explain which information was unavailable. "
            "Respond in the language of the user. "
            "Output plain text suitable for a terminal. "
            "Do not use Markdown tables, HTML, <br>, or bold markers like **. "
            "Use short headings and bullet points with one line per appointment."
            "For every date explicitly requested by the user, "
            "call get_calendar, including dates outside the known "
            "calendar range. Do not assume missing calendar data "
            "without checking the tool. "
            "Only give the final answer after all requested dates "
            "have been checked."
        ),
    },
    {"role": "user", "content": question},
    ]   

    log_event(
        log_path, run_id, "user_task", model,
        args={"question": question}
    )

    # 3. Agent loop: each iteration is one model call
    for step in range(1, max_steps + 1):

        start = time.perf_counter()

        response = call_model_with_rate_limit(
            client=client,
            model=model,
            messages=messages,
            tools=[CALENDAR_TOOL],
            tool_choice="required" if step == 1 else "auto",
            log_path=log_path,
            run_id=run_id,
        )

        model_duration_ms = (time.perf_counter() - start) * 1000

        assistant_message = response.choices[0].message
        messages.append(assistant_message)

        calls = assistant_message.tool_calls or []

        # 4. No tool calls: the agent has finished
        if not calls:
            answer = assistant_message.content

            if not answer:
                raise RuntimeError("The model returned no final answer.")

            log_event(
                log_path, run_id, "final", model,
                duration_ms=model_duration_ms,
                answer=answer,
                steps=step,
                reason="completed",
            )

            return answer

        # 5. Execute all requested tool calls
        for call in calls:
            tool_name = call.function.name

            try:
                arguments = json.loads(call.function.arguments)
            except (ValueError, TypeError):
                arguments = None

            log_event(
                log_path, run_id, "tool_call", model,
                tool=tool_name,
                args=arguments if isinstance(arguments, dict)
                     else {"raw": call.function.arguments},
                duration_ms=model_duration_ms,
                step=step,
            )

            start_tool = time.perf_counter()

            # 6. Handle errors without stopping the entire agent
            try:
                if tool_name != "get_calendar":
                    raise ValueError("Unknown tool: " + tool_name)

                if not isinstance(arguments, dict):
                    raise ValueError("Invalid tool arguments.")

                date = arguments.get("date")

                if not isinstance(date, str):
                    raise ValueError("The date must be a string.")

                result = get_calendar(
                    date,
                    os.environ["DATA_BASE_URL"]
                )

                if not isinstance(result, dict):
                    raise ValueError("Invalid calendar response.")

            except Exception as exc:
                result = {
                    "status": "error",
                    "error": str(exc),
                }

            tool_duration_ms = (time.perf_counter() - start_tool) * 1000

            log_event(
                log_path, run_id, "tool_result", model,
                tool=tool_name,
                args=arguments,
                ok=result.get("status") != "error",
                duration_ms=tool_duration_ms,
                result=result,
                step=step,
            )

            # 7. Return the tool result to the model
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    # 8. The agent did not finish within the allowed steps
    raise RuntimeError(
        f"step_limit_reached: No final answer after {max_steps} model calls."
    )
   
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
