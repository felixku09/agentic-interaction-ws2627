"""First Tool Call: one model -> calendar tool -> model exchange."""

import json
import os
import re
import sys
import time
import uuid
import math
from openai import OpenAI, RateLimitError, BadRequestError
from datetime import datetime, timedelta, timezone
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

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": (
            "Retrieve weather forecast data for one city and one date "
            "from the mock weather API. Use this tool for questions "
            "about weather, rain, temperature or umbrellas. "
            "Use city slugs such as 'ulm' or 'muenchen', "
            "not 'munich' or 'München'. "
            "Missing data does not mean good weather."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": (
                        "City slug from index.json, "
                        "e.g. 'ulm' or 'muenchen'."
                    ),
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Date in YYYY-MM-DD format, "
                        "e.g. '2026-10-29'."
                    ),
                },
            },
            "required": ["city", "date"],
        },
    },
}

TRANSIT_TOOL = {
    "type": "function",
    "function": {
        "name": "get_transit",
        "description": (
            "Retrieve train connections between two cities "
            "for a specific date from the mock transit API. "
            "Use this tool for questions about trains, "
            "departures, arrivals and travel connections. "
            "Use city slugs such as 'ulm' and 'muenchen'. "
            "Do not invent connections when no data is available."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "from_city": {
                    "type": "string",
                    "description": (
                        "Departure city slug, e.g. 'ulm'."
                    ),
                },
                "to_city": {
                    "type": "string",
                    "description": (
                        "Destination city slug, e.g. 'muenchen'."
                    ),
                },
                "date": {
                    "type": "string",
                    "description": (
                        "Travel date in YYYY-MM-DD format."
                    ),
                },
            },
            "required": ["from_city", "to_city", "date"],
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

# Parameter date and adress of the mock calendar are passed to the function,
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

def get_weather(city: str, date: str, base_url: str) -> dict:
    """Retrieve weather data for one city and date."""

    # 1. Validate the date format
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return {
            "city": city,
            "date": date,
            "status": "error",
            "error": "Use YYYY-MM-DD."
        }

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "city": city,
            "date": date,
            "status": "error",
            "error": "Invalid date."
        }

    # 2. Construct the API URL
    url = f"{base_url.rstrip('/')}/weather/{city}/{date}.json"

    # 3. Retrieve weather data
    try:
        with urlopen(url, timeout=15) as response:
            return json.load(response)

    # 4. Handle missing data
    except HTTPError as exc:
        if exc.code == 404:
            return {
                "city": city,
                "date": date,
                "status": "no_data",
                "message": "No weather data available."
            }

        return {
            "city": city,
            "date": date,
            "status": "error",
            "error": f"HTTP {exc.code}"
        }

    # 5. Handle connection and data errors
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "city": city,
            "date": date,
            "status": "error",
            "error": f"Could not read weather: {type(exc).__name__}"
        }

def get_transit(from_city: str, to_city: str, date: str, base_url: str) -> dict:
    """Retrieve train connections between two cities for one date."""

    # 1. Validate date
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return {
            "status": "error",
            "error": "Use YYYY-MM-DD."
        }

    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        return {
            "status": "error",
            "error": "Invalid date."
        }

    # 2. Validate city slugs
    for city in (from_city, to_city):
        if not re.fullmatch(r"[a-z0-9_-]+", city):
            return {
                "status": "error",
                "error": "Invalid city slug."
            }

    # 3. Construct URL
    url = (
        f"{base_url.rstrip('/')}/transit/"
        f"{from_city}/{to_city}/{date}.json"
    )

    # 4. Retrieve transit data
    try:
        with urlopen(url, timeout=15) as response:
            return json.load(response)

    # 5. Handle missing data
    except HTTPError as exc:
        if exc.code == 404:
            return {
                "from": from_city,
                "to": to_city,
                "date": date,
                "status": "no_data",
                "message": "No transit data available."
            }

        return {
            "status": "error",
            "error": f"HTTP {exc.code}"
        }

    # 6. Handle connection and data errors
    except (URLError, TimeoutError, ValueError, OSError) as exc:
        return {
            "status": "error",
            "error": f"Could not read transit: {type(exc).__name__}"
        }


def call_model_with_rate_limit(
    client, model, messages, tools, tool_choice, log_path, run_id
):
    """Call the model and wait if a rate limit is reached."""
    #Spielraum für Agenten bei vorübergehenden Fehlern im Modellaufruf, z.B. Rate-Limit oder ungültiger Tool-Aufruf.
    #verhindert unbegrenzte Wiederholungen und Wartezeiten.
    max_retries = 6
    max_total_wait = 300  #5min
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

        #Ein Sprachmodell arbeitet nicht wie eine gewöhnliche Funktion, 
        #die bei identischer Eingabe immer garantiert dieselbe Ausgabe erzeugt.
        #Falsches Format der Tool Argumente(ungültiger Tool-Aufruf) get_calendar<|channel|>commentary → Fehler
        #Ein erneuter Modellaufruf kann einen gültigen Tool-Aufruf erzeugen,
        #obwohl der vorherige Versuch fehlgeschlagen ist.
        
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

    # Current date based on the computer's local timezone
    today = datetime.now().astimezone().date()

    #Der System-Prompt legt Rolle, Regeln und Ausgabeformat fest.
    # 2. Prepare messages

    messages = [
        {
            "role": "system",
            "content": (
                # Role and environment
                "You are a calendar, weather and transit assistant. "
                "The mock data covers 2026-10-26 through 2026-11-08. "

                # Current date and relative dates
                f"Today's date is {today.isoformat()}. "
                "This date uses the computer's local timezone. "
                "Resolve relative expressions such as 'today', "
                "'tomorrow', 'yesterday', 'heute', 'morgen' and "
                "'gestern' using today's date. "
                "Always convert dates to YYYY-MM-DD before calling tools. "
                "Never assume that today is within the mock data period. "

                # Tool selection
                "Use get_calendar for questions about appointments "
                "and calendar events. "
                "Use get_weather for questions about weather, "
                "temperature, rain and umbrellas. "
                "Use get_transit for questions about train connections, "
                "departures, arrivals and travel routes. "

                # City names
                "Use city slugs from the mock data, such as "
                "'ulm' and 'muenchen'. "
                "Convert city names such as Munich or München "
                "to 'muenchen'. "

                # Multiple tools and dates
                "If a question requires multiple tools, call all "
                "necessary tools before giving the final answer. "
                "For every explicitly requested date, check the "
                "relevant tools for that date. "
                "Do not assume missing data without checking the tool. "
                "Only provide a factual final answer after all "
                "necessary tool calls have been completed. "

                # Missing information
                "If a required city, departure location, "
                "destination or date cannot be determined, "
                "ask the user instead of guessing. "

                # Error handling and reliability
                "Never invent appointments, weather information "
                "or train connections. "
                "Distinguish days without events from days "
                "without data. "
                "If a tool returns status 'no_data', explain "
                "that no data is available. "
                "If a tool returns status 'error', explain "
                "that the information could not be retrieved. "
                "If one tool call fails, continue with other "
                "independent tool calls whenever possible. "
                "Do not interpret missing weather data as no rain "
                "or missing transit data as no trains. "

                # Output format
                "Respond in the language of the user. "
                "Output plain text suitable for a terminal. "
                "Do not use Markdown tables, HTML, <br>, "
                "or bold markers like **. "
                "Use short headings and bullet points. "
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
            tools=[CALENDAR_TOOL, WEATHER_TOOL, TRANSIT_TOOL],
            tool_choice="auto",
            log_path=log_path,
            run_id=run_id,
        )

        model_duration_ms = (time.perf_counter() - start) * 1000    #Dauer des Modellaufrufs in Millisekunden

        assistant_message = response.choices[0].message             #Modellantwort
        messages.append(assistant_message)                          #Modellantwort wird dem Gesprächsverlauf hinzugefügt

        calls = assistant_message.tool_calls or []                  #Prüfe on das Modell Tools verwenden möchte
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
                if not isinstance(arguments, dict):
                    raise ValueError("Invalid tool arguments.")

                date = arguments.get("date")

                if not isinstance(date, str) or not date:
                    raise ValueError("The date must be a string.")

                base_url = os.environ["DATA_BASE_URL"]

                # Calendar tool
                if tool_name == "get_calendar":
                    result = get_calendar(date, base_url)

                # Weather tool
                elif tool_name == "get_weather":
                    city = arguments.get("city")

                    if not isinstance(city, str) or not city:
                        raise ValueError("Invalid city.")

                    result = get_weather(city, date, base_url)

                # Transit tool
                elif tool_name == "get_transit":
                    from_city = arguments.get("from_city")
                    to_city = arguments.get("to_city")

                    if not isinstance(from_city, str) or not from_city:
                        raise ValueError("Invalid departure city.")

                    if not isinstance(to_city, str) or not to_city:
                        raise ValueError("Invalid destination city.")

                    result = get_transit(
                        from_city, to_city, date, base_url
                    )

                # Unknown tool
                else:
                    raise ValueError("Unknown tool: " + tool_name)

                if not isinstance(result, dict):
                    raise ValueError("Invalid tool response.")

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
