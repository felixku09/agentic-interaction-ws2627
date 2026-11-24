"""Groq/OpenAI-compatible model call and API retry policy."""

import math
import sys
import time
from openai import RateLimitError, BadRequestError

from .event_log import log_event

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
