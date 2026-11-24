"""The agent loop: call model, execute tools, observe results, and finish."""

import json
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI

from .config import load_settings
from .event_log import log_event
from .model_client import call_model_with_rate_limit
from .prompts import initial_messages
from .tools.definitions import ALL_TOOLS
from .tools.executor import execute_tool


def run_agent(question: str, log_path: Path, run_id: str, model: str) -> str:
    """Run the agent until it finishes or reaches the step limit."""
    # Keep the same public signature as the original implementation.
    # As before, the MODEL configuration determines the model actually used.
    settings = load_settings()
    model = settings["MODEL"]
    max_steps = settings["AGENT_MAX_STEPS"]

    client = OpenAI(
        api_key=settings["OPENAI_API_KEY"],
        base_url=settings["OPENAI_BASE_URL"],
        max_retries=0,
    )
    today = datetime.now().astimezone().date()
    messages = initial_messages(question, today)

    log_event(log_path, run_id, "user_task", model, args={"question": question})

    # One loop iteration = one agent model step. Network retries stay inside that step.
    for step in range(1, max_steps + 1):
        start = time.perf_counter()
        response = call_model_with_rate_limit(
            client=client,
            model=model,
            messages=messages,
            tools=ALL_TOOLS,
            tool_choice="auto",
            log_path=log_path,
            run_id=run_id,
        )
        model_duration_ms = (time.perf_counter() - start) * 1000
        assistant_message = response.choices[0].message
        messages.append(assistant_message)
        calls = assistant_message.tool_calls or []

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
            try:
                result = execute_tool(tool_name, arguments, settings["DATA_BASE_URL"])
            except Exception as exc:
                # A single failed tool must not stop the complete run.
                result = {"status": "error", "error": str(exc)}

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
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, ensure_ascii=False),
            })

    raise RuntimeError(
        f"step_limit_reached: No final answer after {max_steps} model calls."
    )
