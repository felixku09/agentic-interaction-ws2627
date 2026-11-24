"""Validate model-proposed tool arguments and dispatch to Python tools."""

import sys

from .data import get_calendar, get_weather, get_transit


def execute_tool(tool_name: str, arguments: dict, base_url: str) -> dict:
    """Execute a named tool. Validation errors are handled by the agent loop."""
    if not isinstance(arguments, dict):
        raise ValueError("Invalid tool arguments.")

    if tool_name == "ask_user":
        clarification = arguments.get("question")
        if not isinstance(clarification, str) or not clarification.strip():
            raise ValueError("Invalid clarification question.")

        print(f"\nAgent: {clarification}", file=sys.stderr)
        print("Du: ", end="", flush=True, file=sys.stderr)
        try:
            user_answer = input().strip()
        except EOFError:
            user_answer = ""

        if user_answer:
            return {"answer": user_answer}
        return {
            "status": "error",
            "error": "No user answer provided. Ask again.",
        }

    # All three data tools require a date.
    date = arguments.get("date")
    if not isinstance(date, str) or not date:
        raise ValueError("The date must be a string.")

    if tool_name == "get_calendar":
        result = get_calendar(date, base_url)
    elif tool_name == "get_weather":
        city = arguments.get("city")
        if not isinstance(city, str) or not city:
            raise ValueError("Invalid city.")
        result = get_weather(city, date, base_url)
    elif tool_name == "get_transit":
        from_city = arguments.get("from_city")
        to_city = arguments.get("to_city")
        if not isinstance(from_city, str) or not from_city:
            raise ValueError("Invalid departure city.")
        if not isinstance(to_city, str) or not to_city:
            raise ValueError("Invalid destination city.")
        result = get_transit(from_city, to_city, date, base_url)
    else:
        raise ValueError("Unknown tool: " + tool_name)

    if not isinstance(result, dict):
        raise ValueError("Invalid tool response.")
    return result
