"""Tool schemas shown to the language model; no tool is executed here."""

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


ASK_USER_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user a clarification question when essential "
            "information is missing to complete the original task. "
            "For example, ask for a missing destination, departure "
            "city or date. Do not guess missing information. "
            "Only ask when necessary. After receiving the answer, "
            "continue processing the original user request."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The clarification question to display to "
                        "the user, e.g. 'Where do you want to go?'"
                    ),
                }
            },
            "required": ["question"],
        },
    },
}


ALL_TOOLS = [CALENDAR_TOOL, WEATHER_TOOL, TRANSIT_TOOL, ASK_USER_TOOL]
