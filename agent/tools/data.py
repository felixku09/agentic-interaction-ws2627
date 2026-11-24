"""Read-only HTTP tools for the course mock-data API."""

import json
import re
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

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
