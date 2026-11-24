"""Load project configuration without overriding process environment variables."""

import os
from pathlib import Path
from dotenv import load_dotenv

# The project's .env and README.md live above the agent/ package.
ROOT = Path(__file__).resolve().parent.parent


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)


def load_settings() -> dict:
    """Validate required settings; return the values used for this run."""
    load_environment()
    required = ("OPENAI_BASE_URL", "MODEL", "OPENAI_API_KEY", "DATA_BASE_URL")
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError("Missing configuration: " + ", ".join(missing))

    max_steps = int(os.getenv("AGENT_MAX_STEPS", "10"))
    if max_steps < 1:
        raise ValueError("AGENT_MAX_STEPS must be at least 1.")

    return {
        "OPENAI_BASE_URL": os.environ["OPENAI_BASE_URL"],
        "MODEL": os.environ["MODEL"],
        "OPENAI_API_KEY": os.environ["OPENAI_API_KEY"],
        "DATA_BASE_URL": os.environ["DATA_BASE_URL"],
        "AGENT_MAX_STEPS": max_steps,
    }
