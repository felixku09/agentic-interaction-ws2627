# Agentic Interaction WS 2026/27 — Exercise 1

A small Python calendar agent demonstrating **model → tool → model** using an OpenAI-compatible endpoint and a public mock calendar API.

## Requirements

- Python 3.11 or newer (Python 3.13/3.14 recommended)
- An OpenAI-compatible API key (e.g. Groq)

## Setup (Windows)

From the **repository root**:

```cmd
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a local `.env` file and set `OPENAI_API_KEY` to your own key. The file is excluded from Git. Environment variables from the system have priority over `.env`.

## Run

From the repository root:

```cmd
.venv\Scripts\python.exe -m agent "Was steht am 29.10.2026 im Kalender?"
```

Or, if the virtual environment is activated:

```cmd
python -m agent "Was steht am 29.10.2026 im Kalender?"
```

Each execution writes one JSONL log under `logs/`, with fields `ts`, `run_id`, `event`, `tool`, `args`, `ok`, `model`, `duration_ms`. Events include `user_task`, `tool_call`, `tool_result`, `final`, and `error` when a run fails. Logs are not committed.

## Design

1. Give the model a description of `get_calendar(date)` as a tool.
2. Execute the returned tool call locally with an HTTP request to `{DATA_BASE_URL}/calendar/{YYYY-MM-DD}.json`.
3. Give the tool's JSON data back to the model, which composes the final answer.

`404` responses are treated as *no available data* rather than as a fatal error. Other failed requests are surfaced clearly to the model.

## Security

**Do not commit `.env` or any API key.**
