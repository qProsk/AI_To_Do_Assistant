# AI To-Do Assistant

A lightweight task management application built with FastAPI and SQLite. Supports both rule-based and AI-powered (OpenAI) natural-language task parsing, with a clean browser-based interface served directly from the backend.

---

## Features

- **Natural language input** — paste a list of tasks in plain text; the parser extracts titles, priorities, time estimates, tags, and next steps automatically
- **Two parse modes** — a fast rule-based parser (no API key required) and an AI-powered parser via OpenAI GPT
- **Full CRUD** — create, read, update, and delete tasks via a REST API
- **Task notes** — attach freeform notes to any task
- **Bulk actions** — mark multiple tasks done or delete them in a single request
- **Today view** — tasks due today and overdue tasks, sorted by priority
- **Filtering and sorting** — filter by status, priority, tag, or full-text search; sort by date, priority, estimate, or deadline
- **Statistics** — completion rate, average estimate, priority breakdown, and top tags
- **Persistent storage** — SQLite with automatic schema migration on startup
- **Single-file frontend** — no build step required; served as a static file

---

## Requirements

- Python 3.10 or later
- An OpenAI API key (optional — only required for AI parsing)

---

## Installation

```bash
git clone https://github.com/qProsk/AI_To_Do_Assistant.git
cd AI_To_Do_Assistant

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...        # optional
OPENAI_MODEL=gpt-4o-mini     # optional, default: gpt-4o-mini
TODO_DB_PATH=todo.db         # optional, default: todo.db
```

If `OPENAI_API_KEY` is not set, the application falls back to the rule-based parser for all requests.

---

## Running

```bash
uvicorn main:app --reload
```

The API is available at `http://127.0.0.1:8000`.
The browser interface is available at `http://127.0.0.1:8000/app`.
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

---

## Project Structure

```
AI_To_Do_Assistant/
├── main.py                  # FastAPI application and all backend logic
├── frontend/
│   └── index.html           # Browser interface
├── requirements.txt
├── .env                     # Local environment variables (not committed)
├── .gitignore
└── todo.db                  # SQLite database (auto-created on first run)
```

---

## API Overview

### System

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check — returns server, database, and OpenAI status |

### Parser

| Method | Path | Description |
|--------|------|-------------|
| POST | `/parse_fast` | Rule-based parsing, no API key required |
| POST | `/parse_ai` | AI parsing via OpenAI |
| POST | `/parse` | Auto mode — AI with fallback to rule-based |
| POST | `/smart_add` | Parse and immediately save tasks (rule-based) |
| POST | `/smart_add_ai` | Parse and immediately save tasks (AI) |

### Tasks

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tasks` | Create one or more tasks |
| GET | `/tasks` | List tasks with optional filters and pagination |
| GET | `/tasks/{id}` | Get a single task |
| PATCH | `/tasks/{id}` | Update task fields |
| PATCH | `/tasks/{id}/done` | Toggle completion status |
| DELETE | `/tasks/{id}` | Delete a task |
| POST | `/tasks/bulk/done` | Mark multiple tasks as done |
| POST | `/tasks/bulk/delete` | Delete multiple tasks |

### Notes

| Method | Path | Description |
|--------|------|-------------|
| POST | `/tasks/{id}/notes` | Add a note to a task |
| GET | `/tasks/{id}/notes` | List notes for a task |
| DELETE | `/tasks/{id}/notes/{note_id}` | Delete a note |

### Dashboard

| Method | Path | Description |
|--------|------|-------------|
| GET | `/today` | Tasks due today and overdue tasks |
| GET | `/stats` | Aggregate statistics |

Full request/response schemas are available in the interactive docs at `/docs`.

---

## Database

The application uses SQLite and requires no external database server. The schema is created automatically on first run. If an existing database from a previous version is detected, missing columns are added automatically without data loss.

The database file location can be changed via the `TODO_DB_PATH` environment variable.

---

## License

MIT
