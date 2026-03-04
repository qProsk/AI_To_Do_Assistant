"""
AI To-Do Assistant - Backend API
FastAPI + SQLite, optional OpenAI integration
"""

from __future__ import annotations

import json
import logging
import os
import re
import traceback
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Dict, Generator, List, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    import sqlite3
except ImportError as exc:
    raise RuntimeError("sqlite3 is required") from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DB_PATH: str = os.getenv("TODO_DB_PATH", "todo.db")

log.info("OpenAI API key present: %s", bool(API_KEY))
log.info("Model: %s", MODEL)

openai_client: Optional[Any] = None
if API_KEY and OpenAI is not None:
    openai_client = OpenAI(api_key=API_KEY)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AI To-Do Assistant",
    version="2.0.0",
    description="Natural-language task parsing with full CRUD over SQLite.",
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")


@app.get("/app", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse("frontend/index.html")


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    title        TEXT    NOT NULL,
    priority     TEXT    NOT NULL,
    estimate_min INTEGER NOT NULL,
    deadline     TEXT,
    tags_json    TEXT    NOT NULL DEFAULT '[]',
    next_step    TEXT    NOT NULL DEFAULT '',
    done         INTEGER NOT NULL DEFAULT 0,
    done_at      TEXT,
    created_at   TEXT    NOT NULL,
    updated_at   TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS task_notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    INTEGER NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    body       TEXT    NOT NULL,
    created_at TEXT    NOT NULL
);
"""


def _migrate(conn: sqlite3.Connection) -> None:
    """Apply incremental schema migrations on existing databases."""
    existing = {
        row[1]
        for row in conn.execute("PRAGMA table_info(tasks)").fetchall()
    }
    migrations = [
        ("done_at",    "ALTER TABLE tasks ADD COLUMN done_at    TEXT"),
        ("updated_at", "ALTER TABLE tasks ADD COLUMN updated_at TEXT NOT NULL DEFAULT ''"),
    ]
    for column, sql in migrations:
        if column not in existing:
            log.info("Migration: adding column '%s' to tasks", column)
            conn.execute(sql)

    # Back-fill updated_at for rows that have an empty string (just migrated)
    conn.execute(
        "UPDATE tasks SET updated_at = created_at WHERE updated_at = '' OR updated_at IS NULL"
    )


def init_db() -> None:
    with db_connection() as conn:
        for statement in SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        _migrate(conn)


@contextmanager
def db_connection() -> Generator[sqlite3.Connection, None, None]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


init_db()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def encode_tags(tags: List[str]) -> str:
    return json.dumps(tags, ensure_ascii=False)


def decode_tags(raw: str) -> List[str]:
    try:
        return json.loads(raw) if raw else []
    except (json.JSONDecodeError, ValueError):
        return []


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def today_iso() -> str:
    return date.today().isoformat()


def row_to_task_row(r: sqlite3.Row) -> "TaskRow":
    return TaskRow(
        id=r["id"],
        title=r["title"],
        priority=r["priority"],
        estimate_min=r["estimate_min"],
        deadline=r["deadline"],
        tags=decode_tags(r["tags_json"]),
        next_step=r["next_step"],
        done=bool(r["done"]),
        done_at=r["done_at"],
        created_at=r["created_at"],
        updated_at=r["updated_at"],
    )


def priority_sort_key(priority: str) -> int:
    return {"high": 0, "medium": 1, "low": 2}.get(priority, 9)


PRIORITY_SQL_CASE = """
    CASE priority
        WHEN 'high'   THEN 0
        WHEN 'medium' THEN 1
        WHEN 'low'    THEN 2
        ELSE 9
    END
"""


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

Priority = Literal["high", "medium", "low"]
SortMode = Literal["newest", "oldest", "prio", "estimate", "deadline"]


class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=8000)


class Task(BaseModel):
    title: str = Field(..., min_length=1)
    priority: Priority
    estimate_min: int = Field(..., ge=1, le=1440)
    deadline: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    next_step: str = Field(default="")


class TaskCreate(Task):
    pass


class ParseResponse(BaseModel):
    tasks: List[Task]
    mode: str


class TaskRow(Task):
    id: int
    done: bool
    done_at: Optional[str] = None
    created_at: str
    updated_at: str


class NoteRow(BaseModel):
    id: int
    task_id: int
    body: str
    created_at: str


class SaveTasksRequest(BaseModel):
    tasks: List[Task] = Field(default_factory=list)


class SaveTasksResponse(BaseModel):
    inserted_ids: List[int]


class UpdateTaskRequest(BaseModel):
    title: Optional[str] = None
    priority: Optional[Priority] = None
    estimate_min: Optional[int] = Field(default=None, ge=1, le=1440)
    deadline: Optional[str] = None
    tags: Optional[List[str]] = None
    next_step: Optional[str] = None


class AddNoteRequest(BaseModel):
    body: str = Field(..., min_length=1)


class TodayResponse(BaseModel):
    today: List[TaskRow]
    overdue: List[TaskRow]
    total_estimated_minutes_today: int


class StatsResponse(BaseModel):
    total: int
    done: int
    pending: int
    completion_rate: float
    by_priority: Dict[str, int]
    top_tags: List[Dict[str, Any]]
    avg_estimate_min: float


class BulkActionRequest(BaseModel):
    ids: List[int] = Field(..., min_items=1)


class HealthResponse(BaseModel):
    status: str
    version: str
    db: str
    openai: bool


# ---------------------------------------------------------------------------
# Rule-based (fast) parser
# ---------------------------------------------------------------------------

BULLET_RE = re.compile(r"^\s*([-*+]|(\d+[.\)]))\s+")
WHITESPACE_RE = re.compile(r"\s+")

_URGENT = {"urgent", "hned", "asap", "dnes", "zajtra", "termin", "deadline"}
_WORK = {"faktura", "invoice", "klient", "email", "mail", "meeting", "call", "zmluva", "ponuka", "report"}
_FINANCE = {"faktura", "platba", "zaplatit", "bank", "ucet", "dph", "dan"}
_HOME = {"upratat", "upratovanie", "umyvanie", "vysvat", "riad", "dom", "izba"}
_SHOP = {"kupit", "nakupit", "shop", "obchod", "objednat"}


def _normalize(text: str) -> str:
    """Lowercase and strip diacritics for keyword matching."""
    table = str.maketrans(
        "áäčďéíľĺňóôŕšťúýžÁÄČĎÉÍĽĹŇÓÔŔŠŤÚÝŽ",
        "aacdeilnoorstuyyzAACDEILNOORSTUYYZ",
    )
    return text.lower().translate(table)


def _split_to_lines(text: str) -> List[str]:
    lines: List[str] = []
    for line in text.splitlines():
        line = BULLET_RE.sub("", line.strip()).strip()
        line = WHITESPACE_RE.sub(" ", line)
        if len(line) >= 3:
            lines.append(line)

    if not lines:
        for part in text.split(";"):
            part = WHITESPACE_RE.sub(" ", part.strip())
            if len(part) >= 3:
                lines.append(part)

    return lines


def _infer_tags(title: str) -> List[str]:
    t = _normalize(title)
    tags: List[str] = []
    if any(w in t for w in _WORK):
        tags.append("praca")
    if any(w in t for w in _FINANCE):
        tags.append("financie")
    if any(w in t for w in _HOME):
        tags.append("domacnost")
    if any(w in t for w in _SHOP):
        tags.append("nakup")
    return list(dict.fromkeys(tags)) or ["osobne"]


def _infer_priority(title: str) -> Priority:
    t = _normalize(title)
    if any(w in t for w in _URGENT):
        return "high"
    if any(w in t for w in _WORK) or any(w in t for w in _FINANCE):
        return "medium"
    return "low"


def _infer_estimate(title: str) -> int:
    t = _normalize(title)
    if any(x in t for x in ["zavolat", "call", "email", "mail", "poslat", "napisat"]):
        return 5
    if any(x in t for x in ["kupit", "nakupit", "objednat"]):
        return 10
    if any(x in t for x in ["upratat", "upratovanie", "vysvat", "umyvanie", "riad"]):
        return 20
    return 15


def _infer_next_step(title: str) -> str:
    return f"Begin with the first concrete action for: {title}"


def rule_based_parse(text: str) -> List[Task]:
    tasks: List[Task] = []
    for line in _split_to_lines(text):
        tasks.append(
            Task(
                title=line,
                priority=_infer_priority(line),
                estimate_min=_infer_estimate(line),
                tags=_infer_tags(line),
                next_step=_infer_next_step(line),
            )
        )
    tasks.sort(key=lambda t: (priority_sort_key(t.priority), t.estimate_min))
    return tasks


def fast_parse_response(text: str, mode: str) -> ParseResponse:
    return ParseResponse(tasks=rule_based_parse(text), mode=mode)


# ---------------------------------------------------------------------------
# AI parser (OpenAI)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """
You are a task parser. Return ONLY valid JSON matching this schema exactly:

{
  "tasks": [
    {
      "title": "string",
      "priority": "high|medium|low",
      "estimate_min": integer,
      "deadline": "YYYY-MM-DD or null",
      "tags": ["string"],
      "next_step": "string"
    }
  ]
}

Rules:
- Produce one task object per distinct task found in the input.
- Estimate times in minutes (1-1440).
- Tags: 1-5 short lowercase labels (e.g. work, finance, home, shopping, personal).
- next_step: one concrete, actionable sentence.
- Output JSON only. No markdown fences, no extra text.
""".strip()

_REPAIR_PROMPT = (
    "Fix the following so it is valid JSON matching the required schema. "
    "Return JSON only, no commentary."
)


def _call_model(messages: List[Dict[str, str]]) -> str:
    if openai_client is None:
        raise RuntimeError("OpenAI client is not initialized")

    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("Model returned empty response")
    return content.strip()


def _strip_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    return text.strip()


def ai_parse(text: str) -> ParseResponse:
    raw = _call_model(
        [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Tasks:\n{text}"},
        ]
    )
    cleaned = _strip_fences(raw)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        log.warning("Initial JSON parse failed; attempting repair")
        repaired = _call_model(
            [
                {"role": "system", "content": _REPAIR_PROMPT},
                {"role": "user", "content": cleaned},
            ]
        )
        data = json.loads(_strip_fences(repaired))

    tasks = [Task(**t) for t in data.get("tasks", [])]
    return ParseResponse(tasks=tasks, mode="ai")


def _is_quota_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "insufficient_quota" in msg or "429" in msg


# ---------------------------------------------------------------------------
# DB helpers shared by routes
# ---------------------------------------------------------------------------

def _insert_tasks(tasks: List[Task], conn: sqlite3.Connection) -> List[int]:
    now = now_iso()
    ids: List[int] = []
    for task in tasks:
        cursor = conn.execute(
            """
            INSERT INTO tasks
                (title, priority, estimate_min, deadline, tags_json, next_step, done, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (
                task.title,
                task.priority,
                task.estimate_min,
                task.deadline,
                encode_tags(task.tags),
                task.next_step,
                now,
                now,
            ),
        )
        ids.append(cursor.lastrowid)
    return ids


def _fetch_task_or_404(task_id: int, conn: sqlite3.Connection) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return row


# ---------------------------------------------------------------------------
# Routes - System
# ---------------------------------------------------------------------------

@app.get("/", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    try:
        with db_connection() as conn:
            conn.execute("SELECT 1")
        db_status = "ok"
    except Exception:
        db_status = "error"

    return HealthResponse(
        status="running",
        version=app.version,
        db=db_status,
        openai=openai_client is not None,
    )


# ---------------------------------------------------------------------------
# Routes - Parser
# ---------------------------------------------------------------------------

@app.post("/parse_fast", response_model=ParseResponse, tags=["Parser"])
def parse_fast(req: ParseRequest) -> ParseResponse:
    """Rule-based parsing. Always available, no API key required."""
    return fast_parse_response(req.text, mode="fast")


@app.post("/parse_ai", response_model=ParseResponse, tags=["Parser"])
def parse_ai_route(req: ParseRequest) -> ParseResponse:
    """AI-powered parsing. Requires a valid OpenAI API key."""
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured")
    try:
        return ai_parse(req.text)
    except Exception as exc:
        log.error("AI parse error: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/parse", response_model=ParseResponse, tags=["Parser"])
def parse_auto(req: ParseRequest) -> ParseResponse:
    """Parse using AI when available; fall back to rule-based on quota errors."""
    if openai_client is None:
        return fast_parse_response(req.text, mode="auto-fast")

    try:
        result = ai_parse(req.text)
        result.mode = "auto-ai"
        return result
    except Exception as exc:
        if _is_quota_error(exc):
            log.warning("OpenAI quota exceeded, using fallback parser")
            return fast_parse_response(req.text, mode="auto-fast")
        log.error("AI parse error: %s", exc)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/smart_add", response_model=SaveTasksResponse, tags=["Parser"])
def smart_add(req: ParseRequest) -> SaveTasksResponse:
    """Parse text and immediately persist all detected tasks."""
    parsed = fast_parse_response(req.text, mode="smart-add")
    with db_connection() as conn:
        ids = _insert_tasks(parsed.tasks, conn)
    return SaveTasksResponse(inserted_ids=ids)


@app.post("/smart_add_ai", response_model=SaveTasksResponse, tags=["Parser"])
def smart_add_ai(req: ParseRequest) -> SaveTasksResponse:
    """Parse text with AI and immediately persist all detected tasks."""
    if openai_client is None:
        raise HTTPException(status_code=503, detail="OpenAI client not configured")
    try:
        parsed = ai_parse(req.text)
    except Exception as exc:
        if _is_quota_error(exc):
            log.warning("Quota exceeded, falling back to rule-based parser")
            parsed = fast_parse_response(req.text, mode="auto-fast")
        else:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    with db_connection() as conn:
        ids = _insert_tasks(parsed.tasks, conn)
    return SaveTasksResponse(inserted_ids=ids)


# ---------------------------------------------------------------------------
# Routes - Tasks CRUD
# ---------------------------------------------------------------------------

@app.post("/tasks", response_model=SaveTasksResponse, tags=["Tasks"])
def create_tasks(req: SaveTasksRequest) -> SaveTasksResponse:
    """Persist one or more pre-structured tasks."""
    if not req.tasks:
        return SaveTasksResponse(inserted_ids=[])
    with db_connection() as conn:
        ids = _insert_tasks(req.tasks, conn)
    return SaveTasksResponse(inserted_ids=ids)


@app.get("/tasks", response_model=List[TaskRow], tags=["Tasks"])
def list_tasks(
    done: Optional[bool] = Query(default=None, description="Filter by completion status"),
    q: Optional[str] = Query(default=None, description="Fulltext search in title and next_step"),
    tag: Optional[str] = Query(default=None, description="Filter by tag"),
    priority: Optional[Priority] = Query(default=None),
    sort: SortMode = Query(default="newest"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> List[TaskRow]:
    where: List[str] = []
    params: List[Any] = []

    if done is not None:
        where.append("done = ?")
        params.append(1 if done else 0)
    if priority is not None:
        where.append("priority = ?")
        params.append(priority)
    if q:
        where.append("(title LIKE ? OR next_step LIKE ?)")
        like = f"%{q.strip()}%"
        params.extend([like, like])
    if tag:
        where.append("tags_json LIKE ?")
        params.append(f'%"{tag.strip()}"%')

    sql = "SELECT * FROM tasks"
    if where:
        sql += " WHERE " + " AND ".join(where)

    sort_map: Dict[str, str] = {
        "newest":   "created_at DESC",
        "oldest":   "created_at ASC",
        "estimate": f"estimate_min ASC, created_at DESC",
        "deadline": "CASE WHEN deadline IS NULL THEN 1 ELSE 0 END, deadline ASC, created_at DESC",
        "prio":     f"{PRIORITY_SQL_CASE}, estimate_min ASC, created_at DESC",
    }
    sql += f" ORDER BY {sort_map.get(sort, 'created_at DESC')}"
    sql += " LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with db_connection() as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()

    return [row_to_task_row(r) for r in rows]


@app.get("/tasks/{task_id}", response_model=TaskRow, tags=["Tasks"])
def get_task(task_id: int) -> TaskRow:
    with db_connection() as conn:
        row = _fetch_task_or_404(task_id, conn)
    return row_to_task_row(row)


@app.patch("/tasks/{task_id}", response_model=TaskRow, tags=["Tasks"])
def update_task(task_id: int, req: UpdateTaskRequest) -> TaskRow:
    fields: List[str] = []
    values: List[Any] = []

    if req.title is not None:
        fields.append("title = ?")
        values.append(req.title)
    if req.priority is not None:
        fields.append("priority = ?")
        values.append(req.priority)
    if req.estimate_min is not None:
        fields.append("estimate_min = ?")
        values.append(req.estimate_min)
    if req.deadline is not None:
        fields.append("deadline = ?")
        values.append(req.deadline)
    if req.tags is not None:
        fields.append("tags_json = ?")
        values.append(encode_tags(req.tags))
    if req.next_step is not None:
        fields.append("next_step = ?")
        values.append(req.next_step)

    if not fields:
        raise HTTPException(status_code=400, detail="No fields provided for update")

    fields.append("updated_at = ?")
    values.append(now_iso())
    values.append(task_id)

    with db_connection() as conn:
        _fetch_task_or_404(task_id, conn)
        conn.execute(
            f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?",
            tuple(values),
        )
        row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()

    return row_to_task_row(row)


@app.patch("/tasks/{task_id}/done", response_model=TaskRow, tags=["Tasks"])
def toggle_done(task_id: int) -> TaskRow:
    with db_connection() as conn:
        row = _fetch_task_or_404(task_id, conn)
        new_done = 0 if row["done"] else 1
        done_at = now_iso() if new_done else None
        conn.execute(
            "UPDATE tasks SET done = ?, done_at = ?, updated_at = ? WHERE id = ?",
            (new_done, done_at, now_iso(), task_id),
        )
        updated = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    return row_to_task_row(updated)


@app.delete("/tasks/{task_id}", tags=["Tasks"])
def delete_task(task_id: int) -> Dict[str, Any]:
    with db_connection() as conn:
        result = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return {"deleted": True, "id": task_id}


@app.post("/tasks/bulk/done", tags=["Tasks"])
def bulk_mark_done(req: BulkActionRequest) -> Dict[str, Any]:
    """Mark multiple tasks as done in one request."""
    now = now_iso()
    placeholders = ",".join("?" * len(req.ids))
    with db_connection() as conn:
        result = conn.execute(
            f"UPDATE tasks SET done = 1, done_at = ?, updated_at = ? WHERE id IN ({placeholders})",
            (now, now, *req.ids),
        )
    return {"updated": result.rowcount, "ids": req.ids}


@app.post("/tasks/bulk/delete", tags=["Tasks"])
def bulk_delete(req: BulkActionRequest) -> Dict[str, Any]:
    """Delete multiple tasks in one request."""
    placeholders = ",".join("?" * len(req.ids))
    with db_connection() as conn:
        result = conn.execute(
            f"DELETE FROM tasks WHERE id IN ({placeholders})",
            tuple(req.ids),
        )
    return {"deleted": result.rowcount, "ids": req.ids}


# ---------------------------------------------------------------------------
# Routes - Notes
# ---------------------------------------------------------------------------

@app.post("/tasks/{task_id}/notes", response_model=NoteRow, tags=["Notes"])
def add_note(task_id: int, req: AddNoteRequest) -> NoteRow:
    """Attach a free-text note to a task."""
    now = now_iso()
    with db_connection() as conn:
        _fetch_task_or_404(task_id, conn)
        cursor = conn.execute(
            "INSERT INTO task_notes (task_id, body, created_at) VALUES (?, ?, ?)",
            (task_id, req.body, now),
        )
        note_id = cursor.lastrowid
    return NoteRow(id=note_id, task_id=task_id, body=req.body, created_at=now)


@app.get("/tasks/{task_id}/notes", response_model=List[NoteRow], tags=["Notes"])
def list_notes(task_id: int) -> List[NoteRow]:
    with db_connection() as conn:
        _fetch_task_or_404(task_id, conn)
        rows = conn.execute(
            "SELECT * FROM task_notes WHERE task_id = ? ORDER BY created_at ASC",
            (task_id,),
        ).fetchall()
    return [NoteRow(id=r["id"], task_id=r["task_id"], body=r["body"], created_at=r["created_at"]) for r in rows]


@app.delete("/tasks/{task_id}/notes/{note_id}", tags=["Notes"])
def delete_note(task_id: int, note_id: int) -> Dict[str, Any]:
    with db_connection() as conn:
        _fetch_task_or_404(task_id, conn)
        result = conn.execute(
            "DELETE FROM task_notes WHERE id = ? AND task_id = ?",
            (note_id, task_id),
        )
        if result.rowcount == 0:
            raise HTTPException(status_code=404, detail=f"Note {note_id} not found")
    return {"deleted": True, "note_id": note_id}


# ---------------------------------------------------------------------------
# Routes - Today & Stats
# ---------------------------------------------------------------------------

@app.get("/today", response_model=TodayResponse, tags=["Dashboard"])
def today_tasks(
    include_done: bool = Query(default=False, description="Include completed tasks"),
) -> TodayResponse:
    d = today_iso()
    done_clause = "" if include_done else "AND done = 0"

    with db_connection() as conn:
        rows_today = conn.execute(
            f"""
            SELECT * FROM tasks
            WHERE deadline = ?
              {done_clause}
            ORDER BY {PRIORITY_SQL_CASE}, estimate_min ASC, created_at DESC
            """,
            (d,),
        ).fetchall()

        rows_overdue = conn.execute(
            f"""
            SELECT * FROM tasks
            WHERE deadline IS NOT NULL
              AND deadline < ?
              {done_clause}
            ORDER BY deadline ASC, {PRIORITY_SQL_CASE}, estimate_min ASC
            """,
            (d,),
        ).fetchall()

    today_list = [row_to_task_row(r) for r in rows_today]
    overdue_list = [row_to_task_row(r) for r in rows_overdue]

    return TodayResponse(
        today=today_list,
        overdue=overdue_list,
        total_estimated_minutes_today=sum(t.estimate_min for t in today_list),
    )


@app.get("/stats", response_model=StatsResponse, tags=["Dashboard"])
def stats() -> StatsResponse:
    with db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        done_cnt = conn.execute("SELECT COUNT(*) FROM tasks WHERE done = 1").fetchone()[0]

        by_priority: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
        for row in conn.execute("SELECT priority, COUNT(*) AS c FROM tasks GROUP BY priority").fetchall():
            by_priority[row["priority"]] = row["c"]

        avg_row = conn.execute("SELECT AVG(estimate_min) FROM tasks").fetchone()
        avg_estimate = round(float(avg_row[0] or 0), 1)

        tag_counts: Dict[str, int] = {}
        for row in conn.execute("SELECT tags_json FROM tasks").fetchall():
            for tag in decode_tags(row["tags_json"]):
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

    top_tags = [
        {"tag": k, "count": v}
        for k, v in sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
    ]
    completion_rate = round(done_cnt / total * 100, 1) if total else 0.0

    return StatsResponse(
        total=total,
        done=done_cnt,
        pending=total - done_cnt,
        completion_rate=completion_rate,
        by_priority=by_priority,
        top_tags=top_tags,
        avg_estimate_min=avg_estimate,
    )