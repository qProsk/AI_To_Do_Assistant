from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import re
import os
import json
import traceback
import sqlite3
from datetime import datetime

from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# =========================
# ENV
# =========================
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print("API KEY loaded:", bool(API_KEY))
print("MODEL:", MODEL)

client = None
if API_KEY and OpenAI is not None:
    client = OpenAI(api_key=API_KEY)


# =========================
# FASTAPI APP (Swagger tags order)
# =========================
openapi_tags = [
    {"name": "System", "description": "Healthcheck a systémové endpointy"},
    {"name": "Parser", "description": "Prevod textu na štruktúrované tasky"},
    {"name": "Tasks", "description": "CRUD operácie nad tasks v databáze"},
]

app = FastAPI(openapi_tags=openapi_tags)


# =========================
# DB
# =========================
DB_PATH = "todo.db"


def db_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = db_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            priority TEXT NOT NULL,
            estimate_min INTEGER NOT NULL,
            deadline TEXT,
            tags_json TEXT NOT NULL,
            next_step TEXT NOT NULL,
            done INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# =========================
# Pydantic MODELS
# =========================
class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1)


class Task(BaseModel):
    title: str
    priority: str  # "high" | "medium" | "low"
    estimate_min: int
    deadline: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    next_step: str


class ParseResponse(BaseModel):
    tasks: List[Task]
    mode: str  # "fast" | "ai" | "auto-fast" | "auto-ai" | "smart-add"


class TaskRow(Task):
    id: int
    done: bool
    created_at: str


class SaveTasksRequest(BaseModel):
    tasks: List[Task] = Field(default_factory=list)


class SaveTasksResponse(BaseModel):
    inserted_ids: List[int]


class UpdateTaskRequest(BaseModel):
    title: Optional[str] = None
    priority: Optional[str] = None
    estimate_min: Optional[int] = None
    deadline: Optional[str] = None
    tags: Optional[List[str]] = None
    next_step: Optional[str] = None


# =========================
# FALLBACK PARSER (FAST)
# =========================
BULLET_PREFIX = re.compile(r"^\s*([-*•]|(\d+[\.\)]))\s+")
MULTISPACE = re.compile(r"\s+")

URGENT_WORDS = {"urgent", "hneď", "asap", "dnes", "zajtra", "do piatku", "do pondelka", "termín", "deadline"}
WORK_WORDS = {"faktúra", "invoice", "klient", "email", "mail", "meeting", "call", "zmluva", "ponuka"}
FINANCE_WORDS = {"faktúra", "platba", "zaplatiť", "bank", "účet", "dph", "daň"}
HOME_WORDS = {"upratať", "upratovanie", "umývanie", "vysávať", "riad", "dom", "izba"}
SHOP_WORDS = {"kúpiť", "nakúpiť", "shop", "obchod", "objednať"}


def split_to_tasks(text: str) -> List[str]:
    lines = text.splitlines()
    out: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        line = BULLET_PREFIX.sub("", line).strip()
        line = MULTISPACE.sub(" ", line)

        if len(line) < 3:
            continue

        out.append(line)

    # fallback: keď nič nevyjde z riadkov, rozdeľ podľa ";"
    if not out:
        parts = [p.strip() for p in text.split(";") if p.strip()]
        out = [MULTISPACE.sub(" ", p) for p in parts if len(p) >= 3]

    return out


def smart_tags(title: str) -> List[str]:
    t = title.lower()
    tags: List[str] = []

    if any(w in t for w in WORK_WORDS):
        tags.append("práca")
    if any(w in t for w in FINANCE_WORDS):
        tags.append("financie")
    if any(w in t for w in HOME_WORDS):
        tags.append("domácnosť")
    if any(w in t for w in SHOP_WORDS):
        tags.append("nákup")

    if not tags:
        tags.append("osobné")

    return tags[:5]


def smart_priority(title: str) -> str:
    t = title.lower()
    if any(w in t for w in URGENT_WORDS):
        return "high"
    if any(w in t for w in WORK_WORDS) or any(w in t for w in FINANCE_WORDS):
        return "medium"
    return "low"


def smart_estimate(title: str) -> int:
    t = title.lower()
    if any(x in t for x in ["zavolať", "call", "email", "mail", "poslať", "napísať"]):
        return 5
    if any(x in t for x in ["kúpiť", "nakúpiť", "objednať"]):
        return 10
    if any(x in t for x in ["upratať", "upratovanie", "vysávať", "umývanie", "riad"]):
        return 20
    return 15


def fallback_tasks(text: str) -> List[Task]:
    titles = split_to_tasks(text)
    tasks: List[Task] = []
    for t in titles:
        tasks.append(
            Task(
                title=t,
                priority=smart_priority(t),
                estimate_min=smart_estimate(t),
                deadline=None,
                tags=smart_tags(t),
                next_step=f"Urob prvý konkrétny krok: začni s '{t}'.",
            )
        )
    return tasks


def fallback_response(text: str, mode: str) -> ParseResponse:
    tasks = fallback_tasks(text)
    prio_order = {"high": 0, "medium": 1, "low": 2}
    tasks.sort(key=lambda x: (prio_order.get(x.priority, 9), x.estimate_min))
    return ParseResponse(tasks=tasks, mode=mode)


# =========================
# AI PARSER (optional)
# =========================
SYSTEM_PROMPT = """
Vráť VÝHRADNE validný JSON podľa schémy:

{
  "tasks": [
    {
      "title": "string",
      "priority": "high|medium|low",
      "estimate_min": integer,
      "deadline": "YYYY-MM-DD" alebo null,
      "tags": ["string", ...],
      "next_step": "string"
    }
  ]
}
"""

REPAIR_PROMPT = "Oprav tento obsah tak, aby bol validný JSON presne podľa schémy. Bez ďalšieho textu."


def call_model(messages) -> str:
    if client is None:
        raise RuntimeError("OpenAI client not initialized")

    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.2,
    )
    content = resp.choices[0].message.content
    if content is None or not content.strip():
        raise ValueError("Model returned empty content")
    return content.strip()


def ai_parse(text: str) -> ParseResponse:
    raw = call_model(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Text s úlohami:\n{text}"},
        ]
    )
    try:
        data = json.loads(raw)
    except Exception:
        fixed = call_model(
            [
                {"role": "system", "content": REPAIR_PROMPT},
                {"role": "user", "content": raw},
            ]
        )
        data = json.loads(fixed)

    tasks = [Task(**t) for t in data.get("tasks", [])]
    return ParseResponse(tasks=tasks, mode="ai")


def is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("insufficient_quota" in msg) or ("error code: 429" in msg) or ("429" in msg)


# =========================
# ENDPOINTS: SYSTEM
# =========================
@app.get("/", tags=["System"])
def root():
    return {"status": "running"}


# =========================
# ENDPOINTS: PARSER
# =========================
@app.post("/parse_fast", response_model=ParseResponse, tags=["Parser"])
def parse_fast(req: ParseRequest):
    return fallback_response(req.text, mode="fast")


@app.post("/parse_ai", response_model=ParseResponse, tags=["Parser"])
def parse_ai(req: ParseRequest):
    if client is None:
        raise HTTPException(status_code=400, detail="OPENAI client not initialized")
    try:
        return ai_parse(req.text)
    except Exception as e:
        print("AI ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/parse", response_model=ParseResponse, tags=["Parser"])
def parse_auto(req: ParseRequest):
    # Auto: ak nemáš client alebo dôjde quota, padne na fast
    if client is None:
        return fallback_response(req.text, mode="auto-fast")

    try:
        res = ai_parse(req.text)
        res.mode = "auto-ai"
        return res
    except Exception as e:
        if is_quota_error(e):
            return fallback_response(req.text, mode="auto-fast")
        print("AI ERROR:", repr(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/smart_add", response_model=SaveTasksResponse, tags=["Parser"])
def smart_add(req: ParseRequest):
    """
    1) Parse text -> tasks (fast parser)
    2) Save tasks -> DB
    """
    parsed = fallback_response(req.text, mode="smart-add")

    if not parsed.tasks:
        return SaveTasksResponse(inserted_ids=[])

    conn = db_conn()
    cur = conn.cursor()

    inserted: List[int] = []
    now = datetime.utcnow().isoformat()

    for t in parsed.tasks:
        cur.execute(
            """
            INSERT INTO tasks (title, priority, estimate_min, deadline, tags_json, next_step, done, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (t.title, t.priority, t.estimate_min, t.deadline, json.dumps(t.tags), t.next_step, now),
        )
        inserted.append(int(cur.lastrowid))

    conn.commit()
    conn.close()

    return SaveTasksResponse(inserted_ids=inserted)


# =========================
# ENDPOINTS: TASKS (DB)
# =========================
@app.get("/tasks", response_model=List[TaskRow], tags=["Tasks"])
def list_tasks(done: Optional[bool] = None):
    conn = db_conn()
    cur = conn.cursor()

    if done is None:
        cur.execute("SELECT * FROM tasks ORDER BY done ASC, created_at DESC")
    else:
        cur.execute("SELECT * FROM tasks WHERE done = ? ORDER BY created_at DESC", (1 if done else 0,))

    rows = cur.fetchall()
    conn.close()

    out: List[TaskRow] = []
    for r in rows:
        out.append(
            TaskRow(
                id=r["id"],
                title=r["title"],
                priority=r["priority"],
                estimate_min=r["estimate_min"],
                deadline=r["deadline"],
                tags=json.loads(r["tags_json"]) if r["tags_json"] else [],
                next_step=r["next_step"],
                done=bool(r["done"]),
                created_at=r["created_at"],
            )
        )
    return out


@app.get("/tasks/{task_id}", response_model=TaskRow, tags=["Tasks"])
def get_task(task_id: int):
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return TaskRow(
        id=row["id"],
        title=row["title"],
        priority=row["priority"],
        estimate_min=row["estimate_min"],
        deadline=row["deadline"],
        tags=json.loads(row["tags_json"]) if row["tags_json"] else [],
        next_step=row["next_step"],
        done=bool(row["done"]),
        created_at=row["created_at"],
    )


@app.post("/tasks", response_model=SaveTasksResponse, tags=["Tasks"])
def save_tasks(req: SaveTasksRequest):
    if not req.tasks:
        return SaveTasksResponse(inserted_ids=[])

    conn = db_conn()
    cur = conn.cursor()

    inserted: List[int] = []
    now = datetime.utcnow().isoformat()

    for t in req.tasks:
        cur.execute(
            """
            INSERT INTO tasks (title, priority, estimate_min, deadline, tags_json, next_step, done, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (t.title, t.priority, t.estimate_min, t.deadline, json.dumps(t.tags), t.next_step, now),
        )
        inserted.append(int(cur.lastrowid))

    conn.commit()
    conn.close()

    return SaveTasksResponse(inserted_ids=inserted)


@app.patch("/tasks/{task_id}", response_model=TaskRow, tags=["Tasks"])
def update_task(task_id: int, req: UpdateTaskRequest):
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

    fields = []
    values = []

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
        values.append(json.dumps(req.tags))
    if req.next_step is not None:
        fields.append("next_step = ?")
        values.append(req.next_step)

    if not fields:
        conn.close()
        raise HTTPException(status_code=400, detail="No fields to update")

    values.append(task_id)
    sql = f"UPDATE tasks SET {', '.join(fields)} WHERE id = ?"
    cur.execute(sql, tuple(values))
    conn.commit()

    cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row2 = cur.fetchone()
    conn.close()

    return TaskRow(
        id=row2["id"],
        title=row2["title"],
        priority=row2["priority"],
        estimate_min=row2["estimate_min"],
        deadline=row2["deadline"],
        tags=json.loads(row2["tags_json"]) if row2["tags_json"] else [],
        next_step=row2["next_step"],
        done=bool(row2["done"]),
        created_at=row2["created_at"],
    )


@app.patch("/tasks/{task_id}/done", response_model=TaskRow, tags=["Tasks"])
def toggle_done(task_id: int):
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

    new_done = 0 if row["done"] else 1
    cur.execute("UPDATE tasks SET done = ? WHERE id = ?", (new_done, task_id))
    conn.commit()

    cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row2 = cur.fetchone()
    conn.close()

    return TaskRow(
        id=row2["id"],
        title=row2["title"],
        priority=row2["priority"],
        estimate_min=row2["estimate_min"],
        deadline=row2["deadline"],
        tags=json.loads(row2["tags_json"]) if row2["tags_json"] else [],
        next_step=row2["next_step"],
        done=bool(row2["done"]),
        created_at=row2["created_at"],
    )


@app.delete("/tasks/{task_id}", tags=["Tasks"])
def delete_task(task_id: int):
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    deleted = cur.rowcount
    conn.close()

    if deleted == 0:
        raise HTTPException(status_code=404, detail="Task not found")

    return {"deleted": True, "id": task_id}