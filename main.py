from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict, Any, Tuple
import re
import os
import json
import traceback
import sqlite3
from datetime import datetime, date
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# ---------- ENV ----------
load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

print("API KEY loaded:", bool(API_KEY))
print("MODEL:", MODEL)

client = None
if API_KEY and OpenAI is not None:
    client = OpenAI(api_key=API_KEY)


# ---------- APP ----------
app = FastAPI(
    title="AI To-Do Assistant",
    version="1.0.0",
    description="Prevod textu na tasky + jednoduché CRUD nad SQLite",
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

@app.get("/app", include_in_schema=False)
def app_page():
    return FileResponse("frontend/index.html")

# ---------- DB ----------
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


# ---------- HELPERS ----------
def dumps_tags(tags: List[str]) -> str:
    # Dôležité: nech v DB ostane diakritika a tag filter funguje
    return json.dumps(tags, ensure_ascii=False)


def loads_tags(s: str) -> List[str]:
    try:
        return json.loads(s) if s else []
    except Exception:
        return []


def row_to_taskrow(r: sqlite3.Row) -> "TaskRow":
    return TaskRow(
        id=r["id"],
        title=r["title"],
        priority=r["priority"],
        estimate_min=r["estimate_min"],
        deadline=r["deadline"],
        tags=loads_tags(r["tags_json"]),
        next_step=r["next_step"],
        done=bool(r["done"]),
        created_at=r["created_at"],
    )


def today_yyyy_mm_dd() -> str:
    return date.today().isoformat()


# ---------- MODELS ----------
Priority = Literal["high", "medium", "low"]
SortMode = Literal["newest", "oldest", "prio", "estimate"]

class ParseRequest(BaseModel):
    text: str = Field(..., min_length=1)


class Task(BaseModel):
    title: str
    priority: Priority
    estimate_min: int
    deadline: Optional[str] = None  # YYYY-MM-DD alebo None
    tags: List[str] = Field(default_factory=list)
    next_step: str


class ParseResponse(BaseModel):
    tasks: List[Task]
    mode: str  # "fast" | "ai" | "auto-fast" | "auto-ai" | ...


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
    priority: Optional[Priority] = None
    estimate_min: Optional[int] = None
    deadline: Optional[str] = None
    tags: Optional[List[str]] = None
    next_step: Optional[str] = None


class TodayResponse(BaseModel):
    today: List[TaskRow]
    overdue: List[TaskRow]
    total_estimated_minutes_today: int


class StatsResponse(BaseModel):
    total: int
    done: int
    pending: int
    by_priority: Dict[str, int]
    top_tags: List[Dict[str, Any]]  # [{"tag": "...", "count": 3}, ...]


# ---------- FALLBACK PARSER ----------
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

    # unikátne + limit
    uniq: List[str] = []
    for x in tags:
        if x not in uniq:
            uniq.append(x)
    return uniq[:5]


def smart_priority(title: str) -> Priority:
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


# ---------- AI (voliteľné) ----------
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


# ---------- ROUTES ----------
@app.get("/", tags=["System"])
def root():
    return {"status": "running"}


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
    # 1) parse text -> tasks (fast parser)
    parsed = fallback_response(req.text, mode="smart-add")

    # 2) save tasks -> DB
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
            (t.title, t.priority, t.estimate_min, t.deadline, dumps_tags(t.tags), t.next_step, now),
        )
        inserted.append(int(cur.lastrowid))

    conn.commit()
    conn.close()
    return SaveTasksResponse(inserted_ids=inserted)


# ---------- TASKS / CRUD ----------
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
            (t.title, t.priority, t.estimate_min, t.deadline, dumps_tags(t.tags), t.next_step, now),
        )
        inserted.append(int(cur.lastrowid))

    conn.commit()
    conn.close()
    return SaveTasksResponse(inserted_ids=inserted)


@app.get("/tasks", response_model=List[TaskRow], tags=["Tasks"])
def list_tasks(
    done: Optional[bool] = Query(default=None),
    q: Optional[str] = Query(default=None, description="Fulltext (LIKE) cez title + next_step"),
    tag: Optional[str] = Query(default=None, description="Filter podľa tagu"),
    priority: Optional[Priority] = Query(default=None),
    sort: SortMode = Query(default="newest", description="newest|oldest|prio|estimate"),
):
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
        # Keď ukladáme tags_json s ensure_ascii=False, stačí tento LIKE.
        # (Ak máš staré dáta s \u00e1, najjednoduchšie je zmazať todo.db počas vývoja.)
        where.append("tags_json LIKE ?")
        params.append(f'%"{tag.strip()}"%')

    sql = "SELECT * FROM tasks"
    if where:
        sql += " WHERE " + " AND ".join(where)

    if sort == "newest":
        sql += " ORDER BY created_at DESC"
    elif sort == "oldest":
        sql += " ORDER BY created_at ASC"
    elif sort == "estimate":
        sql += " ORDER BY estimate_min ASC, created_at DESC"
    elif sort == "prio":
        # high -> medium -> low
        sql += """
        ORDER BY
          CASE priority
            WHEN 'high' THEN 0
            WHEN 'medium' THEN 1
            WHEN 'low' THEN 2
            ELSE 9
          END,
          estimate_min ASC,
          created_at DESC
        """

    conn = db_conn()
    cur = conn.cursor()
    cur.execute(sql, tuple(params))
    rows = cur.fetchall()
    conn.close()

    return [row_to_taskrow(r) for r in rows]


@app.get("/tasks/{task_id}", response_model=TaskRow, tags=["Tasks"])
def get_task(task_id: int):
    conn = db_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return row_to_taskrow(row)


@app.patch("/tasks/{task_id}", response_model=TaskRow, tags=["Tasks"])
def update_task(task_id: int, req: UpdateTaskRequest):
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
    row = cur.fetchone()
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="Task not found")

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
        values.append(dumps_tags(req.tags))
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

    return row_to_taskrow(row2)


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

    return row_to_taskrow(row2)


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


# ---------- TODAY + STATS ----------
@app.get("/today", response_model=TodayResponse, tags=["Tasks"])
def today_tasks(include_done: bool = Query(default=False, description="Ak true, ukáže aj dokončené úlohy")):
    d = today_yyyy_mm_dd()

    conn = db_conn()
    cur = conn.cursor()

    base_done_clause = "" if include_done else "AND done = 0"

    # today
    cur.execute(
        f"""
        SELECT * FROM tasks
        WHERE deadline = ?
        {base_done_clause}
        ORDER BY
          CASE priority
            WHEN 'high' THEN 0
            WHEN 'medium' THEN 1
            WHEN 'low' THEN 2
            ELSE 9
          END,
          estimate_min ASC,
          created_at DESC
        """,
        (d,),
    )
    rows_today = cur.fetchall()

    # overdue
    cur.execute(
        f"""
        SELECT * FROM tasks
        WHERE deadline IS NOT NULL
          AND deadline < ?
        {base_done_clause}
        ORDER BY deadline ASC,
          CASE priority
            WHEN 'high' THEN 0
            WHEN 'medium' THEN 1
            WHEN 'low' THEN 2
            ELSE 9
          END,
          estimate_min ASC
        """,
        (d,),
    )
    rows_overdue = cur.fetchall()

    conn.close()

    today_list = [row_to_taskrow(r) for r in rows_today]
    overdue_list = [row_to_taskrow(r) for r in rows_overdue]
    total_est = sum(t.estimate_min for t in today_list)

    return TodayResponse(today=today_list, overdue=overdue_list, total_estimated_minutes_today=total_est)


@app.get("/stats", response_model=StatsResponse, tags=["Tasks"])
def stats():
    conn = db_conn()
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) AS c FROM tasks")
    total = int(cur.fetchone()["c"])

    cur.execute("SELECT COUNT(*) AS c FROM tasks WHERE done = 1")
    done_cnt = int(cur.fetchone()["c"])

    pending = total - done_cnt

    cur.execute("SELECT priority, COUNT(*) AS c FROM tasks GROUP BY priority")
    by_priority: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}
    for r in cur.fetchall():
        by_priority[str(r["priority"])] = int(r["c"])

    # top tags: načítame tags_json a spočítame v Pythone (jednoduché a spoľahlivé)
    cur.execute("SELECT tags_json FROM tasks")
    tag_counts: Dict[str, int] = {}
    for r in cur.fetchall():
        tags = loads_tags(r["tags_json"])
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    conn.close()

    top = sorted(tag_counts.items(), key=lambda x: (-x[1], x[0]))[:10]
    top_tags = [{"tag": k, "count": v} for k, v in top]

    return StatsResponse(
        total=total,
        done=done_cnt,
        pending=pending,
        by_priority=by_priority,
        top_tags=top_tags,
    )