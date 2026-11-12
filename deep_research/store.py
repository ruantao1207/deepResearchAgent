"""运行记录持久化（SQLite）：任务元数据 + 事件流。

LangGraph 的执行状态由 AsyncSqliteSaver（checkpoints.db）保存，用于断点续跑；
这里保存的是 Web 界面需要的任务列表与事件历史（runs.db），服务重启后仍可查看。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    topic TEXT NOT NULL,
    options TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    run_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    data TEXT NOT NULL,
    PRIMARY KEY (run_id, seq)
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class RunStore:
    def __init__(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- runs ----
    def create_run(self, run_id: str, topic: str, options: dict[str, Any], status: str, created: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO runs(id, topic, options, status, created, updated) VALUES (?,?,?,?,?,?)",
                (run_id, topic, json.dumps(options, ensure_ascii=False), status, created, _now()),
            )
            self._conn.commit()

    def set_status(self, run_id: str, status: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE runs SET status=?, updated=? WHERE id=?", (status, _now(), run_id))
            self._conn.commit()

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, topic, options, status, created, updated FROM runs ORDER BY created DESC"
            ).fetchall()
        return [{**dict(r), "options": json.loads(r["options"])} for r in rows]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT id, topic, options, status, created, updated FROM runs WHERE id=?", (run_id,)
            ).fetchone()
        return {**dict(r), "options": json.loads(r["options"])} if r else None

    def delete_run(self, run_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM events WHERE run_id=?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE id=?", (run_id,))
            self._conn.commit()

    # ---- events ----
    def add_event(self, run_id: str, seq: int, event: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO events(run_id, seq, data) VALUES (?,?,?)",
                (run_id, seq, json.dumps(event, ensure_ascii=False)),
            )
            self._conn.execute("UPDATE runs SET updated=? WHERE id=?", (_now(), run_id))
            self._conn.commit()

    def load_events(self, run_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT data FROM events WHERE run_id=? ORDER BY seq", (run_id,)).fetchall()
        return [json.loads(r["data"]) for r in rows]
