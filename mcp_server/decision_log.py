"""decision_log.py — the record of what Friday actually did.

Every tool call: its name, its arguments, what came back, whether it failed,
how long it took, and when. This is the thing you read at 1am when the robot
did something inexplicable, and later the thing the dashboard shows and that
Friday can learn from.

SQLite rather than a flat file so it can be queried by time or by tool without
parsing the whole history, and so concurrent writes from the movement thread
and the tool thread do not interleave into a corrupt line.

Logging must never be the reason a tool fails. Every function here swallows
its own errors — a robot that stops moving because its logfile is unwritable
is a worse robot than one with a gap in its log.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any

import config

_lock = threading.Lock()
_ready = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    at          REAL    NOT NULL,      -- unix time
    at_iso      TEXT    NOT NULL,      -- human readable, for reading by eye
    tool        TEXT    NOT NULL,
    arguments   TEXT    NOT NULL,      -- json
    result      TEXT,
    ok          INTEGER NOT NULL,      -- 1 success, 0 failure
    error       TEXT,
    duration_ms INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS decisions_at   ON decisions(at);
CREATE INDEX IF NOT EXISTS decisions_tool ON decisions(tool);
"""


def _connect() -> sqlite3.Connection:
    config.DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.DECISION_LOG, timeout=5.0)
    # WAL so a reader (the dashboard) never blocks the robot's writer.
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _ensure() -> None:
    global _ready
    if _ready:
        return
    with _connect() as conn:
        conn.executescript(SCHEMA)
    _ready = True


def record(tool: str, arguments: dict[str, Any], result: str | None,
           ok: bool, error: str | None, duration_ms: int) -> None:
    """Append one tool call. Never raises."""
    try:
        with _lock:
            _ensure()
            now = time.time()
            with _connect() as conn:
                conn.execute(
                    "INSERT INTO decisions (at, at_iso, tool, arguments, "
                    "result, ok, error, duration_ms) VALUES (?,?,?,?,?,?,?,?)",
                    (now,
                     time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                     tool,
                     json.dumps(arguments, default=str, ensure_ascii=False),
                     result, 1 if ok else 0, error, duration_ms),
                )
    except Exception:       # noqa: BLE001 - logging must never break a tool
        pass


def recent(limit: int = 40, tool: str | None = None) -> list[dict[str, Any]]:
    """The newest calls, newest first. Never raises; [] if unreadable."""
    try:
        with _lock:
            _ensure()
            with _connect() as conn:
                conn.row_factory = sqlite3.Row
                if tool:
                    rows = conn.execute(
                        "SELECT * FROM decisions WHERE tool = ? "
                        "ORDER BY id DESC LIMIT ?", (tool, limit)).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM decisions ORDER BY id DESC LIMIT ?",
                        (limit,)).fetchall()
                return [dict(r) for r in rows]
    except Exception:       # noqa: BLE001
        return []


def main() -> int:
    """Print the tail of the log, for reading by eye."""
    import argparse
    ap = argparse.ArgumentParser(description="Friday's decision log.")
    ap.add_argument("-n", "--limit", type=int, default=30)
    ap.add_argument("--tool", help="only this tool")
    args = ap.parse_args()

    rows = recent(args.limit, args.tool)
    if not rows:
        print(f"nothing logged yet ({config.DECISION_LOG})")
        return 0
    for row in reversed(rows):
        mark = "ok " if row["ok"] else "ERR"
        detail = row["error"] if not row["ok"] else (row["result"] or "")
        print(f'{row["at_iso"]}  {mark}  {row["tool"]:16} '
              f'{row["duration_ms"]:>6}ms  {row["arguments"]}')
        if detail:
            print(f'{"":21}   -> {detail.splitlines()[0][:120]}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
