"""run_summary.py — File 19 step 7: what a run amounted to, said once she is back.

    s = run_summary.build(kind="explore", started=t0, ended=t1, area_gained_m2=4.2,
                          came_back=True)
    run_summary.record(s)       # bounded log, state/run_summaries.json
    run_summary.say(s)          # spoken + the proud face, if she came back

TWO KINDS OF RETURN, because there are two ways home in this robot:
  * "explore" -- the RL run (rl/, started by explore_and_return) drives itself
    back ("I am back where I started."). Its end is read from
    state/explore_mode.json and its run log -- nothing in rl/ is touched.
    Area gained is measured on the explore map before/after (rl_map.prev.npz
    vs rl_map.npz), which is honest WITHIN that one map.
  * "return" -- file 14's navigation arriving at home / where she set off
    from, read from state/nav_log.jsonl lines as they are written.

BOUNDED like file 09's reflection log: the last RUN_SUMMARY_KEEP runs are kept
whole; anything older is folded into one running total, so the file never
grows without limit.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("friday.run_summary")


def _file() -> Path:
    return Path(config.RUN_SUMMARY_FILE)


def read() -> dict:
    try:
        data = json.loads(_file().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def build(kind: str, started: float, ended: float,
          area_gained_m2: float | None = None, came_back: bool | None = None,
          destination: str = "", driven_m: float | None = None) -> dict:
    """Counts what happened in [started, ended] from the records."""
    import environments
    env = environments.active() or {}
    db = environments._db()
    pending_new = db.execute(
        "SELECT COUNT(*) FROM pending_landmarks WHERE env_id=? AND "
        "created_at BETWEEN ? AND ?", (env.get("env_id"), started, ended)
    ).fetchone()[0]
    named_new = db.execute(
        "SELECT COUNT(*) FROM pending_landmarks WHERE env_id=? AND "
        "named_at BETWEEN ? AND ?", (env.get("env_id"), started, ended)
    ).fetchone()[0]
    obstacles = sum(1 for items in (env.get("obstacle_log") or {}).values()
                    for e in items if started <= float(e.get("at", 0)) <= ended)
    return {"kind": kind, "env": env.get("name"), "started": started,
            "ended": ended, "minutes": round(max(0.0, ended - started) / 60, 1),
            "area_gained_m2": (round(area_gained_m2, 1)
                               if area_gained_m2 is not None else None),
            "came_back": came_back, "destination": destination,
            "driven_m": driven_m, "pending_new": int(pending_new),
            "named_new": int(named_new), "obstacles": int(obstacles),
            "waiting_names": len(environments.pending())}


def line(s: dict) -> str:
    """The spoken sentence. Short: this is said as she arrives."""
    parts = []
    if s["kind"] == "explore":
        parts.append(f"That was a {s['minutes']:.0f} minute run" if
                     s["minutes"] >= 1 else "That was a short run")
        gained = s.get("area_gained_m2")
        if gained is not None and gained >= 0.5:
            parts[-1] += f", and I mapped about {gained:.0f} new square metres"
        parts[-1] += "."
    else:
        dest = s.get("destination") or "home"
        parts.append(f"Back at {dest} after {s['minutes']:.0f} minutes."
                     if s["minutes"] >= 1 else f"Back at {dest}.")
    if s["named_new"]:
        parts.append(f"{s['named_new']} new place"
                     f"{'s' if s['named_new'] != 1 else ''} named.")
    if s["waiting_names"]:
        parts.append(f"{s['waiting_names']} area"
                     f"{'s are' if s['waiting_names'] != 1 else ' is'} waiting "
                     f"for a name on the dashboard.")
    if s["obstacles"]:
        parts.append(f"I noted {s['obstacles']} new obstacle"
                     f"{'s' if s['obstacles'] != 1 else ''} near named places.")
    return " ".join(parts)


def record(s: dict) -> None:
    """Append, keeping the last RUN_SUMMARY_KEEP; fold the rest into 'older'."""
    data = read()
    runs = list(data.get("runs") or []) + [{**s, "line": line(s)}]
    older = dict(data.get("older") or {"runs": 0, "minutes": 0.0,
                                       "area_m2": 0.0})
    keep = int(config.RUN_SUMMARY_KEEP)
    while len(runs) > keep:
        gone = runs.pop(0)
        older["runs"] += 1
        older["minutes"] = round(older["minutes"] + float(gone.get("minutes") or 0), 1)
        older["area_m2"] = round(older["area_m2"]
                                 + float(gone.get("area_gained_m2") or 0), 1)
    path = _file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps({"runs": runs, "older": older}))
    tmp.replace(path)


def say(s: dict) -> None:
    """Spoken through the MCP process's own voice queue (nav_voice, which
    goes through tts.speak), with the proud face. Never raises; never blocks
    the caller -- the voice queue speaks on its own thread."""
    if not s.get("came_back"):
        return                               # she already said she could not
    try:
        from mcp_server.tools.expression import set_face
        set_face("proud")
    except Exception as exc:                        # noqa: BLE001
        log.debug("no proud face: %s", exc)
    try:
        import nav_voice
        nav_voice.VOICE.say_text(line(s))
    except Exception as exc:                        # noqa: BLE001
        log.info("summary not spoken: %s", exc)


def finish(s: dict) -> dict:
    record(s)
    say(s)
    log.info("run summary: %s", line(s))
    return s


def panel() -> dict[str, Any]:
    data = read()
    return {"runs": list(reversed(data.get("runs") or []))[:10],
            "older": data.get("older")}
