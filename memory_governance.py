"""memory_governance.py — File 17: what kinds of memory exist, and who may delete one.

    memory_governance.TYPES                      the documented type convention
    memory_governance.remember(text, protected)  an explicit "remember that ..."
    memory_governance.start_forget("my sister")  -> the question she asks
    memory_governance.answer("yes")              -> what happened
    memory_governance.request_delete(id)         the dashboard's first click
    memory_governance.confirm_delete(token, id)  THE ONLY DELETE, voice or dashboard

NOTHING IS DELETED ON THE FIRST ASK. Every path -- "forget that ...", the
brain's forget tool, the dashboard's bin -- makes a PENDING request
(state/memory_pending.json, MEMORY_PENDING_TTL_S) and only confirm_delete
with that request's token removes a row. One match: "I found: ... Delete it?".
Several: "Which one: 1 ..., 2 ...?" and then the same yes/no. None: said so,
nothing touched.

PROTECTED. facts.protected (added here, default 0). "Always remember" / "never
forget" saves with it set, and the dashboard toggles it. File 09's reflection
never flags a protected row -- that one guard is the only edit to an earlier
file (reflection.py, see CLAUDE.md PART 8). Harsh can still forget a protected
memory himself: his manual control takes priority, and it still needs the yes.

The fuzzy match is memory.recall (the same TF-IDF the brain uses) plus a plain
word match, then ONE small Haiku call that picks which candidates the
description actually means. With no key or no internet the scores decide, and
anything doubtful becomes a question rather than a deletion.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("friday.memory_governance")

# THE TYPE CONVENTION (step 1) -- what is ALREADY in use, written down, not
# invented. `facts.kind` unless the store is named. New code that writes a
# memory picks from here or adds a line here; it does not make a new store.
TYPES: dict[str, str] = {
    "fact": "facts: the default. The brain's remember tool, reply-object "
            "memory notes (source brain-note), idle study (idle-study), the "
            "memory CLI, and 'Friday, remember that ...' (explicit).",
    "learned_preference": "facts: file 09 corrections — how he wants her to "
                          "behave. Reflection may FLAG these; only he removes.",
    "exploration_sighting": "facts: what she saw on an explore trip (one per "
                            "trip, trip_objects.py).",
    "calendar": "facts: notes about calendar events (gcal.py) AND the "
                "calendar_events table (day_calendar.py) — file 16.",
    "reminder": "the reminders table (reminders.py): reminders (due_at) and "
                "to-dos (no due_at) — file 16.",
    "environment": "the environments table (environments.py, file 19): one "
                   "row per place she lives in -- its map file, landmark "
                   "names, per-place obstacle log. Inactive ones keep their "
                   "map + places in state/environments/<env_id>/.",
    "research": "facts.kind: notes Friday put on the laptop (laptop_research."
                "py, file 20) -- topic, summary, date, notes path; source "
                "'laptop'.",
    "place": "the places table: named spots on the ACTIVE environment's map "
             "(file 14; file 19 swaps it when the environment changes).",
    "home": "the places table: the one spot called home.",
}

SCHEMA_PROTECTED = "ALTER TABLE facts ADD COLUMN protected INTEGER NOT NULL DEFAULT 0"

_ready = False


def _db():
    global _ready
    import memory
    connection = memory.db()
    if not _ready:
        with memory._lock:
            columns = {r[1] for r in connection.execute("PRAGMA table_info(facts)")}
            if "protected" not in columns:
                connection.execute(SCHEMA_PROTECTED)
                connection.commit()
                log.info("facts.protected added (default 0)")
        _ready = True
    return connection


def is_protected(fact_id: int) -> bool:
    try:
        row = _db().execute("SELECT protected FROM facts WHERE id=?",
                            (int(fact_id),)).fetchone()
    except Exception:                               # noqa: BLE001
        return False
    return bool(row and row["protected"])


def set_protected(fact_id: int, on: bool) -> bool:
    """Set the flag. True if the row exists."""
    import memory
    with memory._lock:
        cursor = _db().execute("UPDATE facts SET protected=? WHERE id=?",
                               (1 if on else 0, int(fact_id)))
        _db().commit()
    if cursor.rowcount:
        log.info("fact #%d %s", fact_id, "PROTECTED" if on else "unprotected")
    return cursor.rowcount > 0


def protected_ids() -> set[int]:
    return {r["id"] for r in _db().execute(
        "SELECT id FROM facts WHERE protected=1").fetchall()}


def entry(fact_id: int) -> dict | None:
    row = _db().execute("SELECT id, text, kind, source, created_at, protected "
                        "FROM facts WHERE id=?", (int(fact_id),)).fetchone()
    return dict(row) if row else None


# --- step 2: remember, plainly or for ever -------------------------------------

def remember(text: str, protected: bool = False,
             source: str = "explicit") -> dict:
    """Save it, whatever file 09's automatic capture would have done.

    A plain remember keeps an existing protected flag (re-stating a protected
    memory must not quietly unprotect it); an emphatic one sets it.
    """
    import memory
    text = (text or "").strip(" .")
    if not text:
        raise ValueError("there is nothing to remember")
    stored = memory.remember(text, kind="fact", source=source)
    if protected:
        set_protected(stored["id"], True)
    stored["protected"] = is_protected(stored["id"])
    return stored


# --- step 3: forget, only after a yes ------------------------------------------

def _pending_file() -> Path:
    return Path(getattr(config, "MEMORY_PENDING_FILE",
                        config.ROOT / "state" / "memory_pending.json"))


def pending() -> dict | None:
    """The open request, or None once it has expired or been answered."""
    try:
        data = json.loads(_pending_file().read_text())
    except (OSError, ValueError):
        return None
    ttl = float(getattr(config, "MEMORY_PENDING_TTL_S", 120.0))
    if not isinstance(data, dict) or time.time() - float(data.get("at") or 0) > ttl:
        return None
    return data


def _set_pending(data: dict | None) -> None:
    path = _pending_file()
    if data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps({**data, "at": time.time()}))
    tmp.replace(path)


def _words(text: str) -> set[str]:
    stop = {"the", "a", "an", "my", "i", "me", "that", "about", "is", "was",
            "what", "said", "to", "of", "and", "his", "her", "harsh", "s"}
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if w not in stop and len(w) > 1}


def candidates(description: str, limit: int = 6) -> list[dict]:
    """Rows that might be what he means, best first, with a 0-1 score."""
    import memory
    want = _words(description)
    scored: dict[int, dict] = {}
    try:
        for fact in memory.recall(description, limit=limit * 2):
            scored[fact.id] = {"id": fact.id, "text": fact.text,
                               "score": float(getattr(fact, "score", 0) or 0)}
    except Exception as exc:                        # noqa: BLE001
        log.info("recall unavailable for the match: %s", exc)
    for fact in memory.all_facts(limit=500):
        overlap = len(want & _words(fact.text)) / max(1, len(want))
        if overlap >= 0.5:
            best = scored.setdefault(fact.id, {"id": fact.id, "text": fact.text,
                                               "score": 0.0})
            best["score"] = max(best["score"], overlap)
    ranked = sorted(scored.values(), key=lambda r: -r["score"])
    return [r for r in ranked if r["score"] > 0][:limit]


def _haiku_pick(description: str, rows: list[dict]) -> list[int] | None:
    """Which of these rows does he mean? None when it cannot be asked."""
    if not rows:
        return []
    try:
        import brain
        import scene_describe
        key = scene_describe._api_key()
        if not key or brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
            return None
        import anthropic
        listing = "\n".join(f"{r['id']}: {r['text']}" for r in rows)
        reply = anthropic.Anthropic(api_key=key).messages.create(
            model=config.MODEL_FAST, max_tokens=40,
            system=("You match a spoken description to stored memories. Reply "
                    "with ONLY a JSON list of the ids that the description "
                    "refers to -- [] if none clearly does. Never include a "
                    "memory that is merely about a similar topic."),
            messages=[{"role": "user", "content":
                       f"Description: {description}\n\nMemories:\n{listing}"}])
        brain._record_spend(config.MODEL_FAST, reply.usage,
                            brain._cost_usd(config.MODEL_FAST, reply.usage))
        text = "".join(b.text for b in reply.content
                       if getattr(b, "type", "") == "text")
        ids = json.loads(re.search(r"\[.*?\]", text, re.S).group(0))
        allowed = {r["id"] for r in rows}
        return [int(i) for i in ids if int(i) in allowed]
    except Exception as exc:                        # noqa: BLE001
        log.info("memory match without the model: %s", exc)
        return None


def match(description: str) -> list[dict]:
    """The rows the description means: the model's pick, else the scores."""
    rows = candidates(description)
    picked = _haiku_pick(description, rows)
    if picked is None:
        floor = float(getattr(config, "MEMORY_MATCH_MIN_SCORE", 0.5))
        top = [r for r in rows if r["score"] >= floor]
        # A clear winner is one match; several close ones are a question.
        if len(top) > 1 and top[0]["score"] - top[1]["score"] >= 0.3:
            top = top[:1]
        return top[:5]
    order = {i: n for n, i in enumerate(picked)}
    return sorted([r for r in rows if r["id"] in order],
                  key=lambda r: order[r["id"]])[:5]


def _ask_to_confirm(row: dict, via: str) -> str:
    token = secrets.token_hex(8)
    _set_pending({"stage": "confirm", "token": token, "via": via,
                  "candidates": [{"id": row["id"], "text": row["text"]}],
                  "chosen": row["id"]})
    extra = " It's one you asked me never to forget." \
        if is_protected(row["id"]) else ""
    return f"I found this: {row['text']}.{extra} Should I delete it? Say yes or no."


def start_forget(description: str) -> str:
    """Step 3's first half: find, then ASK. Never deletes."""
    description = (description or "").strip(" .?")
    if not description:
        return "What should I forget?"
    found = match(description)
    if not found:
        _set_pending(None)
        return f"I don't have anything saved about {description}."
    if len(found) == 1:
        return _ask_to_confirm(found[0], via="voice")
    _set_pending({"stage": "choose", "token": secrets.token_hex(8),
                  "via": "voice",
                  "candidates": [{"id": r["id"], "text": r["text"]}
                                 for r in found]})
    listing = "; ".join(f"{n}, {r['text']}" for n, r in enumerate(found, 1))
    return (f"I found {len(found)} things: {listing}. Which one should I "
            f"forget? Say the number, or no.")


_ORDINALS = {"first": 1, "one": 1, "1": 1, "second": 2, "two": 2, "2": 2,
             "third": 3, "three": 3, "3": 3, "fourth": 4, "four": 4, "4": 4,
             "fifth": 5, "five": 5, "5": 5}
_YES = re.compile(r"\b(yes|yeah|yep|yup|sure|do it|delete it|go ahead|confirm|"
                  r"haan|ha)\b", re.I)
_NO = re.compile(r"\b(no|nope|don'?t|keep it|cancel|never ?mind|nahi|na)\b",
                 re.I)


def answer(reply: str) -> str:
    """Step 3's second half: his yes / no / which one."""
    request = pending()
    if request is None:
        return "There's nothing waiting for an answer."
    reply = (reply or "").strip().lower()
    if request["stage"] == "choose":
        pick = next((n for word, n in _ORDINALS.items()
                     if re.search(rf"\b{word}\b", reply)), None)
        if pick is None:
            if _NO.search(reply):
                _set_pending(None)
                return "Okay, I won't forget anything."
            return "Which number? Or say no."
        rows = request["candidates"]
        if not 1 <= pick <= len(rows):
            return f"There are only {len(rows)}. Which number?"
        return _ask_to_confirm(rows[pick - 1], via=request.get("via", "voice"))
    if _NO.search(reply) and not _YES.search(reply):
        _set_pending(None)
        return "Okay, I'll keep it."
    if _YES.search(reply):
        row = request["candidates"][0]
        if confirm_delete(request["token"], row["id"]):
            return f"Done. I've forgotten that {row['text'].rstrip('.')}."
        return "I couldn't delete it — it may already be gone."
    return "Should I delete it? Say yes or no."


def request_delete(fact_id: int, via: str = "dashboard") -> dict | None:
    """The dashboard's first click: a pending request, no deletion."""
    row = entry(fact_id)
    if row is None:
        return None
    _ask_to_confirm(row, via=via)
    request = pending() or {}
    return {"token": request.get("token"), "id": row["id"], "text": row["text"],
            "protected": bool(row.get("protected"))}


def confirm_delete(token: str, fact_id: int) -> bool:
    """THE ONLY DELETE in file 17, voice or dashboard. Needs the pending
    request's token for exactly this row; anything else deletes nothing."""
    import memory
    request = pending()
    if (request is None or request.get("stage") != "confirm"
            or not token or token != request.get("token")
            or int(request.get("chosen") or -1) != int(fact_id)):
        log.warning("delete of #%s refused: no matching confirmed request",
                    fact_id)
        return False
    _set_pending(None)
    gone = memory.forget(int(fact_id))
    log.info("fact #%s deleted after confirmation (%s)", fact_id,
             request.get("via"))
    return gone
