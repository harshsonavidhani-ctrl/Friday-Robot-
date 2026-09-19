"""reflection.py — File 09 step 4: one reflection, three triggers, one debounce.

    reflection.run_self_reflection(reason="idle")   one look back, or None
    reflection.last()                               the latest note
    reflection.flags()                              what she wants ruled on
    reflection.resolve(flag_id, keep=True)          his answer, from the phone
    reflection.start(...)                           the watcher thread

Harsh's spec: "One function, run_self_reflection(), called from three places:
end of a conversation session, a daily scheduled time, and robot idle
detection. Before running, check a last_reflection_at timestamp -- if a
reflection already ran within the last few hours, skip. This prevents the
three triggers from overlapping and running reflection repeatedly, which would
waste API calls and cost."

THE DEBOUNCE IS THE POINT, so it lives in the one function rather than in each
caller. All three triggers fire within minutes of each other on a normal
evening -- the conversation ends, the robot goes idle, and it is late -- and
without the timestamp that is three Haiku calls for one quiet room.
`last_reflection_at` is on DISK, not in memory, because the brain restarts far
more often than four hours (it restarted eight times tonight).

WHAT IT LOOKS AT (file 09 step 5), all of it already stored:
  * the learned preferences (step 1), with their dates;
  * the gaps she logged while answering (step 3);
  * a SAMPLE of the recent event log, with outcomes;
  * the newest things she was told or looked up;
  * the running totals (step 6, growth.py).
A sample, not a transcript: the spec asks for a short self-assessment, "not a
full transcript replay", and a model handed two hundred rows writes about the
rows. It has no tools, no web, no camera, and it never speaks.

THE NOTE IS BOUNDED AND SO IS ITS HISTORY. The latest note is what the next
turns see (situation.py folds `to_adjust` into the context), and the last
REFLECTION_LOG_MAX of them are kept in state/reflection.json so the file
cannot grow forever -- the spec again: "don't just keep appending forever".

⚠ IT NEVER DELETES ANYTHING. The spec: "if two conflict, or one looks
stale/no longer relevant, flag it for the user rather than silently deleting
it." A preference is something Harsh told her to do; a model that quietly
drops one has overruled him, and he would find out by her going back to a
habit he had corrected. So reflection writes FLAGS carrying the row id, the
dashboard shows them with keep/remove, and `resolve()` -- called only by a
button press -- is the only path that can call memory.forget().

WHAT IT DOES WITH WHAT IT FINDS. Nothing, out loud. Harsh, 2026-09-20, asked
for it "written down, spoken only when you ask": the note goes to
state/reflection.json and to the events table, the dashboard shows it, and
`recent_reflection()` reads it back when he asks. It never opens a
conversation of its own -- that is the one thing file 08's scene remarks are
already allowed to do, and one is enough.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import config

log = logging.getLogger("friday.reflection")

TASK = "reflection"
_lock = threading.Lock()
_last_activity = time.monotonic()
_had_turns = False
_thread: threading.Thread | None = None


def _file() -> Path:
    return Path(getattr(config, "REFLECTION_FILE",
                        config.ROOT / "state" / "reflection.json"))


def _read() -> dict:
    """The whole stored state. Never raises; a torn file is an empty one."""
    try:
        data = json.loads(_file().read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def _write(data: dict) -> None:
    try:
        path = _file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
    except OSError as exc:
        log.warning("could not save the reflection: %s", exc)


# --- what main.py tells it -------------------------------------------------

def turn_ended() -> None:
    """A turn just finished: the quiet starts again from now."""
    global _last_activity, _had_turns
    with _lock:
        _last_activity = time.monotonic()
        _had_turns = True


# --- what the dashboard and the voice read back ----------------------------

def last() -> dict | None:
    """The most recent reflection note, or None."""
    note = _read().get("note")
    return note if isinstance(note, dict) and note.get("summary") else None


def log_entries(limit: int = 10) -> list[dict]:
    """The rolling reflection log, newest last. Bounded on write."""
    entries = [e for e in (_read().get("log") or []) if isinstance(e, dict)]
    return entries[-limit:]


def flags(unresolved_only: bool = True) -> list[dict]:
    """Preferences she wants a decision on. Never raises."""
    rows = _read().get("flags")
    if not isinstance(rows, list):
        return []
    return [f for f in rows
            if isinstance(f, dict) and (not unresolved_only or not f.get("resolved"))]


def resolve(flag_id: str, keep: bool) -> dict | None:
    """His answer to one flag. THE ONLY PATH THAT MAY DELETE A PREFERENCE.

    keep=True leaves the row alone and stops asking about it. keep=False
    forgets it -- and that only ever happens from a button he pressed.
    """
    data = _read()
    rows = data.get("flags")
    if not isinstance(rows, list):
        return None
    for flag in rows:
        if not isinstance(flag, dict) or str(flag.get("id")) != str(flag_id):
            continue
        flag["resolved"] = "kept" if keep else "removed"
        flag["resolved_at"] = time.time()
        if not keep:
            try:
                import memory
                fact_id = int(flag.get("fact_id") or 0)
                flag["forgotten"] = bool(fact_id) and memory.forget(fact_id)
            except Exception as exc:            # noqa: BLE001
                log.warning("could not forget preference %s: %s",
                            flag.get("fact_id"), exc)
                flag["forgotten"] = False
        _write(data)
        log.info("reflection flag %s -> %s", flag_id, flag["resolved"])
        return flag
    return None


# --- the debounce ----------------------------------------------------------

def _too_soon() -> float:
    """Seconds still to wait before another reflection is allowed. 0 = go."""
    last_at = float(_read().get("last_reflection_at") or 0.0)
    gap = float(getattr(config, "REFLECTION_MIN_GAP_S", 4 * 3600.0))
    waited = time.time() - last_at
    if waited < 0:                      # clock moved back; do not lock her out
        return 0.0
    return max(0.0, gap - waited)


def _why_not_now(reason: str) -> str | None:
    """The reason this reflection should not run, or None. Cheap checks only."""
    if not getattr(config, "SELF_REFLECTION", True):
        return "switched off"
    if not config.ANTHROPIC_API_KEY:
        return "no API key"
    try:
        import brain
        if brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
            return "over the daily spend cap"
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not read the spend: %s", exc)
    wait = _too_soon()
    if wait:
        return f"one ran {wait / 3600:.1f}h ago (min gap)"
    return None


# --- the reflection itself -------------------------------------------------

_SYSTEM = (
    "You are Friday, a home robot, looking back over what you have learned. "
    "Nobody is listening: nothing you write here is spoken aloud, and it is "
    "read later by Harsh on a screen.\n\n"
    "You are given his standing instructions to you with their dates, the "
    "topics you answered without really knowing, a sample of the last few "
    "things you did and how they went, the newest things you learned, and "
    "your running totals.\n\n"
    "Reply with ONE JSON object and nothing else:\n"
    '{"went_well": "...", "to_adjust": "...", "themes": ["..."], '
    '"flags": [{"fact_id": 1, "kind": "conflict"|"stale", "why": "..."}]}\n\n'
    "- went_well: one or two plain sentences on what is working — what you "
    "got right, what you have learned to do differently, what he seems "
    "happy with.\n"
    "- to_adjust: one or two plain sentences on what to do differently next "
    "time. Specific and about YOU, not a wish about him.\n"
    "⚠ Both are a SELF-ASSESSMENT, not a replay: do not list what happened, "
    "do not quote the log back, do not count things (the totals are given to "
    "you). Write them to be read on a screen, not said out loud.\n"
    "- themes: at most three short phrases naming what you are repeatedly "
    "unsure about (\"her own hardware\", \"cricket scores\"). [] if there is "
    "no pattern -- two unrelated gaps are not a theme.\n"
    "- flags: preferences that need HIS decision. Two that contradict each "
    "other (kind \"conflict\", flag both), or one that plainly no longer "
    "applies (kind \"stale\"). `why` is one short sentence he can rule on.\n\n"
    "⚠ FLAGGING IS NOT DELETING and you cannot delete: he decides, you only "
    "ask. So flag only what really needs him -- an empty list is the normal "
    "answer, and an instruction being old is not a reason to doubt it. Two "
    "instructions about different things are not a conflict.\n"
    "Never invent a preference or a gap that is not in the lists."
)


def _recent_work(limit: int = 12) -> list[dict]:
    """The last few things she DID, with how they went. Never raises."""
    try:
        import memory
        rows = memory.recent_events(limit=limit * 3)
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not read the recent events: %s", exc)
        return []
    # The gaps and the reflections are already in their own sections; showing
    # them here again would have her reflecting on her own reflecting.
    skip = {TASK, "uncertain_response", "uncertain_resolved"}
    return [r for r in rows if r.get("task") not in skip][:limit]


def _recent_memories(limit: int = 10) -> list[str]:
    """The newest things she was told or looked up. Never raises."""
    try:
        import memory
        kind = getattr(config, "LEARNED_PREFERENCE_KIND", "learned_preference")
        return [f.text for f in memory.all_facts(limit=limit * 3)
                if f.kind != kind][:limit]
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not read the recent memories: %s", exc)
        return []


def _lists(preferences: list[dict], gaps: list[dict],
           work: list[dict] | None = None,
           memories: list[str] | None = None,
           counts: dict | None = None) -> str:
    lines = ["HIS STANDING INSTRUCTIONS TO YOU:"]
    if preferences:
        for row in preferences:
            lines.append(f"- [fact_id {row.get('id')}] ({row.get('date')}) "
                         f"{row.get('text', '')}")
    else:
        lines.append("- (none yet)")
    lines.append("")
    lines.append("WHAT YOU ANSWERED WITHOUT REALLY KNOWING:")
    if gaps:
        for row in gaps:
            when = time.strftime("%Y-%m-%d",
                                 time.localtime(row.get("created_at") or 0))
            lines.append(f"- ({when}) {row.get('text', '')}")
    else:
        lines.append("- (nothing logged)")

    # ⚠ THE LOG IS A SAMPLE, NOT A TRANSCRIPT. The spec asks for a short
    # self-assessment, "not a full transcript replay" -- and a model handed
    # two hundred rows writes about the rows.
    if work:
        lines.append("")
        lines.append("THE LAST FEW THINGS YOU DID:")
        for row in work:
            outcome = row.get("outcome") or "no outcome recorded"
            lines.append(f"- {row.get('task') or 'something'}: "
                         f"{str(row.get('text', ''))[:90]} [{outcome}]")
    if memories:
        lines.append("")
        lines.append("THE NEWEST THINGS YOU LEARNED OR WERE TOLD:")
        for text in memories:
            lines.append(f"- {text[:140]}")
    if counts:
        lines.append("")
        lines.append("THE RUNNING TOTALS (file 09 step 6): " + ", ".join(
            f"{name} {value}" for name, value in counts.items() if value))
    return "\n".join(lines)


def _json_object(text: str) -> dict:
    """The JSON object in a model reply, or {}. Never raises."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        return data if isinstance(data, dict) else {}
    except (ValueError, TypeError):
        return {}


def run_self_reflection(reason: str = "idle", force: bool = False) -> dict | None:
    """One look back over what she has learned. Returns the note, or None.

    `force` is the path Harsh asked for himself -- a button, or "what have you
    learned" -- and it skips the debounce only. Everything else (the switch,
    the key, the spend cap) still applies, because those are about whether she
    CAN, not about whether it is worth doing again.

    Never raises: a reflection that fails is a quiet nothing, not a crash.
    """
    if not force:
        why_not = _why_not_now(reason)
        if why_not:
            log.debug("no reflection (%s): %s", reason, why_not)
            return None
    else:
        for check in (_why_not_now(reason),):
            if check and "min gap" not in check:
                log.info("reflection asked for but not possible: %s", check)
                return None

    try:
        import anthropic
        import brain
        import corrections
        import memory
        import uncertainty
    except Exception as exc:                    # noqa: BLE001
        log.warning("reflection could not load what it reads: %s", exc)
        return None

    preferences = corrections.preferences(
        int(getattr(config, "REFLECTION_MAX_PREFERENCES", 25)))
    gaps = uncertainty.gaps(int(getattr(config, "REFLECTION_MAX_GAPS", 25)))
    work = _recent_work(int(getattr(config, "REFLECTION_MAX_EVENTS", 12)))
    memories = _recent_memories(int(getattr(config, "REFLECTION_MAX_MEMORIES",
                                            10)))
    try:
        import growth
        counts = growth.report()["counts"]
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not read the growth counts: %s", exc)
        counts = {}
    if not preferences and not gaps:
        # NOTHING TO REFLECT ON IS NOT A FAILURE, and it must not cost money
        # or move the timestamp -- otherwise the first real reflection waits
        # four hours behind an empty one.
        log.info("nothing to reflect on yet (no preferences, no gaps)")
        return None

    started = time.monotonic()
    try:
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY,
                                     timeout=60.0, max_retries=1)
        message = client.messages.create(
            model=getattr(config, "REFLECTION_MODEL", config.MODEL_FAST),
            max_tokens=int(getattr(config, "REFLECTION_MAX_TOKENS", 500)),
            system=_SYSTEM,
            messages=[{"role": "user",
                       "content": _lists(preferences, gaps, work, memories,
                                         counts)}])
    except Exception as exc:                    # noqa: BLE001
        log.warning("reflection call failed (%s: %s)", type(exc).__name__, exc)
        return None

    raw = "".join(b.text for b in message.content
                  if getattr(b, "type", None) == "text")
    parsed = _json_object(raw)
    went_well = str(parsed.get("went_well") or "").strip()
    to_adjust = str(parsed.get("to_adjust") or "").strip()
    # `summary` stays, as the one plain-language note everything else reads --
    # the dashboard, the voice read-back and situation.py were built on it.
    summary = " ".join(x for x in (went_well, to_adjust) if x) or str(
        parsed.get("summary") or "").strip()
    if not summary:
        log.info("reflection returned nothing usable")
        return None

    cost = 0.0
    try:
        cost = brain._cost_usd(getattr(config, "REFLECTION_MODEL",
                                       config.MODEL_FAST), message.usage)
        brain._record_spend(getattr(config, "REFLECTION_MODEL",
                                    config.MODEL_FAST), message.usage, cost)
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not record the reflection's cost: %s", exc)

    themes = [str(t).strip() for t in (parsed.get("themes") or [])
              if str(t).strip()][:3]
    known = {int(p["id"]): p.get("text", "") for p in preferences
             if str(p.get("id", "")).isdigit()}
    new_flags = []
    for item in (parsed.get("flags") or []):
        if not isinstance(item, dict):
            continue
        try:
            fact_id = int(item.get("fact_id"))
        except (TypeError, ValueError):
            continue
        # ⚠ ONLY A ROW IT WAS ACTUALLY SHOWN. A flag against an id that was
        # not in the list is a hallucinated row number, and the dashboard's
        # remove button would then delete whatever happens to have that id.
        if fact_id not in known:
            log.info("reflection flagged an id it was not given (%s) — dropped",
                     fact_id)
            continue
        kind = str(item.get("kind") or "").lower()
        new_flags.append({
            "id": f"{int(time.time())}-{fact_id}",
            "fact_id": fact_id,
            "text": known[fact_id],
            "kind": kind if kind in ("conflict", "stale") else "review",
            "why": str(item.get("why") or "").strip()[:240],
            "at": time.time(),
        })

    note = {
        "summary": summary[:1200],
        "went_well": went_well[:600],
        "to_adjust": to_adjust[:600],
        "themes": themes,
        "reason": reason,
        "at": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "preferences": len(preferences),
        "gaps": len(gaps),
        "usd": round(cost, 5),
        "seconds": round(time.monotonic() - started, 1),
    }

    data = _read()
    data["last_reflection_at"] = time.time()
    data["note"] = note
    # THE ROLLING LOG, bounded. Only the small fields: the whole note carries
    # costs and counts that are already in the events table, and a log that
    # keeps everything is the thing the spec asks this not to become.
    entries = [e for e in (data.get("log") or []) if isinstance(e, dict)]
    entries.append({k: note[k] for k in
                    ("date", "reason", "went_well", "to_adjust", "themes")})
    data["log"] = entries[-int(getattr(config, "REFLECTION_LOG_MAX", 20)):]
    kept = [f for f in (data.get("flags") or [])
            if isinstance(f, dict) and not f.get("resolved")
            and f.get("fact_id") not in {n["fact_id"] for n in new_flags}]
    data["flags"] = (kept + new_flags)[-int(
        getattr(config, "REFLECTION_MAX_FLAGS", 12)):]
    _write(data)

    try:
        memory.log_event(summary[:400], task=TASK,
                         meta={"reason": reason, "themes": themes,
                               "flags": len(new_flags)})
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not log the reflection: %s", exc)

    log.info("reflected (%s): %d preferences, %d gaps, %d flagged, $%.4f, %.1fs",
             reason, len(preferences), len(gaps), len(new_flags), cost,
             note["seconds"])
    return note


# --- the three triggers ----------------------------------------------------

def _daily_due() -> bool:
    """Is it past the scheduled time, and has today's reflection not run?"""
    at = str(getattr(config, "REFLECTION_DAILY_AT", "") or "").strip()
    if not at:
        return False
    try:
        hour, minute = (int(part) for part in at.split(":", 1))
    except (ValueError, TypeError):
        return False
    now = time.localtime()
    if (now.tm_hour, now.tm_min) < (hour, minute):
        return False
    last_at = float(_read().get("last_reflection_at") or 0.0)
    return time.strftime("%Y-%m-%d", time.localtime(last_at)) != \
        time.strftime("%Y-%m-%d", now)


def tick() -> dict | None:
    """Evaluate all three triggers once. The debounce makes that safe.

    Returns the note if one ran. Called by the watcher thread every
    REFLECTION_POLL_S, and directly by the tests.
    """
    global _had_turns
    with _lock:
        quiet_for = time.monotonic() - _last_activity
        had_turns = _had_turns

    session_end = float(getattr(config, "REFLECTION_SESSION_END_S", 600.0))
    idle_after = float(getattr(config, "REFLECTION_IDLE_S", 1800.0))

    reason = None
    if had_turns and quiet_for >= session_end:
        reason = "the conversation ended"
    elif quiet_for >= idle_after:
        reason = "the robot is idle"
    elif _daily_due():
        reason = "the daily look back"
    if reason is None:
        return None

    note = run_self_reflection(reason)
    if note is not None:
        with _lock:
            _had_turns = False
    return note


def start() -> threading.Thread | None:
    """Watch for the three triggers, on a thread.

    Its own thread for the same reason as idle_study's: main.py's wake loop is
    a blocking generator, and anything that waits beside it swallows a "hey
    Friday".
    """
    global _thread
    if not getattr(config, "SELF_REFLECTION", True):
        return None
    if _thread is not None and _thread.is_alive():
        return _thread

    def watch() -> None:
        log.info("self-reflection on: after %.0fs quiet, daily at %s, and at "
                 "most one every %.1fh",
                 float(getattr(config, "REFLECTION_SESSION_END_S", 600.0)),
                 getattr(config, "REFLECTION_DAILY_AT", "(never)"),
                 float(getattr(config, "REFLECTION_MIN_GAP_S", 14400.0)) / 3600)
        while True:
            try:
                tick()
            except Exception as exc:            # noqa: BLE001
                log.warning("reflection watcher error (%s: %s)",
                            type(exc).__name__, exc)
            time.sleep(float(getattr(config, "REFLECTION_POLL_S", 60.0)))

    _thread = threading.Thread(target=watch, name="reflection", daemon=True)
    _thread.start()
    return _thread


def main() -> int:
    """Run one now, and print it -- the on-demand path from a terminal."""
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(message)s")
    note = run_self_reflection("asked from the command line", force=True)
    if note is None:
        print("nothing to reflect on, or it could not run")
        return 1
    print(f"\n{note['date']}  ({note['reason']}, ${note['usd']:.4f})\n")
    print(note["summary"])
    if note["themes"]:
        print("\nthemes:", ", ".join(note["themes"]))
    for flag in flags():
        print(f"\n[{flag['kind']}] {flag['text']}\n    {flag['why']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
