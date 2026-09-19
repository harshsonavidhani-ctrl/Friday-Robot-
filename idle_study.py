"""idle_study.py — in the quiet, Friday looks up the thing she did not know.

File 08 step 8: "during idle gaps, if there's an unresolved question or
unfamiliar topic from recent conversation or memory, use web_search to look it
up and save_memory to store what was learned. On the next interaction Friday
can bring it up naturally. Rate-limit this -- once per idle period is enough,
not continuous background searching."

IT RUNS ON ITS OWN THREAD WITH ITS OWN SYNC CLIENT, and that is not an
accident. main.py's wake loop is `for trigger in source.listen()`, so anything
awaited inside it blocks the loop: a search takes 3-15 s, and a "hey Friday"
arriving in that window would be handled after it -- the capture would start
long after Harsh had finished talking, and his words would be gone. So this
never goes near the loop or the Brain's history. Same pattern, and the same
reason, as main._pick_face and mcp_server/tools/narrate.py.

IT NEVER SPEAKS. Nobody asked it anything; it reads, saves one sentence, and
stops. The next turn can mention it, because situation.py puts a "you looked
this up earlier" line in front of exactly one turn (see pending/mark_offered).

WHAT KEEPS IT CHEAP -- all of them, or the thread just sleeps:
  * idle for IDLE_STUDY_AFTER_S, where idle means no turn has ended since
  * ONE study per idle period, and never within IDLE_STUDY_MIN_INTERVAL_S
  * at most IDLE_STUDY_MAX_PER_DAY, counted from the spend log's own day
  * nothing to be curious about = nothing spent: with no conversation to read
    it does not call at all, and the model itself answers NOTHING when the
    conversation holds no open question. Measured: ~$0.002 for a NOTHING,
    ~$0.02 when it searches.

⚠ WHAT IT MUST NEVER DO IS INVENT. The fact it saves is read back as memory on
later turns and spoken as something she knows (see the 23:09 "cat" in
situation.block for what that costs). So it may only save what a search
actually returned, it says NOTHING rather than guessing, and every fact it
stores is marked source="idle-study" in the database so its origin is visible.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Callable

import config

log = logging.getLogger("friday.idle_study")

NOTHING = "NOTHING"

_lock = threading.Lock()
_last_activity = time.monotonic()
_last_study = 0.0                       # time.monotonic()
_studied_today: tuple[str, int] = ("", 0)
_thread: threading.Thread | None = None


def _file() -> Path:
    return Path(getattr(config, "IDLE_STUDY_FILE",
                        config.ROOT / "state" / "idle_study.json"))


# --- what main.py tells it ------------------------------------------------

def turn_ended() -> None:
    """A turn just finished: the idle period starts again from now."""
    global _last_activity
    with _lock:
        _last_activity = time.monotonic()


# --- what the next turn reads --------------------------------------------

def pending() -> dict | None:
    """The thing she looked up and has not mentioned yet, if it is fresh."""
    try:
        note = json.loads(_file().read_text())
        if note.get("mentioned") or not note.get("fact"):
            return None
        age = time.time() - float(note.get("at", 0))
        if age < -5 or age > float(getattr(config, "IDLE_STUDY_FRESH_S", 7200)):
            return None
        note["age_s"] = age
        return note
    except (OSError, ValueError, TypeError, AttributeError):
        return None


def mark_offered() -> None:
    """She has now had one turn with it in front of her. Don't offer it again.

    Once, not until-she-says-it: there is no way to tell from the outside
    whether she worked it into what she said, and a note re-offered every turn
    is a robot that keeps telling you the same thing it read.
    """
    note = pending()
    if note is None:
        return
    note["mentioned"] = True
    note.pop("age_s", None)
    try:
        _file().write_text(json.dumps(note))
    except OSError as exc:
        log.warning("could not mark the idle note as offered: %s", exc)


# --- the study itself -----------------------------------------------------

def _recent_conversation(history: list[dict[str, Any]], limit: int = 8) -> str:
    """The last few turns as plain text: what she might still be curious about.

    Tool blocks and images are dropped -- what matters is what was said.
    """
    lines: list[str] = []
    for message in history[-limit:]:
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = " ".join(
                block.get("text", "") for block in content
                if isinstance(block, dict) and block.get("type") == "text")
        else:
            text = ""
        text = " ".join(text.split())
        if text:
            who = "Harsh" if message.get("role") == "user" else "You"
            lines.append(f"{who}: {text[:400]}")
    return "\n".join(lines[-limit:])


_SMALL_WORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to", "for", "is",
    "was", "are", "were", "her", "his", "its", "it", "she", "he", "they",
    "what", "how", "why", "when", "where", "which", "who", "does", "do", "did",
    "about", "with", "from", "that", "this", "these", "those", "own", "you",
}


def _about_the_same_thing(topic: str, text: str) -> bool:
    """Do the gap and what was found share a real word? Never raises.

    Deliberately crude and deliberately strict: it is a check against counting
    a look-up about something else as a gap closed. A real word means one that
    is not in _SMALL_WORDS and is longer than three letters.
    """
    try:
        def words(s: str) -> set[str]:
            return {w.strip(".,!?;:'\"()").lower() for w in s.split()}
        gap = {w for w in words(topic) if len(w) > 3 and w not in _SMALL_WORDS}
        found = {w for w in words(text) if len(w) > 3}
        return bool(gap & found)
    except Exception:                                   # noqa: BLE001
        return False


def _why_not_now() -> str | None:
    global _studied_today
    if not getattr(config, "IDLE_STUDY", False):
        return "switched off"
    if not config.ANTHROPIC_API_KEY:
        return "no API key"
    with _lock:
        idle_for = time.monotonic() - _last_activity
        since_study = time.monotonic() - _last_study
    after = float(getattr(config, "IDLE_STUDY_AFTER_S", 300.0))
    if idle_for < after:
        return None                     # not idle yet; not worth logging
    interval = float(getattr(config, "IDLE_STUDY_MIN_INTERVAL_S", 1800.0))
    if _last_study and since_study < interval:
        return None                     # already studied this quiet stretch
    day = time.strftime("%Y-%m-%d")
    if _studied_today[0] != day:
        _studied_today = (day, 0)
    if _studied_today[1] >= int(getattr(config, "IDLE_STUDY_MAX_PER_DAY", 8)):
        return "already studied the most she may today"
    import brain                                        # noqa: PLC0415
    if brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
        return "over the daily spend cap"
    return None


def study(history: list[dict[str, Any]]) -> dict | None:
    """One quiet look-up. Returns what was saved, or None. Never raises."""
    global _last_study, _studied_today

    conversation = _recent_conversation(history)
    if not conversation:
        return None                     # nothing said yet: nothing to wonder about

    import anthropic                                    # noqa: PLC0415
    import brain                                        # noqa: PLC0415
    import memory                                       # noqa: PLC0415

    client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY,
                                 timeout=60.0, max_retries=1)
    system = (
        "You are Friday, a robot, alone in a quiet room. Nobody is listening "
        "and nothing you write is spoken aloud.\n\n"
        "The message you are given is a real transcript of what Harsh and you "
        "said to each other a few minutes ago. Take it as yours and do not "
        "comment on it.\n\n"
        "Pick the ONE thing in it that was left unresolved, that you did not "
        "know, that you were unsure about, or that he plainly cares about and "
        "you know little about — and that a web search can settle. Search for "
        "it. Then reply with ONE sentence worth remembering: third person, "
        "plain, specific, and true to what the search returned — a fact, not a "
        "summary of your search, and not a sentence about searching.\n\n"
        f"Your whole reply is either that one sentence, or exactly {NOTHING}. "
        f"Nothing else is allowed: no preamble, no explanation of what you "
        f"found or did not find, no sources, no offer to help. {NOTHING} is a "
        f"perfectly good answer when nothing there needs looking up — better "
        f"than a sentence you are not sure of. Never write a fact the search "
        f"did not show you."
    )
    tool = {"type": "web_search_20250305", "name": "web_search",
            "max_uses": int(getattr(config, "IDLE_STUDY_MAX_SEARCHES", 2))}
    # FILE 09 STEP 3 FEEDS THIS. A gap she logged during a turn is better than
    # anything this thread can infer from a transcript: she was there, she
    # answered, and she knew at the time that she did not really know. Offered,
    # not forced -- the conversation may have moved on to something better.
    gap = ""
    offered_gap = ""
    try:
        import uncertainty                              # noqa: PLC0415
        topic = uncertainty.next_topic()
        if topic:
            offered_gap = topic
            gap = (f"Earlier today you answered about \"{topic}\" without "
                   f"really knowing. If that is still the most useful thing "
                   f"to settle, start there.\n\n")
    except Exception as exc:                            # noqa: BLE001
        log.debug("could not read the logged gaps: %s", exc)

    messages: list[dict[str, Any]] = [{"role": "user", "content": (
        f"This is what was said in the room a few minutes ago:\n\n"
        f"{conversation}\n\n"
        f"{gap}"
        f"Search for the one thing worth looking up, then give the one "
        f"sentence — or {NOTHING}.")}]

    model = getattr(config, "BRAIN_FAST_MODEL", config.MODEL_FAST)
    text = ""
    query = ""
    searched = False
    try:
        for _ in range(int(getattr(config, "IDLE_STUDY_MAX_ROUNDS", 4))):
            reply = client.messages.create(
                model=model,
                max_tokens=int(getattr(config, "IDLE_STUDY_MAX_TOKENS", 400)),
                system=system, messages=messages, tools=[tool])
            try:
                brain._record_spend(model, reply.usage,
                                    brain._cost_usd(model, reply.usage))
            except Exception:                           # noqa: BLE001
                log.warning("could not record the cost of the study")
            for block in reply.content:
                kind = getattr(block, "type", None)
                if kind == "text":
                    text += block.text
                elif kind in ("server_tool_use", "web_search_tool_result"):
                    # ⚠ ANYTHING WRITTEN BEFORE THE SEARCH IS NOT A FACT, and
                    # it was being saved as one: "I need to search for
                    # information about robotics competitions in India.The
                    # e-Yantra Robotics Competition..." went into the database
                    # and would have been said aloud later as memory. Only what
                    # comes after the last search survives.
                    text = ""
                    if kind == "server_tool_use":
                        searched = True
                        inputs = (block.input if isinstance(block.input, dict)
                                  else {})
                        query = str(inputs.get("query", "")) or query
                        log.info("idle study searching %r", query)
            # A hosted search can pause the turn; the history already ends on
            # that assistant message, so asking again resumes it.
            if reply.stop_reason != "pause_turn":
                break
            messages.append({"role": "assistant", "content": reply.content})
    except Exception as exc:                            # noqa: BLE001
        log.info("idle study skipped (%s: %s)", type(exc).__name__, exc)
        return None

    with _lock:
        _last_study = time.monotonic()
    day = time.strftime("%Y-%m-%d")
    _studied_today = (day, (_studied_today[1] if _studied_today[0] == day
                            else 0) + 1)

    fact = " ".join(text.split())
    if not fact or fact.strip(" .!\"'`").upper() == NOTHING:
        log.info("idle study: nothing worth looking up in the last "
                 "conversation (%s)", NOTHING)
        return None
    if not searched:
        # It answered from training instead of looking. That is exactly the
        # thing that must not become a remembered "fact".
        log.info("idle study: answered without searching — not saved: %r",
                 fact[:80])
        return None

    try:
        stored = memory.remember(
            fact, source=getattr(config, "IDLE_STUDY_SOURCE", "idle-study"))
    except Exception as exc:                            # noqa: BLE001
        log.warning("idle study could not save what it learned: %s", exc)
        return None

    note = {"topic": query, "fact": stored["text"], "at": time.time(),
            "mentioned": False, "updated": stored["updated"]}
    try:
        _file().parent.mkdir(parents=True, exist_ok=True)
        _file().write_text(json.dumps(note))
    except OSError as exc:
        log.warning("could not write the idle note: %s", exc)
    log.info("idle study learned (%s): %r", query or "no query", stored["text"])

    # FILE 09 STEP 6 COUNTS THE GAPS THAT ACTUALLY CLOSED. Being offered a gap
    # is not the same as settling it -- the model is told to start there "if
    # that is still the most useful thing", and it is free to look up
    # something else entirely. So this only counts when the gap's own words
    # turn up in what was searched for or in what was saved. Over-claiming
    # here would inflate a growth score whose whole point is being honest.
    if offered_gap and _about_the_same_thing(offered_gap, f"{query} {fact}"):
        try:
            import growth                               # noqa: PLC0415
            growth.mark_resolved(offered_gap, stored["text"])
        except Exception as exc:                        # noqa: BLE001
            log.debug("could not mark the gap resolved: %s", exc)
    return note


def start(history: Callable[[], list[dict[str, Any]]]) -> threading.Thread | None:
    """Watch for quiet, on a thread. `history` returns the brain's messages."""
    global _thread
    if not getattr(config, "IDLE_STUDY", False):
        return None
    if _thread is not None and _thread.is_alive():
        return _thread

    def watch() -> None:
        log.info("idle curiosity on: after %.0fs quiet, at most one look-up "
                 "every %.0fs and %d a day",
                 float(getattr(config, "IDLE_STUDY_AFTER_S", 300.0)),
                 float(getattr(config, "IDLE_STUDY_MIN_INTERVAL_S", 1800.0)),
                 int(getattr(config, "IDLE_STUDY_MAX_PER_DAY", 8)))
        said = None
        while True:
            try:
                why_not = _why_not_now()
                if why_not != said and why_not:
                    log.info("idle curiosity waiting: %s", why_not)
                    said = why_not
                if why_not is None:
                    with _lock:
                        idle_for = time.monotonic() - _last_activity
                    if idle_for >= float(getattr(config, "IDLE_STUDY_AFTER_S",
                                                 300.0)):
                        study(list(history() or []))
            except Exception as exc:                    # noqa: BLE001
                # A background curiosity must never take the robot down.
                log.warning("idle curiosity error (%s: %s)",
                            type(exc).__name__, exc)
            time.sleep(float(getattr(config, "IDLE_STUDY_POLL_S", 20.0)))

    _thread = threading.Thread(target=watch, name="idle-study", daemon=True)
    _thread.start()
    return _thread


def main() -> int:
    """Run one study now against a made-up conversation, and print it."""
    import argparse

    ap = argparse.ArgumentParser(description="Friday's idle curiosity.")
    ap.add_argument("said", nargs="*",
                    default=["What's the tallest building in Pune?"])
    ap.add_argument("--pending", action="store_true",
                    help="print the note waiting to be mentioned")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    if args.pending:
        print(pending())
        return 0
    history = [{"role": "user", "content": " ".join(args.said)},
               {"role": "assistant", "content": "I'm not sure about that one."}]
    print(study(history))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
