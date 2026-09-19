"""growth.py — File 09 step 6: how much Friday has actually learned and done.

    growth.report()     every count, the level, and one plain sentence
    growth.say()        that sentence on its own, for the voice
    ./venv/bin/python growth.py

NO NEW DATABASE, AND NO NEW COLUMNS. Harsh's spec: "Extend the existing SQLite
logging schema (don't start a new database)". It did not need extending: the
facts table has carried `kind` and `source` since it was written, and the
events table carries `task` and `outcome`, so every number here is a query
over rows that were already being stored. The same thing was true of step 1 --
a correction is a `kind`, not a new table -- and it is worth saying out loud
twice, because the reflex on being asked for statistics is to build somewhere
to put them.

WHAT EACH NUMBER MEANS, because a count nobody can explain is worse than none:
  * corrections_learned     facts with kind=learned_preference. Things Harsh
                            told her to do differently, that she kept.
  * memories_added          everything else in the facts table.
  * tasks_worked/failed     events with an outcome. `partial` is counted
                            separately and honestly -- a move that was cut
                            short by the safety stop is not a failure and is
                            not a success.
  * reflections_completed   events with task="reflection" (step 4).
  * uncertain_logged        gaps she admitted while answering (step 3).
  * uncertain_resolved      of those, the ones the quiet look-up later went
                            and settled (file 08 step 8, idle_study).

⚠ THE LEVEL IS A ROUGH MILESTONE AND IT SAYS SO. The spec: "not a fake
gamified number, just a readable summary". So the weights are stated in the
code, the points never decay, nothing is random, and `report()` returns every
raw count beside the level -- the counts are the answer, the level is a way of
saying "further along than last week" in one word. A correction is worth the
most because it is the hardest thing to earn: it means she got something wrong,
he bothered to say so, and she kept it.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import config

log = logging.getLogger("friday.growth")

# points per thing. Stated here rather than hidden in a formula below.
WEIGHTS = {
    "corrections_learned": 5.0,
    "uncertain_resolved": 3.0,
    "reflections_completed": 2.0,
    "memories_added": 1.0,
    "tasks_worked": 0.2,
}
# ⚠ AND A CEILING ON THE ONE THAT REPEATS. Measured on the live database the
# first time this ran: 165 finished jobs at 0.2 each was 33 of her 50 points,
# so two thirds of "how much she has learned" was answering "what time is it"
# over and over. Doing the same thing two hundred times is not two hundred
# times the growth, and a number that says otherwise is the gamified thing
# the spec asks this not to be. Learning has no ceiling; activity does.
MAX_POINTS = {"tasks_worked": 10.0}
POINTS_PER_LEVEL = 25.0

RESOLVED_TASK = "uncertain_resolved"


def _counts() -> dict[str, int]:
    """Every number, straight out of the tables that already hold them."""
    import memory                                        # noqa: PLC0415
    preference_kind = getattr(config, "LEARNED_PREFERENCE_KIND",
                              "learned_preference")
    con = memory.db()
    def one(sql: str, *args) -> int:
        try:
            row = con.execute(sql, args).fetchone()
            return int(row[0] if row and row[0] is not None else 0)
        except Exception as exc:                         # noqa: BLE001
            log.debug("growth query failed (%s): %s", sql[:40], exc)
            return 0

    return {
        "corrections_learned": one(
            "SELECT COUNT(*) FROM facts WHERE kind=?", preference_kind),
        "memories_added": one(
            "SELECT COUNT(*) FROM facts WHERE kind IS NULL OR kind<>?",
            preference_kind),
        "tasks_worked": one(
            "SELECT COUNT(*) FROM events WHERE outcome='worked'"),
        "tasks_failed": one(
            "SELECT COUNT(*) FROM events WHERE outcome='failed'"),
        "tasks_partial": one(
            "SELECT COUNT(*) FROM events WHERE outcome='partial'"),
        "reflections_completed": one(
            "SELECT COUNT(*) FROM events WHERE task='reflection'"),
        "uncertain_logged": one(
            "SELECT COUNT(*) FROM events WHERE task='uncertain_response'"),
        "uncertain_resolved": one(
            "SELECT COUNT(*) FROM events WHERE task=?", RESOLVED_TASK),
        "looked_up_herself": one(
            "SELECT COUNT(*) FROM facts WHERE source=?",
            getattr(config, "IDLE_STUDY_SOURCE", "idle-study")),
    }


def _sentence(counts: dict[str, int], level: int) -> str:
    """One plain line, the way she would say it. No numbers she cannot back."""
    bits = []
    if counts["corrections_learned"]:
        bits.append(f"{counts['corrections_learned']} thing"
                    f"{'s' if counts['corrections_learned'] != 1 else ''} "
                    f"you've corrected me on")
    if counts["memories_added"]:
        bits.append(f"{counts['memories_added']} things I remember")
    if counts["uncertain_resolved"]:
        bits.append(f"{counts['uncertain_resolved']} gap"
                    f"{'s' if counts['uncertain_resolved'] != 1 else ''} "
                    f"I went and closed")
    elif counts["uncertain_logged"]:
        bits.append(f"{counts['uncertain_logged']} things I know I don't know")
    done = counts["tasks_worked"] + counts["tasks_failed"] + counts["tasks_partial"]
    if done:
        bits.append(f"{counts['tasks_worked']} of {done} jobs that went right")
    if not bits:
        return "I haven't learned anything worth counting yet."
    return f"Level {level}: " + ", ".join(bits) + "."


def _next_step(counts: dict[str, int]) -> str:
    """The honest answer to "how do I make that number go up"."""
    if not counts["corrections_learned"]:
        return ("tell me when I get something wrong — that is the thing I "
                "learn most from")
    if counts["uncertain_logged"] and not counts["uncertain_resolved"]:
        return "let me sit quiet for a while and I'll go and look up what I missed"
    if not counts["reflections_completed"]:
        return "ask me to reflect, or leave me be for a few hours"
    return "keep talking to me and keep correcting me"


_cache: tuple[float, dict] = (0.0, {})


def report(max_age_s: float | None = None) -> dict[str, Any]:
    """Every count, the level, and one sentence. Never raises.

    CACHED FOR A FEW SECONDS, because the dashboard rides the /api/state
    heartbeat (every 4 s) and this is nine COUNT(*) queries. The numbers move
    at the speed of a conversation; re-counting them nine times a minute is
    work the Pi has better uses for. Pass max_age_s=0 for a fresh count.
    """
    global _cache
    ttl = float(getattr(config, "GROWTH_CACHE_S", 20.0)
                if max_age_s is None else max_age_s)
    when, cached = _cache
    if cached and time.time() - when <= ttl:
        return cached
    try:
        counts = _counts()
    except Exception as exc:                             # noqa: BLE001
        log.warning("could not read the growth counts: %s", exc)
        return {"counts": {}, "points": 0.0, "level": 0,
                "summary": "", "next": "", "at": time.time()}
    points = 0.0
    for name, value in counts.items():
        earned = WEIGHTS.get(name, 0.0) * value
        points += min(earned, MAX_POINTS.get(name, earned))
    level = int(points // POINTS_PER_LEVEL) + 1 if points else 0
    data = {"counts": counts,
            "points": round(points, 1),
            "capped": dict(MAX_POINTS),
            "level": level,
            "weights": dict(WEIGHTS),
            "points_per_level": POINTS_PER_LEVEL,
            "summary": _sentence(counts, level),
            "next": _next_step(counts),
            "at": time.time()}
    _cache = (time.time(), data)
    return data


def say() -> str:
    """The one line, for the voice path."""
    return report()["summary"]


def mark_resolved(topic: str, fact: str = "", source: str = "idle-study") -> dict | None:
    """A gap she logged has now been settled. Never raises.

    File 08's quiet look-up calls this when the thing it just searched for was
    a gap from step 3 rather than something it inferred from a transcript.
    It is its OWN event task, deliberately: an `uncertain_response` row must
    keep its empty outcome (see uncertainty.py -- an outcome would make a gap
    resurface as a lesson that worked), so "it got answered" is a new row
    rather than an edit of the old one.
    """
    if not (topic or "").strip():
        return None
    try:
        import memory                                    # noqa: PLC0415
        row = memory.log_event(topic.strip()[:200], task=RESOLVED_TASK,
                               meta={"source": source, "fact": fact[:200]})
        log.info("gap closed: %r", topic[:60])
        return row
    except Exception as exc:                             # noqa: BLE001
        log.warning("could not mark the gap resolved: %s", exc)
        return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    data = report()
    print(data["summary"] or "(nothing counted yet)")
    print()
    for name, value in data["counts"].items():
        print(f"  {name:24s} {value}")
    print(f"\n  points {data['points']} "
          f"({POINTS_PER_LEVEL:g} per level) -> level {data['level']}")
    print(f"  next: {data['next']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
