"""Curiosity: she asks to go exploring by herself (File 21 step 10, spec 4.15).

    mode off   never (what config ships)
    mode ask   "I'm curious about the house. Can I explore for a while?"
               -> listen 10 s -> yes: the mission starts, exactly as "explore"
                                 no / stop: not again for no_cooldown_h (2 h)
                                 silence / anything else: silence_cooldown_min (30 min)
    mode auto  "I'm going to explore for a while." -> the mission starts

THE SCORE (0-1, only while IDLE, at most once per check_period_s):

    w1 * clamp(frontier_m / 10)              unexplored edge the LAST mission
                                             left reachable (0 when it finished)
  + w2 * clamp(hours_since_last_mission / 24)
  + w3 * (stale + unnamed areas) / all areas

and she speaks up only when score >= threshold AND every gate is open: a home
anchor exists, she is idle >= min_idle_min, nobody has talked for
min_quiet_conversation_min, it is outside quiet_hours, no cooldown, and -- the
last and only gate that costs anything -- a LiDAR LOOK says she is standing at
the home anchor (`localized`, spec 4.15). Nothing here moves her.

TWO READINGS OF THE SPEC, both stricter than its words:

  * "localized" is the home look, measured NOW, never the boot's placement
    (that goes stale the moment a mission drives her anywhere). A mission
    started at the home spot adopts the home anchor's frame, so curiosity's
    trips extend the house she already knows instead of starting a frame
    somewhere else and moving her home anchor there.
  * Not in the spec, and blocking: a run or a journey already has the robot,
    quiet mode, a mission a power cut interrupted that is still waiting for
    "continue" / "go back", and -- in ask mode -- nobody able to hear the
    answer (the mic is friday-brain's; it arrives with the voice hook, step 11).

"Nobody has talked" needs a clock the mission's process can read. Every "he
spoke" time lives inside friday-brain's wake sources, so the voice hook (step
11) marks state/exploration_mission/heard.json on every turn. Until that file
exists the quiet time is UNKNOWN, and unknown is not quiet: she never asks.

The cooldowns, and what the last mission left, are kept per environment in
envs/<env>/curiosity.json (durable.py: atomic, with a .bak) so a reboot does
not reset a "no".
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from exploration_mission import durable, label_dialog

log = logging.getLogger("friday.exploration_mission.curiosity")

MODES = ("off", "ask", "auto")
FRONTIER_FULL_M = 10.0
HOURS_FULL = 24.0
KEEP_ANSWERS = 10


def clamp01(x: float) -> float:
    if x is None or not math.isfinite(x):
        return 0.0
    return min(1.0, max(0.0, float(x)))


def _minutes(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def in_quiet_hours(local: _dt.datetime, quiet_hours) -> bool:
    """[start, end) in local time; a window that wraps midnight (22:00-08:00)
    is the usual case. start == end is an empty window."""
    start, end = (_minutes(t) for t in quiet_hours)
    m = local.hour * 60 + local.minute
    if start == end:
        return False
    if start < end:
        return start <= m < end
    return m >= start or m < end


def score(frontier_m: float | None, hours_since_mission: float | None,
          stale_unnamed: int, areas: int, weights) -> float:
    """Spec 4.15. Never a mission: hours = full (the only way to have a home
    anchor and no mission is "this is home" said while idle)."""
    w1, w2, w3 = (float(w) for w in weights)
    f = clamp01((frontier_m or 0.0) / FRONTIER_FULL_M)
    h = 1.0 if hours_since_mission is None else clamp01(hours_since_mission / HOURS_FULL)
    a = clamp01(stale_unnamed / areas) if areas else 0.0
    return w1 * f + w2 * h + w3 * a


# Answers to THIS question only, on top of the label dialog's yes/no words
# (those were tuned on 1057 real transcripts in step 6 and stay as they are).
# "Can I explore for a while?" -- "sure", "go ahead", "chalo" mean yes here.
CURIOUS_YES = ("sure", "okay", "ok", "go ahead", "go for it", "of course",
               "chalo", "explore", "go explore", "karo", "kar lo")
CURIOUS_NO = ("not now", "later", "maybe later", "no thanks", "abhi nahi",
              "baad mein", "baad me", "mat karo", "don't", "dont", "do not")


def _starts(norm: str, phrases) -> bool:
    return any(norm == p or norm.startswith(p + " ") for p in phrases)


def classify(answer: str | None) -> str:
    """"yes" | "no" | "silence". Only a yes starts a trip; "stop" and "finish"
    are a no; a sentence that is neither counts as no answer (the short
    cooldown), never as a yes."""
    norm = label_dialog.normalize(answer) if answer else ""
    if not norm:
        # The dialog's normaliser drops "ok"/"okay" as fillers; said ALONE to
        # this question they are the answer.
        bare = re.sub(r"[^a-z ]+", " ", (answer or "").lower()).split()
        said = set(bare)
        return ("yes" if said & {"ok", "okay"} and said <= {"ok", "okay", "please"}
                else "silence")
    if (label_dialog.HALT_RE.match(norm) or label_dialog.FINISH_RE.match(norm)
            or _starts(norm, CURIOUS_NO)):
        return "no"
    yn = label_dialog.yes_no(answer)
    if yn is True:
        return "yes"
    if yn is False:
        return "no"
    if _starts(norm, CURIOUS_YES):
        return "yes"
    return "silence"


@dataclass
class Facts:
    """What the mission knows when it checks. Everything but `at_home`."""
    mode: str
    has_home: bool
    idle_s: float
    quiet_s: float | None           # since anyone last spoke; None = unknown
    busy: str | None                # why the robot is not free, or None
    can_listen: bool
    frontier_m: float | None
    hours_since_mission: float | None
    stale_unnamed: int
    areas: int


class Curiosity:
    def __init__(self, cfg_section, state_path_for):
        """`cfg_section` = cfg.curiosity; `state_path_for(env_id)` -> Path."""
        self.c = cfg_section
        self._path_for = state_path_for
        self.last: dict = {}                    # the last decision, for the dashboard

    # --- the persisted part -------------------------------------------------------
    def load(self, env_id: str) -> dict:
        data, _src = durable.load_json(self._path_for(env_id))
        return data if isinstance(data, dict) else {}

    def _save(self, env_id: str, data: dict) -> None:
        try:
            durable.save_json(self._path_for(env_id), data)
        except OSError as exc:
            log.warning("curiosity state not saved: %s", exc)

    def cooldown_left_s(self, env_id: str, wall: float) -> float:
        return max(0.0, float(self.load(env_id).get("next_ok_at") or 0.0) - wall)

    def frontier_m(self, env_id: str) -> float | None:
        v = self.load(env_id).get("frontier_m_left")
        return None if v is None else float(v)

    def mission_ended(self, env_id: str, mission_id: str, frontier_m_left,
                      wall: float) -> None:
        data = self.load(env_id)
        data.update(frontier_m_left=(None if frontier_m_left is None
                                     else round(float(frontier_m_left), 2)),
                    frontier_mission=mission_id, frontier_at=wall)
        self._save(env_id, data)

    def answered(self, env_id: str, outcome: str, wall: float) -> float:
        """Record the outcome of a question (or an announcement) and set the
        cooldown. Returns the cooldown in seconds."""
        data = self.load(env_id)
        if outcome == "no":
            cool = float(self.c.no_cooldown_h) * 3600.0
        elif outcome == "silence":
            cool = float(self.c.silence_cooldown_min) * 60.0
        else:                                   # yes / auto: the mission is the rest
            cool = 0.0
        data["next_ok_at"] = wall + cool
        answers = list(data.get("answers") or [])
        answers.append({"at": round(wall, 1), "outcome": outcome})
        data["answers"] = answers[-KEEP_ANSWERS:]
        self._save(env_id, data)
        log.info("curiosity %s -> next ask in %.0f min", outcome, cool / 60.0)
        return cool

    # --- the decision -------------------------------------------------------------
    def score_of(self, f: Facts) -> float:
        return score(f.frontier_m, f.hours_since_mission, f.stale_unnamed, f.areas,
                     self.c.weights)

    def why_not(self, f: Facts, env_id: str, wall: float,
                local: _dt.datetime | None = None) -> str | None:
        """None = ask (or announce) now, as far as anything but the LiDAR look
        can tell. Otherwise the first reason, cheapest checks first."""
        local = local or _dt.datetime.fromtimestamp(wall)
        if f.mode not in ("ask", "auto"):
            return "off"
        if not f.has_home:
            return "no home anchor: the first mission is started by voice"
        if f.busy:
            return f.busy
        if in_quiet_hours(local, self.c.quiet_hours):
            return "quiet hours"
        left = self.cooldown_left_s(env_id, wall)
        if left > 0:
            return f"cooldown ({left / 60.0:.0f} min left)"
        if f.idle_s < float(self.c.min_idle_min) * 60.0:
            return f"idle only {f.idle_s / 60.0:.1f} min"
        need = float(self.c.min_quiet_conversation_min) * 60.0
        if f.quiet_s is None:
            return "no conversation clock yet (the voice hook marks it)"
        if f.quiet_s < need:
            return f"someone talked {f.quiet_s / 60.0:.1f} min ago"
        if f.mode == "ask" and not f.can_listen:
            return "nobody can hear the answer yet (the voice hook, step 11)"
        s = self.score_of(f)
        if s < float(self.c.threshold):
            return f"score {s:.2f} < {float(self.c.threshold):.2f}"
        return None

    def note(self, **decision) -> None:
        self.last = dict(decision)


def facts_dict(f: Facts) -> dict:
    return asdict(f)


def state_path(store, env_id: str) -> Path:
    return Path(store.env_dir(env_id)) / "curiosity.json"
