"""The offline command router (File 21 step 6, spec 4.8).

    router = Router(actions, resolve_area, resolve_place)
    handled = router.handle("Friday, kitchen chalo", ctx)     # -> bool

`handle` sees a transcript BEFORE the file 08 brain. True = it was a mission
command, dealt with here, and the brain is skipped. False = not ours; the
existing pipeline (offline.py's reflexes, then the brain) carries on exactly
as it does today. No model, no network: regular expressions over the 4.7
normalisation, and that is all.

EVERY PATTERN MATCHES THE WHOLE UTTERANCE, so ordinary talk is never captured
("go back to sleep" is not "go back"; "stop exploring" is FINISH, not HALT).
Checked in the spec's order:

    HALT (whole) -> FINISH -> CONTINUE -> LABEL_NOW -> GO_TO -> RENAME ->
    FORGET -> LIST -> STATUS -> SET_HOME -> START -> HALT (contains stop/ruko)

WHO CLAIMS WHAT.
  * Name-slot intents (GO_TO, RENAME, FORGET) claim the utterance only when
    the name resolves to a saved area / landmark: "go to sleep" and "forget
    it" still reach the existing pipeline. The one exception is the spec's:
    GO_TO during a mission answers "I don't know where X is yet."
  * A matched intent in a mission state where it is not allowed gets "I can't
    do that right now." (True).
  * ⚠ ONE NARROWING OF THE SPEC: with NO mission at all (IDLE, nothing
    interrupted), the mission-only words -- stop, continue, chalo, go back,
    "this is X" -- are NOT ours and return False. "Continue" said to the brain
    mid-story, "come back" (File 14's own reflex) and "this is the kitchen"
    (File 14's remember_place) keep doing what they do today.
  * Step 11, with the existing pipeline behind it (offline.py measured on
    each phrase): GO_TO with no mission running is NOT ours -- File 14's
    nav_go already drives "go to the kitchen" offline, and the mission's own
    go-to has no engine outside a run. A bare "status" is offline.py's robot
    status unless a mission is running. And Harsh, 2026-10-01: "this is home"
    / "yahan home hai" stay his PERMANENT home (File 10); the mission's start
    is marked only by "set home here" / "mark the start here".

This module decides and delegates. Everything that moves, saves or speaks
goes through the `actions` object the mission manager supplies, so the router
is tested with a recording fake and nothing drives.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable

from exploration_mission import area_store
from exploration_mission.label_dialog import (FINISH_RE, HALT_RE, name_slot,
                                              normalize, normalize_cased, yes_no)

# --- mission states (spec 3.3) -------------------------------------------------------

IDLE_STATES = frozenset({"BOOT_RESTORE", "RELOCALIZE", "IDLE"})
STATES = frozenset(IDLE_STATES | {
    "MISSION_INIT", "EXPLORE", "ASK_LABEL", "SAVE_AREA", "TARGET_NAV", "PAUSED",
    "SUMMARIZE", "RETURN_TO_START", "ARRIVED_START", "SAVE_MISSION", "HELP",
    "HELP_WAIT", "FAILSAFE"})
CONFIRM_TTL_S = 30.0


@dataclass
class Context:
    state: str = "IDLE"
    localized: bool = True
    interrupted: bool = False       # a mission a power cut interrupted is waiting
    moving: bool = False
    dialog: str | None = None       # "label" | "curiosity": a dialog owns the mic

    @property
    def mission_active(self) -> bool:
        return self.state not in IDLE_STATES


@dataclass
class Intent:
    name: str
    args: dict = field(default_factory=dict)


# --- patterns (whole normalised utterance) ------------------------------------------

CONTINUE_RE = re.compile(r"^(continue|carry on|resume|keep exploring|"
                         r"continue exploring|continue karo|chalo|aage badho)$")
START_RE = re.compile(
    r"^(?:explore|explore the house|explore the home|explore the room|"
    r"start exploring|start exploration|go explore|explore karo|"
    r"ghar explore karo|ghoom ke dekho)(?: for (?P<n>\d+) (?:minutes?|mins?|min))?$")
# Matched AFTER normalisation, where "one" has already become "1".
NEW_RE = re.compile(r"^(?:explore|new|(?:start )?(?:a )?new (?:one|1)|start new|"
                    r"start again|naya|naya wala|explore again)$")
GO_TO_RES = (re.compile(r"^go to (?P<x>.+)$"),
             re.compile(r"^(?P<x>.+?) (?:pe|par|mein|me) jao$"),
             re.compile(r"^(?P<x>.+?) chalo$"))
_AFTER = re.compile(r" (?:and|aur|then) .*$")
RENAME_RES = (re.compile(r"^(?:rename|change) (?P<x>.+?) to (?P<y>.+)$"),
              re.compile(r"^(?P<x>.+?) ka naam (?P<y>.+?) "
                         r"(?:kar do|kardo|karo|kr do|krdo|rakho|rakh do)$"))
FORGET_RES = (re.compile(r"^(?:forget|delete) (?P<x>.+)$"),
              re.compile(r"^(?P<x>.+?) (?:ko )?(?:bhool jao|bhul jao)$"))
LIST_RE = re.compile(r"^(what have you saved|list areas|list the areas|"
                     r"which rooms do you know|what rooms do you know|"
                     r"kya kya save kiya|kya kya save kiya hai)$")
STATUS_RE = re.compile(r"^(status|how much have you explored|kitna explore hua|"
                       r"kitna explore hua hai)$")
SET_HOME_RE = re.compile(r"^(set home here|set the home here|mark the start here|"
                         r"set the start here|mark this as the start)$")
HALT_ANY_RE = re.compile(r"(?:^| )(stop|ruko)(?: |$)")


def _span(cased: str, norm: str, m: re.Match, group: str) -> str:
    a, b = m.span(group)
    return cased[a:b] if len(cased) == len(norm) else norm[a:b]


def candidates(text: str) -> list[Intent]:
    """Every intent whose pattern matches, in the spec's check order. State is
    not considered here; Router.handle decides which one, if any, claims it."""
    norm = normalize(text)
    if not norm:
        return []
    cased = normalize_cased(text)
    out: list[Intent] = []
    if HALT_RE.match(norm):
        out.append(Intent("HALT"))
    if FINISH_RE.match(norm):
        out.append(Intent("FINISH"))
    if CONTINUE_RE.match(norm):
        out.append(Intent("CONTINUE"))
    # Only a sentence that ORDERS a name labels an area here. "It's good."
    # and "That's it." are statements; they name an area only as the answer
    # to her question (label_dialog), never out of the blue.
    slot = name_slot(text, declarative=False)
    if slot is not None:
        try:
            name, display = area_store.validate_name(slot[1])
            out.append(Intent("LABEL_NOW", {"name": name, "display": display}))
        except area_store.InvalidName:
            pass                         # "this is home" -> SET_HOME, further down
    for rx in GO_TO_RES:
        m = rx.match(norm)
        if m:
            # No post-arrival actions offline: X ends at " and " / " aur " /
            # " then ", and the rest is ignored.
            x = _AFTER.sub("", m.group("x")).strip() or m.group("x")
            out.append(Intent("GO_TO", {"name": area_store.normalize_name(x)}))
            break
    for rx in RENAME_RES:
        m = rx.match(norm)
        if m:
            out.append(Intent("RENAME", {
                "name": area_store.normalize_name(m.group("x")),
                "new": _span(cased, norm, m, "y")}))
            break
    for rx in FORGET_RES:
        m = rx.match(norm)
        if m:
            out.append(Intent("FORGET", {"name": area_store.normalize_name(m.group("x"))}))
            break
    if LIST_RE.match(norm):
        out.append(Intent("LIST"))
    if STATUS_RE.match(norm):
        out.append(Intent("STATUS", {"text": norm}))
    if SET_HOME_RE.match(norm):
        out.append(Intent("SET_HOME"))
    m = START_RE.match(norm)
    if m:
        out.append(Intent("START", {"minutes": int(m.group("n")) if m.group("n")
                                    else None}))
    if HALT_ANY_RE.search(norm) and not any(i.name == "HALT" for i in out):
        out.append(Intent("HALT", {"loose": True}))
    return out


def classify(text: str) -> Intent | None:
    """The first matching intent, by pattern alone (for tests and logs)."""
    found = candidates(text)
    return found[0] if found else None


# --- the router ---------------------------------------------------------------------

class Router:
    """Decides, then delegates to `actions` (the mission manager):

        halt()  finish()  resume()  start(minutes, fresh)  label_now(name, display)
        go_to(name, target)  rename(area, new_display) -> bool  forget(area)
        say_list()  say_status()  set_home()  say(key, **fields)
        dialog_answer(text)

    `resolve_area(name)` -> a saved area dict or None (exact name / alias).
    `resolve_place(name)` -> a File 14 landmark dict or None (exact).
    """

    _NOT_OURS = object()

    def __init__(self, actions, resolve_area: Callable[[str], dict | None],
                 resolve_place: Callable[[str], dict | None] | None = None,
                 monotonic: Callable[[], float] = time.monotonic,
                 confirm_ttl_s: float = CONFIRM_TTL_S):
        self.actions = actions
        self.resolve_area = resolve_area
        self.resolve_place = resolve_place or (lambda name: None)
        self.monotonic = monotonic
        self.confirm_ttl_s = confirm_ttl_s
        self.pending: dict | None = None

    # -- a question of the router's own that is waiting for yes / no --
    def _ask(self, kind: str, **data) -> None:
        self.pending = dict(data, kind=kind, at=self.monotonic())

    def _pending_live(self) -> dict | None:
        if self.pending and self.monotonic() - self.pending["at"] > self.confirm_ttl_s:
            self.pending = None
        return self.pending

    def _answer(self, text: str, ctx: Context) -> bool | None:
        """True/False = the waiting question consumed it; None = it was not an
        answer, so the question is dropped and the words are routed normally."""
        p, norm = self.pending, normalize(text)
        self.pending = None
        if p["kind"] == "forget":
            if yes_no(text) is True:
                self.actions.forget(p["area"])
                self.actions.say("forgot", name=p["area"]["display_name"])
                return True
            return True if yes_no(text) is False else None
        if p["kind"] == "start":
            if CONTINUE_RE.match(norm) or yes_no(text) is True:
                self.actions.resume()
                return True
            if NEW_RE.match(norm) or START_RE.match(norm):
                self.actions.start(p.get("minutes"), True)
                return True
            return True if yes_no(text) is False else None
        return None

    def _target(self, name: str) -> dict | None:
        area = self.resolve_area(name)
        if area is not None:
            return {"kind": "area", "name": area["name"], "x": area["center_x"],
                    "y": area["center_y"], "radius_m": area["radius_m"],
                    "area": area}
        for candidate in (name, f"the {name}"):
            place = self.resolve_place(candidate)
            if place is not None:
                return {"kind": "landmark", "name": place["name"], "x": place["x"],
                        "y": place["y"], "radius_m": place.get("radius_m")}
        return None

    def _cant(self) -> bool:
        self.actions.say("cant_now")
        return True

    # -- one candidate intent against the state --
    def _decide(self, it: Intent, ctx: Context):
        a, st, active = self.actions, ctx.state, ctx.mission_active
        n = it.name
        if n == "HALT":
            if active or ctx.moving:
                a.halt()
                return True
            return self._NOT_OURS            # nothing of ours to stop: the existing stop
        if n == "FINISH":
            if st in ("EXPLORE", "PAUSED", "HELP_WAIT"):
                a.finish()
                return True
            return self._cant() if active else self._NOT_OURS
        if n == "CONTINUE":
            if st == "PAUSED" or (st == "IDLE" and ctx.interrupted):
                a.resume()
                return True
            return self._cant() if active else self._NOT_OURS
        if n == "LABEL_NOW":
            if st in ("EXPLORE", "PAUSED"):
                a.label_now(it.args["name"], it.args["display"])
                return True
            return self._cant() if active else self._NOT_OURS
        if n == "GO_TO":
            if not active:
                return self._NOT_OURS        # File 14's nav_go drives it today
            target = self._target(it.args["name"])
            if target is None:
                if st in ("EXPLORE", "PAUSED"):
                    a.say("unknown_place", name=it.args["name"])
                    return True
                return self._NOT_OURS
            if st in ("IDLE", "EXPLORE", "PAUSED"):
                if st == "IDLE" and not ctx.localized:
                    a.say("cant_start")
                    return True
                a.go_to(target["name"], target)
                return True
            return self._cant()
        if n in ("RENAME", "FORGET"):
            area = self.resolve_area(it.args["name"])
            if area is None:
                return self._NOT_OURS
            if ctx.moving:
                return self._cant()
            if n == "FORGET":
                a.say("forget_ask", name=area["display_name"])
                self._ask("forget", area=area)
                return True
            if a.rename(area, it.args["new"]):
                return True
            return self._cant()
        if n == "LIST":
            a.say_list()
            return True
        if n == "STATUS":
            if not active and it.args.get("text") == "status":
                return self._NOT_OURS        # offline.py's robot status today
            a.say_status()
            return True
        if n == "SET_HOME":
            if st in ("IDLE", "PAUSED"):
                a.set_home()
                return True
            return self._cant()
        if n == "START":
            if active:
                return self._cant()
            # Still booting / relocalising counts as not localized: refused
            # here, never handed on to start a run with no mission round it.
            if st != "IDLE" or not ctx.localized:
                a.say("cant_start")
                return True
            if ctx.interrupted:
                a.say("continue_or_new")
                self._ask("start", minutes=it.args.get("minutes"))
                return True
            a.start(it.args.get("minutes"), False)
            return True
        return None

    def handle(self, text: str, ctx: Context) -> bool:
        if not normalize(text):
            return False
        if self._pending_live() is not None:
            norm = normalize(text)
            if HALT_RE.match(norm) or FINISH_RE.match(norm):
                self.pending = None              # they still work; the question is dropped
            else:
                answered = self._answer(text, ctx)
                if answered is not None:
                    return answered
        if ctx.dialog:
            norm = normalize(text)
            if HALT_RE.match(norm):
                self.actions.halt()
            elif FINISH_RE.match(norm):
                self.actions.finish()
            else:
                self.actions.dialog_answer(text)
            return True
        for intent in candidates(text):
            verdict = self._decide(intent, ctx)
            if verdict is self._NOT_OURS:
                return False
            if verdict is not None:
                return bool(verdict)
        return False
