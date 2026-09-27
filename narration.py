"""narration.py — File 15 steps 3-7: "Friday, keep describing what you see".

    narration.set_mode(True, by="voice")      # his words, or the dashboard
    narration.mode()                          # {"on", "since", "until", "by"}
    narration.panel()                         # what the dashboard shows

THE SWITCH (step 5). state/narration.json, written by the voice intent
(offline.py -> main._run_local), the dashboard button, the timeout, and the
boot reset -- nothing else. OFF on every start of friday-brain (`boot_reset`),
and it NEVER turns itself on: no brain tool can set it, only Harsh's own words
or his finger. NARRATION_TIMEOUT_S after he turned it on it turns itself OFF,
and says so, in case he forgot.

TWO TIERS (steps 3-4), both inside NarrationWatch.listen(), on its own thread:
  Tier 1 -- free. Every NARRATION_POLL_S, the scene watcher's latest reading
    (state/scene.json + scene_latest.jpg, written by scene.py inside the MCP
    server) is compared with the BASELINE: the reading she last described. A
    change is a label that came or went, a label that moved, or a big enough
    share of the picture that changed (see config for why the pixel test is
    needed in this room). No model, no camera -- two file reads.
  Tier 2 -- paid, and only on a Tier 1 change. ONE vision call with the new
    frame and the last thing she said: NO_CHANGE, or only what is new
    (scene_describe.narrate). The baseline moves to the new reading either
    way, so the same change is never paid for twice.

WHY A WAKE SOURCE (step 6). What Tier 2 says reaches main.py's loop the same
way "hey Friday" does, so it can only be spoken BETWEEN turns, never on top
of one -- the same reason scene_bridge is one. The paid call itself runs here,
on this source's own pump thread, so the loop never waits on it. Before the
call, the no-interrupt gate (initiative.py -- the spec's "file 11") must say
nobody is talking; while it does not, the change stays PENDING and is looked
at again next poll. A description that went stale waiting in the queue is
dropped by main.py and handed back (`missed`) so the change is re-checked
against a fresh frame rather than lost.

The camera handle is never touched here: scene.py owns the only one.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import config
from wake_word import WakeSource

log = logging.getLogger("friday.narration")

NAME = "narration"

TIMEOUT_LINE = ("I've been describing for a while, so I'll stop now. Say keep "
                "describing if you want more.")


# --- the switch -------------------------------------------------------------

def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def _read(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def mode(now: float | None = None) -> dict[str, Any]:
    """The switch as it stands. Missing or unreadable is OFF, never on.

    A switch left on past its `until` reads as off here too, so the dashboard
    shows the truth even when friday-brain is not running to turn it off.
    """
    now = time.time() if now is None else now
    raw = _read(config.NARRATION_FILE)
    on = bool(raw.get("on"))
    until = float(raw.get("until") or 0.0)
    if on and until and now >= until:
        on = False
    return {"on": on, "since": float(raw.get("since") or 0.0),
            "until": until, "by": str(raw.get("by") or ""),
            "expired": bool(raw.get("on")) and not on}


def set_mode(on: bool, by: str = "") -> dict[str, Any]:
    """Turn narration on or off. `by`: voice, dashboard, stop, timeout, boot.

    Turning it on while it is already on RESTARTS the timeout -- "keep
    describing" said again is him asking for more -- but keeps the session,
    so she does not start over with a first look.
    """
    now = time.time()
    current = mode(now)
    if on:
        since = current["since"] if current["on"] else now
        state = {"on": True, "since": since,
                 "until": now + float(config.NARRATION_TIMEOUT_S), "by": by,
                 "at": now}
    else:
        state = {"on": False, "since": current["since"], "until": 0.0,
                 "by": by, "at": now}
    try:
        _atomic_write(Path(config.NARRATION_FILE), state)
    except OSError as exc:
        log.warning("could not write the narration switch: %s", exc)
    if on != current["on"]:
        log.info("narration %s (%s)", "ON" if on else "off", by or "?")
    return mode(now)


def boot_reset() -> None:
    """Step 5: off on every start. Never raises."""
    try:
        if _read(config.NARRATION_FILE).get("on"):
            set_mode(False, by="boot")
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not reset narration at start: %s", exc)


def status() -> dict[str, Any]:
    return _read(config.NARRATION_STATUS_FILE)


def panel(now: float | None = None) -> dict[str, Any]:
    """The dashboard's whole payload: the switch plus what she is doing."""
    now = time.time() if now is None else now
    m = mode(now)
    st = status()
    fresh = now - float(st.get("at") or 0.0) <= 3 * float(
        config.NARRATION_POLL_S) + 5
    if m["on"] and not fresh:
        doing = "waiting for her voice loop (friday-brain) to pick it up"
    elif m["on"]:
        doing = st.get("status") or "watching"
    elif m["expired"] or m["by"] == "timeout":
        doing = "turned itself off after the time limit"
    else:
        doing = "off"
    session = st.get("session") == m["since"]
    return {"on": m["on"], "by": m["by"],
            "left_s": max(0.0, m["until"] - now) if m["on"] else 0.0,
            "timeout_s": float(config.NARRATION_TIMEOUT_S),
            "status": doing,
            "last_said": st.get("last_said") or "",
            "last_said_at": st.get("last_said_at") or 0.0,
            "looks": int(st.get("looks") or 0) if session else 0,
            "no_change": int(st.get("no_change") or 0) if session else 0,
            "spoken": int(st.get("spoken") or 0) if session else 0}


# --- Tier 1 -----------------------------------------------------------------

def thumb(jpeg_or_path: bytes | str | Path):
    """A 32x24 grey thumbnail as float32, or None. JPEG draft mode: ~ms."""
    try:
        import io

        import numpy as np
        from PIL import Image
        src = (io.BytesIO(jpeg_or_path) if isinstance(jpeg_or_path, bytes)
               else jpeg_or_path)
        with Image.open(src) as image:
            image.draft("L", (80, 60))
            return np.asarray(image.convert("L").resize((32, 24)),
                              dtype="float32")
    except Exception:                           # noqa: BLE001
        return None


def pixel_change(before, after) -> float:
    """Share of thumbnail cells that moved more than NARRATION_PIXEL_DELTA."""
    if before is None or after is None or before.shape != after.shape:
        return 0.0
    import numpy as np
    moved = np.abs(after - before) > float(config.NARRATION_PIXEL_DELTA)
    return float(moved.mean())


@dataclass
class Snapshot:
    """One scene reading, reduced to what Tier 1 compares."""
    taken_at: float
    labels: dict[str, dict] = field(default_factory=dict)   # label -> biggest
    counts: dict[str, int] = field(default_factory=dict)
    thumb: Any = None

    @classmethod
    def of(cls, reading: dict, image) -> "Snapshot":
        min_score = float(config.NARRATION_MIN_SCORE)
        labels: dict[str, dict] = {}
        counts: dict[str, int] = {}
        # scene.py sorts biggest first, so the first of each label is the
        # one to follow across the view.
        for obj in reading.get("objects") or []:
            label = obj.get("label")
            if not label or float(obj.get("score") or 0) < min_score:
                continue
            counts[label] = counts.get(label, 0) + 1
            labels.setdefault(label, obj)
        return cls(float(reading.get("taken_at") or 0.0), labels, counts,
                   image)


def what_changed(before: Snapshot, after: Snapshot) -> str | None:
    """Tier 1: a short account of the change, or None. Never a model."""
    parts = []
    for label in sorted(set(after.counts) - set(before.counts)):
        side = after.labels[label].get("side")
        parts.append(f"new: {label}" + (f" ({side})" if side else ""))
    for label in sorted(set(before.counts) - set(after.counts)):
        parts.append(f"gone: {label}")
    for label in sorted(set(before.counts) & set(after.counts)):
        if after.counts[label] != before.counts[label]:
            parts.append(f"{label}: {before.counts[label]} -> "
                         f"{after.counts[label]}")
            continue
        was = before.labels[label].get("bearing_deg")
        now = after.labels[label].get("bearing_deg")
        try:
            moved = abs(float(now) - float(was))
        except (TypeError, ValueError):
            continue
        if moved >= float(config.NARRATION_MOVE_DEG):
            parts.append(f"moved: {label} ({before.labels[label].get('side')}"
                         f" -> {after.labels[label].get('side')})")
    share = pixel_change(before.thumb, after.thumb)
    if share >= float(config.NARRATION_PIXEL_CHANGE):
        parts.append(f"{share:.0%} of the picture changed")
    return "; ".join(parts) or None


# --- the watch --------------------------------------------------------------

class NarrationWatch(WakeSource):
    """Yields `json` {"say", "at", "kind", "hint", "cost", "model"}.

    kind: "first" (the first look after he turned it on), "change", or
    "timeout" (the line that says she is stopping).
    """

    name = NAME

    def __init__(self) -> None:
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._paused = False
        self._asleep_until = 0.0
        self._last_heard = 0.0           # time.time(): his last turn ended
        self._backoff_until = 0.0
        self._reset()
        self._said_status = ""

    def _reset(self) -> None:
        self._session: float | None = None
        self._baseline: Snapshot | None = None
        self._last_said: str | None = None
        self._last_said_at = 0.0
        self._last_taken: float | None = None
        self._pending = False
        self._undo: tuple | None = None
        self._counts = {"looks": 0, "no_change": 0, "spoken": 0}

    # -- told by main.py -----------------------------------------------------

    def heard_harsh(self) -> None:
        self._last_heard = time.time()

    def delivered(self) -> None:
        """What was yielded was actually said."""
        with self._lock:
            self._undo = None
            self._counts["spoken"] += 1

    def missed(self) -> None:
        """What was yielded was NOT said (stale, or he started talking).

        Put the baseline back so the same change is looked at again, against
        a fresh frame, once the way is clear. Queued, not lost (step 6).
        """
        with self._lock:
            if self._undo is not None:
                self._baseline, self._last_said, self._last_said_at = self._undo
                self._undo = None
                self._pending = True
        self._write_status("a description went stale waiting; looking again")

    # -- WakeSource ------------------------------------------------------------

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def sleep(self, seconds: float) -> float:
        self._asleep_until = time.monotonic() + seconds
        return seconds

    def wake_up(self) -> None:
        self._asleep_until = 0.0

    def sleeping_for(self) -> float:
        return max(0.0, self._asleep_until - time.monotonic())

    def close(self) -> None:
        self._closed.set()

    def listen(self) -> Iterator[str]:
        log.info("narration watch ready (off until asked; poll %.0fs, "
                 "timeout %.0f min)", float(config.NARRATION_POLL_S),
                 float(config.NARRATION_TIMEOUT_S) / 60)
        while not self._closed.wait(float(config.NARRATION_POLL_S)):
            try:
                item = self.step()
            except Exception as exc:            # noqa: BLE001
                log.warning("narration step failed: %s: %s",
                            type(exc).__name__, exc)
                continue
            if item is not None:
                yield json.dumps(item)

    # -- one poll --------------------------------------------------------------

    def step(self, now: float | None = None) -> dict | None:
        """One Tier 1 look and, if it earns one, one Tier 2 call."""
        now = time.time() if now is None else now
        switch = mode(now)
        raw_on = bool(_read(config.NARRATION_FILE).get("on"))

        if not switch["on"]:
            if raw_on and switch["expired"]:
                # THE TIMEOUT. Turned off here, where it is noticed, and said
                # out loud -- a mode that quietly stops looks like a broken one.
                set_mode(False, by="timeout")
                self._reset()
                self._write_status("turned itself off after the time limit")
                return {"say": TIMEOUT_LINE, "at": now, "kind": "timeout",
                        "hint": "", "cost": 0.0, "model": ""}
            if self._session is not None:
                self._reset()
                self._write_status("off")
            return None

        with self._lock:
            if self._session != switch["since"]:
                self._reset()
                self._session = switch["since"]
                self._pending = True             # the first look is owed

        try:
            reading = json.loads(Path(config.SCENE_JSON).read_text())
            taken_at = float(reading["taken_at"])
        except (OSError, ValueError, KeyError, TypeError):
            self._write_status("no picture from the camera yet")
            return None
        age = now - taken_at
        if age > float(config.NARRATION_FRAME_MAX_AGE_S):
            self._write_status(
                f"no fresh picture ({age:.0f}s old) — the camera is busy: the "
                f"dashboard's up-camera stream or a run pauses the scene "
                f"watcher")
            return None
        if taken_at == self._last_taken and not self._pending:
            self._write_status(self._said_status)   # still alive, same frame
            return None                          # nothing new to compare
        self._last_taken = taken_at

        try:
            jpeg = Path(config.SCENE_IMAGE).read_bytes()
        except OSError:
            return None
        snap = Snapshot.of(reading, thumb(jpeg))

        first = self._baseline is None
        if first:
            change = "first look"
        else:
            change = what_changed(self._baseline, snap)
            if change is None:
                self._pending = False
                self._write_status("watching — nothing has changed")
                return None

        # THE EXECUTIVE LAYER (step 6). Not paying for words that could not be
        # said now anyway; the change stays pending and is re-checked.
        why_not = self._why_not_now(now)
        if why_not:
            self._pending = True
            self._write_status(f"saw a change ({change}) — waiting: {why_not}")
            return None

        import scene_describe
        with self._lock:
            previous = self._last_said
        text, why = scene_describe.narrate(jpeg, previous=previous,
                                           hint=change)
        if text is None:
            self._pending = True
            if not why.startswith("rate limited"):
                self._backoff_until = time.time() + float(
                    config.NARRATION_ERROR_BACKOFF_S)
            self._write_status(f"could not look ({why})")
            log.info("narration: %s — %s", change, why)
            return None

        cost = scene_describe.last_cost_usd
        self._counts["looks"] += 1
        with self._lock:
            self._undo = (self._baseline, self._last_said, self._last_said_at)
            self._baseline = snap
            self._pending = False
            if text == scene_describe.NO_CHANGE:
                self._undo = None
                self._counts["no_change"] += 1
                log.info("narration: %s — NO_CHANGE ($%.4f)", change, cost)
                self._write_status(f"checked ({change}) — nothing new to say")
                return None
            self._last_said, self._last_said_at = text, time.time()
        log.info("narration: %s — %r ($%.4f)", change, text[:80], cost)
        self._write_status("described a change")
        return {"say": text, "at": time.time(),
                "kind": "first" if first else "change", "hint": change,
                "cost": cost, "model": scene_describe.last_model}

    def _why_not_now(self, now: float) -> str | None:
        if self._paused:
            return "mid-turn"
        if self.sleeping_for() > 0:
            return "she has been told to stop listening"
        if now < self._backoff_until:
            return f"the last call failed; retrying in {self._backoff_until - now:.0f}s"
        quiet = float(config.NARRATION_QUIET_AFTER_TURN_S)
        if now - self._last_heard < quiet:
            return "he just spoke"
        try:
            import initiative                   # noqa: PLC0415
            if initiative.he_is_talking():
                return "he is talking"
            if initiative.quiet_mode():
                return "quiet mode is on (the face only)"
        except Exception:                       # noqa: BLE001
            pass
        try:
            import tts                          # noqa: PLC0415
            if tts.quiet_at() > now:
                return "she is still speaking"
        except Exception:                       # noqa: BLE001
            pass
        return None

    def _write_status(self, text: str) -> None:
        """The dashboard's line. Written when it changes, or every few polls
        so the dashboard can tell a live watch from a dead one."""
        now = time.time()
        if (text == self._said_status
                and now - getattr(self, "_status_at", 0.0)
                < 2 * float(config.NARRATION_POLL_S)):
            return
        self._said_status, self._status_at = text, now
        try:
            _atomic_write(Path(config.NARRATION_STATUS_FILE), {
                "at": now, "status": text, "session": self._session,
                "last_said": self._last_said or "",
                "last_said_at": self._last_said_at, **self._counts})
        except OSError as exc:
            log.debug("could not write the narration status: %s", exc)


def parse_trigger(trigger: str) -> dict | None:
    """The item a `narration: {...}` trigger carries, or None."""
    prefix = f"{NAME}: "
    if not trigger.startswith(prefix):
        return None
    try:
        item = json.loads(trigger[len(prefix):])
    except ValueError:
        return None
    return item if isinstance(item, dict) and item.get("say") else None
