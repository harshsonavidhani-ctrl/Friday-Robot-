"""scene_bridge.py — something new came into view, and Friday says so.

File 08 step 7: "on a new/unrecognized object appearing, feed that into the
brain's context so it can react (curious emotion, a short comment). Throttle
this — do not send a Haiku call on every camera frame."

WHERE THE DETECTIONS COME FROM. There is no get_visible_objects(). The scene
watcher (scene.py, inside the MCP server, which owns the cameras) runs the
SSD-MobileNet detector on the up camera every SCENE_PERIOD_S and writes
state/scene.json. This reads that file and nothing else: no camera, no model,
no second detector, no CPU worth measuring.

WHY IT IS A WAKE SOURCE. main.py's loop is `for trigger in source.listen()`, and
the brain may only be used from inside it -- it is a blocking generator in an
async function, so anything running beside it starves (see
brain.warm_in_background). Arriving as a trigger through the same AnyOf as the
wake word means a remark is queued behind Harsh's turn and never on top of it,
and pausing the detector for a turn pauses this too.

WHAT "MEANINGFULLY NEW" MEANS -- all of these, or no call is made:
  * a label she has NOT seen in the last SCENE_BRIDGE_FORGET_S of watching. The
    first tick after start (or after a gap in watching) is only the baseline:
    the furniture that was already there is not news.
  * seen on SCENE_BRIDGE_CONFIRM_TICKS ticks in a row at SCENE_BRIDGE_MIN_SCORE
    or better. The detector flickers; one frame of "aeroplane" is not a plane.
  * not in SCENE_BRIDGE_IGNORE (a person is somebody, not something).
  * SCENE_BRIDGE_COOLDOWN_S since her last remark, and QUIET_AFTER_TURN_S since
    Harsh last spoke to her.
  * someone plausibly there to hear it: Harsh spoke to her, or a person was in
    view, within SCENE_BRIDGE_AUDIENCE_S. A robot remarking on a cup to an empty
    room at 3 am is a bug, not curiosity.
A label that fails a gate is still LEARNED -- it will not come back as "new"
the moment the cooldown ends. So the worst case is one Haiku call per cooldown,
and the usual case is none.

THE BRAIN SEES THE PICTURE, not just the label: brain.Brain.remark() gets the
frame the detection came from, so a wrong label is caught by looking and
answered with IGNORE rather than said out loud.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import config
from wake_word import WakeSource

log = logging.getLogger("friday.scene_bridge")

NAME = "scene"


@dataclass
class Novelty:
    """The rules above, with the clock passed in so they can be tested.

        novelty = Novelty()
        new = novelty.tick(objects, taken_at)     # labels newly confirmed
    """

    seen: dict[str, float] = field(default_factory=dict)   # label -> last seen
    streak: dict[str, int] = field(default_factory=dict)   # label -> ticks in a row
    last_tick: float | None = None
    last_person: float | None = None

    def tick(self, objects: list[dict[str, Any]], taken_at: float) -> list[dict]:
        """One scene.json reading. Returns the NEW objects, biggest first."""
        min_score = float(getattr(config, "SCENE_BRIDGE_MIN_SCORE", 0.6))
        forget = float(getattr(config, "SCENE_BRIDGE_FORGET_S", 1800.0))
        confirm = int(getattr(config, "SCENE_BRIDGE_CONFIRM_TICKS", 2))
        ignore = set(getattr(config, "SCENE_BRIDGE_IGNORE", ("person",)))
        gap = float(getattr(config, "SCENE_BRIDGE_RESEED_GAP_S", 120.0))

        kept = [o for o in objects
                if o.get("label") and float(o.get("score", 0)) >= min_score]
        labels = {o["label"] for o in kept}
        if "person" in labels:
            self.last_person = taken_at

        # THE BASELINE. First look after starting, or after not looking for a
        # while (the dashboard held the camera, a run took it): whatever is in
        # view now is the room, not news.
        baseline = self.last_tick is None or taken_at - self.last_tick > gap
        self.last_tick = taken_at
        if baseline:
            for label in labels:
                self.seen[label] = taken_at
            self.streak = {label: 1 for label in labels}
            return []

        self.streak = {label: self.streak.get(label, 0) + 1 for label in labels}
        new: list[dict] = []
        for obj in kept:                      # scene.py sorts biggest first
            label = obj["label"]
            familiar = (label in self.seen
                        and taken_at - self.seen[label] <= forget)
            if familiar or label in ignore:
                self.seen[label] = taken_at
            elif self.streak[label] >= confirm:
                self.seen[label] = taken_at   # decided once; familiar from now
                if all(n["label"] != label for n in new):
                    new.append(obj)
            # else: a candidate still waiting for its next tick. Deliberately
            # NOT marked seen, or it could never be confirmed.
        return new


class NewObjectWatch(WakeSource):
    """Yields `json` describing one new object, at most once per cooldown."""

    name = NAME

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or config.SCENE_JSON)
        self.novelty = Novelty()
        self._closed = threading.Event()
        self._paused = False
        self._asleep_until = 0.0
        self._last_remark = 0.0          # time.time()
        self._last_heard = 0.0           # time.time(): Harsh's last turn ended

    # -- told by main.py ---------------------------------------------------

    def heard_harsh(self) -> None:
        """A voice turn just ended: someone is here, and wants a moment."""
        self._last_heard = time.time()

    # -- WakeSource ----------------------------------------------------------

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
        log.info("watching %s for new objects (cooldown %.0fs, confirm %d "
                 "ticks, ignoring %s)", self.path.name,
                 float(getattr(config, "SCENE_BRIDGE_COOLDOWN_S", 180.0)),
                 int(getattr(config, "SCENE_BRIDGE_CONFIRM_TICKS", 2)),
                 ", ".join(getattr(config, "SCENE_BRIDGE_IGNORE", ())) or "nothing")
        last_taken = None
        while not self._closed.wait(float(
                getattr(config, "SCENE_BRIDGE_POLL_S", 2.0))):
            try:
                reading = json.loads(self.path.read_text())
                taken_at = float(reading["taken_at"])
            except (OSError, ValueError, KeyError, TypeError):
                continue
            if taken_at == last_taken:
                continue
            last_taken = taken_at
            if time.time() - taken_at > float(config.SCENE_MAX_AGE_S):
                continue                  # an old file is not what is in view
            new = self.novelty.tick(reading.get("objects") or [], taken_at)
            if not new:
                continue
            why_not = self._why_not_now(taken_at)
            labels = ", ".join(o["label"] for o in new)
            if why_not:
                # Learned anyway (Novelty already marked it seen): it will not
                # come back as new when the gate opens.
                log.info("new in view: %s — not remarking (%s)", labels, why_not)
                continue
            self._last_remark = time.time()
            log.info("new in view: %s — asking the brain", labels)
            first = new[0]
            yield json.dumps({
                "label": first["label"],
                "side": first.get("side", ""),
                "score": first.get("score"),
                "also": [o["label"] for o in new[1:]],
                "taken_at": taken_at,
            })

    def _why_not_now(self, taken_at: float) -> str | None:
        now = time.time()
        if self._paused:
            return "mid-turn"
        if self.sleeping_for() > 0:
            return "hushed"
        cooldown = float(getattr(config, "SCENE_BRIDGE_COOLDOWN_S", 180.0))
        if now - self._last_remark < cooldown:
            return f"cooldown, {cooldown - (now - self._last_remark):.0f}s left"
        quiet = float(getattr(config, "SCENE_BRIDGE_QUIET_AFTER_TURN_S", 20.0))
        if now - self._last_heard < quiet:
            return "Harsh just spoke"
        audience = float(getattr(config, "SCENE_BRIDGE_AUDIENCE_S", 900.0))
        person = self.novelty.last_person or 0.0
        if now - self._last_heard > audience and taken_at - person > audience:
            return "nobody around to hear it"
        return None


def parse_trigger(trigger: str) -> dict | None:
    """The object a `scene: {...}` trigger describes, or None if it is not one."""
    prefix = f"{NAME}: "
    if not trigger.startswith(prefix):
        return None
    try:
        seen = json.loads(trigger[len(prefix):])
    except ValueError:
        return None
    return seen if isinstance(seen, dict) and seen.get("label") else None


def event_text(seen: dict) -> str:
    """What the brain is told. Nobody asked it anything -- it is noticing."""
    where = f", {seen['side']}" if seen.get("side") else ""
    also = (f" (also new: {', '.join(seen['also'])})"
            if seen.get("also") else "")
    return (
        f"[Nobody spoke to you. Your up camera just noticed something that was "
        f"not around before: {seen['label']}{where}{also}. That name comes from "
        f"an object detector and can be wrong — the picture is what is really "
        f"there. If it is worth remarking on, react out loud in ONE short, "
        f"natural sentence, the way a curious friend would: a comment or a "
        f"light question, not a report, and no numbers. If the picture does not "
        f"show that, or a remark would be odd or unwelcome, reply IGNORE.]")


def frame_blocks(max_age_s: float | None = None) -> list[dict] | None:
    """The scene watcher's latest frame as an image block, if still fresh."""
    import base64

    max_age_s = float(max_age_s or config.SCENE_FRESH_IMAGE_S)
    try:
        taken_at = float(json.loads(Path(config.SCENE_JSON).read_text())["taken_at"])
        if time.time() - taken_at > max_age_s:
            return None
        data = base64.b64encode(Path(config.SCENE_IMAGE).read_bytes()).decode()
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return [{"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg",
                                         "data": data}}]
