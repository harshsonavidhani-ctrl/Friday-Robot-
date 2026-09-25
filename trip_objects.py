"""trip_objects.py — what she saw while exploring: collected, remembered, said.

    seen = TripSightings(remember=..., camera_reason=...)
    seen.poll()                 # every couple of seconds during the trip
    seen.finish()               # -> the sentence she says, e.g.
    "On this trip I saw a chair, two bottles and a television."

Harsh, 2026-09-25: "the offline model uses cameras to see objects and remember
the objects it saw and also report and speak what all it saw during
exploration". Offline, no model: the camera and the detector are the ones
scene.py already runs (the up camera, SSD-MobileNet, every SCENE_PERIOD_S),
and this only READS what it writes. The sentence is built from strings.

WHY IT READS scene.json AND NOTHING ELSE. scene.py owns the camera and is
careful with it: it never takes it from the dashboard stream Harsh watches on
his phone, or from an autonomous run. A second reader of the camera during a
trip would have to relearn all of that. scene.json keeps only the LATEST
frame, so a trip has to collect as it goes; `taken_at` says whether a frame is
new, and nothing taken before the trip began is counted.

ONE PICTURE IS NOT A SIGHTING. The detector has named a "stop sign" in the
hostel room and, on its first live tick, an "aeroplane" filling half the
frame. So a thing counts once it is named in at least CONFIRM_FRAMES pictures,
or once with a score of STRONG_SCORE or more. "How many" is the most seen in
any ONE picture, never the sum: one chair seen in ten pictures is one chair.

IF THE CAMERA GAVE HER NOTHING, SHE SAYS SO. The commonest reason is the
dashboard streaming the up camera, where scene.py skips every tick; "I saw
nothing" would be false.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import config

log = logging.getLogger("friday.trip_objects")

CONFIRM_FRAMES = 2
STRONG_SCORE = 0.80
# The memory note's kind, so "what did you see when you explored" can find
# every trip's note without matching on its wording.
MEMORY_KIND = "exploration_sighting"

_NUMBERS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine"}
_PLURAL = {"person": "people", "mouse": "mice", "knife": "knives",
           "sofa": "sofas", "bus": "buses", "wine glass": "wine glasses",
           "sandwich": "sandwiches", "hot dog": "hot dogs",
           "toothbrush": "toothbrushes", "skis": "pairs of skis",
           "scissors": "pairs of scissors", "sheep": "sheep"}
# What the detector calls something is not always what a person would say.
_SPOKEN = {"sofa": "sofa", "aeroplane": "aeroplane",
           "dining table": "table", "mobile phone": "phone",
           "sports ball": "ball", "hair drier": "hair drier"}


def spoken(label: str, count: int) -> str:
    """ "a chair", "an apple", "two bottles", "three people"."""
    name = _SPOKEN.get(label, label)
    if count <= 1:
        if name in ("skis", "scissors"):
            return f"a pair of {name}"
        return ("an " if name[:1] in "aeiou" else "a ") + name
    plural = _PLURAL.get(name, name + "s")
    many = _NUMBERS.get(count, "lots of")
    return f"{many} {plural}"


def listing(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def _save_note(text: str) -> None:
    import memory
    memory.remember(text, kind=MEMORY_KIND, source="exploration")


class TripSightings:
    """Everything the detector named during one trip."""

    def __init__(self, scene_json: Path | None = None,
                 remember: Callable[[str], None] | None = None,
                 camera_reason: Callable[[], str | None] | None = None,
                 report_file: Path | None = None,
                 now: float | None = None) -> None:
        self.scene_json = Path(scene_json or config.SCENE_JSON)
        self._remember = remember if remember is not None else _save_note
        self._camera_reason = camera_reason or (lambda: None)
        self.report_file = report_file
        self.started_at = time.time() if now is None else float(now)
        self._last_taken = self.started_at
        self.frames = 0
        self.frames_seen: dict[str, int] = {}
        self.most_at_once: dict[str, int] = {}
        self.best_score: dict[str, float] = {}
        self.camera_note: str | None = None

    # --- during the trip ---------------------------------------------------

    def poll(self) -> bool:
        """Count the latest frame if it is new. True if it was."""
        try:
            data = json.loads(self.scene_json.read_text())
            taken = float(data.get("taken_at", 0.0))
        except (OSError, ValueError, TypeError):
            return False
        if taken <= self._last_taken:
            return False
        self._last_taken = taken
        self.frames += 1
        here: dict[str, int] = {}
        for obj in data.get("objects") or []:
            label = str(obj.get("label") or "").strip()
            if not label:
                continue
            here[label] = here.get(label, 0) + 1
            score = float(obj.get("score") or 0.0)
            self.best_score[label] = max(self.best_score.get(label, 0.0),
                                         score)
        for label, n in here.items():
            self.frames_seen[label] = self.frames_seen.get(label, 0) + 1
            self.most_at_once[label] = max(self.most_at_once.get(label, 0), n)
        return True

    def note_camera(self) -> None:
        """Remember why the camera is quiet, while it still is."""
        if self.camera_note is None:
            try:
                self.camera_note = self._camera_reason()
            except Exception:                    # noqa: BLE001
                self.camera_note = None

    # --- the answer ------------------------------------------------------

    def confirmed(self) -> list[tuple[str, int]]:
        """(label, how many) she can honestly say she saw, most-seen first."""
        out = [(label, self.most_at_once[label])
               for label, n in self.frames_seen.items()
               if n >= CONFIRM_FRAMES
               or self.best_score.get(label, 0.0) >= STRONG_SCORE]
        out.sort(key=lambda item: (-self.frames_seen[item[0]], item[0]))
        return out

    def sentence(self) -> str:
        if self.frames == 0:
            if self.camera_note and "dashboard" in self.camera_note:
                return ("I could not use my camera on this trip, because the "
                        "dashboard was showing it, so I cannot say what I "
                        "saw.")
            return ("My camera did not give me any pictures on this trip, so "
                    "I cannot say what I saw.")
        seen = self.confirmed()
        if not seen:
            return "I did not spot anything I know the name of on this trip."
        return ("On this trip I saw "
                + listing([spoken(label, n) for label, n in seen]) + ".")

    def as_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "frames": self.frames,
            "seen": [{"label": label, "count": n,
                      "pictures": self.frames_seen[label],
                      "best_score": round(self.best_score.get(label, 0.0), 2)}
                     for label, n in self.confirmed()],
            "glimpsed": sorted(label for label in self.frames_seen
                               if label not in dict(self.confirmed())),
            "camera": self.camera_note,
        }

    def finish(self) -> str:
        """Final poll, then remember what was seen. Returns what to say."""
        self.poll()
        if self.frames == 0:
            self.note_camera()
        said = self.sentence()
        seen = self.confirmed()
        if seen:
            when = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(self.started_at))
            note = (f"When I explored on {when} I saw "
                    + listing([spoken(label, n) for label, n in seen]) + ".")
            try:
                self._remember(note)
            except Exception as exc:             # noqa: BLE001
                # Forgetting must never become a trip that failed.
                log.warning("could not remember what I saw: %s", exc)
        if self.report_file is not None:
            try:
                self.report_file.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.report_file.with_name(f".{self.report_file.name}.tmp")
                tmp.write_text(json.dumps(dict(self.as_dict(), said=said),
                                          indent=2))
                tmp.replace(self.report_file)
            except OSError as exc:
                log.warning("could not write the trip's sightings: %s", exc)
        log.info("trip sightings: %s", self.as_dict())
        return said
