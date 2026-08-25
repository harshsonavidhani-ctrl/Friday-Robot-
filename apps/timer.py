"""apps/timer.py — a countdown you can read from across the room.

    app_control.add_timer("10 minutes", "pasta")   # stores it
    app_control.launch("timer")                    # shows the soonest one

The app only counts. Reaching zero is the schedule's business, not the
screen's: the face watches state/schedule.json and brings up the alarm view
when the moment arrives, so a timer rings whether or not anyone left this
view open -- and whether or not the face was restarted in between.
"""

from __future__ import annotations

import math
import time

import app_control

import apps
from apps.base import DIM_COLOR, App


@apps.register
class Timer(App):
    NAME = "timer"

    def setup(self, timer_id: str = "", **_) -> None:
        self.item = _find(timer_id)
        self.missing = self.item is None
        self.total = float(self.item.get("seconds", 60)) if self.item else 0.0
        self._checked = 0.0

    def tick(self, dt: float) -> None:
        if self.missing:
            if self.elapsed > 2.5:
                self.done = True
            return
        # Cancelled from somewhere else, or fired: stop showing it.
        if time.monotonic() - self._checked > 1.0:
            self._checked = time.monotonic()
            if _find(self.item["id"]) is None:
                self.done = True
        if self.remaining <= 0:
            # The schedule launches the ring; this just gets out of the way.
            self.done = True

    @property
    def remaining(self) -> float:
        return max(0.0, self.item["at"] - time.time()) if self.item else 0.0

    def draw(self, surface) -> None:
        paint = self.paint
        surface.fill((0, 0, 0))
        cx, cy = self.width // 2, int(self.height * 0.48)

        if self.missing:
            paint.text(surface, "NO TIMER", paint.px(0.13), (cx, cy))
            return

        left = self.remaining
        radius, thickness = paint.px(0.36), paint.px(0.014)
        fraction = left / self.total if self.total else 0.0
        paint.arc(surface, (cx, cy), radius, thickness, 0, 2 * math.pi,
                  color=DIM_COLOR)
        # Drains anticlockwise from 12, the way an hourglass empties.
        paint.arc(surface, (cx, cy), radius, thickness,
                  -2 * math.pi * fraction, 2 * math.pi * fraction)

        paint.text(surface, _digits(left), paint.px(0.30), (cx, cy))
        self._label(surface, self.item.get("label", "").upper(), 0.71)

    def launched_message(self) -> str:
        if self.missing:
            return "there is no timer running"
        return f"{app_control.say_duration(self.remaining)} left"


def _digits(seconds: float) -> str:
    """M:SS under an hour, H:MM:SS over it. Rounded UP, so a timer never
    shows 0:00 while it is still going."""
    total = int(math.ceil(max(0.0, seconds)))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _find(timer_id: str = "") -> dict | None:
    """The named timer, or the soonest one if no name was given."""
    timers = [i for i in app_control.pending() if i["kind"] == "timer"]
    if not timers:
        return None
    if not timer_id:
        return timers[0]
    return next((t for t in timers if t["id"] == timer_id), None)
