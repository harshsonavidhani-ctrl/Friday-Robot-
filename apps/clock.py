"""apps/clock.py — the time, and what Friday shows when nobody is there.

    app_control.launch("clock")                  # on command
    app_control.launch("clock", screensaver=True)  # after a long idle

Digital because that is what gets read at a glance across a room, with one
analog element: a ring that fills once a minute, so the panel is visibly
alive even when the digits have not changed for fifty seconds.

As a screensaver it draws dimmer and never finishes by itself -- the face
takes the screen back the moment anything happens.
"""

from __future__ import annotations

import math
import time
from datetime import datetime

import config

import apps
from apps.base import DIM_COLOR, App


@apps.register
class Clock(App):
    NAME = "clock"

    def setup(self, screensaver: bool = False, hold: float = 0.0) -> None:
        self.screensaver = bool(screensaver)
        self.hold = float(hold)
        self.alpha = 150 if self.screensaver else 255

    def tick(self, dt: float) -> None:
        if self.hold and self.elapsed >= self.hold:
            self.done = True

    def draw(self, surface) -> None:
        paint = self.paint
        surface.fill((0, 0, 0))
        now = datetime.now()
        cx, cy = self.width // 2, int(self.height * 0.48)

        # The minute ring, drawn first so the digits sit inside it. Every
        # line below is placed to clear it -- text crossing the ring reads
        # as a mistake from two metres away, which is the viewing distance.
        radius = paint.px(0.36)
        thickness = paint.px(0.014)
        fraction = (now.second + now.microsecond / 1e6) / 60.0
        paint.arc(surface, (cx, cy), radius, thickness, 0, 2 * math.pi,
                  color=DIM_COLOR, alpha=self.alpha)
        paint.arc(surface, (cx, cy), radius, thickness, 0,
                  2 * math.pi * fraction, alpha=self.alpha)

        fmt = "%H:%M" if config.APP_CLOCK_24H else "%-I:%M"
        # On a 12-hour clock the digits shift left by half of what AM/PM
        # takes, so the PAIR is centred in the ring rather than the digits
        # alone with a tag hanging over the edge.
        shift = 0 if config.APP_CLOCK_24H else paint.px(0.05)
        digits = paint.text(surface, now.strftime(fmt), paint.px(0.30),
                            (cx - shift, cy), alpha=self.alpha)
        if not config.APP_CLOCK_24H:
            # Hung off the digits' own box rather than a guessed offset, so
            # 9:05 and 12:45 both sit right.
            paint.text(surface, now.strftime("%p"), paint.px(0.075),
                       (digits.right + paint.px(0.055),
                        digits.bottom - paint.px(0.045)), alpha=self.alpha)

        paint.text(surface, now.strftime("%A %-d %B").upper(),
                   paint.px(0.055), (cx, int(self.height * 0.71)),
                   alpha=int(self.alpha * 0.8))

        # What is coming, if anything is. A clock that knows about the 7am
        # alarm is worth more than one that only knows the time.
        soon = _next_line()
        if soon and not self.screensaver:
            paint.text(surface, soon, paint.px(0.045),
                       (cx, int(self.height * 0.95)),
                       alpha=int(self.alpha * 0.7))

    def launched_message(self) -> str:
        fmt = "%H:%M" if config.APP_CLOCK_24H else "%-I:%M %p"
        return f"it is {datetime.now().strftime(fmt)}"


_next_cache: tuple[float, str] = (0.0, "")


def _next_line() -> str:
    """The soonest alarm or timer, re-read at most once a second -- this runs
    every frame and reading a file 30 times a second for nothing is exactly
    the kind of cost this Pi cannot pay."""
    global _next_cache
    now = time.monotonic()
    if now - _next_cache[0] < 1.0:
        return _next_cache[1]
    import app_control
    items = app_control.pending()
    line = ""
    if items:
        item = items[0]
        label = f" — {item['label']}" if item.get("label") else ""
        line = f"{item['kind']} {app_control.say_time(item['at'])}{label}"
    _next_cache = (now, line.upper())
    return _next_cache[1]
