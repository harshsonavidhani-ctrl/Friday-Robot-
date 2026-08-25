"""apps/alarm.py — the view that means WAKE UP, and the only thing that rings.

Both an alarm and a finished timer land here, because from the room's point of
view they are the same event: something you asked for earlier is happening
now, and it will keep asking until you answer it.

    app_control.launch("alarm", label="school", at=..., item_id="alarm1")

Answering it is dismiss(): a touch anywhere on the panel, or "stop" spoken to
Friday. It gives up on its own after APP_ALARM_MAX_RING_S rather than ringing
into an empty house forever.

This is the one app marked URGENT, so it keeps the screen even while a safety
expression is up. Everything else yields to the face.
"""

from __future__ import annotations

import math

import config

import apps
from apps.base import App

from . import ring


@apps.register
class Alarm(App):
    NAME = "alarm"
    URGENT = True

    def setup(self, label: str = "", at: float = 0.0, kind: str = "alarm",
              item_id: str = "", silent: bool = False) -> None:
        self.label = (label or "").upper()
        self.at = float(at or 0.0)
        self.kind = kind if kind in ("alarm", "timer") else "alarm"
        self.item_id = item_id
        self.silent = bool(silent)          # for tests and for the demo
        self._rotations: dict[int, object] = {}
        if not self.silent:
            ring.start()

    def tick(self, dt: float) -> None:
        if self.elapsed >= config.APP_ALARM_MAX_RING_S:
            self.done = True

    def dismiss(self) -> bool:
        ring.stop()
        self.done = True
        return True

    def close(self) -> None:
        ring.stop()

    # --- drawing -----------------------------------------------------

    def draw(self, surface) -> None:
        paint = self.paint
        surface.fill((0, 0, 0))
        cx = self.width // 2

        # The whole view breathes on the ring's own rhythm, so the picture
        # and the sound are obviously the same event.
        beat = 0.5 + 0.5 * math.sin(2 * math.pi * self.elapsed / 1.1)

        self._draw_bell(surface, (cx, int(self.height * 0.40)), beat)

        heading = (app_time(self.at) if self.kind == "alarm" else "TIME UP")
        paint.text(surface, heading, paint.px(0.16),
                   (cx, int(self.height * 0.74)))
        if self.label:
            paint.text(surface, self.label, paint.px(0.07),
                       (cx, int(self.height * 0.86)), alpha=210)

        # The way out, stated plainly and pulsing so it is not missed.
        paint.text(surface, "TOUCH OR SAY STOP", paint.px(0.045),
                   (cx, int(self.height * 0.95)),
                   alpha=int(120 + 100 * beat))

    def _draw_bell(self, surface, center, beat: float) -> None:
        paint = self.paint
        size = paint.px(0.34)
        art = paint.stencil("bell", (size, size), _paint_bell)

        # Swings left and right. The angle is QUANTISED to 4 degrees and the
        # rotations are cached: rotating a lit surface every frame is exactly
        # the per-frame cost this Pi cannot afford.
        angle = int(round(14 * math.sin(2 * math.pi * self.elapsed / 0.55)
                          / 4.0)) * 4
        swung = self._rotations.get(angle)
        if swung is None:
            import pygame
            swung = self._rotations[angle] = pygame.transform.rotate(art, angle)
        surface.blit(swung, swung.get_rect(center=center))

        # Two sound waves either side, riding the same beat. Both are 72
        # degrees wide and centred on 3 and 9 o'clock, so they mirror.
        span = 2 * math.pi / 5
        for middle in (math.pi / 2, 3 * math.pi / 2):
            for index, scale in enumerate((0.62, 0.80)):
                radius = int(size * scale * (0.95 + 0.12 * beat))
                paint.arc(surface, center, radius, max(3, paint.px(0.009)),
                          middle - span / 2, span,
                          alpha=int((150 - index * 45) * (0.4 + 0.6 * beat)))


def _paint_bell(mask) -> None:
    """The silhouette: a flared bell, its rim, and the clapper under it."""
    import pygame
    w, h = mask.get_size()
    body = [
        (w * 0.50, h * 0.10),                     # crown
        (w * 0.74, h * 0.24), (w * 0.80, h * 0.60),
        (w * 0.88, h * 0.74), (w * 0.12, h * 0.74),
        (w * 0.20, h * 0.60), (w * 0.26, h * 0.24),
    ]
    pygame.draw.polygon(mask, (255, 255, 255), body)
    pygame.draw.rect(mask, (255, 255, 255),
                     pygame.Rect(int(w * 0.08), int(h * 0.74),
                                 int(w * 0.84), int(h * 0.08)))
    pygame.draw.circle(mask, (255, 255, 255),
                       (int(w * 0.50), int(h * 0.92)), int(w * 0.09))
    # The handle on top, so it reads as a hand bell and not a lampshade.
    # It overlaps the crown by design -- a gap there reads as two objects.
    pygame.draw.rect(mask, (255, 255, 255),
                     pygame.Rect(int(w * 0.42), int(h * 0.00),
                                 int(w * 0.16), int(h * 0.13)))


def app_time(epoch: float) -> str:
    """The alarm's own time, big. Falls back to now if it was not given."""
    from datetime import datetime
    when = datetime.fromtimestamp(epoch) if epoch else datetime.now()
    return when.strftime("%H:%M" if config.APP_CLOCK_24H else "%-I:%M %p")
