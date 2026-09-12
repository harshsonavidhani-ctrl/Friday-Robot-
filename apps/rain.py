"""apps/rain.py — falling rain, as ambience.

    ./venv/bin/python app_control.py rain        # stays until dismissed
    ./venv/bin/python app_control.py --dismiss

Harsh, 2026-09-12, after seeing the weather card's rain icon on the panel:
"the raining effect was good pls like permanent". So the drops moved out of the
icon and across the whole screen, and they do not stop.

IT CARRIES NO NUMBERS, DELIBERATELY. The weather card next door reports the
sky and has to stay true to it; a panel reading "24 RAIN" under a clear sky is
a robot lying quietly, and at a demo somebody reads that and believes it. This
one claims nothing at all -- it is rain the way a fish tank is fish. When you
want the real sky, `app_control.py weather` fetches it.

Drawn, never a bitmap, for the same reason as the face and the icons: it has to
hold up at 1920x1080, and it has to be cheap. The face process already spends
most of a core (see FACE_FPS), so this is plain lines over a black fill -- no
per-frame surfaces, no alpha blending, no allocation once it has started.
"""

from __future__ import annotations

import random

import config

import apps
from apps.base import DIM_COLOR, App

BRIGHT = (255, 106, 0)          # Friday's orange, the same one the icons use


@apps.register
class Rain(App):
    NAME = "rain"

    def setup(self, hold: float | None = None, drops: int | None = None,
              **_) -> None:
        # hold 0 means "until something takes the screen back", which is the
        # whole point of this one. The face reclaims it on any expression.
        self.hold = float(config.APP_RAIN_HOLD_S if hold is None else hold)
        count = int(config.APP_RAIN_DROPS if drops is None else drops)
        self._rng = random.Random(7)          # same sky every time it starts
        # Each drop: [x, y, fall speed px/s, length px, near?]. NEAR drops are
        # brighter, longer and faster; far ones dim and slow. That parallax is
        # the whole depth effect and it costs one boolean.
        self._drops: list[list] = []
        for _ in range(count):
            self._drops.append(self._new_drop(seed_y=True))
        self._thick_near = max(3, self.paint.px(0.004))
        self._thick_far = max(2, self.paint.px(0.002))

    def _new_drop(self, seed_y: bool = False) -> list:
        near = self._rng.random() < 0.45
        speed = self.height * (self._rng.uniform(0.55, 0.85) if near
                               else self._rng.uniform(0.22, 0.42))
        length = self.paint.px(0.055 if near else 0.03) * self._rng.uniform(0.7, 1.3)
        y = self._rng.uniform(-self.height, self.height) if seed_y \
            else -self._rng.uniform(0, self.height * 0.35)
        return [self._rng.uniform(0, self.width), y, speed, length, near]

    def tick(self, dt: float) -> None:
        # Clamped: a stalled frame (whisper, a camera spawning) must not
        # teleport every drop through the screen in one step.
        step = min(dt, 0.1)
        height = self.height
        for drop in self._drops:
            drop[1] += drop[2] * step
            if drop[1] > height:
                drop[:] = self._new_drop()
        if self.hold and self.elapsed >= self.hold:
            self.done = True

    def draw(self, surface) -> None:
        import pygame

        surface.fill((0, 0, 0))
        for x, y, _speed, length, near in self._drops:
            pygame.draw.line(surface, BRIGHT if near else DIM_COLOR,
                             (int(x), int(y)), (int(x), int(y + length)),
                             self._thick_near if near else self._thick_far)
