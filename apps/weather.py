"""apps/weather.py — the sky, as a card.

    app_control.set_weather({"place": "Pune", "condition": "rain",
                             "temp_c": 24, "high_c": 29, "low_c": 22})

FRIDAY HAS NO WEATHER API KEY AND NO WAY TO SEE THE SKY. The brain has web
search; it looks the weather up and hands the numbers here. That keeps one
more key out of the project and means the card can show whatever the brain
found, including a place that is not this one.

The icon is drawn, not an image file -- same reason the face is drawn: it has
to match at 1920x1080 on a panel, and a bitmap of a sun would not.
"""

from __future__ import annotations

import math

import config

import apps
from apps.base import DIM_COLOR, App


@apps.register
class Weather(App):
    NAME = "weather"

    def setup(self, weather: dict | None = None,
              hold: float | None = None, **_) -> None:
        self.data = dict(weather or {})
        self.hold = float(config.APP_WEATHER_HOLD_S if hold is None else hold)
        self.icon = _icon_for(str(self.data.get("condition") or ""))

    def tick(self, dt: float) -> None:
        if self.hold and self.elapsed >= self.hold:
            self.done = True

    def draw(self, surface) -> None:
        paint = self.paint
        surface.fill((0, 0, 0))
        left = int(self.width * 0.30)
        right = int(self.width * 0.68)
        mid = int(self.height * 0.48)

        self.icon(self, surface, (left, mid))

        temp = self.data.get("temp_c")
        paint.text(surface, f"{round(temp)}°" if temp is not None else "--",
                   paint.px(0.30), (right, mid))

        condition = str(self.data.get("condition") or "").upper()
        if condition:
            paint.text(surface, condition, paint.px(0.075),
                       (right, mid + paint.px(0.24)), alpha=215)

        place = str(self.data.get("place") or "").upper()
        if place:
            paint.text(surface, place, paint.px(0.065),
                       (self.width // 2, int(self.height * 0.14)), alpha=215)

        paint.text(surface, self._footer(), paint.px(0.05),
                   (self.width // 2, int(self.height * 0.90)), alpha=190)

    def _footer(self) -> str:
        bits = []
        high, low = self.data.get("high_c"), self.data.get("low_c")
        if high is not None:
            bits.append(f"HIGH {round(high)}°")
        if low is not None:
            bits.append(f"LOW {round(low)}°")
        if self.data.get("wind_kph") is not None:
            bits.append(f"WIND {round(self.data['wind_kph'])} KPH")
        if self.data.get("humidity") is not None:
            bits.append(f"HUMIDITY {round(self.data['humidity'])}%")
        return "   ".join(bits)

    def launched_message(self) -> str:
        summary = self.data.get("summary")
        if summary:
            return str(summary)
        temp = self.data.get("temp_c")
        place = self.data.get("place") or "here"
        condition = self.data.get("condition") or "unknown"
        if temp is None:
            return f"{condition} in {place}"
        return f"{round(temp)} degrees and {condition} in {place}"


# =====================================================================
# the icons -- all drawn, all animated, all cheap
# =====================================================================

def _sun(app, surface, center) -> None:
    import pygame
    paint = app.paint
    r = paint.px(0.11)
    art = paint.stencil("sun-disc", (r * 2, r * 2), lambda m: pygame.draw.circle(
        m, (255, 255, 255), (r, r), r))
    surface.blit(art, art.get_rect(center=center))
    # Rays turn slowly, one full revolution a minute.
    for i in range(8):
        angle = 2 * math.pi * (i / 8 + app.elapsed / 60.0)
        inner, outer = r * 1.35, r * 1.85
        pygame.draw.line(
            surface, (255, 106, 0),
            (center[0] + inner * math.cos(angle),
             center[1] + inner * math.sin(angle)),
            (center[0] + outer * math.cos(angle),
             center[1] + outer * math.sin(angle)),
            max(4, paint.px(0.014)))


def _cloud(app, surface, center, dy: int = 0) -> None:
    import pygame
    paint = app.paint
    w, h = paint.px(0.34), paint.px(0.17)
    art = paint.stencil("cloud", (w, h), _paint_cloud)
    drift = int(paint.px(0.012) * math.sin(2 * math.pi * app.elapsed / 7.0))
    surface.blit(art, art.get_rect(center=(center[0] + drift,
                                           center[1] + dy)))


def _paint_cloud(mask) -> None:
    import pygame
    w, h = mask.get_size()
    pygame.draw.circle(mask, (255, 255, 255),
                       (int(w * 0.28), int(h * 0.55)), int(h * 0.33))
    pygame.draw.circle(mask, (255, 255, 255),
                       (int(w * 0.50), int(h * 0.34)), int(h * 0.34))
    pygame.draw.circle(mask, (255, 255, 255),
                       (int(w * 0.72), int(h * 0.52)), int(h * 0.30))
    pygame.draw.rect(mask, (255, 255, 255),
                     pygame.Rect(int(w * 0.16), int(h * 0.55),
                                 int(w * 0.68), int(h * 0.30)))


def _partly(app, surface, center) -> None:
    _sun(app, surface, (center[0] + app.paint.px(0.07),
                        center[1] - app.paint.px(0.07)))
    _cloud(app, surface, center, dy=app.paint.px(0.04))


def _rain(app, surface, center) -> None:
    import pygame
    paint = app.paint
    _cloud(app, surface, center, dy=-paint.px(0.05))
    length = paint.px(0.055)
    for i in range(5):
        phase = (app.elapsed * 1.6 + i * 0.2) % 1.0
        x = center[0] + paint.px(0.055) * (i - 2)
        y = center[1] + paint.px(0.06) + phase * paint.px(0.16)
        pygame.draw.line(surface, (255, 106, 0), (x, y), (x, y + length),
                         max(3, paint.px(0.010)))


def _storm(app, surface, center) -> None:
    import pygame
    paint = app.paint
    _cloud(app, surface, center, dy=-paint.px(0.05))
    flash = 255 if (app.elapsed % 1.6) < 0.16 else 90
    w = paint.px(0.09)
    art = paint.stencil("bolt", (w, int(w * 1.6)), _paint_bolt, alpha=flash)
    surface.blit(art, art.get_rect(center=(center[0],
                                           center[1] + paint.px(0.11))))


def _paint_bolt(mask) -> None:
    import pygame
    w, h = mask.get_size()
    pygame.draw.polygon(mask, (255, 255, 255), [
        (w * 0.58, 0), (w * 0.10, h * 0.58), (w * 0.44, h * 0.58),
        (w * 0.32, h), (w * 0.92, h * 0.38), (w * 0.54, h * 0.38),
    ])


def _snow(app, surface, center) -> None:
    import pygame
    paint = app.paint
    _cloud(app, surface, center, dy=-paint.px(0.05))
    for i in range(5):
        phase = (app.elapsed * 0.5 + i * 0.19) % 1.0
        x = center[0] + paint.px(0.055) * (i - 2) + paint.px(0.012) * math.sin(
            2 * math.pi * (phase + i / 5))
        y = center[1] + paint.px(0.06) + phase * paint.px(0.17)
        pygame.draw.circle(surface, (255, 106, 0), (int(x), int(y)),
                           max(3, paint.px(0.011)))


def _fog(app, surface, center) -> None:
    paint = app.paint
    for i in range(4):
        drift = paint.px(0.02) * math.sin(2 * math.pi * app.elapsed / 6.0 + i)
        width = paint.px(0.30) - i * paint.px(0.02)
        rect = (center[0] - width // 2 + drift,
                center[1] - paint.px(0.09) + i * paint.px(0.06),
                width, max(4, paint.px(0.022)))
        paint.box(surface, rect, radius=paint.px(0.011),
                  lit=False, color=DIM_COLOR if i % 2 else (255, 106, 0))


_CONDITIONS = (
    (("thunder", "storm", "lightning"), _storm),
    (("snow", "sleet", "hail", "blizzard"), _snow),
    (("rain", "drizzle", "shower", "wet", "monsoon"), _rain),
    (("fog", "mist", "haze", "smoke"), _fog),
    (("partly", "part cloud", "some cloud", "few cloud"), _partly),
    (("cloud", "overcast", "grey", "gray"), _cloud),
    (("clear", "sun", "fair", "hot"), _sun),
)


def _icon_for(condition: str):
    text = condition.lower()
    for words, icon in _CONDITIONS:
        if any(word in text for word in words):
            return icon
    return _partly            # an honest shrug: something, and some cloud
