"""apps/base.py — what every screen app gets for free.

An app is a full-screen view with a life of its own: it comes up, says its
one thing, and hands the panel back. Subclass App, give it a NAME, and put
the drawing in draw().

    from apps.base import App
    import apps

    @apps.register
    class Clock(App):
        NAME = "clock"
        def setup(self): ...
        def tick(self, dt): ...
        def draw(self, surface): ...

WHY A PAINT KIT AND NOT JUST pygame.draw. Friday's face is not flat orange on
black -- every shape carries a vertical gradient, an upper-left gloss and a
hard drop shadow (friday_face.py's _finish()). An app drawn with plain fills
sits on the same panel looking like a different robot. Paint below applies the
same finish to text and rectangles, so an app matches the face without
importing it -- friday_face imports this package, so this package must never
import friday_face.

EVERYTHING IS CACHED BY WHAT IT LOOKS LIKE, NOT BY WHEN IT IS DRAWN. This Pi
is CPU-bound and the face already eats most of a core; a clock that rebuilt a
gradient every frame would be felt everywhere else in the robot. A lit string
is built once per distinct string, so the big digits cost a blit for 29 of
every 30 frames.
"""

from __future__ import annotations

import math

import pygame

# The face's palette, deliberately repeated rather than imported -- see the
# module docstring. If friday_face.py's colours change, change them here too.
BG_COLOR = (0, 0, 0)
FG_COLOR = (255, 106, 0)          # #FF6A00
GRADIENT_TOP = (255, 168, 74)
GRADIENT_BOTTOM = (168, 58, 0)
HIGHLIGHT_ALPHA = 105
SHADOW_COLOR = (74, 27, 0)
SHADOW_ALPHA = 235
DIM_COLOR = (128, 53, 0)          # inactive track, secondary rules


class AppError(RuntimeError):
    """An app that cannot start or cannot run. Never fatal to the face."""


# =====================================================================
# paint
# =====================================================================

class Paint:
    """The face's finish, applied to text and boxes, and cached."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.width, self.height = size
        self._fonts: dict[int, pygame.font.Font] = {}
        self._lit: dict[tuple, pygame.Surface] = {}
        self._grad: dict[tuple[int, int], pygame.Surface] = {}

    # --- units -------------------------------------------------------

    def px(self, fraction: float) -> int:
        """A size as a fraction of the screen's short side, so every app
        lays out proportionally on any panel -- same rule as the face."""
        return max(1, int(min(self.width, self.height) * fraction))

    # --- materials ---------------------------------------------------

    def font(self, size: int) -> pygame.font.Font:
        f = self._fonts.get(size)
        if f is None:
            if not pygame.font.get_init():
                pygame.font.init()
            f = self._fonts[size] = pygame.font.SysFont(None, size, bold=True)
        return f

    def _gradient(self, w: int, h: int) -> pygame.Surface:
        key = (w, h)
        mat = self._grad.get(key)
        if mat is None:
            if len(self._grad) > 48:
                self._grad.clear()
            strip = pygame.Surface((1, 64), pygame.SRCALPHA)
            for y in range(64):
                t = y / 63.0
                strip.set_at((0, y), tuple(
                    int(GRADIENT_TOP[i] + (GRADIENT_BOTTOM[i]
                                           - GRADIENT_TOP[i]) * t)
                    for i in range(3)))
            mat = pygame.transform.smoothscale(strip, (w, h)).convert_alpha()
            gloss = pygame.Surface((w, h), pygame.SRCALPHA)
            ell = pygame.Rect(0, 0, max(1, int(w * 0.78)),
                              max(1, int(h * 0.5)))
            ell.center = (int(w * 0.36), int(h * 0.27))
            pygame.draw.ellipse(gloss, (255, 255, 255, HIGHLIGHT_ALPHA), ell)
            mat.blit(gloss, (0, 0))
            self._grad[key] = mat
        return mat

    def _light(self, mask: pygame.Surface) -> pygame.Surface:
        """A white silhouette -> the lit shape, with its shadow behind it."""
        w, h = mask.get_size()
        offset = max(2, self.px(0.008))
        out = pygame.Surface((w + offset, h + offset), pygame.SRCALPHA)

        shadow = pygame.Surface((w, h), pygame.SRCALPHA)
        shadow.fill((*SHADOW_COLOR, SHADOW_ALPHA))
        shadow.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        out.blit(shadow, (offset, offset))

        body = self._gradient(w, h).copy()
        body.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        out.blit(body, (0, 0))
        return out

    # --- the two things apps actually draw ---------------------------

    def text(self, surface, string: str, size: int, center,
             lit: bool = True, color=FG_COLOR, alpha: int = 255) -> pygame.Rect:
        """Draw a string centred on `center`. Lit text is cached per string,
        so a clock costs one blit for most frames."""
        string = str(string)
        if not string:
            return pygame.Rect(center, (0, 0))
        if lit:
            key = (string, size)
            art = self._lit.get(key)
            if art is None:
                if len(self._lit) > 120:
                    self._lit.clear()
                mask = self.font(size).render(string, True, (255, 255, 255))
                art = self._lit[key] = self._light(mask)
        else:
            art = self.font(size).render(string, True, color)
        if alpha < 255:
            art = art.copy()
            art.set_alpha(max(0, min(255, int(alpha))))
        rect = art.get_rect(center=(int(center[0]), int(center[1])))
        surface.blit(art, rect)
        return rect

    def box(self, surface, rect, radius: int = 0, lit: bool = True,
            color=FG_COLOR, alpha: int = 255) -> None:
        """A filled rectangle with the same finish. radius=0 keeps the sharp
        corners the face uses; a rounded one is for progress tracks."""
        rect = pygame.Rect(rect)
        if rect.width < 1 or rect.height < 1:
            return
        if not lit:
            shape = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(shape, color, shape.get_rect(),
                             border_radius=radius)
            if alpha < 255:
                shape.set_alpha(max(0, min(255, int(alpha))))
            surface.blit(shape, rect.topleft)
            return
        mask = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255), mask.get_rect(),
                         border_radius=radius)
        art = self._light(mask)
        if alpha < 255:
            art = art.copy()
            art.set_alpha(max(0, min(255, int(alpha))))
        surface.blit(art, rect.topleft)

    def stencil(self, key, size, paint_mask, alpha: int = 255):
        """A lit shape of any outline, cached.

        `paint_mask(surface)` draws the silhouette in white; what comes back
        is that silhouette with the face's finish on it. Cache by the KEY,
        including any size that changes -- and quantise sizes that animate,
        or every frame is a cache miss and the cost doubles. (The face learnt
        this the hard way; see _q() in friday_face.py.)
        """
        size = (max(1, int(size[0])), max(1, int(size[1])))
        cache_key = (key, size)
        art = self._lit.get(cache_key)
        if art is None:
            if len(self._lit) > 120:
                self._lit.clear()
            mask = pygame.Surface(size, pygame.SRCALPHA)
            paint_mask(mask)
            art = self._lit[cache_key] = self._light(mask)
        if alpha < 255:
            art = art.copy()
            art.set_alpha(max(0, min(255, int(alpha))))
        return art

    def arc(self, surface, center, radius: int, thickness: int,
            start: float, sweep: float, color=FG_COLOR,
            alpha: int = 255) -> None:
        """A ring segment, clockwise from 12 o'clock. pygame's own arc is
        one pixel wide and ragged, so this is drawn as short segments."""
        if not sweep:
            return                      # a negative sweep runs anticlockwise
        steps = max(3, int(abs(sweep) / (math.pi / 90)))
        layer = surface
        if alpha < 255:
            layer = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        points = []
        for i in range(steps + 1):
            a = -math.pi / 2 + start + sweep * (i / steps)
            points.append((center[0] + radius * math.cos(a),
                           center[1] + radius * math.sin(a)))
        pygame.draw.lines(layer, color, False, points, thickness)
        for point in (points[0], points[-1]):
            pygame.draw.circle(layer, color, (int(point[0]), int(point[1])),
                               thickness // 2)
        if layer is not surface:
            layer.set_alpha(max(0, min(255, int(alpha))))
            surface.blit(layer, (0, 0))


# =====================================================================
# the app
# =====================================================================

class App:
    """One full-screen view. Subclasses override setup/tick/draw."""

    NAME = ""
    # An app that says it is URGENT keeps the screen even while a safety
    # expression is up -- an alarm going off IS the thing to look at. Only
    # the alarm sets this, and the face still shows the emotion afterwards.
    URGENT = False

    def __init__(self, size: tuple[int, int], **kwargs) -> None:
        self.size = size
        self.width, self.height = size
        self.paint = Paint(size)
        self.elapsed = 0.0
        self.done = False
        self.setup(**kwargs)

    # --- subclass hooks ----------------------------------------------

    def setup(self, **kwargs) -> None:
        """Called with whatever the launcher passed. Raise TypeError for an
        argument you do not take -- the manager turns that into a sentence."""

    def tick(self, dt: float) -> None:
        """Advance. Set self.done = True to hand the screen back."""

    def draw(self, surface) -> None:
        raise NotImplementedError

    def handle_event(self, event) -> bool:
        """True means the app consumed it. Touch arrives as MOUSEBUTTONDOWN;
        the default is that any touch anywhere dismisses -- a 7" panel in
        front of you is a stop button before it is a UI."""
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.FINGERDOWN):
            return self.dismiss()
        return False

    def dismiss(self) -> bool:
        """A spoken or touched "stop". True closes the app."""
        self.done = True
        return True

    def close(self) -> None:
        """Release anything held -- a ring, a file. Always called."""

    def launched_message(self) -> str:
        return f"{self.NAME} is on the screen"

    # --- the manager's side ------------------------------------------

    def update(self, dt: float) -> None:
        self.elapsed += dt
        self.tick(dt)

    # --- shared furniture --------------------------------------------

    def _label(self, surface, text: str, y_frac: float = 0.80,
               alpha: int = 205) -> None:
        if not text:
            return
        self.paint.text(surface, text, self.paint.px(0.075),
                        (self.width // 2, int(self.height * y_frac)),
                        alpha=alpha)
