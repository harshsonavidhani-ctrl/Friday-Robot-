"""friday_face.py — Friday's face on the 7" display.

Public API is one method:

    face = FridayFace(fullscreen=True)
    face.set_emotion("happy")
    face.run()                       # blocking loop

Geometry is authored on a 1024x600 canvas and scaled to whatever the real
display reports, so the face keeps its proportions on any panel.

    python modules/friday_face.py                 windowed demo, cycles all
    python modules/friday_face.py --fullscreen    on the real 7" screen
    python modules/friday_face.py --emotion happy pin one emotion
    python modules/friday_face.py --shots out/    render a PNG each, headless
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from dataclasses import dataclass, field, replace

import pygame

# =====================================================================
# PALETTE — tune these
# =====================================================================
# Sampled from the studio-lit reference photos. Photo 1 (the real screen,
# shot on a phone) reads much brighter than this because the camera blew
# out the backlight; trust these values over that photo.

BG_COLOR = (52, 64, 150)          # dark blue background
EYE_COLOR = (247, 209, 156)       # warm amber
MOUTH_COLOR = (247, 209, 156)     # same amber as the eyes

# Background has a very slight centre lift in the photos. 0 disables it.
BG_VIGNETTE = 18

# --- 3D finish -------------------------------------------------------
# Applied ONLY to emotions marked dimensional=True. The seven from the
# photos are deliberately flat, because the photos are flat.
GRADIENT_TOP_LIFT = 26            # how much lighter the top of a shape is
GRADIENT_BOTTOM_DROP = 34         # how much deeper the bottom is
HIGHLIGHT_ALPHA = 70              # top-left gloss strength
HIGHLIGHT_ALPHA_GLOSSY = 145      # used by heart eyes
SHADOW_COLOR = (16, 20, 56)
SHADOW_ALPHA = 95
SHADOW_OFFSET = (0, 7)
BLUR_AMOUNT = 5

# =====================================================================
# TIMING — tune these
# =====================================================================
TRANSITION_S = 0.45               # emotion cross-fade length
BLINK_CLOSED_S = 0.16             # one blink, close+open
BLINK_GAP_MIN_S = 2.6
BLINK_GAP_MAX_S = 6.5
BLINK_DOUBLE_CHANCE = 0.18
DEMO_HOLD_S = 2.6

# =====================================================================
# GEOMETRY — the design canvas
# =====================================================================
BASE_W, BASE_H = 1024, 600


@dataclass(frozen=True)
class Eye:
    """One eye. `kind` picks the silhouette; the rest are tweenable numbers."""

    kind: str = "rect"        # rect chevron diamond arc spiral heart circle
    w: float = 118
    h: float = 138
    radius: float = 26
    tilt: float = 0.0         # degrees, positive = counter-clockwise
    dx: float = 0.0
    dy: float = 0.0
    apex: float = 0.58        # chevron only: where the point sits, 0..1
    pupil: float = 0.0        # circle only: pupil size as a fraction


@dataclass(frozen=True)
class Mouth:
    kind: str = "line"        # line smile wavy o box smirk none
    w: float = 428
    h: float = 20             # box only
    thickness: float = 20
    curve: float = 0.0        # +down = smile, -up = frown
    radius: float = 10
    dx: float = 0.0


@dataclass(frozen=True)
class Face:
    left: Eye
    right: Eye
    mouth: Mouth
    eye_y: float = 250        # centre of the eyes
    eye_gap: float = 272      # centre-to-centre
    mouth_y: float = 421
    dimensional: bool = False
    extras: tuple = ()
    blink: bool = True


# =====================================================================
# THE SEVEN FROM THE PHOTOS
# =====================================================================
# Each entry notes which photo it came from. These are flat on purpose.

EMOTIONS: dict[str, Face] = {
    # Photo 1 — big rounded squares, long flat mouth.
    "neutral": Face(
        left=Eye(w=118, h=138, radius=26),
        right=Eye(w=118, h=138, radius=26),
        mouth=Mouth(kind="line", w=428, thickness=20),
        eye_y=250, eye_gap=272, mouth_y=421,
    ),

    # Photo 4 — downward wedges pointing inward, wide thin smile.
    "happy": Face(
        left=Eye(kind="chevron", w=156, h=146, apex=0.72),
        right=Eye(kind="chevron", w=156, h=146, apex=0.28),
        mouth=Mouth(kind="smile", w=384, thickness=14, curve=58),
        eye_y=244, eye_gap=272, mouth_y=396,
    ),

    # Photo 5 — the SARAS dashboard. Diamond eyes, `sad` selected.
    # Flagged: that screenshot is a different robot, and shows no mouth.
    # The small frown here is my inference, not something in your photo.
    "sad": Face(
        left=Eye(w=112, h=112, radius=22, tilt=38),
        right=Eye(w=112, h=112, radius=22, tilt=38),
        mouth=Mouth(kind="smile", w=196, thickness=14, curve=-30),
        eye_y=254, eye_gap=262, mouth_y=436,
    ),

    # Photo 2 — left eye skewed, short mouth pushed left.
    "thinking": Face(
        left=Eye(w=124, h=118, radius=20, tilt=-5),
        right=Eye(w=116, h=128, radius=24),
        mouth=Mouth(kind="line", w=112, thickness=13, dx=-64),
        eye_y=252, eye_gap=270, mouth_y=430,
    ),

    # Photo 3 — eyes squashed to bars, tiny mouth.
    "focused": Face(
        left=Eye(w=152, h=52, radius=14),
        right=Eye(w=152, h=52, radius=14),
        mouth=Mouth(kind="line", w=108, thickness=14),
        eye_y=258, eye_gap=266, mouth_y=430,
    ),

    # Photos 6-9 — the yawn. Eyes squeeze shut as the mouth opens.
    # `extras=("yawn",)` drives that cycle over time.
    "yawning": Face(
        left=Eye(w=140, h=118, radius=20),
        right=Eye(w=140, h=118, radius=20),
        mouth=Mouth(kind="box", w=64, h=58, radius=22),
        eye_y=250, eye_gap=268, mouth_y=418,
        extras=("yawn",),
    ),
}

DEMO_ORDER = ["neutral", "happy", "sad", "thinking", "focused", "yawning"]


# =====================================================================
# Drawing helpers
# =====================================================================

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease(t: float) -> float:
    """Ease in-out cubic, so transitions start and land softly."""
    return 4 * t * t * t if t < 0.5 else 1 - pow(-2 * t + 2, 3) / 2


def _shift(color, amount: int):
    return tuple(max(0, min(255, c + amount)) for c in color[:3])


def _blur(surf: pygame.Surface, amount: int = BLUR_AMOUNT) -> pygame.Surface:
    """Cheap blur: shrink then grow. Keeps per-pixel alpha."""
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(
        surf, (max(1, w // amount), max(1, h // amount)))
    return pygame.transform.smoothscale(small, (w, h))


def _finish(mask: pygame.Surface, color, dimensional: bool,
            glossy: bool = False) -> pygame.Surface:
    """Turn a white silhouette into the final shape.

    Flat mode just tints it. Dimensional mode adds the vertical gradient,
    the top-left gloss, and a soft drop shadow.
    """
    w, h = mask.get_size()

    if not dimensional:
        out = pygame.Surface((w, h), pygame.SRCALPHA)
        out.fill((*color, 255))
        out.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        return out

    # Vertical gradient: lighter at the top, deeper at the bottom.
    grad = pygame.Surface((w, h), pygame.SRCALPHA)
    top = _shift(color, GRADIENT_TOP_LIFT)
    bottom = _shift(color, -GRADIENT_BOTTOM_DROP)
    for y in range(h):
        t = y / max(1, h - 1)
        row = tuple(int(_lerp(top[i], bottom[i], t)) for i in range(3))
        pygame.draw.line(grad, (*row, 255), (0, y), (w, y))
    grad.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)

    # Gloss near the top-left.
    gloss = pygame.Surface((w, h), pygame.SRCALPHA)
    alpha = HIGHLIGHT_ALPHA_GLOSSY if glossy else HIGHLIGHT_ALPHA
    pygame.draw.ellipse(
        gloss, (255, 255, 255, alpha),
        pygame.Rect(int(w * 0.12), int(h * 0.08),
                    max(2, int(w * 0.46)), max(2, int(h * 0.34))))
    gloss = _blur(gloss)
    gloss.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    grad.blit(gloss, (0, 0))

    # Soft drop shadow underneath.
    shadow = pygame.Surface((w, h), pygame.SRCALPHA)
    shadow.fill((*SHADOW_COLOR, 255))
    shadow.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    shadow = _blur(shadow)
    shadow.set_alpha(SHADOW_ALPHA)

    out = pygame.Surface((w, h), pygame.SRCALPHA)
    out.blit(shadow, SHADOW_OFFSET)
    out.blit(grad, (0, 0))
    return out


def _thick_path(mask: pygame.Surface, points, thickness: float) -> None:
    """Stroke a path by stamping circles along it — smooth, no AA needed."""
    r = max(1, int(round(thickness / 2)))
    for x, y in points:
        pygame.draw.circle(mask, (255, 255, 255, 255), (int(x), int(y)), r)


# --- eye silhouettes -------------------------------------------------

def _eye_mask(eye: Eye, scale: float) -> pygame.Surface:
    """White silhouette of one eye, on a padded transparent surface."""
    w = max(2, int(eye.w * scale))
    h = max(2, int(eye.h * scale))
    pad = int(max(24, max(w, h) * 0.45))
    surf = pygame.Surface((w + pad * 2, h + pad * 2), pygame.SRCALPHA)
    box = pygame.Rect(pad, pad, w, h)
    white = (255, 255, 255, 255)

    if eye.kind == "rect":
        pygame.draw.rect(surf, white, box,
                         border_radius=max(0, int(eye.radius * scale)))

    elif eye.kind == "chevron":
        # Flat top, sides falling to a point below. Photo 4's eye language.
        pygame.draw.polygon(surf, white, [
            (box.left, box.top),
            (box.right, box.top),
            (box.left + w * eye.apex, box.bottom),
        ])

    elif eye.kind == "diamond":
        pygame.draw.rect(surf, white, box,
                         border_radius=max(0, int(eye.radius * scale)))

    elif eye.kind == "arc":
        # Closed eye: a downward arc, like a lash line.
        pts = []
        for i in range(41):
            t = i / 40
            pts.append((box.left + w * t,
                        box.top + h * 0.5 - math.sin(math.pi * t) * h * 0.42))
        _thick_path(surf, pts, max(3, eye.radius * scale))

    elif eye.kind == "circle":
        pygame.draw.ellipse(surf, white, box)

    elif eye.kind == "spiral":
        pts = []
        turns, steps = 2.6, 150
        for i in range(steps + 1):
            t = i / steps
            ang = t * turns * math.tau
            rad = t * min(w, h) * 0.5
            pts.append((box.centerx + math.cos(ang) * rad,
                        box.centery + math.sin(ang) * rad))
        _thick_path(surf, pts, max(3, min(w, h) * 0.13))

    elif eye.kind == "heart":
        pts = []
        for i in range(121):
            a = i / 120 * math.tau
            hx = 16 * math.sin(a) ** 3
            hy = -(13 * math.cos(a) - 5 * math.cos(2 * a)
                   - 2 * math.cos(3 * a) - math.cos(4 * a))
            pts.append((box.centerx + hx / 17 * (w / 2),
                        box.centery + hy / 17 * (h / 2)))
        pygame.draw.polygon(surf, white, pts)

    return surf


def _mouth_mask(mouth: Mouth, scale: float) -> pygame.Surface:
    w = max(2, int(mouth.w * scale))
    th = max(2, int(mouth.thickness * scale))
    curve = mouth.curve * scale
    h = max(2, int(mouth.h * scale))
    pad = int(max(30, abs(curve) + th + 24))
    sw = w + pad * 2
    sh = int(max(h, abs(curve) * 2) + th + pad * 2)
    surf = pygame.Surface((sw, sh), pygame.SRCALPHA)
    cx, cy = sw / 2, sh / 2
    white = (255, 255, 255, 255)

    if mouth.kind == "none":
        return surf

    if mouth.kind == "line":
        rect = pygame.Rect(0, 0, w, th)
        rect.center = (cx, cy)
        pygame.draw.rect(surf, white, rect, border_radius=th // 2)

    elif mouth.kind == "box":
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (cx, cy)
        pygame.draw.rect(surf, white, rect,
                         border_radius=max(0, int(mouth.radius * scale)))

    elif mouth.kind == "o":
        rect = pygame.Rect(0, 0, w, h)
        rect.center = (cx, cy)
        pygame.draw.ellipse(surf, white, rect)

    elif mouth.kind == "smile":
        # Quadratic arc. Positive curve bows downward into a smile.
        pts = []
        for i in range(61):
            t = i / 60
            x = cx - w / 2 + w * t
            y = cy + curve * (1 - (2 * t - 1) ** 2) - curve / 2
            pts.append((x, y))
        _thick_path(surf, pts, th)

    elif mouth.kind == "wavy":
        pts = []
        for i in range(61):
            t = i / 60
            pts.append((cx - w / 2 + w * t,
                        cy + math.sin(t * math.tau * 1.5) * (curve or 10 * scale)))
        _thick_path(surf, pts, th)

    elif mouth.kind == "smirk":
        pts = []
        for i in range(41):
            t = i / 40
            x = cx - w / 2 + w * t
            pts.append((x, cy + curve * (t ** 1.9) - curve * 0.5))
        _thick_path(surf, pts, th)

    return surf


# =====================================================================
# Tweening
# =====================================================================

def _lerp_eye(a: Eye, b: Eye, t: float, kind: str) -> Eye:
    return Eye(
        kind=kind,
        w=_lerp(a.w, b.w, t), h=_lerp(a.h, b.h, t),
        radius=_lerp(a.radius, b.radius, t),
        tilt=_lerp(a.tilt, b.tilt, t),
        dx=_lerp(a.dx, b.dx, t), dy=_lerp(a.dy, b.dy, t),
        apex=_lerp(a.apex, b.apex, t),
        pupil=_lerp(a.pupil, b.pupil, t),
    )


def _lerp_mouth(a: Mouth, b: Mouth, t: float, kind: str) -> Mouth:
    return Mouth(
        kind=kind,
        w=_lerp(a.w, b.w, t), h=_lerp(a.h, b.h, t),
        thickness=_lerp(a.thickness, b.thickness, t),
        curve=_lerp(a.curve, b.curve, t),
        radius=_lerp(a.radius, b.radius, t),
        dx=_lerp(a.dx, b.dx, t),
    )


# =====================================================================
# The face
# =====================================================================

class FridayFace:
    def __init__(self, size=(BASE_W, BASE_H), fullscreen=False,
                 caption="Friday"):
        pygame.init()
        pygame.display.set_caption(caption)
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode(size, flags)
        if fullscreen:
            pygame.mouse.set_visible(False)
        self.clock = pygame.time.Clock()

        self._current = EMOTIONS["neutral"]
        self._previous = self._current
        self._name = "neutral"
        self._t = 1.0

        self._elapsed = 0.0
        self._blink_at = random.uniform(BLINK_GAP_MIN_S, BLINK_GAP_MAX_S)
        self._blink_t = None
        self._blink_queued = 0
        self.running = True

    # --- public ------------------------------------------------------

    @property
    def emotion(self) -> str:
        return self._name

    def set_emotion(self, name: str) -> None:
        """Cross-fade to `name`. Unknown names raise, so typos surface early."""
        if name not in EMOTIONS:
            raise KeyError(
                f"unknown emotion {name!r}; have: {', '.join(sorted(EMOTIONS))}")
        if name == self._name:
            return
        self._previous = self._resolved()
        self._current = EMOTIONS[name]
        self._name = name
        self._t = 0.0

    # --- internals ---------------------------------------------------

    def _scale(self) -> float:
        w, h = self.screen.get_size()
        return min(w / BASE_W, h / BASE_H)

    def _resolved(self) -> Face:
        """The face as it looks right now, mid-transition included."""
        if self._t >= 1.0:
            return self._current
        p = _ease(self._t)
        a, b = self._previous, self._current
        return Face(
            left=_lerp_eye(a.left, b.left, p, b.left.kind),
            right=_lerp_eye(a.right, b.right, p, b.right.kind),
            mouth=_lerp_mouth(a.mouth, b.mouth, p, b.mouth.kind),
            eye_y=_lerp(a.eye_y, b.eye_y, p),
            eye_gap=_lerp(a.eye_gap, b.eye_gap, p),
            mouth_y=_lerp(a.mouth_y, b.mouth_y, p),
            dimensional=b.dimensional if p > 0.5 else a.dimensional,
            extras=b.extras, blink=b.blink,
        )

    def _blink_factor(self) -> float:
        """1.0 open, 0.0 shut."""
        if self._blink_t is None:
            return 1.0
        p = self._blink_t / BLINK_CLOSED_S
        return _ease(abs(2 * p - 1))

    def _apply_extras(self, face: Face) -> Face:
        if "yawn" not in face.extras:
            return face
        # One yawn every 4.2s: mouth opens, eyes squeeze down to bars.
        phase = (self._elapsed % 4.2) / 4.2
        y = 0.0 if phase > 0.72 else math.sin(phase / 0.72 * math.pi) ** 0.7
        return replace(
            face,
            left=replace(face.left, h=_lerp(118, 30, y), radius=_lerp(20, 14, y)),
            right=replace(face.right, h=_lerp(118, 30, y), radius=_lerp(20, 14, y)),
            mouth=replace(face.mouth, w=_lerp(64, 92, y), h=_lerp(58, 116, y)),
        )

    def update(self, dt: float) -> None:
        self._elapsed += dt

        if self._t < 1.0:
            self._t = min(1.0, self._t + dt / TRANSITION_S)

        if self._blink_t is not None:
            self._blink_t += dt
            if self._blink_t >= BLINK_CLOSED_S:
                self._blink_t = None
                if self._blink_queued:
                    self._blink_queued -= 1
                    self._blink_t = 0.0
                else:
                    self._blink_at = self._elapsed + random.uniform(
                        BLINK_GAP_MIN_S, BLINK_GAP_MAX_S)
        elif self._elapsed >= self._blink_at:
            self._blink_t = 0.0
            if random.random() < BLINK_DOUBLE_CHANCE:
                self._blink_queued = 1

    def _draw_background(self) -> None:
        self.screen.fill(BG_COLOR)
        if BG_VIGNETTE:
            w, h = self.screen.get_size()
            glow = pygame.Surface((w, h), pygame.SRCALPHA)
            pygame.draw.ellipse(
                glow, (*_shift(BG_COLOR, BG_VIGNETTE), 120),
                pygame.Rect(int(-w * 0.15), int(-h * 0.2),
                            int(w * 1.3), int(h * 1.4)))
            self.screen.blit(_blur(glow, 8), (0, 0))

    def _blit_eye(self, eye: Eye, center, scale: float, dimensional: bool,
                  alpha: int) -> None:
        if eye.h * scale < 1.5 or eye.w * scale < 1.5:
            return
        mask = _eye_mask(eye, scale)
        surf = _finish(mask, EYE_COLOR, dimensional, glossy=eye.kind == "heart")
        if eye.tilt:
            surf = pygame.transform.rotate(surf, eye.tilt)
        if alpha < 255:
            surf.set_alpha(alpha)
        rect = surf.get_rect(center=(center[0] + eye.dx * scale,
                                     center[1] + eye.dy * scale))
        self.screen.blit(surf, rect)

        if eye.pupil > 0:
            pr = max(2, int(min(eye.w, eye.h) * scale * eye.pupil * 0.5))
            pygame.draw.circle(self.screen, BG_COLOR, rect.center, pr)

    def _blit_mouth(self, mouth: Mouth, center, scale: float,
                    dimensional: bool, alpha: int) -> None:
        mask = _mouth_mask(mouth, scale)
        surf = _finish(mask, MOUTH_COLOR, dimensional)
        if alpha < 255:
            surf.set_alpha(alpha)
        rect = surf.get_rect(center=(center[0] + mouth.dx * scale, center[1]))
        self.screen.blit(surf, rect)

    def draw(self) -> None:
        self._draw_background()

        face = self._apply_extras(self._resolved())
        scale = self._scale()
        w, h = self.screen.get_size()
        ox, oy = w / 2, h / 2 - BASE_H / 2 * scale

        blink = self._blink_factor() if face.blink else 1.0
        half = face.eye_gap / 2 * scale

        eye_c = oy + face.eye_y * scale
        left_c = (ox - half, eye_c)
        right_c = (ox + half, eye_c)
        mouth_c = (ox, oy + face.mouth_y * scale)

        # Mid-transition between two different silhouettes, cross-fade the
        # pair. Same silhouette, one pass at full alpha.
        p = _ease(self._t)
        fading = self._t < 1.0

        def pair(prev_kind, cur_kind):
            if not fading or prev_kind == cur_kind:
                return [(cur_kind, 255)]
            return [(prev_kind, int(255 * (1 - p))), (cur_kind, int(255 * p))]

        for kind, a in pair(self._previous.left.kind, face.left.kind):
            eye = replace(face.left, kind=kind,
                          h=face.left.h * blink if kind != "arc" else face.left.h)
            self._blit_eye(eye, left_c, scale, face.dimensional, a)

        for kind, a in pair(self._previous.right.kind, face.right.kind):
            eye = replace(face.right, kind=kind,
                          h=face.right.h * blink if kind != "arc" else face.right.h)
            self._blit_eye(eye, right_c, scale, face.dimensional, a)

        for kind, a in pair(self._previous.mouth.kind, face.mouth.kind):
            self._blit_mouth(replace(face.mouth, kind=kind), mouth_c, scale,
                             face.dimensional, a)

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif event.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()

    def run(self, fps: int = 60) -> None:
        while self.running:
            dt = self.clock.tick(fps) / 1000.0
            self.handle_events()
            self.update(dt)
            self.draw()
            pygame.display.flip()
        pygame.quit()

    def run_demo(self, hold: float = DEMO_HOLD_S, fps: int = 60) -> None:
        """Cycle every emotion so you can check them against the photos."""
        font = pygame.font.SysFont(None, 30)
        order, index, since = DEMO_ORDER, 0, 0.0
        self.set_emotion(order[0])

        while self.running:
            dt = self.clock.tick(fps) / 1000.0
            self.handle_events()
            self.update(dt)

            since += dt
            if since >= hold:
                since = 0.0
                index = (index + 1) % len(order)
                self.set_emotion(order[index])

            self.draw()
            label = font.render(self._name, True, (255, 255, 255))
            label.set_alpha(120)
            self.screen.blit(label, (18, 14))
            pygame.display.flip()
        pygame.quit()


# =====================================================================
# CLI
# =====================================================================

def _render_shots(out_dir: str) -> None:
    """Headless: one PNG per emotion, for checking against the photos."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.makedirs(out_dir, exist_ok=True)
    face = FridayFace()
    for name in DEMO_ORDER:
        face._name = name
        face._current = face._previous = EMOTIONS[name]
        face._t = 1.0
        face._blink_t = None
        # Catch the yawn mid-open so the shot shows the mouth actually wide.
        face._elapsed = 4.2 * 0.36 if name == "yawning" else 0.0
        face.draw()
        path = os.path.join(out_dir, f"{name}.png")
        pygame.image.save(face.screen, path)
        print(f"  {path}")
    pygame.quit()


def main() -> int:
    ap = argparse.ArgumentParser(description="Friday's face")
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--emotion", help="pin one emotion instead of cycling")
    ap.add_argument("--shots", metavar="DIR", help="render a PNG per emotion")
    ap.add_argument("--size", default=f"{BASE_W}x{BASE_H}")
    args = ap.parse_args()

    if args.shots:
        _render_shots(args.shots)
        return 0

    w, h = (int(v) for v in args.size.lower().split("x"))
    face = FridayFace(size=(w, h), fullscreen=args.fullscreen)
    if args.emotion:
        face.set_emotion(args.emotion)
        face.run()
    else:
        face.run_demo()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
