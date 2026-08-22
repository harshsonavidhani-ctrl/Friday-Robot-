"""friday_face.py — Friday's face on the 7" display.

    python3 friday_face.py            fullscreen, follows state/emotion.txt
    python3 friday_face.py --demo     cycle all 20 emotions + play the yawn
    python3 friday_face.py --window   windowed, for testing over SSH/VNC

Nothing drives this directly. It watches state/emotion.txt (and
state/yawn.txt for the yawn trigger) and tweens to whatever it finds
there; use face_control.py to write those files.

Every eye and mouth is a flat white silhouette until _finish() runs: a
top-lit vertical gradient, an upper-left gloss, and a soft drop shadow.
Geometry is authored on a 1024x600 canvas and letterboxed onto whatever the
real panel reports, so the proportions hold on any screen.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import dataclass, replace

import pygame

from face_control import (EMOTIONS as EMOTION_NAMES, DEFAULT_EMOTION,
                          EMOTION_FILE, YAWN_FILE, get_emotion)

# =====================================================================
# COLOURS — tune these
# =====================================================================
BG_COLOR = (17, 25, 50)          # dark navy background, darker than before
ORANGE = (236, 132, 64)          # base colour; every shape's gradient mid-point
PUPIL_COLOR = (18, 26, 52)       # only used by `scared`
MASK_WHITE = (255, 255, 255, 255)  # silhouettes are drawn white, then tinted

# --- dimensional finish -----------------------------------------------
# Every eye and mouth is a flat white silhouette until _finish() runs: a
# top-to-bottom gradient (lit from above), a soft upper-left gloss, and a
# drop shadow. Deliberately subtle -- from a normal viewing distance this
# should still read as one clean orange shape, not chrome.
GRADIENT_LIFT = 22               # how much lighter the top gets
GRADIENT_DROP = 28               # how much darker the bottom gets
GRADIENT_TOP = tuple(min(255, c + GRADIENT_LIFT) for c in ORANGE)
GRADIENT_BOTTOM = tuple(max(0, c - GRADIENT_DROP) for c in ORANGE)
HIGHLIGHT_ALPHA = 46              # upper-left gloss strength
HIGHLIGHT_ALPHA_GLOSSY = 110      # heart eyes and the dizzy spiral shine more
SHADOW_COLOR = (10, 14, 34)
SHADOW_ALPHA = 70
SHADOW_OFFSET = (0, 5)            # px, before supersampling
BLUR_AMOUNT = 5                   # smoothscale down/up divisor -- softness

# =====================================================================
# GEOMETRY — authored at 1024x600, tune these
# =====================================================================
BASE_W, BASE_H = 1024, 600

EYE_W = 130                      # neutral eye width
EYE_H = 130                      # neutral eye height
EYE_RADIUS = 28                  # corner rounding
EYE_GAP = 200                    # bare gap between the two eyes
EYE_CY = int(BASE_H * 0.45)      # eye centre line: 45% down the screen

MOUTH_W = 260                    # neutral mouth width
MOUTH_T = 14                     # bar thickness, ends are rounded
MOUTH_DROP = 100                 # gap from the BOTTOM of the eyes to the mouth
MOUTH_CY = EYE_CY + EYE_H // 2 + MOUTH_DROP

# =====================================================================
# TIMING — tune these
# =====================================================================
FPS = 30
TRANSITION_S = 0.40              # emotion tween length
BLINK_CLOSED_S = 0.13            # one blink, close and open again
BLINK_GAP_MIN_S = 2.0            # idle blinking, random in this range
BLINK_GAP_MAX_S = 5.0
POLL_S = 0.25                    # how often the state files are checked
DEMO_HOLD_S = 2.6                # --demo dwell per emotion

# --- yawn overlay -------------------------------------------------
# Not one of the 20 emotions -- a transient reflex that plays on top of
# whatever emotion is showing, then returns to it. See play_yawn().
YAWN_EYE_W = round(EYE_W * 1.05)
YAWN_EYE_H = round(EYE_H / 3)     # squinted nearly shut
YAWN_EYE_RADIUS = 18
YAWN_MOUTH_W = 46                 # a small vertical pill: taller than wide
YAWN_MOUTH_H = 92
YAWN_IN_S = 0.18                  # eyes/mouth close into the yawn shape
YAWN_HOLD_S = 0.62                # held open
YAWN_OUT_S = 0.30                 # eases back to whatever was showing

SUPERSAMPLE = 2                  # shapes are drawn 2x then shrunk, for clean
                                 # edges without per-pixel antialiasing cost


# =====================================================================
# The shape of a face
# =====================================================================

@dataclass(frozen=True)
class Eye:
    """One eye. `kind` picks the silhouette; everything else is tweenable."""
    kind: str = "rect"           # rect | heart | spiral | arc
    w: float = EYE_W
    h: float = EYE_H
    radius: float = EYE_RADIUS
    tilt: float = 0.0            # degrees, + rotates counter-clockwise
    bevel: float = 0.0           # px chamfered off ONE top corner of a rect
                                  # eye, not the whole shape: + cuts this
                                  # eye's own top-RIGHT corner, - cuts top-LEFT
    taper: float = 0.0           # px chamfered off ONE bottom corner (the
                                  # "happy squint" trapezoid): + cuts this
                                  # eye's own bottom-RIGHT corner, - cuts
                                  # bottom-LEFT
    dx: float = 0.0
    dy: float = 0.0
    pupil: float = 0.0           # dark pupil radius as a fraction of width


@dataclass(frozen=True)
class Mouth:
    kind: str = "bar"            # bar | curve | o | wavy | smirk | grin
    w: float = MOUTH_W
    h: float = MOUTH_T           # only `o` and `grin` use height
    thickness: float = MOUTH_T
    curve: float = 0.0           # + smiles up, - frowns down
    dx: float = 0.0
    dy: float = 0.0


@dataclass(frozen=True)
class Face:
    left: Eye
    right: Eye
    mouth: Mouth
    eye_gap: float = EYE_GAP
    eye_y: float = EYE_CY
    mouth_y: float = MOUTH_CY
    tilt: float = 0.0            # whole-face rotation, degrees
    extras: tuple = ()           # animated overlays, see _animate/_overlays
    blink: bool = True


def _eyes(**kw) -> tuple[Eye, Eye]:
    """A matched pair. `tilt`, `bevel` and `taper` all mirror between the two
    eyes, so +tilt/+bevel/+taper on the LEFT eye becomes -tilt/-bevel/-taper
    on the right one -- call sites only ever specify one signed value for a
    symmetric pair."""
    tilt = kw.pop("tilt", 0.0)
    bevel = kw.pop("bevel", 0.0)
    taper = kw.pop("taper", 0.0)
    return (Eye(tilt=tilt, bevel=bevel, taper=taper, **kw),
            Eye(tilt=-tilt, bevel=-bevel, taper=-taper, **kw))


def _face(left: Eye, right: Eye, mouth: Mouth, **kw) -> Face:
    return Face(left=left, right=right, mouth=mouth, **kw)


# =====================================================================
# THE 20 EMOTIONS
# =====================================================================
# Rotation sign: + is counter-clockwise, so it lifts a shape's right edge
# and drops its left. `_eyes(tilt=+n)` therefore droops both OUTER edges
# (sad); a negative tilt drops both INNER edges (angry).

_L, _R = _eyes()

EMOTIONS: dict[str, Face] = {

    # --- core --------------------------------------------------------
    "neutral": _face(_L, _R, Mouth()),

    "happy": _face(
        # Bottom-inner corner tapered off deep, into a pointed trapezoid --
        # a squinting, smiling-eyes shape, not a whole-eye rotation.
        *_eyes(h=124, radius=18, taper=92),
        Mouth("curve", w=250, thickness=16, curve=52),
    ),

    "sad": _face(
        # Eyes stay the plain neutral square -- only the mouth changes,
        # into a frown the same width as the neutral bar.
        _L, _R,
        Mouth("curve", w=260, thickness=14, curve=-32),
    ),

    "angry": _face(
        # Inner top corners chamfered off -- eyebrows angled down toward
        # the nose, same clean-cut language as sad's outer chamfer.
        *_eyes(h=100, radius=20, bevel=40),
        Mouth("bar", w=150, thickness=17),
    ),

    "surprised": _face(
        *_eyes(w=142, h=172, radius=32),
        Mouth("o", w=66, h=80),
    ),

    # --- functional --------------------------------------------------
    "listening": _face(
        # Eyes stay the neutral square -- the pulse is what reads as
        # "listening", not a different resting shape.
        _L, _R,
        Mouth("bar", w=140),
        extras=("pulse_eyes",),
    ),

    "thinking": _face(
        # Eyes are the neutral square, breathing open/contract in place --
        # plus the dots overlay above.
        _L, _R,
        Mouth("bar", w=130, thickness=12),
        extras=("dots", "think_eyes"),
    ),

    "speaking": _face(
        _L, _R,
        Mouth("o", w=170, h=MOUTH_T),
        extras=("speak",),
    ),

    "confused": _face(
        Eye(w=152, h=152, radius=32, dy=-8),
        Eye(w=104, h=104, radius=22, dy=10),
        Mouth("wavy", w=190, thickness=14, curve=13),
        extras=("question",),
    ),

    "dizzy": _face(
        *_eyes(kind="spiral", w=136, h=136),
        Mouth("wavy", w=210, thickness=14, curve=15),
        extras=("spin",),
        blink=False,
    ),

    "success": _face(
        # Only a slight happy squint -- close to the neutral square, not
        # the deep flatten used elsewhere.
        *_eyes(h=104, radius=24),
        Mouth("curve", w=270, thickness=18, curve=64),
        extras=("sparkle_brief",),
    ),

    # --- personality -------------------------------------------------
    "excited": _face(
        *_eyes(w=152, h=156, radius=32),
        Mouth("grin", w=210, h=96, dy=14),
        extras=("sparkle", "bounce"),
    ),

    "wink": _face(
        Eye(),
        Eye(h=22, radius=11, dy=26),
        Mouth("smirk", w=200, thickness=15, curve=34),
    ),

    "love": _face(
        *_eyes(kind="heart", w=150, h=136),
        Mouth("curve", w=230, thickness=16, curve=46),
        extras=("pulse",),
        blink=False,
    ),

    "curious": _face(
        Eye(h=140, radius=30, dy=-14),
        Eye(h=124, radius=28, dy=6),
        Mouth("bar", w=170),
        tilt=9.0,
    ),

    "proud": _face(
        *_eyes(h=66, radius=22, dy=-6),
        Mouth("smirk", w=200, thickness=16, curve=30),
    ),

    "bored": _face(
        *_eyes(h=64, radius=22, dy=10),
        Mouth("bar", w=230),
    ),

    # --- low energy --------------------------------------------------
    "sleepy": _face(
        *_eyes(h=26, radius=13, dy=16),
        Mouth("bar", w=70),
        extras=("z_one",),
        blink=False,
    ),

    "sleeping": _face(
        *_eyes(kind="arc", w=130, h=44),
        Mouth("bar", w=90),
        extras=("zzz", "breathe"),
        blink=False,
    ),

    "scared": _face(
        *_eyes(w=150, h=166, radius=32, pupil=0.20),
        Mouth("wavy", w=200, thickness=14, curve=15),
        extras=("shake",),
    ),
}

# The face and face_control must agree on the name list, always.
_missing = [n for n in EMOTION_NAMES if n not in EMOTIONS]
_extra = [n for n in EMOTIONS if n not in EMOTION_NAMES]
if _missing or _extra:
    raise SystemExit(
        f"emotion list mismatch — missing {_missing}, unexpected {_extra}"
    )


# =====================================================================
# Drawing primitives
# =====================================================================

def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease(t: float) -> float:
    """Ease in and out, so transitions start and stop softly."""
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _new(w: int, h: int) -> pygame.Surface:
    return pygame.Surface((max(1, int(w)), max(1, int(h))), pygame.SRCALPHA)


def _shrink(surf: pygame.Surface) -> pygame.Surface:
    """Undo the supersample. This is what gives the shapes clean edges."""
    w, h = surf.get_size()
    return pygame.transform.smoothscale(
        surf, (max(1, w // SUPERSAMPLE), max(1, h // SUPERSAMPLE))
    )


def _blur(surf: pygame.Surface, amount: int = BLUR_AMOUNT) -> pygame.Surface:
    """Cheap blur: shrink then grow back. Used for the gloss and the shadow."""
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(
        surf, (max(1, w // amount), max(1, h // amount)))
    return pygame.transform.smoothscale(small, (w, h))


def _gradient_seed() -> pygame.Surface:
    seed = _new(1, 2)
    seed.set_at((0, 0), (*GRADIENT_TOP, 255))
    seed.set_at((0, 1), (*GRADIENT_BOTTOM, 255))
    return seed


_GRADIENT_SEED = _gradient_seed()


def _finish(mask: pygame.Surface, glossy: bool = False) -> pygame.Surface:
    """Turn a flat white silhouette into a dimensional one: a top-lit
    vertical gradient, an upper-left gloss, and a soft drop shadow. Runs on
    the supersampled mask so the shrink at the end also smooths the finish.
    Returns a padded surface -- callers just centre on it as usual."""
    w, h = mask.get_size()
    pad = int(SUPERSAMPLE * (max(SHADOW_OFFSET) + 4 * BLUR_AMOUNT))
    canvas = _new(w + 2 * pad, h + 2 * pad)

    # 1. drop shadow: the silhouette recoloured, blurred, offset, and laid
    #    down first so the coloured shape below covers most of it.
    shadow = _new(w, h)
    shadow.fill((*SHADOW_COLOR, SHADOW_ALPHA))
    shadow.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    shadow = _blur(shadow)
    ox = pad + SHADOW_OFFSET[0] * SUPERSAMPLE
    oy = pad + SHADOW_OFFSET[1] * SUPERSAMPLE
    canvas.blit(shadow, (ox, oy))

    # 2. the shape itself, tinted with a vertical gradient (lighter on top).
    #    A 1x2 strip stretched by smoothscale interpolates it for us --
    #    far cheaper than a Python loop drawing one rect per row, and this
    #    runs on every cache miss (i.e. every frame of every tween).
    grad = pygame.transform.smoothscale(_GRADIENT_SEED, (w, h))
    colored = mask.copy()
    colored.blit(grad, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    canvas.blit(colored, (pad, pad))

    # 3. a soft gloss highlight near the upper-left, clipped to the shape.
    hi = _new(w, h)
    rect = pygame.Rect(0, 0, w * 0.72, h * 0.56)
    rect.center = (int(w * 0.33), int(h * 0.30))
    alpha = HIGHLIGHT_ALPHA_GLOSSY if glossy else HIGHLIGHT_ALPHA
    pygame.draw.ellipse(hi, (255, 255, 255, alpha), rect)
    hi = _blur(hi)
    hi.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
    canvas.blit(hi, (pad, pad))

    return canvas


def _thick_path(surf, points, thickness, color=MASK_WHITE) -> None:
    """Stamp a rounded-ended stroke along a point list."""
    r = max(1, int(round(thickness / 2)))
    for x, y in points:
        pygame.draw.circle(surf, color, (int(round(x)), int(round(y))), r)


def _rotate(surf: pygame.Surface, degrees: float) -> pygame.Surface:
    if abs(degrees) < 0.05:
        return surf
    return pygame.transform.rotate(surf, degrees)


# --- eye silhouettes -------------------------------------------------

def _eye_surface(eye: Eye) -> pygame.Surface:
    """One eye, drawn oversized then shrunk. Result is already dimensional."""
    s = SUPERSAMPLE
    w = max(2, int(round(eye.w * s)))
    h = max(2, int(round(eye.h * s)))
    glossy = eye.kind in ("heart", "spiral")

    if eye.kind == "heart":
        mask = _new(w, h)
        pts = []
        for i in range(121):
            a = 2 * math.pi * i / 120
            hx = 16 * math.sin(a) ** 3
            hy = (13 * math.cos(a) - 5 * math.cos(2 * a)
                  - 2 * math.cos(3 * a) - math.cos(4 * a))
            pts.append((w / 2 + hx * w / 34, h / 2 - hy * h / 30))
        pygame.draw.polygon(mask, MASK_WHITE, pts)

    elif eye.kind == "spiral":
        mask = _new(w, h)
        turns, steps = 2.6, 190
        thick = max(2, int(9 * s))
        outer = (min(w, h) - thick) / 2
        pts = []
        for i in range(steps + 1):
            a = turns * 2 * math.pi * i / steps
            rad = outer * i / steps
            pts.append((w / 2 + rad * math.cos(a), h / 2 + rad * math.sin(a)))
        _thick_path(mask, pts, thick)

    elif eye.kind == "arc":
        # A closed, sleeping eye: peak in the middle, ends dropping away.
        thick = max(2, int(round(min(eye.h, 20) * s)))
        mask = _new(w, h + thick)
        span, depth = w - thick, h
        pts = []
        for i in range(81):
            u = i / 80 * 2 - 1                      # -1 .. 1
            pts.append((thick / 2 + (u + 1) / 2 * span,
                        thick / 2 + depth * (u * u)))
        _thick_path(mask, pts, thick)

    else:                                            # rounded square
        mask = _new(w, h)
        radius = int(round(min(eye.radius * s, min(w, h) / 2)))
        pygame.draw.rect(mask, MASK_WHITE, (0, 0, w, h), border_radius=radius)
        if eye.bevel:
            # A clean diagonal chamfer on ONE top corner -- not a rotation
            # of the whole eye. `keep` is opaque everywhere except a
            # triangle over the corner being cut; BLEND_RGBA_MIN intersects
            # it with the mask, punching a hard-edged notch with the other
            # three corners (and their rounding) untouched.
            cut = min(abs(eye.bevel) * s, min(w, h) * 0.85)
            keep = _new(w, h)
            if eye.bevel > 0:      # this eye's own top-RIGHT corner
                pts = [(0, 0), (w - cut, 0), (w, cut), (w, h), (0, h)]
            else:                  # this eye's own top-LEFT corner
                pts = [(cut, 0), (w, 0), (w, h), (0, h), (0, cut)]
            pygame.draw.polygon(keep, MASK_WHITE, pts)
            mask.blit(keep, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        if eye.taper:
            # Same chamfer technique, on ONE bottom corner instead -- the
            # "happy squint" trapezoid.
            cut = min(abs(eye.taper) * s, min(w, h) * 0.85)
            keep = _new(w, h)
            if eye.taper > 0:      # this eye's own bottom-RIGHT corner
                pts = [(0, 0), (w, 0), (w, h - cut), (w - cut, h), (0, h)]
            else:                  # this eye's own bottom-LEFT corner
                pts = [(0, 0), (w, 0), (w, h), (cut, h), (0, h - cut)]
            pygame.draw.polygon(keep, MASK_WHITE, pts)
            mask.blit(keep, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    mask = _rotate(mask, eye.tilt)                    # rotate the SHAPE...
    art = _finish(mask, glossy=glossy)                # ...so the gradient
                                                       # underneath stays
                                                       # screen-vertical.
    if eye.pupil > 0:
        # Flat dark hole, added after the finish -- it's a marking on the
        # eye, not a shape that itself needs the gradient/gloss/shadow.
        pr = max(2, int(round(eye.pupil * w)))
        cx, cy = art.get_width() // 2, art.get_height() // 2
        pygame.draw.circle(art, PUPIL_COLOR, (cx, cy), pr)

    return _shrink(art)


# --- mouth silhouettes -----------------------------------------------

def _mouth_surface(mouth: Mouth) -> pygame.Surface:
    s = SUPERSAMPLE
    w = max(2, int(round(mouth.w * s)))
    thick = max(2, int(round(mouth.thickness * s)))
    curve = mouth.curve * s

    if mouth.kind == "o":
        h = max(2, int(round(mouth.h * s)))
        mask = _new(w, h)
        pygame.draw.ellipse(mask, MASK_WHITE, (0, 0, w, h))

    elif mouth.kind == "grin":
        # Open smile: the bottom half of an ellipse, so the top is flat.
        h = max(2, int(round(mouth.h * s)))
        tall = _new(w, h * 2)
        pygame.draw.ellipse(tall, MASK_WHITE, (0, 0, w, h * 2))
        mask = _new(w, h)
        mask.blit(tall, (0, -h))

    elif mouth.kind == "bar" or abs(curve) < 0.5:
        mask = _new(w, thick)
        pygame.draw.rect(mask, MASK_WHITE, (0, 0, w, thick),
                         border_radius=thick // 2)

    else:
        # Sample the stroke, then size the surface to the points we actually
        # got. Doing it the other way round silently clips the arc.
        span = w - thick
        pts = []
        for i in range(101):
            u = i / 100 * 2 - 1                      # -1 .. 1
            if mouth.kind == "wavy":
                y = curve * math.sin(2 * math.pi * 0.75 * u)
            elif mouth.kind == "smirk":
                y = curve * 0.75 * (1 - u * u) - curve * 0.45 * u
            else:                                    # "curve": smile / frown
                y = curve * (1 - u * u)
            pts.append((thick / 2 + (u + 1) / 2 * span, y))

        lo = min(y for _, y in pts)
        hi = max(y for _, y in pts)
        mask = _new(w, hi - lo + thick)
        _thick_path(mask, [(x, y - lo + thick / 2) for x, y in pts], thick)

    return _shrink(_finish(mask))


# --- cache -----------------------------------------------------------
# Static emotions redraw nothing; only tweens and animated extras miss.

_CACHE: dict = {}
_CACHE_MAX = 500


def _cached(key, build):
    surf = _CACHE.get(key)
    if surf is None:
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.clear()
        surf = _CACHE[key] = build()
    return surf


def _eye_art(eye: Eye) -> pygame.Surface:
    key = ("e", eye.kind, round(eye.w), round(eye.h), round(eye.radius),
           round(eye.tilt, 1), round(eye.bevel), round(eye.taper),
           round(eye.pupil, 2))
    return _cached(key, lambda: _eye_surface(eye))


def _mouth_art(mouth: Mouth) -> pygame.Surface:
    key = ("m", mouth.kind, round(mouth.w), round(mouth.h),
           round(mouth.thickness), round(mouth.curve, 1))
    return _cached(key, lambda: _mouth_surface(mouth))


# =====================================================================
# Tweening
# =====================================================================

def _lerp_eye(a: Eye, b: Eye, t: float) -> Eye:
    return Eye(
        kind=b.kind if t >= 0.5 else a.kind,
        w=_lerp(a.w, b.w, t), h=_lerp(a.h, b.h, t),
        radius=_lerp(a.radius, b.radius, t),
        tilt=_lerp(a.tilt, b.tilt, t),
        bevel=_lerp(a.bevel, b.bevel, t),
        taper=_lerp(a.taper, b.taper, t),
        dx=_lerp(a.dx, b.dx, t), dy=_lerp(a.dy, b.dy, t),
        pupil=_lerp(a.pupil, b.pupil, t),
    )


def _lerp_mouth(a: Mouth, b: Mouth, t: float) -> Mouth:
    return Mouth(
        kind=b.kind if t >= 0.5 else a.kind,
        w=_lerp(a.w, b.w, t), h=_lerp(a.h, b.h, t),
        thickness=_lerp(a.thickness, b.thickness, t),
        curve=_lerp(a.curve, b.curve, t),
        dx=_lerp(a.dx, b.dx, t), dy=_lerp(a.dy, b.dy, t),
    )


def _lerp_face(a: Face, b: Face, t: float) -> Face:
    """Sizes and angles interpolate. Silhouette changes cross-fade instead,
    which is handled at draw time -- here we just carry both kinds."""
    return Face(
        left=_lerp_eye(a.left, b.left, t),
        right=_lerp_eye(a.right, b.right, t),
        mouth=_lerp_mouth(a.mouth, b.mouth, t),
        eye_gap=_lerp(a.eye_gap, b.eye_gap, t),
        eye_y=_lerp(a.eye_y, b.eye_y, t),
        mouth_y=_lerp(a.mouth_y, b.mouth_y, t),
        tilt=_lerp(a.tilt, b.tilt, t),
        extras=b.extras if t >= 0.5 else a.extras,
        blink=b.blink if t >= 0.5 else a.blink,
    )


# =====================================================================
# Overlay glyphs — the "?" , the "z"s, the sparkles
# =====================================================================

_FONTS: dict = {}


def _font(size: int) -> pygame.font.Font:
    f = _FONTS.get(size)
    if f is None:
        f = _FONTS[size] = pygame.font.SysFont(None, size, bold=True)
    return f


def _glyph(surf, text, size, x, y, alpha=255) -> None:
    art = _font(size).render(text, True, ORANGE)
    if alpha < 255:
        art = art.copy()
        art.set_alpha(int(alpha))
    surf.blit(art, art.get_rect(center=(int(x), int(y))))


def _dot_art(diameter: int) -> pygame.Surface:
    def build():
        s = SUPERSAMPLE
        d = diameter * s
        mask = _new(d, d)
        pygame.draw.circle(mask, MASK_WHITE, (d // 2, d // 2), d // 2)
        return _shrink(_finish(mask))
    return _cached(("dot", diameter), build)


def _sparkle(surf, x, y, r, alpha=255) -> None:
    """A four-pointed star, the same flat orange as everything else."""
    pts = []
    for i in range(8):
        a = math.pi * i / 4
        rad = r if i % 2 == 0 else r * 0.26
        pts.append((x + rad * math.cos(a), y + rad * math.sin(a)))
    if alpha >= 255:
        pygame.draw.polygon(surf, ORANGE, pts)
        return
    layer = _new(int(r * 2 + 4), int(r * 2 + 4))
    pygame.draw.polygon(layer, ORANGE,
                        [(px - x + r + 2, py - y + r + 2) for px, py in pts])
    layer.set_alpha(int(alpha))
    surf.blit(layer, (int(x - r - 2), int(y - r - 2)))


# =====================================================================
# The face
# =====================================================================

class FridayFace:
    """Renders Friday's current emotion. One public method: set_emotion()."""

    def __init__(self, fullscreen: bool = True,
                 size: tuple[int, int] = (BASE_W, BASE_H)) -> None:
        pygame.init()
        pygame.display.set_caption("Friday")

        # Draw everything on a fixed BASE_W x BASE_H canvas, then scale that
        # onto the real screen ourselves (see _present()). Letting SDL's
        # FULLSCREEN | SCALED do this under Wayland (labwc) produced a
        # cropped picture -- its letterboxing didn't come out right on this
        # panel -- so we do the aspect-preserving fit by hand instead.
        flags = pygame.FULLSCREEN if fullscreen else 0
        self.screen = pygame.display.set_mode((0, 0) if fullscreen else size,
                                              flags)
        self.canvas = pygame.Surface((BASE_W, BASE_H))
        if fullscreen:
            pygame.mouse.set_visible(False)

        self.clock = pygame.time.Clock()
        self.running = True

        self._name = DEFAULT_EMOTION
        self._from = self._to = EMOTIONS[self._name]
        self._t = 1.0                     # 0 = start of tween, 1 = arrived
        self._elapsed = 0.0               # since start, drives animation
        self._since_change = 0.0          # since the last set_emotion()

        self._blink_next = random.uniform(BLINK_GAP_MIN_S, BLINK_GAP_MAX_S)
        self._blink_t = None              # seconds into the current blink

        self._poll_next = 0.0
        self._file_stamp = None

        self._yawn_start = None       # elapsed-time timestamp, or None
        self._yawn_poll_next = 0.0
        self._yawn_stamp = None

    # --- control -----------------------------------------------------

    @property
    def emotion(self) -> str:
        return self._name

    def set_emotion(self, name: str) -> None:
        """Switch emotion. Always tweens; never a hard jump."""
        name = name.strip().lower()
        if name not in EMOTIONS:
            raise KeyError(f"unknown emotion {name!r}")
        if name == self._name:
            return
        self._from = self._current_shape()   # tween out of where we ARE,
        self._to = EMOTIONS[name]            # not where we were headed
        self._name = name
        self._t = 0.0
        self._since_change = 0.0

    def _current_shape(self) -> Face:
        if self._t >= 1.0:
            return self._to
        return _lerp_face(self._from, self._to, _ease(self._t))

    def _poll_state_file(self, now: float) -> None:
        """Cheap: stat first, only read when the file actually changed."""
        if now < self._poll_next:
            return
        self._poll_next = now + POLL_S
        try:
            st = os.stat(EMOTION_FILE)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return
        if stamp == self._file_stamp:
            return
        self._file_stamp = stamp
        try:
            name = EMOTION_FILE.read_text().strip().lower()
        except (OSError, UnicodeDecodeError):
            return
        if name in EMOTIONS:
            self.set_emotion(name)

    def play_yawn(self) -> None:
        """Trigger the yawn overlay. Plays on top of whatever emotion is
        showing, then returns to it -- never touches self._name."""
        self._yawn_start = self._elapsed

    def _yawn_amount(self) -> float:
        """0 = not yawning, 1 = fully into the yawn shape, eased in between."""
        if self._yawn_start is None:
            return 0.0
        dt = self._elapsed - self._yawn_start
        if dt < YAWN_IN_S:
            return _ease(dt / YAWN_IN_S)
        if dt < YAWN_IN_S + YAWN_HOLD_S:
            return 1.0
        if dt < YAWN_IN_S + YAWN_HOLD_S + YAWN_OUT_S:
            return 1.0 - _ease((dt - YAWN_IN_S - YAWN_HOLD_S) / YAWN_OUT_S)
        self._yawn_start = None
        return 0.0

    def _poll_yawn_file(self, now: float) -> None:
        if now < self._yawn_poll_next:
            return
        self._yawn_poll_next = now + POLL_S
        try:
            st = os.stat(YAWN_FILE)
            stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            return
        if stamp == self._yawn_stamp:
            return
        self._yawn_stamp = stamp
        self.play_yawn()

    # --- per-frame state ---------------------------------------------

    def update(self, dt: float) -> None:
        self._elapsed += dt
        self._since_change += dt

        if self._t < 1.0:
            self._t = min(1.0, self._t + dt / TRANSITION_S)

        if self._blink_t is not None:
            self._blink_t += dt
            if self._blink_t >= BLINK_CLOSED_S:
                self._blink_t = None
                self._blink_next = random.uniform(BLINK_GAP_MIN_S,
                                                  BLINK_GAP_MAX_S)
        else:
            self._blink_next -= dt
            if self._blink_next <= 0:
                self._blink_t = 0.0

    def _blink_factor(self, face: Face) -> float:
        """1.0 open, ~0 shut. Only rounded-square eyes blink."""
        if self._blink_t is None or not face.blink:
            return 1.0
        half = BLINK_CLOSED_S / 2
        p = (self._blink_t / half if self._blink_t < half
             else (BLINK_CLOSED_S - self._blink_t) / half)
        return 1.0 - 0.92 * max(0.0, min(1.0, p))

    def _animate(self, face: Face) -> tuple[Face, float, float]:
        """Apply the animated extras. Returns the face plus a whole-face
        offset for the ones that move it (bounce, shake, breathe)."""
        t, ex = self._elapsed, face.extras
        left, right, mouth = face.left, face.right, face.mouth
        ox = oy = 0.0

        if "pulse_eyes" in ex:
            k = 1 + 0.07 * math.sin(2 * math.pi * t / 1.6)
            left, right = replace(left, h=left.h * k), replace(right, h=right.h * k)
        if "think_eyes" in ex:
            # A deep, deliberate open/contract squint -- eyes shrink almost
            # to a flat bar and open back up, much more pronounced than the
            # subtle pulse_eyes wobble used elsewhere.
            cyc = (1 + math.sin(2 * math.pi * t / 2.6)) / 2   # 0..1
            k = 0.32 + 0.68 * cyc
            left = replace(left, h=left.h * k, radius=left.radius * k)
            right = replace(right, h=right.h * k, radius=right.radius * k)
        if "pulse" in ex:
            k = 1 + 0.06 * math.sin(2 * math.pi * t / 1.4)
            left = replace(left, w=left.w * k, h=left.h * k)
            right = replace(right, w=right.w * k, h=right.h * k)
        if "spin" in ex:
            # Quantised so the spiral art stays cacheable instead of being
            # rebuilt every single frame.
            a = round((t * 150) / 6) * 6 % 360
            left, right = replace(left, tilt=a), replace(right, tilt=a)
        if "speak" in ex:
            env = ((0.5 + 0.5 * math.sin(2 * math.pi * t * 3.1))
                   * (0.5 + 0.5 * math.sin(2 * math.pi * t * 1.7 + 1.1)))
            mouth = replace(mouth, h=MOUTH_T + 62 * env)

        if "bounce" in ex:
            oy -= 7 * abs(math.sin(2 * math.pi * t / 0.9))
        if "shake" in ex:
            ox += 3 * math.sin(2 * math.pi * t * 11)
        if "breathe" in ex:
            oy += 5 * math.sin(2 * math.pi * t / 3.4)

        k = self._blink_factor(face)
        if k < 1.0:
            for name, eye in (("l", left), ("r", right)):
                if eye.kind != "rect":
                    continue
                h = max(4.0, eye.h * k)
                shut = replace(eye, h=h, radius=min(eye.radius, h / 2))
                left, right = (shut, right) if name == "l" else (left, shut)

        return replace(face, left=left, right=right, mouth=mouth), ox, oy

    # --- drawing -----------------------------------------------------

    @staticmethod
    def _variants(base, from_kind, to_kind, p, fading):
        """During a tween between two different silhouettes, cross-fade the
        old shape out and the new one in. Same silhouette -> one opaque draw."""
        if not fading or from_kind == to_kind:
            return ((replace(base, kind=to_kind), 255),)
        return ((replace(base, kind=from_kind), int(255 * (1 - p))),
                (replace(base, kind=to_kind), int(255 * p)))

    def _blit(self, art, cx, cy, alpha, spin=0.0) -> None:
        if alpha <= 0:
            return
        if abs(spin) > 0.05:
            art = pygame.transform.rotate(art, spin)
        if alpha < 255:
            art = art.copy()
            art.set_alpha(alpha)
        self.canvas.blit(art, art.get_rect(center=(int(cx), int(cy))))

    def draw(self) -> None:
        self.canvas.fill(BG_COLOR)

        fading = self._t < 1.0
        p = _ease(self._t)
        face, ox, oy = self._animate(self._current_shape())

        y = self._yawn_amount()
        if y > 0:
            yawn_eye = Eye(kind="rect", w=YAWN_EYE_W, h=YAWN_EYE_H,
                           radius=YAWN_EYE_RADIUS)
            yawn_mouth = Mouth(kind="o", w=YAWN_MOUTH_W, h=YAWN_MOUTH_H)
            face = replace(
                face,
                left=_lerp_eye(face.left, yawn_eye, y),
                right=_lerp_eye(face.right, yawn_eye, y),
                mouth=_lerp_mouth(face.mouth, yawn_mouth, y),
            )

        cx0 = BASE_W / 2
        cy0 = (face.eye_y + face.mouth_y) / 2
        rot = math.radians(face.tilt)
        cos_a, sin_a = math.cos(rot), math.sin(rot)

        def place(x, y):
            """Rotate a point about the face centre, then apply the offset."""
            dx, dy = x - cx0, y - cy0
            return (cx0 + dx * cos_a + dy * sin_a + ox,
                    cy0 - dx * sin_a + dy * cos_a + oy)

        # While yawning, the eye/mouth kind is decided by the yawn amount,
        # not by the settled emotion -- and the two cross-fades don't
        # compose, so yawning suspends the emotion cross-fade (in practice
        # yawn only fires once an emotion has already settled anyway).
        eye_fading = fading and y <= 0
        for eye, side, src_kind, dst_kind in (
            (face.left, -1, self._from.left.kind,
             "rect" if y >= 0.5 else self._to.left.kind),
            (face.right, +1, self._from.right.kind,
             "rect" if y >= 0.5 else self._to.right.kind),
        ):
            ex = cx0 + side * (face.eye_gap / 2 + eye.w / 2) + eye.dx
            ey = face.eye_y + eye.dy
            px, py = place(ex, ey)
            for shape, alpha in self._variants(eye, src_kind, dst_kind,
                                               p, eye_fading):
                self._blit(_eye_art(shape), px, py, alpha, spin=face.tilt)

        mouth = face.mouth
        mouth_dst_kind = "o" if y >= 0.5 else self._to.mouth.kind
        mx, my = place(cx0 + mouth.dx, face.mouth_y + mouth.dy)
        for shape, alpha in self._variants(
                mouth, self._from.mouth.kind, mouth_dst_kind, p, eye_fading):
            self._blit(_mouth_art(shape), mx, my, alpha, spin=face.tilt)

        self._overlays(face, ox, oy)

    def _overlays(self, face: Face, ox: float, oy: float) -> None:
        t, ex = self._elapsed, face.extras
        if not ex:
            return
        cx = BASE_W / 2 + ox
        top = face.eye_y + oy

        if "dots" in ex:
            for i in range(3):
                phase = (t * 1.5 - i * 0.22) % 1.0
                lift = 8 * math.sin(math.pi * min(1.0, phase * 1.6))
                alpha = 90 + 165 * math.sin(math.pi * min(1.0, phase * 1.6))
                dot = _dot_art(22).copy()
                dot.set_alpha(int(alpha))
                self.canvas.blit(dot, dot.get_rect(
                    center=(int(cx + (i - 1) * 52), int(top - 132 - lift))))

        if "question" in ex:
            bob = 9 * math.sin(2 * math.pi * t / 1.8)
            _glyph(self.canvas, "?", 170, cx + 312, top - 46 + bob)

        if "sparkle" in ex:
            spots = ((-330, -96, 26), (318, -110, 30),
                     (-268, 108, 20), (296, 96, 22))
            for i, (dx, dy, r) in enumerate(spots):
                tw = 0.55 + 0.45 * math.sin(2 * math.pi * t / 1.1 + i * 1.7)
                _sparkle(self.canvas, cx + dx, top + dy, r * tw, 255 * tw)

        if "sparkle_brief" in ex:
            age = self._since_change
            if age < 1.3:
                fade = max(0.0, 1.0 - age / 1.3)
                for i, (dx, dy, r) in enumerate(
                        ((-306, -84, 30), (300, -96, 26), (0, -190, 22))):
                    _sparkle(self.canvas, cx + dx, top + dy,
                             r * (0.5 + 0.5 * fade), 255 * fade)

        if "z_one" in ex:
            phase = (t / 2.8) % 1.0
            if phase < 0.55:
                k = phase / 0.55
                _glyph(self.canvas, "z", 58, cx + 268 + 26 * k,
                       top - 60 - 74 * k, 255 * (1 - k))

        if "zzz" in ex:
            for i, size in enumerate((44, 62, 82)):
                phase = (t / 3.2 + i * 0.28) % 1.0
                _glyph(self.canvas, "z", int(size * (0.7 + 0.4 * phase)),
                       cx + 244 + 66 * phase,
                       top - 34 - 150 * phase,
                       255 * min(1.0, (1 - phase) * 2.2))

    def _present(self) -> None:
        """Scale the BASE_W x BASE_H canvas to fit the real screen, aspect
        preserved, and centre it -- done by hand so the whole picture always
        lands on screen instead of being cropped by SDL/Wayland scaling."""
        sw, sh = self.screen.get_size()
        scale = min(sw / BASE_W, sh / BASE_H)
        w, h = round(BASE_W * scale), round(BASE_H * scale)
        frame = pygame.transform.smoothscale(self.canvas, (w, h))
        self.screen.fill(BG_COLOR)
        self.screen.blit(frame, ((sw - w) // 2, (sh - h) // 2))
        pygame.display.flip()

    # --- loop --------------------------------------------------------

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key in (
                    pygame.K_ESCAPE, pygame.K_q):
                self.running = False

    def run(self, follow_file: bool = True) -> None:
        now = 0.0
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            now += dt
            self.handle_events()
            if follow_file:
                self._poll_state_file(now)
                self._poll_yawn_file(now)
            self.update(dt)
            self.draw()
            self._present()
        pygame.quit()

    def run_demo(self, hold: float = DEMO_HOLD_S) -> None:
        order, index, since = list(EMOTION_NAMES), 0, 0.0
        self.set_emotion(order[0])
        yawn_at = order.index("sleepy") if "sleepy" in order else 0
        yawned = False
        while self.running:
            dt = self.clock.tick(FPS) / 1000.0
            self.handle_events()
            since += dt
            if index == yawn_at and not yawned and since >= hold * 0.35:
                self.play_yawn()
                yawned = True
            if since >= hold:
                since = 0.0
                index = (index + 1) % len(order)
                self.set_emotion(order[index])
            self.update(dt)
            self.draw()
            self._present()
        pygame.quit()


# =====================================================================
# CLI
# =====================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Friday's face")
    ap.add_argument("--demo", action="store_true",
                    help="cycle all 20 emotions and play the yawn once")
    ap.add_argument("--window", action="store_true",
                    help="windowed, for testing over SSH/VNC")
    args = ap.parse_args()

    face = FridayFace(fullscreen=not args.window)
    if args.demo:
        face.run_demo()
    else:
        face.set_emotion(get_emotion())
        face.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
