"""friday_face.py — Friday's face on the 7" display.

    python3 friday_face.py            fullscreen, follows state/emotion.txt
    python3 friday_face.py --demo     cycle all 20 emotions + play the yawn
    python3 friday_face.py --window   windowed, for testing over SSH/VNC

Nothing drives this directly. It watches state/emotion.txt (and
state/yawn.txt for the yawn trigger) and switches to whatever it finds
there; use face_control.py to write those files.

Retro-futuristic block geometry on a pure black screen: solid deep-amber
shapes, sharp corners, flat fills. No gradients, no gloss, no shadows, no
corner rounding. The five reference moods (neutral, thinking, happy, sad,
listening) and the speaking mouth reproduce hardware/saras_face.py's
measured geometry exactly -- those proportions were derived from photos of
the real panel, so they are copied verbatim rather than re-invented. The
remaining emotions are built from the same primitives in the same
language.

Geometry is proportional to the actual screen size, so nothing is ever
cropped or letterboxed regardless of what the panel reports.
"""

from __future__ import annotations

import argparse
import math
import os
import random

import pygame

from face_control import (EMOTIONS as EMOTION_NAMES, DEFAULT_EMOTION,
                          EMOTION_FILE, YAWN_FILE, get_emotion)

# =====================================================================
# COLOURS — tune these
# =====================================================================
BG_COLOR = (0, 0, 0)              # pure black
FG_COLOR = (255, 106, 0)          # #FF6A00 deep amber -- matches the chassis

# =====================================================================
# TIMING — tune these
# =====================================================================
FPS = 60
TRANSITION_S = 0.28               # cross-fade between emotions
BLINK_MIN_INTERVAL = 2.0
BLINK_MAX_INTERVAL = 6.0
BLINK_DURATION = 0.16
POLL_S = 0.25                     # how often the state files are checked
DEMO_HOLD_S = 2.6                 # --demo dwell per emotion

# --- yawn overlay ----------------------------------------------------
# Not one of the 20 emotions -- a transient reflex that plays on top of
# whatever emotion is showing, then returns to it. See play_yawn().
YAWN_IN_S = 0.18
YAWN_HOLD_S = 0.62
YAWN_OUT_S = 0.30

# The real 7" HDMI TFT panel (EDID "Mediatrix Peripherals Inc MPI7002")
# reports 1920x1080 as its native mode -- confirmed here via wlr-randr.
# Geometry below is proportional, so any of these renders correctly.
SCREEN_CANDIDATES = [(1920, 1080), (1024, 600), (800, 480)]
DEV_PREVIEW_SIZE = (960, 540)     # exact half-scale of the panel, 16:9


def _detect_screen_size() -> tuple[tuple[int, int], bool]:
    pygame.display.init()
    info = pygame.display.Info()
    current = (info.current_w, info.current_h)
    if current in SCREEN_CANDIDATES:
        return current, True          # the real panel -> fullscreen native
    return DEV_PREVIEW_SIZE, False    # a dev monitor -> small test window


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


class FridayFace:
    """Renders Friday's current emotion. One public method: set_emotion()."""

    def __init__(self, fullscreen: bool | None = True,
                 screen_size: tuple[int, int] | None = None) -> None:
        pygame.init()
        pygame.display.set_caption("Friday")

        detected, is_real_panel = _detect_screen_size()
        size = screen_size or detected
        use_fullscreen = is_real_panel if fullscreen is None else fullscreen

        if use_fullscreen:
            self.screen = pygame.display.set_mode(size, pygame.FULLSCREEN)
            pygame.mouse.set_visible(False)
        else:
            os.environ.setdefault("SDL_VIDEO_WINDOW_POS", "center")
            self.screen = pygame.display.set_mode(size)

        self.width, self.height = self.screen.get_size()
        self.clock = pygame.time.Clock()
        self.running = True

        self._layout()

        self._name = DEFAULT_EMOTION
        self._elapsed = 0.0
        self._since_change = 0.0

        # Cross-fade: a snapshot of the last frame of the previous emotion,
        # faded out over the new one. Shapes stay pixel-exact throughout --
        # nothing is interpolated geometrically.
        self._fade_from = None
        self._fade_t = 1.0

        self._blink_next = random.uniform(BLINK_MIN_INTERVAL,
                                          BLINK_MAX_INTERVAL)
        self._blink_t = None

        self._poll_next = 0.0
        self._file_stamp = None
        self._yawn_start = None
        self._yawn_poll_next = 0.0
        self._yawn_stamp = None

        # Per-frame whole-face transform, applied by _xf() to every shape:
        # shake (scared), breathing bob (sleeping), head tilt (curious).
        self._off_x = 0.0
        self._off_y = 0.0
        self._tilt = 0.0

        self._fonts: dict[int, pygame.font.Font] = {}

    # --- geometry ----------------------------------------------------

    def _layout(self) -> None:
        """All proportional to the real screen, so nothing is ever cropped.
        These ratios come from hardware/saras_face.py, where they were
        measured off photographs of the panel -- do not re-derive them."""
        w, h = self.width, self.height
        cx = w // 2

        self.eye_size = int(min(w, h) * 0.22)
        self.eye_gap = int(w * 0.16)
        self.eye_y = int(h * 0.30)
        self.left_eye_x = cx - self.eye_gap // 2 - self.eye_size
        self.right_eye_x = cx + self.eye_gap // 2

        # happy has its own, larger, closer-set eyes and its own vertical
        # nudge -- measured separately from the shared values above.
        self.happy_eye_size = int(min(w, h) * 0.32)
        self.happy_y_shift = int(h * 0.075)
        self.happy_eye_gap = int(self.happy_eye_size * 0.47)

        self.mouth_width = int(w * 0.46)
        self.mouth_thin_h = max(6, int(h * 0.02))
        self.mouth_tall_h = int(h * 0.16)
        self.mouth_y = int(h * 0.70)

    # --- control -----------------------------------------------------

    @property
    def emotion(self) -> str:
        return self._name

    def set_emotion(self, name: str) -> None:
        """Switch emotion. Cross-fades; never a hard jump."""
        name = name.strip().lower()
        if name not in EMOTION_NAMES:
            raise KeyError(f"unknown emotion {name!r}")
        if name == self._name:
            return
        self._fade_from = self.screen.copy()
        self._fade_t = 0.0
        self._name = name
        self._since_change = 0.0

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
        if name in EMOTION_NAMES:
            self.set_emotion(name)

    def play_yawn(self) -> None:
        """Trigger the yawn overlay. Plays on top of whatever emotion is
        showing, then returns to it -- never touches self._name."""
        self._yawn_start = self._elapsed

    def _yawn_amount(self) -> float:
        """0 = not yawning, 1 = fully into the yawn shape."""
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

        if self._fade_t < 1.0:
            self._fade_t = min(1.0, self._fade_t + dt / TRANSITION_S)
            if self._fade_t >= 1.0:
                self._fade_from = None

        if self._blink_t is not None:
            self._blink_t += dt
            if self._blink_t >= BLINK_DURATION:
                self._blink_t = None
                self._blink_next = random.uniform(BLINK_MIN_INTERVAL,
                                                  BLINK_MAX_INTERVAL)
        else:
            self._blink_next -= dt
            if self._blink_next <= 0:
                self._blink_t = 0.0

    def _blink_factor(self) -> float:
        """1.0 = eyes open, ~0.0 = fully closed. Triangle wave."""
        if self._blink_t is None or self._name in _NO_BLINK:
            return 1.0
        t = self._blink_t / BLINK_DURATION
        return abs(2 * t - 1)

    # --- drawing primitives ------------------------------------------

    def _xf(self, x, y) -> tuple[float, float]:
        """Apply this frame's head tilt and offset to one point. With no
        tilt and no offset this is the identity, so the reference moods
        render exactly as they did before any of this existed."""
        if self._tilt:
            cx = self.width / 2
            cy = (self.eye_y + self.mouth_y) / 2
            dx, dy = x - cx, y - cy
            a = math.radians(self._tilt)
            ca, sa = math.cos(a), math.sin(a)
            x, y = cx + dx * ca - dy * sa, cy + dx * sa + dy * ca
        return x + self._off_x, y + self._off_y

    def _rect_eye(self, x, y, w, h, color=FG_COLOR) -> None:
        """A sharp, flat, filled rectangle. No rounding anywhere. Falls back
        to a rotated polygon when the face is tilted."""
        if w < 1 or h < 1:
            return
        if self._tilt:
            self._poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)],
                       color=color)
            return
        ox, oy = self._xf(x, y)
        pygame.draw.rect(self.screen, color,
                         pygame.Rect(int(ox), int(oy), int(w), int(h)))

    def _rect_c(self, cx, cy, w, h, color=FG_COLOR) -> None:
        """A centred filled rectangle, routed through the transform so it
        picks up tilt/shake/bob like everything else."""
        if w < 1 or h < 1:
            return
        self._rect_eye(cx - w / 2, cy - h / 2, w, h, color=color)

    def _poly(self, points, color=FG_COLOR) -> None:
        pts = [self._xf(px, py) for px, py in points]
        pygame.draw.polygon(self.screen, color,
                            [(int(px), int(py)) for px, py in pts])

    def _thick_polyline(self, points, width) -> None:
        """Solid (non anti-aliased) thick stroke through the given points."""
        pts = [self._xf(px, py) for px, py in points]
        r = max(int(width) // 2, 1)
        for px, py in pts:
            pygame.draw.circle(self.screen, FG_COLOR, (int(px), int(py)), r)
        pygame.draw.lines(self.screen, FG_COLOR, False,
                          [(int(px), int(py)) for px, py in pts],
                          max(1, int(width)))

    # --- overlay glyphs and sparkles ---------------------------------

    def _font(self, size: int) -> pygame.font.Font:
        f = self._fonts.get(size)
        if f is None:
            if not pygame.font.get_init():
                pygame.font.init()
            f = self._fonts[size] = pygame.font.SysFont(None, size, bold=True)
        return f

    def _glyph(self, text, size, x, y, alpha=255) -> None:
        """A floating character -- the '?' and the 'z's."""
        art = self._font(int(size)).render(text, True, FG_COLOR)
        if alpha < 255:
            art = art.copy()
            art.set_alpha(int(max(0, min(255, alpha))))
        px, py = self._xf(x, y)
        self.screen.blit(art, art.get_rect(center=(int(px), int(py))))

    def _sparkle(self, x, y, r, alpha=255) -> None:
        """A sharp four-pointed star -- flat fill, same as everything else."""
        if r < 1 or alpha <= 0:
            return
        pts = []
        for i in range(8):
            a = math.pi * i / 4
            rad = r if i % 2 == 0 else r * 0.26
            pts.append((x + rad * math.cos(a), y + rad * math.sin(a)))
        if alpha >= 255:
            self._poly(pts)
            return
        pts = [self._xf(px, py) for px, py in pts]
        pad = int(r * 2 + 4)
        layer = pygame.Surface((pad, pad), pygame.SRCALPHA)
        ox, oy = self._xf(x, y)
        pygame.draw.polygon(layer, FG_COLOR,
                            [(int(px - ox + r + 2), int(py - oy + r + 2))
                             for px, py in pts])
        layer.set_alpha(int(max(0, min(255, alpha))))
        self.screen.blit(layer, (int(ox - r - 2), int(oy - r - 2)))

    def _blinked(self, y, size, factor):
        """Apply the blink squeeze to one eye box: returns (y, height)."""
        h = max(2, int(size * factor))
        return y + (size - h) // 2, h

    # --- eyes --------------------------------------------------------

    def _plain_eyes(self, size=None, dy=0, wide=None, pupil=0.0) -> None:
        """The default pair of solid squares, with blink applied. `pupil`
        punches a small dark square in each eye (scared), sized as a
        fraction of the eye."""
        size = self.eye_size if size is None else size
        w = size if wide is None else wide
        blink = self._blink_factor()
        y, h = self._blinked(self.eye_y + dy, size, blink)
        cx = self.width // 2
        lx = cx - self.eye_gap // 2 - w
        rx = cx + self.eye_gap // 2
        self._rect_eye(lx, y, w, h)
        self._rect_eye(rx, y, w, h)
        if pupil > 0 and h > 4:
            pw = max(2, int(w * pupil))
            ph = max(2, int(min(h * 0.9, w * pupil)))
            for ex in (lx, rx):
                self._rect_eye(ex + (w - pw) // 2, y + (h - ph) // 2,
                               pw, ph, color=BG_COLOR)

    def _bar_eyes(self, w_mul, h_mul, dy=0, pulse=0.0) -> None:
        """Thin wide horizontal bars -- listening / bored / sleepy / sleeping."""
        w = int(self.eye_size * w_mul)
        h = max(4, int(self.eye_size * h_mul))
        y = self.eye_y + (self.eye_size - h) // 2 + dy
        extra = 0
        if pulse:
            extra = int((math.sin(self._elapsed * 5) * 0.5 + 0.5) * h * pulse)
        for ex in (self.left_eye_x, self.right_eye_x):
            self._rect_c(ex + self.eye_size // 2, y + h // 2, w, h + extra)

    def _happy_eye_points(self, x, y, w, h, inner_side):
        """Uplifted anime-squint eye: flat top, a single straight diagonal
        bottom edge sloping upward from the outer corner (full height) to
        the inner corner near the nose bridge (shortened) -- a plain
        4-point trapezoid with no intermediate kink."""
        if inner_side == "right":          # left eye: inner edge on the right
            return [(x + w * 0.02, y + h * 0.02),
                    (x + w * 0.98, y + h * 0.02),
                    (x + w * 0.98, y + h * 0.40),
                    (x + w * 0.02, y + h * 0.97)]
        return [(x + w * 0.98, y + h * 0.02),   # right eye: mirrored
                (x + w * 0.02, y + h * 0.02),
                (x + w * 0.02, y + h * 0.40),
                (x + w * 0.98, y + h * 0.97)]

    def _draw_happy_eyes(self) -> None:
        """Solid filled trapezoids -- sharp edges, flat fill. Uses the
        happy-only size and gap measured off the reference photo."""
        w = self.happy_eye_size
        h = int(self.happy_eye_size * 0.81)
        y = self.eye_y - self.happy_y_shift + (self.happy_eye_size - h) // 2
        cx = self.width // 2
        lx = cx - self.happy_eye_gap // 2 - w
        rx = cx + self.happy_eye_gap // 2
        self._poly(self._happy_eye_points(lx, y, w, h, inner_side="right"))
        self._poly(self._happy_eye_points(rx, y, w, h, inner_side="left"))

    def _draw_thinking_eyes(self) -> None:
        """Left eye: right-trapezoid -- vertical outer edge, level bottom,
        inner-top corner sheared down ~15% of the height (a slanted top edge
        only, not a notch). Right eye: plain open square, untouched."""
        left_h = int(self.eye_size * 0.82)
        ly = self.eye_y + (self.eye_size - left_h) // 2
        lx = self.left_eye_x
        drop = int(left_h * 0.15)
        self._poly([(lx, ly),
                    (lx + self.eye_size, ly + drop),
                    (lx + self.eye_size, ly + left_h),
                    (lx, ly + left_h)])
        self._rect_eye(self.right_eye_x, self.eye_y,
                       self.eye_size, self.eye_size)

    def _draw_drooped_eyes(self, corner) -> None:
        """Square eyes with one top corner drooped down. corner="outer"
        gives sad's outward/downward brow; "inner" gives angry's."""
        blink = self._blink_factor()
        y, h = self._blinked(self.eye_y, self.eye_size, blink)
        droop = int(h * 0.35)
        for lx, is_left in ((self.left_eye_x, True),
                            (self.right_eye_x, False)):
            right = lx + self.eye_size
            drop_left = (corner == "outer") == is_left
            tl = (lx, y + droop) if drop_left else (lx, y)
            tr = (right, y) if drop_left else (right, y + droop)
            self._poly([tl, tr, (right, y + h), (lx, y + h)])

    def _draw_heart_eyes(self) -> None:
        """A sharp-edged block heart -- built from rectangles on a 5x5 grid,
        so it stays in the same flat pixel-art language as everything else:
        two lobes on top, tapering to a single point at the bottom."""
        u = self.eye_size // 5
        spans = (((0, 2), (3, 5)),   # row 0: two lobes with a dip between
                 ((0, 5),),          # row 1: solid across
                 ((1, 4),),          # row 2: tapering
                 ((2, 3),))          # row 3: the point
        for lx in (self.left_eye_x, self.right_eye_x):
            for row, row_spans in enumerate(spans):
                for a, b in row_spans:
                    self._rect_eye(lx + a * u, self.eye_y + row * u,
                                   (b - a) * u, u)

    def _draw_spiral_eyes(self) -> None:
        """Dizzy: a swirl that actually spins. Drawn as a thick spiral
        stroke whose start angle advances with time, so the whole swirl
        rotates."""
        s = self.eye_size
        thick = max(3, s // 9)
        outer = (s - thick) / 2
        spin = self._elapsed * 2.6
        turns, steps = 2.4, 64
        for lx in (self.left_eye_x, self.right_eye_x):
            ecx, ecy = lx + s / 2, self.eye_y + s / 2
            pts = []
            for i in range(steps + 1):
                a = spin + turns * 2 * math.pi * i / steps
                rad = outer * i / steps
                pts.append((ecx + rad * math.cos(a), ecy + rad * math.sin(a)))
            self._thick_polyline(pts, thick)

    def _draw_closed_arc_eyes(self, dy=0) -> None:
        """Sleeping: each eye closed into a down-turned arc -- peak in the
        middle, both ends dropping away."""
        w = int(self.eye_size * 1.05)
        depth = int(self.eye_size * 0.30)
        thick = max(4, int(self.eye_size * 0.13))
        y = self.eye_y + int(self.eye_size * 0.35) + dy
        for lx in (self.left_eye_x, self.right_eye_x):
            x0 = lx + (self.eye_size - w) // 2
            pts = []
            for i in range(41):                     # dense enough that the
                u = i / 40 * 2 - 1                  # stamped circles overlap
                pts.append((x0 + (u + 1) / 2 * w, y + depth * (u * u)))
            self._thick_polyline(pts, thick)

    # --- mouths ------------------------------------------------------

    def _mouth_bar(self, width_mul=1.0, dy=0) -> None:
        self._rect_c(self.width // 2, self.mouth_y + dy,
                     int(self.mouth_width * width_mul), self.mouth_thin_h)

    def _mouth_open(self, width_mul=1.0, height_mul=1.0) -> None:
        self._rect_c(self.width // 2, self.mouth_y,
                     int(self.mouth_width * width_mul),
                     int(self.mouth_tall_h * height_mul))

    def _mouth_speaking(self) -> None:
        """The mouth-morph: two detuned sines so it never looks periodic."""
        t = self._elapsed
        wobble = ((math.sin(t * 14) * 0.5 + 0.5) * 0.6
                  + (math.sin(t * 27.3) * 0.5 + 0.5) * 0.4)
        h = int(self.mouth_thin_h
                + (self.mouth_tall_h - self.mouth_thin_h) * wobble)
        self._rect_c(self.width // 2, self.mouth_y,
                     self.mouth_width, max(self.mouth_thin_h, h))

    def _mouth_frown(self) -> None:
        """Two straight segments meeting at a peak -- sad's downturned mouth."""
        cx = self.width // 2
        half_w = self.mouth_width // 2
        drop = int(self.height * 0.05)
        centre = (cx, self.mouth_y)
        pygame.draw.line(self.screen, FG_COLOR, centre,
                         (cx - half_w, self.mouth_y + drop), self.mouth_thin_h)
        pygame.draw.line(self.screen, FG_COLOR, centre,
                         (cx + half_w, self.mouth_y + drop), self.mouth_thin_h)

    def _mouth_smile(self) -> None:
        """Thin, elegant curved smile with a flat-bottomed middle -- the
        depth profile was measured per-column off the reference photo, so
        it plateaus across the middle ~30% and eases up through the outer
        ~35% on each side rather than being a plain circular arc."""
        cx = self.width // 2
        width = int(self.mouth_width * 0.69)
        depth = int(self.height * 0.05)
        happy_mouth_drop = int(self.height * 0.15)
        centre = (cx, self.mouth_y - self.happy_y_shift + happy_mouth_drop)
        thickness = max(3, int(self.height * 0.014))

        corner_frac, steps = 0.35, 20
        points = []
        for i in range(steps + 1):
            xn = -0.5 + i / steps                  # -0.5 (left tip) -> +0.5
            corner_t = (0.5 - abs(xn)) / corner_frac
            depth_frac = (1.0 if corner_t >= 1
                          else 2 * corner_t - corner_t * corner_t)
            points.append((centre[0] + xn * width,
                           centre[1] + depth_frac * depth))
        self._thick_polyline(points, thickness)

    def _mouth_tilted(self, degrees=-8.5, width_mul=0.22) -> None:
        """Very short, thin bar tilted counterclockwise (right end raised) --
        angle measured directly off the reference photo. Built as a rotated
        filled polygon, not a rotated blit, so the edges stay crisp."""
        cx = self.width // 2
        w = int(self.mouth_width * width_mul)
        thickness = max(3, int(self.height * 0.01))
        angle = math.radians(degrees)
        hw, ht = w / 2, thickness / 2
        pts = []
        for dx, dy in ((-hw, -ht), (hw, -ht), (hw, ht), (-hw, ht)):
            pts.append((cx + dx * math.cos(angle) - dy * math.sin(angle),
                        self.mouth_y + dx * math.sin(angle)
                        + dy * math.cos(angle)))
        self._poly(pts)

    def _mouth_o(self, size_mul=1.0) -> None:
        """A small round open mouth -- surprised."""
        d = int(self.mouth_tall_h * size_mul)
        cx, cy = self._xf(self.width // 2, self.mouth_y)
        pygame.draw.ellipse(self.screen, FG_COLOR,
                            pygame.Rect(int(cx - d // 2), int(cy - d // 2),
                                        d, d))

    def _mouth_smirk(self, width_mul=0.55) -> None:
        """A one-sided half smile -- rises toward the right only."""
        cx = self.width // 2
        w = int(self.mouth_width * width_mul)
        depth = int(self.height * 0.035)
        thickness = max(3, int(self.height * 0.012))
        pts = []
        for i in range(13):
            u = i / 12                              # 0 .. 1, left -> right
            pts.append((cx - w / 2 + u * w,
                        self.mouth_y + depth * (1 - u) * (1 - u) * 1.6
                        - depth * 0.5))
        self._thick_polyline(pts, thickness)

    def _mouth_zigzag(self, width_mul=0.5) -> None:
        """A sharp sawtooth -- confused / dizzy / scared."""
        cx = self.width // 2
        w = int(self.mouth_width * width_mul)
        amp = max(4, int(self.height * 0.018))
        thickness = max(3, int(self.height * 0.01))
        pts = []
        for i in range(7):
            pts.append((cx - w // 2 + w * i / 6,
                        self.mouth_y + (amp if i % 2 else -amp)))
        self._thick_polyline(pts, thickness)

    # --- the emotion table -------------------------------------------

    def _draw_emotion(self, name: str) -> None:
        """Draw one emotion's eyes and mouth. The first six reproduce
        saras_face.py exactly; the rest use the same primitives."""

        # --- the five reference moods, plus speaking ------------------
        if name == "neutral":
            self._plain_eyes()
            self._mouth_bar()
        elif name == "thinking":
            self._draw_thinking_eyes()
            self._mouth_tilted()
        elif name == "happy":
            self._draw_happy_eyes()
            self._mouth_smile()
        elif name == "sad":
            self._draw_drooped_eyes("outer")
            self._mouth_frown()
        elif name == "listening":
            self._bar_eyes(1.35, 0.22, pulse=0.4)
            self._mouth_bar()
        elif name == "speaking":
            self._plain_eyes()
            self._mouth_speaking()

        # --- the rest, same visual language ---------------------------
        elif name == "angry":
            self._draw_drooped_eyes("inner")
            self._mouth_bar(0.55)

        elif name == "surprised":
            # bigger and TALLER eyes, small round "o" mouth
            self._plain_eyes(size=int(self.eye_size * 1.15),
                             wide=int(self.eye_size * 0.92),
                             dy=-int(self.eye_size * 0.1))
            self._mouth_o(0.62)

        elif name == "confused":
            # one eye plainly bigger than the other, wavy mouth, floating ?
            blink = self._blink_factor()
            big = int(self.eye_size * 1.18)
            small = int(self.eye_size * 0.72)
            ly, lh = self._blinked(self.eye_y - int(self.eye_size * 0.1),
                                   big, blink)
            ry, rh = self._blinked(self.eye_y + int(self.eye_size * 0.18),
                                   small, blink)
            self._rect_eye(self.left_eye_x, ly, big, lh)
            self._rect_eye(self.right_eye_x, ry, small, rh)
            self._mouth_zigzag(0.42)
            bob = self.height * 0.012 * math.sin(2 * math.pi
                                                 * self._elapsed / 1.8)
            self._glyph("?", self.eye_size * 0.95,
                        self.right_eye_x + self.eye_size * 1.75,
                        self.eye_y + bob)

        elif name == "dizzy":
            self._draw_spiral_eyes()
            self._mouth_zigzag(0.5)

        elif name == "success":
            # short happy squint + big smile + a quick sparkle burst that
            # fades out shortly after the emotion is set
            self._draw_happy_eyes()
            self._mouth_smile()
            age = self._since_change
            if age < 1.3:
                fade = max(0.0, 1.0 - age / 1.3)
                r = self.eye_size * 0.22
                for dx, dy in ((-1.45, -0.5), (1.45, -0.62), (0.0, -1.15)):
                    self._sparkle(self.width / 2 + dx * self.eye_size,
                                  self.eye_y + dy * self.eye_size,
                                  r * (0.5 + 0.5 * fade), 255 * fade)

        elif name == "excited":
            # big wide squares, open smile, sparkles twinkling around
            self._plain_eyes(size=int(self.eye_size * 1.22))
            self._mouth_open(0.42, 0.7)
            for i, (dx, dy) in enumerate(((-2.0, -0.55), (2.0, -0.66),
                                          (-1.7, 0.75), (1.75, 0.66))):
                tw = 0.55 + 0.45 * math.sin(2 * math.pi * self._elapsed / 1.1
                                            + i * 1.7)
                self._sparkle(self.width / 2 + dx * self.eye_size,
                              self.eye_y + dy * self.eye_size,
                              self.eye_size * 0.17 * tw, 255 * tw)

        elif name == "wink":
            # one eye a normal open square, the other closed to a short bar
            blink = self._blink_factor()
            y, h = self._blinked(self.eye_y, self.eye_size, blink)
            self._rect_eye(self.left_eye_x, y, self.eye_size, h)
            bar_h = max(4, int(self.eye_size * 0.18))
            self._rect_eye(self.right_eye_x,
                           self.eye_y + (self.eye_size - bar_h) // 2,
                           self.eye_size, bar_h)
            self._mouth_smirk(0.5)

        elif name == "love":
            self._draw_heart_eyes()
            self._mouth_smile()

        elif name == "curious":
            # square eyes, one slightly raised, whole head tilted a little
            blink = self._blink_factor()
            ly, lh = self._blinked(self.eye_y - int(self.eye_size * 0.16),
                                   self.eye_size, blink)
            ry, rh = self._blinked(self.eye_y, self.eye_size, blink)
            self._rect_eye(self.left_eye_x, ly, self.eye_size, lh)
            self._rect_eye(self.right_eye_x, ry, self.eye_size, rh)
            self._mouth_bar(0.34)

        elif name == "proud":
            # half-height squares (smug, lids lowered) + a smirk
            self._plain_eyes(size=int(self.eye_size * 0.5),
                             wide=self.eye_size,
                             dy=-int(self.eye_size * 0.05))
            self._mouth_smirk(0.5)

        elif name == "bored":
            # squished to about half height, droopy, flat mouth
            self._plain_eyes(size=int(self.eye_size * 0.46),
                             wide=self.eye_size,
                             dy=int(self.eye_size * 0.3))
            self._mouth_bar(0.34)

        elif name == "sleepy":
            # very thin, nearly closed, tiny mouth, an occasional rising z
            self._bar_eyes(1.05, 0.12, dy=int(self.eye_size * 0.28))
            self._mouth_bar(0.16)
            phase = (self._elapsed / 2.8) % 1.0
            if phase < 0.55:
                k = phase / 0.55
                self._glyph("z", self.eye_size * 0.42,
                            self.right_eye_x + self.eye_size * 1.5
                            + self.eye_size * 0.2 * k,
                            self.eye_y - self.eye_size * (0.35 + 0.6 * k),
                            255 * (1 - k))

        elif name == "sleeping":
            # eyes closed as down-arcs, three drifting z's; the gentle
            # breathing bob is applied in _frame_motion()
            self._draw_closed_arc_eyes()
            self._mouth_bar(0.16)
            for i, mul in enumerate((0.34, 0.48, 0.64)):
                phase = (self._elapsed / 3.2 + i * 0.28) % 1.0
                self._glyph("z", self.eye_size * mul * (0.7 + 0.4 * phase),
                            self.right_eye_x + self.eye_size * 1.4
                            + self.eye_size * 0.5 * phase,
                            self.eye_y - self.eye_size * (0.2 + 1.2 * phase),
                            255 * min(1.0, (1 - phase) * 2.2))

        elif name == "scared":
            # big eyes with small dark pupils, wavy mouth; shake in
            # _frame_motion()
            self._plain_eyes(size=int(self.eye_size * 1.26), pupil=0.26)
            self._mouth_zigzag(0.45)

        else:                                  # unreachable, but be safe
            self._plain_eyes()
            self._mouth_bar()

    def _draw_yawn(self, amount: float) -> None:
        """The yawn, following the order in the reference photos: the eyes
        squeeze down to wide short bars while the mouth pulls in to a small
        rounded pill, hold, then ease back open to full squares. `amount`
        runs 0 -> 1 -> 0, so this interpolates the whole way and there is no
        jump at either end."""
        # eyes: full square -> wide short bar
        w = int(_lerp(self.eye_size, self.eye_size * 1.08, amount))
        h = max(3, int(_lerp(self.eye_size, self.eye_size * 0.17, amount)))
        cx = self.width // 2
        y = self.eye_y + (self.eye_size - h) // 2
        self._rect_eye(cx - self.eye_gap // 2 - w, y, w, h)
        self._rect_eye(cx + self.eye_gap // 2, y, w, h)

        # mouth: wide thin bar -> small rounded pill, taller than it is wide
        mw = max(3, int(_lerp(self.mouth_width, self.mouth_width * 0.095,
                              amount)))
        mh = max(3, int(_lerp(self.mouth_thin_h, self.mouth_tall_h * 0.62,
                              amount)))
        mcx, mcy = self._xf(cx, self.mouth_y)
        rect = pygame.Rect(0, 0, mw, mh)
        rect.center = (int(mcx), int(mcy))
        if amount > 0.25:                 # rounded once it has pulled in
            pygame.draw.ellipse(self.screen, FG_COLOR, rect)
        else:
            self._rect_c(cx, self.mouth_y, mw, mh)

    # --- frame -------------------------------------------------------

    def _frame_motion(self, name: str) -> None:
        """Set this frame's whole-face offset and tilt. _xf() applies them
        to every shape, so nothing needs to know about them individually."""
        self._off_x = 0.0
        self._off_y = 0.0
        self._tilt = 0.0
        t = self._elapsed
        if name == "scared":                     # a fast nervous jitter
            self._off_x = self.width * 0.0022 * math.sin(2 * math.pi * t * 11)
        elif name == "sleeping":                 # slow breathing rise/fall
            self._off_y = self.height * 0.011 * math.sin(2 * math.pi * t / 3.4)
        elif name == "curious":                  # a slight, held head tilt
            self._tilt = 6.5
        elif name == "excited":                  # a light bounce
            self._off_y = -self.height * 0.009 * abs(
                math.sin(2 * math.pi * t / 0.9))

    def draw(self) -> None:
        self.screen.fill(BG_COLOR)

        yawn = self._yawn_amount()
        if yawn > 0.0:
            self._frame_motion("")
            self._draw_yawn(yawn)
        else:
            self._frame_motion(self._name)
            self._draw_emotion(self._name)
            self._off_x = self._off_y = self._tilt = 0.0

        # Cross-fade out of the previous emotion's last frame. The shapes
        # themselves are never interpolated, so every frame is exact.
        if self._fade_from is not None:
            alpha = int(255 * (1.0 - _ease(self._fade_t)))
            if alpha > 0:
                self._fade_from.set_alpha(alpha)
                self.screen.blit(self._fade_from, (0, 0))

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
        pygame.quit()


# Emotions whose eyes are already closed or non-square -- blinking them
# would either do nothing visible or look wrong.
_NO_BLINK = frozenset({
    "happy", "thinking", "listening", "dizzy", "success", "love",
    "proud", "bored", "sleepy", "sleeping",
})


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
