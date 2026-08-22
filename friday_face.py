"""friday_face.py — Friday's face on the 7" display.

    python3 friday_face.py            fullscreen, follows state/emotion.txt
    python3 friday_face.py --demo     cycle all 20 emotions + play the yawn
    python3 friday_face.py --window   windowed, for testing over SSH/VNC

Nothing drives this directly. It watches state/emotion.txt (and
state/yawn.txt for the yawn trigger) and switches to whatever it finds
there; use face_control.py to write those files.

Block geometry on a pure black screen, finished to read as raised,
polished orange plastic: every shape gets a strong vertical gradient
(lighter at the top, deeper at the bottom), a glossy highlight near its
upper left, and a drop shadow. See _finish().

Shapes are recorded as white silhouettes onto a scratch surface between
_begin()/_end() -- the `with self._shape():` blocks -- and that silhouette
is what the finish is applied to, so adding a new shape needs no knowledge
of the lighting at all.

The reference moods (neutral, thinking, happy, sad, listening) and the
speaking mouth keep hardware/saras_face.py's measured geometry, which was
derived from photos of the real panel -- copied verbatim, not re-invented.

Geometry is proportional to the actual screen size, so nothing is ever
cropped or letterboxed regardless of what the panel reports.
"""

from __future__ import annotations

import argparse
import contextlib
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

# --- dimensional finish ----------------------------------------------
# Bold and glossy: a strong top-to-bottom gradient, a clear upper-left
# highlight, and a defined drop shadow, so each shape reads as raised
# polished plastic rather than a flat fill.
GRADIENT_TOP = (255, 168, 74)     # lit top edge
GRADIENT_BOTTOM = (168, 58, 0)    # deep bottom edge
HIGHLIGHT_ALPHA = 105             # upper-left gloss strength
HIGHLIGHT_ALPHA_GLOSSY = 165      # heart eyes and the dizzy spiral shine more
# The background is pure black, so a black shadow would be invisible --
# this is a deep warm brown, which still reads as the shape's own shadow.
SHADOW_COLOR = (74, 27, 0)
SHADOW_ALPHA = 235
SHADOW_OFFSET = (0, 9)
BLUR_AMOUNT = 4                   # smoothscale down/up divisor -- softness
SMALL_SHAPE_PX = 56               # below this, skip the material entirely

# Neutral's eyes are drawn wider than tall so the resting face reads bold
# and chunky. Only neutral uses this -- every other mood keeps the square.
NEUTRAL_EYE_W_MUL = 1.2

# =====================================================================
# TIMING — tune these
# =====================================================================
FPS = 30
TRANSITION_S = 0.28               # cross-fade between emotions
BLINK_MIN_INTERVAL = 2.0
BLINK_MAX_INTERVAL = 6.0
BLINK_DURATION = 0.16
WINK_PERIOD_S = 3.0               # one full wink of the right eye
POLL_S = 0.25                     # how often the state files are checked
DEMO_HOLD_S = 2.6                 # --demo dwell per emotion

# --- yawn overlay ----------------------------------------------------
# Not one of the 20 emotions -- a transient reflex that plays on top of
# whatever emotion is showing, then returns to it. See play_yawn().
# Paced like someone surfacing from sleep. Four seconds end to end -- this
# is what plays once on every boot before the face settles to neutral.
YAWN_IN_S = 1.05
YAWN_HOLD_S = 1.60
YAWN_OUT_S = 1.35
YAWN_TOTAL_S = YAWN_IN_S + YAWN_HOLD_S + YAWN_OUT_S

# The real 7" HDMI TFT panel (EDID "Mediatrix Peripherals Inc MPI7002")
# reports 1920x1080 as its native mode -- confirmed here via wlr-randr.
# Geometry below is proportional, so any of these renders correctly.
SCREEN_CANDIDATES = [(1920, 1080), (1024, 600), (800, 480)]
DEV_PREVIEW_SIZE = (960, 540)     # exact half-scale of the panel, 16:9

MASK_WHITE = (255, 255, 255, 255)


def _detect_screen_size() -> tuple[tuple[int, int], bool]:
    pygame.display.init()
    info = pygame.display.Info()
    current = (info.current_w, info.current_h)
    if current in SCREEN_CANDIDATES:
        return current, True          # the real panel -> fullscreen native
    return DEV_PREVIEW_SIZE, False    # a dev monitor -> small test window


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _q(v: float, step: int = 4) -> int:
    """Snap an animated dimension to a step. Sizes that change every single
    frame miss the material cache every frame, which is what makes a pulse
    expensive; quantising costs nothing visible and restores the cache."""
    return max(step, int(round(v / step)) * step)


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return 4 * t * t * t if t < 0.5 else 1 - (-2 * t + 2) ** 3 / 2


def _blur(surf: pygame.Surface, amount: int = BLUR_AMOUNT) -> pygame.Surface:
    """Cheap blur: shrink then grow back."""
    w, h = surf.get_size()
    small = pygame.transform.smoothscale(
        surf, (max(1, w // amount), max(1, h // amount)))
    return pygame.transform.smoothscale(small, (w, h))


def _gradient_seed() -> pygame.Surface:
    seed = pygame.Surface((1, 2), pygame.SRCALPHA)
    seed.set_at((0, 0), (*GRADIENT_TOP, 255))
    seed.set_at((0, 1), (*GRADIENT_BOTTOM, 255))
    return seed


_GRADIENT_SEED = _gradient_seed()


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

        # Per-frame whole-face transform, applied by _xf() to every shape.
        self._off_x = 0.0
        self._off_y = 0.0
        self._tilt = 0.0

        self._fonts: dict[int, pygame.font.Font] = {}

        # Shapes are drawn white onto this, then lit by _finish().
        self._scratch = pygame.Surface((self.width, self.height),
                                       pygame.SRCALPHA)
        self._bbox = None
        self._mat_cache: dict = {}
        self._shadow_cache: dict = {}

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

        # Outer edge of the left eye to outer edge of the right eye. The
        # resting mouth matches this span -- it used to run much wider.
        self.eye_span = self.eye_gap + 2 * self.eye_size

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

    def _prime_state_stamp(self) -> None:
        """Record the state file's current stamp WITHOUT applying it. On
        boot the file may still hold whatever was showing before shutdown;
        priming means the face stays on neutral after the yawn and only
        moves when something actually writes a new emotion."""
        try:
            st = os.stat(EMOTION_FILE)
            self._file_stamp = (st.st_mtime_ns, st.st_size)
        except OSError:
            self._file_stamp = None

    def _poll_state_file(self, now: float) -> None:
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
        self._yawn_start = self._elapsed

    def _yawn_amount(self) -> float:
        """0 -> 1 -> 0 across the whole yawn. Pure: update() is what ends
        the yawn and drops the face back to neutral."""
        if self._yawn_start is None:
            return 0.0
        dt = self._elapsed - self._yawn_start
        if dt < YAWN_IN_S:
            return _ease(dt / YAWN_IN_S)
        if dt < YAWN_IN_S + YAWN_HOLD_S:
            return 1.0
        if dt < YAWN_TOTAL_S:
            return 1.0 - _ease((dt - YAWN_IN_S - YAWN_HOLD_S) / YAWN_OUT_S)
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

        # The yawn always lands back on neutral, whatever it interrupted.
        if (self._yawn_start is not None
                and self._elapsed - self._yawn_start >= YAWN_TOTAL_S):
            self._yawn_start = None
            self._name = DEFAULT_EMOTION
            self._since_change = 0.0

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
        if self._blink_t is None or self._name in _NO_BLINK:
            return 1.0
        t = self._blink_t / BLINK_DURATION
        return abs(2 * t - 1)

    # --- transform ---------------------------------------------------

    def _xf(self, x, y) -> tuple[float, float]:
        """Apply this frame's head tilt and offset to one point."""
        if self._tilt:
            cx = self.width / 2
            cy = (self.eye_y + self.mouth_y) / 2
            dx, dy = x - cx, y - cy
            a = math.radians(self._tilt)
            ca, sa = math.cos(a), math.sin(a)
            x, y = cx + dx * ca - dy * sa, cy + dx * sa + dy * ca
        return x + self._off_x, y + self._off_y

    # --- shape recording and lighting --------------------------------

    def _mark(self, rect: pygame.Rect) -> None:
        self._bbox = rect if self._bbox is None else self._bbox.union(rect)

    @contextlib.contextmanager
    def _shape(self, glossy: bool = False):
        """Record one visual shape as a white silhouette, then light it."""
        self._bbox = None
        try:
            yield
        finally:
            self._finish(glossy)

    def _material(self, w: int, h: int, glossy: bool) -> pygame.Surface:
        """The lit orange surface a shape is cut out of: vertical gradient
        with the gloss highlight already baked in. Cached per size, so the
        expensive blur only runs when a shape changes size -- not every
        frame of every shape."""
        key = (w, h, glossy)
        mat = self._mat_cache.get(key)
        if mat is None:
            if len(self._mat_cache) > 160:
                self._mat_cache.clear()
            mat = pygame.transform.smoothscale(_GRADIENT_SEED, (w, h))
            mat = mat.convert_alpha()
            gloss = pygame.Surface((w, h), pygame.SRCALPHA)
            ell = pygame.Rect(0, 0, max(1, int(w * 0.78)),
                              max(1, int(h * 0.5)))
            ell.center = (int(w * 0.36), int(h * 0.27))
            alpha = HIGHLIGHT_ALPHA_GLOSSY if glossy else HIGHLIGHT_ALPHA
            pygame.draw.ellipse(gloss, (255, 255, 255, alpha), ell)
            mat.blit(_blur(gloss), (0, 0))
            self._mat_cache[key] = mat
        return mat

    def _shadow_layer(self, size) -> pygame.Surface:
        sh = self._shadow_cache.get(size)
        if sh is None:
            if len(self._shadow_cache) > 160:
                self._shadow_cache.clear()
            sh = pygame.Surface(size, pygame.SRCALPHA)
            sh.fill((*SHADOW_COLOR, SHADOW_ALPHA))
            self._shadow_cache[size] = sh
        return sh

    def _finish(self, glossy: bool) -> None:
        """Light one recorded silhouette: a hard offset drop shadow, then
        the shape cut out of the lit material."""
        if self._bbox is None:
            return
        screen_rect = self.screen.get_rect()
        tight = self._bbox.clip(screen_rect)
        self._bbox = None
        if tight.width < 1 or tight.height < 1:
            return

        mask = self._scratch.subsurface(tight).copy()
        size = tight.size

        # Small shapes -- confetti, sparkles, the z's -- get a flat fill
        # instead of the full material. At this size the gradient and gloss
        # are invisible anyway, and they tumble/twinkle constantly, so they
        # would miss the cache on every frame.
        if size[0] < SMALL_SHAPE_PX and size[1] < SMALL_SHAPE_PX:
            flat = pygame.Surface(size, pygame.SRCALPHA)
            flat.fill(FG_COLOR)
            flat.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
            self.screen.blit(flat, tight.topleft)
            self._scratch.fill((0, 0, 0, 0), tight)
            return

        # 1. drop shadow -- the silhouette recoloured and offset, laid down
        #    first so the lit shape covers most of it. Hard-edged on
        #    purpose: a blurred shadow all but vanishes against pure black.
        shadow = self._shadow_layer(size).copy()
        shadow.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        self.screen.blit(shadow, (tight.x + SHADOW_OFFSET[0],
                                  tight.y + SHADOW_OFFSET[1]))

        # 2. the shape, cut out of the lit material. BLEND_RGBA_MIN takes
        #    the mask's alpha (the silhouette) and the material's colour.
        body = self._material(size[0], size[1], glossy).copy()
        body.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        self.screen.blit(body, tight.topleft)

        self._scratch.fill((0, 0, 0, 0),
                           tight.union(tight.move(*SHADOW_OFFSET)))

    # --- primitives (draw white onto the scratch silhouette) ---------

    def _rect_eye(self, x, y, w, h) -> None:
        if w < 1 or h < 1:
            return
        if self._tilt:
            self._poly([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
            return
        ox, oy = self._xf(x, y)
        r = pygame.Rect(int(ox), int(oy), int(w), int(h))
        pygame.draw.rect(self._scratch, MASK_WHITE, r)
        self._mark(r)

    def _rect_c(self, cx, cy, w, h) -> None:
        if w < 1 or h < 1:
            return
        self._rect_eye(cx - w / 2, cy - h / 2, w, h)

    def _poly(self, points) -> None:
        pts = [self._xf(px, py) for px, py in points]
        ipts = [(int(px), int(py)) for px, py in pts]
        pygame.draw.polygon(self._scratch, MASK_WHITE, ipts)
        xs = [p[0] for p in ipts]
        ys = [p[1] for p in ipts]
        self._mark(pygame.Rect(min(xs), min(ys),
                               max(1, max(xs) - min(xs)),
                               max(1, max(ys) - min(ys))))

    def _ellipse(self, cx, cy, w, h) -> None:
        if w < 1 or h < 1:
            return
        ox, oy = self._xf(cx, cy)
        r = pygame.Rect(int(ox - w / 2), int(oy - h / 2), int(w), int(h))
        pygame.draw.ellipse(self._scratch, MASK_WHITE, r)
        self._mark(r)

    def _thick_polyline(self, points, width) -> None:
        pts = [self._xf(px, py) for px, py in points]
        r = max(int(width) // 2, 1)
        for px, py in pts:
            pygame.draw.circle(self._scratch, MASK_WHITE, (int(px), int(py)), r)
        pygame.draw.lines(self._scratch, MASK_WHITE, False,
                          [(int(px), int(py)) for px, py in pts],
                          max(1, int(width)))
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        self._mark(pygame.Rect(int(min(xs)) - r, int(min(ys)) - r,
                               int(max(xs) - min(xs)) + 2 * r,
                               int(max(ys) - min(ys)) + 2 * r))

    def _hole(self, x, y, w, h) -> None:
        """Punch a hole straight through the finished pixels on screen --
        used for scared's pupils, which are a marking on the eye rather
        than a raised shape of their own."""
        ox, oy = self._xf(x, y)
        pygame.draw.rect(self.screen, BG_COLOR,
                         pygame.Rect(int(ox), int(oy), int(w), int(h)))

    # --- overlay glyphs and sparkles ---------------------------------

    def _font(self, size: int) -> pygame.font.Font:
        f = self._fonts.get(size)
        if f is None:
            if not pygame.font.get_init():
                pygame.font.init()
            f = self._fonts[size] = pygame.font.SysFont(None, size, bold=True)
        return f

    def _glyph(self, text, size, x, y, alpha=255) -> None:
        """A floating character -- the '?' and the 'z's. Rendered white into
        the silhouette so it picks up the same finish as everything else."""
        art = self._font(int(size)).render(text, True, (255, 255, 255))
        if alpha < 255:
            art = art.copy()
            art.set_alpha(int(max(0, min(255, alpha))))
        px, py = self._xf(x, y)
        r = art.get_rect(center=(int(px), int(py)))
        self._scratch.blit(art, r)
        self._mark(r)

    def _sparkle(self, x, y, r, alpha=255) -> None:
        """A sharp four-pointed star. Self-contained: it opens its own
        _shape() block, because it is a shape in its own right and would
        otherwise be discarded when the next block resets the bounds."""
        if r < 1 or alpha <= 0:
            return
        pts = []
        for i in range(8):
            a = math.pi * i / 4
            rad = r if i % 2 == 0 else r * 0.26
            pts.append((x + rad * math.cos(a), y + rad * math.sin(a)))
        if alpha >= 250:
            with self._shape():
                self._poly(pts)
            return
        ipts = [self._xf(px, py) for px, py in pts]
        xs = [p[0] for p in ipts]
        ys = [p[1] for p in ipts]
        rect = pygame.Rect(int(min(xs)), int(min(ys)),
                           max(1, int(max(xs) - min(xs))),
                           max(1, int(max(ys) - min(ys))))
        layer = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.polygon(layer, MASK_WHITE,
                            [(int(px - rect.x), int(py - rect.y))
                             for px, py in ipts])
        layer.set_alpha(int(max(0, min(255, alpha))))
        with self._shape():
            self._scratch.blit(layer, rect.topleft)
            self._mark(rect)

    def _confetti(self, count=12) -> None:
        """Party-popper fallout: little slips of paper tumbling down past
        the face. Deterministic per index, so it never needs state."""
        t = self._elapsed
        top = self.eye_y - self.eye_size * 1.5
        fall = self.height * 0.85
        for i in range(count):
            # cheap deterministic scatter -- no RNG, no stored particles
            a = (i * 0.6180339887) % 1.0
            b = (i * 0.7548776662) % 1.0
            phase = (t * (0.34 + 0.16 * b) + a) % 1.0
            x = (self.width / 2 + (a * 2 - 1) * self.eye_span * 0.85
                 + math.sin(t * (1.1 + b) + i) * self.eye_size * 0.22)
            y = top + phase * fall
            if y > self.height:
                continue
            s = self.eye_size * (0.075 + 0.045 * b)
            spin = t * (2.2 + 3.0 * b) + i
            ca, sa = math.cos(spin), math.sin(spin)
            # a thin slip: wide in one axis, narrow in the other
            hw, hh = s * 0.5, s * 0.22
            pts = [(x + dx * ca - dy * sa, y + dx * sa + dy * ca)
                   for dx, dy in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh))]
            with self._shape():
                self._poly(pts)

    def _blinked(self, y, size, factor):
        h = max(2, int(size * factor))
        return y + (size - h) // 2, h

    # --- eyes --------------------------------------------------------

    def _plain_eyes(self, size=None, dy=0, wide=None, pupil=0.0) -> None:
        size = self.eye_size if size is None else size
        w = size if wide is None else wide
        blink = self._blink_factor()
        y, h = self._blinked(self.eye_y + dy, size, blink)
        cx = self.width // 2
        lx = cx - self.eye_gap // 2 - w
        rx = cx + self.eye_gap // 2
        for ex in (lx, rx):
            with self._shape():
                self._rect_eye(ex, y, w, h)
        if pupil > 0 and h > 4:
            pw = max(2, int(w * pupil))
            ph = max(2, int(min(h * 0.9, w * pupil)))
            for ex in (lx, rx):
                self._hole(ex + (w - pw) // 2, y + (h - ph) // 2, pw, ph)

    def _bar_eyes(self, w_mul, h_mul, dy=0, pulse=0.0) -> None:
        w = int(self.eye_size * w_mul)
        h = max(4, int(self.eye_size * h_mul))
        y = self.eye_y + (self.eye_size - h) // 2 + dy
        extra = 0
        if pulse:
            extra = _q((math.sin(self._elapsed * 5) * 0.5 + 0.5) * h * pulse, 4)
        for ex in (self.left_eye_x, self.right_eye_x):
            with self._shape():
                self._rect_c(ex + self.eye_size // 2, y + h // 2, w, h + extra)

    def _happy_eye_points(self, x, y, w, h, inner_side):
        """Uplifted anime-squint eye: flat top, a single straight diagonal
        bottom edge sloping up from the outer corner to the inner corner."""
        if inner_side == "right":
            return [(x + w * 0.02, y + h * 0.02),
                    (x + w * 0.98, y + h * 0.02),
                    (x + w * 0.98, y + h * 0.40),
                    (x + w * 0.02, y + h * 0.97)]
        return [(x + w * 0.98, y + h * 0.02),
                (x + w * 0.02, y + h * 0.02),
                (x + w * 0.02, y + h * 0.40),
                (x + w * 0.98, y + h * 0.97)]

    def _draw_happy_eyes(self) -> None:
        w = self.happy_eye_size
        h = int(self.happy_eye_size * 0.81)
        y = self.eye_y - self.happy_y_shift + (self.happy_eye_size - h) // 2
        cx = self.width // 2
        lx = cx - self.happy_eye_gap // 2 - w
        rx = cx + self.happy_eye_gap // 2
        with self._shape():
            self._poly(self._happy_eye_points(lx, y, w, h, inner_side="right"))
        with self._shape():
            self._poly(self._happy_eye_points(rx, y, w, h, inner_side="left"))

    def _draw_thinking_eyes(self) -> None:
        """Left eye: right-trapezoid, inner-top corner sheared down ~15%.
        Right eye: plain open square."""
        left_h = int(self.eye_size * 0.82)
        ly = self.eye_y + (self.eye_size - left_h) // 2
        lx = self.left_eye_x
        drop = int(left_h * 0.15)
        with self._shape():
            self._poly([(lx, ly),
                        (lx + self.eye_size, ly + drop),
                        (lx + self.eye_size, ly + left_h),
                        (lx, ly + left_h)])
        with self._shape():
            self._rect_eye(self.right_eye_x, self.eye_y,
                           self.eye_size, self.eye_size)

    def _draw_drooped_eyes(self, corner) -> None:
        """Square eyes with one top corner drooped down."""
        blink = self._blink_factor()
        y, h = self._blinked(self.eye_y, self.eye_size, blink)
        droop = int(h * 0.35)
        for lx, is_left in ((self.left_eye_x, True),
                            (self.right_eye_x, False)):
            right = lx + self.eye_size
            drop_left = (corner == "outer") == is_left
            tl = (lx, y + droop) if drop_left else (lx, y)
            tr = (right, y) if drop_left else (right, y + droop)
            with self._shape():
                self._poly([tl, tr, (right, y + h), (lx, y + h)])

    @staticmethod
    def _heartbeat(t: float) -> float:
        """A lub-dub: a strong beat, a softer one just after, then rest."""
        p = t % 1.15
        return (math.exp(-((p / 0.085) ** 2))
                + 0.62 * math.exp(-(((p - 0.26) / 0.085) ** 2)))

    def _draw_heart_eyes(self) -> None:
        """The classic symmetric heart silhouette -- two rounded lobes on
        top meeting a point at the bottom, from the standard parametric
        heart curve. Scales with the heartbeat."""
        beat = 1.0 + 0.17 * self._heartbeat(self._elapsed)
        w = _q(self.eye_size * 1.12 * beat, 6)
        h = _q(self.eye_size * 1.02 * beat, 6)
        cx = self.width // 2
        for ecx in (cx - self.eye_gap // 2 - self.eye_size // 2,
                    cx + self.eye_gap // 2 + self.eye_size // 2):
            pts = []
            for i in range(97):
                a = 2 * math.pi * i / 96
                hx = 16 * math.sin(a) ** 3
                hy = (13 * math.cos(a) - 5 * math.cos(2 * a)
                      - 2 * math.cos(3 * a) - math.cos(4 * a))
                pts.append((ecx + hx * w / 34,
                            self.eye_y + h / 2 - hy * h / 30))
            with self._shape(glossy=True):
                self._poly(pts)

    def _draw_chevron_eyes(self, dy=0) -> None:
        """Happy closed eyes as angular ^ ^ chevrons -- used by success."""
        w = int(self.eye_size * 1.0)
        rise = int(self.eye_size * 0.34)
        thick = max(5, int(self.eye_size * 0.16))
        y = self.eye_y + int(self.eye_size * 0.46) + dy
        for ex in (self.left_eye_x, self.right_eye_x):
            x0 = ex + (self.eye_size - w) // 2
            with self._shape():
                self._thick_polyline([(x0, y), (x0 + w / 2, y - rise),
                                      (x0 + w, y)], thick)

    def _draw_spiral_eyes(self) -> None:
        """Dizzy: a swirl that spins."""
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
            with self._shape(glossy=True):
                self._thick_polyline(pts, thick)

    # --- mouths ------------------------------------------------------

    def _mouth_bar(self, width_mul=1.0, dy=0, width=None) -> None:
        w = self.mouth_width * width_mul if width is None else width
        with self._shape():
            self._rect_c(self.width // 2, self.mouth_y + dy, w,
                         self.mouth_thin_h)

    def _mouth_open(self, width_mul=1.0, height_mul=1.0, width=None) -> None:
        w = self.mouth_width * width_mul if width is None else width
        with self._shape():
            self._rect_c(self.width // 2, self.mouth_y, w,
                         int(self.mouth_tall_h * height_mul))

    def _mouth_speaking(self) -> None:
        """The mouth-morph. Baseline width matches the eye span."""
        t = self._elapsed
        wobble = ((math.sin(t * 14) * 0.5 + 0.5) * 0.6
                  + (math.sin(t * 27.3) * 0.5 + 0.5) * 0.4)
        h = int(self.mouth_thin_h
                + (self.mouth_tall_h - self.mouth_thin_h) * wobble)
        with self._shape():
            self._rect_c(self.width // 2, self.mouth_y,
                         self.eye_span, max(self.mouth_thin_h, h))

    def _mouth_frown(self) -> None:
        cx = self.width // 2
        half_w = self.mouth_width // 2
        drop = int(self.height * 0.05)
        with self._shape():
            self._thick_polyline(
                [(cx - half_w, self.mouth_y + drop), (cx, self.mouth_y),
                 (cx + half_w, self.mouth_y + drop)], self.mouth_thin_h)

    def _mouth_smile(self) -> None:
        """Thin curved smile with a flat-bottomed middle -- the depth
        profile was measured per-column off the reference photo."""
        cx = self.width // 2
        width = int(self.mouth_width * 0.69)
        depth = int(self.height * 0.05)
        centre = (cx, self.mouth_y - self.happy_y_shift
                  + int(self.height * 0.15))
        thickness = max(3, int(self.height * 0.014))
        corner_frac, steps = 0.35, 20
        points = []
        for i in range(steps + 1):
            xn = -0.5 + i / steps
            corner_t = (0.5 - abs(xn)) / corner_frac
            depth_frac = (1.0 if corner_t >= 1
                          else 2 * corner_t - corner_t * corner_t)
            points.append((centre[0] + xn * width,
                           centre[1] + depth_frac * depth))
        with self._shape():
            self._thick_polyline(points, thickness)

    def _mouth_curve(self, width, depth, thickness=None, dy=0) -> None:
        """A plain parabolic arc. +depth smiles up, -depth frowns down."""
        cx = self.width // 2
        thickness = thickness or max(4, int(self.height * 0.016))
        pts = []
        for i in range(25):
            u = i / 24 * 2 - 1
            pts.append((cx + u * width / 2,
                        self.mouth_y + dy + depth * (u * u - 1) * -1))
        with self._shape():
            self._thick_polyline(pts, thickness)

    def _mouth_tilted(self, degrees=-8.5, width_mul=0.22) -> None:
        """Very short, thin bar tilted counterclockwise (right end raised)."""
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
        with self._shape():
            self._poly(pts)

    def _mouth_smirk(self, width_mul=0.55) -> None:
        """A one-sided half smile -- rises toward the right only."""
        cx = self.width // 2
        w = int(self.mouth_width * width_mul)
        depth = int(self.height * 0.055)
        thickness = max(6, int(self.height * 0.019))
        pts = []
        for i in range(17):
            u = i / 16
            # level across the left, lifting through the right -- never
            # dipping below the mouth line, which would read as a droop
            pts.append((cx - w / 2 + u * w,
                        self.mouth_y + depth * 0.12 - depth * 1.15 * u ** 1.6))
        with self._shape():
            self._thick_polyline(pts, thickness)

    def _mouth_zigzag(self, width_mul=0.5) -> None:
        cx = self.width // 2
        w = int(self.mouth_width * width_mul)
        amp = max(4, int(self.height * 0.018))
        thickness = max(3, int(self.height * 0.01))
        pts = []
        for i in range(7):
            pts.append((cx - w // 2 + w * i / 6,
                        self.mouth_y + (amp if i % 2 else -amp)))
        with self._shape():
            self._thick_polyline(pts, thickness)

    def _mouth_teeth(self, width=None, segments=3) -> None:
        """The angry mouth from the reference photo: ONE solid bar split
        into three segments by two thin vertical gaps. No upper/lower
        split, no points -- earlier sawtooth and grid versions both read
        as something other than a face."""
        cx = self.width // 2
        w = int(width if width is not None else self.eye_span * 0.55)
        h = int(self.mouth_thin_h * 2.0)
        with self._shape():
            self._rect_c(cx, self.mouth_y, w, h)
        vgap = max(3, int(w * 0.028))
        for j in range(1, segments):
            self._hole(cx - w / 2 + w * j / segments - vgap / 2,
                       self.mouth_y - h / 2, vgap, h)

    def _mouth_o(self, size_mul=1.0) -> None:
        d = int(self.mouth_tall_h * size_mul)
        with self._shape():
            self._ellipse(self.width // 2, self.mouth_y, d, d)

    def _mouth_grin(self, width=None, height_mul=1.0) -> None:
        """A big open smile: the bottom half of an ellipse, flat on top."""
        w = int(width if width is not None else self.eye_span * 0.62)
        h = int(self.mouth_tall_h * height_mul)
        cx = self.width // 2
        with self._shape():
            ox, oy = self._xf(cx, self.mouth_y - h / 2)
            tall = pygame.Surface((w, h * 2), pygame.SRCALPHA)
            pygame.draw.ellipse(tall, MASK_WHITE, pygame.Rect(0, 0, w, h * 2))
            clip = tall.subsurface(pygame.Rect(0, 0, w, h)).copy()
            r = pygame.Rect(int(ox - w / 2), int(oy), w, h)
            self._scratch.blit(clip, r.topleft)
            self._mark(r)

    # --- the emotion table -------------------------------------------

    def _draw_emotion(self, name: str) -> None:
        if name == "neutral":
            wide = int(self.eye_size * NEUTRAL_EYE_W_MUL)
            self._plain_eyes(wide=wide)
            self._mouth_bar(width=self.eye_gap + 2 * wide)

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
            self._mouth_bar(width=self.eye_span)

        elif name == "speaking":
            self._plain_eyes()
            self._mouth_speaking()

        elif name == "angry":
            self._draw_drooped_eyes("inner")
            self._mouth_teeth()

        elif name == "surprised":
            self._plain_eyes(size=int(self.eye_size * 1.15),
                             wide=int(self.eye_size * 0.92),
                             dy=-int(self.eye_size * 0.1))
            self._mouth_o(0.62)

        elif name == "confused":
            blink = self._blink_factor()
            big = int(self.eye_size * 1.18)
            small = int(self.eye_size * 0.72)
            ly, lh = self._blinked(self.eye_y - int(self.eye_size * 0.1),
                                   big, blink)
            ry, rh = self._blinked(self.eye_y + int(self.eye_size * 0.18),
                                   small, blink)
            with self._shape():
                self._rect_eye(self.left_eye_x, ly, big, lh)
            with self._shape():
                self._rect_eye(self.right_eye_x, ry, small, rh)
            self._mouth_zigzag(0.42)
            bob = self.height * 0.012 * math.sin(2 * math.pi
                                                 * self._elapsed / 1.8)
            with self._shape():
                self._glyph("?", self.eye_size * 0.95,
                            self.right_eye_x + self.eye_size * 1.75,
                            self.eye_y + bob)

        elif name == "dizzy":
            self._draw_spiral_eyes()
            self._mouth_zigzag(0.5)

        elif name == "success":
            # Eyes squeezed shut in a grin, a bold smile, and party-popper
            # confetti tumbling down -- the confetti is what separates this
            # from happy at a glance.
            self._draw_chevron_eyes()
            self._mouth_curve(self.eye_span * 0.66, int(self.height * 0.075),
                              thickness=max(8, int(self.height * 0.024)))
            self._confetti()

        elif name == "excited":
            # Wide eyes and an open shout, both throbbing in time with the
            # bounce so the whole face reads as buzzing with energy.
            pulse = 0.5 + 0.5 * math.sin(2 * math.pi * self._elapsed / 0.38)
            # Kept deliberately modest: at full size the eyes reach almost
            # to the mouth, and the two shapes collide.
            self._plain_eyes(size=_q(self.eye_size * (1.06 + 0.10 * pulse), 6))
            with self._shape():
                self._ellipse(self.width // 2,
                              self.mouth_y + self.mouth_tall_h * 0.12,
                              _q(self.eye_span * (0.36 + 0.06 * pulse), 6),
                              _q(self.mouth_tall_h * (0.8 + 0.22 * pulse), 6))
            for i, (dx, dy) in enumerate(((-2.7, -0.75), (2.7, -0.85),
                                          (-2.5, 0.95), (2.5, 0.85),
                                          (0.0, -1.45))):
                tw = 0.5 + 0.5 * math.sin(2 * math.pi * self._elapsed / 0.55
                                          + i * 1.5)
                self._sparkle(self.width / 2 + dx * self.eye_size,
                              self.eye_y + dy * self.eye_size,
                              self.eye_size * 0.21 * tw, 255)

        elif name == "wink":
            # The left eye stays open; the right one winks on a 3s loop --
            # briefly open, snap shut, hold the wink, then open again.
            p = (self._elapsed % WINK_PERIOD_S) / WINK_PERIOD_S
            if p < 0.10:
                openness = 1.0
            elif p < 0.20:
                openness = 1.0 - _ease((p - 0.10) / 0.10)
            elif p < 0.86:
                openness = 0.0
            else:
                openness = _ease((p - 0.86) / 0.14)
            with self._shape():
                self._rect_eye(self.left_eye_x, self.eye_y,
                               self.eye_size, self.eye_size)
            h = max(5, _q(self.eye_size * (0.17 + 0.83 * openness), 6))
            with self._shape():
                self._rect_eye(self.right_eye_x,
                               self.eye_y + (self.eye_size - h) // 2,
                               self.eye_size, h)
            self._mouth_curve(self.eye_span * 0.62,
                              int(self.height * 0.052))

        elif name == "love":
            self._draw_heart_eyes()
            self._mouth_curve(self.eye_span * 0.6, int(self.height * 0.05))

        elif name == "curious":
            # Head cocked, one eye opened wider than the other, and a small
            # round "oh?" mouth. Tilt comes from _frame_motion().
            blink = self._blink_factor()
            big = int(self.eye_size * 1.2)
            ly, lh = self._blinked(self.eye_y - int(self.eye_size * 0.2),
                                   big, blink)
            ry, rh = self._blinked(self.eye_y + int(self.eye_size * 0.04),
                                   self.eye_size, blink)
            with self._shape():
                self._rect_eye(self.left_eye_x, ly, big, lh)
            with self._shape():
                self._rect_eye(self.right_eye_x, ry, self.eye_size, rh)
            self._mouth_o(0.6)

        elif name == "proud":
            # Lids half down over wide eyes, chin lifted (tilt lives in
            # _frame_motion), and a BOLD closed-mouth smile. The old thin
            # one-sided smirk kept reading as a droop no matter how it was
            # curved, so this is symmetric and heavy instead.
            h = int(self.eye_size * 0.6)
            y = self.eye_y + int(self.eye_size * 0.06)
            wide = int(self.eye_size * 1.14)
            cx = self.width // 2
            for ex in (cx - self.eye_gap // 2 - wide, cx + self.eye_gap // 2):
                with self._shape():
                    self._rect_eye(ex, y, wide, h)
            self._mouth_curve(self.eye_span * 0.5, int(self.height * 0.045),
                              thickness=max(9, int(self.height * 0.026)))

        elif name == "bored":
            self._plain_eyes(size=int(self.eye_size * 0.46),
                             wide=self.eye_size,
                             dy=int(self.eye_size * 0.3))
            self._mouth_bar(0.34)

        elif name == "sleepy":
            self._bar_eyes(1.05, 0.12, dy=int(self.eye_size * 0.28))
            self._mouth_bar(0.16)
            phase = (self._elapsed / 2.8) % 1.0
            if phase < 0.55:
                k = phase / 0.55
                with self._shape():
                    self._glyph("z", self.eye_size * 0.42,
                                self.right_eye_x + self.eye_size * 1.5
                                + self.eye_size * 0.2 * k,
                                self.eye_y - self.eye_size * (0.35 + 0.6 * k),
                                255 * (1 - k))

        elif name == "sleeping":
            # Closed eyes are SQUARES -- the same rounded-square identity as
            # every other mood, just squeezed very short. Not arcs.
            h = max(5, int(self.eye_size * 0.15))
            y = self.eye_y + (self.eye_size - h) // 2
            for ex in (self.left_eye_x, self.right_eye_x):
                with self._shape():
                    self._rect_eye(ex, y, self.eye_size, h)
            self._mouth_bar(0.16)
            for i, mul in enumerate((0.34, 0.48, 0.64)):
                phase = (self._elapsed / 3.2 + i * 0.28) % 1.0
                with self._shape():
                    self._glyph(
                        "z", self.eye_size * mul * (0.7 + 0.4 * phase),
                        self.right_eye_x + self.eye_size * 1.4
                        + self.eye_size * 0.5 * phase,
                        self.eye_y - self.eye_size * (0.2 + 1.2 * phase),
                        255 * min(1.0, (1 - phase) * 2.2))

        elif name == "scared":
            self._plain_eyes(size=int(self.eye_size * 1.26), pupil=0.26)
            self._mouth_zigzag(0.45)

        else:
            self._plain_eyes()
            self._mouth_bar(width=self.eye_span)

    def _draw_yawn(self, amount: float) -> None:
        """Follows the reference photos: the EYES do not change -- they stay
        the neutral squares the whole way through. Only the mouth moves,
        drawing in from the wide neutral bar to a small open square and back
        out again. (Earlier versions squeezed the eyes down to bars, which
        the photos do not show.)"""
        wide = int(self.eye_size * NEUTRAL_EYE_W_MUL)
        blink = self._blink_factor()
        y, h = self._blinked(self.eye_y, self.eye_size, blink)
        cx = self.width // 2
        for ex in (cx - self.eye_gap // 2 - wide, cx + self.eye_gap // 2):
            with self._shape():
                self._rect_eye(ex, y, wide, h)

        # mouth: wide flat bar -> small open square. Quantised so the
        # material cache still hits while it animates.
        open_px = self.eye_size * 0.62
        mw = _q(_lerp(self.eye_gap + 2 * wide, open_px, amount), 6)
        mh = _q(_lerp(self.mouth_thin_h, open_px, amount), 6)
        with self._shape():
            self._rect_c(cx, self.mouth_y, mw, mh)

    # --- frame -------------------------------------------------------

    def _frame_motion(self, name: str) -> None:
        self._off_x = 0.0
        self._off_y = 0.0
        self._tilt = 0.0
        t = self._elapsed
        if name == "scared":
            self._off_x = self.width * 0.0022 * math.sin(2 * math.pi * t * 11)
        elif name == "sleeping":
            self._off_y = self.height * 0.011 * math.sin(2 * math.pi * t / 3.4)
        elif name == "curious":
            self._tilt = 8.0
        elif name == "proud":
            self._tilt = -3.0
        elif name == "excited":
            self._off_y = -self.height * 0.012 * abs(
                math.sin(2 * math.pi * t / 0.75))
        elif name == "success":
            self._off_y = -self.height * 0.007 * abs(
                math.sin(2 * math.pi * t / 1.1))

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
    "proud", "bored", "sleepy", "sleeping", "wink",
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
    ap.add_argument("--boot-yawn", action="store_true",
                    help="wake up with the yawn, then settle on neutral "
                         "(what the boot service uses)")
    args = ap.parse_args()

    face = FridayFace(fullscreen=not args.window)
    if args.demo:
        face.run_demo()
    elif args.boot_yawn:
        # Wake up: ignore whatever emotion.txt happens to hold, yawn once,
        # and land on neutral. The yawn itself returns to neutral (update()).
        face._prime_state_stamp()
        face.play_yawn()
        face.run()
    else:
        face.set_emotion(get_emotion())
        face.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
