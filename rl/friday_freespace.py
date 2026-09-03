"""Where the floor is, from Friday's cameras, as a free-space profile.

Harsh, 2026-09-03: "I suggest an approach where YOU access both the cameras and
analyse all the available space in the environment and move according to the
space available. Also, make sure that when the sensors are not working the
device (YOU) uses the camera's vision to navigate through the environment like a
self-driving vehicle."

FIRST, THE THING THAT MADE THIS URGENT. `pi_deploy` imports cv2 inside a
try/except and sets it to None when it is missing, and Friday's venv is built
with `include-system-site-packages = false` while OpenCV was only ever installed
system-wide. So cv2 was None in EVERY autonomous run this project has ever done:
both cameras dead, both camera threads no-ops, and the visual hazard gate
fail-open from the first run to the six-minute one on 2026-09-03. The cameras
were never "not helping much" -- they were never there. Fixed by installing
opencv-python-headless into the venv, which is the one kind of install
friday/CLAUDE.md allows.

WHAT THIS COMPUTES. A forward camera on a flat floor encodes distance as image
ROW: the bottom of the frame is at her bumper and rows climb toward the horizon.
So for each column, walking up from the bottom until the floor stops being floor
gives the distance to the first obstacle in that direction -- a depth profile
across the field of view, which is what a lane planner steers on.

  free_profile(gray)      -> (bins,) floor coverage in [0, 1], left to right
  FreeSpaceNavigator      -> the profile as a heading, a clearance and a
                             confidence, for both cameras, with hysteresis

TWO USES, AND THEY HAVE VERY DIFFERENT STANDARDS OF PROOF:

  * ADVISORY STEERING, always on. Only the SHAPE of the profile is used -- which
    way is more open -- so it survives any calibration error. It nudges; it
    cannot stop her and it cannot outvote the LiDAR.
  * FALLBACK NAVIGATION, only when a critical sensor has gone stale. Here the
    profile is all she has, so it is read for distance too, and distance depends
    on camera height and tilt that are ESTIMATED, not measured. That is why the
    fallback is capped hard and slowly: driving on a number you cannot verify is
    exactly when to go slowly.

THE WIDEST GAP, NOT THE CENTRE OF MASS. Two doorways either side of a pillar
average to the pillar. Every opening is measured as a contiguous run of open
bins and the widest one wins, which is the same reason a person picks a door
instead of splitting the difference.
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

LOGGER = logging.getLogger("friday.freespace")

try:
    import cv2
except ImportError:                                       # pragma: no cover
    cv2 = None

# ---- the working image ---------------------------------------------------------
# The camera threads publish 320x240 grayscale. Everything here runs on a further
# 4x reduction: the floor boundary is a low-frequency feature, and this loop has a
# 100 ms budget on a Pi that is already missing it (measured 680 overruns in 1459
# ticks). 80x60 keeps the boundary and costs about a millisecond.
WORK_W, WORK_H = 80, 60

# ---- the floor model -----------------------------------------------------------
# The bottom band is assumed to be floor. On this robot it is: the up camera sits
# above the screen looking forward, so the last rows are the metre in front of her
# bumper, and if something is standing THERE the ToF has already stopped her.
FLOOR_BAND_ROWS = 6
# Deviation from the floor's own intensity, in robust MAD units, before a pixel is
# called not-floor. Generous on purpose -- lighting across a room varies far more
# than a texture does, and a false boundary reads as a phantom wall.
FLOOR_TOL_MAD = 4.0
# ...with an absolute floor under it, because a perfectly uniform carpet has a MAD
# of nearly zero and would then call every pixel an obstacle...
FLOOR_TOL_MIN = 12.0
# ...and a ceiling over it, because a bimodal band -- half floor, half the object
# standing on it -- has an enormous MAD, and a tolerance derived from that calls
# the object floor too. Measured: a synthetic half-blocked frame reported 1.00
# open in every bin before this cap existed.
FLOOR_TOL_MAX = 45.0
# The floor model is seeded from the bottom-CENTRE strip, not the whole band. That
# strip is the ground directly in front of the wheels, which is the most likely
# floor in the picture; if something is standing THERE the ToF has already stopped
# her and the camera's opinion is moot.
FLOOR_SEED_COLS = 0.40
# The floor/wall junction is a horizontal EDGE, and in this house it is often the
# only evidence there is -- a grey floor meeting a grey glass door differs by a few
# levels of intensity and by a very visible gradient. Intensity alone measured
# 4.00 m of clearance against a wall the LiDAR put at 1.11 m.
GRAD_TOL_MAD = 5.0
GRAD_TOL_MIN = 6.0
# A boundary must persist over this many rows. One row is noise or a floor seam;
# an obstacle continues upward from where it meets the ground.
BOUNDARY_RUN = 2

# ---- geometry ------------------------------------------------------------------
# Logitech C270: 60 deg diagonal on 4:3 -> 49.8 h, 38.2 v. Taken from the spec, not
# measured on this mount.
FOV_H_DEG = 49.8
FOV_V_DEG = 38.2
# ESTIMATED, NOT MEASURED. The up camera sits above the 7" panel. Distance is only
# ever used by the fallback, and the fallback is speed-capped precisely because
# these two numbers are estimates -- see FALLBACK_V_MAX in pi_deploy.
CAM_HEIGHT_M = 0.30
CAM_TILT_DEG = 0.0                 # positive = pointing down
# CONSEQUENCE WORTH KNOWING: a camera this high, level, with this vertical FOV
# cannot see the ground closer than h / tan(FOV_V/2) = 0.87 m. The bottom row of
# the frame IS 0.87 m away, and everything nearer than that is invisible to it.
# That blind zone is exactly what the ToF's 8x8 bumper patch covers, so the two
# sensors tile the range rather than overlapping -- but it does mean the camera
# can never be the thing that stops her at close quarters.
# Beyond this the row-to-distance map is too flat to mean anything: one row of an
# 80x60 image near the horizon is several metres.
MAX_RANGE_M = 4.0

# ---- profile -------------------------------------------------------------------
BINS = 9                           # ~5.5 deg each across the horizontal FOV
# A bin counts as open if it has at least this fraction of the best bin's floor.
# Relative, not absolute: a dim room and a bright one have different boundaries but
# the same shape, and shape is what steering needs.
OPEN_FRACTION = 0.60
# Below this the frame is not telling us anything -- a lens against a wall, a dark
# room, a blown exposure. Reported, never guessed around.
MIN_CONFIDENCE = 0.35
# A lens pressed against something is the failure that looks most like success: a
# uniform frame has no boundary anywhere, so the floor model fits it perfectly and
# reports wide open in every direction. Structure is therefore a PRECONDITION for
# believing anything, not a bonus on top of it. Standard deviation rather than MAD
# because a two-tone scene -- floor below, wall above -- has a MAD of exactly zero
# while being perfectly informative.
TEXTURE_REF_STD = 6.0


def _prepare(frame: np.ndarray) -> np.ndarray | None:
    """Any camera frame -> the 80x60 blurred grayscale everything else works on."""
    if frame is None:
        return None
    img = np.asarray(frame)
    if img.ndim == 3:
        img = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if cv2 is not None
               else img.mean(axis=2))
    if img.ndim != 2 or img.size == 0:
        return None
    img = img.astype(np.float32)
    if cv2 is not None:
        img = cv2.resize(img, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        img = cv2.GaussianBlur(img, (3, 3), 0)
    elif img.shape != (WORK_H, WORK_W):
        # No OpenCV: nearest-neighbour so the module still works in tests and in
        # --simulate. Never the production path.
        rows = (np.linspace(0, img.shape[0] - 1, WORK_H)).astype(int)
        cols = (np.linspace(0, img.shape[1] - 1, WORK_W)).astype(int)
        img = img[np.ix_(rows, cols)]
    return img


def floor_boundary(img: np.ndarray, detail: bool = False):
    """Per column, the topmost row that is still floor. Row indices, 0 = top.

    Walks UP from the bottom, which is the only direction that answers the
    question being asked: the first thing standing on the ground in front of her,
    not the most obvious edge somewhere in the frame.
    """
    h, w = img.shape
    band = img[h - FLOOR_BAND_ROWS:, :]

    # Seed from the strip straight ahead of the wheels, then refit on the pixels
    # that agreed with it. One reweighting is enough to stop a bimodal band -- half
    # floor, half obstacle -- from adopting the obstacle as its model.
    half = max(1, int(w * FLOOR_SEED_COLS / 2))
    mid = w // 2
    seed = band[:, max(0, mid - half):min(w, mid + half)]
    mu = float(np.median(seed))
    keep = band[np.abs(band - mu) <= max(FLOOR_TOL_MIN, 2.0 * float(
        np.median(np.abs(seed - mu))))]
    if keep.size >= band.size // 8:
        mu = float(np.median(keep))
        mad = float(np.median(np.abs(keep - mu)))
    else:
        mad = float(np.median(np.abs(seed - mu)))
    tol = float(np.clip(FLOOR_TOL_MAD * mad, FLOOR_TOL_MIN, FLOOR_TOL_MAX))

    # Two independent kinds of evidence, because either alone has a blind spot: a
    # rug is a colour change with no edge, and a wall the same shade as the floor
    # is an edge with no colour change.
    vgrad = np.zeros_like(img)
    vgrad[:-1, :] = np.abs(img[1:, :] - img[:-1, :])
    gband = vgrad[h - FLOOR_BAND_ROWS:, :]
    gtol = max(GRAD_TOL_MIN, GRAD_TOL_MAD * float(np.median(np.abs(
        gband - np.median(gband)))) + float(np.median(gband)))

    not_floor = (np.abs(img - mu) > tol) | (vgrad > gtol)   # (h, w) bool
    # How much of the band the model rejects. The band IS the floor by assumption,
    # so this is the assumption auditing itself -- see analyse().
    band_bad = float(not_floor[h - FLOOR_BAND_ROWS:, :].mean())
    boundary = np.zeros(w, np.int32)
    for col in range(w):
        column = not_floor[:, col]
        row = h - 1
        run = 0
        while row >= 0:
            if column[row]:
                run += 1
                if run >= BOUNDARY_RUN:
                    row += run                            # first row of the run
                    break
            else:
                run = 0
            row -= 1
        boundary[col] = max(0, min(h, row + 1))
    return (boundary, not_floor, band_bad) if detail else boundary


def analyse(frame: np.ndarray, bins: int = BINS):
    """One pass over a frame -> (profile, confidence, not_floor_fraction).

    ONE PASS ON PURPOSE. free_profile() and confidence() each used to call
    _prepare(), so a tick with two cameras resized and blurred four images to
    answer two questions. This loop already misses its 100 ms budget on 47% of
    ticks; the cameras must not be the reason it misses more.
    """
    img = _prepare(frame)
    if img is None:
        return None, 0.0, 1.0
    h, w = img.shape
    boundary, not_floor, band_bad = floor_boundary(img, detail=True)
    free_rows = (h - boundary).astype(np.float32)
    edges = np.linspace(0, w, bins + 1).astype(int)
    profile = np.empty(bins, np.float32)
    for i in range(bins):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        # The MINIMUM across the bin, not the mean: a bin is only as passable as
        # its most blocked column. Averaging lets a door frame's open half hide
        # the half she would drive into.
        profile[i] = float(free_rows[lo:hi].min()) / float(h)
    profile = np.clip(profile, 0.0, 1.0)

    mean = float(img.mean())
    # Exposure: full marks in the middle of the range, nothing at the rails.
    exposure = min(1.0, max(0.0, min(mean - 12.0, 243.0 - mean) / 28.0))
    # THE MODEL CHECKING ITSELF. The bottom band is the floor by assumption; if the
    # model built from it then calls the band itself not-floor, the assumption was
    # wrong -- a bimodal band, an object at the bumper, a camera looking at a wall.
    # A synthetic half-blocked frame drove this to 1.0 and reported every bin
    # blocked, which is exactly the reading that must not be believed.
    trust = float(np.clip(1.0 - 2.0 * band_bad, 0.0, 1.0))
    frac = float(not_floor.mean())
    # A frame that is almost all obstacle, or almost all floor with no structure
    # at all, is equally uninformative about WHICH WAY to go.
    spread = float(np.clip(1.0 - abs(2.0 * frac - 1.0) * 0.5, 0.0, 1.0))
    texture = float(np.clip(img.std() / TEXTURE_REF_STD, 0.0, 1.0))
    # EXPOSURE AND TEXTURE MULTIPLY, they do not add. If she cannot see, nothing
    # else about the frame is worth anything -- a black frame and a lens against a
    # wall both scored 0.50 and passed the usable threshold when these were merely
    # terms in a sum, which is the one reading that must never reach the wheels.
    conf = float(np.clip(exposure * texture * (0.65 * trust + 0.35 * spread),
                         0.0, 1.0))
    return profile, conf, frac


def free_profile(frame: np.ndarray, bins: int = BINS) -> np.ndarray | None:
    """Floor coverage per angular bin, left to right, each in [0, 1].

    1.0 means the floor runs all the way to the horizon in that direction; 0.0
    means something is standing at her bumper there.
    """
    return analyse(frame, bins)[0]


def confidence(frame: np.ndarray) -> float:
    """How much the floor model can be believed for this frame, in [0, 1]."""
    return analyse(frame)[1]


def row_to_range(free_fraction: float) -> float:
    """Floor coverage -> metres, by flat-ground projection. APPROXIMATE.

    The row where the floor stops sits `free_fraction * h` above the bottom of the
    frame. Its depression below the optical axis fixes the ground distance for a
    camera of known height. Both the height and the tilt here are estimates, so
    this is honest about magnitude and no better -- which is enough for a capped
    fallback and not enough for anything else.
    """
    half_v = math.radians(FOV_V_DEG) / 2.0
    # Bottom row = -half_v below the axis, top row = +half_v above it.
    ang = -half_v + free_fraction * 2.0 * half_v
    depression = math.radians(CAM_TILT_DEG) - ang
    if depression <= math.radians(1.0):                   # at or above the horizon
        return MAX_RANGE_M
    return float(min(MAX_RANGE_M, CAM_HEIGHT_M / math.tan(depression)))


def widest_gap(profile: np.ndarray) -> tuple[int, int, float]:
    """The widest contiguous run of open bins: (start, stop_exclusive, openness).

    NOT the centre of mass. Two doorways either side of a pillar have their
    centre of mass ON the pillar, and a robot that steers by centroid drives into
    it. This is the same choice a person makes walking through a room.
    """
    best = float(profile.max()) if profile.size else 0.0
    if best <= 0.0:
        return 0, 0, 0.0
    open_bins = profile >= OPEN_FRACTION * best
    start = run = best_start = best_len = 0
    for i, is_open in enumerate(open_bins):
        if is_open:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_start = run, start
        else:
            run = 0
    stop = best_start + best_len
    return best_start, stop, float(profile[best_start:stop].mean()) if best_len else 0.0


def gap_bearing(profile: np.ndarray, bins: int | None = None) -> float:
    """Bearing of the widest gap's centre, radians, positive = LEFT (CCW).

    Left is positive because that is the sign convention omega already uses
    everywhere else in this stack; getting it backwards would steer her
    confidently into the thing she can see.
    """
    n = bins or profile.size
    if n <= 0:
        return 0.0
    start, stop, _ = widest_gap(profile)
    if stop <= start:
        return 0.0
    centre = (start + stop) / 2.0                         # in bin units
    # Bin 0 is the LEFT of the image, which is +FOV/2 in robot bearing.
    frac = centre / n                                     # 0 = left edge, 1 = right
    return float(math.radians(FOV_H_DEG) * (0.5 - frac))


class FreeSpaceView:
    """One reading, from whichever cameras were available."""

    __slots__ = ("profile", "bearing", "centre_range", "confidence", "source",
                 "near_blocked", "stamp")

    def __init__(self, profile, bearing, centre_range, conf, source,
                 near_blocked=False):
        self.profile = profile
        self.bearing = bearing                 # [rad] where the most space is
        self.centre_range = centre_range       # [m] straight ahead, APPROXIMATE
        self.confidence = conf
        self.source = source                   # "up", "down", "up+down", "none"
        self.near_blocked = near_blocked       # down camera: floor gone at the bumper
        self.stamp = time.monotonic()

    @property
    def usable(self) -> bool:
        return self.confidence >= MIN_CONFIDENCE and self.source != "none"

    def __repr__(self) -> str:                            # pragma: no cover
        return (f"<FreeSpace {self.source} bearing={math.degrees(self.bearing):+.0f}deg "
                f"ahead={self.centre_range:.2f}m conf={self.confidence:.2f}"
                f"{' NEAR-BLOCKED' if self.near_blocked else ''}>")


# How much of the down camera's view has to stop looking like floor before the
# bumper is called blocked. It sees the half-metre in front of the wheels, so this
# is a last-ditch check for something the ToF's 8x8 patch missed between its zones.
DOWN_BLOCKED_FRACTION = 0.35
# Bearing is smoothed across ticks. The profile is noisy at 15 fps on a moving
# robot, and an advisory that jitters is worse than none -- it is a second source
# of the zig-zag this whole change exists to remove.
BEARING_SMOOTH = 0.4


class FreeSpaceNavigator:
    """Both cameras -> one steering opinion. Cheap, stateless except for smoothing.

    Never raises. A camera that has fallen off the USB bus -- which the down
    camera on this robot does, repeatedly, for physical reasons software cannot
    fix -- must degrade the answer, not the run.
    """

    def __init__(self, bins: int = BINS) -> None:
        self.bins = bins
        self.last = FreeSpaceView(np.zeros(bins, np.float32), 0.0, 0.0, 0.0, "none")
        self.updates = 0
        self.failures = 0
        self._bearing = 0.0

    def update(self, cam_up=None, cam_down=None) -> FreeSpaceView:
        try:
            return self._update(cam_up, cam_down)
        except Exception as exc:                          # noqa: BLE001
            self.failures += 1
            if self.failures <= 3:
                LOGGER.debug("free-space update failed (%s)", exc)
            self.last = FreeSpaceView(np.zeros(self.bins, np.float32),
                                      0.0, 0.0, 0.0, "none")
            return self.last

    def _update(self, cam_up, cam_down) -> FreeSpaceView:
        sources = []
        profile = None
        conf = 0.0
        if cam_up is not None:
            profile, conf, _ = analyse(cam_up, self.bins)
            if profile is not None:
                sources.append("up")

        near_blocked = False
        if cam_down is not None:
            down, down_conf, _ = analyse(cam_down, self.bins)
            if down is not None:
                sources.append("down")
                # Only from a frame worth believing. A featureless down view is
                # the ToF's problem, not something to invent a block from.
                near_blocked = bool(down_conf >= MIN_CONFIDENCE
                                    and float(down.mean()) < DOWN_BLOCKED_FRACTION)
                if profile is None:
                    # Down-camera only. It looks at the floor just ahead, so it can
                    # say WHICH WAY is clear but not how far -- which is still a
                    # steering opinion, and better than none.
                    profile, conf = down, down_conf * 0.7
                else:
                    # Both: the near view can only ever REDUCE what the far view
                    # promises. Something at the bumper is not made passable by
                    # open floor behind it.
                    profile = np.minimum(profile, down)

        if profile is None:
            self.last = FreeSpaceView(np.zeros(self.bins, np.float32),
                                      0.0, 0.0, 0.0, "none")
            return self.last

        raw_bearing = gap_bearing(profile, self.bins)
        # Smoothed, and only while the reading is worth smoothing toward.
        if conf >= MIN_CONFIDENCE:
            self._bearing += BEARING_SMOOTH * (raw_bearing - self._bearing)
        centre = int(self.bins // 2)
        lo, hi = max(0, centre - 1), min(self.bins, centre + 2)
        centre_range = row_to_range(float(profile[lo:hi].min()))
        self.updates += 1
        self.last = FreeSpaceView(profile, self._bearing, centre_range, conf,
                                  "+".join(sources), near_blocked)
        return self.last
