"""Does the scan matcher recover a translation, and does the map hold a room?

    ./venv/bin/python scripts/test_slam.py
    ./venv/bin/python scripts/test_slam.py --show     # print the built map

No robot needed. A synthetic room is ray-cast from two known poses and the
matcher is asked to work out which way the robot went; the answer is checked
against the truth. This exists because the first attempt at measuring
LIDAR_FORWARD_DEG fitted a cosine to the range deltas, got a confident-looking
number, and was wrong -- the model was wrong, and nothing in the pipeline was
able to say so. A test with a known answer would have caught it in seconds.

The room is deliberately awkward: oblique walls and a pillar, because a cosine
fit works on a circular room and fails on a real one.

The second half builds an occupancy map of that same room from six poses and
checks it against the geometry it was made from: walls where the walls are,
free space where the floor is, and -- the one that matters most -- the pillar
CLEARING once it is taken away. A map that cannot forget is a map that fills
up with furniture that has already moved.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                              # noqa: E402
from slam.core import match_translation, points_from_scan  # noqa: E402
from slam.grid import OccupancyGrid                         # noqa: E402
from slam.mapper import Mapper                              # noqa: E402

# (x0, y0, x1, y1) segments in metres. A skewed room with a pillar in it.
ROOM = [
    (-3.0, -2.5, 3.5, -2.2),      # bottom wall, slightly off square
    (3.5, -2.2, 3.1, 2.6),        # right wall, oblique
    (3.1, 2.6, -2.6, 2.9),        # top wall
    (-2.6, 2.9, -3.0, -2.5),      # left wall
    (0.6, 0.4, 1.0, 0.4),         # a pillar, four short sides
    (1.0, 0.4, 1.0, 0.9),
    (1.0, 0.9, 0.6, 0.9),
    (0.6, 0.9, 0.6, 0.4),
]


def cast(pose: tuple[float, float], bearing_deg: float,
         room: list | None = None) -> float | None:
    """Distance in metres from pose along bearing to the nearest wall."""
    ox, oy = pose
    rad = math.radians(bearing_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    best = None
    for x0, y0, x1, y1 in (room if room is not None else ROOM):
        sx, sy = x1 - x0, y1 - y0
        denom = dx * sy - dy * sx
        if abs(denom) < 1e-12:
            continue
        t = ((x0 - ox) * sy - (y0 - oy) * sx) / denom       # along the ray
        u = ((x0 - ox) * dy - (y0 - oy) * dx) / denom       # along the segment
        if t > 0.05 and 0.0 <= u <= 1.0 and (best is None or t < best):
            best = t
    return best


def sweep(pose: tuple[float, float], noise_mm: float = 8.0,
          rng: np.random.Generator | None = None,
          room: list | None = None) -> dict[int, int]:
    """A full 360-degree scan from a pose, as sensors.Lidar.scan() returns."""
    rng = rng or np.random.default_rng(0)
    out = {}
    for bearing in range(360):
        if any(low <= bearing <= high for low, high in config.LIDAR_SELF_MASK):
            continue                                # Friday's own chassis
        hit = cast(pose, bearing, room)
        if hit is None:
            continue
        mm = hit * 1000.0 + rng.normal(0.0, noise_mm)
        if 0 < mm <= config.LIDAR_MAX_RANGE_MM:
            out[bearing] = int(mm)
    return out


# Poses the robot is imagined to have stood at while mapping. Spread out, and
# all of them with line of sight past the pillar, so every one of its four
# sides is seen by something.
MAP_POSES = [(0.0, 0.0), (-1.6, -1.4), (2.2, 1.9), (2.2, -1.4),
             (-1.6, 2.0), (1.4, 1.9)]

# Points that are unambiguously open floor -- no wall or pillar within 30 cm.
OPEN_FLOOR = [(0.0, 0.0), (-2.0, 0.0), (2.0, -1.0), (0.0, 2.0), (2.5, 1.5)]


def wall_points(room: list, per_metre: int = 30) -> np.ndarray:
    """Sample the room's surfaces, as ground truth for "there is a wall here"."""
    out = []
    for x0, y0, x1, y1 in room:
        length = math.hypot(x1 - x0, y1 - y0)
        for t in np.linspace(0.0, 1.0, max(int(length * per_metre), 2)):
            out.append((x0 + (x1 - x0) * t, y0 + (y1 - y0) * t))
    return np.asarray(out)


def build_map(rng: np.random.Generator, room: list | None = None,
              grid: OccupancyGrid | None = None,
              poses: list | None = None) -> OccupancyGrid:
    """Fold a sweep from each pose into a map. theta is 0 and forward_deg is
    0 because the synthetic sweeps are already in world bearings -- there is
    no LiDAR mounting offset to undo in a simulation."""
    grid = grid if grid is not None else OccupancyGrid()
    for pose in (poses if poses is not None else MAP_POSES):
        grid.integrate_scan(sweep(pose, rng=rng, room=room),
                            (pose[0], pose[1], 0.0), forward_deg=0.0)
    return grid


def near_occupied(grid: OccupancyGrid, points: np.ndarray) -> np.ndarray:
    """Is each point on, or within one cell of, an occupied cell?

    One cell of slack is not generosity: a surface sampled at an arbitrary
    point lands anywhere inside a 5 cm cell, and the beam that found it landed
    anywhere inside its own. Demanding the same cell would be testing the
    rounding, not the map.
    """
    fat = binary_dilation(grid.occupied_mask(), np.ones((3, 3), bool))
    ix, iy = grid.cells_of(points)
    here = grid.inside(points)
    out = np.zeros(len(points), dtype=bool)
    out[here] = fat[ix[here], iy[here]]
    return out


def report(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f": {detail}" if detail else ""))
    return ok


def robot_sweep(pose: tuple[float, float, float],
                rng: np.random.Generator,
                room: list | None = None) -> dict[int, int]:
    """A sweep as the ROBOT sees it -- bearings relative to its own nose.

    The synthetic `sweep` casts in world bearings, which is right for the
    matcher tests but wrong here: a LiDAR bolted to a robot turns with it, so
    a sweep taken at heading theta reports everything theta degrees earlier.
    Getting this backwards makes pose tracking look broken in a way that is
    very hard to see, so it is done once, here.

    Bearings are rounded to whole degrees on the way out, exactly as
    sensors.Lidar._ingest does, collisions and all.
    """
    x, y, theta = pose
    out = {}
    for bearing, mm in sweep((x, y), rng=rng, room=room).items():
        out[int(round((bearing - math.degrees(theta)) % 360.0)) % 360] = mm
    return out


def drive(pose: tuple[float, float, float], forward: float,
          turn_deg: float) -> tuple[float, float, float]:
    turn = math.radians(turn_deg)
    mid = pose[2] + turn / 2.0
    return (pose[0] + forward * math.cos(mid),
            pose[1] + forward * math.sin(mid), pose[2] + turn)


def clear_spot(mapper: Mapper, clear_m: float = 0.5,
               from_m: float = 0.8, within_m: float = 2.5) -> tuple[float, float]:
    """Somewhere the robot has MAPPED as open, to stand a person in.

    Not a hard-coded coordinate. Where the drive ends depends on noise that
    every earlier test in this file has already drawn from the same generator,
    so a fixed spot is visible on some runs and behind the pillar on others --
    which produces a test that fails for a reason that has nothing to do with
    the code under test. This asks the map instead: the nearest cell that is
    known free, half a metre clear of anything solid, and close enough to be
    seen well.
    """
    occupied = mapper.grid.occupied_mask()
    room = distance_transform_edt(~occupied) * mapper.grid.cell
    cells = np.argwhere(mapper.grid.free_mask() & (room >= clear_m))
    if not len(cells):
        raise RuntimeError("no open floor on the map to stand anyone in")
    centres = (cells + 0.5) * mapper.grid.cell + mapper.grid.origin
    away = np.hypot(centres[:, 0] - mapper.pose[0], centres[:, 1] - mapper.pose[1])
    # Not right under the wheels: the nearest free cell is the one the robot
    # is standing on, and a person there is not a test of anything.
    usable = centres[(away >= from_m) & (away <= within_m)]
    if not len(usable):
        raise RuntimeError(f"no open floor {from_m}-{within_m} m from the robot")
    return tuple(usable[len(usable) // 2])


def probe_disc(at: tuple[float, float], radius: float = 0.20,
               step: float = 0.025) -> np.ndarray:
    """The patch of floor a person occupies, as points -- not their outline.

    Probing the box's perimeter looks tidier and is wrong: re-localising
    against a map that is itself a couple of centimetres warped moves the
    whole sweep by a cell or two, and a one-cell-thick outline then misses
    entirely. The question being asked is "is there something standing HERE",
    so ask it about the area. clear_spot guarantees half a metre of clearance,
    so a 20 cm radius cannot accidentally include a wall.
    """
    grid = np.arange(-radius, radius + 1e-9, step)
    gx, gy = np.meshgrid(grid, grid)
    keep = gx ** 2 + gy ** 2 <= radius ** 2
    return np.column_stack([at[0] + gx[keep], at[1] + gy[keep]])


def person_box(at: tuple[float, float], half: float = 0.10) -> list:
    """A 20 cm box: near enough a standing person to a LiDAR at leg height."""
    x, y = at
    return [(x - half, y - half, x + half, y - half),
            (x + half, y - half, x + half, y + half),
            (x + half, y + half, x - half, y + half),
            (x - half, y + half, x - half, y - half)]


def mapper_tests(rng: np.random.Generator, show: bool = False) -> int:
    failures = 0
    print("\ntracking a pose with no encoders: odometry is 20% wrong on purpose")

    truth = (0.0, 0.0, 0.0)
    mapper = Mapper(pose=truth, forward_deg=0.0)
    odom_only = truth
    # Sit still for a moment first -- nothing can match against a map that
    # does not exist yet, which is why the first few updates say so.
    for _ in range(4):
        mapper.update(robot_sweep(truth, rng), 0.0, 0.0)

    plan = ([(0.25, 0.0)] * 6 + [(0.0, 45.0)] * 2 + [(0.25, 0.0)] * 6
            + [(0.0, 45.0)] * 2 + [(0.25, 0.0)] * 6 + [(0.0, -30.0)]
            + [(0.25, 0.0)] * 4)
    untrusted = 0
    for forward, turn_deg in plan:
        truth = drive(truth, forward, turn_deg)
        # No encoders: what odometry REPORTS is the timed guess, +/-20%.
        odo_f = forward * (1.0 + rng.normal(0.0, 0.12))
        odo_t = math.radians(turn_deg * (1.0 + rng.normal(0.0, 0.12)))
        odom_only = drive(odom_only, odo_f, math.degrees(odo_t))
        out = mapper.update(robot_sweep(truth, rng), odo_f, odo_t)
        untrusted += not out["trusted"]

    matched = math.hypot(mapper.pose[0] - truth[0], mapper.pose[1] - truth[1])
    naive = math.hypot(odom_only[0] - truth[0], odom_only[1] - truth[1])
    print(f"        drove {mapper.travelled_m:.1f} m over {len(plan)} moves, "
          f"{untrusted} update(s) fell back to odometry")
    print(f"        odometry alone ended {naive * 100:.0f} cm out; "
          f"matched to the map, {matched * 100:.0f} cm out")
    failures += not report("scan matching beats dead reckoning",
                           matched < naive * 0.6,
                           f"{naive / max(matched, 1e-6):.1f}x better")
    failures += not report("and is good to a few centimetres", matched < 0.10,
                           f"{matched * 100:.1f} cm from truth")
    heading_err = abs(math.degrees((mapper.pose[2] - truth[2] + math.pi)
                                   % (2 * math.pi) - math.pi))
    failures += not report("heading holds too", heading_err < 5.0,
                           f"{heading_err:.1f} deg out")

    # --- a person walks in ----------------------------------------------
    print("\nsomeone walks in and stands there -- when does the map believe it?")
    here = mapper.pose
    stand_at = clear_spot(mapper)
    person = person_box(stand_at)
    probe = probe_disc(stand_at)
    with_person = ROOM + person
    print(f"        robot at ({here[0]:.2f}, {here[1]:.2f}), someone steps in "
          f"at ({stand_at[0]:.2f}, {stand_at[1]:.2f})")

    def person_state() -> tuple[int, int]:
        """(sweeps the most-seen part of them was tracked in, cells mapped)"""
        return (int(mapper.transient.count_at(probe).max()),
                int((mapper.grid.classify(probe) == 1).sum()))

    for _ in range(2):
        mapper.update(robot_sweep(here, rng, room=with_person), 0.0, 0.0)
    seen, occupied = person_state()
    failures += not report("seen twice: tracked, but NOT on the map",
                           seen >= 2 and occupied == 0,
                           f"tracked in {seen} sweeps, {occupied} cells mapped")
    blocking = mapper.transient.blocking_points()
    near = (np.hypot(blocking[:, 0] - stand_at[0],
                     blocking[:, 1] - stand_at[1]).min()
            if len(blocking) else 99.0)
    failures += not report("but it IS in the way, and known to be",
                           near < 0.30, f"nearest blocking cell {near:.2f} m")
    flagged = len(mapper.transient.unconfirmed_points())
    failures += not report("and flagged as not-furniture, so it can be said "
                           "out loud", flagged > 0,
                           f"{flagged} unconfirmed cells")

    for _ in range(6):
        mapper.update(robot_sweep(here, rng, room=with_person), 0.0, 0.0)
    seen, occupied = person_state()
    failures += not report("still there after eight sweeps: now it is mapped",
                           occupied > 0, f"{occupied} cells now occupied")

    # --- and walks off again ---------------------------------------------
    print("\nthey walk off -- the map has to let go, or it fills up with ghosts")
    gone = None
    for sweeps in range(1, 31):
        mapper.update(robot_sweep(here, rng), 0.0, 0.0)
        if person_state()[1] == 0:
            gone = sweeps
            break
    failures += not report("the ghost clears", gone is not None,
                           f"gone after {gone} sweeps" if gone
                           else "still on the map after 30 sweeps")
    walls = float(near_occupied(mapper.grid, wall_points(ROOM)).mean())
    failures += not report("the room survived all of that", walls >= 0.85,
                           f"{walls * 100:.1f}% of wall surface still mapped")

    if show:
        print("\n" + mapper.grid.to_ascii())
        print(f"\n{mapper.status()}")
    return failures

def map_tests(rng: np.random.Generator, show: bool = False) -> int:
    failures = 0
    print("\nbuilding an occupancy map of the same room from six poses\n")
    grid = build_map(rng)
    stats = grid.stats()
    print(f"        {stats['shape'][0]}x{stats['shape'][1]} cells at "
          f"{stats['cell_m']} m, {stats['explored_m2']} m2 explored, "
          f"{stats['occupied']} occupied / {stats['free']} free / "
          f"{stats['unknown']} unknown")

    truth = wall_points(ROOM)
    found = near_occupied(grid, truth)
    fraction = float(found.mean())
    failures += not report("the walls are where the walls are", fraction >= 0.90,
                           f"{fraction * 100:.1f}% of sampled surface is mapped "
                           f"occupied (want >= 90%)")

    floor = np.asarray(OPEN_FLOOR)
    states = grid.classify(floor)
    failures += not report("open floor is carved free", bool((states == -1).all()),
                           f"{int((states == -1).sum())}/{len(floor)} known free")

    # The inside of the pillar is never seen by any beam -- they all stop on
    # its surface. Unknown is the correct answer, and it must not read free.
    inside_pillar = grid.classify(np.array([[0.8, 0.65]]))[0]
    failures += not report("the inside of the pillar is unknown, not free",
                           inside_pillar != -1,
                           {1: "occupied", 0: "unknown", -1: "FREE"}[int(inside_pillar)])

    outside = grid.classify(np.array([[6.0, 6.0], [-8.0, 0.0]]))
    failures += not report("space nobody has looked at is never free",
                           bool((outside != -1).all()))

    # --- the one that matters: a map that can change its mind ------------
    print("\nthe pillar is taken away -- how long until the map lets go of it?")
    empty_room = list(ROOM[:4])
    pillar_face = wall_points(ROOM[4:])
    cleared = None
    for sweeps in range(1, 21):
        build_map(rng, room=empty_room, grid=grid)
        states = grid.classify(pillar_face)
        if (states == -1).mean() >= 0.9:
            cleared = sweeps * len(MAP_POSES)
            break
    failures += not report("a moved obstacle clears", cleared is not None,
                           f"gone after {cleared} sweeps" if cleared
                           else "still on the map after 120 sweeps")
    # And it must not have taken the walls with it.
    still = float(near_occupied(grid, wall_points(empty_room)).mean())
    failures += not report("the real walls survived the clearing", still >= 0.90,
                           f"{still * 100:.1f}% still mapped")

    # --- persistence -----------------------------------------------------
    print("\nthe map has to survive a power cut")
    with tempfile.TemporaryDirectory() as tmp:
        path = grid.save(Path(tmp) / "map.npz")
        back = OccupancyGrid.load(path)
        same = (np.array_equal(back.logodds, grid.logodds)
                and np.array_equal(back.origin, grid.origin)
                and back.cell == grid.cell and back.scans == grid.scans)
        failures += not report("saved and loaded unchanged", same,
                               f"{path.stat().st_size / 1024:.0f} KB on disk, "
                               f"{back.scans} scans")

    # --- the growth cap --------------------------------------------------
    print("\na bad pose must not be able to allocate the Pi to death")
    try:
        OccupancyGrid().integrate(np.array([[500.0, 500.0]]), (0.0, 0.0))
        failures += not report("a point 700 m out is refused", False,
                               "it was accepted")
    except ValueError as exc:
        failures += not report("a point 700 m out is refused", True, str(exc)[:60])

    if show:
        print("\n" + grid.to_ascii())
    return failures


def check(name: str, got: float, want: float, tol: float) -> bool:
    delta = abs((got - want + 180.0) % 360.0 - 180.0)
    ok = delta <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: got {got:.1f}째, "
          f"want {want:.1f}째, off by {delta:.1f}째 (tolerance {tol}째)")
    return ok


def main() -> int:
    rng = np.random.default_rng(7)
    start = (0.0, 0.0)
    failures = 0

    print("recovering a known translation from two synthetic sweeps\n")
    for truth, distance in ((0.0, 0.25), (90.0, 0.25), (203.0, 0.25),
                            (317.0, 0.25), (45.0, 0.40), (270.0, 0.15)):
        rad = math.radians(truth)
        end = (start[0] + distance * math.cos(rad),
               start[1] + distance * math.sin(rad))
        before = points_from_scan(sweep(start, rng=rng))
        after = points_from_scan(sweep(end, rng=rng))
        got, score = match_translation(before, after, distance)
        # 25 cm at 5 cm cells with a 12 cm blur: a couple of degrees is the
        # resolution the geometry supports, not a slack tolerance.
        ok = check(f"moved {distance * 100:.0f} cm along {truth:.0f}째",
                   got, truth, 6.0)
        print(f"        score {score:.3f}")
        failures += not ok

    print("\na sweep matched against itself scores 1.0 by construction")
    here = points_from_scan(sweep(start, rng=rng))
    _, score = match_translation(here, here, 0.0, refine=False)
    ok = 0.95 <= score <= 1.05
    print(f"  {'PASS' if ok else 'FAIL'}  self-match score {score:.3f} "
          f"(want ~1.0)")
    failures += not ok

    print("\na WRONG answer must score clearly worse, or the threshold is "
          "meaningless")
    rad = math.radians(30.0)
    end = (start[0] + 0.25 * math.cos(rad), start[1] + 0.25 * math.sin(rad))
    before = points_from_scan(sweep(start, rng=rng))
    after = points_from_scan(sweep(end, rng=rng))
    right = match_translation(before, after, 0.25)[1]
    from slam.core import LikelihoodField, transform
    field = LikelihoodField(before)
    wrong_rad = math.radians(210.0)                  # 180 degrees out
    wrong = field.score(transform(after, 0.25 * math.cos(wrong_rad),
                                  0.25 * math.sin(wrong_rad), 0.0))
    print(f"        correct bearing scores {right:.3f}")
    print(f"        180째 out scores        {wrong:.3f}")
    ok = right > wrong * 1.3 and wrong < config.SLAM_MIN_SCORE < right
    print(f"  {'PASS' if ok else 'FAIL'}  SLAM_MIN_SCORE "
          f"({config.SLAM_MIN_SCORE}) separates them")
    failures += not ok

    failures += map_tests(rng, show="--show" in sys.argv)
    failures += mapper_tests(rng, show="--show" in sys.argv)

    print(f"\n{'all good' if not failures else str(failures) + ' failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
