"""Find which raw LiDAR bearing points at Friday's front.

The D500 reports angles in its own frame, and nothing says which of those
angles is the robot's nose. Until that is known, sensors.py refuses to name a
direction — so scan_environment reports left/right/backward as null and the
brain cannot navigate by them. This measures it in about thirty seconds.

    ./venv/bin/python scripts/calibrate_lidar.py

HOW IT WORKS. It takes a baseline sweep of the room, waits while you put an
object directly in front of the robot, then takes a second sweep and finds the
bearing whose distance dropped the most. That difference is the object, and
the object is in front — so that bearing is forward.

Diffing rather than just taking the nearest reading is deliberate: the nearest
thing to the LiDAR is often part of the robot's own chassis, or a wall it
happens to be parked against, and either would give a confident wrong answer.

THERE ARE TWO METHODS. The object method above needs a person, and on this
robot it turned out to be unreliable: the LiDAR sits high enough that a box on
the floor is BELOW its scan plane and simply never appears (measured
2026-08-26 -- three attempts, the "object" never showed up as more than a
single noisy bearing). Standing in front works, but a person at 1.1 m reads as
TWO narrow returns -- left leg and right leg, about 12 degrees apart -- and
picking the midpoint of those is a guess worth several degrees.

    ./venv/bin/python scripts/calibrate_lidar.py --drive

The drive method has no such ambiguity, needs nobody in the room, and uses all
360 bearings instead of one. Friday drives forward a short measured distance
and sweeps again. The direction she moved IS forward, by definition -- so
every bearing's range must change by exactly

    delta(theta) = -D * cos(theta - forward)

which is one cosine with one unknown. Fitting it over the whole sweep averages
away the noise that defeated the object method, and it cannot be fooled by
which part of a person the beam happened to hit.

It drives through safety.guarded_move, the same interrupt-checked path the MCP
server's own move tool uses -- not motor.py directly -- and it takes a real
observation first, because the look-before-you-move gate is not something a
calibration script gets to skip. It drives back afterwards.

Nothing is written automatically by either method. It prints the number for
you to paste into config.LIDAR_FORWARD_DEG, because a wrong value here makes
the robot confident about directions it cannot actually see.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config            # noqa: E402
import sensors           # noqa: E402

# How far to drive for --drive. Far enough that the cosine is well above the
# D500's noise at room ranges, short enough to stay inside the clearance the
# safety layer just checked.
DRIVE_CM = 25.0

SETTLE_S = 2.5
SWEEP_S = 2.0
# A real object has to move a bearing by more than sensor noise. 150 mm is
# comfortably past the D500's jitter at indoor ranges.
MIN_CHANGE_MM = 150


def sweep(lidar: sensors.Lidar, seconds: float) -> dict[int, int]:
    """Average each bearing over a few revolutions, to beat down noise."""
    samples: dict[int, list[int]] = {}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for bearing, distance in lidar.scan(max_age_s=0.4).items():
            samples.setdefault(bearing, []).append(distance)
        time.sleep(0.1)
    return {b: int(statistics.median(v)) for b, v in samples.items() if v}



def fit_forward(before: dict[int, int], after: dict[int, int],
                travelled_mm: float) -> tuple[int, float, int]:
    """Find the bearing that best explains the whole sweep's change.

    Driving forward by D changes the range at bearing theta by
    -D * cos(theta - forward). Rather than search for a peak, project the
    measured deltas onto cos and sin and read the phase straight off -- that
    is a least-squares fit of exactly this model, in two dot products.

    Bearings whose delta is far larger than D are thrown out first. Those are
    not the room moving relative to the robot, they are a surface the beam
    slid off or a doorway it found, and a handful of them would drag the fit.
    """
    shared = sorted(set(before) & set(after))
    limit = travelled_mm * 2.5
    used = [b for b in shared if abs(after[b] - before[b]) <= limit]
    if len(used) < 60:
        raise RuntimeError(f"only {len(used)} bearings agreed with a "
                           f"{travelled_mm:.0f} mm move -- did she actually "
                           f"move, and did the room stay still?")

    # delta = -D cos(theta - phi) = -D cos(theta) cos(phi) - D sin(theta) sin(phi)
    cos_sum = sum((after[b] - before[b]) * math.cos(math.radians(b)) for b in used)
    sin_sum = sum((after[b] - before[b]) * math.sin(math.radians(b)) for b in used)
    forward = math.degrees(math.atan2(-sin_sum, -cos_sum)) % 360.0

    # How much of the change the fit actually explains. A good fit on real
    # geometry sits well above 0.5; near zero means the sweep did not move
    # like a translation at all and the answer is meaningless.
    amplitude = math.hypot(cos_sum, sin_sum) * 2.0 / len(used)
    model = [-amplitude * math.cos(math.radians(b - forward)) for b in used]
    actual = [after[b] - before[b] for b in used]
    ss_res = sum((a - m) ** 2 for a, m in zip(actual, model))
    ss_tot = sum((a - statistics.fmean(actual)) ** 2 for a in actual) or 1.0
    return int(round(forward)) % 360, max(0.0, 1.0 - ss_res / ss_tot), len(used)


def drive_method() -> int:
    """Measure forward by driving forward. See the module docstring."""
    # Imported here, not at the top: the object method must keep working on a
    # robot with no motors attached.
    from mcp_server import embodiment, hardware, safety   # noqa: PLC0415

    print("Friday will drive forward about "
          f"{DRIVE_CM:.0f} cm and then back again.")
    print("Clear the space in front of her and stand out of the way.\n")

    ranges = hardware.ranges()
    ranges.wait_ready()
    lidar = ranges.lidar
    if not lidar.connected:
        print(f"LiDAR is not connected ({lidar.last_error or 'absent'}).")
        return 1

    # LOOK BEFORE YOU MOVE. The gate in embodiment.py is not something a
    # calibration script gets to bypass, so this is a real observation: the
    # bumper ToF is read now and recorded as what Friday can actually see.
    clearance_mm, _, _ = safety.check_forward_clearance()
    if clearance_mm is None:
        print("The bumper ToF cannot see anything, so forward clearance is")
        print("unknown -- and unknown is never treated as clear. Aborting.")
        return 1
    needed = DRIVE_CM * 10 + config.SAFETY_STOP_MM
    if clearance_mm < needed:
        print(f"Only {clearance_mm} mm ahead; this needs about "
              f"{needed:.0f} mm. Move her somewhere more open.")
        return 1
    embodiment.record(clearance_mm={"forward": clearance_mm},
                      source="calibrate_lidar")
    print(f"  clearance ahead: {clearance_mm} mm\n")

    time.sleep(SETTLE_S)
    before = sweep(lidar, SWEEP_S * 2)
    print(f"  sweep before: {len(before)} bearings")

    seconds = DRIVE_CM / config.load_calibration()["forward_cm_per_s"]
    out = safety.guarded_move("forward", seconds)
    print(f"  {out.as_text()}")
    travelled_mm = out.actual_s * config.load_calibration()["forward_cm_per_s"] * 10
    if out.stopped_early or travelled_mm < 80:
        print("She did not travel far enough to fit anything to. Aborting.")
        # Record before reversing, or the gate refuses the retreat and she is
        # left parked wherever she stopped -- which is what happened the first
        # time this aborted.
        embodiment.record(clearance_mm={"backward": None},
                          source="calibrate_lidar")
        print(f"  returning: {safety.guarded_move('backward', out.actual_s).as_text()}")
        return 1

    time.sleep(SETTLE_S)
    after = sweep(lidar, SWEEP_S * 2)
    print(f"  sweep after:  {len(after)} bearings")
    print(f"  travelled ~{travelled_mm:.0f} mm\n")

    # Put her back where she started before anything else can go wrong.
    embodiment.record(clearance_mm={"backward": None}, source="calibrate_lidar")
    back = safety.guarded_move("backward", out.actual_s)
    print(f"  returning: {back.as_text()}\n")

    try:
        forward, quality, used = fit_forward(before, after, travelled_mm)
    except RuntimeError as exc:
        print(f"Fit failed: {exc}")
        return 1

    print("=" * 62)
    print(f"  Forward is raw bearing {forward}°")
    print(f"  (cosine fit over {used} bearings, explains "
          f"{quality * 100:.0f}% of the change)")
    print("=" * 62)
    if quality < 0.4:
        print("\n⚠ THAT FIT IS TOO POOR TO TRUST. The sweep did not change")
        print("  the way a translation makes it change -- something in the")
        print("  room moved, or she did not really go where she was told.")
        print("  Do not paste this number in. Run it again in a still room.")
        return 1
    print()
    print("Put this in config.py:")
    print()
    print(f"    LIDAR_FORWARD_DEG = {forward}")
    print()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--drive", action="store_true",
                    help="measure it by driving forward instead of by "
                         "putting an object in front (more reliable)")
    if ap.parse_args().drive:
        return drive_method()

    print(__doc__.split("HOW IT WORKS")[0].strip())
    print()

    with sensors.Rangefinders() as ranges:
        lidar = ranges.lidar
        ranges.wait_ready()
        if not lidar.connected:
            print(f"LiDAR is not connected ({lidar.last_error or 'absent'}).")
            print(f"Expected at {config.LIDAR_PORT}")
            return 1

        print("Clear the space directly in front of Friday.")
        input("Press Enter to take the baseline sweep... ")
        time.sleep(SETTLE_S)
        before = sweep(lidar, SWEEP_S)
        print(f"  baseline: {len(before)} bearings, "
              f"{lidar.health}\n")

        print("Now put a box (or a book, or your leg) about 40-60 cm directly")
        print("in front of the robot — squarely in front, not off to a side.")
        input("Press Enter when it is there... ")
        time.sleep(SETTLE_S)
        after = sweep(lidar, SWEEP_S)
        print(f"  second sweep: {len(after)} bearings\n")

    shared = sorted(set(before) & set(after))
    if not shared:
        print("The two sweeps share no bearings — is the LiDAR spinning?")
        return 1

    drops = [(before[b] - after[b], b) for b in shared]
    drops.sort(reverse=True)
    best_drop, best_bearing = drops[0]

    if best_drop < MIN_CHANGE_MM:
        print(f"Nothing moved by more than {MIN_CHANGE_MM} mm "
              f"(largest change was {best_drop} mm at {best_bearing}°).")
        print("The object may be out of the LiDAR's plane — it sits on the")
        print("top dome, so something low on the floor is invisible to it.")
        print("Try again with something tall enough to reach the dome.")
        return 1

    # A real object spans several degrees. Take the middle of the run of
    # bearings that all dropped, rather than the single sharpest one, which
    # is likelier to be an edge.
    hits = sorted(b for drop, b in drops if drop >= best_drop * 0.5)
    contiguous = [b for b in hits if abs(b - best_bearing) <= 20]
    centre = int(round(statistics.median(contiguous or [best_bearing])))

    print("=" * 62)
    print(f"  Forward is raw bearing {centre}°")
    print(f"  ({before.get(centre, '?')} mm -> {after.get(centre, '?')} mm, "
          f"across {len(contiguous)} bearings)")
    print("=" * 62)
    print()
    print("Put this in config.py:")
    print()
    print(f"    LIDAR_FORWARD_DEG = {centre}")
    print()
    print("Then re-run `python -m mcp_server --check` — lidar_bearings should")
    print("say calibrated, and scan_environment will start reporting left,")
    print("right and backward instead of null.")
    print()
    print("Sanity-check it: run `python sensors.py` and walk around the robot.")
    print("The direction you are standing in should be the one that changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
