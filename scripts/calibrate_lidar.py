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

Nothing is written automatically. It prints the number for you to paste into
config.LIDAR_FORWARD_DEG, because a wrong value here makes the robot confident
about directions it cannot actually see.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config            # noqa: E402
import sensors           # noqa: E402

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


def main() -> int:
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
