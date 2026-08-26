"""Does pose tracking survive a real pivot? Turn in place and see.

    ./venv/bin/python scripts/pivot_test.py            # dry run, no motors
    ./venv/bin/python scripts/pivot_test.py --turn     # actually pivot

⚠ THIS ONE MOVES THE ROBOT — pivots only, never translates, and only with
--turn. Ask Harsh before running it with --turn (CLAUDE.md: ask before
motors). It turns 90 degrees one way in 30 degree steps and 90 back, so it
finishes where it started.

⚠ EVERY TURN IS PRECEDED BY scan_environment(). Not decoration: the movement
gate (mcp_server/embodiment.check) refuses to move at all until Friday has
actually looked, and refuses again once that observation goes stale. The first
version of this script did not, so all six turns were REFUSED while it happily
reported the pivot as a PASS -- the robot had not moved, so of course its
apparent translation was zero. A test that scores "nothing happened" as
success is worse than no test.

⚠ MOTORS GO THROUGH turn_degrees, NOT THROUGH motor.py. That tool runs
embodiment.check and safety.guarded_move, which is where the interrupt lives
and which cannot be overridden. Driving the motors directly from a test script
would bypass every one of those, and a test is exactly the wrong place to find
out what that feels like.

WHAT IT IS ACTUALLY TESTING. map_here proved the LiDAR feeds slam/ and a room
comes out. It could not test POSE TRACKING, because the robot never moved. A
pivot is the smallest honest test of that: the robot's position does not
change, so anything the estimator reports as translation is error, and the map
either stays one room or smears into two overlapping ones.

Rotation is also the motion this chassis is worst at estimating -- there are no
encoders, so a turn is a TIMED guess worth about ±20% -- and the motion the
scan matcher should be best at correcting, because a room's walls constrain
rotation strongly. If pose tracking cannot hold through a pivot, nothing that
drives is worth trying.
"""

from __future__ import annotations

import asyncio
import copy
import json
import math
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession                                    # noqa: E402
from mcp.client.streamable_http import streamable_http_client    # noqa: E402

import config                                                    # noqa: E402
from slam import core                                            # noqa: E402
from slam.mapper import Mapper                                   # noqa: E402

# 30 degrees, not 15: at 92 deg/s a 15 degree step is a 163 ms motor
# pulse, and a chassis with no encoders may simply not break static
# friction in that time. A step that never moved looks EXACTLY like
# flawless pose tracking, so the gentler test is the less honest one.
STEP_DEG = 30.0
SWEEP_EACH_STEP = 3


@asynccontextmanager
async def session():
    async with streamable_http_client(config.MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            yield client


def text_of(result) -> str:
    for block in result.content:
        got = getattr(block, "text", None)
        if got:
            return got
    return ""


async def sweep(client) -> dict[int, int]:
    """One real sweep as {bearing: mm}, straight from the running LiDAR."""
    data = json.loads(text_of(await client.call_tool("lidar_sweep",
                                                     {"full": True})))
    return {int(b): int(mm) for b, mm in data.get("bearings_mm", {}).items()}


async def settle(client, mapper: Mapper, sweeps: int, turned_rad: float = 0.0,
                 log: list | None = None):
    """Fold sweeps in, attributing the turn to the FIRST one only.

    The odometry for a step belongs to the sweep taken right after it; telling
    the mapper the robot turned again on every subsequent stationary sweep
    would be a straightforward lie and would drag the estimate round with it.
    """
    # LET THE CLOUD CLEAR BEFORE READING IT. A scan is a rolling window, so
    # the instant a turn ends the newest revolution still contains bearings
    # measured while the robot was rotating. Folding those in transforms
    # points captured at several headings by a single pose, which is what
    # smeared the map. One full revolution of settling and every point in the
    # window postdates the stop. Costs 120 ms per step; it bought the map back.
    if turned_rad:
        await asyncio.sleep(config.LIDAR_SWEEP_MAX_AGE_S)

    for index in range(sweeps):
        scan = await sweep(client)
        if scan:
            turn = turned_rad if index == 0 else 0.0
            result = mapper.update(scan, 0.0, turn)
            # Raw sweep AND the odometry it was given, saved before anything
            # is concluded from it -- scripts/calibrate_lidar.py does the same,
            # for the same reason: re-analysis should never need the robot to
            # move again. Every question asked of this run afterwards is asked
            # of this file, not of the motors.
            if log is not None:
                # ⚠ "at" IS NOT DECORATION. TransientLayer expires returns on
                # a 4 s wall clock, so a replay that fires 30 sweeps in a
                # second confirms cells the live run let expire and rebuilds a
                # DIFFERENT map. Reproducing the pose needs only the odometry;
                # reproducing the MAP needs this timestamp fed back through
                # Mapper.update(now=...). Found the hard way: a replay scored
                # 0.63x where the live run had scored 0.45x, off the same
                # sweeps.
                log.append({"scan": scan, "turned_rad": turn, "at": time.time(),
                            "pose": list(result["pose"]),
                            "score": result["score"],
                            "sharpness": result["sharpness"],
                            "reason": result["reason"]})
        await asyncio.sleep(0.15)


async def main() -> int:
    turn = "--turn" in sys.argv
    print(f"connecting to {config.MCP_HTTP_URL}")

    async with session() as client:
        first = await sweep(client)
        if not first:
            print("no sweep — run scripts/map_here.py first to check the sensor")
            return 1
        print(f"sensor ok, {len(first)} bearings\n")

        log: list = []
        mapper = Mapper(pose=(0.0, 0.0, 0.0))
        print("building a baseline map from standstill")
        await settle(client, mapper, 12, log=log)
        floor_before = mapper.grid.free_mask().sum() * mapper.grid.cell ** 2
        walls_before = int(mapper.grid.occupied_mask().sum())
        print(f"  {floor_before:.1f} m2 floor, {walls_before} wall cells")

        # THE CONTROL, and the whole reason this test can now fail honestly.
        # Built before a wheel turns, so it cannot contain any pivot's error.
        # Everything after this is scored against it rather than against
        # itself -- see the fit check at the bottom.
        baseline_grid = copy.deepcopy(mapper.grid)

        if not turn:
            print("\n(dry run — pass --turn to actually pivot)")
            return 0

        plan = [STEP_DEG] * 3 + [-STEP_DEG] * 3
        commanded = 0.0
        print(f"\npivoting {sum(abs(d) for d in plan):.0f} degrees in "
              f"{len(plan)} steps, finishing where it started")
        for degrees in plan:
            # Counts as having looked, and is what the gate wants. Cheap --
            # range sensors only, no camera.
            await client.call_tool("scan_environment", {})
            said = text_of(await client.call_tool("turn_degrees",
                                                  {"degrees": degrees}))

            if "REFUSED" in said.upper():
                print(f"  asked {degrees:+.0f}deg -> REFUSED: {said}")
                print("\nthe robot did not move, so there is nothing to "
                      "measure. Not reporting a result.")
                return 1

            # What it ACTUALLY turned, which is not always what was asked:
            # guarded_move can cut a turn short, and telling the mapper it
            # turned the full amount would be feeding it a known lie.
            # "Turned about 12 degrees of the 15 asked for" -- the number
            # sits after "about", and _margin ALWAYS writes that word. A
            # pattern that misses it falls through to the commanded angle,
            # which is the one value guaranteed to be wrong here.
            done = re.search(r"Turned\s+(?:about\s+)?([0-9]+(?:\.[0-9]+)?)",
                             said)
            if not done:
                print(f"  could not read the angle out of: {said}")
                print("\n  refusing to feed the mapper a guessed turn.")
                return 1
            actual = math.copysign(float(done.group(1)), degrees)
            commanded += actual

            await settle(client, mapper, SWEEP_EACH_STEP,
                         math.radians(actual), log=log)
            heading = math.degrees(mapper.pose[2])
            drift = math.hypot(mapper.pose[0], mapper.pose[1])
            print(f"  asked {degrees:+.0f}deg, turned {actual:+.0f} -> "
                  f"total {commanded:+.0f}, estimator says {heading:+.1f}, "
                  f"apparent shift {drift * 100:.0f} cm")
            if "stopped" in said.lower():
                print(f"    cut short: {said}")

        heading = math.degrees(mapper.pose[2])
        drift = math.hypot(mapper.pose[0], mapper.pose[1])
        floor_after = mapper.grid.free_mask().sum() * mapper.grid.cell ** 2
        walls_after = int(mapper.grid.occupied_mask().sum())

        print("\nback where it started, so both of these should be near zero")
        ok = True
        # A pivot does not translate. Anything here is estimator error, and
        # 15 cm is already more than the cell size by a factor of three.
        good_drift = drift < 0.15
        print(f"  {'PASS' if good_drift else 'FAIL'}  apparent translation "
              f"{drift * 100:.0f} cm (want under 15)")
        ok &= good_drift
        good_heading = abs(heading - commanded) < 10.0
        print(f"  {'PASS' if good_heading else 'FAIL'}  final heading "
              f"{heading:+.1f}deg vs {commanded:+.0f} commanded "
              f"(want within 10)")
        ok &= good_heading
        # A smeared map paints every wall twice. Seeing MORE floor is expected
        # and fine -- the pivot reveals what was behind her.
        growth = walls_after / max(walls_before, 1)
        good_walls = growth < 1.8
        print(f"  {'PASS' if good_walls else 'FAIL'}  wall cells "
              f"{walls_before} -> {walls_after} ({growth:.2f}x; over 1.8 means "
              f"the map is drawing every wall twice)")
        ok &= good_walls
        print(f"        floor {floor_before:.1f} -> {floor_after:.1f} m2 "
              f"(growth here is good: it saw behind itself)")

        # THE CHECK THE OTHER THREE CANNOT MAKE.
        #
        # Every gate above reads the map the pivot itself built, and that map
        # is built AT the estimated pose -- so a sweep folded in at a heading
        # 5 degrees wrong lands on a map that already leans 5 degrees, fits it
        # perfectly, and reports "matched". The error is invisible from the
        # inside. That is not a bug in match_pose; it is what scan-to-map SLAM
        # does without an outside reference.
        #
        # baseline_grid is that outside reference: built at standstill before
        # a wheel turned, so it cannot carry the pivot's error. Score one
        # fresh sweep against BOTH maps from the same spot. If the pivoted map
        # explains a real sweep much worse than the standstill map does, the
        # pivot damaged it -- however good the numbers above look.
        #
        # The window is deliberately wide. A timed 180 degrees at +/-20% can
        # leave the robot tens of degrees from where it started, and scoring
        # at a prior of zero would blame the map for that.
        print("\ndoes a real sweep still fit the map the pivot built?")
        fresh = core.points_from_scan(await sweep(client),
                                      forward_deg=config.LIDAR_FORWARD_DEG)
        fits = {}
        for label, grid in (("standstill", baseline_grid),
                            ("pivoted", mapper.grid)):
            field = core.LikelihoodField.from_grid(grid)
            at, got, sharp = core.match_pose(field, fresh, prior=(0.0, 0.0, 0.0),
                                             window_m=0.40, window_deg=45.0)
            fits[label] = got
            print(f"  vs the {label} map: score {got:.3f} "
                  f"(sharpness {sharp:.2f}, best at "
                  f"{math.degrees(at[2]):+.0f}deg)")
        ratio = fits["pivoted"] / max(fits["standstill"], 1e-6)
        good_fit = ratio > 0.5
        print(f"  {'PASS' if good_fit else 'FAIL'}  the pivoted map explains "
              f"a real sweep {ratio:.2f}x as well as the standstill one "
              f"(want over 0.50)")
        ok &= good_fit

        record = Path(config.ROOT) / "state" / "pivot_test"
        record.mkdir(parents=True, exist_ok=True)
        where = record / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
        where.write_text(json.dumps({
            "step_deg": STEP_DEG, "sweeps_each_step": SWEEP_EACH_STEP,
            "commanded_deg": commanded, "fits": fits,
            "updates": [{**u, "scan": {str(k): v for k, v in u["scan"].items()}}
                        for u in log],
        }))
        print(f"\n{len(log)} raw sweeps saved to {where.name} — "
              f"re-analysis needs no motors")

        mapper.grid.save()
        print(f"map saved — look at it with "
              f"./venv/bin/python scripts/map_here.py --show")
        print("all good" if ok else "something is off — do not drive yet")
        return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
