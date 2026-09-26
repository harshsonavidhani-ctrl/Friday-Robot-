#!/usr/bin/env python3
"""Re-survey the room: standing-still maps at several spots. MOVES THE ROBOT.

    ./venv/bin/python scripts/survey_spots.py            # capture, then:
    ./venv/bin/python scripts/stitch_survey.py <folder>  # stitch offline

Harsh approved re-mapping on 2026-09-26 and chose this over a driving survey.
A map built WHILE driving smeared (a doubled diagonal wall: each move's small
matching error adds up); a map built STANDING STILL is crisp, like the
09-19 map. So at each spot a fresh Mapper at (0, 0, 0) folds STILL sweeps
without moving, and between spots she turns toward the most open direction
and drives, every move measured sweep-to-sweep (nav_follower.scan_motion).
Everything is written as it happens to state/map-permanent/surveys/<time>/
(spot_<k>.npz, moves.json) -- the dashboard's Maps panel shows it -- so the
stitching can be redone offline without driving again.

Every move goes through the guarded MCP tools with a fresh look first; the
turn tool takes LEFT AS NEGATIVE, the map frame left as positive.
"""
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pivot_test import session, sweep, text_of       # noqa: E402
import nav_follower as nf                            # noqa: E402
from slam.mapper import Mapper                       # noqa: E402

OUT = ROOT / "state" / "map-permanent" / "surveys" / time.strftime("%Y%m%d-%H%M%S")
SPOTS = 5
STILL = 25            # standing-still sweeps per spot
STEP_CM = 25
STEPS = 2
MIN_OPEN_MM = 700     # a leg sets off only toward this much clear floor
KEEP_OPEN_MM = 600    # and each further step needs this much still clear
# Robot-relative clearance names -> turn in the map frame (+ = left).
DIRECTIONS = {"forward": 0, "forward_left": 45, "forward_right": -45,
              "left": 90, "right": -90, "backward_left": 135,
              "backward_right": -135, "backward": 180}


async def look(c) -> dict:
    # scan_environment can throw "deque mutated during iteration" (a sensor
    # thread race, 2026-09-26): look again rather than end the survey.
    for _ in range(4):
        text = text_of(await c.call_tool("scan_environment", {}))
        try:
            return json.loads(text).get("clearance_mm", {})
        except ValueError:
            await asyncio.sleep(0.3)
    return {}


async def still_map(c, k: int) -> None:
    m = Mapper(pose=(0.0, 0.0, 0.0))
    for _ in range(STILL):
        s = await sweep(c)
        if s:
            m.update(s)
        await asyncio.sleep(0.12)
    m.grid.save(OUT / f"spot_{k}.npz")
    g = m.grid
    print(f"spot {k}: walls {int(g.occupied_mask().sum())} cells, "
          f"free {g.free_mask().sum() * g.cell ** 2:.1f} m2", flush=True)


async def measured(c, tool: str, args: dict, moved_m: float,
                   turned_rad: float) -> dict:
    await look(c)
    before = await sweep(c)
    said = text_of(await c.call_tool(tool, args))
    await asyncio.sleep(0.35)
    after = await sweep(c)
    est = (nf.scan_motion(before, after, moved_m, turned_rad)
           if before and after else None)
    rec = {"tool": tool, "args": args, "said": said[:120],
           "asked": [moved_m, turned_rad],
           "est": None if est is None else [est.forward_m, est.turn_rad,
                                            est.score]}
    print(f"   {tool} {args} -> {said[:50]} | est {rec['est']}", flush=True)
    return rec


async def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    moves = []
    t0 = time.monotonic()
    async with session() as c:
        await look(c)
        for k in range(SPOTS):
            await still_map(c, k)
            if k == SPOTS - 1:
                break
            leg = []
            cl = await look(c)
            best = max(DIRECTIONS, key=lambda n: cl.get(n) or 0)
            if (cl.get(best) or 0) < MIN_OPEN_MM:
                print(f"leg {k}: nowhere open enough {cl}")
                break
            left = DIRECTIONS[best]
            while abs(left) > 1:
                step = max(-35, min(35, left))
                leg.append(await measured(c, "turn_degrees", {"degrees": -step},
                                          0.0, math.radians(step)))
                left -= step
            for _ in range(STEPS):
                cl = await look(c)
                if (cl.get("forward") or 0) < KEEP_OPEN_MM:
                    break
                leg.append(await measured(c, "move_distance", {"cm": STEP_CM},
                                          STEP_CM / 100.0, 0.0))
            moves.append(leg)
            (OUT / "moves.json").write_text(json.dumps(moves))
    print(f"captured in {time.monotonic() - t0:.0f} s -> {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
