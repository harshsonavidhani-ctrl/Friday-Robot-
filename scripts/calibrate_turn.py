"""What is the smallest turn this chassis reliably makes?

    ./venv/bin/python scripts/calibrate_turn.py          # dry run
    ./venv/bin/python scripts/calibrate_turn.py --turn   # MOVES THE ROBOT

⚠ MOVES THE ROBOT (pivots only, never translates). Ask Harsh first.

WHY. A commanded pivot on this chassis sometimes does not happen: the motors
energise, buzz, and never break traction. Odometry reports it anyway, so the
robot then plans from a heading that is a lie. The first autonomous run caught
exactly this -- a -128 degree turn moved the world 912 mm, and a -28 degree
turn moved it 0 mm.

That difference is the clue. At 92 deg/s those are a 1.39 s pulse and a 0.30 s
pulse, and MOTOR_KICKSTART_MS is 130 ms of either. A long turn spends most of
its time already moving; a short one can end before it ever starts.

So this measures the thing rather than guessing at it: each angle is commanded
and the sweep before is compared with the sweep after. A turn that happened
moves the world by hundreds of millimetres. A turn that did not leaves it at
sensor noise. Every angle is driven twice, once each way, so the robot ends
where it started and a one-off scuff on the carpet cannot pass as a result.

The number that comes out is the smallest angle that is reliable, which is
what MISSION_MIN_TURN_DEG and the planner should be built around.
"""

from __future__ import annotations

import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession                                    # noqa: E402
from mcp.client.streamable_http import streamable_http_client    # noqa: E402

import config                                                    # noqa: E402
from slam.mission import scan_shift_mm                           # noqa: E402

ANGLES = (10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 60.0)


@asynccontextmanager
async def session():
    async with streamable_http_client(config.MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            yield client


def text_of(result) -> str:
    for block in result.content:
        if getattr(block, "text", None):
            return block.text
    return ""


async def sweep(client) -> dict[int, int]:
    data = json.loads(text_of(await client.call_tool("lidar_sweep",
                                                     {"full": True})))
    return {int(b): int(mm) for b, mm in data.get("bearings_mm", {}).items()}


async def one(client, degrees: float) -> float | None:
    await client.call_tool("scan_environment", {})
    before = await sweep(client)
    said = text_of(await client.call_tool("turn_degrees", {"degrees": degrees}))
    if "REFUSED" in said.upper():
        print(f"      {degrees:+6.1f} deg REFUSED: {said[:70]}")
        return None
    await asyncio.sleep(config.LIDAR_SWEEP_MAX_AGE_S)
    return scan_shift_mm(before, await sweep(client))


async def main() -> int:
    if "--turn" not in sys.argv:
        print("dry run — pass --turn to actually pivot")
        print(f"would test {ANGLES} degrees, each way, "
              f"{len(ANGLES) * 2} pivots ending where it started")
        return 0

    async with session() as client:
        print(f"{'angle':>7s} {'right':>10s} {'left':>10s}   verdict")
        results = {}
        for angle in ANGLES:
            right = await one(client, angle)
            left = await one(client, -angle)
            pair = [x for x in (right, left) if x is not None]
            ok = bool(pair) and all(
                x >= config.PIVOT_MIN_SCAN_SHIFT_MM for x in pair)
            results[angle] = ok
            print(f"{angle:7.0f} {right if right is None else f'{right:10.0f}'}"
                  f" {left if left is None else f'{left:10.0f}'}"
                  f"   {'moved' if ok else 'STALLED'}")

        reliable = [a for a in ANGLES if results.get(a)]
        smallest = min(reliable) if reliable else None
        print()
        if smallest is None:
            print("no angle turned reliably — this is not a tuning problem")
            return 1
        print(f"smallest reliable turn: {smallest:.0f} degrees")
        print(f"config.MISSION_MIN_TURN_DEG is "
              f"{config.MISSION_MIN_TURN_DEG:.0f}")
        if smallest > config.MISSION_MIN_TURN_DEG:
            print(f"  -> raise MISSION_MIN_TURN_DEG to {smallest:.0f}: below "
                  f"that she cannot be trusted to turn at all")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
