"""Build a map of the room Friday is standing in, and look at it.

    ./venv/bin/python scripts/map_here.py            # check the sensor only
    ./venv/bin/python scripts/map_here.py --build    # take sweeps, build a map
    ./venv/bin/python scripts/map_here.py --show     # draw the saved map

⚠ THIS DOES NOT MOVE THE ROBOT and cannot: it calls read-only tools only.
Run it with Friday sitting still, anywhere, even held.

⚠ IT TALKS TO THE RUNNING MCP SERVER OVER HTTP RATHER THAN OPENING THE LiDAR.
That is not politeness, it is the single-doorway rule: friday-mcp.service
already holds /dev/serial/by-id/...CP2102... and two processes on one serial
port produces garbage in both, intermittently, in a way that looks like a
sensor fault for a week. If the service is not running, start it -- do not
work around it by opening the port here.

WHAT THIS IS FOR. Everything in slam/ has been tested against a ray-cast
simulation and has never seen a real room. Simulated walls are perfectly flat,
perfectly opaque, and exactly where the geometry says. Real rooms have glass,
gloss, chair legs, and skirting boards. This is the first contact, and it is
deliberately the version with no motors in it: if the map of a room Friday is
STANDING STILL in does not look like the room, nothing built on top of it is
worth testing.

A good result is a recognisable outline of this room with a shadow behind
every solid thing -- NOT a floor plan. One standing position can only ever map
what is in line of sight.
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
from slam.grid import OccupancyGrid                              # noqa: E402


@asynccontextmanager
async def session():
    async with streamable_http_client(config.MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            yield client


def payload(result) -> dict:
    """The JSON a tool returned, or an explanation of what came back instead."""
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return {"raw": text}
    return {"error": "the tool returned nothing"}


def check_sweep(data: dict) -> bool:
    """Is this sensor worth building a map from? Prints why, either way."""
    if data.get("error") or not data.get("beams"):
        print(f"  the LiDAR is not answering: {data.get('error', data)}")
        return False

    beams = data["beams"]
    gap = data["widest_gap_deg"]
    print(f"  {beams} bearings answered, "
          f"nearest {data['nearest_mm']} mm, furthest {data['furthest_mm']} mm")
    print(f"  widest gap in coverage: {gap}deg")

    ok = True
    if beams < 150:
        print(f"  ⚠ only {beams} bearings — expected 250-350 in a normal room")
        ok = False
    if gap >= 90:
        print(f"  ⚠ a {gap}deg hole in the sweep. That is a whole quadrant "
              f"returning nothing, which is a fault (a loose cable, or "
              f"something resting on the dome) rather than a quiet room")
        ok = False
    if data["nearest_mm"] < 200:
        print(f"  ⚠ nearest return is {data['nearest_mm']} mm, inside the "
              f"chassis radius — the self-mask may need re-measuring")
        ok = False
    if ok:
        print("  the sensor looks healthy")
    return ok


def draw(path: Path | None = None) -> int:
    try:
        grid = OccupancyGrid.load(path)
    except FileNotFoundError:
        print(f"no map at {path or config.SLAM_MAP_FILE} — run with --build")
        return 1
    stats = grid.stats()
    floor = grid.free_mask().sum() * grid.cell ** 2
    print(f"\n{stats['shape'][0]}x{stats['shape'][1]} cells at {grid.cell} m, "
          f"{floor:.1f} m2 of floor, {stats['occupied']} wall cells, "
          f"{stats['scans']} sweeps")
    print(f"extent {stats['bounds_m']} metres\n")
    print(grid.to_ascii())
    print("\n'#' wall   '.' floor   ' ' never seen")
    print("Friday is at the middle of the picture, facing UP.")
    return 0


async def main() -> int:
    build = "--build" in sys.argv
    show = "--show" in sys.argv
    if show and not build:
        return draw()

    print(f"connecting to the MCP server at {config.MCP_HTTP_URL}")
    try:
        async with session() as client:
            print("\nchecking the sensor before trusting it")
            data = payload(await client.call_tool("lidar_sweep", {}))
            healthy = check_sweep(data)
            if not build:
                print("\n(sensor check only — pass --build to make a map)")
                return 0 if healthy else 1
            if not healthy:
                print("\nnot building a map from a sensor that looks wrong. "
                      "Fix that first.")
                return 1

            sweeps = 30
            print(f"\ntaking {sweeps} sweeps — hold still, this is ~4 seconds")
            built = payload(await client.call_tool("map_here",
                                                   {"sweeps": sweeps}))
            if built.get("error"):
                print(f"  {built['error']}")
                return 1
            print(f"  {built['sweeps_used']} sweeps used, "
                  f"{built['sweeps_empty']} empty, {built['seconds']}s")
            print(f"  {built['floor_m2']} m2 of floor, "
                  f"{built['occupied_cells']} wall cells")
            if not built["looks_like_a_room"]:
                print("  ⚠ this does not look like a room. Do not build on it.")
                return 1
            print(f"  saved to {built['saved_to']}")
    except Exception as exc:                       # noqa: BLE001
        print(f"\ncould not talk to the MCP server: {exc}")
        print("Is friday-mcp running? Do NOT open the LiDAR from here instead "
              "— see the note at the top of this file.")
        return 1

    return draw() if show else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
