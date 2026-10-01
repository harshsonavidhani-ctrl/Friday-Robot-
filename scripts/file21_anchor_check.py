#!/usr/bin/env python3
"""File 21 hardware check H2: does the home anchor's scan signature work here?

    ./venv/bin/python scripts/file21_anchor_check.py capture    # at the home spot
    ./venv/bin/python scripts/file21_anchor_check.py compare    # after turning / moving her

READ-ONLY AND IT NEVER MOVES HER. It reads the LiDAR through friday-mcp's own
`lidar_sweep` tool (friday-mcp owns the port; nothing else may open it),
builds the mission's 360-bin signature from 5 sweeps, and judges a match
with the mission's own numbers (config.EXPLORATION_MISSION["signature"]) and
the same function the mission uses at arrival (rth_ladder.verify).

What H2 expects:
  * turned ~90 degrees by hand, same spot  -> MATCH, turned ~ +/-90
  * pushed ~1 m away                        -> NO MATCH
The captured signature goes to a file in /tmp (never the mission's data).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import home_anchor, rth_ladder  # noqa: E402

DEFAULT_FILE = Path("/tmp/friday-anchor-check.json")
SWEEPS = 5


async def _sweeps(n: int) -> list[dict]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    out = []
    async with streamable_http_client(config.MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            for _ in range(n):
                res = await client.call_tool("lidar_sweep", {"full": True})
                text = next((getattr(b, "text", "") for b in res.content
                             if getattr(b, "text", "")), "")
                raw = json.loads(text).get("bearings_mm", {})
                out.append(home_anchor.raw_to_robot(
                    {int(b): int(mm) for b, mm in raw.items()},
                    config.LIDAR_FORWARD_DEG))
                await asyncio.sleep(0.3)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="H2: the home anchor's signature.")
    ap.add_argument("action", choices=("capture", "compare"))
    ap.add_argument("--file", default=str(DEFAULT_FILE))
    args = ap.parse_args(argv)
    sig = mcfg.load().signature
    path = Path(args.file)
    try:
        sweeps = asyncio.run(_sweeps(SWEEPS))
    except Exception as exc:                            # noqa: BLE001
        print(f"could not read the LiDAR through friday-mcp: {type(exc).__name__}: "
              f"{exc}\n(is friday-mcp up, and is no exploration run holding the LiDAR?)")
        return 2
    bearings = [len(s) for s in sweeps]
    print(f"{len(sweeps)} sweeps, {min(bearings)}-{max(bearings)} bearings each")
    if args.action == "capture":
        ref = home_anchor.signature(sweeps, sig.max_range_m)
        seen = int(sum(1 for v in ref if v == v))
        path.write_text(json.dumps({"at": time.time(), "signature": [
            None if v != v else round(float(v), 4) for v in ref]}))
        print(f"captured: {seen}/360 bearings see a wall within {sig.max_range_m:g} m "
              f"(the mission needs {int(sig.min_overlap * 360)} to match) -> {path}")
        if seen < sig.min_overlap * 360:
            print("⚠ too open here: a start at this spot could never be recognised")
        return 0
    if not path.exists():
        print(f"no capture at {path}: run 'capture' at the home spot first")
        return 2
    ref = json.loads(path.read_text())["signature"]
    v = rth_ladder.verify(sweeps, ref, sig.max_range_m, sig.match_mad_m,
                          sig.min_overlap)
    print(f"{'MATCH' if v['verified'] else 'NO MATCH'}: she turned {v['shift_deg']:+.0f} "
          f"deg (CCW +) · walls differ by {v['mad_m'] * 100:.0f} cm on average "
          f"(match <= {sig.match_mad_m * 100:.0f} cm) · overlap {v['overlap'] * 100:.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
