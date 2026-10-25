#!/usr/bin/env python3
"""Keep the laptop's autopilot driving a game from the Pi (2026-10-25: Harsh
watching Hill Climb Racing, "its slow processing to decide"). The helper does
the fast part on the laptop; this only renews the lease every RENEW_S, prints
its report, saves a picture every SHOT_S to state/game_drive/, and switches
it off at the end (also on Ctrl-C).

    ./venv/bin/python scripts/game_drive.py "Hill Climb Racing" --seconds 90 --mode cruise --hold right
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laptop_games as lg                                 # noqa: E402
import laptop_research as lr                              # noqa: E402

OUT = ROOT / "state" / "game_drive"
RENEW_S, SHOT_S = 2.0, 8.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("game")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--mode", default="")
    ap.add_argument("--hold", nargs="*", default=None)
    ap.add_argument("--tune", default="", help='JSON, e.g. {"tilt_back": 35}')
    args = ap.parse_args()
    auto = lg.autopilot_for(args.game) or {"mode": "cruise", "hold": ["right"]}
    if args.mode:
        auto["mode"] = args.mode
    if args.hold is not None:
        auto["hold"] = args.hold
    if args.tune:
        auto["tune"] = {**auto.get("tune", {}), **json.loads(args.tune)}
    OUT.mkdir(parents=True, exist_ok=True)
    run = time.strftime("%Y%m%d-%H%M%S")
    log = (OUT / f"{run}.jsonl").open("w")
    print(f"driving {args.game}: {json.dumps(auto)} for {args.seconds:.0f} s", flush=True)
    t0 = next_shot = time.monotonic()
    shots = 0
    try:
        while time.monotonic() - t0 < args.seconds:
            got, why = lr.autopilot(args.game, **auto, for_s=30.0)
            now = time.monotonic()
            row = {"t": round(now - t0, 1), "why": why, **(got or {})}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"{row['t']:6.1f}s ticks {(got or {}).get('ticks')} "
                  f"{json.dumps((got or {}).get('last', {}))[:140]} {why}", flush=True)
            if got is None or (not got.get("running") and got.get("why")):
                print("the driver stopped:", why or got.get("why"), flush=True)
                break
            if now >= next_shot:
                png, _ = lr.screen(args.game)
                if png:
                    shots += 1
                    (OUT / f"{run}_{shots:02d}.png").write_bytes(png)
                next_shot = now + SHOT_S
            time.sleep(RENEW_S)
    except KeyboardInterrupt:
        print("stopped by hand", flush=True)
    finally:
        lr.autopilot(args.game, "off")
        log.close()
        print(f"driver off. {OUT / run}*", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
