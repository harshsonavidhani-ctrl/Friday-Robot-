#!/usr/bin/env python3
"""Watch the laptop drive GTA (2026-10-25, "try the gta driving now").

Starts the helper's autopilot with GTA's config (config.GAME_AUTOPILOT, mode
"mix": the minimap steers, the car network works the pedals), renews its lease
every RENEW_S, prints what each renewal reports and saves a game picture every
SHOT_S to state/gta_drive/. Always switches the driver off at the end, also on
Ctrl-C. The helper's own rules still hold: game keys only, only while GTA is
in front, everything let go when it stops.

    ./venv/bin/python scripts/gta_drive.py --seconds 60
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                             # noqa: E402
import laptop_games as lg                                 # noqa: E402
import laptop_research as lr                              # noqa: E402

GAME = "Grand Theft Auto V Enhanced"
OUT = ROOT / "state" / "gta_drive"
RENEW_S, SHOT_S = 2.0, 10.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--mode", default="")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    auto = lg.autopilot_for(GAME) or {"mode": "road"}
    if args.mode:
        auto["mode"] = args.mode
    OUT.mkdir(parents=True, exist_ok=True)
    run = time.strftime("%Y%m%d-%H%M%S")
    log = (OUT / f"{run}.jsonl").open("w")
    print(f"driving {GAME} in {auto['mode']} mode for {args.seconds:.0f} s", flush=True)
    t0 = next_shot = time.monotonic()
    shots = 0
    try:
        while time.monotonic() - t0 < args.seconds:
            got, why = lr.autopilot(GAME, **auto, speed=args.speed, for_s=30.0)
            now = time.monotonic()
            row = {"t": round(now - t0, 1), "why": why, **(got or {})}
            log.write(json.dumps(row) + "\n")
            log.flush()
            last = (got or {}).get("last", {})
            print(f"{row['t']:6.1f}s  mode {(got or {}).get('mode')}  ticks {(got or {}).get('ticks')}  "
                  f"stucks {(got or {}).get('stucks')}  {json.dumps(last)[:150]}  {why or (got or {}).get('why', '')}",
                  flush=True)
            if got is None or (not got.get("running") and got.get("why")):
                print("the driver stopped:", why or got.get("why"), flush=True)
                break
            if now >= next_shot:
                png, _ = lr.screen(GAME)
                if png:
                    shots += 1
                    (OUT / f"{run}_{shots:02d}.png").write_bytes(png)
                next_shot = now + SHOT_S
            time.sleep(RENEW_S)
    except KeyboardInterrupt:
        print("stopped by hand", flush=True)
    finally:
        lr.autopilot(GAME, "off")
        log.close()
        print(f"driver off. log {OUT / (run + '.jsonl')}, {shots} pictures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
