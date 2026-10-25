#!/usr/bin/env python3
"""Friday plays Hill Climb Racing on Harsh's laptop, run after run
(2026-10-25: "try another game like car racing game", "hill climb racing").

The laptop's helper drives (autopilot mode "balance": gas, easing off and
braking as the nose rears up, ~10 times a second). This loop, every LOOK_S,
looks at the game: the jeep in view -> keep the driver's lease; otherwise it
works out which screen is up (scripts/hcr_screens.py) and gets to the next run
-- closing "second chance" / "bonus reward" ad offers with their X (it never
watches an ad), touching "continue", START, Normal mode START. Anything else
(a video ad, loading) it waits out. Pictures of each new screen go to
state/game_drive/. Switches the driver off at the end and on Ctrl-C.

    ./venv/bin/python scripts/hcr_play.py --minutes 5
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "laptop"))
sys.path.insert(0, str(ROOT / "scripts"))

import friday_laptop_agent as fa                          # noqa: E402
import hcr_screens as hs                                  # noqa: E402
import laptop_games as lg                                 # noqa: E402
import laptop_research as lr                              # noqa: E402

GAME = "Hill Climb Racing"
OUT = ROOT / "state" / "game_drive"
LOOK_S = 1.0
AD_WAIT_S, AD_BACK_EVERY_S = 30.0, 15.0     # an unknown screen this long: Android "back"


def jeep_tilt(png: bytes) -> float | None:
    rgb = np.array(Image.open(io.BytesIO(png)).convert("RGB").resize((480, 270)))
    bgra = np.dstack([rgb[..., ::-1], np.full(rgb.shape[:2], 255, np.uint8)]).tobytes()
    return fa.head_tilt(480, 270, bgra)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=5.0)
    ap.add_argument("--tune", default="", help='JSON, e.g. {"tilt_back": 35}')
    args = ap.parse_args()
    auto = lg.autopilot_for(GAME) or {"mode": "balance", "keys": {"gas": "right", "brake": "left"}}
    if args.tune:
        auto["tune"] = {**auto.get("tune", {}), **json.loads(args.tune)}
    OUT.mkdir(parents=True, exist_ok=True)
    tag = time.strftime("hcr-%Y%m%d-%H%M%S")
    log = (OUT / f"{tag}.jsonl").open("w")
    if not fa.title_matches((lr.idle() or {}).get("title", ""), GAME):
        lr.launch(GAME)                               # to the front (he may have clicked his terminal);
        time.sleep(1.5)                               # when it is already there this would restart it
    t0, last_screen, runs, shots = time.monotonic(), None, 0, 0
    other_since, last_back = None, 0.0
    try:
        while time.monotonic() - t0 < args.minutes * 60:
            png, why = lr.screen(GAME)
            t = round(time.monotonic() - t0, 1)
            if not png:
                print(f"{t:6.1f}s no picture ({why}) -- is the game in front?", flush=True)
                if why and "front" in why and "hill" not in str((lr.idle() or {}).get("title", "")).lower():
                    lr.launch(GAME)
                time.sleep(2.0)
                continue
            screen = hs.which(png)
            tilt = None
            if screen not in ("second_chance", "bonus") and hs.pedals(png):
                screen, tilt = "driving", jeep_tilt(png)     # a menu's jeep PICTURE is not driving
            if screen != last_screen:
                shots += 1
                (OUT / f"{tag}_{shots:03d}_{screen}.png").write_bytes(png)
                if screen == "driving":
                    runs += 1
                print(f"{t:6.1f}s {screen}" + (f" (run {runs})" if screen == "driving" else ""), flush=True)
                last_screen = screen
            row = {"t": t, "screen": screen, "tilt": None if tilt is None else round(tilt)}
            if screen == "driving":
                got, why = lr.autopilot(GAME, **auto, for_s=30.0)
                row["last"] = (got or {}).get("last")
                if got is None:
                    row["why"] = why
            elif screen == "other":
                # a full-screen video ad (2026-10-26: Clash of Clans, 4 minutes): never
                # click inside it; after AD_WAIT_S press Escape = Android back
                now = time.monotonic()
                other_since = other_since or now
                if now - other_since > AD_WAIT_S and now - last_back > AD_BACK_EVERY_S:
                    lr.step(GAME, ["escape"], 100, shot=False)
                    last_back = now
                    row["back"] = True
            elif screen in hs.CLICK:
                lr.autopilot(GAME, "off")
                x, y = hs.CLICK[screen]
                lr.click(GAME, x, y)
                row["click"] = screen
                time.sleep(1.5)
            if screen != "other":
                other_since = None
            log.write(json.dumps(row) + "\n")
            log.flush()
            time.sleep(LOOK_S)
    except KeyboardInterrupt:
        print("stopped by hand", flush=True)
    finally:
        lr.autopilot(GAME, "off")
        log.close()
        print(f"driver off. {runs} runs; pictures and log: {OUT / tag}*", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
