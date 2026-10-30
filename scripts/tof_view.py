"""What the bumper ToF really sees, frame by frame. Read-only: no motors.

    ./venv/bin/python scripts/tof_view.py [seconds] [log.jsonl]

Prints the 8x8 raw grid (mm) and which cells the reflex would count as an obstacle,
through the same tof_obstacles_mm() and k-th-smallest rule the run uses. Needs the
ToF port free: nothing else may be reading it (a run, or friday-mcp's sensors).
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT))

import numpy as np                                          # noqa: E402
import serial                                               # noqa: E402

import config                                               # noqa: E402
from pi_deploy import (RobotConfig, TOF_COLS, TOF_ROWS,     # noqa: E402
                       tof_obstacles_mm)

seconds = float(sys.argv[1]) if len(sys.argv) > 1 else 10.0
log = open(sys.argv[2], "a") if len(sys.argv) > 2 else None
cfg = RobotConfig()
port = next(iter(sorted(Path("/dev/serial/by-id").glob("*Pico*"))))
ser = serial.Serial(str(port), 115200, timeout=0.5)
ser.reset_input_buffer()
partial, frames, last = {}, 0, 0.0
end = time.monotonic() + seconds
buf = bytearray()
while time.monotonic() < end:
    buf.extend(ser.read(ser.in_waiting or 1))
    while b"\n" in buf:
        cut = buf.find(b"\n")
        raw, buf = bytes(buf[:cut]), buf[cut + 1:]
        line = raw.decode(errors="replace").strip()
        if not line.startswith("y") or ":" not in line:
            continue
        head, _, body = line.partition(":")
        try:
            row = int(head[1:])
        except ValueError:
            continue
        vals = [int(v) for v in body.split(",") if v.strip().isdigit()]
        if len(vals) != TOF_COLS or not 0 <= row < TOF_ROWS:
            continue
        partial[row] = vals
        if len(partial) < TOF_ROWS:
            continue
        frame = [partial[r] for r in range(TOF_ROWS)]
        partial = {}
        frames += 1
        obstacles, intact = tof_obstacles_mm(frame)
        flat = np.sort(obstacles.reshape(-1))
        kth = float(flat[min(cfg.tof_min_valid_zones, flat.size) - 1])
        if log:
            log.write(json.dumps({"t": time.time(), "frame": frame, "kth": kth,
                                  "intact": bool(intact)}) + "\n")
        if time.monotonic() - last >= 1.0:
            last = time.monotonic()
            print(f"\nframe {frames}  reflex reading {kth:.0f} mm "
                  f"({'STOP' if kth <= cfg.tof_stop_distance * 1000 else 'clear'})  "
                  f"floor intact {intact}  raw (* = counted as an obstacle)")
            for r in range(TOF_ROWS):
                print(" ".join(f"{frame[r][c]:5d}{'*' if obstacles[r, c] < 4000 else ' '}"
                               for c in range(TOF_COLS)), flush=True)
print(f"\n{frames} frames in {seconds:.0f} s")
