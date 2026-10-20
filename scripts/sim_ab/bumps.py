"""Where on her body the sim bumps landed, and what she was doing.

    python scripts/sim_ab/bumps.py b1:0904 b1:today b2:careful
"""
import collections
import glob
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "state" / "sim_ab" / "out"
for spec in sys.argv[1:] or ["b1:0904", "b1:today", "b2:careful"]:
    tag, code = spec.split(":")
    zones, doing, n, files = collections.Counter(), collections.Counter(), 0, 0
    still, driven = [], 0.0
    for f in sorted(glob.glob(str(OUT / f"{tag}_{code}_*.json"))):
        s = json.loads(Path(f).read_text())
        files += 1
        still.append(s.get("still_pct", 0))
        driven += s.get("driven_m", 0)
        for c in s.get("contacts", []):
            b = abs(c["bearing"])
            n += 1
            zones["front" if b <= 45 else "side" if b <= 135 else "rear"] += 1
            doing["reversing" if c["v"] < -0.02 else
                  "forward" if c["v"] > 0.02 else "turning/still"] += 1
    if files:
        print(f"{spec:12} rooms {files}  bumps {n:3}  driven {driven:5.1f} m  "
              f"still {min(still):.0f}-{max(still):.0f}%  where {dict(zones)}  doing {dict(doing)}")
    else:
        print(f"{spec:12} (no results)")
