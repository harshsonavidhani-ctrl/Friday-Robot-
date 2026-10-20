"""Table of A/B sim results: python summarise.py <tag> [code ...]"""
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "state" / "sim_ab" / "out"
TAG = sys.argv[1] if len(sys.argv) > 1 else "r1"
CODES = sys.argv[2:] or ["0904", "today"]
WORLDS = ["corridor", "chairs", "two_rooms", "low_things", "popup"]


def loop_ms(log):
    try:
        txt = log.read_text(errors="replace")
    except FileNotFoundError:
        return None, None
    m = re.findall(r"loop ms[^:]*: mean ([\d.]+) \| p95 ([\d.]+)", txt)
    return (float(m[-1][0]), float(m[-1][1])) if m else (None, None)


hdr = ("room", "code", "bumps", "touch s", "closest m", "still %", "hesitate %",
       "longest stop s", "driven m", "top v", "seen m2", "reverse %", "loop ms", "p95 ms")
print(" | ".join(hdr))
pops = []
for w in WORLDS:
    for v in CODES:
        p = OUT / f"{TAG}_{v}_{w}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        lm, lp = loop_ms(OUT / f"{TAG}_{v}_{w}.log")
        print(" | ".join(str(x) for x in (
            w, v, s["bumps"], s["contact_s"], s["min_clearance"], s["still_pct"],
            s["hesitate_pct"], s["longest_still_s"], s["driven_m"], s.get("v_max_seen"),
            s["seen_m2"], s["reverse_pct"], lm, lp)))
        if s.get("popups"):
            pops.append((v, s["popups"]))
for v, ps in pops:
    print(f"popup {v}: " + "; ".join(
        f"at {p['at_s']}s v={p['v_at']} reacted {p['react_s']}s closest {p['min_gap_m']} m"
        for p in ps))
