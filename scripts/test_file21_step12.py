#!/usr/bin/env python3
"""File 21 step 12: the sim, the H2 anchor check, the hardware checklist.

    ./venv/bin/python scripts/test_file21_step12.py

Network blocked; no robot; no sound. The sim runs a whole mission on each of
its four worlds (the three-room house, the uneven house with its saved map,
one closed room, a COPY of the real state/map.npz) with typed answers, and
everything it prints is checked. CLAUDE.md is checked to be UNCHANGED: the
PART 12 addendum is only a proposal until Harsh approves it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))

import netguard                                         # noqa: E402
netguard.block()

from exploration_mission import sim                     # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def args(**kw) -> argparse.Namespace:
    base = dict(map="three", say="explore the house", replies="", policy="matched",
                minutes=None, matched_within=None, fail_return=None, max_s=1800.0,
                keep=None, verbose=False)
    base.update(kw)
    return argparse.Namespace(**base)


def run(**kw):
    lines: list[str] = []
    res = sim.run(args(**kw), out=lines.append)
    return res, lines


SAVED_MAP = ROOT / "state" / "map.npz"
map_hash = hashlib.sha256(SAVED_MAP.read_bytes()).hexdigest() if SAVED_MAP.exists() else None
state_before = {p.name: p.stat().st_mtime for p in (ROOT / "state").glob("*")
                if p.is_file()}

# =============================================================================
print("--- the sim: the three-room house, answered in English and Hinglish ---")
res, lines = run(map="three", replies="hall,isko kitchen bolo,ye bedroom hai", keep=None)
text = "\n".join(lines)
check("she is started by the offline router ('explore the house')",
      'HARSH : "explore the house"' in text and "Starting exploration" in text)
check("three questions, three names saved",
      text.count("What should I call it?") == 3
      and all(f"Saved as {n}." in text for n in ("hall", "kitchen", "bedroom")), text[-800:])
check("the summary names every area", "hall, kitchen and bedroom" in text)
check("she comes back and says so",
      "I've reached where I started. Exploration saved." in text
      and res["truth_m"] < 0.35, f"{res['truth_m']:.2f} m")
check("the states run IDLE -> MISSION_INIT -> EXPLORE <-> ASK_LABEL -> "
      "RETURN_TO_START -> IDLE",
      res["states"][:3] == ["BOOT_RESTORE", "RELOCALIZE", "IDLE"]
      and "ASK_LABEL" in res["states"] and res["states"][-1] == "IDLE"
      and "RETURN_TO_START" in res["states"], str(res["states"]))
check("every line she says is printed (none to a speaker)",
      sum(1 for ln in lines if ln.startswith("  FRIDAY: ")) == len(res["said"])
      and len(res["said"]) >= 8)
check("the faces are printed too", "[face: curious]" in text and "[face: proud]" in text)
check("timings printed against the spec's budgets",
      "spec < 150 ms" in text and "spec < 300 ms" in text and "p95" in text)
check("the temp data is removed afterwards (no --keep)", not Path(res["tmp"]).exists())

print("--- silence and a non-name, typed ---")
res, lines = run(map="uneven", replies="kitchen,,not a name at all really,hall")
text = "\n".join(lines)
check("silence -> she asks once more", "Sorry, I didn't catch that." in text)
check("a sentence that is not a name -> saved unnamed, for later",
      "I'll save it as area 1 for now. You can rename it later." in text)
check("the uneven house's mission map FITS its saved map: areas move into "
      "frame 'map'", "FITTED the saved map" in text and "frame map" in text, text[-700:])

print("--- a COPY of the real saved map ---")
if map_hash is None:
    check("state/map.npz exists", False)
else:
    res, lines = run(map="real", replies="bedroom")
    text = "\n".join(lines)
    check("a whole mission on a copy of the real room", "Saved as bedroom." in text
          and res["states"][-1] == "IDLE" and res["truth_m"] < 0.35, text[-600:])
    check("...the end fit REFUSES the look-alike hostel room: nothing guessed",
          "did not take" in text and "frame mission:" in text, text[-600:])
    check("...and the real map was only COPIED, never written",
          hashlib.sha256(SAVED_MAP.read_bytes()).hexdigest() == map_hash)

print("--- a way back that fails: the help line with the true distance ---")
res, lines = run(map="closed", replies="study", fail_return=1.5)
text = "\n".join(lines)
check("HELP_WAIT, and she says how far she is",
      res["states"][-1] == "HELP_WAIT" and "I'm stuck about 1.1 meters" in text,
      text[-500:])

print("--- not a mission command ---")
res, lines = run(map="closed", say="what time is it")
check("'what time is it' is not the mission's: nothing starts",
      any("not a mission command" in ln for ln in lines) and res["mission"] is None
      and not Path(res["tmp"]).exists())

print("--- offline, and nothing of hers touched ---")
try:
    socket.create_connection(("1.1.1.1", 53), timeout=1)
    blocked = False
except Exception:                                       # noqa: BLE001
    blocked = True
check("the network is blocked for the whole run", blocked)
src = (ROOT / "exploration_mission" / "sim.py").read_text()
check("sim.py never imports a speaker, the brain or a network library",
      not any(w in src for w in ("import nav_voice", "import tts", "import brain",
                                 "import anthropic", "import httpx", "import requests")))
after = {p.name: p.stat().st_mtime for p in (ROOT / "state").glob("*") if p.is_file()}
touched = [n for n in after if n in state_before and after[n] != state_before[n]
           and n in ("map.npz", "memory.db", "places.json")]
check("state/map.npz and memory.db untouched by every sim run", touched == [], str(touched))
check("no state/exploration_mission folder appeared",
      not (ROOT / "state" / "exploration_mission").exists())

# =============================================================================
print("--- H2's anchor check (its sweep source faked) ---")
sys.path.insert(0, str(ROOT / "scripts"))
import file21_anchor_check as h2                        # noqa: E402
import synth                                            # noqa: E402
import make_maps as mm                                  # noqa: E402

room = mm.closed_room()
occ = room.grid.occupied()
pose = [(3.0, 2.5, 0.0)]


async def fake_sweeps(n):
    """What the real _sweeps returns: robot-frame sweeps, ray-cast here."""
    g = room.grid
    return [synth.grid_ray_cast(occ, g.origin, g.resolution, pose[0])] * n

h2._sweeps = fake_sweeps
f = Path("/tmp") / "friday-file21-step12-anchor.json"
out: list[str] = []
import builtins                                          # noqa: E402
real_print = builtins.print
builtins.print = lambda *a, **k: out.append(" ".join(str(x) for x in a))
try:
    h2.main(["capture", "--file", str(f)])
    pose[0] = (3.0, 2.5, 1.5708)                        # turned 90 degrees by hand
    h2.main(["compare", "--file", str(f)])
    pose[0] = (2.0, 2.5, 0.0)                           # pushed 1 m
    h2.main(["compare", "--file", str(f)])
finally:
    builtins.print = real_print
check("capture reports how many bearings see a wall", "captured:" in out[1], str(out))
turned = next((ln for ln in out if ln.startswith(("MATCH", "NO MATCH"))), "")
check("turned 90 deg in place -> MATCH, about +/-90", turned.startswith("MATCH")
      and ("+90" in turned or "-90" in turned or "+89" in turned or "-89" in turned
           or "+91" in turned or "-91" in turned), turned)
pushed = [ln for ln in out if ln.startswith(("MATCH", "NO MATCH"))][-1]
check("pushed 1 m -> NO MATCH", pushed.startswith("NO MATCH"), pushed)
f.unlink(missing_ok=True)

# =============================================================================
print("--- the checklist, and CLAUDE.md left alone ---")
doc = ROOT / "docs" / "file21_hardware_checklist.md"
body = doc.read_text() if doc.exists() else ""
check("docs/file21_hardware_checklist.md has H1-H7, each with a 'Pass when'",
      all(f"## H{i} " in body for i in range(1, 8)) and body.count("**Pass when:**") == 7)
check("H7 never switches Wi-Fi or the Pi's network off (internet cable / mobile "
      "data only)", "Never change any network setting on the Pi" in body
      and "Wi-Fi left ON" in body)
check("the CLAUDE.md PART 12 text is a PROPOSAL in the checklist",
      "proposed CLAUDE.md PART 12 (NOT appended" in body)
diff = subprocess.run(["git", "-C", str(ROOT), "diff", "--name-only", "HEAD", "--",
                       "CLAUDE.md", "rl", "slam", "exploration_mode.py",
                       "mcp_server/tools/explore.py", "offline.py"],
                      capture_output=True, text=True).stdout.split()
check("CLAUDE.md, rl/, slam/, exploration_mode.py, explore.py, offline.py "
      "untouched", diff == [], str(diff))
cfg = (ROOT / "config.py").read_text()
check("the mission and curiosity both still ship OFF",
      "EXPLORATION_MISSION_SERVICE = False" in cfg
      and '"curiosity": {"mode": "off"' in cfg)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
