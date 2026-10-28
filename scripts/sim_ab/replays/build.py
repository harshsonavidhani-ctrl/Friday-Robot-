"""Build the replay page from the sim_ab results that recorded a path.

Published (pinned in Harsh's claude.ai sidebar): https://claude.ai/artifact/Puo1QhtdkU5bm9kaAsTZiJ
Rebuild: python3 scripts/sim_ab/replays/build.py, then republish friday_sim_replays.html to that URL.
"""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import simworlds as sw

OUT = HERE.parents[2] / "state" / "sim_ab" / "out"
ROOMS = [("two_rooms", "Two rooms, 0.8 m door"), ("corridor", "L corridor, 1.4 m"),
         ("chairs", "Dining room, chair legs"), ("low_things", "Low obstacles"),
         ("popup", "Open room, person steps in")]
RUNS = [("rep1", "act", "ACT (today)", "act", "run 1"), ("rep1", "ppo_ft", "Round 3", "r3", "run 1"),
        ("p3", "act", "ACT (today)", "act", "run 2"), ("p3", "ft3", "Round 3", "r3", "run 2"),
        ("rep2", "act", "ACT (today)", "act", "run 2"), ("rep2", "ppo_ft", "Round 3", "r3", "run 2"),
        ("rep3", "act", "ACT (today)", "act", "run 3"), ("rep3", "ppo_ft", "Round 3", "r3", "run 3"),
        ("b1", "ppo_ft", "Round 3, continued", "r3b", "run 1"),
        ("b2", "ppo_ft", "Round 3, continued", "r3b", "run 2"),
        ("b3", "ppo_ft", "Round 3, continued", "r3b", "run 3"),
        ("m4", "ppo_ft", "Round 4", "r4", "run 1"),
        ("r5a", "ppo_ft", "Round 5 (door practice)", "r5", "run 1"),
        ("r5b", "ppo_ft", "Round 5 (door practice)", "r5", "run 2"),
        ("r5c", "ppo_ft", "Round 5 (door practice)", "r5", "run 3")]
# the averages also count m3 (the first masked run of each, recorded before paths were)
AVG = [("act", "ACT (today)", ["m3_act", "rep1_act", "rep2_act", "rep3_act"]),
       ("r3", "Round 3", ["m3_ppo_ft", "rep1_ppo_ft", "rep2_ppo_ft", "rep3_ppo_ft"]),
       ("r3b", "Round 3, continued", ["b1_ppo_ft", "b2_ppo_ft", "b3_ppo_ft"]),
       ("r5", "Round 5 (door practice)", ["r5a_ppo_ft", "r5b_ppo_ft", "r5c_ppo_ft"])]
data = {"rooms": []}
for key, title in ROOMS:
    walls, circles, low_w, low_c, popup = sw.build(key)
    room = {"key": key, "title": title, "walls": walls, "circles": circles, "low_walls": low_w,
            "low_circles": low_c, "runs": []}
    for tag, code, label, model, nth in RUNS:
        f = OUT / f"{tag}_{code}_{key}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if len(d.get("path", [])) < 110:          # an unfinished run
            continue
        room["runs"].append({"label": label, "model": model, "nth": nth if tag != "p3" else "extra run",
                             "path": d["path"], "seen": d["seen_m2"], "driven": d["driven_m"],
                             "bumps": d["bumps"], "still": d["still_pct"], "reverse": d["reverse_pct"]})
    room["avg"] = []
    for model, label, tags in AVG:
        ds = [json.loads((OUT / f"{t}_{key}.json").read_text()) for t in tags if (OUT / f"{t}_{key}.json").exists()]
        if not ds:
            continue
        mean = lambda k: round(sum(d[k] for d in ds) / len(ds), 1)
        room["avg"].append({"model": model, "label": label, "n": len(ds), "seen": mean("seen_m2"),
                            "driven": mean("driven_m"), "bumps": mean("bumps"), "still": mean("still_pct"),
                            "range": [min(d["seen_m2"] for d in ds), max(d["seen_m2"] for d in ds)]})
    data["rooms"].append(room)
html = (HERE / "template.html").read_text()
(HERE / "friday_sim_replays.html").write_text(html.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
print({r["key"]: len(r["runs"]) for r in data["rooms"]})
