#!/usr/bin/env python3
"""File 21 step 1: config, grid_utils, adapters, the RL seam, offline.

    ./venv/bin/python scripts/test_file21_step1.py

No motors, no LiDAR, no speech, no money, and THE NETWORK IS BLOCKED for the
whole run (tests/exploration_mission/netguard.py). Hold and live files go to a
temp dir. The pi_deploy hold is exercised on an in-process ControlLoop with
the NullMotorDriver, the way rl/tests build one. The probe runs for real
against the live state, read-only, and is checked to have written nothing.
"""

from __future__ import annotations

import json
import logging
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "exploration_mission"))
sys.path.insert(0, str(ROOT / "rl"))

import netguard                                         # noqa: E402
netguard.block()

import config                                           # noqa: E402
from exploration_mission import adapters as ad          # noqa: E402
from exploration_mission import config as mcfg          # noqa: E402
from exploration_mission import grid_utils as gu        # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-file21-s1-"))
PY = sys.executable
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:                                   # noqa: BLE001
        return False
    return False


# =============================================================================
print("--- offline (hard rule 6) ---")

FORBIDDEN = ("anthropic", "openai", "requests", "httpx", "aiohttp",
             "urllib.request", "http.client", "websocket", "http://", "https://",
             "api_key", "import brain", "from brain", "brain_router")
hits = []
for path in sorted((ROOT / "exploration_mission").rglob("*.py")):
    text = path.read_text().lower()
    for word in FORBIDDEN:
        if word in text:
            hits.append(f"{path.name}: {word}")
check("no network, cloud-model, key or brain token anywhere in exploration_mission/",
      not hits, "; ".join(hits))

check("the guard blocks a connection to the internet",
      raises(lambda: socket.create_connection(("203.0.113.7", 80), timeout=0.2),
             netguard.NetworkBlocked))
check("the guard blocks a name that would need DNS",
      raises(lambda: socket.create_connection(("example.com", 443), timeout=0.2),
             netguard.NetworkBlocked))
srv = socket.socket()
srv.bind(("127.0.0.1", 0))
srv.listen(1)
cli = socket.socket()
try:
    cli.connect(srv.getsockname())
    loop_ok = True
except OSError:
    loop_ok = False
cli.close()
srv.close()
check("the guard still allows loopback", loop_ok)

BLOCKED_IMPORT = r'''
import importlib.abc, os, sys
BLOCK = {"brain", "brain_router", "anthropic", "openai", "httpx", "requests",
         "aiohttp", "websocket", "websockets"}
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCK:
            raise ImportError(f"BLOCKED IMPORT: {name}")
        return None
sys.meta_path.insert(0, Refuse())
sys.path[:0] = [ROOT, ROOT + "/tests/exploration_mission"]
import netguard; netguard.block()
assert "ANTHROPIC_API_KEY" not in os.environ
import exploration_mission
from exploration_mission import adapters as ad, config as mcfg, grid_utils as gu
cfg = mcfg.load()
fake = ad.fake_adapters()
run = ad.RunAdapter(state_dir=TMP)
run.hold("test", 2.0); assert run.held(); run.release(); assert run.held() is None
assert run.live() is None
assert ad.SlamAdapter(map_file=TMP + "/none.npz").grid() is None
assert len(ad.FaceAdapter().names()) == 20
ad.EnvAdapter().env_id(); ad.DetectorAdapter().status(); ad.MemoryAdapter().backend()
bad = sorted(n for n in sys.modules if n.split(".")[0] in BLOCK)
print("LOADED:" + ",".join(bad))
'''.replace("ROOT", repr(str(ROOT))).replace("TMP", repr(str(TMP)))
env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
done = subprocess.run([PY, "-c", BLOCKED_IMPORT], capture_output=True, text=True,
                      env=env, cwd=str(ROOT), timeout=120)
check("the package imports and runs its adapters with no API key and the brain "
      "blocked", done.returncode == 0 and "LOADED:\n" in done.stdout + "\n",
      (done.stdout + done.stderr)[-800:])


# =============================================================================
print("--- config (spec 4.17, numbers in config.py) ---")

cfg = mcfg.load()
check("config.EXPLORATION_MISSION loads and validates", cfg.loop_hz == 10.0)
check("attribute access, nested", cfg.explore.mission_cap_min == 25.0
      and cfg["return"].arrive_tolerance_m == 0.35)
check("ints are accepted as floats", isinstance(cfg.explore.progress_timeout_s, float))
check("settings are read-only", raises(lambda: setattr(cfg, "loop_hz", 5),
                                       AttributeError))
check("an override applies", mcfg.load({"explore": {"mission_cap_min": 5}})
      .explore.mission_cap_min == 5.0)
check("wrong type -> ConfigError",
      raises(lambda: mcfg.load({"loop_hz": "fast"}), mcfg.ConfigError))
check("a bool is not a number",
      raises(lambda: mcfg.load({"loop_hz": True}), mcfg.ConfigError))
check("a float is not a whole number",
      raises(lambda: mcfg.load({"events_cap": 5.5}), mcfg.ConfigError))
check("out of range -> ConfigError",
      raises(lambda: mcfg.load({"grid": {"free_below": 1.5}}), mcfg.ConfigError))
missing = json.loads(json.dumps(config.EXPLORATION_MISSION))
del missing["signature"]["scans"]
check("a missing key -> ConfigError",
      raises(lambda: mcfg.load(base=missing), mcfg.ConfigError))
logs: list[str] = []


class _Grab(logging.Handler):
    def emit(self, record):
        logs.append(record.getMessage())


mcfg.log.addHandler(_Grab())
c2 = mcfg.load({"explore": {"warp_speed": 9}})
check("an unknown key warns and is ignored",
      any("warp_speed" in m for m in logs) and "warp_speed" not in c2.explore)
check("free_below must be below occupied_above",
      raises(lambda: mcfg.load({"grid": {"free_below": 0.7}}), mcfg.ConfigError))
check("label.trigger must be a known mode",
      raises(lambda: mcfg.load({"label": {"trigger": "vibes"}}), mcfg.ConfigError))
check("curiosity.mode must be off/ask/auto",
      raises(lambda: mcfg.load({"curiosity": {"mode": "always"}}), mcfg.ConfigError))
check("quiet hours must be HH:MM",
      raises(lambda: mcfg.load({"curiosity": {"quiet_hours": ["22", "8"]}}),
             mcfg.ConfigError))
check("three curiosity weights",
      raises(lambda: mcfg.load({"curiosity": {"weights": [1, 1]}}), mcfg.ConfigError))
check("a hold refresh slower than its expiry is refused",
      raises(lambda: mcfg.load({"run": {"hold_refresh_s": 5.0}}), mcfg.ConfigError))
check("curiosity ships 'ask' (the file's default, Harsh's go 2026-10-03): asks first",
      cfg.curiosity.mode == "ask")


# =============================================================================
print("--- grid_utils (spec 4.1) ---")

g = gu.Grid(np.full((40, 30), gu.UNKNOWN, np.int8), 0.05, (-1.23, 4.56))
pts = np.random.default_rng(1).uniform([-1.2, 4.6], [0.7, 6.0], size=(500, 2))
ix, iy = g.world_to_cell(pts[:, 0], pts[:, 1])
cx, cy = g.cell_to_world(ix, iy)
jx, jy = g.world_to_cell(cx, cy)
check("world -> cell -> world(centre) -> cell round trip",
      np.array_equal(ix, jx) and np.array_equal(iy, jy))
check("the centre is within half a cell of the point",
      float(np.max(np.abs(cx - pts[:, 0]))) <= 0.025 + 1e-9
      and float(np.max(np.abs(cy - pts[:, 1]))) <= 0.025 + 1e-9)
check("axis 0 is x, axis 1 is y (slam/grid.py order, no flip)",
      g.world_to_cell(-1.23 + 0.051, 4.56) == (1, 0)
      and g.world_to_cell(-1.23, 4.56 + 0.051) == (0, 1))
check("scalar in, ints out", isinstance(g.world_to_cell(0.0, 5.0)[0], int))
check("inside() bounds", bool(g.inside(0, 0)) and not bool(g.inside(40, 0))
      and not bool(g.inside(-1, 5)))

from slam.grid import OccupancyGrid                    # noqa: E402
from slam.plan import passable_mask                     # noqa: E402

og = OccupancyGrid(cell_m=0.05)
og.logodds = np.zeros((80, 60), np.float32)
og.origin = np.array([-2.0, -1.5])
og._empty = False
og.logodds[5:75, 5:55] = -3.0                           # a room
og.logodds[5:75, 5] = og.logodds[5:75, 54] = 3.0        # walls
og.logodds[5, 5:55] = og.logodds[74, 5:55] = 3.0
og.logodds[30:34, 20:24] = 3.0                          # a box
og.logodds[50:60, 30:40] = 0.0                          # an unseen patch
og.logodds[60, 45] = -0.62                              # right on a threshold
ng = gu.from_occupancy_grid(og)
check("SLAM map: FREE == slam's free_mask", np.array_equal(ng.free(), og.free_mask()))
check("SLAM map: OCC == slam's occupied_mask",
      np.array_equal(ng.occupied(), og.occupied_mask()))
check("SLAM map: the rest is UNKNOWN",
      np.array_equal(ng.unknown(), ~og.free_mask() & ~og.occupied_mask()))
check("origin and resolution carried over",
      ng.origin == (-2.0, -1.5) and ng.resolution == 0.05)
trav = gu.traversable(ng, config.NAV_INFLATE_M)
check("traversable == File 14's passable_mask, cell for cell",
      np.array_equal(trav, passable_mask(og)),
      f"{int((trav != passable_mask(og)).sum())} cells differ")
check("unknown is never traversable", not bool(trav[ng.unknown()].any()))

pl = gu.from_probability(np.array([[0.25, 0.26, 0.64, 0.65]]), (0, 0), 0.05,
                         0.25, 0.65)
check("probability thresholds are inclusive (spec 4.1 values)",
      pl.occ.tolist() == [[gu.FREE, gu.UNKNOWN, gu.UNKNOWN, gu.OCC]])
lo = gu.from_logodds(np.array([[-5.0, 0.0, 5.0]]), (0, 0), 0.05, 0.25, 0.65)
check("log-odds use slam's probability formula",
      lo.occ.tolist() == [[gu.FREE, gu.UNKNOWN, gu.OCC]])

fine = gu.Grid(np.full((5, 4), gu.UNKNOWN, np.int8), 0.05, (1.0, 2.0))
fine.occ[0:2, 0:2] = gu.FREE                            # block (0,0): all free
fine.occ[0:2, 2:4] = gu.FREE                            # block (0,1): 3 free + 1 occ
fine.occ[1, 3] = gu.OCC
fine.occ[2, 0] = fine.occ[3, 0] = gu.FREE               # block (1,0): 2 of 4 free
fine.occ[2, 2] = gu.FREE                                # block (1,1): 1 of 4 free
fine.occ[4, :] = gu.FREE                                # row 4: padded block
coarse = gu.downsample(fine, 0.10)
check("downsample: shape padded up, resolution x2, origin kept",
      coarse.shape == (3, 2) and abs(coarse.resolution - 0.10) < 1e-12
      and coarse.origin == (1.0, 2.0))
check("downsample: any OCC -> OCC", coarse.occ[0, 1] == gu.OCC)
check("downsample: >= 50% FREE -> FREE", coarse.occ[0, 0] == gu.FREE
      and coarse.occ[1, 0] == gu.FREE)
check("downsample: < 50% FREE -> UNKNOWN", coarse.occ[1, 1] == gu.UNKNOWN)
check("downsample: the padding counts as UNKNOWN (2 of 4 free -> FREE)",
      coarse.occ[2, 0] == gu.FREE and coarse.occ[2, 1] == gu.FREE)
check("explored m2 = FREE cells x res^2",
      abs(ng.explored_m2() - float(og.free_mask().sum()) * 0.0025) < 1e-9)


# =============================================================================
print("--- adapters (temp files, fake clock) ---")

clock = ad.FakeClock(5_000.0)
run = ad.RunAdapter(state_dir=TMP / "seam", autonomy_file=TMP / "rl_autonomy.json",
                    clock=clock)
check("no live file -> None", run.live() is None)
run.hold("asking", 3.0)
check("hold() writes a hold the adapter reads back",
      (run.held() or {}).get("why") == "asking")
clock.advance(3.5)
check("a hold lapses on its own", run.held() is None)
run.hold("asking", 999.0)
check("a hold is capped at HOLD_MAX_AHEAD_S",
      (run.held() or {}).get("until", 0) - clock.wall() <= ad.HOLD_MAX_AHEAD_S)
run.release()
check("release() removes it", run.held() is None and not run.hold_file.exists())
run.release()
check("release() twice is harmless", True)
check("atomic writes leave no temp files behind",
      not [p for p in (TMP / "seam").iterdir() if p.name.endswith(".tmp")])

(TMP / "rl_autonomy.json").write_text(json.dumps({
    "TICK": 7, "EXPLORATION_TARGET": {"frontiers": 3, "exhausted": False,
                                      "error": ""},
    "MAP_STATE": {"explored_m2": 12.5}, "SAFETY_STATUS": {"detail": ""},
    "way_back": {"phase": "exploring", "outcome": "", "distance_m": None}}))
st = run.status()
check("status() reads the run's frontiers / exhausted / phase",
      st and st["frontiers"] == 3 and st["exhausted"] is False
      and st["way_back_phase"] == "exploring" and st["explored_m2_rl"] == 12.5)

bins = ad.scan_to_bins({0: 1500, 90.4: 2000, 359: 800, 10: 0, 20: -5})
check("scan_to_bins: metres per degree, NaN = no return",
      bins.shape == (360,) and abs(bins[0] - 1.5) < 1e-6 and abs(bins[90] - 2.0) < 1e-6
      and abs(bins[359] - 0.8) < 1e-6 and math.isnan(bins[10]) and math.isnan(bins[20])
      and int(np.isfinite(bins).sum()) == 3)

fa = ad.fake_adapters(voice=ad.FakeVoice(["kitchen", None]))
check("fakes: scripted voice replies, then silence",
      fa.voice.listen_once(10) == "kitchen" and fa.voice.listen_once(10) is None
      and fa.voice.listen_once(10) is None)
fa.clock.advance(1.0)
fa.run.hold("x", 2.0)
check("fakes: hold follows the fake clock", fa.run.held() is not None)
fa.clock.advance(2.5)
check("fakes: ...and lapses on it", fa.run.held() is None)
check("fakes: the face refuses a name that is not one of the 20",
      raises(lambda: fa.face.set("worried"), ValueError))
check("the real face list is face_control's 20",
      ad.FaceAdapter().names() == ad.FakeFace.NAMES)
check("power_low is unknown, never invented", ad.PowerAdapter().low() is None)


# =============================================================================
print("--- the RL seam: rl/way_back.py publisher + Hold ---")

import way_back as wb                                    # noqa: E402

live = TMP / "seam" / "live.json"
res = {"pose": (0.5, -0.25, 0.1), "reason": "matched", "ms": 12.3}
ok = wb._publish({0.0: 1500, 1.0: 1510, 359.0: 800}, (1.0, 2.0, 0.0), None,
                 (0.9, 1.9, 0.0), res, 8.0, path=live)
fix = ad.RunAdapter(state_dir=TMP / "seam").live()
check("a published fine sweep is read back by the mission's adapter",
      ok and fix and fix["pose"] == [0.5, -0.25, 0.1] and fix["reason"] == "matched"
      and fix["scan"] == {0: 1500, 1: 1510, 359: 800} and fix["rl"] == [1.0, 2.0, 0.0])
check("its age is measured", fix and 0.0 <= fix["age_s"] < 5.0)
check("...and a stale one is refused", ad.RunAdapter(
    state_dir=TMP / "seam", clock=ad.FakeClock(time.time() + 60)).live(3.0) is None)
bins72 = np.full(72, 8.0, np.float32)
bins72[0] = 1.2
bins72[18] = 2.4
ok = wb._publish(bins72, (0, 0, 0), np.array([36]), None, res, 8.0, path=live)
fix = ad.RunAdapter(state_dir=TMP / "seam").live()
check("72-bin sweeps publish as {deg: mm}, max range and blind bins left out",
      ok and fix["scan"] == {0: 1200, 90: 2400}, str(fix and fix["scan"]))
check("publishing never raises on garbage", wb._publish(object(), None, None, None,
                                                        {}, 8.0, path=live) is False)

hp = TMP / "seam" / "hold.json"
h = wb.Hold(path=hp)
h.poll()
check("Hold: no file -> inactive", not h.active())
ad.RunAdapter(state_dir=TMP / "seam").hold("asking", 3.0)
h.poll()
check("Hold: the mission's hold file is seen by the run", h.active()
      and h.snapshot() == {"active": True, "why": "asking"})
hp.write_text(json.dumps({"until": time.time() - 1}))
h.poll()
check("Hold: an expired hold is inactive", not h.active())
hp.write_text(json.dumps({"until": time.time() + 3600}))
h.poll()
check("Hold: an hour ahead (clock stepped back) is refused", not h.active())
hp.write_text("{not json")
h.poll()
check("Hold: garbage is inactive", not h.active())
hp.write_text(json.dumps({"until": time.time() + 0.3}))
h.poll()
was = h.active()
time.sleep(0.4)
check("Hold: lapses by itself with nobody refreshing it", was and not h.active())


# =============================================================================
print("--- the RL seam: pi_deploy's held tick ---")

from pi_deploy import (ControlLoop, HERE, NullMotorDriver, PolicyRunner,  # noqa: E402
                       RobotConfig, SensorSlot)

rcfg = RobotConfig()
rcfg.tof_enabled = False
rcfg.cam_enabled = False
slots = {"lidar": SensorSlot("lidar", rcfg.lidar_deadline, True),
         "tof": SensorSlot("tof", rcfg.tof_deadline, True)}
loop = ControlLoop(rcfg, PolicyRunner(HERE / "exploration_model.onnx", rcfg, 1),
                   NullMotorDriver(rcfg), slots)
check("a new ControlLoop has a hold, inactive", hasattr(loop, "hold")
      and not loop.hold.active())
loop.hold.path = TMP / "seam" / "loop_hold.json"
open_scan = np.full(rcfg.n_beams, 3.0, np.float32)
calls = {"n": 0}
real_infer = loop.policy.infer


def counting_infer(obs):
    calls["n"] += 1
    return real_infer(obs)


loop.policy.infer = counting_infer
for _ in range(5):
    slots["lidar"].publish(open_scan)
    free_row = loop.tick()
check("without a hold the policy runs as before", calls["n"] == 5
      and free_row["gate"] != "mission_hold")
ticks_before, infers_before = loop.stats["ticks"], calls["n"]
ad.atomic_write_json(loop.hold.path, {"until": time.time() + 3.0, "why": "asking"})
loop.hold.poll()
rows = []
for _ in range(10):
    slots["lidar"].publish(open_scan)
    rows.append(loop.tick())
check("held: every tick is gate mission_hold with zero out",
      all(r["gate"] == "mission_hold" and r["v"] == 0.0 and r["omega"] == 0.0
          for r in rows))
check("held: the policy is never asked", calls["n"] == infers_before)
check("held: the motors read zero", tuple(loop.motors.executed()) == (0.0, 0.0),
      str(loop.motors.executed()))
check("held: ticks still count (state keeps publishing)",
      loop.stats["ticks"] == ticks_before + 10
      and loop.stats.get("mission_hold_ticks") == 10)
st = loop.autonomy_state()
check("held: the published state says so (REASON_CODE, zero executed)",
      st and st.get("REASON_CODE") == "mission_hold"
      and st["RUN_TOTALS"].get("mission_hold_ticks") == 10, str(st and st.get("REASON_CODE")))
stale = ControlLoop(rcfg, PolicyRunner(HERE / "exploration_model.onnx", rcfg, 1),
                    NullMotorDriver(rcfg), {
                        "lidar": SensorSlot("lidar", rcfg.lidar_deadline, True),
                        "tof": SensorSlot("tof", rcfg.tof_deadline, True)})
stale.hold.path = loop.hold.path
stale.hold.poll()
srow = stale.tick()                                    # no LiDAR published at all
check("held with no LiDAR: still a zero, no crash",
      srow["gate"] == "mission_hold" and srow["v"] == 0.0
      and math.isnan(srow["lidar_min"]))
loop.hold.path.unlink()
loop.hold.poll()
slots["lidar"].publish(open_scan)
back = loop.tick()
check("released: the policy drives again", back["gate"] != "mission_hold"
      and calls["n"] == infers_before + 1)


# =============================================================================
print("--- the probe (real state, read-only) ---")

session = ROOT / "state" / "map_session.json"
watch = [session, Path(config.MEMORY_DB), Path(config.SLAM_MAP_FILE)]
before = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in watch}
hold_real = ROOT / "state" / "exploration_mission" / "hold.json"
PROBE = ("import sys; sys.path[:0]=[%r, %r]; import netguard; netguard.block(); "
         "sys.argv=['adapters','--probe']; import runpy; "
         "runpy.run_module('exploration_mission.adapters', run_name='__main__')"
         % (str(ROOT), str(ROOT / "tests" / "exploration_mission")))
done = subprocess.run([PY, "-c", PROBE], capture_output=True, text=True,
                      cwd=str(ROOT), timeout=180, env=env)
out = done.stdout
print("\n".join("        " + ln for ln in out.splitlines()))
check("the probe runs (network blocked) and exits 0", done.returncode == 0,
      done.stderr[-600:])
for key in ("pose", "map", "map_id", "inflate_m", "file14_engine", "emotions",
            "detector", "memory", "env_id", "power_low"):
    check(f"the probe reports {key}", any(ln.startswith(key) for ln in out.splitlines()))
after = {p: (p.stat().st_mtime_ns if p.exists() else None) for p in watch}
check("the probe wrote nothing (map, map session, memory.db untouched)",
      before == after, str({str(p): (before[p], after[p]) for p in watch
                            if before[p] != after[p]}))
check("the probe never held or moved the run", not hold_real.exists())


# =============================================================================
print("--- existing code: only the approved lines changed ---")

changed = subprocess.run(
    ["git", "diff", "--name-only", "HEAD", "--", "rl", "slam", "exploration_mode.py",
     "mcp_server/tools/explore.py", ":(exclude)rl/modes/*"],  # mode config, his OK
    capture_output=True, text=True,
    cwd=str(ROOT)).stdout.split()
check("rl/ + slam/ + exploration: only way_back.py, pi_deploy.py, friday_voice.py (the hold mute), "
      "unstick.py + friday_localnav_deploy.py (keep trying), run_localnav.sh (the "
      "logger that survives a stop) differ",
      set(changed) <= {"rl/way_back.py", "rl/pi_deploy.py", "rl/friday_voice.py",
                       "rl/unstick.py", "rl/friday_localnav_deploy.py",
                       "rl/run_localnav.sh"}, str(changed))
diff = subprocess.run(["git", "diff", "-U0", "HEAD", "--", "rl/pi_deploy.py"],
                      capture_output=True, text=True, cwd=str(ROOT)).stdout
removed = [ln for ln in diff.splitlines()
           if ln.startswith("-") and not ln.startswith("---")]
check("pi_deploy.py: lines only added, none removed or rewritten",
      not removed, "\n".join(removed))
check("the network guard saw no attempt from the code under test",
      all("203.0.113.7" in a or "example.com" in a for a in netguard.attempts),
      str(netguard.attempts))

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
