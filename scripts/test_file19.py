#!/usr/bin/env python3
"""File 19: environments, landmark naming, obstacle logs, run summaries.

    ./venv/bin/python scripts/test_file19.py

No LiDAR, no motors, no speech, no money. memory.db, the navigation map, the
environment folders, the status/summary files, the explore status + maps and
the nav log are ALL temp copies; the relocaliser and the voice are fakes. The
last cases check that exploration's own files are unchanged against git.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-file19-"))
config.MEMORY_DB = TMP / "memory.db"
config.SLAM_MAP_FILE = TMP / "map.npz"
config.ENV_DIR = TMP / "environments"
config.ENV_STATUS_FILE = TMP / "environment_status.json"
config.RUN_SUMMARY_FILE = TMP / "run_summaries.json"
config.SCENE_JSON = TMP / "scene.json"
config.MODE_FILE = TMP / "mode.json"
config.LISTENING_FILE = TMP / "listening.json"
config.SPEAKING_FILE = TMP / "speaking.json"

import environment_watch                                # noqa: E402
import environments                                     # noqa: E402
import nav_state_machine as nsm                         # noqa: E402
import offline                                          # noqa: E402
import run_summary                                      # noqa: E402
from slam import places                                 # noqa: E402
from slam.grid import OccupancyGrid                     # noqa: E402

environment_watch.STATE = TMP
nsm.NAV_LOG = TMP / "nav_log.jsonl"
nsm._bundle = lambda reason: None                       # no bundle writes
said: list[str] = []
import nav_voice                                        # noqa: E402
nav_voice.VOICE.say_text = lambda text: said.append(text) or text

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def grid_with(free_boxes=(), walls=(), size=(120, 120)) -> OccupancyGrid:
    g = OccupancyGrid(cell_m=0.05)
    g.logodds = np.zeros(size, dtype=np.float32)
    for x0, y0, x1, y1 in free_boxes:
        g.logodds[x0:x1, y0:y1] = -3.0
    for x0, y0, x1, y1 in walls:
        g.logodds[x0:x1, y0:y1] = 3.0
    g._empty = False
    return g


# ---------------------------------------------------------------------------
print("\nstep 1: the migration keeps everything")
grid_with([(10, 10, 40, 40)]).save(config.SLAM_MAP_FILE)
places.remember_here("kitchen", (1.0, 1.0, 0.0))
places.remember_here("desk", (1.5, 0.8, 0.0))
before = config.SLAM_MAP_FILE.read_bytes()
environments.ensure()
home = environments.active()
check("today's map + places became the first environment",
      home["name"] == config.ENV_FIRST_NAME and home["active"]
      and sorted(home["landmarks"]) == ["desk", "kitchen"], str(home))
check("...and nothing was moved or reset",
      config.SLAM_MAP_FILE.read_bytes() == before
      and {p["name"] for p in environments.live_places()} == {"desk", "kitchen"})
environments.ensure()
check("the migration runs once", len(environments.all_envs()) == 1)
check("its type is the documented 'environment'",
      home["kind"] == "environment")
import memory_governance                                # noqa: E402
check("'environment' is in the memory type convention",
      "environment" in memory_governance.TYPES)

# ---------------------------------------------------------------------------
print("\nstep 2: boot check — recognise, or ASK (never create)")


class Fix:
    def __init__(self, pose=None, score=0.0, candidate=None, why=""):
        self.pose, self.score, self.candidate, self.why = pose, score, candidate, why

    @property
    def ok(self):
        return self.pose is not None


check("a confident fit is the same place",
      environments.judge(Fix((1, 1, 0), 0.9))[0] == "matched")
check("two look-alike spots still means this map",
      environments.judge(Fix(None, 0.8, candidate=(1, 1, 0)))[0] == "matched")
check("no fit anywhere is a mismatch",
      environments.judge(Fix(None, 0.1, why="no fit"))[0] == "no_match")
import global_localise                                  # noqa: E402
global_localise.localise = lambda grid, scan: Fix(None, 0.1, why="no fit")
envs_before = len(environments.all_envs())
result = environments.boot_check({0: 1000})
check("a mismatch at boot is FLAGGED with a question",
      result["state"] == "no_match" and "new place" in result["ask"])
check("...and NO environment is auto-created",
      len(environments.all_envs()) == envs_before)
global_localise.localise = lambda grid, scan: Fix((1, 1, 0), 0.9)
check("a match is recorded as such",
      environments.boot_check({0: 1000})["state"] == "matched"
      and environments.status()["state"] == "matched")

# ---------------------------------------------------------------------------
print("\nstep 3: landmark names are per environment")
said_new = environments.create_new("Office")
office = environments.active()
check("'this is a new place' makes a new, empty environment",
      office["name"] == "Office" and environments.live_places() == []
      and not config.SLAM_MAP_FILE.exists(), said_new)
home_dir = environments._dir(home["env_id"])
check("...and Home's map and places are kept in its folder",
      (home_dir / "map.npz").read_bytes() == before
      and {p["name"] for p in json.loads((home_dir / "places.json").read_text())}
      == {"desk", "kitchen"})
places.remember_here("kitchen", (3.0, 3.0, 0.0))
check("'kitchen' in Office does not clash with Home's 'kitchen'",
      places.find("kitchen")["x"] == 3.0)
check(environments.switch_to("Home"), environments.active()["name"] == "Home")
check("back in Home, 'kitchen' is Home's kitchen again",
      places.find("kitchen")["x"] == 1.0
      and config.SLAM_MAP_FILE.read_bytes() == before)
environments.switch_to("Office")
check("and Office's kitchen came back with Office",
      places.find("kitchen")["x"] == 3.0)
environments.switch_to("Home")
check("an unknown place name is not a switch",
      "don't know" in environments.switch_to("the moon"))
check("a second 'Office' is refused", "already" in environments.create_new("Office"))

# ---------------------------------------------------------------------------
print("\nstep 4: new areas queue for a name")
watch = environment_watch.Watch()
watch.env_id = environments.active()["env_id"]
check("first look at the map is the baseline, not a discovery",
      watch.map_changed() == [])
grid_with([(10, 10, 40, 40), (45, 10, 90, 60)]).save(config.SLAM_MAP_FILE)   # +5.6 m²
found = watch.map_changed()
check("a big new free region becomes a pending area",
      len(found) == 1 and environments.pending()[-1]["area_m2"] >= 4.0,
      str(found))
check("an unchanged map adds nothing", watch.map_changed() == [])
pid = environments.pending()[-1]["id"]
for bad, why in (("", "empty"), ("x" * 60, "under"), ("rm -rf /", "letters")):
    check(f"a bad name is refused ({why})",
          environments.validate_name(bad)[0] is None)
check("an existing name is refused",
      "already" in environments.name_pending(pid, "kitchen"))
check("a good name saves a landmark there",
      environments.name_pending(pid, "Living Room").startswith("Saved")
      and places.find("living room") is not None
      and environments.pending() == [])

print("\n  voice")
environments.add_pending(2.0, 2.0, 5.0, 1.0)
for said_, name in (("Friday, this is a new place", "env_new"),
                    ("this is a new place called the lab", "env_new"),
                    ("we're at home", "env_switch"),
                    ("call it the study", "env_name")):
    intent = offline.understand(said_, strict=True)
    check(f"{said_!r} -> {name}", intent is not None and intent.name == name,
          str(intent))
check("'we're at the park' (unknown) is NOT a switch",
      getattr(offline.understand("we're at the park", strict=True), "name",
              None) != "env_switch")
check("'this place is the kitchen' is still file 14's remember_place",
      offline.understand("this place is the kitchen", strict=True).name
      == "remember_place")

print("\n  the voice half of the queue (only mid-conversation)")
import reminders                                        # noqa: E402
rw = reminders.ReminderWatch()
config.BRIEFING = False
check("nobody talking to her: the dashboard queue only", rw.step() is None)
rw._last_heard = time.time() - 5
item = rw.step()
check("mid-conversation: she also asks by voice",
      item is not None and item["kind"] == "landmark"
      and "What should I call it" in item["say"], str(item))
rw.delivered(item)
check("...once", rw.step() is None)

print("\n  step 5: the dashboard's one write")
from dashboard import server as dash                    # noqa: E402


class Req:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def post(body):
    try:
        return asyncio.run(dash.api_landmarks_post(Req(body))).status
    except Exception as exc:                            # noqa: BLE001
        return getattr(exc, "status", type(exc).__name__)


pid = environments.pending()[-1]["id"]
check("anything besides pending_id + name is refused",
      post({"pending_id": pid, "name": "study", "action": "drive"}) == 400)
check("an empty name is refused", post({"pending_id": pid, "name": " "}) == 400)
check("a valid name is saved", post({"pending_id": pid, "name": "study"}) == 200
      and places.find("study") is not None)
routes = (ROOT / "dashboard" / "server.py").read_text()
check("the route is registered once, as POST /api/landmarks",
      routes.count('"/api/landmarks"') == 1)

# ---------------------------------------------------------------------------
print("\nstep 6: obstacles near named places")
kitchen = places.find("kitchen")
kx, ky = int(kitchen["x"] / 0.05), int(kitchen["y"] / 0.05)
a = grid_with([(10, 10, 40, 40)])
b = grid_with([(10, 10, 40, 40)], walls=[(kx - 2, ky - 2, kx + 2, ky + 2)])
entries = environments.obstacles_near_places(a, b)
check("a new obstacle inside the kitchen is logged to the kitchen",
      [e["place"] for e in entries] == ["kitchen"]
      and entries[0]["class"] == "unclassified", str(entries))
far = grid_with([(10, 10, 40, 40)], walls=[(110, 110, 114, 114)])
check("one far from every place is not",
      environments.obstacles_near_places(a, far) == [])
for _ in range(config.ENV_OBSTACLE_KEEP + 5):
    environments.log_obstacles(entries)
book = environments.active()["obstacle_log"]["kitchen"]
check("the per-place log stays bounded", len(book) == config.ENV_OBSTACLE_KEEP)

# ---------------------------------------------------------------------------
print("\nstep 7: run summaries")
rl_prev = grid_with([(10, 10, 30, 30)])
rl_after = grid_with([(10, 10, 60, 60)])
rl_prev.save(TMP / "rl_map.prev.npz")
rl_after.save(TMP / "rl_map.npz")
log_file = TMP / "run.log"
log_file.write_text("...\nI am back where I started.\n")
status = {"state": "running", "started": "2026-09-27T20:00:00+05:30",
          "log": str(log_file)}
(TMP / "explore_mode.json").write_text(json.dumps(status))
check("mid-run: no summary", watch.explore_finished() is None)
status.update(state="completed", ended="2026-09-27T20:05:00+05:30")
(TMP / "explore_mode.json").write_text(json.dumps(status))
said.clear()
s = watch.explore_finished()
check("run end: a summary with the minutes and the area gained",
      s is not None and s["minutes"] == 5.0 and s["area_gained_m2"] > 4
      and s["came_back"], str(s))
check("...spoken, because she came back",
      said and "5 minute run" in said[-1], str(said))
check("the same end is not summarised twice", watch.explore_finished() is None)

nsm.NAV_LOG.write_text("")
watch.nav_arrivals()                                    # sets the offset
with nsm.NAV_LOG.open("a") as handle:
    handle.write(json.dumps({"at": time.time(), "destination": "home",
                             "arrived": True, "seconds": 90.0,
                             "driven_m": 3.2}) + "\n")
    handle.write(json.dumps({"at": time.time(), "destination": "the kitchen",
                             "arrived": True, "seconds": 30.0}) + "\n")
done = watch.nav_arrivals()
check("navigation arriving HOME is summarised; a kitchen errand is not",
      len(done) == 1 and done[0]["kind"] == "return"
      and "Back at home" in said[-1], str(done))
for _ in range(config.RUN_SUMMARY_KEEP + 7):
    run_summary.record(run_summary.build("return", time.time() - 60,
                                         time.time(), came_back=True))
data = run_summary.read()
check("the summary log is bounded; older runs fold into a total",
      len(data["runs"]) == config.RUN_SUMMARY_KEEP
      and data["older"]["runs"] >= 7, str(data.get("older")))

# ---------------------------------------------------------------------------
print("\nthe hard rule: exploration itself is unchanged")
diff = subprocess.run(
    ["git", "-C", str(ROOT), "status", "--porcelain", "--", "rl", "slam",
     "exploration_mode.py", "mcp_server/tools/explore.py"],
    capture_output=True, text=True).stdout.strip()
check("rl/, slam/, exploration_mode.py, explore.py: no changes", diff == "",
      diff)

print(f"\n{sum(_results)}/{len(_results)} passed")
raise SystemExit(0 if all(_results) else 1)
