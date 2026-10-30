"""The map while she makes it: written off the loop, read by the dashboard, gone at the end."""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

import numpy as np                                          # noqa: E402
import live_map                                             # noqa: E402
from world import WorldMap                                  # noqa: E402
from dashboard import map_view                              # noqa: E402

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail if not ok else ''}")
    fails += 0 if ok else 1


tmp = Path(tempfile.mkdtemp())
w = WorldMap(max_range=8.0)
w.map_file = tmp / "rl_map.npz"
live = tmp / "rl_map.live.npz"
pts = np.array([[1.0, 0.0], [1.0, 0.1], [0.0, 1.2], [-1.0, 0.3]], np.float32)

m = live_map.LiveMap(w, live)
check("nothing mapped yet: nothing written", m.write_once() is False and not live.exists())
try:
    w.grid.integrate(pts, (0.0, 0.0))
except TypeError:
    w.grid.integrate(pts, (0.0, 0.0), 8.0)
w.set_pose(0.5, 0.25, 0.3)
w._mark_visited(0.5, 0.25)
t0 = time.perf_counter()
check("a map writes one file", m.write_once() is True and live.exists())
print(f"  one write {1000 * (time.perf_counter() - t0):.1f} ms, {live.stat().st_size} bytes")
check("no temporary file is left behind", not list(tmp.glob("*.tmp*")))

map_view.LIVE_FILE, map_view.MAP_FILE = live, tmp / "none.npz"
map_view.VISITS_FILE, map_view.POSE_FILE = tmp / "n1", tmp / "n2"
map_view._cache.update(key=None, png=b"", meta={})
png, meta = map_view.current()
check("the dashboard draws it", png[:4] == b"\x89PNG" and meta.get("ok"), str(meta))
check("it says it is live", meta.get("live") is True)
check("her position comes from it", meta.get("pose") == [0.5, 0.25, 0.3], str(meta.get("pose")))
v1 = meta["version"]
time.sleep(0.02)
w.set_pose(0.9, 0.25, 0.3)
m.write_once()
import os
os.utime(live, (time.time() + 1, time.time() + 1))
_, meta = map_view.current()
check("a new write is a new picture", meta["version"] != v1 and meta["pose"][0] == 0.9)

old = time.time() - map_view.LIVE_FRESH_S - 5
os.utime(live, (old, old))
_, meta = map_view.current()
check("a live file that stopped changing is not believed", not meta.get("live"), str(meta))

m.stop()
check("the run ending removes it", not live.exists())

m = live_map.LiveMap(w, tmp / "t.npz", period_s=0.05)
m.start()
time.sleep(0.3)
m.stop()
m.join(1.0)
check("the thread writes on its own and stops", m.writes >= 2 and not m.is_alive(), str(m.writes))

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
