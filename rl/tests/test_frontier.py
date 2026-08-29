"""The frontier channel is 72 of the policy's 146 inputs. It was a dead constant.

`update_and_frontier` integrated the scan and THEN asked "is there an unknown cell
before the hit?" -- but the free-marking had just made every such cell known, so the
answer was always no. Measured in the simulator with the robot driving through
128 m^2 of unexplored world: 300 ticks x 72 rays produced ONE distinct value.

Order of operations is the whole test.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import LocalExplorationMap, RobotConfig, masked_beam_indices

cfg = RobotConfig()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def scan(dist=2.0):
    return np.full(cfg.n_beams, float(dist), np.float32)


print("--- frontier channel ---")

# 1. A fresh map is entirely unknown, so the frontier is immediately ahead on
#    every beam. If this reports max range, the channel is dead.
m = LocalExplorationMap(cfg)
f1 = m.update_and_frontier(scan())
check("fresh map reports a NEAR frontier on every beam",
      float(f1.max()) < 0.2, f"max={float(f1.max()):.3f} m")

# 2. Seen from the same pose a second time, that space is now known: no frontier.
f2 = m.update_and_frontier(scan())
check("re-observing the same space clears the frontier",
      float(f2.min()) >= cfg.lidar_max_range - 1e-3, f"min={float(f2.min()):.3f} m")

# 3. The regression itself: the channel must take more than one value over a run
#    in which the robot moves through unexplored space.
m = LocalExplorationMap(cfg)
seen = set()
for step in range(60):
    seen.update(np.unique(m.update_and_frontier(scan())).tolist())
    m.dead_reckon(0.30, 0.15, 0.1)
check("the channel VARIES over a drive", len(seen) > 1, f"{len(seen)} distinct values")

# 4. A beam pointing at freshly reachable space must report a distance, not max.
m = LocalExplorationMap(cfg)
m.update_and_frontier(scan(1.0))          # learn a 1 m bubble
m.dead_reckon(0.5, 0.0, 1.0)              # step forward out of it
f = m.update_and_frontier(scan(3.0))
check("stepping into unexplored space produces a finite frontier",
      float(f.min()) < cfg.lidar_max_range - 1e-3, f"min={float(f.min()):.3f} m")

print("--- blind beams (the chassis) ---")

# 5. A masked bearing has NO DATA, and the LD19 publishes max_range for an empty
#    bin -- indistinguishable from "nothing within 5 m". Marking it free carves a
#    5 m corridor through the robot's own body on every tick.
idx = masked_beam_indices(((88.0, 99.0),), -51.8, cfg.n_beams)
check("the chassis mask maps onto beams", idx.size > 0, f"beams {list(idx)}")

m = LocalExplorationMap(cfg)
m.blind_beams = idx
s = scan(2.0)
s[idx] = cfg.lidar_max_range              # what an empty bin publishes
f = m.update_and_frontier(s)
check("blind beams report no frontier",
      bool(np.all(f[idx] >= cfg.lidar_max_range - 1e-3)), f"{f[idx]}")

# The control: the same scan with no mask DOES carve the extra free space.
m2 = LocalExplorationMap(cfg)
m2.update_and_frontier(s)
check("masking reduces the space marked explored (control)",
      m.explored_cells < m2.explored_cells,
      f"masked={m.explored_cells} unmasked={m2.explored_cells}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
