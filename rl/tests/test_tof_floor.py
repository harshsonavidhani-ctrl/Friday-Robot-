"""The bumper ToF looks DOWN. Its bottom rows are floor, and reading them as
obstacles is why Friday once could not drive forward at all, in any room, forever.

Every case here is a failure that actually happened on the robot. The measured
floor profile is (295, 223, 161, 120) mm for rows 4-7.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pi_deploy
from pi_deploy import (RobotConfig, ToFReflex, tof_obstacles_mm,
                       TOF_FLOOR_PROFILE_MM, TOF_NO_RETURN_MM, TOF_FLOOR_FROM_ROW)

GROUND0 = TOF_FLOOR_FROM_ROW          # index of the FIRST (furthest) ground row

cfg = RobotConfig()
FLOOR = TOF_FLOOR_PROFILE_MM          # rows 4..7
AHEAD_CLEAR = 1700                    # the room ahead, from the measured profile
# The ahead rows correctly report the room as a thing -- a wall at 1.7 m IS a thing,
# and the reflex only acts below tof_slow_distance. So "clear" here means nothing
# NEAR, not nothing at all.
NEAR_MM = 600                         # cfg.tof_slow_distance
fails = 0


def nearest(obs):
    return float(obs.min())


def near_cells(obs):
    return int((obs < NEAR_MM).sum())


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def frame(ahead=AHEAD_CLEAR, pitch=1.0):
    """A clear-floor frame: four rows of room, four rows of floor."""
    f = [[int(ahead)] * 8 for _ in range(4)]
    f += [[int(round(FLOOR[r] * pitch))] * 8 for r in range(4)]
    return f


print("--- ToF floor semantics (rows 4-7 are the ground) ---")

# 1. THE regression: a clear floor must not read as an obstacle.
obs, intact = tof_obstacles_mm(frame())
check("clear floor reports nothing NEAR",
      near_cells(obs) == 0, f"nearest={nearest(obs):.0f} mm")
check("clear floor is intact", intact)

# 2. Pitch: the chassis noses down as the motors take up and EVERY ground row reads
#    short together. Un-normalised this read as an object at 246 mm and braked the
#    robot 0.2 s into its first move -- braking for its own acceleration.
for p in (0.88, 0.95, 1.05):
    obs, intact = tof_obstacles_mm(frame(pitch=p))
    check(f"pitched floor x{p} is still floor",
          near_cells(obs) == 0, f"nearest={nearest(obs):.0f} mm")

# 3. An object standing on the floor OCCLUDES the ground rows: they read SHORT.
#    Deleting that evidence is what let her drive into something until the reflex
#    caught it at SIXTEEN MILLIMETRES.
f = frame()
for r in (4, 5):
    for c in (3, 4, 5):
        f[r][c] = 150
obs, intact = tof_obstacles_mm(f)
check("occluded ground rows ARE an obstacle",
      abs(float(obs.min()) - 150) < 1, f"nearest={float(obs.min()):.0f} mm")
check("only the occluded cells read near",
      near_cells(obs) == 6, f"{near_cells(obs)} near cells")

# 4. SPAD crosstalk: the row just above the horizon echoes the strong floor return
#    below it at ~295 mm. About 1 frame in 7. One of them braked her mid-move.
f = frame()
for c in (3, 4):
    f[3][c] = FLOOR[0]                    # 295 mm, exactly the floor's first row
obs, intact = tof_obstacles_mm(f)
check("horizon-row floor echo is rejected",
      near_cells(obs) == 0, f"nearest={nearest(obs):.0f} mm")

# 5. ...but a REAL object at that range also occludes the floor, so it survives.
f = frame()
for c in (3, 4):
    f[3][c] = 290
    f[4][c] = 150                         # the same object, occluding the ground row
obs, intact = tof_obstacles_mm(f)
check("a real near object is NOT rejected as an echo",
      float(obs.min()) < 200, f"nearest={float(obs.min()):.0f} mm")

# 6. A floor that stopped reading is what a CLIFF looks like: cannot-see, never clear.
f = frame()
f[7] = [0] * 8
_, intact = tof_obstacles_mm(f)
check("a floor that stopped reading is NOT intact", not intact)

f = frame()
f[6] = [FLOOR[2] + 400] * 8               # reads much further than floor: a drop
_, intact = tof_obstacles_mm(f)
check("a drop-off is NOT intact", not intact)

# 7. End to end through the reflex -- the layer that actually stops the wheels.
reflex = ToFReflex(cfg)
obs, _ = tof_obstacles_mm(frame())
scale, stop = reflex.evaluate(np.minimum(obs / 1000.0, cfg.tof_max_range))
check("reflex does NOT fire on clear floor", not stop and scale == 1.0,
      f"scale={scale:.2f} stop={stop}")

f = frame()
for r in (4, 5):
    for c in (2, 3, 4, 5):
        f[r][c] = 200                     # an object at 0.20 m, under the 0.30 stop
obs, _ = tof_obstacles_mm(f)
scale, stop = reflex.evaluate(np.minimum(obs / 1000.0, cfg.tof_max_range))
check("reflex FIRES on a real close object", stop and scale == 0.0,
      f"scale={scale:.2f} stop={stop}")

# 8. Speckle immunity must survive the floor logic: one hot cell is not a stop.
f = frame()
f[2][4] = 120
obs, _ = tof_obstacles_mm(f)
scale, stop = reflex.evaluate(np.minimum(obs / 1000.0, cfg.tof_max_range))
check("a single hot cell does not stop the robot", not stop, f"stop={stop}")



# --- the far ground row is allowed to be patchy -------------------------------------
#
# REGRESSION, 2026-09-03. floor_intact was unanimous over all 32 ground cells, so one
# non-returning cell anywhere set it False. Row 4 looks along the shallowest incidence
# over the longest path and drops cells on an ordinary floor every frame -- measured at
# 5.59/8 returning against 8.00/8 for rows 5-7. floor_intact was therefore False on 177
# of 182 frames, DFRobotToFThread publishes nothing when the floor is not intact and
# nothing is ahead, the slot went stale, and Friday announced "A sensor has stopped
# responding" about a sensor streaming a clean 15 Hz.
f = frame()
for col in (2, 4, 6):
    f[GROUND0][col] = 0            # far row drops cells, as it does
_, intact = tof_obstacles_mm(f)
check("a patchy FAR ground row is still an intact floor", intact)

# ...and the near rows still carry the claim. They are what "the path immediately in
# front of the bumper is empty" is actually an argument about, so losing them must
# still read as cannot-see.
f = frame()
for off in range(1, len(TOF_FLOOR_PROFILE_MM)):
    for col in range(8):
        f[GROUND0 + off][col] = 0
_, intact = tof_obstacles_mm(f)
check("...but losing the NEAR ground rows is still cannot-see", not intact)

# A single dead near cell is noise, not a cliff -- the quorum has to tolerate one.
f = frame()
f[GROUND0 + 2][3] = 0
_, intact = tof_obstacles_mm(f)
check("one dead near cell does not condemn the floor", intact)

# Occlusion detection must be untouched on the FAR row: a cell that reads short is an
# object whichever row it is in, and that is the bug that let her bump something at
# 16 mm in 2026-08-27.
f = frame()
for col in (3, 4, 5):
    f[GROUND0][col] = TOF_FLOOR_PROFILE_MM[0] - 120
obstacles, _ = tof_obstacles_mm(f)
check("a short FAR ground cell is still reported as an obstacle",
      float(obstacles.min()) < TOF_FLOOR_PROFILE_MM[0],
      f"nearest={float(obstacles.min()):.0f} mm")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
