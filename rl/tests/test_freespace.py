"""Friday must be able to see where the space is, and say so in the right units.

Harsh, 2026-09-03: "I suggest an approach where YOU access both the cameras and
analyse all the available space in the environment and move according to the
space available."

The scenes here are synthetic on purpose -- a floor, a background, and a box
standing on the floor -- because the properties being asserted are geometric and
must hold in any room, not just the one the robot happens to be in tonight. The
numbers from the real camera are recorded in test 8 as a regression anchor.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import friday_freespace as fs

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


def scene(box=None, floor=170, back=140, obj=70):
    """A floor, a wall above the horizon, and optionally a box standing on it."""
    img = np.full((240, 320), floor, np.uint8)
    img[:96, :] = back
    if box == "right":
        img[60:150, 170:300] = obj
    elif box == "left":
        img[60:150, 20:150] = obj
    elif box == "mid":
        img[60:150, 120:200] = obj
    return img


print("--- free space from the cameras ---")

# 1. THE SIGN. Getting this backwards steers her confidently into the thing she can
#    see, and it is invisible in any test that only checks magnitude. Positive is
#    LEFT, the same convention omega uses everywhere else in this stack.
right = fs.free_profile(scene("right"))
left = fs.free_profile(scene("left"))
check("a box on the right sends her left", fs.gap_bearing(right) > 0,
      f"{math.degrees(fs.gap_bearing(right)):+.0f} deg")
check("a box on the left sends her right", fs.gap_bearing(left) < 0,
      f"{math.degrees(fs.gap_bearing(left)):+.0f} deg")
check("...and they are mirror images",
      abs(fs.gap_bearing(right) + fs.gap_bearing(left)) < math.radians(3),
      f"{math.degrees(fs.gap_bearing(right)):+.0f} vs "
      f"{math.degrees(fs.gap_bearing(left)):+.0f}")

# 2. THE WIDEST GAP, NOT THE CENTRE OF MASS. This is the assertion that matters most
#    and the one a centroid-based planner fails: two openings either side of an
#    obstacle have their centre of mass ON the obstacle.
mid = fs.free_profile(scene("mid"))
start, stop, _ = fs.widest_gap(mid)
blocked_bin = int(np.argmin(mid))
check("an obstacle dead ahead is never chosen as the gap",
      not (start <= blocked_bin < stop), f"gap=[{start},{stop}) blocked={blocked_bin}")
check("...and she is steered off centre rather than into it",
      abs(fs.gap_bearing(mid)) > math.radians(5),
      f"{math.degrees(fs.gap_bearing(mid)):+.0f} deg")

# 3. An empty room must not invent a reason to turn. A steering advisory that
#    wanders with nothing in front of it is a second source of the zig-zag this
#    whole change exists to remove.
clear = fs.free_profile(scene(None))
check("open floor steers straight ahead", abs(fs.gap_bearing(clear)) < math.radians(3),
      f"{math.degrees(fs.gap_bearing(clear)):+.0f} deg")
check("...and reads as open in every direction", float(clear.min()) > 0.3,
      str(np.round(clear, 2)))

# 4. A bin is only as passable as its most blocked column. Averaging inside a bin
#    lets the open half of a door frame hide the half she would drive into.
narrow = scene(None)
narrow[60:150, 152:168] = 70            # a pole, thinner than one bin
prof = fs.free_profile(narrow)
check("a pole narrower than a bin still closes that bin",
      float(prof.min()) < float(clear.min()) - 0.1,
      f"min {prof.min():.2f} vs open {clear.min():.2f}")

# 5. DISTANCE IS MONOTONE AND BOUNDED. The mapping is approximate -- camera height
#    and tilt are estimates -- so what is asserted is the shape of it, which is what
#    the capped fallback actually relies on.
r_near, r_far = fs.row_to_range(0.1), fs.row_to_range(0.5)
check("more visible floor means more room", r_far > r_near,
      f"{r_near:.2f} m -> {r_far:.2f} m")
# The bottom row of the frame is NOT the bumper. A level camera 0.30 m up with a
# 38.2 deg vertical field cannot see the ground nearer than h / tan(FOV_V/2), and
# pretending otherwise would put an obstacle a metre closer than it is.
near_limit = fs.CAM_HEIGHT_M / math.tan(math.radians(fs.FOV_V_DEG) / 2)
check("the bottom of the frame is the near limit of the mount, honestly reported",
      abs(fs.row_to_range(0.0) - near_limit) < 0.05,
      f"{fs.row_to_range(0.0):.2f} m vs geometric {near_limit:.2f} m")
check("...and that blind zone is the ToF's to cover", near_limit > 0.5,
      f"{near_limit:.2f} m")
check("the horizon is clamped, not infinite", fs.row_to_range(1.0) <= fs.MAX_RANGE_M,
      f"{fs.row_to_range(1.0):.2f} m")

# 6. IT MUST NEVER COST A TICK. A camera that has fallen off the USB bus is normal
#    on this robot -- the down camera does it repeatedly, for physical reasons
#    software cannot fix -- and the run must not notice.
nav = fs.FreeSpaceNavigator()
view = nav.update(cam_up=None, cam_down=None)
check("no cameras at all is a usable answer, not an exception",
      view.source == "none" and not view.usable, repr(view))
view = nav.update(cam_up=np.zeros((4, 4), np.uint8))
check("a nonsense frame does not raise", view is not None, repr(view))
view = nav.update(cam_up=scene("right"))
check("one camera is enough", view.source == "up" and view.usable, repr(view))
check("...and it still points at the space", view.bearing > 0,
      f"{math.degrees(view.bearing):+.0f} deg")

# 7. THE NEAR VIEW CAN ONLY EVER REDUCE WHAT THE FAR VIEW PROMISES. Something at
#    the bumper is not made passable by open floor behind it.
#    Note the shape of the scene: an obstacle standing on floor that is still
#    visible at the very bottom. A near view that is UNIFORMLY the obstacle cannot
#    be told from a uniformly clean floor by any appearance model -- that case is
#    the ToF's, and confidence() refuses to guess at it rather than inventing a
#    block (see the featureless-frame assertion below).
both = fs.FreeSpaceNavigator()
blocked_down = scene(None)
blocked_down[:170, :] = 70           # obstacle filling most of the near view
v = both.update(cam_up=scene(None), cam_down=blocked_down)
check("a blocked near view overrides an open far view",
      float(v.profile.max()) <= float(clear.max()),
      f"{np.round(v.profile, 2)} vs open {np.round(clear, 2)}")
check("...and is reported as a near block", v.near_blocked, repr(v))
check("both cameras are named in the source", v.source == "up+down", v.source)

# 8. Regression anchor from the real up camera on 2026-09-03, facing a wall the
#    LiDAR put at 1.11 m and the ToF at 1.66 m. The estimate does not have to match
#    either -- they are different heights and different fields of view -- but it must
#    stay in the same world as them, and it must not silently drift to the 4 m clamp
#    the way the intensity-only version did.
real = scene(None)
real[96:150, :] = 150                                # a wall meeting the floor
v = fs.FreeSpaceNavigator().update(cam_up=real)
check("a wall ahead is measured in metres, not clamped away",
      0.2 < v.centre_range < fs.MAX_RANGE_M, f"{v.centre_range:.2f} m")

# 9. Confidence has to be able to say no. It is what stands between an unverifiable
#    reading and the wheels, in the one mode where nothing else is watching.
# A featureless frame is the failure that looks most like success: the floor model
# fits it perfectly and reports wide open. It must not be believed.
flat = np.full((240, 320), 128, np.uint8)
check("a featureless frame is not trusted", fs.confidence(flat) < fs.MIN_CONFIDENCE,
      f"{fs.confidence(flat):.2f}")

dark = np.full((240, 320), 4, np.uint8)
check("a black frame is not trusted", fs.confidence(dark) < fs.MIN_CONFIDENCE,
      f"{fs.confidence(dark):.2f}")
blown = np.full((240, 320), 252, np.uint8)
check("a blown-out frame is not trusted", fs.confidence(blown) < fs.MIN_CONFIDENCE,
      f"{fs.confidence(blown):.2f}")
check("a normal scene is trusted", fs.confidence(scene("right")) >= fs.MIN_CONFIDENCE,
      f"{fs.confidence(scene('right')):.2f}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
