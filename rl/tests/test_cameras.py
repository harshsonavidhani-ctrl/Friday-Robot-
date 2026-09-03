"""Camera identity and self-healing.

Two real defects are locked here.

  1. IDENTITY. `cam_indices = (0, 1)` opened the up camera's METADATA node as the
     second camera and never touched the down camera at all. Every UVC device on this
     machine occupies two consecutive video numbers, so index arithmetic cannot work.
     Confirmed live: cam0 updates=87, cam1 updates=0 stale=True.

  2. SURVIVAL. The down camera (349c:2317 on usb 1-1.1.4) re-enumerates on its own --
     "USB disconnect", then a new device number, minutes later, unprompted. The old
     `SensorThread.run()` caught the setup exception and EXITED, so one glitch meant
     the camera was gone for the rest of the run. The root cause is physical and
     software cannot fix it; coming back without a restart is what software can do.

The reconnect test uses a stubbed resolver rather than the real bus, because the
failure being reproduced is a device that is absent and then present, and a test that
depended on the hardware actually flapping would pass or fail for reasons unrelated to
the code.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cameras as cams
from pi_deploy import CameraThread, RobotConfig, SensorSlot

cfg = RobotConfig()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


print("--- camera identity and self-healing ---")

# 1. IDENTITY. Resolution is by USB serial and must verify the node is a real capture
#    device, not a metadata node.
found = cams.resolve()
check("resolve() returns only real capture nodes",
      all(d.path.startswith("/dev/video") for d in found.values()),
      cams.describe(found))
check("a resolved camera carries the serial it was found by",
      all(d.serial in d.by_id for d in found.values()))
check("a missing camera is ABSENT, never a guessed index",
      set(found) <= {"up", "down"}, f"roles: {sorted(found)}")

# The up camera is the one reliably present on this robot; assert against it when it
# is there, and skip rather than fail when it is not (the bus is genuinely flaky).
if "up" in found:
    check("the up camera resolves to a Video Capture node",
          cams._is_capture_node(found["up"].path), found["up"].path)
    # The metadata sibling is the -video-index1 link, NOT video<n+1> by arithmetic.
    # This used to be found["up"].path.replace("video0", "video1"), which silently
    # did NOTHING once the up camera enumerated as /dev/video2 -- so the test handed
    # the REAL capture node to a check that asserts rejection, and failed. The by-id
    # name is the only thing that survives a re-enumeration; index arithmetic is the
    # exact mistake this whole module exists to avoid.
    meta = found["up"].by_id.replace("-video-index0", "-video-index1")
    if Path(meta).exists():
        check("...and its metadata sibling is correctly REJECTED",
              not cams._is_capture_node(meta), meta)
else:
    print("  [skip] up camera not on the bus right now")

# 2. SELF-HEALING. Absent -> present must be picked up, with no restart.
class _FakeCap:
    """A camera that opens and yields frames."""

    def __init__(self, ok=True):
        self._ok = ok
        self.released = False

    def isOpened(self):
        return self._ok

    def set(self, *a):
        return True

    def read(self):
        return (True, np.zeros((240, 320, 3), np.uint8)) if self._ok else (False, None)

    def release(self):
        self.released = True


slot = SensorSlot("cam_down", cfg.cam_deadline, critical=False)
th = CameraThread(slot, cfg, "down", serial="TESTSERIAL")

present = {"now": False}
th._resolve_path = lambda: "/dev/video99" if present["now"] else None
th._try_open = lambda: (setattr(th, "_cap", _FakeCap()),
                        setattr(th, "device", "/dev/video99"),
                        True)[-1] if present["now"] else False

# Absent at startup: must not raise, must not publish, must schedule a retry.
th._setup()
check("an absent camera does not raise at setup", True)
check("...and publishes nothing", slot.updates == 0)
check("...and records the open failure", th.open_failures == 1)

th._next_try = 0.0
th._poll()
check("polling while absent still publishes nothing", slot.updates == 0)
check("...and counts another failure", th.open_failures == 2)
check("...and backs off rather than spinning", th._backoff > CameraThread.RETRY_MIN_S,
      f"backoff {th._backoff:.1f}s")
check("...but the backoff is capped", th._backoff <= CameraThread.RETRY_MAX_S)

# The device comes back.
present["now"] = True
th._next_try = 0.0
th._poll()
check("the camera is picked up when it returns", th._cap is not None)
check("...and the reconnect is counted", th.reconnects == 1)
check("...and the device path is recorded", th.device == "/dev/video99")

# 3. It must survive vanishing again mid-run, without dying.
th._cap = _FakeCap(ok=False)          # reads now fail
present["now"] = False                # ...and the device is gone from the bus
th._poll()
check("a camera that vanishes mid-run is dropped, not fatal", th._cap is None)
check("...and a retry is scheduled", th._next_try > 0)

present["now"] = True
th._next_try = 0.0
th._poll()
check("...and it is picked up AGAIN on the next return", th.reconnects == 2)

# 4. A read failure while the device is STILL on the bus must not drop the camera --
#    that is a transient, and tearing down for it costs far more frames than the fault.
th._cap = _FakeCap(ok=False)
present["now"] = True
before = th.reconnects
th._poll()
check("a transient read failure does NOT drop an enumerated camera",
      th._cap is not None)
check("...and does not count as a reconnect", th.reconnects == before)

# 5. ContextMonitor: measured statistics only, and no invented fields.
cm = cams.ContextMonitor()
blank = cm.snapshot()
check("context reports None before any frame, never 0.0",
      blank["brightness"] is None and blank["change"] is None)
cm.update(np.full((240, 320), 120, np.uint8))
s1 = cm.snapshot()
check("brightness is measured", abs(s1["brightness"] - 120.0) < 1e-6, str(s1["brightness"]))
check("change is None until there are two frames", s1["change"] is None)
cm.update(np.full((240, 320), 140, np.uint8))
s2 = cm.snapshot()
check("change is measured between frames", abs(s2["change"] - 20.0) < 1e-6, str(s2["change"]))
cm.update(np.zeros((240, 320), np.uint8))
check("occlusion is measured", cm.snapshot()["occluded"] == 1.0)
check("context invents no semantic fields",
      set(cm.snapshot()) == {"brightness", "change", "occluded", "frames"},
      str(sorted(cm.snapshot())))

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)
