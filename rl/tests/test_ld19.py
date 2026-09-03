"""Validate the LD19/STL-19P frame parser against synthesized, protocol-correct frames."""
import struct
import sys

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import LD19Thread, RobotConfig, SensorSlot, ld19_crc8

cfg = RobotConfig()
# Centre the obstacle INSIDE one beam. At 72 beams a beam spans 5 deg, so beam 18 is
# [90, 95). A +-3 deg window around 90 would straddle beams 17 and 18 and make argmin
# ambiguous -- that is a test artefact, not a driver behaviour.
OBSTACLE_DEG, OBSTACLE_HALF_WIDTH = 92.5, 1.5
OBSTACLE_M, BACKGROUND_M = 1.00, 3.00


def build_frame(start_deg, end_deg, dists_mm):
    assert len(dists_mm) == 12
    f = bytearray()
    f += bytes((0x54, 0x2C))
    f += struct.pack("<H", 3600)                       # speed deg/s
    f += struct.pack("<H", int(round(start_deg * 100)))
    for d in dists_mm:
        f += struct.pack("<H", int(d)) + bytes((200,))  # dist + intensity
    f += struct.pack("<H", int(round(end_deg * 100)))
    f += struct.pack("<H", 12345)                      # timestamp
    f += bytes((ld19_crc8(bytes(f)),))
    assert len(f) == 47, len(f)
    return bytes(f)


def prime(th, span=11.0):
    """Feed one revolution and its wrap, which the driver DISCARDS by contract.

    A real thread attaches mid-sweep, so the bins behind its first wrap cover only
    the arc the sensor happened to cross after start-up -- measured on hardware as
    47 of 72 beams reading max_range. LD19Thread therefore treats the first wrap as
    nothing but the marker for where a revolution ends. Tests have to get past that
    marker before any sweep of theirs can publish.
    """
    a = 0.0
    while a < 360.0:
        end = min(a + span, 359.99)
        th._handle_frame(build_frame(a, end, [BACKGROUND_M * 1000] * 12))
        a += span
    th._handle_frame(build_frame(0.0, 11.0, [BACKGROUND_M * 1000] * 12))


def make_thread(primed=True):
    """primed=False for tests that assert on raw frame counters -- priming feeds
    a revolution of frames, which would inflate _frames and _crc_ok."""
    slot = SensorSlot("lidar", 1.0, True)
    th = LD19Thread(slot, cfg, "/dev/null")
    if primed:
        prime(th)
    return th, slot


def sweep(th, span=11.0):
    """Emit one full revolution, then one wrap frame to trigger publication."""
    a = 0.0
    while a < 360.0:
        end = min(a + span, 359.99)
        angles = np.linspace(a, end, 12)
        dists = [OBSTACLE_M * 1000 if abs(x - OBSTACLE_DEG) < OBSTACLE_HALF_WIDTH
                 else BACKGROUND_M * 1000 for x in angles]
        th._handle_frame(build_frame(a, end, dists))
        a += span
    th._handle_frame(build_frame(0.0, 11.0, [BACKGROUND_M * 1000] * 12))   # wrap -> publish


fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


print("--- LD19 / STL-19P parser tests ---")

# 1. a full sweep publishes a scan
th, slot = make_thread()
sweep(th)
scan, age = slot.read()
check("full revolution publishes a scan", scan is not None and age < 1.0)

# 2. obstacle lands on the correct beam
if scan is not None:
    expected = int(OBSTACLE_DEG / 360.0 * cfg.n_beams)
    got = int(np.argmin(scan))
    check("obstacle at correct beam", got == expected, f"expected #{expected}, got #{got}")
    check("obstacle range correct", abs(float(scan[got]) - OBSTACLE_M) < 0.02,
          f"{float(scan[got]):.3f} m")
    others = np.delete(scan, got)
    check("background range correct", abs(float(np.median(others)) - BACKGROUND_M) < 0.02,
          f"{float(np.median(others)):.3f} m")
    check("scan width matches config", scan.shape == (cfg.n_beams,), str(scan.shape))

# 3. CRC rejects corruption
th2, slot2 = make_thread(primed=False)
good = build_frame(0.0, 11.0, [1000] * 12)
bad = bytearray(good)
bad[10] ^= 0xFF                       # flip bits in a distance field
th2._handle_frame(bytes(bad))
check("corrupt frame rejected by CRC", th2._crc_ok == 0 and th2._frames == 1)
th2._handle_frame(good)
check("valid frame accepted by CRC", th2._crc_ok == 1)

# 4. 0 mm (no return) must not read as a contact at 0 m
th3, slot3 = make_thread()
a = 0.0
while a < 360.0:
    end = min(a + 11.0, 359.99)
    th3._handle_frame(build_frame(a, end, [0] * 12))     # every point invalid
    a += 11.0
th3._handle_frame(build_frame(0.0, 11.0, [0] * 12))
scan3, _ = slot3.read()
check("zero-range points become max_range, not 0 m",
      scan3 is not None and float(np.min(scan3)) == cfg.lidar_max_range,
      f"min={float(np.min(scan3)):.2f}" if scan3 is not None else "no scan")

# 5. offset shifts the obstacle predictably
th4, slot4 = make_thread()
th4.offset_deg = 90.0
sweep(th4)
scan4, _ = slot4.read()
if scan4 is not None:
    expected4 = int(((OBSTACLE_DEG + 90.0) % 360) / 360.0 * cfg.n_beams)
    got4 = int(np.argmin(scan4))
    check("angle offset applied correctly", got4 == expected4,
          f"expected #{expected4}, got #{got4}")


# 6. the chassis self-mask
#
# Regression: Friday's own body sits in the beam at raw bearings 89-98 and returns a
# rock-steady 206-219 mm that does not move when she pivots. Bins keep the MINIMUM
# range, so that one return pinned the published scan minimum at 0.21 m -- below
# hard_stop (0.36 m) -- and forward motion was forbidden for an entire run. It was
# visible in blackbox.csv as a lidar_min that never changed while omega did.
th5, slot5 = make_thread()
th5.self_mask_deg = ((88.0, 99.0),)   # half-open [88, 99)
BODY_M = 0.21


def sweep_with_body(th):
    a = 0.0
    while a < 360.0:
        end = min(a + 11.0, 359.99)
        angles = np.linspace(a, end, 12)
        dists = [BODY_M * 1000 if 88.0 <= x < 99.0 else BACKGROUND_M * 1000
                 for x in angles]
        th._handle_frame(build_frame(a, end, dists))
        a += 11.0
    th._handle_frame(build_frame(0.0, 11.0, [BACKGROUND_M * 1000] * 12))


sweep_with_body(th5)
scan5 = slot5.read()[0]
check("masked bearings do not pin the scan minimum",
      float(scan5.min()) > BODY_M + 0.01, f"min={float(scan5.min()):.2f} m")
check("the room behind the mask is still reported",
      abs(float(scan5.min()) - BACKGROUND_M) < 0.05, f"min={float(scan5.min()):.2f} m")

# Same sweep with the mask off must still see the body -- otherwise the test above
# would pass for the wrong reason (e.g. the frames never reaching the bins at all).
# The geometric floor (cfg.lidar_floor) now rejects sub-footprint ranges too, so the
# control has to lower it as well; leaving it in place would catch the body and the
# control would pass whether or not the mask ever did anything.
th6, slot6 = make_thread()
th6.self_mask_deg = ()
floor_was = cfg.lidar_min_valid_range
cfg.lidar_min_valid_range = 0.01
try:
    sweep_with_body(th6)
    scan6 = slot6.read()[0]
finally:
    cfg.lidar_min_valid_range = floor_was
check("without the mask the body IS seen (control)",
      abs(float(scan6.min()) - BODY_M) < 0.02, f"min={float(scan6.min()):.2f} m")

# ...and the geometric floor is a SECOND, independent net under the mask: with the mask
# off and the floor at its real value the body must not reach the scan at all. This is
# the property that makes a leaking or mis-measured self-mask a degradation rather than
# a robot that cannot drive -- the failure mode Harsh actually hit.
th7, slot7 = make_thread()
th7.self_mask_deg = ()
sweep_with_body(th7)
scan7 = slot7.read()[0]
check("the geometric floor catches the body when the mask does not",
      float(scan7.min()) >= cfg.lidar_floor,
      f"min={float(scan7.min()):.2f} m vs floor {cfg.lidar_floor:.2f} m")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)

