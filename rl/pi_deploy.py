#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pi_deploy.py
============
Real-time sensor-fusion inference runtime for the PPO exploration policy trained by
`train.py`, targeting a Raspberry Pi 4B driving a differential-drive robot.

Fuses three sensor modalities into one 10 Hz control loop:

    RPLIDAR (360 deg)  -> policy observation + speed governor   [primary]
    VL53L5CX 8x8 ToF   -> emergency reflex braking              [hard override]
    2x USB cameras     -> visual hazard gate                    [advisory]

WHY EACH SENSOR EXISTS (they are not redundant)
-----------------------------------------------
    LiDAR   sees far and all-round, but only in ONE horizontal plane. It is blind to
            anything below/above that plane, to glass, and to floor drop-offs.
    ToF     sees a dense 8x8 depth patch straight ahead at close range. It catches the
            low obstacle the LiDAR plane passes over -- a shoe, a cable, a step.
    Cameras see texture and geometry the other two cannot: glass doors, stair edges,
            dark voids. Used ONLY to slow down, never to steer (see VisualHazardMonitor).

COMMAND PIPELINE (each stage can only ever REDUCE authority)
------------------------------------------------------------
    policy(obs) -> LidarGovernor -> VisualGate -> ToFReflex -> RateLimiter -> motors
                        |               |            |
                   plane-based      advisory     HARD VETO, own thread,
                   braking law      slowdown     bypasses everything

    The reflex additionally runs in its own high-rate thread with a direct line to
    `MotorDriver.emergency_stop()`, so a close obstacle stops the robot without waiting
    for the next policy tick.

FAIL-SAFE POSTURE
-----------------
    Every sensor slot is timestamped. If a SAFETY-CRITICAL source (LiDAR, ToF) goes
    stale beyond its deadline, the loop stops the robot -- a missing sensor is treated
    as an obstacle, never as "clear". Cameras are advisory: if they go stale the robot
    continues at reduced speed rather than stopping, because a dead USB camera should
    not strand the robot.

HONEST SCOPE -- read before trusting this on hardware
-----------------------------------------------------
  * Everything here is verified ONLY in `--simulate` mode on a dev machine. No part of
    the hardware path (rplidar / VL53L5CX / V4L2 / your motor driver) has been run
    against real devices by me. Treat driver code as a reviewed starting point.
  * `MotorDriver` is ABSTRACT. `NullMotorDriver` only logs. You must implement
    `_apply()` for your hardware, and you must bench-test with wheels off the ground.
  * The visual layer is a calibrated HEURISTIC (edge density, looming, dark-region
    ratio). It is not object detection and it does not know what it is looking at.
    It needs `--calibrate` in YOUR environment and it will still produce false
    positives under changing light. That is why it may only slow the robot down.
  * Dead-reckoned pose drifts. The frontier channel of the observation degrades over
    minutes without an external pose source (see `--pose-source`).

Install (Raspberry Pi OS 64-bit)
--------------------------------
    sudo apt install -y python3-opencv libatlas-base-dev
    pip install onnxruntime numpy rplidar-roboticia vl53l5cx-python pyserial

Usage
-----
    python3 pi_deploy.py --simulate --selftest      # no hardware; validates the pipeline
    python3 pi_deploy.py --simulate --duration 30   # watch the arbitration live
    python3 pi_deploy.py --calibrate 20             # record clear-floor visual baseline
    python3 pi_deploy.py --model exploration_model.onnx
    sudo chrt -f 20 python3 pi_deploy.py ...        # real-time priority (recommended)
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import signal
import statistics
import struct
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

LOGGER = logging.getLogger("pi_deploy")
HERE = Path(__file__).resolve().parent

# Decision provenance (telemetry.py, beside this file). Imported by path so it travels
# with pi_deploy.py and does not depend on the cwd the process was launched from.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import telemetry as tlm  # noqa: E402
from explorer import ExplorationPlanner, steering_bias  # noqa: E402
import cameras as _cameras  # noqa: E402
from narrator import Narrator  # noqa: E402
from recovery import RecoveryController  # noqa: E402
from world import MapperThread, WorldMap  # noqa: E402

# Optional dependencies are imported lazily so `--simulate` runs on a bare machine.
try:
    import onnxruntime as ort
except ImportError:
    ort = None
try:
    import cv2
except ImportError:
    cv2 = None


# ======================================================================================
# 1. Configuration -- MUST mirror the training constants
# ======================================================================================
@dataclass
class RobotConfig:
    """Physical + policy constants.

    CRITICAL: every field down to `omega_max` must match the EnvConfig used in
    `train.py` for the checkpoint you are deploying. A mismatch here does not raise --
    it silently feeds the network out-of-distribution input. `verify_against_model()`
    catches the one mismatch that is mechanically checkable (observation width).
    """

    # ---- must match train.py EnvConfig ----
    n_beams: int = 72
    lidar_max_range: float = 5.00
    robot_radius: float = 0.22
    wheel_base: float = 0.35
    v_max: float = 0.50
    v_min: float = -0.15
    omega_max: float = 1.60
    resolution: float = 0.10
    use_frontier_rays: bool = True

    # ---- control loop ----
    control_hz: float = 10.0            # policy rate; matches training dt = 0.1 s
    reflex_hz: float = 30.0             # reflex thread target rate (ToF-bound in practice)

    # ---- LiDAR safety governor (mirrors SafetyGovernor in train.py) ----
    safe_distance: float = 0.55         # begin braking below this clearance [m]
    collision_margin: float = 0.04
    hard_stop_extra: float = 0.10       # extra margin over robot_radius + collision_margin

    # ---- ToF emergency reflex ----
    tof_enabled: bool = True
    tof_stop_distance: float = 0.30     # [m] any zone closer than this -> emergency stop
    tof_slow_distance: float = 0.60     # [m] begin proportional braking
    tof_min_valid_zones: int = 3        # ignore isolated speckle; need N zones to agree
    tof_max_range: float = 4.00
    # Minimum BUMPER clearance before an emergency stop will still allow rotation.
    #
    # Measured from the bumper, like every other ToF range -- NOT from the chassis
    # centre. The first version of this compared tof_min against robot_radius (0.22,
    # a centre-relative number) and so double-counted the robot's own length: the
    # allowance then never fired, because after braking she rests around 0.215 m and
    # every reflex tick tested "no (too close)" while the policy was asking to turn
    # at 0.6 rad/s.
    #
    # 0.12 m is the right kind of number because a pivot does not translate: a roughly
    # circular chassis rotating about its centre barely changes its distance to
    # anything, so the only real risk is a protruding corner scrubbing something
    # already almost touching the bumper. Forward motion stays vetoed regardless.
    tof_pivot_min_clearance: float = 0.12   # [m] from the BUMPER; below this, freeze

    # ---- visual hazard gate ----
    cam_enabled: bool = True
    cam_indices: tuple[int, int] = (0, 1)
    cam_width: int = 320
    cam_height: int = 240
    cam_fps: int = 15
    visual_hazard_threshold: float = 0.65   # sustained score above this -> full stop
    visual_consec_frames: int = 3           # temporal consistency -> fewer false stops
    visual_min_speed_scale: float = 0.25    # a hazard never brakes below this fraction

    # ---- staleness deadlines [s] ----
    lidar_deadline: float = 0.40        # SAFETY CRITICAL -> stop when exceeded
    tof_deadline: float = 0.50          # SAFETY CRITICAL -> stop when exceeded
    cam_deadline: float = 1.50          # advisory -> degrade, do not stop

    # ---- actuation limits ----
    max_lin_accel: float = 0.8          # [m/s^2] command slew limit -> no wheel slip
    max_ang_accel: float = 4.0          # [rad/s^2]
    # MEASURED ON THIS ROBOT 2026-08-28, at the ceiling duty of 191 (max_duty 0.75).
    # The bumper ToF closed 1.393 -> 0.280 m under power in 2.34 s while driving
    # straight at one surface: 1.113 m / 2.34 s = 0.476 m/s.
    #
    # History, because two wrong values are on record and both get quoted:
    #   0.85  -- train.py's simulator. Never measured on hardware.
    #   0.185 -- friday/state/calibration.json forward_cm_per_s 18.5, but that was
    #            taken at a LOWER effective duty (move_speed 170 x speed_percent 75),
    #            not at the ceiling, so it does not describe this constant.
    #
    # SINGLE-POINT and worth refining: it is one clean run, not a fitted curve, and
    # a loaded brushed motor is not linear. Re-measure with a closing-RATE fit (slope
    # of ToF distance vs time, R^2 > 0.9) in a clear lane of at least 2 m.
    #
    # ⚠ STALE SINCE 2026-09-02. MOTOR_MAX_DUTY was raised 0.75 -> 0.85 with Harsh's
    # explicit approval, moving the ceiling from duty 191 to 217. This constant means
    # "the speed reached at the ceiling duty", so it no longer describes the hardware:
    #     duty = 255 * max_duty * (wheel_speed / wheel_speed_max)
    # asks for ~14% more duty than it did when 0.476 was taken, and Friday therefore
    # moves FASTER than the commanded m/s. The safety chain still bounds her by
    # measured distance, so this is a calibration error and not a collision risk --
    # but the policy's idea of how far it travels per tick is now wrong.
    # Re-measure AT 0.85 (blocker B3) before trusting any commanded velocity.
    wheel_speed_max: float = 0.476      # [m/s] per wheel -- MEASURED AT THE OLD 0.75 CEILING

    # ---- LiDAR chassis self-mask ----
    # Raw sensor bearings (degrees, pre-offset) that look at the robot's own body.
    # Empty for a generic robot; the deployment sets it. See --lidar-self-mask.
    self_mask_deg: tuple[tuple[float, float], ...] = ()
    lidar_offset_deg: float = 0.0        # mounting rotation, set from --lidar-offset-deg

    # ---- exploration map ----
    map_size_m: float = 24.0            # rolling world grid edge (robot starts centred)

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def hard_stop(self) -> float:
        """LiDAR range below which forward motion is forbidden."""
        return self.robot_radius + self.collision_margin + self.hard_stop_extra

    @property
    def obs_dim(self) -> int:
        return (2 if self.use_frontier_rays else 1) * self.n_beams + 2

    def to_si(self, action: np.ndarray) -> tuple[float, float]:
        """Normalised [-1,1]^2 -> (v, omega). Only used if the ONNX outputs normalised."""
        a = np.clip(action, -1.0, 1.0)
        v = float(a[0]) * (self.v_max - self.v_min) / 2.0 + (self.v_max + self.v_min) / 2.0
        return v, float(a[1]) * self.omega_max

    def to_normalised(self, v: float, omega: float) -> np.ndarray:
        """(v, omega) -> normalised, for the `last_action` observation channel."""
        a0 = (v - (self.v_max + self.v_min) / 2.0) / ((self.v_max - self.v_min) / 2.0)
        return np.clip(np.array([a0, omega / self.omega_max], np.float32), -1.0, 1.0)


# ======================================================================================
# 2. Thread-safe sensor plumbing
# ======================================================================================
class SensorSlot:
    """Single-slot latest-value mailbox with a timestamp.

    A queue is the wrong structure for control: a backlog of stale scans is worse than
    no scan at all, because it makes the robot act on where the world USED to be. This
    keeps only the newest sample and reports its age so the loop can reject it.
    """

    def __init__(self, name: str, deadline: float, critical: bool) -> None:
        self.name, self.deadline, self.critical = name, deadline, critical
        self._lock = threading.Lock()
        self._value: Any = None
        self._stamp: float = 0.0
        self._updates = 0

    def publish(self, value: Any) -> None:
        with self._lock:
            self._value, self._stamp = value, time.monotonic()
            self._updates += 1

    def read(self) -> tuple[Any, float]:
        """Returns (value, age_seconds). age = inf when nothing has ever arrived."""
        with self._lock:
            if self._value is None:
                return None, float("inf")
            return self._value, time.monotonic() - self._stamp

    @property
    def updates(self) -> int:
        with self._lock:
            return self._updates

    def is_stale(self) -> bool:
        return self.read()[1] > self.deadline


class SensorThread(threading.Thread):
    """Base class: a daemon loop that publishes into a SensorSlot and never dies quietly."""

    def __init__(self, name: str, slot: SensorSlot) -> None:
        super().__init__(name=name, daemon=True)
        self.slot = slot
        self._stop = threading.Event()
        self.error: Exception | None = None

    def stop(self) -> None:
        self._stop.set()

    def _setup(self) -> None:
        pass

    def _poll(self) -> None:
        raise NotImplementedError

    def _teardown(self) -> None:
        pass

    def run(self) -> None:
        try:
            self._setup()
            while not self._stop.is_set():
                self._poll()
        except Exception as exc:                        # noqa: BLE001
            # Record and exit; the control loop notices via staleness and fails safe.
            self.error = exc
            LOGGER.error("Sensor thread %s died: %s", self.name, exc, exc_info=True)
        finally:
            try:
                self._teardown()
            except Exception:                            # noqa: BLE001
                pass


# ---------------------------------------------------------------------- LiDAR --------
def resample_scan(angles_rad: np.ndarray, ranges_m: np.ndarray, cfg: RobotConfig) -> np.ndarray:
    """Resample a raw LiDAR sweep onto the fixed `n_beams` grid used in training.

    Uses the MINIMUM range within each angular bin, not the mean. For a safety system
    the nearest return in a sector is the number that matters; averaging would smear a
    thin table leg into open space.
    """
    out = np.full(cfg.n_beams, cfg.lidar_max_range, np.float32)
    if angles_rad.size == 0:
        return out
    valid = np.isfinite(ranges_m) & (ranges_m > 0.02) & (ranges_m <= cfg.lidar_max_range)
    if not valid.any():
        return out
    a = np.mod(angles_rad[valid], 2.0 * np.pi)
    r = ranges_m[valid].astype(np.float32)
    idx = np.floor(a / (2.0 * np.pi) * cfg.n_beams).astype(np.int32) % cfg.n_beams
    np.minimum.at(out, idx, r)          # per-bin min, vectorised
    return out


def masked_beam_indices(mask: tuple[tuple[float, float], ...], offset_deg: float,
                        n_beams: int) -> np.ndarray:
    """Which resampled beams fall inside the chassis mask, after the mounting offset.

    A masked bearing has NO DATA, and the LD19 publishes max_range for any empty bin
    -- which is indistinguishable from "nothing within 5 m". The exploration map reads
    that as free space and carves a 5 m corridor straight through the robot's own body
    on every tick. Static geometry, so this is computed once.
    """
    if not mask:
        return np.zeros(0, np.int32)
    out = set()
    for low, high in mask:
        span = (high - low) % 360.0
        # Sample the half-open range finely enough that no beam inside it is missed.
        for k in range(int(span * 4)):
            raw = (low + k * 0.25) % 360.0
            rel = (raw + offset_deg) % 360.0
            out.add(int(rel / 360.0 * n_beams) % n_beams)
    return np.array(sorted(out), np.int32)


def parse_self_mask(spec: str | None) -> tuple[tuple[float, float], ...]:
    """Parse "88:99,350:10" into HALF-OPEN [low, high) raw-bearing ranges.

    Half-open because the chassis is a continuous wedge and the LiDAR reports
    continuous angles: a bearing bin that merely straddles the edge still carries
    body returns. Empty string disables the mask.
    """
    if not spec or not spec.strip():
        return ()
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        low, _, high = part.partition(":")
        out.append((float(low) % 360.0, float(high) % 360.0))
    return tuple(out)


class RPLidarThread(SensorThread):
    """RPLIDAR A1/A2 reader. Publishes an (n_beams,) float32 array of metres.

    Beam 0 points along the robot's heading and beams advance counter-clockwise, which
    is the convention `train.py` used. `angle_offset_deg` compensates for how the unit
    is physically bolted on -- get this wrong and the policy's world is rotated.
    """

    def __init__(self, slot: SensorSlot, cfg: RobotConfig, port: str,
                 angle_offset_deg: float = 0.0, baud: int = 115200):
        super().__init__("lidar", slot)
        self.cfg, self.port, self.baud = cfg, port, baud
        self.offset = math.radians(angle_offset_deg)
        self._lidar = None

    def _setup(self) -> None:
        from rplidar import RPLidar          # imported here so --simulate needs no driver

        # Baud MUST match the model: A1/A1M8 = 115200, A3/S1 = 256000 (A2 varies).
        # A mismatch does not raise a clear error -- it just times out looking exactly
        # like a dead or wrongly-chosen port, which is why it is exposed as a flag.
        self._lidar = RPLidar(self.port, baudrate=self.baud, timeout=3)
        info = self._lidar.get_info()
        LOGGER.info("RPLIDAR connected on %s @ %d: %s", self.port, self.baud, info)
        health = self._lidar.get_health()
        if health and str(health[0]).lower() != "good":
            LOGGER.warning("RPLIDAR health is %s -- readings may be unreliable", health)
        self._lidar.start_motor()

    def _poll(self) -> None:
        for sweep in self._lidar.iter_scans(max_buf_meas=3000):
            if self._stop.is_set():
                return
            arr = np.asarray(sweep, dtype=np.float64)          # (N, 3): quality, angle_deg, dist_mm
            ang = np.radians(arr[:, 1]) + self.offset
            dist = arr[:, 2] / 1000.0
            self.slot.publish(resample_scan(ang, dist, self.cfg))

    def _teardown(self) -> None:
        if self._lidar is not None:
            self._lidar.stop()
            self._lidar.stop_motor()
            self._lidar.disconnect()


def _make_ld19_crc_table() -> bytes:
    """CRC-8, polynomial 0x4D, init 0x00, no reflection -- as used by LDROBOT."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ 0x4D) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return bytes(table)


_LD19_CRC_TABLE = _make_ld19_crc_table()


def ld19_crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc = _LD19_CRC_TABLE[(crc ^ b) & 0xFF]
    return crc


class LD19Thread(SensorThread):
    """LDROBOT LD06 / LD19 / STL-19P reader. Publishes an (n_beams,) array of metres.

    Unlike the RPLIDAR, this sensor is push-only: it streams the moment it has power
    and accepts no commands, which is exactly why probing it with RPLIDAR's GET_INFO
    returned mid-stream scan bytes instead of a descriptor.

    Frame, 47 bytes at 230400 8N1:
        [0]     0x54                header
        [1]     0x2C                VerLen (3 bits type | 5 bits point count = 12)
        [2:4]   speed               u16, degrees/second
        [4:6]   start_angle         u16, hundredths of a degree
        [6:42]  12 x {dist u16 mm, intensity u8}
        [42:44] end_angle           u16, hundredths of a degree
        [44:46] timestamp           u16, ms
        [46]    crc8

    One frame is only ~12 points of a sweep, so points are accumulated into per-beam
    bins and published as a complete scan when the angle wraps past 360 deg. Each bin
    keeps the MINIMUM range seen, matching `resample_scan` -- for obstacle avoidance the
    nearest return in a sector is the one that matters.
    """

    HEADER, VERLEN, FRAME_LEN, POINTS = 0x54, 0x2C, 47, 12

    def __init__(self, slot: SensorSlot, cfg: RobotConfig, port: str,
                 angle_offset_deg: float = 0.0, baud: int = 230400,
                 invert: bool = False, ignore_crc: bool = False,
                 self_mask_deg: tuple[tuple[float, float], ...] | None = None) -> None:
        super().__init__("lidar", slot)
        self.cfg, self.port, self.baud = cfg, port, baud
        self.offset_deg = angle_offset_deg
        self.self_mask_deg = (cfg.self_mask_deg if self_mask_deg is None
                              else tuple(self_mask_deg))
        self.invert = invert
        self.ignore_crc = ignore_crc
        self._ser = None
        self._buf = bytearray()
        self._bins = np.full(cfg.n_beams, np.inf, np.float32)
        self._prev_start: float | None = None
        self._frames = self._crc_ok = self._scans = self._reconnects = 0
        self._read_errors = self._soft_retries = 0

    def _setup(self) -> None:
        import serial

        # exclusive=True so a second process cannot silently steal half the byte stream
        # -- that failure mode looks exactly like corrupt frames and wasted us an hour.
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5, exclusive=True)
        except TypeError:                            # older pyserial / non-POSIX
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
        self._ser.reset_input_buffer()
        LOGGER.info("LD19/STL-19P opened on %s @ %d baud", self.port, self.baud)
        if self.ignore_crc:
            LOGGER.warning("LD19 CRC checking DISABLED -- diagnostic only, corrupt frames "
                           "will be treated as real range data. Do not drive like this.")

    # A USB re-enumeration takes a second or two, sometimes longer under load. Five
    # attempts at a flat 0.5 s only covered a 2.5 s window, which is shorter than the
    # event we are trying to survive -- so: more attempts, with growing backoff.
    MAX_RECONNECTS = 10

    def _reconnect(self) -> None:
        """Close and reopen the port after a read fault.

        USB serial on a Pi glitches: a brownout, a jostled connector, or a re-enumeration
        raises SerialException and would otherwise kill this thread for good, leaving the
        robot permanently blind. Recovering is safe because it does NOT bypass any safety
        logic -- while we are reconnecting no scans are published, the slot goes stale,
        and the control loop stops the robot on its own. After MAX_RECONNECTS consecutive
        failures we give up for real and let the thread die, because at that point the
        hardware is genuinely gone and pretending otherwise would be worse.
        """
        import serial

        self._reconnects += 1
        if self._reconnects > self.MAX_RECONNECTS:
            raise RuntimeError(
                f"LD19: {self._reconnects} consecutive reconnect attempts failed on "
                f"{self.port}. Check the cable/power, and that no other process holds the "
                f"port (pkill -f pi_deploy.py; sudo fuser -k {self.port})."
            )
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:                            # noqa: BLE001
            pass
        self._ser = None
        time.sleep(min(0.5 * self._reconnects, 3.0))     # growing backoff, capped
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
            self._ser.reset_input_buffer()
            self._buf.clear()
            self._prev_start = None                  # partial revolution is now invalid
            self._bins[:] = np.inf
            LOGGER.info("LD19 reconnected to %s (attempt %d)", self.port, self._reconnects)
        except Exception as exc:                     # noqa: BLE001
            # Reopen failure must CONSUME a retry, not escape and kill the thread on the
            # first try -- the port often needs a moment after a USB glitch.
            LOGGER.warning("LD19 reopen failed (%d/%d): %s",
                           self._reconnects, self.MAX_RECONNECTS, exc)

    # A read fault while the device is still enumerated is usually transient (a UART
    # overrun, a short read the driver reports as "no data"). Tearing the port down and
    # sleeping seconds for that costs far more data than the fault itself. So: try a
    # cheap recovery first -- flush and carry on -- and only escalate to a full reopen
    # once the cheap path has clearly failed.
    MAX_SOFT_RETRIES = 20

    def _poll(self) -> None:
        try:
            if self._ser is None:                    # previous reopen failed; retry
                self._reconnect()
                return
            data = self._ser.read(512)
        except RuntimeError:
            raise                                    # retry budget exhausted -> let it die
        except Exception as exc:                     # noqa: BLE001
            self._read_errors += 1
            self._soft_retries += 1
            if self._soft_retries <= self.MAX_SOFT_RETRIES:
                if self._read_errors in (1, 50) or self._read_errors % 500 == 0:
                    LOGGER.warning("LD19 transient read error #%d (%s) -- flushing and "
                                   "continuing.", self._read_errors, exc)
                try:
                    self._ser.reset_input_buffer()
                except Exception:                    # noqa: BLE001
                    pass
                time.sleep(0.02)
                return
            LOGGER.warning("LD19 read failed %d times in a row (%s) -- reopening %s.",
                           self._soft_retries, exc, self.port)
            self._soft_retries = 0
            self._reconnect()
            return
        self._soft_retries = 0                       # a completed read clears the streak
        if not data:
            return
        self._reconnects = 0                         # a good read clears the failure streak
        self._buf.extend(data)
        if len(self._buf) > 8192:                    # bound growth if we fall behind
            del self._buf[:-4096]

        needle = bytes((self.HEADER, self.VERLEN))
        while True:
            i = self._buf.find(needle)
            if i < 0:
                del self._buf[:-1]                   # keep 1 byte: header may straddle reads
                return
            if len(self._buf) - i < self.FRAME_LEN:
                del self._buf[:i]                    # wait for the rest of the frame
                return
            frame = bytes(self._buf[i:i + self.FRAME_LEN])
            del self._buf[:i + self.FRAME_LEN]
            self._handle_frame(frame)

    def _handle_frame(self, f: bytes) -> None:
        self._frames += 1
        if not self.ignore_crc and ld19_crc8(f[:-1]) != f[-1]:
            # If EVERY frame fails, the problem is systematic (wrong baud or wrong CRC
            # polynomial), not noise -- say so instead of silently starving the loop.
            if self._frames >= 200 and self._crc_ok == 0 and self._frames % 200 == 0:
                LOGGER.error(
                    "LD19: %d frames read, ZERO passed CRC. Header is being found, so the "
                    "baud is probably right but the frames are corrupt. Re-check --lidar-baud "
                    "(should be 230400); to inspect anyway, re-run with --ld19-ignore-crc "
                    "(diagnostic only).", self._frames,
                )
            return
        self._crc_ok += 1

        start_deg = struct.unpack_from("<H", f, 4)[0] / 100.0
        end_deg = struct.unpack_from("<H", f, 42)[0] / 100.0

        raw = np.frombuffer(f, dtype=np.uint8, count=36, offset=6)
        dist_mm = raw[0::3].astype(np.uint16) | (raw[1::3].astype(np.uint16) << 8)
        dist_m = dist_mm.astype(np.float32) / 1000.0

        span = (end_deg - start_deg) % 360.0
        angles = start_deg + np.arange(self.POINTS, dtype=np.float32) * (span / (self.POINTS - 1))

        if self.invert:
            angles = -angles

        # A bearing that looks at the robot's own chassis is not the room. Bins keep
        # the MINIMUM range, so ONE unmasked body return pins the scan minimum below
        # hard_stop and forward motion is forbidden for the whole run -- it never
        # moves when the robot pivots, so nothing ever clears it. Masked in RAW
        # sensor space, BEFORE the mounting offset, because that is where it was
        # measured.
        body = np.zeros(angles.shape, dtype=bool)
        if self.self_mask_deg:
            raw = np.mod(angles, 360.0)
            for low, high in self.self_mask_deg:
                # HALF-OPEN [low, high). The chassis occupies a continuous angular
                # wedge, and a bearing whose 1-degree bin merely CONTAINS the edge
                # still carries body returns. Inclusive [88, 98] left the whole bin
                # at bearing 98 (raw 98.0-99.0, body returns down to 0.209 m)
                # unmasked, so the body leaked on 12% of scans -- and because the
                # governor takes the minimum over the WHOLE scan, each leak blocked
                # ALL forward motion. Intermittently, which is worse than never.
                body |= (((raw >= low) & (raw < high)) if low <= high
                         else ((raw >= low) | (raw < high)))

        angles = np.mod(angles + self.offset_deg, 360.0)

        # 0 mm means "no valid return" on this sensor -- never treat it as a contact.
        good = (dist_mm > 0) & (dist_m <= self.cfg.lidar_max_range) & ~body
        if good.any():
            idx = (angles[good] / 360.0 * self.cfg.n_beams).astype(np.int32) % self.cfg.n_beams
            np.minimum.at(self._bins, idx, dist_m[good])

        # A wrap past 0 deg means one full revolution is complete -> publish it.
        if self._prev_start is not None and start_deg < self._prev_start - 180.0:
            scan = np.where(np.isinf(self._bins), self.cfg.lidar_max_range, self._bins)
            self.slot.publish(np.clip(scan, 0.0, self.cfg.lidar_max_range).astype(np.float32))
            self._bins[:] = np.inf
            self._scans += 1
            if self._scans % 100 == 0:
                LOGGER.debug("LD19: %d scans, CRC pass %.1f%%",
                             self._scans, 100.0 * self._crc_ok / max(self._frames, 1))
        self._prev_start = start_deg

    def _teardown(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:                        # noqa: BLE001
                pass


# ------------------------------------------------------------------------ ToF --------
# MEASURED on this robot (friday/config.py, friday/sensors.py). The bumper sensor is
# NOT the VL53L5CX-over-I2C this file was written for -- it is a DFRobot 8x8 matrix ToF
# behind a Raspberry Pi Pico that presents as USB serial and streams ASCII rows:
#     y0:d0,d1,...,d7,\n   ... one line per row, TOF_ROWS lines per frame, mm
TOF_ROWS, TOF_COLS = 8, 8
TOF_NO_RETURN_MM = 4000

# THE BOTTOM ROWS LOOK AT THE GROUND. Rows at or below this index are floor, not room.
# Reading them as obstacles is why Friday once could not drive forward at all, in any
# room, forever: the nearest return in the frame was always the floor ~12 cm ahead.
TOF_FLOOR_FROM_ROW = 4
# What each ground row reads over empty floor, row 4 first. Re-measure if the bumper
# is re-seated (friday/scripts/calibrate_tof_floor.py).
TOF_FLOOR_PROFILE_MM = (295, 223, 161, 120)
# A ground row this much SHORTER than its pitch-corrected floor distance is an object
# standing on the floor. Comfortably above the +-27 mm of noise on the worst row.
TOF_FLOOR_OBSTACLE_MM = 45
# The chassis noses down as the motors take up and every ground row reads short
# together. Pitch scales all four rows; an object shortens only the rows it intercepts
# -- so the common factor is the MEDIAN ratio across the ground cells, and it is
# clamped so a surface filling the lower field of view cannot normalise itself flat.
TOF_FLOOR_PITCH_MIN, TOF_FLOOR_PITCH_MAX = 0.80, 1.25
# How far a ground row may drift before the floor is no longer "intact". Beyond this
# the answer is cannot-see, never clear -- a floor that stopped reading is a cliff.
TOF_FLOOR_INTACT_MM = 70
# ~0.7% of frames carry a spurious 250-315 mm return in the centre cells just above
# the horizon -- SPAD crosstalk echoing the strong floor return below. One of them was
# enough to brake the robot 0.2 s into a move. A real object is in every frame; an
# artefact cannot move a median.
TOF_CORROBORATE_FRAMES = 3


def tof_obstacles_mm(frame: list[list[int]]) -> tuple[np.ndarray, bool]:
    """One raw 8x8 frame -> (per-cell obstacle range in mm, floor_intact).

    Cells that are not an obstacle come back as TOF_NO_RETURN_MM, so the array can be
    fed straight to ToFReflex without the floor tripping it. Ported from
    friday/sensors.py `_forward_from_frame`; every rule below cost a real debugging
    session and the reasons are recorded there.
    """
    horizon = TOF_FLOOR_FROM_ROW
    profile = TOF_FLOOR_PROFILE_MM
    out = np.full((TOF_ROWS, TOF_COLS), float(TOF_NO_RETURN_MM), np.float32)

    # --- pitch, from the ground rows, before anything is called an obstacle ---
    ground = frame[horizon:horizon + len(profile)]
    ratios = [v / profile[off]
              for off, row in enumerate(ground) for v in row
              if 0 < v < TOF_NO_RETURN_MM]
    pitch = 1.0
    if len(ratios) >= 8:
        pitch = min(max(statistics.median(ratios), TOF_FLOOR_PITCH_MIN),
                    TOF_FLOOR_PITCH_MAX)

    # --- ground rows: only shortfalls against the stretched profile ---
    floor_intact = True
    for off, row in enumerate(ground):
        expected = profile[off] * pitch
        for col, v in enumerate(row):
            if not 0 < v < TOF_NO_RETURN_MM:
                floor_intact = False                     # the floor stopped reading
                continue
            if v < expected - TOF_FLOOR_OBSTACLE_MM:
                out[horizon + off, col] = float(v)       # an occlusion IS the object
            elif abs(v - expected) > TOF_FLOOR_INTACT_MM:
                floor_intact = False                     # further than floor: a drop

    # --- ahead rows: every return is a thing, minus the horizon-row echo ---
    echo = profile[0] if profile else None
    ahead: list[tuple[int, int, int]] = []
    for r, row in enumerate(frame[:horizon]):
        is_horizon = r == horizon - 1
        for col, v in enumerate(row):
            if not 0 < v < TOF_NO_RETURN_MM:
                continue
            if is_horizon and echo is not None and abs(v - echo) <= TOF_FLOOR_INTACT_MM:
                continue                                 # SPAD crosstalk, not an object
            ahead.append((r, col, v))

    # An INTACT floor is positive evidence the near path is empty: an object standing
    # nearer than row 4's intercept would have occluded the ground rows first. So an
    # ahead-row return closer than the floor we can still see is multipath.
    # THIS FILTER IS FOR THE AHEAD ROWS ONLY. Applying it to the ground rows deleted
    # every genuine near obstacle -- an occlusion reads SHORT, which is exactly what
    # the filter discarded -- and the robot drove into something until the reflex
    # caught it at 16 mm.
    if floor_intact and ahead:
        reach = profile[0] * pitch if profile else 0.0
        ahead = [(r, c, v) for r, c, v in ahead if v >= reach]
    for r, c, v in ahead:
        out[r, c] = float(v)

    return out, floor_intact


class DFRobotToFThread(SensorThread):
    """DFRobot 8x8 matrix ToF over USB serial. Publishes an (8, 8) float32 of METRES.

    Publishes OBSTACLE ranges, not raw ranges: a cell looking at clean floor reads
    tof_max_range, so the reflex's k-th-smallest rule sees only real things. Raw
    ranges here would put the floor (120-295 mm) in every frame and the robot would
    emergency-stop forever, which is the exact bug this logic exists to prevent.

    Corroborated across TOF_CORROBORATE_FRAMES frames by a per-cell median, because
    one frame is not enough to stop the robot (see TOF_CORROBORATE_FRAMES).

    Publishes NOTHING when the floor is not intact and no obstacle is visible -- that
    is cannot-see, and the slot then goes stale, and a stale critical sensor stops the
    robot. Never publish a clear frame to represent blindness.
    """

    def __init__(self, slot: SensorSlot, cfg: RobotConfig, port: str,
                 baud: int = 115200) -> None:
        super().__init__("tof", slot)
        self.cfg, self.port, self.baud = cfg, port, baud
        self._ser = None
        self._buf = bytearray()
        self._partial: dict[int, list[int]] = {}
        self._recent: deque = deque(maxlen=TOF_CORROBORATE_FRAMES)
        self._frames = self._blind = 0

    def _setup(self) -> None:
        import serial
        try:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5, exclusive=True)
        except TypeError:
            self._ser = serial.Serial(self.port, self.baud, timeout=0.5)
        self._ser.reset_input_buffer()
        LOGGER.info("DFRobot 8x8 ToF opened on %s @ %d baud", self.port, self.baud)

    def _poll(self) -> None:
        # Chunked reads, NOT readline(): pyserial reads one byte per syscall there,
        # and at 8 rows x 15 fps that measured 13% of a core on this Pi.
        try:
            data = self._ser.read(self._ser.in_waiting or 1)
        except Exception as exc:                          # noqa: BLE001
            LOGGER.warning("ToF read failed (%s)", exc)
            time.sleep(0.05)
            return
        if not data:
            return
        self._buf.extend(data)
        if len(self._buf) > 4096:                         # a stream with no newlines
            del self._buf[:-256]
        while True:
            cut = self._buf.find(b"\n")
            if cut < 0:
                return
            raw, self._buf = bytes(self._buf[:cut]), self._buf[cut + 1:]
            self._ingest(raw)

    def _ingest(self, raw: bytes) -> None:
        """One "yN:d0,...,d7," line. TOF_ROWS of them make a frame."""
        line = raw.decode(errors="replace").strip()
        # The first line after a connect is usually joined mid-stream and has no
        # "yN:" prefix. Dropping it is correct.
        if not line.startswith("y") or ":" not in line:
            return
        head, _, body = line.partition(":")
        try:
            row = int(head[1:])
        except ValueError:
            return
        values = [int(v) for v in body.split(",") if v.strip().isdigit()]
        if row < 0 or row >= TOF_ROWS or len(values) != TOF_COLS:
            return
        self._partial[row] = values
        if len(self._partial) < TOF_ROWS:
            return

        frame = [self._partial[r] for r in range(TOF_ROWS)]
        self._partial = {}
        self._frames += 1
        obstacles, intact = tof_obstacles_mm(frame)
        self._recent.append((obstacles, intact))

        # Cannot-see is the majority view -> publish nothing and let the slot go
        # stale. A stale critical sensor stops the robot; a clear frame would not.
        blind = sum(1 for _, ok in self._recent if not ok)
        anything = min(float(o.min()) for o, _ in self._recent) < TOF_NO_RETURN_MM
        if blind * 2 > len(self._recent) and not anything:
            self._blind += 1
            return

        # Per-cell median across frames: a real object is in all of them, a one-frame
        # multipath artefact cannot move a median.
        stack = np.stack([o for o, _ in self._recent])
        med = np.median(stack, axis=0).astype(np.float32) / 1000.0
        self.slot.publish(np.minimum(med, self.cfg.tof_max_range))

    def _teardown(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:                             # noqa: BLE001
                pass


class VL53L5CXThread(SensorThread):
    """8x8 multizone ToF reader. Publishes an (8, 8) float32 array of metres.

    LATENCY REALITY: the VL53L5CX caps at ~15 Hz in 8x8 mode (60 Hz only at 4x4). So
    the reflex cannot react faster than ~67 ms no matter what `reflex_hz` says. At
    v_max = 0.5 m/s that is ~3.4 cm of travel before the sample even arrives, before
    any braking. `tof_stop_distance` must stay comfortably above that -- the 0.30 m
    default leaves large margin. Verify the timing against your own datasheet revision.
    """

    def __init__(self, slot: SensorSlot, cfg: RobotConfig) -> None:
        super().__init__("tof", slot)
        self.cfg = cfg
        self._dev = None

    def _setup(self) -> None:
        import vl53l5cx_ctypes as vl53l5cx   # driver package name varies by distro

        self._dev = vl53l5cx.VL53L5CX()
        self._dev.set_resolution(8 * 8)
        self._dev.set_ranging_frequency_hz(15)
        self._dev.start_ranging()
        LOGGER.info("VL53L5CX ranging at 8x8 / 15 Hz")

    def _poll(self) -> None:
        if not self._dev.data_ready():
            time.sleep(0.002)
            return
        data = self._dev.get_data()
        mm = np.asarray(data.distance_mm, dtype=np.float32).reshape(8, 8)
        status = np.asarray(data.target_status, dtype=np.int32).reshape(8, 8)
        depth = mm / 1000.0
        # Status 5 (and 9) are the valid measurements; everything else is untrustworthy
        # and is pushed to max range rather than being believed as a close reading.
        depth[~np.isin(status, (5, 9))] = self.cfg.tof_max_range
        self.slot.publish(np.clip(depth, 0.0, self.cfg.tof_max_range))

    def _teardown(self) -> None:
        if self._dev is not None:
            self._dev.stop_ranging()


# --------------------------------------------------------------------- camera --------
class CameraThread(SensorThread):
    """Self-healing USB camera grabber. Publishes a small grayscale frame (uint8).

    Low resolution and MJPG on purpose: two USB cameras on one Pi share USB 2.0
    bandwidth, and raw YUYV at 640x480 from both will saturate the bus and stall them.

    IDENTIFIED BY SERIAL, NEVER BY INDEX. `cam_indices = (0, 1)` opened the up
    camera's METADATA node as the second camera and never touched the down camera at
    all -- every UVC device on this machine occupies two consecutive video numbers,
    so index arithmetic cannot work here. See cameras.py.

    RETRIES FOREVER, unlike LD19Thread which gives up after MAX_RECONNECTS. The
    inversion is deliberate and follows from criticality: the LiDAR is safety-critical,
    so when it is genuinely gone the right move is to die and let the control loop stop
    the robot on staleness. A camera is ADVISORY -- it may slow the robot, never steer
    it, and its absence is already handled by the fail-open gate. So a camera that
    vanishes must never kill anything, and one that comes back must be picked up again.

    That is not hypothetical here. MEASURED on this robot: the down camera
    (349c:2317 on usb 1-1.1.4) re-enumerates repeatedly -- "USB disconnect" then a new
    device number, minutes apart, unprompted. The root cause is physical (that hub
    leg, its cable, or its power) and software cannot fix it; what software can do is
    come back every time without a restart, and say honestly how often it happened.
    """

    # Backoff between reopen attempts, seconds. Capped so a camera that returns after
    # twenty minutes is still picked up promptly rather than on an ever-lengthening
    # schedule, and floored so a permanently absent device does not spin a core.
    RETRY_MIN_S = 1.0
    RETRY_MAX_S = 10.0

    def __init__(self, slot: SensorSlot, cfg: RobotConfig, role: str,
                 serial: str | None = None, path: str | None = None) -> None:
        super().__init__(f"cam_{role}", slot)
        self.cfg, self.role, self.serial = cfg, role, serial
        self.explicit_path = path
        self._cap = None
        self.device: str | None = None
        self.reconnects = 0
        self.open_failures = 0
        self._next_try = 0.0
        self._backoff = self.RETRY_MIN_S

    # ---- opening -----------------------------------------------------------------
    def _resolve_path(self) -> str | None:
        """Where is this camera right now? Re-asked on every reopen, never cached.

        A device that re-enumerates comes back under a DIFFERENT /dev/videoN. Caching
        the path is the same mistake as caching the index, one level up.
        """
        if self.explicit_path:
            return self.explicit_path
        if not self.serial:
            return None
        found = _cameras.resolve({self.role: self.serial})
        d = found.get(self.role)
        return d.path if d else None

    def _try_open(self) -> bool:
        if cv2 is None:
            return False
        path = self._resolve_path()
        if path is None:
            return False
        cap = cv2.VideoCapture(path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap.release()
            return False
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cfg.cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg.cam_height)
        cap.set(cv2.CAP_PROP_FPS, self.cfg.cam_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always the freshest frame
        # Warm up. Friday's own config.py has CAM_WARMUP_FRAMES = 10 for this reason:
        # the first reads off a freshly opened UVC device commonly fail, and treating
        # that as "camera broken" is how a working camera gets written off.
        for _ in range(getattr(self.cfg, "cam_warmup_frames", 10)):
            if cap.read()[0]:
                break
        self._cap = cap
        self.device = path
        self._backoff = self.RETRY_MIN_S
        return True

    def _drop(self) -> None:
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:                            # noqa: BLE001
            pass
        self._cap = None
        self.device = None
        self._next_try = time.monotonic() + self._backoff
        self._backoff = min(self._backoff * 2.0, self.RETRY_MAX_S)

    # ---- SensorThread hooks ------------------------------------------------------
    def _setup(self) -> None:
        # Never raises. A camera absent at startup is a degraded run, not a failed one,
        # and this thread's whole job is to keep looking.
        if self._try_open():
            LOGGER.info("camera %s opened on %s (%dx%d)", self.role, self.device,
                        self.cfg.cam_width, self.cfg.cam_height)
        else:
            self.open_failures += 1
            LOGGER.warning("camera %s not present at startup (serial %s) -- will keep "
                           "retrying; the visual gate stays fail-open meanwhile.",
                           self.role, self.serial)
            self._next_try = time.monotonic() + self._backoff

    def _poll(self) -> None:
        if self._cap is None:
            if time.monotonic() < self._next_try:
                time.sleep(0.1)
                return
            if self._try_open():
                self.reconnects += 1
                LOGGER.info("camera %s came back on %s (reconnect %d)",
                            self.role, self.device, self.reconnects)
            else:
                self.open_failures += 1
                self._next_try = time.monotonic() + self._backoff
                self._backoff = min(self._backoff * 2.0, self.RETRY_MAX_S)
            return
        try:
            ok, frame = self._cap.read()
        except Exception:                            # noqa: BLE001
            ok, frame = False, None
        if not ok or frame is None:
            # A read failure on an enumerated device is usually transient; a read
            # failure on a device that has been unplugged never recovers. Telling them
            # apart cheaply: ask where the camera is now. If it is gone, drop and wait.
            if self._resolve_path() is None:
                LOGGER.warning("camera %s disappeared from the bus -- dropping and "
                               "waiting for it to come back.", self.role)
                self._drop()
            else:
                time.sleep(0.02)
            return
        self.slot.publish(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    def _teardown(self) -> None:
        if self._cap is not None:
            self._cap.release()


# ======================================================================================
# 3. Simulated sensors (no hardware) -- this is what makes the pipeline testable
# ======================================================================================
class SimulatedWorld:
    """Tiny kinematic world so the full control pipeline can be exercised off-robot.

    Not a physics engine -- just enough to produce coherent LiDAR/ToF/camera streams
    that respond to commanded motion, including a scripted close-approach event that
    forces the reflex layer to fire.
    """

    def __init__(self, cfg: RobotConfig, seed: int = 0) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.pose = np.zeros(3, np.float32)
        self.t0 = time.monotonic()
        # Random box obstacles in a 12 m room, robot starting at the centre.
        self.walls = 6.0
        self.boxes = self.rng.uniform(-4.5, 4.5, size=(8, 2)).astype(np.float32)

    def step(self, v: float, omega: float, dt: float) -> None:
        x, y, th = self.pose
        if abs(omega) < 1e-6:
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
        else:
            th_new = th + omega * dt
            x += (v / omega) * (math.sin(th_new) - math.sin(th))
            y -= (v / omega) * (math.cos(th_new) - math.cos(th))
            th = th_new
        self.pose[:] = (x, y, math.atan2(math.sin(th), math.cos(th)))

    def lidar(self) -> np.ndarray:
        """Analytic ray-cast against the room walls and circular box obstacles."""
        c = self.cfg
        x, y, th = self.pose
        ang = np.linspace(0, 2 * np.pi, c.n_beams, endpoint=False) + th
        dx, dy = np.cos(ang), np.sin(ang)
        rng = np.full(c.n_beams, c.lidar_max_range, np.float64)
        for i in range(c.n_beams):
            t = np.inf
            for wall, comp, d in ((self.walls, dx[i], x), (-self.walls, dx[i], x),
                                  (self.walls, dy[i], y), (-self.walls, dy[i], y)):
                if abs(comp) > 1e-9:
                    tt = (wall - d) / comp
                    if 0 < tt < t:
                        t = tt
            for bx, by in self.boxes:                    # circles of radius 0.4 m
                ox, oy = bx - x, by - y
                proj = ox * dx[i] + oy * dy[i]
                if proj <= 0:
                    continue
                perp2 = ox * ox + oy * oy - proj * proj
                if perp2 < 0.16:
                    tt = proj - math.sqrt(0.16 - perp2)
                    if 0 < tt < t:
                        t = tt
            rng[i] = min(t, c.lidar_max_range)
        noise = self.rng.normal(0, 0.015, c.n_beams)
        return np.clip(rng + noise, 0.0, c.lidar_max_range).astype(np.float32)

    def tof(self) -> np.ndarray:
        """8x8 patch ahead. Scripted: a low obstacle appears at t = 8-12 s.

        This is the event the LiDAR CANNOT see (below its plane) -- exactly the case
        the ToF reflex exists for.
        """
        forward = float(self.lidar()[0])
        depth = np.full((8, 8), min(forward, self.cfg.tof_max_range), np.float32)
        elapsed = time.monotonic() - self.t0
        if 8.0 < elapsed < 12.0:
            depth[4:8, 2:6] = 0.22          # low obstacle, well inside the stop distance
        return depth + self.rng.normal(0, 0.005, (8, 8)).astype(np.float32)

    def frame(self) -> np.ndarray:
        """Synthetic grayscale floor, plus a dark blob during the obstacle event.

        Deliberately a SMOOTH lit floor (gradient + mild sensor noise) rather than pure
        noise: uniform noise has maximal edge density and would make every frame look
        like a hazard, which tests nothing. This gives the calibrator a realistic
        low-edge baseline that the dark blob can then exceed.
        """
        h, w = self.cfg.cam_height, self.cfg.cam_width
        gradient = np.linspace(150.0, 95.0, h, dtype=np.float32)[:, None]
        img = gradient + self.rng.normal(0.0, 2.5, (h, w)).astype(np.float32)
        elapsed = time.monotonic() - self.t0
        if 8.0 < elapsed < 12.0:
            img[int(h * 0.62):, int(w * 0.30):int(w * 0.70)] = 22.0   # dark void / obstacle
        return img.clip(0, 255).astype(np.uint8)


class SimSensorThread(SensorThread):
    """Publishes from a SimulatedWorld at a fixed rate."""

    def __init__(self, name: str, slot: SensorSlot, producer: Callable[[], Any], hz: float) -> None:
        super().__init__(name, slot)
        self.producer, self.period = producer, 1.0 / hz

    def _poll(self) -> None:
        self.slot.publish(self.producer())
        time.sleep(self.period)


# ======================================================================================
# 4. Local exploration map -- reproduces the training observation's frontier channel
# ======================================================================================
class LocalExplorationMap:
    """Occupancy + known-space grid with dead-reckoned pose, mirroring `train.py`.

    The policy was trained on TWO ray arrays: LiDAR ranges, and distance-to-nearest-
    unknown-cell along each beam. The second one carries the agent's memory of where it
    has been. Feeding it zeros (or max range) at deployment puts the network far outside
    its training distribution, so it has to be reconstructed here with the same geometry
    and the same units as the simulator used.

    DRIFT WARNING: pose comes from integrating COMMANDED velocities. There is no
    encoder or IMU feedback, so error accumulates -- wheel slip and turning error are
    unobservable. Over a few minutes the frontier channel degrades. Feed real odometry
    through `set_pose()` whenever you have it (`--pose-source`).
    """

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg
        self.n = int(round(cfg.map_size_m / cfg.resolution))
        self.occ = np.zeros((self.n, self.n), np.uint8)
        self.known = np.zeros((self.n, self.n), np.uint8)
        self.pose = np.array([cfg.map_size_m / 2, cfg.map_size_m / 2, 0.0], np.float32)

        self._beam_angles = np.linspace(0, 2 * np.pi, cfg.n_beams, endpoint=False).astype(np.float32)
        self._step = cfg.resolution * 0.5
        self._n_samples = int(math.ceil(cfg.lidar_max_range / self._step))
        self._depths = (np.arange(1, self._n_samples + 1, dtype=np.float32) * self._step)
        self.explored_cells = 0
        # Beams that look at the chassis carry no information about the room. They are
        # excluded from free-marking, from hit-marking and from the frontier answer.
        self.blind_beams = np.zeros(0, np.int32)

    def set_pose(self, x: float, y: float, theta: float) -> None:
        self.pose[:] = (x, y, theta)

    def dead_reckon(self, v: float, omega: float, dt: float) -> None:
        x, y, th = (float(p) for p in self.pose)
        if abs(omega) < 1e-6:
            x += v * math.cos(th) * dt
            y += v * math.sin(th) * dt
        else:
            th_new = th + omega * dt
            x += (v / omega) * (math.sin(th_new) - math.sin(th))
            y -= (v / omega) * (math.cos(th_new) - math.cos(th))
            th = th_new
        # Clamp inside the grid: running off the edge would corrupt the frontier channel.
        m = self.cfg.map_size_m - 0.5
        self.pose[:] = (min(max(x, 0.5), m), min(max(y, 0.5), m),
                        math.atan2(math.sin(th), math.cos(th)))

    def update_and_frontier(self, scan_m: np.ndarray) -> np.ndarray:
        """Integrate a scan into the map and return the frontier ranges [m].

        Single vectorised pass shared by both products, exactly as in training.
        """
        c = self.cfg
        x, y, th = (float(p) for p in self.pose)
        ang = self._beam_angles + th
        px = x + np.cos(ang)[:, None] * self._depths[None, :]
        py = y + np.sin(ang)[:, None] * self._depths[None, :]
        cc = np.clip((px / c.resolution).astype(np.int32), 0, self.n - 1)
        rr = np.clip((py / c.resolution).astype(np.int32), 0, self.n - 1)

        hit_idx = np.clip((scan_m / self._step).astype(np.int32), 0, self._n_samples - 1)
        hit_any = scan_m < c.lidar_max_range - 1e-3

        # A blind beam is not "clear to 5 m", it is "no data". Marking it free is how
        # the map came to believe it had explored 81 m^2 from a standstill.
        blind = np.zeros(c.n_beams, bool)
        if self.blind_beams.size:
            blind[self.blind_beams] = True
            hit_any &= ~blind

        # ---- FRONTIER FIRST, against the map as it stands BEFORE this scan ---------
        #
        # ORDER IS THE WHOLE POINT. This used to integrate the scan and then ask
        # "is there an unknown cell before the hit?" -- but the free-marking below
        # has just made every cell before the hit KNOWN, so the answer was always
        # no. `visible` was structurally False on every ray of every tick, and the
        # channel emitted the constant lidar_max_range forever. Measured in the
        # simulator with the robot driving through 128 m^2 of unexplored world:
        # 300 ticks x 72 rays produced exactly ONE distinct value.
        #
        # That is 72 of the policy's 146 inputs frozen at a value an exploring
        # agent never sees in training, which is why the network sat at a constant
        # reverse-and-turn on the real robot while the same weights, fed varied
        # frontier values, commanded +0.32 m/s forward on average.
        frontier = np.full(c.n_beams, c.lidar_max_range, np.float32)
        if c.use_frontier_rays:
            unknown = self.known[rr, cc] == 0
            unk_any = unknown.any(axis=1)
            unk_idx = np.where(unk_any, unknown.argmax(axis=1), self._n_samples - 1)
            # Only report unexplored space this beam can actually SEE -- an unknown
            # cell behind a wall is not a frontier, it is a different room.
            visible = unk_any & (unk_idx <= hit_idx) & ~blind
            frontier = np.where(visible, self._depths[unk_idx],
                                c.lidar_max_range).astype(np.float32)

        # ---- then integrate the scan ------------------------------------------------
        sample_idx = np.arange(self._n_samples)[None, :]
        free_mask = sample_idx < hit_idx[:, None]
        if blind.any():
            free_mask &= ~blind[:, None]
        flat_free = np.unique(rr[free_mask] * self.n + cc[free_mask])
        if flat_free.size:
            known_flat = self.known.reshape(-1)
            self.explored_cells += int((known_flat[flat_free] == 0).sum())
            known_flat[flat_free] = 1
        if hit_any.any():
            self.known[rr[hit_any, hit_idx[hit_any]], cc[hit_any, hit_idx[hit_any]]] = 1
            self.occ[rr[hit_any, hit_idx[hit_any]], cc[hit_any, hit_idx[hit_any]]] = 1

        return frontier

    @property
    def explored_m2(self) -> float:
        return self.explored_cells * self.cfg.resolution ** 2


# ======================================================================================
# 5. Safety layers
# ======================================================================================
class LidarGovernor:
    """Plane-based braking law (ported verbatim from `SafetyGovernor` in train.py)."""

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg

    def __call__(self, v: float, omega: float, scan_m: np.ndarray) -> tuple[float, float, bool]:
        c = self.cfg
        d_min = float(np.min(scan_m))
        v_out = float(np.clip(v, c.v_min, c.v_max))
        omega_out = float(np.clip(omega, -c.omega_max, c.omega_max))
        if d_min <= c.hard_stop:
            v_out = min(v_out, 0.0)                  # rotate / reverse only
        elif d_min < c.safe_distance:
            span = max(c.safe_distance - c.hard_stop, 1e-6)
            v_out = min(v_out, c.v_max * (d_min - c.hard_stop) / span)
        return v_out, omega_out, v_out != v


class ToFReflex:
    """Emergency reflex braking from the 8x8 ToF patch. The hard override.

    Requires `tof_min_valid_zones` zones to agree before triggering. A single hot zone
    on a ToF array is frequently speckle (reflective floor, ambient IR, crosstalk);
    stopping on one pixel would make the robot undriveable, while requiring a small
    cluster keeps genuine obstacles -- which always span several zones at these
    distances -- fully detectable.
    """

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg
        self.last_min = float("inf")
        self.triggered = False
        self.trigger_count = 0

    def evaluate(self, depth: np.ndarray | None) -> tuple[float, bool]:
        """Returns (speed_scale in [0,1], emergency_stop)."""
        c = self.cfg
        if depth is None or not c.tof_enabled:
            return 1.0, False
        flat = np.sort(depth.reshape(-1))
        # The k-th smallest zone, not the single smallest -> speckle-immune.
        k = min(c.tof_min_valid_zones, flat.size) - 1
        d = float(flat[k])
        self.last_min = d
        if d <= c.tof_stop_distance:
            if not self.triggered:
                self.trigger_count += 1
            self.triggered = True
            return 0.0, True
        self.triggered = False
        if d < c.tof_slow_distance:
            span = max(c.tof_slow_distance - c.tof_stop_distance, 1e-6)
            return float((d - c.tof_stop_distance) / span), False
        return 1.0, False


@dataclass
class VisualBaseline:
    """Clear-floor statistics recorded by `--calibrate`."""

    edge_density: float = 0.05
    dark_ratio: float = 0.02
    flow_energy: float = 2.0

    @staticmethod
    def load(path: Path) -> "VisualBaseline | None":
        """Returns None when no calibration exists -- the caller must then FAIL OPEN.

        An uncalibrated hazard heuristic is not a safety device. Left enabled with
        guessed baselines it saturates on ordinary floor texture and vetoes all forward
        motion, which looks exactly like a broken robot. Since LiDAR and ToF are the
        actual safety layers, the correct behaviour for this advisory layer is to
        disable itself and say so loudly.
        """
        if path.exists():
            try:
                return VisualBaseline(**json.loads(path.read_text()))
            except Exception as exc:                       # noqa: BLE001
                LOGGER.error("Corrupt visual baseline %s (%s) -- visual gate DISABLED.", path, exc)
                return None
        LOGGER.warning(
            "No visual baseline at %s -- visual hazard gate DISABLED (fail-open). "
            "Run `--calibrate 20` on clear floor to enable it.", path,
        )
        return None

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(dataclasses.asdict(self), indent=2))


class VisualHazardMonitor:
    """Camera-based hazard score in [0, 1]. ADVISORY ONLY -- it may slow, never steer.

    Three cheap cues over a near-field floor ROI, all vectorised so OpenCV releases the
    GIL and the reflex thread keeps its timing:

        edge density : structure where flat floor is expected -> an object
        dark ratio   : a void where floor should be -> a drop-off / stair edge
        flow energy  : frame-to-frame change -> something looming

    This is a heuristic, NOT object detection: it cannot tell a doormat from a step, and
    it is sensitive to lighting. Hence advisory-only, a temporal-consistency requirement
    before it stops anything, and a floor on how much it may brake.
    """

    def __init__(self, cfg: RobotConfig, baseline: VisualBaseline | None) -> None:
        self.cfg = cfg
        self.enabled = baseline is not None      # fail-open when uncalibrated
        self.base = baseline or VisualBaseline()
        self._prev: dict[str, np.ndarray] = {}
        self._consec = 0
        self.last_score = 0.0
        self.last_components: dict[str, float] = {}

    @staticmethod
    def _roi(frame: np.ndarray) -> np.ndarray:
        """Bottom-centre band: the floor the robot is about to drive onto."""
        h, w = frame.shape[:2]
        return frame[int(h * 0.55):, int(w * 0.20):int(w * 0.80)]

    def _score_one(self, key: str, frame: np.ndarray) -> float:
        roi = self._roi(frame).astype(np.float32)
        if cv2 is not None:
            gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
        else:                                            # numpy fallback for --simulate
            mag = np.abs(np.diff(roi, axis=0, prepend=roi[:1])) + \
                  np.abs(np.diff(roi, axis=1, prepend=roi[:, :1]))
        edge_density = float((mag > 40.0).mean())
        dark_ratio = float((roi < 45.0).mean())

        prev = self._prev.get(key)
        flow_energy = 0.0 if prev is None or prev.shape != roi.shape \
            else float(np.abs(roi - prev).mean())
        self._prev[key] = roi

        # Each cue is expressed as excess over its calibrated clear-floor baseline.
        e = max(0.0, edge_density - self.base.edge_density) / max(self.base.edge_density, 0.02)
        d = max(0.0, dark_ratio - self.base.dark_ratio) / max(self.base.dark_ratio, 0.02)
        f = max(0.0, flow_energy - self.base.flow_energy) / max(self.base.flow_energy, 1.0)
        self.last_components = {"edge": e, "dark": d, "flow": f}
        return float(np.clip(0.45 * e + 0.40 * d + 0.15 * f, 0.0, 1.0))

    def evaluate(self, frames: dict[str, np.ndarray | None]) -> tuple[float, bool]:
        """Returns (speed_scale in [visual_min_speed_scale, 1], hard_stop)."""
        scores = [self._score_one(k, f) for k, f in frames.items() if f is not None]
        if not scores:
            self._consec = 0
            self.last_score = 0.0
            return 1.0, False

        score = max(scores)                              # worst camera wins
        self.last_score = score

        # Uncalibrated: score is still computed and logged (so you can see what it WOULD
        # have done) but it is given no authority over the command.
        if not self.enabled:
            return 1.0, False

        if score >= self.cfg.visual_hazard_threshold:
            self._consec += 1
        else:
            self._consec = 0

        # Only a SUSTAINED hazard stops the robot; a single bad frame merely slows it.
        if self._consec >= self.cfg.visual_consec_frames:
            return 0.0, True
        scale = 1.0 - score * (1.0 - self.cfg.visual_min_speed_scale)
        return float(np.clip(scale, self.cfg.visual_min_speed_scale, 1.0)), False


class RateLimiter:
    """Slew-limits (v, omega) so commands are physically achievable.

    Without this the policy can step from +0.5 to -0.15 m/s in one tick. The wheels
    slip, dead reckoning corrupts, and the drivetrain suffers. Braking is deliberately
    NOT limited -- an emergency stop must be immediate.
    """

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg
        self.v = 0.0
        self.omega = 0.0

    def reset(self) -> None:
        self.v = self.omega = 0.0

    def __call__(self, v: float, omega: float, dt: float, emergency: bool = False,
                 allow_rotation: bool = False) -> tuple[float, float]:
        if emergency:
            self.v = 0.0                                 # never rate-limit a stop
            if not allow_rotation:
                self.omega = 0.0
                return 0.0, 0.0
            # The caller has established that turning in place cannot reach the
            # obstacle (it lies outside robot_radius), so rotation is the escape and
            # must survive the stop. It is deliberately NOT slew limited either --
            # the same reasoning that makes a stop immediate makes turning AWAY from
            # something immediate. Forward speed stays zero regardless.
            self.omega = omega
            return 0.0, omega
        dv = self.cfg.max_lin_accel * dt
        dw = self.cfg.max_ang_accel * dt
        # Slowing down is always allowed at full rate; only acceleration is limited.
        self.v = v if abs(v) < abs(self.v) else float(np.clip(v, self.v - dv, self.v + dv))
        self.omega = float(np.clip(omega, self.omega - dw, self.omega + dw))
        return self.v, self.omega


# ======================================================================================
# 6. Motor driver
# ======================================================================================
class MotorDriver:
    """Abstract actuator interface. Implement `_apply()` for your hardware."""

    def __init__(self, cfg: RobotConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.last_cmd = (0.0, 0.0)
        self.last_executed = (0.0, 0.0)
        # The literal line handed to the hardware, and the duties inside it. The status
        # block reports THESE rather than recomputing the mapping, because recomputing
        # would re-derive the answer from the same assumptions instead of observing the
        # result -- and the creep pulse, stiction floor, kick and lead inversions all
        # sit between the commanded twist and the bytes that reach the Arduino.
        self.last_wire: str = ""
        self.last_duties: tuple[int, int] | None = None

    def to_wheel_speeds(self, v: float, omega: float) -> tuple[float, float]:
        """(v, omega) -> (left, right) wheel linear speeds [m/s], scaled to stay legal.

        Scales the pair uniformly rather than clipping each wheel, which preserves the
        commanded turn radius instead of distorting the path.
        """
        L = self.cfg.wheel_base
        vr, vl = v + omega * L / 2.0, v - omega * L / 2.0
        peak = max(abs(vr), abs(vl))
        if peak > self.cfg.wheel_speed_max:
            s = self.cfg.wheel_speed_max / peak
            vr, vl = vr * s, vl * s
        return vl, vr

    def drive(self, v: float, omega: float) -> None:
        with self._lock:
            self.last_cmd = (v, omega)
            # Assume the command is executed verbatim; a driver that knows better
            # (because it has real duties, a floor and a cap) overwrites this inside
            # _apply. See `executed()`.
            self.last_executed = (v, omega)
            self._apply(*self.to_wheel_speeds(v, omega))

    def min_speed_mps(self) -> float:
        """Slowest wheel speed this driver can actually execute. 0.0 = no floor."""
        return 0.0

    def executed(self) -> tuple[float, float]:
        """(v, omega) the wheels were actually driven at -- NOT what was commanded.

        These differ by a lot on this drivetrain, and the difference is structural
        rather than noise: the creep cutoff turns a sub-floor request into a full
        STOP, the stiction floor lifts a small request UP to floor speed, and the
        duty cap holds a large request DOWN to the ceiling. Dead reckoning from the
        commanded value therefore accumulates error in jumps, not as slow drift.

        Still an approximation: duty -> m/s uses the same linear, uncalibrated map as
        m/s -> duty, so it cannot see motor non-linearity or wheel slip. It removes
        the structural error, not the calibration error.
        """
        return self.last_executed

    def emergency_stop(self) -> None:
        """Callable from ANY thread, including the reflex. Must be fast and idempotent."""
        with self._lock:
            self.last_cmd = (0.0, 0.0)
            self.last_executed = (0.0, 0.0)
            self._apply(0.0, 0.0)

    def _apply(self, left: float, right: float) -> None:
        raise NotImplementedError

    def close(self) -> None:
        self.emergency_stop()


class NullMotorDriver(MotorDriver):
    """Logs instead of moving. The default, so a misconfigured run cannot drive away."""

    def __init__(self, cfg: RobotConfig, verbose: bool = False) -> None:
        super().__init__(cfg)
        self.verbose = verbose
        self.commands: list[tuple[float, float]] = []

    def _apply(self, left: float, right: float) -> None:
        self.commands.append((left, right))
        # NOTHING TURNS. Reporting the command as executed is what made the local map
        # dead-reckon a phantom pose across the room during a bench run: 400 ticks of
        # an un-driven reverse-and-turn swept the 5 m disc through 3 rotations and
        # marked 81 m^2 "explored" from a parked robot, which pinned all 72 frontier
        # rays at max. The ControlLoop uses the SIMULATED pose in --simulate, so
        # honesty here costs the sim nothing.
        self.last_executed = (0.0, 0.0)
        self.last_wire = f"(null driver -- nothing turns) L={left:+.3f} R={right:+.3f}"
        self.last_duties = None
        if self.verbose:
            LOGGER.debug("motor L=%+.3f R=%+.3f", left, right)


class FridayMotorDriver(MotorDriver):
    """Driver for the Friday L298N/Arduino-Uno stack (firmware `friday_motors.ino`).

    Wire protocol, ASCII newline-terminated at 115200:
        P            -> "FRIDAY-MOTORS..."   identity handshake
        W<ms>        -> "OK"                 arm the firmware watchdog
        M<l> <r>     -> "OK"                 duties, -255..255
        S            -> "OK"                 stop
        <empty line> -> (no reply)           keepalive

    FOURTH SAFETY LAYER. The Arduino stops the motors on its own if it hears
    nothing for the watchdog window. So if this process crashes, the Pi locks up, or
    the USB cable is pulled, the robot stops without any help from the Pi. Our 10 Hz
    control loop naturally pets it -- every tick sends an M command -- so the window
    is set to a few control periods, not seconds.

    THE UNCALIBRATED PART, stated plainly. The policy commands wheel speed in m/s;
    the firmware wants PWM duty. Duty is not proportional to speed on a brushed motor
    under load, and nothing in your calibration file maps between them. What this does
    is the honest first approximation:

        duty = 255 * max_duty * (wheel_speed / cfg.wheel_speed_max)

    linear, then a stiction floor (`min_duty`) so small commands do not just buzz, and
    a short boost out of standstill (the same idea as the firmware helper's kickstart,
    but spread over control ticks instead of a blocking sleep, which would stall the
    loop). Expect the real speed to differ from the commanded speed until you measure
    it. That error lands in dead reckoning, so the frontier channel degrades faster
    than it would with real odometry.
    """

    IDENT = "FRIDAY-MOTORS"

    def __init__(self, cfg: RobotConfig, port: str, baud: int = 115200,
                 max_duty: float = 0.60, watchdog_s: float = 0.5,
                 invert_left: bool = False, invert_right: bool = False,
                 min_duty: int = 0, kick_ticks: int = 2, kick_scale: float = 1.6,
                 creep_cutoff: float = 0.5, turn_min_duty: int | None = None,
                 turn_kick_duty: float | None = None,
                 turn_kick_cooldown_s: float = 1.0,
                 pulse_creep_duty: int = 0) -> None:
        super().__init__(cfg)
        import serial

        self.max_duty = max(0.05, min(max_duty, 1.0))
        self.invert_left, self.invert_right = invert_left, invert_right
        self.min_duty = max(0, min(min_duty, 255))
        if self.min_duty == 0:
            LOGGER.warning(
                "STICTION FLOOR DISABLED (--motor-min-duty 0). Small commands will be "
                "sent as small duties, which this hardware answers by HUMMING rather "
                "than moving, and the creep cutoff is disabled with it. Measured floor "
                "on this robot is 72.")
        # Default: 40% more than the straight-line floor. Pivots scrub, so they need it.
        self.turn_min_duty = max(0, min(turn_min_duty if turn_min_duty
                                        else int(self.min_duty * 1.4), 255))
        self.creep_cutoff = creep_cutoff
        # ⚠ DELIBERATE, APPROVED EXEMPTION FROM THE max_duty CAP -- pivots only.
        #
        # HANDOFF section 4 invariant 4 says duty is capped at max_duty "including
        # the kickstart boost". This breaks that on purpose, with Harsh's explicit
        # approval on 2026-08-28, and only for the first few ticks of a PIVOT.
        #
        # Why it is needed: a pivot scrubs all four wheels sideways and needs far
        # more torque to break static friction than rolling straight. At the
        # approved 0.75 ceiling (duty 191) she still would not turn cleanly -- she
        # hummed. The cap is right for STEADY current; the problem is the initial
        # break-away, which is a transient.
        #
        # Why it is bounded, because the risk this cap exists for is real (the L298N
        # heats, the battery sags, a brownout mid-drive could reset the Pi):
        #   - pivots ONLY; straight driving is still capped at max_duty
        #   - only while _kick_left is counting down (kick_ticks, default 2)
        #   - a cooldown, so an oscillating policy cannot re-trigger it every tick
        #     and turn a transient boost into sustained current
        self.turn_kick_duty = (max_duty if turn_kick_duty is None
                               else max(max_duty, min(turn_kick_duty, 1.0)))
        self.turn_kick_cooldown_s = turn_kick_cooldown_s
        self._was_pivot = False
        self._last_turn_kick = -1e9

        # ⚠ MODIFIES HANDOFF SECTION 4 INVARIANT 5 (the creep cutoff), with Harsh's
        # explicit approval on 2026-08-28. Read this before touching it.
        #
        # The invariant says a request far below the stiction floor becomes STOP, not
        # floor speed, so the governor's proportional braking cannot silently turn
        # into a lurch at floor speed. That reasoning is sound and is preserved --
        # what changes is that STOP is no longer the only alternative.
        #
        # Why it had to change: MEASURED on the real robot, the policy commands
        # 0.05-0.09 m/s in an ordinary room while the slowest speed this drivetrain
        # can execute is 0.179 m/s. Every command it produced was cut to STOP, so the
        # robot thought, decided, and never moved. The gap is a sim-to-real mismatch
        # (train.py assumed continuous control from -0.15 m/s), not a tuning error.
        #
        # What this does instead: drive at a duty that has real AUTHORITY for a
        # FRACTION of the ticks, so the AVERAGE speed is what the policy asked for.
        #   fraction = requested_peak / pulse_speed,  and on a driving tick both
        #   wheels are scaled by the same factor, so the turn geometry is preserved.
        # Proportional braking stays proportional -- as the governor lowers the
        # request, the pulses get rarer and she slides to a stop. It does not become
        # a constant floor-speed lurch, which is what the invariant guards against.
        #
        # It pulses at pulse_creep_duty, NOT at min_duty. Pulsing at the bare floor
        # is what produced Harsh's humming: one wheel at 72 dragging a dead one.
        # 0 disables pulsing and restores the original stop-only behaviour.
        self.pulse_creep_duty = max(0, min(pulse_creep_duty, 255))
        self._pulse_acc = 0.0
        self.kick_ticks, self.kick_scale = kick_ticks, kick_scale
        self._kick_left = 0
        self._moving = False
        self._err_count = 0
        self._cmds = 0
        self._closed = False

        # The Uno resets when DTR toggles on open; nothing sent during the bootloader
        # window is heard, so wait it out before the handshake.
        self._ser = serial.Serial(port, baud, timeout=1.0, write_timeout=0.2)
        time.sleep(2.0)
        self._ser.reset_input_buffer()

        reply = self._command("P")
        if not reply.startswith(self.IDENT):
            self._ser.close()
            raise RuntimeError(
                f"{port} answered {reply!r}, not {self.IDENT}. Refusing to drive an "
                f"unidentified board -- flash friday_motors.ino, or check you are on the "
                f"Arduino's port (yours is the FTDI FT232R, not the LiDAR's CP2102)."
            )
        LOGGER.info("Motors: %s on %s", reply, port)

        wd_ms = int(watchdog_s * 1000)
        if self._command(f"W{wd_ms}") != "OK":
            LOGGER.warning("Watchdog arm command was not acknowledged")
        else:
            LOGGER.info("Firmware watchdog armed at %d ms (control period %.0f ms)",
                        wd_ms, cfg.dt * 1000)
        # A watchdog shorter than the control period would trip between our own
        # commands and make the robot stutter.
        if wd_ms < cfg.dt * 1000 * 2:
            LOGGER.warning("Watchdog %d ms is tight against a %.0f ms control period; "
                           "consider raising --motor-watchdog.", wd_ms, cfg.dt * 1000)

        # Surface the speed range this drivetrain can ACTUALLY produce. A stiction floor
        # does not just slow the robot down -- it deletes the bottom of the speed range,
        # and with it the governor's ability to creep. Better to see that at startup
        # than to wonder later why the robot lurches instead of easing up to a wall.
        if self.min_duty:
            ceiling = int(255 * self.max_duty)
            min_mps = self.min_duty * cfg.wheel_speed_max / max(ceiling, 1)
            LOGGER.warning(
                "Stiction floor %d => slowest achievable wheel speed ~%.2f m/s "
                "(policy range is %.2f..%.2f m/s).", self.min_duty, min_mps, cfg.v_min, cfg.v_max)
            if min_mps > abs(cfg.v_min):
                LOGGER.warning(
                    "  REVERSE IS UNREACHABLE: v_min=%.2f m/s is below the floor, so every "
                    "reverse command becomes a stop.", cfg.v_min)
            if min_mps > 0.5 * cfg.v_max:
                LOGGER.warning(
                    "  The usable forward band is only %.2f..%.2f m/s -- the robot is close to "
                    "on/off. Raise --motor-max-duty, or re-measure wheel_speed_max "
                    "(--motor-calibrate); the %.2f m/s in the config is a training-sim value.",
                    min_mps, cfg.v_max, cfg.wheel_speed_max)

        # Short timeout from here on: a blocking readline inside the control loop would
        # turn a 100 ms budget into a 1 s one.
        self._ser.timeout = 0.05

    def _command(self, line: str) -> str:
        """Blocking write/read. Handshake only -- called from __init__ before any
        thread can reach this object, so it deliberately takes no lock. The control
        loop uses `_write_raw` instead, which never waits for a reply."""
        self._ser.write((line + "\n").encode())
        self._ser.flush()
        return self._ser.readline().decode(errors="replace").strip()

    def _mps_to_duty(self, mps: float, floor: int | None = None) -> int:
        # Clamp LAST, on the integer, and use an integer ceiling. Clamping the float
        # first let rounding push the result one count past the cap (127.5 -> 128),
        # which quietly breaches the limit that exists to keep the speed down.
        ceiling = int(255 * self.max_duty)
        duty = int(round(255.0 * self.max_duty * (mps / max(self.cfg.wheel_speed_max, 1e-6))))
        min_duty = self.min_duty if floor is None else floor

        if min_duty and duty != 0 and abs(duty) < min_duty:
            # SAFETY: below stiction the robot physically cannot go this slowly. Two
            # readings are possible and only one is safe.
            #
            # Lifting EVERY small request up to the floor would mean the governor's
            # proportional braking -- the thing that slows the robot as it nears an
            # obstacle -- silently becomes a lurch at floor speed. A commanded 0.05 m/s
            # creep would come out as whatever duty 72 produces.
            #
            # So: only lift requests that are already close to the floor. Anything well
            # below it meant "creep", and since creeping is impossible, stop instead.
            # The cutoff is measured against the STRAIGHT floor, never the (higher) pivot
            # floor. Scaling it with the pivot floor would make ordinary turns fall under
            # the cutoff and stop dead -- the robot could not turn at all.
            if abs(duty) < self.min_duty * self.creep_cutoff:
                return 0
            duty = int(math.copysign(min(min_duty, ceiling), duty))
        return max(-ceiling, min(ceiling, duty))

    def min_speed_mps(self) -> float:
        """The stiction floor expressed in m/s. Below this the wheels only hum."""
        ceiling = 255.0 * self.max_duty
        if ceiling <= 0:
            return 0.0
        return self.min_duty / ceiling * self.cfg.wheel_speed_max

    def _duties_to_twist(self, l_duty: int, r_duty: int) -> tuple[float, float]:
        """Integer duties -> the (v, omega) they represent. Inverse of _mps_to_duty."""
        ceiling = 255.0 * self.max_duty
        if ceiling <= 0:
            return 0.0, 0.0
        scale = self.cfg.wheel_speed_max / ceiling
        vl, vr = l_duty * scale, r_duty * scale
        return (vl + vr) / 2.0, (vr - vl) / self.cfg.wheel_base

    def pulse_speed_mps(self) -> float:
        """Wheel speed a creep pulse drives at. 0.0 when pulsing is disabled."""
        ceiling = 255.0 * self.max_duty
        if ceiling <= 0 or self.pulse_creep_duty <= 0:
            return 0.0
        return self.pulse_creep_duty / ceiling * self.cfg.wheel_speed_max

    def _creep_pulse(self, left: float, right: float) -> tuple[float, float]:
        """Turn a too-slow request into occasional bursts with the right average.

        Returns the wheel speeds to execute THIS tick. See the note in __init__ for
        why this exists and which invariant it modifies.
        """
        pulse = self.pulse_speed_mps()
        peak = max(abs(left), abs(right))
        if pulse <= 0.0 or peak <= 0.0 or peak >= self.min_speed_mps():
            return left, right                      # fast enough to execute directly
        self._pulse_acc += min(peak / pulse, 1.0)
        if self._pulse_acc < 1.0:
            return 0.0, 0.0                         # coast this tick
        self._pulse_acc -= 1.0
        scale = pulse / peak                        # both wheels: geometry preserved
        return left * scale, right * scale

    def _apply(self, left: float, right: float) -> None:
        left, right = self._creep_pulse(left, right)
        # Pivoting scrubs all four wheels sideways at once, which needs materially more
        # torque than rolling straight -- your own motor_1.py calls it the worst case on
        # this robot. So a pivot gets its own, higher stiction floor; sharing the
        # straight-line floor is what leaves turns humming while forward works fine.
        pivot = (left > 0) != (right > 0) and abs(left) > 1e-6 and abs(right) > 1e-6
        # The break-away happens when the pivot STARTS, which is not necessarily from
        # standstill -- turning while already rolling has to break the same friction.
        now = time.monotonic()
        entering_pivot = (pivot and not self._was_pivot
                          and now - self._last_turn_kick >= self.turn_kick_cooldown_s)
        self._was_pivot = pivot
        floor = self.turn_min_duty if pivot else self.min_duty
        l_duty = self._mps_to_duty(left, floor)
        r_duty = self._mps_to_duty(right, floor)

        if l_duty == 0 and r_duty == 0:
            self._moving = False
            self._kick_left = 0
            self.last_executed = (0.0, 0.0)               # the creep cutoff bites here
            self.last_duties = (0, 0)
            self._write_raw("S")                          # firmware's dedicated stop
            return

        # Boost out of standstill, spread across ticks so the loop never blocks.
        if not self._moving or entering_pivot:
            self._kick_left = self.kick_ticks
            if entering_pivot:
                self._last_turn_kick = now
        if self._kick_left > 0:
            self._kick_left -= 1
            # The exempt ceiling applies to a PIVOT kick only. Straight driving --
            # including its own standstill kickstart -- stays at max_duty.
            frac = self.turn_kick_duty if pivot else self.max_duty
            ceiling = int(255 * frac)
            l_duty = int(max(-ceiling, min(ceiling, l_duty * self.kick_scale)))
            r_duty = int(max(-ceiling, min(ceiling, r_duty * self.kick_scale)))
        self._moving = True

        # What these duties actually mean in m/s, for dead reckoning. Taken BEFORE
        # the inversions, which are wiring corrections and not changes in motion.
        self.last_executed = self._duties_to_twist(l_duty, r_duty)

        if self.invert_left:
            l_duty = -l_duty
        if self.invert_right:
            r_duty = -r_duty
        # POST-inversion, because these are the numbers the firmware actually receives.
        # The lead inversion is precisely where a wiring fault hides (a commanded
        # STRAIGHT went out as a spin for the whole of this stack's early life), so the
        # status block has to show the wire, not the intent.
        self.last_duties = (l_duty, r_duty)
        self._write_raw(f"M{l_duty} {r_duty}")

    def drive_duty(self, left_duty: int, right_duty: int) -> None:
        """Send raw PWM duties, bypassing the m/s mapping. Bench calibration only.

        The normal path goes through `_mps_to_duty`, which is the very thing being
        calibrated -- so measuring the stiction threshold has to sidestep it. Still
        capped by max_duty: this is a diagnostic, not a way around the speed limit.
        """
        with self._lock:
            ceiling = int(255 * self.max_duty)
            l = max(-ceiling, min(ceiling, int(left_duty)))
            r = max(-ceiling, min(ceiling, int(right_duty)))
            if l == 0 and r == 0:
                self._moving = False
                self._write_raw("S")
                return
            self._moving = True
            if self.invert_left:
                l = -l
            if self.invert_right:
                r = -r
            self._write_raw(f"M{l} {r}")

    def _write_raw(self, line: str) -> None:
        """Write without waiting for the reply, then drain non-blocking.

        Waiting for "OK" on every tick would put a serial round trip inside the control
        budget. Instead we drain whatever is pending and watch for ERR, reporting the
        rate rather than stalling on each command.
        """
        # Shutdown runs the stop path twice (ControlLoop.shutdown, then main's finally).
        # Once the port is closed the second pass is a no-op, not an error -- printing
        # "Motor write failed" there would cry wolf about a failure to stop.
        if self._closed or self._ser is None or not self._ser.is_open:
            return
        try:
            self._ser.write((line + "\n").encode())
            self.last_wire = line            # set on SUCCESS only: a failed write did
                                             # not command anything, and reporting it
                                             # as though it had would hide a dead port.
            self._cmds += 1
            pending = self._ser.in_waiting
            if pending:
                blob = self._ser.read(pending)
                if b"ERR" in blob:
                    self._err_count += 1
                    if self._err_count in (1, 10) or self._err_count % 100 == 0:
                        LOGGER.warning("Motor firmware returned ERR (%d of %d commands) "
                                       "-- last sent: %r", self._err_count, self._cmds, line)
        except Exception as exc:                          # noqa: BLE001
            LOGGER.error("Motor write failed: %s", exc)

    def close(self) -> None:
        """Idempotent: safe to call from several shutdown paths."""
        if self._closed:
            return
        try:
            self._write_raw("S")                          # last command is always a stop
        except Exception:                                 # noqa: BLE001
            pass
        self._closed = True
        try:
            if self._ser is not None:
                self._ser.close()
        except Exception:                                 # noqa: BLE001
            pass


class SerialMotorDriver(MotorDriver):
    """Example: newline-delimited 'L,R\\n' wheel speeds to a microcontroller.

    Adapt the wire format to your firmware. Keep the write non-blocking or short-timeout:
    a blocking serial write inside the control loop will wreck the loop's timing.
    """

    def __init__(self, cfg: RobotConfig, port: str, baud: int = 115200) -> None:
        super().__init__(cfg)
        import serial

        self._ser = serial.Serial(port, baud, timeout=0.05, write_timeout=0.05)
        LOGGER.info("Motor serial open on %s @ %d", port, baud)

    def _apply(self, left: float, right: float) -> None:
        line = f"{left:.3f},{right:.3f}"
        try:
            self._ser.write((line + "\n").encode())
            self.last_wire = line                         # success only -- see FridayMotorDriver
        except Exception as exc:                          # noqa: BLE001
            LOGGER.error("Motor write failed: %s", exc)

    def close(self) -> None:
        super().close()
        try:
            self._ser.close()
        except Exception:                                 # noqa: BLE001
            pass


# ======================================================================================
# 7. Policy runner
# ======================================================================================
class PolicyRunner:
    """ONNX Runtime wrapper with a hard observation-width check at load time."""

    def __init__(self, model_path: Path, cfg: RobotConfig, threads: int = 2) -> None:
        if ort is None:
            raise RuntimeError("onnxruntime not installed: pip install onnxruntime")
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads     # leave cores for sensors on a 4-core Pi
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])

        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.output_name = self.sess.get_outputs()[0].name

        # This check exists because a 72-beam (74-dim) and a 72+72-beam (146-dim) model
        # are both plausible exports of this project. Silently feeding the wrong width
        # is the single easiest way to put a robot into a wall.
        expected = cfg.obs_dim
        actual = inp.shape[-1]
        if isinstance(actual, int) and actual != expected:
            raise ValueError(
                f"Observation width mismatch: model expects {actual}, config produces {expected}. "
                f"Check `use_frontier_rays` ({cfg.use_frontier_rays}) and `n_beams` ({cfg.n_beams}) "
                f"against the training run that produced {model_path.name}."
            )
        self._buf = np.zeros((1, expected), np.float32)   # preallocated: no per-tick alloc
        self.model_name = model_path.name
        self.last_infer_ms: float | None = None
        self.infer_count = 0
        LOGGER.info("Policy loaded: %s (obs_dim=%d, threads=%d)", model_path.name, expected, threads)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """Run the network, timing it. The duration is evidence, not decoration:
        "inference is running" is only credible if THIS tick produced a measurement."""
        self._buf[0, :] = obs
        t0 = time.perf_counter()
        out = self.sess.run([self.output_name], {self.input_name: self._buf})[0][0]
        self.last_infer_ms = (time.perf_counter() - t0) * 1000.0
        self.infer_count += 1
        return out


# ======================================================================================
# 8. Black-box recorder
# ======================================================================================
class BlackBox:
    """Rolling in-memory log of the last N seconds, dumped on stop or fault.

    When a robot does something unexpected, "what did the sensors say 2 seconds before"
    is the only question that matters. A ring buffer costs nothing and answers it.
    """

    # v_raw/omega_raw are THE NETWORK'S OUTPUT and are never overwritten.
    # v_policy/omega_policy are kept, in the same position as in baseline_v1, so the
    # two files stay comparable -- but they are the POST-SUBSTITUTION working values,
    # which is exactly the mislabelling that made the first run's log unable to answer
    # "did my policy drive this robot". Compare v_raw against v_policy to see every
    # tick where a hand-written rule replaced the network's answer.
    FIELDS = ("t", "lidar_min", "tof_min", "visual", "v_policy", "omega_policy",
              "v_out", "omega_out", "gate", "loop_ms",
              "v_raw", "omega_raw", "authority", "wire",
              # Planner columns. `bias` is the rad/s actually added this tick, so
              # "the planner fired" and "the planner mattered" can be told apart --
              # 62% of ticks carrying a 0.05 rad/s nudge is a different fact from 62%
              # carrying a hard turn, and the gate column alone cannot distinguish them.
              "plan_err", "plan_bias",
              # Phase 10 metrics. infer_ms makes policy latency measurable per tick
              # rather than as a single end-of-run mean; explored_m2 turns coverage
              # into a RATE; recovery lets stuck episodes be counted from the file
              # alone, without needing the run's log.
              "infer_ms", "explored_m2", "recovery")

    def __init__(self, seconds: float, hz: float) -> None:
        self.buf: deque[tuple] = deque(maxlen=int(seconds * hz) + 16)

    def record(self, row: tuple) -> None:
        self.buf.append(row)

    def dump(self, path: Path) -> Path | None:
        if not self.buf:
            return None
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(self.FIELDS)
            w.writerows(self.buf)
        LOGGER.info("Black box written: %s (%d rows)", path, len(self.buf))
        return path


# ======================================================================================
# 9. Control loop
# ======================================================================================
class ControlLoop:
    """Fixed-rate fusion loop: read sensors -> observe -> policy -> safety chain -> motors."""

    def __init__(self, cfg: RobotConfig, policy: PolicyRunner, motors: MotorDriver,
                 slots: dict[str, SensorSlot], sim: SimulatedWorld | None = None,
                 baseline: "VisualBaseline | None" = None) -> None:
        self.cfg, self.policy, self.motors, self.slots, self.sim = cfg, policy, motors, slots, sim
        self.map = LocalExplorationMap(cfg)
        self.map.blind_beams = masked_beam_indices(
            cfg.self_mask_deg, getattr(cfg, "lidar_offset_deg", 0.0), cfg.n_beams)
        if self.map.blind_beams.size:
            LOGGER.info("Exploration map ignores %d blind beam(s) (chassis): %s",
                        self.map.blind_beams.size, list(self.map.blind_beams))
        # PERSISTENT MEMORY. `self.map` above is the observation's frontier source and
        # must keep its training geometry; `self.world` is what Friday REMEMBERS. Two
        # objects because they answer different questions and have different lifetimes:
        # one is reset every run by design, the other is the point of not being.
        self.world = WorldMap(max_range=cfg.lidar_max_range)
        self.mapper: MapperThread | None = None
        self.map_enabled = True
        self.planner: ExplorationPlanner | None = None
        self.planner_enabled = True
        # Broader stuck detection than the boxed-in check below. It sees oscillation,
        # spinning and no-progress, which are invisible in any single tick.
        self.recovery = RecoveryController(cfg, self.world)
        self.recovery_enabled = True
        self.governor = LidarGovernor(cfg)
        self.reflex = ToFReflex(cfg)
        self.visual = VisualHazardMonitor(cfg, baseline)   # None => fail-open, disabled
        # Up camera: measured image statistics, reported and never acted on. Explicitly
        # NOT object detection -- nothing here names what is in view, because nothing
        # that could is implemented.
        self.context = _cameras.ContextMonitor()
        # Set by main() so telemetry can report per-role device paths and reconnects.
        self.cam_threads: list = []
        LOGGER.info("Visual hazard gate: %s", "ACTIVE" if self.visual.enabled else "DISABLED (uncalibrated)")
        self.limiter = RateLimiter(cfg)
        self.reverse_fallback = True
        self.escape_turns = True
        self.status_every = 0          # ticks between full status blocks; 0 = off
        # Machine-readable state for anything that is not a person watching a terminal:
        # the dashboard, a log, the voice layer. Written atomically at state_every ticks.
        self.state_file: Path | None = HERE.parent / "state" / "rl_autonomy.json"
        self.state_every = 5
        # Voice. Disabled by default: a robot that starts talking unprompted during a
        # bench test is a surprise, and surprises around motors are the wrong kind.
        self.narrator = Narrator(enabled=False)
        # 0.5 s at 10 Hz: long enough that a passing brake is not mistaken for being
        # stuck, short enough that she does not sit against a wall.
        self.escape_after_ticks = 5
        self._stuck_ticks = 0
        self._warned_no_reverse = False
        self.blackbox = BlackBox(30.0, cfg.control_hz)

        self.last_action = np.zeros(2, np.float32)
        self._obs = np.zeros(cfg.obs_dim, np.float32)
        self._stop = threading.Event()
        self._reflex_thread: threading.Thread | None = None
        self.stats = {"ticks": 0, "reflex_stops": 0, "visual_stops": 0,
                      "governor_limits": 0, "stale_stops": 0, "overruns": 0,
                      "no_reverse_turns": 0, "escape_turns": 0, "planner_biases": 0,
                      "recovery_ticks": 0}
        self.loop_ms: deque[float] = deque(maxlen=200)
        # Populated by every tick, including the fail-safe early returns. Exists from
        # construction so a status render before the first tick is empty, not a crash.
        self.last_record: tlm.DecisionRecord | None = None

    # ------------------------------------------------------------------ reflex ------
    def _reflex_worker(self) -> None:
        """High-rate ToF watchdog with a direct line to the motors.

        Runs independently of the 10 Hz policy tick so a close obstacle is acted on as
        soon as the ToF reports it, not up to 100 ms later.
        """
        period = 1.0 / self.cfg.reflex_hz
        while not self._stop.is_set():
            depth, age = self.slots["tof"].read()
            if depth is not None and age <= self.cfg.tof_deadline:
                _, emergency = self.reflex.evaluate(depth)
                if emergency:
                    self.motors.emergency_stop()
                    self.limiter.reset()
            time.sleep(period)

    # ------------------------------------------------------------ observation ------
    def _open_side(self, scan: np.ndarray) -> float:
        """+1 to turn left, -1 to turn right -- toward whichever side is more open.

        Read from the LIVE scan, never a fixed direction: turning a fixed way into
        the wall you are already against is how a robot gets permanently stuck.
        """
        n = self.cfg.n_beams
        q = max(1, n // 12)
        left = float(np.mean(np.roll(scan, -((n // 4 - q) % n))[:2 * q + 1]))
        right = float(np.mean(np.roll(scan, -((3 * n // 4 - q) % n))[:2 * q + 1]))
        # MEAN, not minimum. The minimum lets one stray beam veto an entire side, and
        # when both sides contain any close return at all the two minima tie and the
        # choice collapses to a fixed direction -- which is how a robot ends up
        # turning into the wall it is already against. What is wanted here is which
        # way is more open OVERALL. A pivot does not translate, so a near return on
        # the chosen side is not a collision risk; the governor and reflex still run.
        return 1.0 if left >= right else -1.0

    def _build_observation(self, scan: np.ndarray) -> np.ndarray:
        c = self.cfg
        frontier = self.map.update_and_frontier(scan)
        n = c.n_beams
        self._obs[:n] = np.clip(scan / c.lidar_max_range, 0.0, 1.0)
        if c.use_frontier_rays:
            self._obs[n:2 * n] = np.clip(frontier / c.lidar_max_range, 0.0, 1.0)
        self._obs[-2:] = self.last_action
        return self._obs

    # ------------------------------------------------------------------- tick ------
    def tick(self) -> dict[str, Any]:
        c = self.cfg
        t_start = time.perf_counter()
        gate = "ok"

        scan, scan_age = self.slots["lidar"].read()
        depth, depth_age = self.slots["tof"].read()

        # ---- FAIL-SAFE: a stale safety-critical sensor means STOP, never "clear" ----
        if scan is None or scan_age > c.lidar_deadline:
            self.motors.emergency_stop()
            self.limiter.reset()
            self.stats["stale_stops"] += 1
            # A record even here -- ESPECIALLY here. "The LiDAR went stale and I
            # stopped" and "I decided to stop" produce identical wheel behaviour and
            # must not produce identical telemetry.
            rec = tlm.DecisionRecord(tick=self.stats["ticks"], obs_dim=c.obs_dim,
                                     model_name=self.policy.model_name, model_loaded=True,
                                     gate="stale_lidar", lidar_age=scan_age)
            rec.wire = self.motors.last_wire
            self.last_record = rec
            return {"gate": "stale_lidar", "v": 0.0, "omega": 0.0,
                    "lidar_min": float("nan"), "tof_min": float("nan"), "visual": 0.0,
                    "v_policy": 0.0, "omega_policy": 0.0,
                    "v_raw": float("nan"), "omega_raw": float("nan"),
                    "authority": tlm.SRC_SAFETY, "loop_ms": 0.0}
        if c.tof_enabled and (depth is None or depth_age > c.tof_deadline):
            self.motors.emergency_stop()
            self.limiter.reset()
            self.stats["stale_stops"] += 1
            rec = tlm.DecisionRecord(tick=self.stats["ticks"], obs_dim=c.obs_dim,
                                     model_name=self.policy.model_name, model_loaded=True,
                                     gate="stale_tof", lidar_min=float(np.min(scan)),
                                     tof_age=depth_age)
            rec.wire = self.motors.last_wire
            self.last_record = rec
            return {"gate": "stale_tof", "v": 0.0, "omega": 0.0,
                    "lidar_min": float(np.min(scan)), "tof_min": float("nan"), "visual": 0.0,
                    "v_policy": 0.0, "omega_policy": 0.0,
                    "v_raw": float("nan"), "omega_raw": float("nan"),
                    "authority": tlm.SRC_SAFETY, "loop_ms": 0.0}

        # ---- 1. policy ------------------------------------------------------------
        obs = self._build_observation(scan)
        action = self.policy.infer(obs)
        v_pol, omega_pol = float(action[0]), float(action[1])

        # THE RAW OUTPUT, CAPTURED ONCE AND NEVER WRITTEN TO AGAIN.
        #
        # `v_pol`/`omega_pol` below are working variables that later stages overwrite
        # in place. The first black box recorded THOSE as "v_policy", so the 34 ticks
        # whose heading came from the no-reverse rule were logged under the policy's
        # name at exactly +-omega_max -- and the log could not answer the one question
        # it existed to answer. These two are the network's actual answer.
        v_raw, omega_raw = v_pol, omega_pol
        cl = tlm.clearances(scan)
        frontier = self._obs[c.n_beams:2 * c.n_beams] if c.use_frontier_rays else None
        rec = tlm.DecisionRecord(
            tick=self.stats["ticks"], obs_dim=c.obs_dim,
            model_name=self.policy.model_name, model_loaded=True,
            infer_ms=self.policy.last_infer_ms,
            v_raw=v_raw, omega_raw=omega_raw,
            lidar_front=cl["front"], lidar_left=cl["left"],
            lidar_right=cl["right"], lidar_rear=cl["rear"],
            lidar_min=float(np.min(scan)), lidar_age=scan_age, tof_age=depth_age,
            # "LIVE" means the channel carries variation. Pinned at one value is the
            # signature of the frontier bug that once cost 41% of explored area, so it
            # is surfaced rather than left to be rediscovered.
            frontier_near=(float(np.min(frontier)) * c.lidar_max_range
                           if frontier is not None else None),
            frontier_live=(bool(np.ptp(frontier) > 1e-6) if frontier is not None else False),
        )
        rec.add("policy", v_raw, omega_raw, "network output")

        # Feed back the POLICY output (pre-safety), matching what training observed.
        self.last_action = c.to_normalised(v_pol, omega_pol)

        # ---- 1b. THIS DRIVETRAIN CANNOT REVERSE -----------------------------------
        #
        # The policy was trained with continuous control from -0.15 to +0.5 m/s. On
        # this robot a reverse of -0.15 m/s maps to duty 60 while the stiction floor
        # is 72, so EVERY reverse command becomes a stop. Measured consequence: parked
        # near an obstacle the policy commanded v=-0.07 for 300 consecutive ticks and
        # the robot never moved -- no safety layer fired, because nothing was wrong.
        # It was asking, correctly, for an escape its wheels cannot perform.
        #
        # A differential drive that cannot back out should TURN out. Substituting a
        # pivot keeps the robot doing the thing the policy actually wanted -- getting
        # away from what is in front of it -- using the escape route it does have.
        # The side is chosen from LIVE LiDAR, not at random: turn toward open space.
        #
        # This is a control-layer adaptation, NOT a safety bypass. It only ever
        # replaces a command that would execute as zero, it cannot increase forward
        # speed, and the governor, visual gate, reflex and rate limiter all still run
        # on the result below. Disable with --no-reverse-fallback to see the raw
        # policy behaviour.
        # THE CHECK RUNS EVERY TICK, not only when it fails. "Checked and executable"
        # and "never checked" look identical in a log otherwise, and this is the exact
        # axis on which the trained action space and this hardware do not overlap.
        floor_mps = self.motors.min_speed_mps()
        rec.capability = tlm.capability_check(
            v_pol, omega_pol, floor_mps, c.v_min, c.v_max, c.omega_max)

        if self.reverse_fallback and not rec.capability["executable"]:
            if not self._warned_no_reverse:
                self._warned_no_reverse = True
                # Said ONCE, loudly. Substituting an action silently is precisely what
                # must not happen, and per-tick logging at 10 Hz is its own kind of
                # silence -- it becomes scrollback nobody reads.
                LOGGER.warning(
                    "CAPABILITY MISMATCH: the policy is asking to reverse and this "
                    "drivetrain cannot. v_min %+.3f m/s vs stiction floor %.3f m/s, so "
                    "NO reverse in the trained action space is executable. Substituting "
                    "a pivot toward open space, labelled CAPABILITY_ADAPTER in "
                    "telemetry -- the raw policy output is preserved unmodified. The "
                    "real fix is retraining with the measured floor, not this. "
                    "--no-reverse-fallback shows the raw behaviour instead.",
                    c.v_min, floor_mps)
            v_pol = 0.0
            omega_pol = self._open_side(scan) * c.omega_max
            self.stats["no_reverse_turns"] += 1
            gate = "no_reverse_turn"
            rec.add("no_reverse", v_pol, omega_pol,
                    "; ".join(rec.capability["findings"])
                    or f"reverse {v_raw:+.3f} m/s is not executable")

        # ---- 1c. EXPLORATION PLANNER (heading only) --------------------------------
        #
        # The policy has no goal input -- 72 LiDAR beams, 72 frontier rays, its own
        # last action, and nothing that says where the unmapped room is. So the target
        # is NOT smuggled into the observation (that would be out-of-distribution input
        # to a network that cannot report its own confusion). It is arbitrated here,
        # after the policy has had its say.
        #
        # Heading only, capped, and additive. Forward speed stays the policy's and then
        # the safety chain's; a planner can never make this robot go faster at
        # anything. The governor, visual gate, reflex and limiter all run below.
        planner_ctx = None
        if self.planner is not None:
            err = self.planner.bearing_error(self.map.pose)
            bias, why = steering_bias(err, omega_pol, c.omega_max)
            planner_ctx = {"bearing_error": err, "bias": bias, "reason": why}
            if bias != 0.0:
                omega_pol = float(np.clip(omega_pol + bias, -c.omega_max, c.omega_max))
                self.stats["planner_biases"] += 1
                gate = "planner"
                rec.add("planner", v_pol, omega_pol, why)
            self.planner.note_progress(self.map.pose)
        rec.planner = planner_ctx

        # ---- 2. LiDAR governor ----------------------------------------------------
        v, omega, limited = self.governor(v_pol, omega_pol, scan)
        rec.add("governor", v, omega,
                f"clearance {rec.lidar_min:.2f} m vs bubble {c.safe_distance:.2f} m"
                if limited else "")
        if limited:
            self.stats["governor_limits"] += 1
            gate = "governor"

        # ---- 2b. ESCAPE: boxed in, and every correct decision says "do nothing" ----
        #
        # MEASURED on the real robot: parked with an obstacle at 0.33 m -- inside
        # hard_stop (0.36) -- the governor zeroed forward on 268 of 300 ticks while
        # the policy asked for +0.13 m/s. Forward was forbidden (correctly), reverse
        # was impossible (this drivetrain), and the turn the policy wanted was
        # +0.05 rad/s, far too gentle for a drivetrain whose pivots need duty 190.
        # Three individually correct decisions left her stuck against a wall until a
        # human moved her, which is exactly what "without human intervention" rules
        # out.
        #
        # So: once forward has been blocked for long enough that this is clearly not
        # a passing brake, turn DECISIVELY toward open space until the way clears.
        # A pivot in place cannot approach anything -- the robot's footprint does not
        # move -- so this reduces no safety margin. The governor, reflex and limiter
        # all still run on the result.
        # Only when reverse is genuinely unavailable. On a drivetrain that CAN
        # reverse, backing away is the natural escape and the policy already knows
        # how -- hijacking that with a pivot would be strictly worse.
        floor_mps = self.motors.min_speed_mps()
        cannot_reverse = floor_mps > 0.0 and abs(c.v_min) < floor_mps
        # "Not going forward" rather than "v is exactly zero": with reverse
        # unavailable a negative v executes as a stop, so it is stuck just the same.
        if (self.escape_turns and cannot_reverse
                and float(np.min(scan)) <= c.hard_stop and v <= 1e-6):
            self._stuck_ticks += 1
        else:
            self._stuck_ticks = 0
        boxed_in = self._stuck_ticks >= self.escape_after_ticks
        if boxed_in:
            v = 0.0
            omega = self._open_side(scan) * c.omega_max
            self.stats["escape_turns"] += 1
            gate = "escape_turn"
            rec.add("escape", v, omega,
                    f"forward blocked {self._stuck_ticks} ticks at "
                    f"{rec.lidar_min:.2f} m and no reverse available")

        # ---- 2c. RECOVERY CONTROLLER ----------------------------------------------
        #
        # Fed the EXECUTED twist, never the commanded one. On this drivetrain the creep
        # cutoff turns a sub-floor request into a full stop and the stiction floor lifts
        # a small one, so a detector watching commands would conclude the robot was
        # travelling nicely while it sat buzzing against a wall.
        #
        # Its output is a pivot in place, which cannot approach anything -- the
        # footprint does not translate -- so it reduces no safety margin, and the
        # governor, visual gate, reflex and limiter all still run on the result below.
        if self.recovery_enabled:
            # SAME SOURCE OF TRUTH AS DEAD RECKONING. In --simulate the world moves
            # exactly as commanded while the null driver honestly reports executing
            # nothing, so asking the driver would have every simulated run trip
            # `no_progress` within five seconds -- a detector firing on a robot that is
            # demonstrably crossing the room. On hardware the driver IS the truth,
            # because the creep cutoff and stiction floor make executed != commanded.
            ex = (v, omega) if self.sim is not None else self.motors.executed()
            forced = self.recovery.update(
                scan, ex[0], ex[1], v, boxed_in, self.map.pose,
                self.stats["ticks"], self._open_side(scan))
            rec._recovery_state = self.recovery.state
            if forced is not None:
                v, omega, why_rec = forced
                self.stats["recovery_ticks"] += 1
                gate = "recovery"
                rec.add("recovery", v, omega, why_rec)
                # At the last escalation the destination itself is suspect. Drop it so
                # the planner stops steering back toward whatever the robot cannot reach.
                if (getattr(self.recovery, "blacklist_requested", False)
                        and self.planner is not None):
                    with self.planner._lock:
                        if self.planner.target is not None:
                            self.planner.blacklist.add(self.planner.target.key())
                            self.planner.target = None
                            self.planner.targets_abandoned += 1
                    self.recovery.blacklist_requested = False

        # ---- 3. visual gate (advisory) --------------------------------------------
        frames = {}
        if c.cam_enabled:
            for key in ("cam_up", "cam_down"):
                if key in self.slots:
                    fr, age = self.slots[key].read()
                    frames[key] = fr if (fr is not None and age <= c.cam_deadline) else None
            # UP camera: measured context only, never motion authority. Its statistics
            # are recorded and reported; nothing downstream reads them.
            self.context.update(frames.get("cam_up"))
        # FLOOR HAZARD IS A DOWN-CAMERA QUESTION ONLY.
        #
        # VisualHazardMonitor scores a bottom-centre band -- "the floor the robot is
        # about to drive onto" -- and used to take max() over BOTH cameras, worst wins.
        # That is right for the down camera, which is mounted above the bumper and sees
        # floor. It is meaningless for the up camera, which is above the screen and
        # sees the room: the bottom of ITS frame is whatever furniture is in the middle
        # distance, so a sofa scored as floor hazard could brake the robot.
        vis_scale, vis_stop = self.visual.evaluate(
            {"cam_down": frames.get("cam_down")})
        rec.cams_reporting = sum(1 for f in frames.values() if f is not None)
        rec.context = self.context.snapshot()
        rec.cam_state = {
            t.role: {"device": t.device, "updates": self.slots[f"cam_{t.role}"].updates,
                     "reconnects": t.reconnects, "open_failures": t.open_failures}
            for t in self.cam_threads}
        rec.visual_score = self.visual.last_score
        rec.visual_enabled = self.visual.enabled
        if vis_stop:
            v = min(v, 0.0)
            self.stats["visual_stops"] += 1
            gate = "visual_stop"
        elif vis_scale < 1.0:
            v = min(v, c.v_max * vis_scale) if v > 0 else v
            if gate == "ok":
                gate = "visual_slow"
        rec.add("visual", v, omega,
                f"hazard score {self.visual.last_score:.2f} vs "
                f"{c.visual_hazard_threshold:.2f}" if (vis_stop or vis_scale < 1.0) else "")

        # ---- 4. ToF reflex (hard veto) --------------------------------------------
        # Set by the emergency branch when the obstacle is far enough that turning in
        # place cannot reach it. It has to be carried down to the rate limiter, which
        # otherwise zeroes rotation along with speed and silently undoes the escape.
        pivot_allowed = False
        tof_scale, emergency = self.reflex.evaluate(depth)
        if emergency:
            # Forward motion is vetoed absolutely -- that is the whole point of the
            # reflex and it does not soften.
            v = 0.0
            # Rotation is a different question, and zeroing it here was a real bug.
            # A differential drive pivoting in place TRANSLATES NOWHERE: it sweeps a
            # circle of robot_radius and cannot close distance on anything outside it.
            # Meanwhile this stage runs LAST, after the escape-turn and planner stages
            # have already chosen "turn toward open space" -- so zeroing omega erased
            # the one manoeuvre that could free her. Measured: stopped 0.20 m from a
            # wall she sat for 316 consecutive ticks with v_out AND omega_out at zero,
            # 100% REFLEX_STOP, unable to turn away and explored 0.1 m^2.
            #
            # So: keep the commanded rotation unless the bumper is almost touching,
            # and only freeze completely there. Approved by Harsh 2026-09-02.
            #
            # The threshold is BUMPER-relative, matching what the ToF measures. An
            # earlier version tested against robot_radius, a CENTRE-relative number,
            # which double-counted the robot's own length: she rests near 0.215 m after
            # braking, so every reflex tick read "too close" and froze while the policy
            # was asking for 0.6 rad/s. Measured over that run: 10 of 11 reflex ticks.
            if self.reflex.last_min <= c.tof_pivot_min_clearance:
                omega = 0.0
            else:
                pivot_allowed = True
            self.stats["reflex_stops"] += 1
            gate = "REFLEX_STOP"
        elif tof_scale < 1.0 and v > 0:
            v = min(v, c.v_max * tof_scale)
            if gate == "ok":
                gate = "tof_slow"
        rec.tof_min = self.reflex.last_min if depth is not None else None
        rec.add("tof", v, omega,
                f"nearest zone {self.reflex.last_min:.2f} m vs stop "
                f"{c.tof_stop_distance:.2f} / slow {c.tof_slow_distance:.2f}"
                if (emergency or tof_scale < 1.0) else "")

        # ---- 5. slew limit + actuate ----------------------------------------------
        v, omega = self.limiter(v, omega, c.dt, emergency=emergency,
                                allow_rotation=pivot_allowed)
        rec.add("limiter", v, omega, "slew limited")
        self.motors.drive(v, omega)

        # Read back what the driver actually put on the wire. Done AFTER drive() so
        # these are this tick's values, not the previous tick's.
        rec.final_v, rec.final_omega = v, omega
        rec.wire = self.motors.last_wire
        rec.duties = self.motors.last_duties
        rec.gate = gate

        # ---- 6. propagate pose with what the WHEELS actually did -------------------
        # Not the commanded twist. The creep cutoff, the stiction floor and the duty
        # cap all make executed != commanded on this drivetrain, and dead reckoning
        # from the command puts the frontier channel's map somewhere the robot is not.
        # In --simulate the world moves exactly as commanded, so the sim is the truth
        # there and the driver's opinion is irrelevant.
        if self.sim is not None:
            self.sim.step(v, omega, c.dt)
            self.map.dead_reckon(v, omega, c.dt)
        else:
            ex_v, ex_omega = self.motors.executed()
            self.map.dead_reckon(ex_v, ex_omega, c.dt)

        # ---- 6b. persistent memory ------------------------------------------------
        # Same pose as the observation map, so the two never disagree about where the
        # robot is. submit() copies and returns immediately; integration happens on the
        # mapper thread, because folding a sweep costs ~24 ms against a 100 ms budget
        # and that cost GROWS with the map.
        if self.mapper is not None:
            self.world.set_pose(*(float(x) for x in self.map.pose))
            self.mapper.submit(scan, self.map.pose, self.map.blind_beams)

        loop_ms = (time.perf_counter() - t_start) * 1000.0
        self.loop_ms.append(loop_ms)
        self.stats["ticks"] += 1
        if loop_ms > c.dt * 1000.0:
            self.stats["overruns"] += 1

        rec.loop_ms = loop_ms
        rec.explored_m2 = self.map.explored_m2
        if self.mapper is not None:
            w = self.world.stats()
            w["dropped"] = self.world.dropped
            rec.world, rec.world_loaded = w, self.world.loaded_from_disk
        rec.pose = tuple(float(x) for x in self.map.pose)
        rec.executed_v, rec.executed_omega = self.motors.executed()
        self.last_record = rec

        row = {"gate": gate, "v": v, "omega": omega,
               "lidar_min": float(np.min(scan)),
               "tof_min": self.reflex.last_min if depth is not None else float("nan"),
               "visual": self.visual.last_score,
               "v_policy": v_pol, "omega_policy": omega_pol, "loop_ms": loop_ms,
               "v_raw": v_raw, "omega_raw": omega_raw, "authority": rec.authority}
        self.blackbox.record((round(time.monotonic(), 3), round(row["lidar_min"], 3),
                              round(row["tof_min"], 3), round(row["visual"], 3),
                              # v_policy and v_raw MUST share a precision. At 3 dp
                              # against 4, 0.436 != 0.4360 and every tick looks like a
                              # rule substitution -- which is the exact false positive
                              # these columns exist to rule out.
                              round(v_pol, 4), round(omega_pol, 4), round(v, 3),
                              round(omega, 3), gate, round(loop_ms, 2),
                              round(v_raw, 4), round(omega_raw, 4), rec.authority,
                              rec.wire,
                              round(planner_ctx["bearing_error"], 4)
                              if planner_ctx and planner_ctx["bearing_error"] is not None
                              else "",
                              round(planner_ctx["bias"], 4) if planner_ctx else "",
                              round(self.policy.last_infer_ms or 0.0, 4),
                              round((rec.world or {}).get("explored_m2", 0.0), 2),
                              (rec.__dict__.get("_recovery_state") or "")))
        return row

    # ------------------------------------------------------------------- run --------
    def run(self, duration: float | None = None, verbose: bool = True) -> None:
        if self.cfg.tof_enabled:
            self._reflex_thread = threading.Thread(target=self._reflex_worker,
                                                   name="reflex", daemon=True)
            self._reflex_thread.start()
            LOGGER.info("Reflex thread running at %.0f Hz", self.cfg.reflex_hz)

        if self.map_enabled:
            if self.world.load():
                s = self.world.stats()
                LOGGER.info("World map restored from %s: %.1f m^2 known, %d sweeps, "
                            "%.1f m^2 previously driven%s",
                            self.world.map_file.name, s["explored_m2"],
                            s["sweeps_total"], s["visited_m2"],
                            "" if self.world.visits_loaded
                            else "  (visit trace MISSING -- map only)")
            else:
                LOGGER.info("No world map at %s -- starting a fresh one.",
                            self.world.map_file)
            # No explicit rate: MapperThread's default is the MEASURED one (5 Hz
            # against 7.4 ms mean per sweep). A literal here silently overrode it.
            self.mapper = MapperThread(self.world)
            self.mapper.start()
            if self.planner_enabled:
                # 0.5 Hz: one cycle costs ~265 ms on this Pi (DistanceField 87 ms +
                # frontier.find 179 ms over a 53 m^2 map), which is 2.65x the whole
                # control budget. It plans against a grid SNAPSHOT for the same reason.
                self.planner = ExplorationPlanner(self.world, period_s=2.0)
                self.planner.start()
                LOGGER.info("Frontier planner running every %.1f s "
                            "(heading bias only, capped at %.0f%% of omega_max)",
                            2.0, 100 * 0.55)

        period = self.cfg.dt
        next_tick = time.monotonic()
        t_end = None if duration is None else time.monotonic() + duration
        LOGGER.info("Control loop starting at %.0f Hz. Ctrl+C to stop.", self.cfg.control_hz)

        try:
            while not self._stop.is_set():
                if t_end is not None and time.monotonic() >= t_end:
                    break
                row = self.tick()
                # Full status block first: when both are on, the one-liner reads as a
                # continuation of the block rather than interrupting it.
                if self.state_every and self.stats["ticks"] % self.state_every == 0:
                    st = self.autonomy_state()
                    self._publish_state_dict(st)
                    # Transition detection + string building only: microseconds, no
                    # I/O. tts.speak() BLOCKS for a second or more and would blow the
                    # 100 ms budget by an order of magnitude, so it happens on the
                    # narrator's own thread.
                    self.narrator.observe(st or {})
                if self.status_every and self.stats["ticks"] % self.status_every == 0 \
                        and self.last_record is not None:
                    # Mode is DERIVED, not asserted. It was a literal here, which
                    # meant the printed block claimed "RL policy" whether or not the
                    # frontier planner was running -- a displayed field that did not
                    # come from internal state, which is the exact defect this
                    # telemetry exists to prevent. `_mode()` is the single source
                    # both the printed block and the JSON state read from.
                    print(tlm.autonomy_status(
                        self.last_record, self.cfg, mode=self._mode(),
                        planner=(self.planner.snapshot() if self.planner else None),
                        recovery=(self.recovery.snapshot() if self.recovery_enabled
                                  else None),
                        stats=self.stats,
                        objective=self._objective()), flush=True)
                if verbose and self.stats["ticks"] % 5 == 0:
                    LOGGER.info(
                        "t=%5d | lidar %.2f | tof %.2f | vis %.2f | pol(%+.2f,%+.2f) "
                        "-> out(%+.2f,%+.2f) | %-12s | %.1f ms",
                        self.stats["ticks"], row["lidar_min"], row["tof_min"], row["visual"],
                        row["v_policy"], row["omega_policy"], row["v"], row["omega"],
                        row["gate"], row["loop_ms"],
                    )
                next_tick += period
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.monotonic()          # overrun: resync, do not spiral
        except KeyboardInterrupt:
            LOGGER.warning("Interrupted -- stopping motors.")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        self.motors.emergency_stop()
        if self._reflex_thread is not None:
            self._reflex_thread.join(timeout=1.0)
        try:
            self.narrator.finish(self.autonomy_state())
        except Exception:                              # noqa: BLE001
            pass
        self.narrator.stop()
        if self.planner is not None:
            self.planner.stop()
            self.planner.join(timeout=3.0)
            p = self.planner.snapshot()
            LOGGER.info("Planner: %d chosen, %d ARRIVED at, %d revealed without "
                        "arriving, %d abandoned, %d cycles, %d frontiers left%s",
                        p["chosen"], p["arrived"], p["revealed"], p["abandoned"],
                        p["cycles"], p["frontiers"],
                        "  (EXHAUSTED)" if p["exhausted"] else "")
        if self.mapper is not None:
            self.mapper.stop()
            self.mapper.join(timeout=2.0)
            saved = self.world.save()
            s = self.world.stats()
            if saved is not None:
                LOGGER.info("World map saved: %s  (%.1f m^2 known, %.1f m^2 driven, "
                            "%.1f m^2 re-entered, revisit ratio %.2f, %d sweeps, "
                            "%d superseded)", saved.name, s["explored_m2"],
                            s["visited_m2"], s["revisited_m2"], s["revisit_ratio"],
                            s["sweeps"], self.world.dropped)
            else:
                LOGGER.info("World map not saved (nothing was observed).")
        self.motors.close()
        # Timestamped, so runs never overwrite each other. The old fixed name meant
        # every run destroyed the previous one's evidence -- and would have destroyed
        # baseline_v1_blackbox.csv, the recording of the first autonomous run, which
        # cannot be re-made. Comparing V1 against V2 requires both to still exist.
        self.blackbox.dump(HERE / time.strftime("blackbox-%Y%m%d-%H%M%S.csv"))
        self.report()

    def autonomy_state(self) -> dict[str, Any] | None:
        """The full machine-readable state, or None before the first tick."""
        if self.last_record is None:
            return None
        return tlm.autonomy_state(
            self.last_record, self.cfg,
            mode=("AUTONOMOUS EXPLORATION (RL policy + frontier planner)"
                  if self.planner is not None else "AUTONOMOUS (RL policy only)"),
            planner=self.planner.snapshot() if self.planner is not None else None,
            recovery=self.recovery.snapshot() if self.recovery_enabled else None,
            objective=self._objective(), stats=self.stats)

    def _publish_state(self) -> None:
        self._publish_state_dict(self.autonomy_state())

    def _publish_state_dict(self, st: dict[str, Any] | None) -> None:
        """Write the autonomy state where other processes can read it.

        Atomic replace via a temp file in the same directory -- the same idiom as
        face_control and the occupancy map, and for the same reason: a reader that
        catches a half-written file gets a JSON error rather than a plausible-looking
        truncated state.
        """
        if st is None or self.state_file is None:
            return
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.state_file.with_suffix(".tmp")
            with tmp.open("w") as fh:
                json.dump(st, fh, indent=1, default=str)
            os.replace(tmp, self.state_file)
        except Exception:                              # noqa: BLE001
            pass          # telemetry must never take the robot down

    def _objective(self) -> str:
        """What Friday is working toward, stated only as far as the code can support.

        There is no frontier planner in this stack -- the frontier rays are an
        OBSERVATION fed to the network, not a target it was given. Claiming "heading
        for the unexplored region to the north-east" would be fabrication. What can be
        said truthfully is whether unknown space is currently visible and what the
        policy is doing about it.
        """
        r = self.last_record
        if r is None:
            return "not started"
        if not self.cfg.use_frontier_rays:
            return "no frontier channel in this observation -- reactive navigation only"
        if not r.frontier_live:
            return ("no unknown space within LiDAR range: the policy is navigating on "
                    "geometry alone until new area opens up")
        return (f"unknown space visible from {r.frontier_near:.2f} m; the policy weighs "
                f"it against clearance each tick (no explicit target is assigned)")

    def report(self) -> None:
        lm = np.array(self.loop_ms) if self.loop_ms else np.array([0.0])
        LOGGER.info("=" * 74)
        LOGGER.info("ticks=%d  explored=%.1f m^2", self.stats["ticks"], self.map.explored_m2)
        LOGGER.info("loop ms: mean %.2f | p95 %.2f | max %.2f (budget %.0f)",
                    lm.mean(), np.percentile(lm, 95), lm.max(), self.cfg.dt * 1000)
        LOGGER.info("escape turns: %d  (boxed in: forward blocked and not moving, so "
                    "she turned toward open space)", self.stats["escape_turns"])
        LOGGER.info("no-reverse turns: %d  (policy asked to back up; this drivetrain "
                    "cannot, so it turned toward open space instead)",
                    self.stats["no_reverse_turns"])
        LOGGER.info("interventions: reflex=%d visual=%d governor=%d stale=%d overruns=%d",
                    self.stats["reflex_stops"], self.stats["visual_stops"],
                    self.stats["governor_limits"], self.stats["stale_stops"],
                    self.stats["overruns"])
        for name, slot in self.slots.items():
            LOGGER.info("  sensor %-6s updates=%-7d stale=%s", name, slot.updates, slot.is_stale())
        LOGGER.info("=" * 74)


# ======================================================================================
# 10. Wiring / CLI
# ======================================================================================
# Distinguishing hints from /dev/serial/by-id. The LiDAR is on a Silicon Labs CP2102;
# the Arduino motor board is an FTDI FT232R clone.
_PORT_HINTS = {
    "lidar": ("CP2102", "Silicon_Labs"),
    "motor": ("FTDI", "FT232R", "Arduino", "CH340"),
    # The DFRobot 8x8 ToF sits behind a Pico that presents as USB CDC. Auto-detection
    # is safe here for the same reason as the others: the frame format is gated --
    # a device that does not stream "yN:d0,...,d7," never publishes anything.
    "tof": ("Raspberry_Pi_Pico", "Pico"),
}


def resolve_serial_port(kind: str, given: str | None,
                        _candidates: "list[str] | None" = None) -> str:
    """Resolve 'auto' (or an empty value) to a stable /dev/serial/by-id path.

    Two reasons this exists. Shell variables do not survive an SSH login, so a
    hand-exported port silently becomes "" and produces a baffling
    `could not open port :`. And /dev/ttyUSBn is assigned by enumeration order, so it
    can move between boots while the by-id name never does.

    Auto-detection is safe here ONLY because both devices are gated downstream: the
    motor board must answer FRIDAY-MOTORS before a single duty is sent, and LiDAR
    frames must pass CRC before they are believed. Picking the wrong port therefore
    fails loudly rather than driving something unexpected.
    """
    if given and given not in ("auto", ""):
        return given

    import glob

    candidates = _candidates if _candidates is not None else sorted(glob.glob("/dev/serial/by-id/*"))
    for hint in _PORT_HINTS[kind]:
        for cand in candidates:
            if hint.lower() in cand.lower():
                LOGGER.info("Auto-detected %s port: %s", kind, cand)
                return cand

    listing = "\n  ".join(candidates) if candidates else "(none present)"
    raise RuntimeError(
        f"Could not auto-detect the {kind} port (looked for {', '.join(_PORT_HINTS[kind])}).\n"
        f"Serial devices present:\n  {listing}\n"
        f"Pass one explicitly with --{kind}-port."
    )


def build_sensors(cfg: RobotConfig, args) -> tuple[dict[str, SensorSlot], list[SensorThread], SimulatedWorld | None]:
    slots = {
        "lidar": SensorSlot("lidar", cfg.lidar_deadline, critical=True),
        "tof": SensorSlot("tof", cfg.tof_deadline, critical=True),
        # Named by ROLE, not by index. The two cameras see different things and are
        # used for different jobs; "cam0"/"cam1" made them interchangeable in the code
        # and they are not.
        "cam_up": SensorSlot("cam_up", cfg.cam_deadline, critical=False),
        "cam_down": SensorSlot("cam_down", cfg.cam_deadline, critical=False),
    }
    threads: list[SensorThread] = []

    # Alignment is a LiDAR-only question. Bringing up the ToF and cameras for it just
    # produces unrelated failures ("camera 0 did not open") that bury the one message
    # you are actually looking for.
    lidar_only = bool(getattr(args, "lidar_check", False))

    if args.simulate:
        sim = SimulatedWorld(cfg, seed=args.seed)
        threads.append(SimSensorThread("lidar", slots["lidar"], sim.lidar, 10.0))
        if not lidar_only:
            threads.append(SimSensorThread("tof", slots["tof"], sim.tof, 15.0))
            if cfg.cam_enabled:
                threads.append(SimSensorThread("cam_up", slots["cam_up"], sim.frame,
                                               float(cfg.cam_fps)))
                threads.append(SimSensorThread("cam_down", slots["cam_down"], sim.frame,
                                               float(cfg.cam_fps)))
        return slots, threads, sim

    kind = getattr(args, "lidar_type", "ld19")
    baud = args.lidar_baud if args.lidar_baud else (230400 if kind == "ld19" else 115200)
    args.lidar_port = resolve_serial_port("lidar", args.lidar_port)
    if kind == "ld19":
        threads.append(LD19Thread(slots["lidar"], cfg, args.lidar_port, args.lidar_offset_deg,
                                  baud, getattr(args, "lidar_invert", False),
                                  getattr(args, "ld19_ignore_crc", False),
                                  cfg.self_mask_deg))
    else:
        threads.append(RPLidarThread(slots["lidar"], cfg, args.lidar_port,
                                     args.lidar_offset_deg, baud))
    if lidar_only:
        return slots, threads, None
    if cfg.tof_enabled:
        # This robot's bumper ToF is the DFRobot matrix on USB serial, not the
        # I2C VL53L5CX this file originally assumed. VL53L5CXThread is kept for
        # other builds; --tof-type selects it.
        if getattr(args, "tof_type", "dfrobot") == "dfrobot":
            threads.append(DFRobotToFThread(
                slots["tof"], cfg, resolve_serial_port("tof", args.tof_port),
                args.tof_baud))
        else:
            threads.append(VL53L5CXThread(slots["tof"], cfg))
    if cfg.cam_enabled:
        found = _cameras.resolve()
        LOGGER.info("Cameras resolved by USB serial: %s", _cameras.describe(found))
        for role, serial in (("up", _cameras.UP_SERIAL), ("down", _cameras.DOWN_SERIAL)):
            override = getattr(args, f"cam_{role}_device", None)
            if role not in found and not override:
                LOGGER.warning("camera %s (serial %s) is not on the bus right now. "
                               "The thread starts anyway and will pick it up when it "
                               "returns -- the down camera on this robot is known to "
                               "re-enumerate on its own.", role, serial)
            threads.append(CameraThread(slots[f"cam_{role}"], cfg, role,
                                        serial=serial, path=override))
    return slots, threads, None


def run_calibration(cfg: RobotConfig, slots: dict[str, SensorSlot], seconds: float, out: Path) -> None:
    """Record clear-floor visual statistics. Point the robot at EMPTY floor first."""
    LOGGER.warning("CALIBRATION: aim the DOWN camera at clear, obstacle-free floor "
                   "for %.0f s.", seconds)
    mon = VisualHazardMonitor(cfg, VisualBaseline(1e-6, 1e-6, 1e-6))   # baselines off
    edges, darks, flows = [], [], []
    t_end = time.monotonic() + seconds
    while time.monotonic() < t_end:
        # DOWN camera only. The baseline describes clear FLOOR, and folding the up
        # camera's view of the room into it produces a baseline that matches neither.
        for key in ("cam_down",):
            fr, age = slots[key].read()
            if fr is not None and age <= cfg.cam_deadline:
                roi = mon._roi(fr).astype(np.float32)
                if cv2 is not None:
                    gx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
                    mag = np.hypot(gx, gy)
                else:
                    mag = np.abs(np.diff(roi, axis=0, prepend=roi[:1]))
                edges.append(float((mag > 40.0).mean()))
                darks.append(float((roi < 45.0).mean()))
                prev = mon._prev.get(key)
                if prev is not None and prev.shape == roi.shape:
                    flows.append(float(np.abs(roi - prev).mean()))
                mon._prev[key] = roi
        time.sleep(0.05)

    if not edges:
        LOGGER.error("No camera frames captured -- calibration aborted.")
        return
    # 90th percentile, not the mean: the baseline must cover normal floor variation,
    # otherwise ordinary texture reads as a hazard and the robot crawls everywhere.
    base = VisualBaseline(
        edge_density=float(np.percentile(edges, 90)),
        dark_ratio=float(np.percentile(darks, 90)),
        flow_energy=float(np.percentile(flows, 90)) if flows else 2.0,
    )
    base.save(out)
    LOGGER.info("Baseline saved to %s: %s", out, dataclasses.asdict(base))


def lidar_alignment_check(cfg: RobotConfig, slots: dict[str, SensorSlot],
                          duration: float | None = None,
                          threads: "list[SensorThread] | None" = None) -> None:
    """Stream per-sector LiDAR diagnostics so you can verify beam 0 points FORWARD.

    Why this exists: the normal runtime log prints min(scan) across all beams, which
    tells you how close something is but not WHERE it is. If the LiDAR is bolted on
    rotated, every range still looks plausible while the policy's entire world is
    turned -- and it will steer confidently into things. This is the only cheap way to
    catch that before the wheels are live.

    Procedure: put ONE object about 1 m directly in front of the robot, nothing else
    close. The reported `nearest` angle is your mounting offset; pass its NEGATIVE to
    --lidar-offset-deg and re-run until `nearest` reads ~0 deg.
    """
    beam_deg = 360.0 / cfg.n_beams
    LOGGER.info("=" * 78)
    LOGGER.info("LiDAR ALIGNMENT CHECK -- place ONE object ~1 m directly in FRONT of the robot.")
    LOGGER.info("Beam resolution %.1f deg. Goal: 'nearest' settles near 0 deg and 'front' is the "
                "smallest sector.", beam_deg)
    LOGGER.info("If nearest reads e.g. 90 deg, re-run with --lidar-offset-deg -90.")
    LOGGER.info("=" * 78)

    def sector_min(scan: np.ndarray, centre_deg: float, half_deg: float = 15.0) -> float:
        k = max(int(round(half_deg / beam_deg)), 1)
        c = int(round(centre_deg / beam_deg)) % cfg.n_beams
        idx = [(c + d) % cfg.n_beams for d in range(-k, k + 1)]
        return float(np.min(scan[idx]))

    t_end = None if duration is None else time.monotonic() + duration
    misses = 0
    try:
        while t_end is None or time.monotonic() < t_end:
            scan, age = slots["lidar"].read()
            if scan is None or age > cfg.lidar_deadline:
                misses += 1
                # If the driver thread already threw, that exception is the real answer.
                # Reporting "wrong port" over the top of a PermissionError or a timeout
                # sends you chasing the wrong problem.
                died = next((t for t in (threads or []) if t.name == "lidar" and t.error), None)
                if died is not None:
                    LOGGER.error("LiDAR driver failed: %s: %s",
                                 type(died.error).__name__, died.error)
                    LOGGER.error(
                        "  PermissionError  -> sudo usermod -aG dialout,plugdev $USER, re-login\n"
                        "  timeout/no reply -> wrong --lidar-baud (A1=115200, A3/S1=256000) "
                        "or no 5V motor power\n"
                        "  Run `python3 lidar_probe.py` to test every port and baud combination."
                    )
                    return
                # Fail fast with a diagnosis. Warning forever looks like the tool is
                # working when it is actually pointed at the wrong device.
                if misses >= 10:
                    LOGGER.error(
                        "No LiDAR data after %.0f s -- giving up. Most likely the WRONG PORT.\n"
                        "  An RPLIDAR enumerates as a Silicon Labs CP2102; an FTDI FT232R is\n"
                        "  usually a motor controller. Identify them with:\n"
                        "    for d in /dev/ttyUSB*; do echo == $d; "
                        "udevadm info -q property -n $d | grep ID_MODEL=; done\n"
                        "  Then check: is the LiDAR motor spinning? Are you in the dialout "
                        "and plugdev groups?", misses * 0.5,
                    )
                    return
                LOGGER.warning("no fresh scan (age=%.2f s) -- check --lidar-port and power", age)
                time.sleep(0.5)
                continue
            misses = 0
            i = int(np.argmin(scan))
            LOGGER.info(
                "nearest beam #%02d @ %5.1f deg = %.2f m | front %.2f | left %.2f | "
                "right %.2f | back %.2f",
                i, i * beam_deg, float(scan[i]),
                sector_min(scan, 0.0), sector_min(scan, 90.0),
                sector_min(scan, 270.0), sector_min(scan, 180.0),
            )
            time.sleep(0.5)
    except KeyboardInterrupt:
        LOGGER.info("alignment check stopped")


def motor_bench_test(cfg: RobotConfig, motors: MotorDriver) -> None:
    """Drive a fixed sequence with NO policy and NO sensors. Wheels must be off the ground.

    Exists because a sensor fault holds the safety chain at v=0, so the full pipeline
    can run start to finish without ever turning a wheel -- which tells you nothing
    about motor wiring. This isolates the actuator path so wiring, direction and
    stiction can be settled while sensor problems are diagnosed separately.
    """
    # Drive RAW DUTIES here, not m/s. The m/s -> duty mapping depends on
    # cfg.wheel_speed_max, which is a value from the training config that nobody has
    # measured on this robot -- so using it would make a wiring test depend on an
    # unvalidated guess. Comfortably above the measured stiction floor, because
    # commanding exactly the threshold sits on the edge and just hums.
    ceiling = int(255 * motors.max_duty)
    duty = int(min(max(motors.min_duty * 1.35, 95), ceiling)) if motors.min_duty \
        else max(ceiling // 2, 95)
    LOGGER.info("Bench duty %d (floor=%d, cap=%d)", duty, motors.min_duty, ceiling)

    steps = [
        ("FORWARD  - both wheels should turn forward", duty, duty, 1.5),
        ("stop", 0, 0, 1.0),
        ("ROTATE LEFT  - left back, right forward", -duty, duty, 1.5),
        ("stop", 0, 0, 1.0),
        ("ROTATE RIGHT - left forward, right back", duty, -duty, 1.5),
        ("stop", 0, 0, 1.0),
        ("BACKWARD - both wheels should turn backward", -duty, -duty, 1.5),
        ("stop", 0, 0, 1.0),
    ]
    LOGGER.warning("=" * 74)
    LOGGER.warning("MOTOR BENCH TEST -- the wheels WILL turn. Robot must be on a stand.")
    LOGGER.warning("=" * 74)
    time.sleep(2.0)

    try:
        for label, l_duty, r_duty, seconds in steps:
            LOGGER.info(">>> %s", label)
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                # Re-send every tick: this also pets the firmware watchdog, exactly as
                # the real control loop does.
                motors.drive_duty(l_duty, r_duty)
                time.sleep(cfg.dt)
    finally:
        motors.emergency_stop()
        LOGGER.info("Motors stopped.")

    LOGGER.info("=" * 74)
    LOGGER.info("If a side ran the wrong way : --motor-invert-left / --motor-invert-right")
    LOGGER.info("If motors buzzed but no spin: --motor-min-duty 70 (raise until they turn)")
    LOGGER.info("If nothing at all moved     : check L298N power and the ENA/ENB jumpers")
    LOGGER.info("=" * 74)


def motor_duty_sweep(cfg: RobotConfig, motors: "FridayMotorDriver",
                     start: int = 20, step: int = 8, hold: float = 1.4) -> None:
    """Ramp PWM duty upward so you can see where the wheels actually start turning.

    This measures the ONE number the m/s -> duty mapping cannot be honest without: the
    stiction threshold. Below it a brushed motor sits energised and humming, which is
    what you just heard. Note the duty at which each side begins to rotate steadily,
    and that becomes --motor-min-duty.
    """
    ceiling = int(255 * motors.max_duty)
    LOGGER.warning("=" * 74)
    LOGGER.warning("DUTY SWEEP -- wheels WILL turn. Robot on a stand.")
    LOGGER.warning("Watch for the duty where the wheels start turning STEADILY (not humming).")
    LOGGER.warning("Cap is %d (max_duty=%.2f). Raise --motor-max-duty if it never moves.", ceiling, motors.max_duty)
    LOGGER.warning("=" * 74)
    time.sleep(2.0)

    first_move = None
    try:
        for duty in range(start, ceiling + 1, step):
            LOGGER.info(">>> duty %3d  (%3.0f%% of cap)  %s",
                        duty, 100.0 * duty / max(ceiling, 1),
                        "<-- watch here" if first_move is None else "")
            deadline = time.monotonic() + hold
            while time.monotonic() < deadline:
                motors.drive_duty(duty, duty)     # re-send: also pets the watchdog
                time.sleep(cfg.dt)
            motors.drive_duty(0, 0)
            time.sleep(0.35)                      # settle back to rest between steps
    finally:
        motors.emergency_stop()

    LOGGER.info("=" * 74)
    LOGGER.info("Note the lowest duty that produced steady rotation, then run with:")
    LOGGER.info("    --motor-min-duty <that value>")
    LOGGER.info("If nothing turned even at %d, raise --motor-max-duty (try 0.8) and repeat.", ceiling)
    LOGGER.info("If it needs most of the cap to move at all, the L298N is likely current-starved:")
    LOGGER.info("  4 motors on 2 channels is ~2x the per-channel load, and the L298N drops ~2 V.")
    LOGGER.info("=" * 74)


def motor_speed_calibration(cfg: RobotConfig, motors: "FridayMotorDriver",
                            duty: int, seconds: float = 3.0) -> None:
    """Drive straight at a fixed duty for a fixed time so you can MEASURE the speed.

    This closes the last unmeasured gap in the stack. Everything upstream -- the
    governor's braking law, the reflex stop distance, dead reckoning, the policy's own
    v_max -- is expressed in metres per second, while the hardware takes PWM duty. Right
    now `wheel_speed_max = 0.85 m/s` in the config is a number from the TRAINING
    simulator that nobody has checked against this robot. If the truth is 0.3, every
    speed in the system is overstated by nearly 3x.

    ON THE FLOOR, clear straight run, tape measure ready.
    """
    LOGGER.warning("=" * 74)
    LOGGER.warning("SPEED CALIBRATION -- robot ON THE FLOOR, clear run ahead.")
    LOGGER.warning("Driving straight at duty %d for %.1f s. Mark start and end positions.", duty, seconds)
    LOGGER.warning("=" * 74)
    for n in (3, 2, 1):
        LOGGER.info("starting in %d...", n)
        time.sleep(1.0)

    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < seconds:
            motors.drive_duty(duty, duty)
            time.sleep(cfg.dt)
    finally:
        motors.emergency_stop()
    elapsed = time.monotonic() - t0

    LOGGER.info("=" * 74)
    LOGGER.info("Drove at duty %d for %.2f s. Measure the distance travelled, then:", duty, elapsed)
    LOGGER.info("    speed [m/s] = distance_cm / 100 / %.2f", elapsed)
    LOGGER.info("")
    LOGGER.info("Report that speed. It sets cfg.wheel_speed_max, which currently reads")
    LOGGER.info("%.2f m/s -- an unverified value inherited from the training simulator.", cfg.wheel_speed_max)
    LOGGER.info("Repeat at two or three duties (e.g. %d, %d, %d) to see whether the",
                duty, int(duty * 1.3), int(duty * 1.6))
    LOGGER.info("relationship is actually linear; on a loaded brushed motor it often is not.")
    LOGGER.info("=" * 74)


def policy_proof(cfg: RobotConfig, loop: ControlLoop) -> int:
    """Prove -- or disprove -- that the ONNX policy controls this robot.

    The question this answers is not "does an .onnx file exist" but "does the network
    determine what the wheels do". A file can exist, load, and still be irrelevant: if
    every command it produced were overruled downstream, the robot would behave
    identically without it. So this runs the REAL pipeline (real sensors unless
    --simulate, real observation builder, real network, real safety chain) and then
    tests the counterfactual directly.

    Four things are established, in order of how easy they are to fake:

      1. The model loads, and its input width matches what this config produces.
      2. Inference runs, with a measured duration, on a real observation.
      3. The network is INPUT-SENSITIVE -- distinct observations give distinct
         actions. A constant output would still "run" while controlling nothing.
      4. The counterfactual: the same observations are replayed through the safety
         chain with the network's action replaced by a fixed one. If the wheel
         commands come out the same, the policy is decoration.

    Motors are never engaged: the chain is exercised, the actuation is not.
    """
    print()
    print("=" * 70)
    print("RL POLICY VERIFICATION".center(70))
    print("=" * 70)

    # ---- 1 + 2: load and infer on real observations -------------------------------
    print(f"\n[1] Model: {loop.policy.model_name}")
    print(f"    input '{loop.policy.input_name}' width {cfg.obs_dim} "
          f"(n_beams={cfg.n_beams}, frontier_rays={cfg.use_frontier_rays})")
    print(f"    output '{loop.policy.output_name}'")

    n = 30
    obs_log, act_log = [], []
    print(f"\n[2] Running {n} live ticks through the real pipeline...")
    for _ in range(n):
        scan, age = loop.slots["lidar"].read()
        if scan is None:
            print("    FAIL: no LiDAR data -- cannot build a real observation.")
            return 6
        o = loop._build_observation(scan).copy()
        a = loop.policy.infer(o)
        obs_log.append(o)
        act_log.append(np.asarray(a, dtype=np.float64).copy())
        time.sleep(cfg.dt)

    A = np.array(act_log)
    O = np.array(obs_log)
    print(f"    inference ran {loop.policy.infer_count} times, "
          f"last {loop.policy.last_infer_ms:.3f} ms")
    print(f"    v     range [{A[:, 0].min():+.4f}, {A[:, 0].max():+.4f}]  "
          f"distinct {len(np.unique(A[:, 0].round(6)))}/{n}")
    print(f"    omega range [{A[:, 1].min():+.4f}, {A[:, 1].max():+.4f}]  "
          f"distinct {len(np.unique(A[:, 1].round(6)))}/{n}")

    # ---- 3: input sensitivity -----------------------------------------------------
    print("\n[3] Input sensitivity (is the output a function of the observation?)")
    probe = O[-1].copy()
    base = np.asarray(loop.policy.infer(probe), dtype=np.float64)
    clear = probe.copy(); clear[:cfg.n_beams] = 1.0
    blocked = probe.copy(); blocked[:cfg.n_beams] = 0.05
    a_clear = np.asarray(loop.policy.infer(clear), dtype=np.float64)
    a_block = np.asarray(loop.policy.infer(blocked), dtype=np.float64)
    print(f"    all-clear  -> v={a_clear[0]:+.4f} omega={a_clear[1]:+.4f}")
    print(f"    all-blocked-> v={a_block[0]:+.4f} omega={a_block[1]:+.4f}")
    dv = abs(a_clear[0] - a_block[0])
    sensitive = dv > 1e-3
    print(f"    delta v = {dv:.4f}  ->  {'SENSITIVE' if sensitive else 'CONSTANT (!!)'}")
    if a_clear[0] <= a_block[0]:
        print("    WARNING: the network does not slow for a blocked scan. Check the "
              "observation scaling against train.py before trusting this policy.")

    # ---- 4: the counterfactual ----------------------------------------------------
    print("\n[4] Counterfactual: would the wheels do the same with NO policy?")
    print("    Replaying the same observations with the network's action replaced by")
    print("    a fixed 'drive forward' command, through the same safety chain.")
    gov, lim = LidarGovernor(cfg), RateLimiter(cfg)
    real_out, null_out = [], []
    for o, a in zip(obs_log, act_log):
        scan = o[:cfg.n_beams] * cfg.lidar_max_range
        rv, ro, _ = gov(float(a[0]), float(a[1]), scan)
        real_out.append((rv, ro))
        nv, no_, _ = gov(cfg.v_max, 0.0, scan)          # a policy-free "just go"
        null_out.append((nv, no_))
    R, N = np.array(real_out), np.array(null_out)
    same_v = np.isclose(R[:, 0], N[:, 0], atol=1e-3).mean()
    same_w = np.isclose(R[:, 1], N[:, 1], atol=1e-3).mean()
    print(f"    v identical on {same_v * 100:.0f}% of ticks, "
          f"omega identical on {same_w * 100:.0f}%")

    print("\n" + "-" * 70)
    if sensitive and same_w < 0.9:
        print("VERDICT: the ONNX policy DETERMINES this robot's commands.")
        print("  The network is input-sensitive, and removing it changes the")
        print("  commanded heading on the majority of ticks.")
        rc = 0
    elif sensitive:
        print("VERDICT: the policy runs and is input-sensitive, but the safety chain")
        print("  is producing nearly the same commands without it. The policy is")
        print("  loaded and live, yet contributing little HERE -- most likely because")
        print("  the robot is boxed in and the governor is doing the deciding.")
        print("  Re-run in open space before concluding anything about the policy.")
        rc = 0
    else:
        print("VERDICT: the network's output does NOT vary with its input.")
        print("  Whatever is driving this robot, it is not learned behaviour.")
        rc = 7
    print("-" * 70)

    # The single most informative artefact: one COMPLETE tick, end to end, through
    # the whole safety chain. Run with the motors swapped for a null driver and then
    # restored, so this keeps its promise not to engage the wheels even when the
    # caller passed --motor friday and the robot is sitting on the floor.
    real_motors = loop.motors
    try:
        loop.motors = NullMotorDriver(cfg)
        loop.tick()
    finally:
        loop.motors = real_motors
    if loop.last_record is not None:
        print("\nOne complete decision chain (motors held inert for this tick):\n")
        print(tlm.policy_status(loop.last_record))
        print()
        print("REASON:", tlm.reason(loop.last_record, cfg))
    print()
    return rc


def selftest(cfg: RobotConfig, loop: ControlLoop) -> int:
    """Assert the safety chain actually behaves, using synthetic inputs.

    These are the invariants that must hold no matter what the policy outputs; if any
    fails, the robot must not be driven.
    """
    LOGGER.info("--- safety self-test ---")
    failures = 0

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        LOGGER.info("  [%s] %s %s", "PASS" if ok else "FAIL", name, detail)
        if not ok:
            failures += 1

    # 1. governor must forbid forward motion at/below hard_stop
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[0] = cfg.hard_stop - 0.05
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("governor blocks forward inside hard_stop", v <= 0.0, f"v={v:.3f}")

    # 2. governor must scale speed inside the bubble
    scan[0] = (cfg.hard_stop + cfg.safe_distance) / 2.0
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("governor brakes inside safety bubble", 0.0 < v < cfg.v_max, f"v={v:.3f}")

    # 3. reflex must fire on a close cluster
    depth = np.full((8, 8), 2.0, np.float32)
    depth[3:6, 3:6] = cfg.tof_stop_distance - 0.05
    _, emergency = loop.reflex.evaluate(depth)
    check("reflex fires on close cluster", emergency)

    # 4. reflex must IGNORE a single speckle zone
    depth = np.full((8, 8), 2.0, np.float32)
    depth[0, 0] = 0.05
    _, emergency = loop.reflex.evaluate(depth)
    check("reflex ignores single-zone speckle", not emergency)

    # 5. rate limiter must not soften an emergency stop
    loop.limiter.v = 0.5
    v, omega = loop.limiter(0.0, 0.0, cfg.dt, emergency=True)
    check("emergency stop is immediate", v == 0.0 and omega == 0.0)

    # 5b. an emergency stop must still let her TURN AWAY when the obstacle is outside
    # the circle a pivot sweeps -- otherwise she freezes facing a wall and the escape
    # turn the earlier stages already chose is erased. Forward must stay zero.
    loop.limiter.v = 0.5
    v, omega = loop.limiter(0.6, 1.2, cfg.dt, emergency=True, allow_rotation=True)
    check("emergency stop still permits a pivot escape",
          v == 0.0 and omega == 1.2, f"v={v:.3f} omega={omega:.3f}")

    # 6. rate limiter must cap acceleration
    loop.limiter.reset()
    v, _ = loop.limiter(cfg.v_max, 0.0, cfg.dt)
    check("acceleration is slew limited", v <= cfg.max_lin_accel * cfg.dt + 1e-6, f"v={v:.3f}")

    # 7. wheel speeds must stay legal for any command
    l, r = loop.motors.to_wheel_speeds(cfg.v_max, cfg.omega_max)
    check("wheel speeds within limit",
          max(abs(l), abs(r)) <= cfg.wheel_speed_max + 1e-6, f"L={l:.3f} R={r:.3f}")

    # 8. stale LiDAR must stop the robot
    loop.slots["lidar"]._stamp = time.monotonic() - 10.0
    row = loop.tick()
    check("stale lidar triggers stop", row["gate"] == "stale_lidar" and row["v"] == 0.0)

    LOGGER.info("--- self-test %s (%d failure(s)) ---", "PASSED" if not failures else "FAILED", failures)
    return failures


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sensor-fusion inference runtime for the Pi 4B.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model", type=str, default=str(HERE / "exploration_model.onnx"))
    p.add_argument("--simulate", action="store_true", help="Synthetic sensors; no hardware needed.")
    p.add_argument("--selftest", action="store_true", help="Run safety-chain assertions and exit.")
    p.add_argument("--voice", action="store_true",
                   help="Speak autonomy events aloud through Friday's Piper TTS. "
                        "Event-driven and throttled; speech runs on its own thread and "
                        "cannot delay the control loop. OFF by default.")
    p.add_argument("--voice-dry-run", action="store_true",
                   help="Like --voice but LOGS what would be said instead of speaking. "
                        "Use this to tune narration without waking the house.")
    p.add_argument("--state-file", default=None, metavar="PATH",
                   help="Where to publish the machine-readable autonomy state "
                        "(default: friday/state/rl_autonomy.json). Set to 'none' to "
                        "disable publishing.")
    p.add_argument("--cam-up-device", default=None, metavar="PATH",
                   help="Override the up camera's device path (normally resolved by "
                        "USB serial via /dev/v4l/by-id).")
    p.add_argument("--cam-down-device", default=None, metavar="PATH",
                   help="Override the down camera's device path.")
    p.add_argument("--no-recovery", action="store_true",
                   help="Disable oscillation/spin/no-progress detection and recovery. "
                        "The boxed-in escape turn is separate and stays on.")
    p.add_argument("--no-planner", action="store_true",
                   help="Disable frontier target selection. The RL policy then drives "
                        "alone, exactly as before Phase 3 -- useful for an A/B of "
                        "planner-on vs planner-off exploration efficiency.")
    p.add_argument("--no-map", action="store_true",
                   help="Disable the persistent world map (no load, no save, no "
                        "mapper thread). The policy is unaffected -- its frontier "
                        "rays come from LocalExplorationMap either way.")
    p.add_argument("--map-file", default=None, metavar="PATH",
                   help="Where the persistent world map lives "
                        "(default: friday/state/rl_map.npz). NOT state/map.npz -- "
                        "that is Friday's scan-matched survey and must not be "
                        "overwritten by a dead-reckoned one.")
    p.add_argument("--status-every", type=int, default=0, metavar="N",
                   help="Print the full FRIDAY AUTONOMOUS STATUS block every N ticks "
                        "(10 = once a second at the default rate). 0 = off.")
    p.add_argument("--policy-proof", action="store_true",
                   help="Prove whether the ONNX policy controls this robot: run the "
                        "real observation pipeline through the real network and print "
                        "the full sensor -> policy -> safety -> motor chain, then exit "
                        "without driving. Works with --simulate.")
    p.add_argument("--lidar-check", action="store_true",
                   help="Stream per-sector LiDAR diagnostics to verify beam 0 points forward, "
                        "then exit. Needs no model and never touches the motors.")
    p.add_argument("--calibrate", type=float, metavar="SECONDS", default=None,
                   help="Record the clear-floor visual baseline and exit.")
    p.add_argument("--duration", type=float, default=None, help="Stop after N seconds.")
    p.add_argument("--lidar-port", type=str, default="auto",
                   help="'auto' finds it under /dev/serial/by-id (stable across reboots).")
    p.add_argument("--lidar-type", choices=["ld19", "rplidar"], default="ld19",
                   help="'ld19' covers LDROBOT LD06/LD19/STL-19P (push-only, 230400). "
                        "'rplidar' covers Slamtec A-series (command/response).")
    p.add_argument("--lidar-baud", type=int, default=None,
                   help="Default depends on --lidar-type: ld19=230400, rplidar=115200. "
                        "A wrong baud looks identical to a dead port.")
    p.add_argument("--lidar-invert", action="store_true",
                   help="Reverse the scan direction. Training assumed beam 0 = heading, "
                        "advancing counter-clockwise; use this if yours advances the other way.")
    p.add_argument("--ld19-ignore-crc", action="store_true",
                   help="DIAGNOSTIC ONLY: accept LD19 frames that fail CRC. Never drive with this.")
    # MEASURED: friday/config.py LIDAR_FORWARD_DEG = 51.8, i.e. raw bearing 51.8 is
    # straight ahead. Friday's own code does `relative = raw - LIDAR_FORWARD_DEG`, so
    # the offset that puts beam 0 on the heading is the NEGATIVE of it.
    p.add_argument("--lidar-offset-deg", type=float, default=-51.8,
                   help="Mounting rotation: beam 0 must point along the robot heading. "
                        "Default -51.8 is measured for Friday (LIDAR_FORWARD_DEG=51.8). "
                        "Pass 0 for a generic robot.")
    # MEASURED on this robot, 197 revolutions parked (scratchpad/body_scan.py):
    # bearing bins 90-96 return 0.206-0.219 m with under 2 mm of jitter -- chassis,
    # not room. Bins 88, 89, 97 and 98 are MIXED: they flicker between the body
    # (~0.20-0.23 m) and the room behind it (1.8-2.2 m), which means the wedge's
    # true edges fall INSIDE them, so the whole bin must go. Bins 87 and 99 are pure
    # room (2.182 and 1.799 m, sd < 0.005) and are left alone. The wedge is therefore
    # the half-open range [88, 99). friday/config.py has ((89, 98),), which leaks at
    # both ends -- bearing 88 at 0.229 m and bearing 98 at 0.209 m.
    p.add_argument("--lidar-self-mask", type=str, default="88:99",
                   help="Raw sensor bearings looking at the robot's own chassis, as "
                        "LOW:HIGH pairs separated by commas (wrapping past 0 is allowed). "
                        "Half-open [low, high). Default 88:99 is measured for "
                        "Friday. Pass '' to disable.")
    p.add_argument("--motor", choices=["null", "friday", "serial"], default="null",
                   help="'null' only logs -- the safe default. 'friday' drives the "
                        "L298N/Arduino stack via friday_motors.ino.")
    p.add_argument("--motor-port", type=str, default="auto",
                   help="'auto' finds the Arduino under /dev/serial/by-id. Safe because the "
                        "FRIDAY-MOTORS handshake gates any actual driving.")
    # MEASURED and already approved for this hardware. friday/config.py raised
    # MOTOR_MAX_DUTY 0.6 -> 0.75 on 2026-08-25 with Harsh's explicit approval,
    # "because turning was weak and 0.6 capped output at 153/255". A pivot scrubs all
    # four wheels sideways and needs materially more torque than rolling straight, so
    # a ceiling that is fine for driving can still leave turns humming in place.
    # DELIBERATELY short of full power: the L298N heats and the battery sags under
    # load, and a brownout mid-drive could reset the Pi. friday/config.py says if
    # turning is ever still too weak, ASK before going higher. Do that, do not raise it.
    p.add_argument("--motor-max-duty", type=float, default=0.85,
                   help="Fraction of full PWM the driver may ever command. 0.85 matches "
                        "friday/config.py MOTOR_MAX_DUTY, which is the approved ceiling "
                        "for this L298N (raised from 0.75 on 2026-09-02 with Harsh's "
                        "explicit approval). Do not raise it without asking. NOTE: "
                        "wheel_speed_max was measured at the OLD ceiling of duty 191, so "
                        "commanded m/s under-reports real speed until it is re-measured.")
    p.add_argument("--motor-watchdog", type=float, default=0.5,
                   help="Firmware watchdog window [s]. The Arduino stops the motors if it "
                        "hears nothing for this long -- the safety layer that survives this "
                        "process dying.")
    # MEASURED on this robot and recorded in HANDOFF.md section 2: "Stiction floor:
    # duty 72 -- below this the motors hum and do not turn."
    #
    # This used to default to 0, and a default of 0 is not "no opinion" -- it DISABLES
    # the stiction floor, the creep cutoff (0.5 x 0 = 0) and anything that asks the
    # driver for its floor. Measured consequence: an autonomous run that omitted
    # --motor-min-duty sent the policy's -0.06 m/s straight through as duty 24, far
    # below break-away, and the motors buzzed for 30 s while the robot sat still.
    # Harsh heard it from across the room: "motors humming is coming when robot is
    # not even moving."
    #
    # A measured hardware constant does not belong in an optional argument that every
    # example command has to remember to pass.
    p.add_argument("--motor-min-duty", type=int, default=72,
                   help="Stiction floor: below this the motors hum and do not turn. "
                        "72 is measured on this robot. 0 DISABLES the floor and the "
                        "creep cutoff -- the motors will buzz instead of moving.")
    # MEASURED BY OBSERVATION 2026-08-28. Harsh watched four labelled pivots (100,
    # 130, 160, 190) and reported that only 190 turned acceptably -- "last pivot was
    # improved but needs more increase in speed while turning".
    #
    # The old default was 1.4x --motor-min-duty = 100, and that is why turns hummed.
    # It is worse than it looks: at omega_max (1.6 rad/s) a pivot's wheel speed is
    # only 0.28 m/s, which maps to duty 112. So EVERY pivot the policy could ask for
    # landed in the 100-112 band -- entirely below the duty at which this robot
    # actually rotates. She was never commanded enough torque to turn, at any omega.
    #
    # A high floor here is not a speed preference, it is the break-away threshold:
    # below it she hums and does not rotate at all. Gentle ARCS are unaffected -- both
    # wheels turn the same way, so they are not pivots and keep the straight floor.
    p.add_argument("--motor-turn-min-duty", type=int, default=190,
                   help="Stiction floor for PIVOTS, which scrub all wheels sideways and need "
                        "more torque than driving straight. 190 is measured on this robot; "
                        "below it she hums instead of turning. Pass 0 for 1.4x --motor-min-duty.")
    p.add_argument("--motor-creep-cutoff", type=float, default=0.5,
                   help="Requests below this fraction of --motor-min-duty STOP instead of "
                        "being lifted to the floor, so the governor's slow braking cannot "
                        "become a lurch at floor speed.")
    # MEASURED, and NOT optional on this robot. friday/config.py:
    #     MOTOR_INVERT_LEFT = False
    #     MOTOR_INVERT_RIGHT = True   <- the right motor leads are reversed
    # "Fixed in software rather than by rewiring: the chassis is glued shut and
    # opening it risks destroying it."
    #
    # This file was written on the dev machine and defaulted BOTH to False, so every
    # command it sent had the right side backwards. The consequences are not subtle
    # and they are dangerous the wrong way round:
    #     commanded STRAIGHT (M d d)   -> wheels opposed -> she SPINS
    #     commanded PIVOT    (M d -d)  -> both forward   -> she DRIVES OFF
    # The second one is how a pivot test turned into a robot heading for the wall.
    # If the leads are ever physically corrected, pass --no-motor-invert-right.
    # APPROVED 2026-08-28: pivots may briefly exceed --motor-max-duty to break the
    # sideways-scrub stiction that left turns humming at the 0.75 ceiling. Bounded to
    # the kick ticks, pivots only, with a cooldown. See FridayMotorDriver.__init__.
    p.add_argument("--motor-turn-kick-duty", type=float, default=0.90,
                   help="Duty fraction a PIVOT may reach during its first few ticks "
                        "only, to break static friction. Straight driving is always "
                        "capped at --motor-max-duty. Set equal to --motor-max-duty to "
                        "disable the exemption.")
    p.add_argument("--motor-turn-kick-cooldown", type=float, default=1.0,
                   help="Minimum seconds between pivot boosts, so an oscillating "
                        "policy cannot turn a transient into sustained current.")
    p.add_argument("--motor-invert-left", action=argparse.BooleanOptionalAction,
                   default=False)
    p.add_argument("--motor-invert-right", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--motor-test", action="store_true",
                   help="Drive a fixed sequence with no sensors and no policy, to check "
                        "wiring/direction/stiction. Requires --motor friday and "
                        "--i-am-watching. WHEELS WILL TURN.")
    p.add_argument("--motor-calibrate", type=int, metavar="DUTY", default=None,
                   help="Drive straight at this raw duty for --calibrate-seconds so you can "
                        "measure actual m/s. ON THE FLOOR. Requires --motor friday and "
                        "--i-am-watching.")
    p.add_argument("--calibrate-seconds", type=float, default=3.0)
    p.add_argument("--motor-sweep", action="store_true",
                   help="Ramp PWM duty upward to find the stiction threshold (the value for "
                        "--motor-min-duty). Requires --motor friday and --i-am-watching.")
    p.add_argument("--i-am-watching", action="store_true",
                   help="Confirms the robot is on a stand or has clear space. Required by "
                        "--motor-test and --motor-sweep.")
    p.add_argument("--no-cameras", action="store_true")
    # APPROVED 2026-08-28. See FridayMotorDriver.__init__ -- this modifies HANDOFF
    # invariant 5. 130 is chosen because a measured straight run at duty 130 has real
    # authority, while pulsing at the bare floor (72) hums instead of moving.
    p.add_argument("--motor-pulse-duty", type=int, default=130,
                   help="Duty used for creep pulses when the policy asks for a speed "
                        "below the stiction floor. The robot drives at this duty for a "
                        "fraction of ticks so the AVERAGE matches the request. 0 "
                        "disables pulsing (the robot then stalls at low speeds).")
    p.add_argument("--no-escape-turns", action="store_true",
                   help="Disable turning to escape when forward is blocked and the "
                        "robot is not moving. With this set she will sit against an "
                        "obstacle until a human moves her.")
    p.add_argument("--no-reverse-fallback", action="store_true",
                   help="Disable turning-instead-of-reversing. This drivetrain cannot "
                        "execute the policy's reverse commands at all, so with this set "
                        "the robot will simply stall whenever it wants to back up.")
    p.add_argument("--no-tof", action="store_true", help="DANGEROUS: disables reflex braking.")
    p.add_argument("--tof-type", choices=["dfrobot", "vl53l5cx"], default="dfrobot",
                   help="'dfrobot' is the 8x8 matrix ToF on USB serial fitted to this "
                        "robot. 'vl53l5cx' is the I2C part this file was written for.")
    p.add_argument("--tof-port", type=str, default="auto",
                   help="Serial port for --tof-type dfrobot. 'auto' resolves by USB id.")
    p.add_argument("--tof-baud", type=int, default=115200)
    p.add_argument("--threads", type=int, default=2, help="ONNX Runtime intra-op threads.")
    p.add_argument("--baseline", type=str, default=str(HERE / "visual_baseline.json"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    cfg = RobotConfig()
    cfg.cam_enabled = not args.no_cameras
    cfg.tof_enabled = not args.no_tof
    cfg.self_mask_deg = parse_self_mask(args.lidar_self_mask)
    cfg.lidar_offset_deg = args.lidar_offset_deg
    if cfg.self_mask_deg:
        LOGGER.info("LiDAR chassis self-mask: %s (raw bearings, pre-offset)",
                    ", ".join(f"{lo:g}-{hi:g}" for lo, hi in cfg.self_mask_deg))
    else:
        LOGGER.warning("No LiDAR self-mask. If any beam sees the robot's own body it "
                       "will pin the scan minimum and forward motion stays forbidden.")
    if not cfg.tof_enabled:
        LOGGER.warning("ToF DISABLED -- emergency reflex braking is OFF. Bench use only.")

    # The motor bench test touches no sensors at all, so it must not start them --
    # a dead LiDAR should not block checking the wheels.
    if args.motor_test or args.motor_sweep or args.motor_calibrate is not None:
        if args.motor != "friday":
            LOGGER.error("Motor bench modes need a real driver: add --motor friday")
            return 2
        if not args.i_am_watching:
            LOGGER.error("Refusing to spin the wheels without --i-am-watching. "
                         "Put the robot on a stand first.")
            return 2
        drv = FridayMotorDriver(
            cfg, resolve_serial_port("motor", args.motor_port),
            max_duty=args.motor_max_duty, watchdog_s=args.motor_watchdog,
            invert_left=args.motor_invert_left, invert_right=args.motor_invert_right,
            min_duty=args.motor_min_duty, creep_cutoff=args.motor_creep_cutoff, turn_min_duty=args.motor_turn_min_duty,
            turn_kick_duty=args.motor_turn_kick_duty,
            turn_kick_cooldown_s=args.motor_turn_kick_cooldown,
            pulse_creep_duty=args.motor_pulse_duty,
        )
        try:
            if args.motor_calibrate is not None:
                motor_speed_calibration(cfg, drv, args.motor_calibrate, args.calibrate_seconds)
            elif args.motor_sweep:
                motor_duty_sweep(cfg, drv)
            else:
                motor_bench_test(cfg, drv)
        finally:
            drv.close()
        return 0

    slots, threads, sim = build_sensors(cfg, args)
    for t in threads:
        t.start()

    motors: MotorDriver
    if args.motor == "friday":
        motors = FridayMotorDriver(
            cfg, resolve_serial_port("motor", args.motor_port),
            max_duty=args.motor_max_duty, watchdog_s=args.motor_watchdog,
            invert_left=args.motor_invert_left, invert_right=args.motor_invert_right,
            min_duty=args.motor_min_duty, creep_cutoff=args.motor_creep_cutoff, turn_min_duty=args.motor_turn_min_duty,
            turn_kick_duty=args.motor_turn_kick_duty,
            turn_kick_cooldown_s=args.motor_turn_kick_cooldown,
            pulse_creep_duty=args.motor_pulse_duty,
        )
        LOGGER.warning("MOTORS ARE LIVE (max_duty=%.2f). Wheels can turn.", args.motor_max_duty)
    elif args.motor == "serial":
        motors = SerialMotorDriver(cfg, args.motor_port)
    else:
        motors = NullMotorDriver(cfg, verbose=args.verbose)
        LOGGER.warning("Motor driver = NULL: commands are logged, wheels will NOT turn. "
                       "Use --motor serial once you have bench-tested.")

    exit_code = 0
    loop = None
    try:
        if args.calibrate is not None:
            time.sleep(1.0)                         # let camera threads spin up
            run_calibration(cfg, slots, args.calibrate, Path(args.baseline))
            return 0

        # Runs before the model is loaded on purpose: alignment is a wiring question,
        # so it must be answerable without a policy present.
        if args.lidar_check:
            lidar_alignment_check(cfg, slots, args.duration, threads)
            return 0

        policy = PolicyRunner(Path(args.model), cfg, threads=args.threads)
        baseline = VisualBaseline.load(Path(args.baseline))
        loop = ControlLoop(cfg, policy, motors, slots, sim=sim, baseline=baseline)
        loop.reverse_fallback = not args.no_reverse_fallback
        loop.escape_turns = not args.no_escape_turns
        if loop.reverse_fallback:
            floor = motors.min_speed_mps()
            if floor > 0 and abs(cfg.v_min) < floor:
                LOGGER.info("Reverse is unreachable (v_min %.2f m/s < floor %.2f m/s): "
                            "reverse commands will TURN toward open space instead of "
                            "stalling. --no-reverse-fallback disables this.",
                            cfg.v_min, floor)

        # Wait for the safety-critical sensors before allowing any motion.
        def _missing() -> list[str]:
            out = []
            if slots["lidar"].updates == 0:
                out.append("lidar")
            if cfg.tof_enabled and slots["tof"].updates == 0:
                out.append("tof")
            return out

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and _missing():
            time.sleep(0.05)

        missing = _missing()
        if missing:
            # Name the sensor and surface its thread's exception. "sensors never
            # reported" on its own sends you guessing at which one and why.
            LOGGER.error("Refusing to run: no data from %s.", ", ".join(missing))
            for t in threads:
                if t.name in missing:
                    if t.error is not None:
                        LOGGER.error("  %s thread failed: %s: %s",
                                     t.name, type(t.error).__name__, t.error)
                    else:
                        LOGGER.error("  %s thread is alive but published nothing "
                                     "(wrong baud, or the device is silent).", t.name)
            if "tof" in missing:
                LOGGER.error("  ToF hints: this robot's bumper sensor is a DFRobot 8x8 "
                             "matrix on USB SERIAL behind a Pico, not an I2C VL53L5CX. "
                             "Check --tof-port/--tof-baud (115200) and that it is "
                             "streaming 'yN:d0,...,d7,' lines; `ls /dev/serial/by-id/` "
                             "should show a Pico. Or pass --no-tof for BENCH USE ONLY "
                             "(it disables reflex braking).")
                LOGGER.error("  Do NOT run raspi-config / enable I2C for this: it edits "
                             "/boot/firmware/config.txt, which CLAUDE.md Part 1 forbids, "
                             "and it would not help -- wrong bus, wrong sensor.")
            if "lidar" in missing:
                LOGGER.error("  LiDAR hints: check --lidar-port/--lidar-baud (LD19 = 230400), "
                             "and that no other process holds the port "
                             "(pkill -f pi_deploy.py).")
            return 4

        loop.status_every = max(0, int(args.status_every))
        loop.map_enabled = not args.no_map
        loop.planner_enabled = not args.no_planner
        loop.recovery_enabled = not args.no_recovery
        loop.cam_threads = [t for t in threads if isinstance(t, CameraThread)]
        if args.voice or args.voice_dry_run:
            speak_fn = None
            if args.voice:
                try:
                    sys.path.insert(0, str(HERE.parent))
                    import tts as _tts
                    speak_fn = _tts.speak
                    LOGGER.info("Voice ON via %s", _tts.engine_name())
                except Exception as exc:              # noqa: BLE001
                    # Degrade to logging rather than refusing to run. Losing narration
                    # is a cosmetic loss; refusing to explore because the speaker is
                    # missing is not.
                    LOGGER.warning("TTS unavailable (%s: %s) -- narrating to the log "
                                   "instead.", type(exc).__name__, exc)
            else:
                LOGGER.info("Voice DRY RUN: narration goes to the log, not the speaker.")
            loop.narrator = Narrator(speak=speak_fn, enabled=True)
            loop.narrator.start()

        if args.state_file:
            loop.state_file = (None if args.state_file.lower() == "none"
                               else Path(args.state_file))
        if args.map_file:
            loop.world.map_file = Path(args.map_file)

        if args.policy_proof:
            return policy_proof(cfg, loop)

        if args.selftest:
            return 0 if selftest(cfg, loop) == 0 else 5

        signal.signal(signal.SIGTERM, lambda *_: loop.shutdown())
        loop.run(duration=args.duration, verbose=True)

    except Exception as exc:                              # noqa: BLE001
        LOGGER.exception("Fatal: %s", exc)
        exit_code = 1
    finally:
        # Motors off before anything else, in every exit path.
        try:
            motors.emergency_stop()
            motors.close()
        except Exception:                                 # noqa: BLE001
            pass
        for t in threads:
            t.stop()
        if loop is not None and not args.selftest:
            pass                                          # shutdown() already ran in loop.run
    return exit_code


if __name__ == "__main__":
    sys.exit(main())



