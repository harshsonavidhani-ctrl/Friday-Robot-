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

# THE GYRO (Harsh, 2026-10-07; motor firmware GYRO 2+, read in FridayMotorDriver).
GYRO_MAX_STEP_DEG = 90.0     # more than this between two replies: a glitch, fall back
PIVOT_STALL_CMD = 0.5        # rad/s commanded in place...
PIVOT_STALL_MEAS = 0.09      # ...while the gyro measures under ~5 deg/s...
PIVOT_STALL_TICKS = 6        # ...for 0.6 s: a stalled turn
import telemetry as tlm  # noqa: E402
from explorer import ExplorationPlanner, steering_bias  # noqa: E402
import cameras as _cameras  # noqa: E402
from dwa import DynamicWindowPlanner  # noqa: E402
from narrator import P_RECOVERY, Narrator  # noqa: E402
import way_back as _way_back  # noqa: E402
import unstick as _unstick  # noqa: E402
from navigator import Navigator  # noqa: E402
from recovery import RecoveryController  # noqa: E402
from world import MapperThread, WorldMap  # noqa: E402
import friday_freespace as fs_mod  # noqa: E402
from friday_freespace import FreeSpaceNavigator  # noqa: E402
from friday_vision import ObjectDetector  # noqa: E402

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
    # Kept only so old --help output, logs and saved configs still parse. The forward
    # braking law no longer uses an arc at all; see corridor_half_width below.
    governor_forward_arc_deg: float = 90.0
    # Mirror of the forward arc, used ONLY to gate reverse. Nothing on this robot
    # looks backwards otherwise: the ToF is a forward 8x8 patch, and the governor's
    # braking law reads the front hemisphere, so before this a reverse command was
    # executed with no obstacle check of any kind behind her.
    governor_rear_arc_deg: float = 90.0
    # ---- swept-corridor forward clearance ----
    #
    # THE ARC ABOVE IS THE WRONG QUESTION and it is what made her zig-zag down an
    # open corridor. Harsh, 2026-09-03: "traversing in a zig-zag way in an open
    # straight hallway/corridor... the collision rate INCREASED".
    #
    # A beam at 90 deg and 0.60 m is a wall BESIDE her. She is 0.22 m in radius, so
    # she drives straight past it -- but `min over the front hemisphere` called it
    # 0.60 m of forward clearance, which is under safe_distance (0.55) in any normal
    # hallway and at hard_stop (0.36) in a narrow one. Measured over the 6-minute run
    # of 2026-09-03 18:44: fwd_min p50 = 0.415 m and 419 degrees of turning per metre
    # travelled, in a corridor the ToF could see 1.6 m down.
    #
    # What a car asks instead: project every return onto the direction of travel and
    # keep only what is inside the body's swept corridor. x = d*cos(bearing) is how
    # far ahead it is, y = d*sin(bearing) how far to the side; a return with
    # |y| > half-width cannot be hit by driving straight, whatever d is.
    corridor_half_width: float = 0.0       # 0 -> robot_radius + collision_margin
    # She turns while she drives, so the swept corridor is wider than the body. Widen
    # it in proportion to the commanded turn rate rather than assuming the worst case
    # all the time, which is what the 90 deg arc effectively did.
    corridor_turn_widen: float = 0.10      # [m] extra half-width per rad/s
    corridor_max_widen: float = 0.18       # [m] ceiling on that
    # The one thing the corridor rule alone would miss: something level with her and
    # close enough to scrape as she turns. Anything inside this radius ANYWHERE in the
    # front hemisphere still brakes, whatever its lateral offset. This is what keeps
    # the old arc's real safety value without its cost.
    scrape_distance: float = 0.0           # 0 -> robot_radius + collision_margin
    # Relocalisation search, deliberately much wider than SLAM's tracking window
    # (0.20 m / 12 deg): that corrects a tick of drift, this has to absorb the robot
    # being nudged between sessions.
    relocalise_window_m: float = 1.00
    relocalise_window_deg: float = 40.0
    # Sharpness, min occupied cells and min judgeable beams all come from config.py
    # at runtime, NOT from numbers invented here. They are measured on this robot and
    # carry warnings: a gate of 5.0 "would reject every good answer it was ever
    # given" because a correct match scores 2.1-3.7, and my own first guess of 3.0
    # duly rejected a valid match at 2.46. See SLAM_POSE_MIN_SHARPNESS.
    # THE ARGUMENT BELOW WAS RIGHT ABOUT SPEED AND WRONG ABOUT MARGIN, and Harsh hit
    # it on 2026-09-03: "it wasn't able to detect a wall behind it and if an obstacle
    # was moved from behind it kept moving backwards without noticing the wall".
    #
    #   Reverse gets robot_radius + collision_margin and NOT the additional
    #   hard_stop_extra the forward law uses. Deliberate: forward runs at up to
    #   0.50 m/s while the only executable reverse is 0.15 m/s, so the stopping
    #   distance being bought is roughly 3x smaller -- and over-gating reverse
    #   recreates the failure that cost a whole session, where she could not back
    #   out of a doorway and could only spin in it.
    #
    # What that reasoning left out is that stopping distance is not the only thing the
    # margin has to cover, and that reverse has NO proportional band ahead of it. Going
    # forward the governor has already bled speed off between safe_distance (0.55) and
    # hard_stop (0.36), so she arrives slow. Reverse is binary -- it must be, because
    # any scaled-down reverse falls under the stiction floor -- so she arrives at the
    # FULL v_min every time, and the gate is the only thing between her and the wall.
    #
    # The budget at 0.15 m/s, with the coast MEASURED at 0.22 m from 0.403 m/s:
    #     coast, scaled by v^2   (0.15/0.403)^2 * 0.22 =  0.031 m
    #     coast, scaled by v     (0.15/0.403)   * 0.22 =  0.082 m   <- pessimistic case
    #     one more control tick at loop p95 217 ms      =  0.033 m
    #                                                     -------
    #     worst case before she can possibly be stopped   0.115 m
    #
    # The old rear_stop of 0.26 m left 0.26 - 0.22 = 0.04 m against a naive reading of
    # the coast and, against the honest budget above, less margin than the error bars.
    # 0.10 m of extra -- the same number forward uses -- puts rear_stop at 0.36 m and
    # buys real room. It does NOT recreate the doorway failure: that was a robot with
    # no executable reverse at all being forced to spin, and reverse-lifting (stage 1b)
    # is what actually fixed it.
    rear_stop_extra: float = 0.10
    # How many rear beams must agree before reverse is refused. 1 would be a bare
    # minimum and is what makes a single bad return veto an escape.
    #
    # 2, NOT the 3 the forward law uses, and the asymmetry is the point: FORWARD HAS
    # TWO SENSORS AND REVERSE HAS ONE. The ToF is a forward-facing bumper patch, so
    # going forward the LiDAR can afford to be conservative -- the reflex is watching
    # the same space with 64 independent zones. Behind her the LiDAR is the only thing
    # there is, and a miss is a collision with nothing to catch it.
    #
    # What k costs in blindness, at 5 deg per beam:
    #     k=3  an object must be  8 cm wide at 0.3 m, 13 cm at 0.5 m, 21 cm at 0.8 m
    #     k=2  an object must be  5 cm wide at 0.3 m,  9 cm at 0.5 m, 14 cm at 0.8 m
    # A chair or table leg is 3-5 cm. At k=3 she reverses into furniture legs by
    # construction; k=2 at least reaches them at close range.
    #
    # The speckle this guards against is also much rarer than when k=3 was chosen:
    # sub-footprint returns are now rejected at the source (lidar_min_valid_range), and
    # the runs since have logged ZERO of them outside the self-mask.
    rear_min_valid_beams: int = 2
    # Same rule, applied to the FORWARD braking law, which was the last stage in the
    # chain still taking a bare np.min(). The rear gate and ToFReflex had both already
    # been moved off the minimum for exactly this reason; forward had not, so one bad
    # beam anywhere in the front hemisphere could pin d_min under hard_stop and forbid
    # all forward motion. MEASURED across the eleven runs of 2026-09-03: the whole-scan
    # minimum sat BELOW robot_radius -- geometrically impossible -- on 63% and 65% of
    # ticks in two of them.
    forward_min_valid_beams: int = 3
    collision_margin: float = 0.04
    hard_stop_extra: float = 0.10       # extra margin over robot_radius + collision_margin

    # ---- LiDAR returns that cannot be real ----
    # A centre-mounted LiDAR cannot see anything closer than the chassis it stands on.
    # A return inside this radius is the robot's own body, a cable in the beam, or
    # sensor speckle -- never a thing the robot could steer around, because by the time
    # something is inside the footprint the robot is already in contact and the ToF and
    # the bumper own that problem, not the planner.
    #
    # WHY AT THE SOURCE rather than in the governor: the scan is read by five different
    # consumers -- governor, escape test, recovery, the observation the policy sees, and
    # the persistent map -- and fixing it in one leaves the other four believing it.
    # MEASURED parked (scratchpad/lidar_raw.py, 25 s): the only sub-radius returns are a
    # single chassis wedge at raw [90, 98) reading 0.206-0.219 m, already inside the
    # self-mask. Anything outside that wedge reading short is by construction spurious.
    #
    # Dropped exactly like a 0 mm return -- "no valid return" -- so the bin keeps
    # whatever real range the same beam saw elsewhere in the revolution instead of the
    # body winning the per-bin minimum.
    lidar_min_valid_range: float = 0.0   # 0 -> use robot_radius; set from --lidar-min-range

    # ---- ToF emergency reflex ----
    tof_enabled: bool = True
    # CALIBRATED FROM THE APPROACH SPEEDS SHE ACTUALLY REACHES, 2026-09-04.
    #
    # At 0.30 the reflex fired on 75% of ticks of a 12-minute run, and the tof_min
    # median DURING those ticks was 0.217 m: she spent the run stopped a hand's width
    # from things, which is the "hesitation" Harsh keeps seeing. 0.30 m of clear space
    # in front of the BUMPER (not the centre -- every ToF range is bumper-relative) is
    # a lot of margin for a robot 0.22 m in radius.
    #
    # What the margin has to cover, measured over the 2158 ticks she spent in the
    # 0.20-0.35 m band of that run:
    #     mean |v| 0.049 m/s, p95 0.150, worst single tick 0.445
    #     worst case = 0.015 m coasted in one 30 Hz reflex period
    #                + 0.124 m braking at max_lin_accel  =  0.139 m
    # So 0.24 leaves 0.10 m of margin against the worst approach ever recorded, and
    # 0.22 m against the p95 one, while cutting the ticks that trip it from 71% (at
    # 0.28) to 42%. The curve is steep exactly here -- 0.26 is 53%, 0.24 is 42% -- so
    # this is where the hesitation is bought back cheapest.
    #
    # It is still the LAST line: the LiDAR governor brakes from safe_distance 0.55 and
    # hard-stops at 0.36, and the navigator's near-field barrier steers around returns
    # at EVERY range (see navigator._tof_constraint) long before any of this.
    tof_stop_distance: float = 0.24     # [m] any zone closer than this -> emergency stop
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
    # THIS CAMERA DELIVERS EXACTLY HALF WHAT IT IS ASKED FOR. Measured through
    # cv2.VideoCapture on the up camera (Logitech C270), 120 frames each:
    #     request 15 fps -> 7.6 fps      request 30 fps -> 15.3 fps
    # at both 320x240 and 160x120, so it is the negotiated frame interval and not
    # bandwidth, resolution or exposure (auto_exposure and exposure_dynamic_framerate
    # were both ruled out by measurement). 15 was therefore buying 7.6 fps of vision
    # for a 10 Hz control loop -- the free-space profile was a third of a second stale
    # whenever it mattered. Asking for 30 costs nothing extra: the decode is 1.4 ms and
    # cap.read() is a blocking WAIT, so the thread was idle for the other 130 ms.
    cam_fps: int = 30
    visual_hazard_threshold: float = 0.65   # sustained score above this -> full stop
    visual_consec_frames: int = 3           # temporal consistency -> fewer false stops
    visual_min_speed_scale: float = 0.25    # a hazard never brakes below this fraction
    # ---- free-space vision (friday_freespace) ----
    # Harsh, 2026-09-03: "analyse all the available space in the environment and move
    # according to the space available... when the sensors are not working the device
    # uses the camera's vision to navigate like a self-driving vehicle."
    freespace_enabled: bool = True
    # ADVISORY. A nudge toward the open side, applied only while she is navigating
    # normally. Small on purpose: the LiDAR is the authority on where things are, and
    # a strong camera term fighting it is just a new way to zig-zag.
    freespace_gain: float = 0.55            # [rad/s per rad of camera bearing]
    freespace_max_nudge: float = 0.45       # [rad/s] ceiling on that
    # FALLBACK. Only when a critical sensor has gone stale and the OTHER one has not,
    # so something still covers the range this cannot. The camera is blind inside
    # 0.87 m by geometry (see friday_freespace.CAM_HEIGHT_M), which is exactly the ToF's
    # patch, so a fallback with both sensors down would be driving on nothing.
    freespace_fallback: bool = True
    freespace_v_max: float = 0.18           # [m/s] hard cap while driving on vision
    freespace_stop_range: float = 1.00      # [m] closer than this ahead -> do not drive
    freespace_heading_gain: float = 1.10    # [rad/s per rad] steering while on vision

    # ---- staleness deadlines [s] ----
    lidar_deadline: float = 0.40        # SAFETY CRITICAL -> stop when exceeded
    tof_deadline: float = 0.50          # SAFETY CRITICAL -> stop when exceeded
    cam_deadline: float = 1.50          # advisory -> degrade, do not stop

    # ---- object detection (perception only; see friday_vision) ----
    # OFF unless a model is present. It names what she is looking at for the
    # narration and the log; nothing in the motion chain reads it. 1 Hz because
    # inference measured 355-375 ms on this Pi and the control loop must keep its
    # 100 ms deadline.
    detect_enabled: bool = True
    detect_period_s: float = 1.0
    detect_threads: int = 2

    # ---- actuation limits ----
    # Fraction of v_max the local navigator may command on an arc it has simulated
    # clear, even when the policy asked for less. 0.0 = off: it may only ever reduce,
    # and she stays exactly as decisive as the network. See stage 1d in tick().
    nav_speed_authority: float = 0.0
    max_lin_accel: float = 0.8          # [m/s^2] command slew limit -> no wheel slip
    max_ang_accel: float = 4.0          # [rad/s^2]
    # A PIVOT IN PLACE IS NOT SLEW LIMITED. The slew limit exists to stop the wheels
    # slipping under a step change in TRANSLATION, which corrupts dead reckoning -- the
    # only odometry this robot has. A differential drive rotating about its own centre
    # translates nowhere, so there is no travel to corrupt and nothing to approach.
    #
    # It was also actively harmful. max_ang_accel * dt = 0.40 rad/s on the first tick of
    # any turn from standstill, which maps to a wheel duty of 37.6 -- BELOW the creep
    # cutoff at 40 -- so the first tick of every turn executed as a full stop. Measured
    # over the 3675 ticks logged on 2026-09-03: 797 of them (22%) commanded a pure turn
    # that the drivetrain silently executed as a stop, 257 of 316 in the worst run.
    # That is precisely the hesitancy and the mid-manoeuvre stalling Harsh reported.
    #
    # The turn kick already drives a standing pivot straight to duty 190-216 by design,
    # so issuing the commanded rotation immediately is no harsher than what the driver
    # was already sanctioned to do.
    pivot_slew_free: bool = True
    pivot_slew_v_eps: float = 0.02      # [m/s] |v| below this counts as "not translating"

    # ---- closing-rate braking (moving obstacles) ----
    # The braking law above is a function of DISTANCE alone, which is correct for walls
    # and wrong for anything that moves: a person walking toward the robot at 1 m/s
    # crosses the whole 0.55 m braking band in half a second, and a distance-only law
    # does not begin braking until they are already inside it.
    #
    # So the forward clearance is differentiated across ticks and, when it is closing,
    # the effective clearance used by the braking law is reduced by what will be lost
    # over `ttc_horizon_s`. A static wall has a closing rate equal to the robot's own
    # speed and is unaffected in practice; a wall the robot is driving at is exactly
    # the case the distance law already handled.
    ttc_enabled: bool = True
    # 0.6 s is not a taste setting: she COASTS ~0.22 m after the motors cut (measured
    # 2026-09-03, 0.405 m under power vs 0.629 m settled), and 0.22 m at her working
    # speed of ~0.4 m/s is 0.55 s. The horizon is therefore how long it actually takes
    # this robot to stop, which is the only defensible answer -- looking further ahead
    # than she can act would just make her timid.
    ttc_horizon_s: float = 0.6          # [s] lookahead applied to the closing rate
    ttc_max_rate: float = 2.0           # [m/s] cap: above this it is a sensor jump
    # MEASURED ON THIS ROBOT 2026-09-03, at the ceiling duty of 216 (max_duty 0.85).
    #
    # Method: driven at duty 216 straight at a wall SQUARE-ON to her (fitted plane
    # normal_x 0.976, planarity 0.018 m). A perpendicular plane is an exact 1:1
    # ruler -- travel d shortens its distance by d, with no scan matching involved.
    # 37 samples, ZERO non-monotonic steps. The acceleration ramp is visible out to
    # ~0.3 s; the steady-state slope after it is 0.403 m/s.
    #
    # CORROBORATED by a method that shares no failure mode: driving ONE wheel with
    # the other stopped arcs her about the stopped wheel at omega = v / wheel_base,
    # and the yaw recovered by circular correlation of the scan gives 0.397 m/s.
    # The two agree to 1.5%.
    #
    # Attempts that FAILED here, recorded so nobody repeats them:
    #   full-scan matching -- degenerate among parallel walls. Many (dx, dy, dtheta)
    #                         fit equally well, so the estimate wandered 0.04-0.33 m
    #                         with uniformly LOW residuals. Low residual is not
    #                         evidence of a good fit when the geometry is degenerate.
    #   oblique wall fit   -- at normal_x ~0.5 the division by the normal's forward
    #                         component amplified error enough to INVERT the sign,
    #                         reporting confident reverse motion for a forward drive.
    # Check normal_x before believing any wall-referenced distance.
    #
    # History, because several wrong values are on record and all get quoted:
    #   0.85  -- train.py's simulator. Never measured on hardware.
    #   0.476 -- measured 2026-08-28, but at the OLD duty-191 ceiling; it stopped
    #            describing the hardware the moment max_duty went to 0.85.
    #   0.185 -- friday/state/calibration.json forward_cm_per_s 18.5, taken at a
    #            LOWER effective duty, so it never described this constant at all.
    #
    # STILL n=1 for the translation run. It is the independent agreement that carries
    # the confidence here, not the run count -- more runs would tighten it.
    #
    # Two things measured alongside this that the constant does NOT capture:
    #   * the wheels differ by ~17% (left 0.428, right 0.366 m/s at identical duty),
    #     so open-loop straight driving always curves, and with no encoders nothing
    #     downstream can correct it;
    #   * she COASTS ~0.22 m after the motors cut (0.405 m under power, 0.629 m
    #     settled). Read that before trusting any stopping-distance assumption.
    wheel_speed_max: float = 0.403      # [m/s] per wheel -- MEASURED AT THE 0.85 CEILING

    # ---- LiDAR chassis self-mask ----
    # Raw sensor bearings (degrees, pre-offset) that look at the robot's own body.
    # Empty for a generic robot; the deployment sets it. See --lidar-self-mask.
    self_mask_deg: tuple[tuple[float, float], ...] = ()
    lidar_offset_deg: float = 0.0        # mounting rotation, set from --lidar-offset-deg

    # ---- exploration map ----
    map_size_m: float = 24.0            # rolling world grid edge (robot starts centred)
    blackbox_seconds: float = 30.0      # ring-buffer depth; set from --blackbox-seconds

    @property
    def dt(self) -> float:
        return 1.0 / self.control_hz

    @property
    def corridor_half(self) -> float:
        """Half-width of the corridor the body sweeps driving straight."""
        return self.corridor_half_width or (self.robot_radius + self.collision_margin)

    @property
    def scrape_radius(self) -> float:
        """Inside this, anything in front of her is a scrape whatever its bearing."""
        return self.scrape_distance or (self.robot_radius + self.collision_margin)

    @property
    def lidar_floor(self) -> float:
        """Ranges below this are the robot's own body, not the room."""
        return self.lidar_min_valid_range or self.robot_radius

    @property
    def rear_stop(self) -> float:
        """Clearance behind at or below which reverse is forbidden."""
        return self.robot_radius + self.collision_margin + self.rear_stop_extra

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
        # THE WAY BACK'S SWEEP (rl/way_back.py). 72 bins are too sparse to match
        # against -- measured, a scan matcher fed them never had enough walls to
        # match at all -- so the same good returns also go, at 1 degree, into a
        # {deg: mm} sweep published to `fine_slot` when one is attached. Nothing
        # else reads it; the 72 bins, the mask and the publish rule are unchanged.
        self.fine_slot: SensorSlot | None = None
        self._fine = np.zeros(360, np.float32)
        self._prev_start: float | None = None
        self._have_full_rev = False
        self._frames = self._crc_ok = self._scans = self._reconnects = 0
        # Sub-footprint returns dropped OUTSIDE the self-mask. A nonzero count is
        # not an error -- it is the sensor speckling -- but a large or growing one
        # means the chassis wedge has moved and the mask needs re-measuring.
        self._body_rejects = 0
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
            self._have_full_rev = False              # ...and so is the next one we assemble
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
        # Neither is a range inside the robot's own footprint: see RobotConfig
        # .lidar_min_valid_range. Both are the same statement -- "this sample carries no
        # information about the room" -- so they are rejected in the same place, before
        # the per-bin minimum, the map and the observation ever see them.
        good = ((dist_mm > 0) & (dist_m >= self.cfg.lidar_floor)
                & (dist_m <= self.cfg.lidar_max_range) & ~body)
        self._body_rejects += int(np.count_nonzero(
            (dist_mm > 0) & (dist_m < self.cfg.lidar_floor) & ~body))
        if good.any():
            idx = (angles[good] / 360.0 * self.cfg.n_beams).astype(np.int32) % self.cfg.n_beams
            np.minimum.at(self._bins, idx, dist_m[good])
            if self.fine_slot is not None:
                deg = angles[good].astype(np.int32) % 360
                self._fine[deg] = dist_m[good]

        # A wrap past 0 deg means one full revolution is complete -> publish it.
        if self._prev_start is not None and start_deg < self._prev_start - 180.0:
            if self._have_full_rev:
                scan = np.where(np.isinf(self._bins), self.cfg.lidar_max_range, self._bins)
                self.slot.publish(np.clip(scan, 0.0, self.cfg.lidar_max_range).astype(np.float32))
                if self.fine_slot is not None:
                    hit = np.nonzero(self._fine > 0.0)[0]
                    self.fine_slot.publish({int(d): int(self._fine[d] * 1000.0)
                                            for d in hit})
                self._scans += 1
                if self._scans % 100 == 0:
                    LOGGER.debug("LD19: %d scans, CRC pass %.1f%%",
                                 self._scans, 100.0 * self._crc_ok / max(self._frames, 1))
            else:
                # THE FIRST WRAP ONLY MARKS WHERE A REVOLUTION ENDS. The bins behind it
                # were filled from wherever in the sweep this thread happened to attach,
                # so every bin the sensor had not swept yet still holds inf and would go
                # out as lidar_max_range -- MEASURED on a cold start as 47 of 72 beams
                # wrong by up to 4.64 m, while the other 25 were already accurate to
                # 0.003 m. Publishing that is worse than publishing nothing: the control
                # loop's only test is `scan_age > lidar_deadline`, which asks whether a
                # scan is FRESH and never whether it is COMPLETE, and MapperThread
                # ray-casts those phantom max-range beams into the PERSISTENT map as
                # free space that was never observed.
                #
                # `_prev_start = None` in _reconnect was written to prevent exactly this
                # and does not: clearing it only defers wrap detection by one frame, and
                # the very next wrap still publishes the same partial sweep. This flag is
                # what that comment intended. Cost is one revolution, ~53 ms at the
                # measured 19 Hz, against a 5 s start-up budget.
                self._have_full_rev = True
                LOGGER.debug("LD19: discarded the partial first revolution "
                             "(%d/%d bins never swept)",
                             int(np.isinf(self._bins).sum()), self.cfg.n_beams)
            self._bins[:] = np.inf
            self._fine[:] = 0.0
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
# Where the object-detection model lives. Absent is a normal, supported state.
DETECT_MODEL_PATH = (Path(__file__).resolve().parent.parent
                     / "assets" / "models" / "vision" / "ssd_mobilenet_v1_10.onnx")

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
# THE FURTHEST GROUND ROW IS ALLOWED TO BE PATCHY, and this is not a tolerance dialled
# in to make a test pass -- it is what the sensor does. Row 4 looks along the shallowest
# incidence over the longest path, so it returns the weakest signal of the four, and on
# an ordinary floor it drops cells every single frame. MEASURED 2026-09-03 over 182
# frames in two different spots: rows 5, 6 and 7 returned 8.00/8 cells in EVERY frame,
# row 4 returned 5.59/8.
#
# floor_intact used to be unanimous over all 32 ground cells -- one non-returning cell
# anywhere set it False. With row 4 patchy by nature that made it False on 177 of 182
# frames, and DFRobotToFThread publishes nothing when the floor is not intact and no
# obstacle is visible. So the slot went stale, the control loop stopped the robot on a
# stale critical sensor, and Friday announced "A sensor has stopped responding" while
# the sensor was streaming a clean 15 Hz the whole time. That is the freeze Harsh saw.
#
# The near rows carry the safety claim anyway: "an intact floor is positive evidence the
# near path is empty" is an argument about the floor immediately in front of the bumper,
# which is rows 5-7. Row 4's job is occlusion detection, which still works from whatever
# cells it does return -- its SILENCE is just not evidence of anything.
TOF_FLOOR_NEAR_FROM_OFFSET = 1        # ground-row offsets >= this must be present
TOF_FLOOR_NEAR_QUORUM = 0.75          # ...and this fraction of their cells must agree
# ~0.7% of frames carry a spurious 250-315 mm return in the centre cells just above
# the horizon -- SPAD crosstalk echoing the strong floor return below. One of them was
# enough to brake the robot 0.2 s into a move. A real object is in every frame; an
# artefact cannot move a median.
TOF_CORROBORATE_FRAMES = 3

# ---- THE FLOOR IS FITTED FROM THE FRAME, NOT FROM THE CONSTANTS ABOVE ----------------
#
# Harsh, 2026-09-04: "The performance is worse than it was before... it is not
# discovering new paths on its own."
#
# MEASURED, 2026-09-03 23:0x vs 2026-09-04 00:0x, same robot, same room. The four
# ground rows read 314/236/168/127 mm, which is TOF_FLOOR_PROFILE_MM scaled by a clean
# and consistent 1.06. Hours later, after the down camera was moved and the bumper
# disturbed, the same rows read 231/141/116/89 -- ratios of 0.78, 0.63, 0.72, 0.74.
#
# THOSE RATIOS ARE NOT A PITCH. A pitch scales every row by the SAME factor. These do
# not, because the sensor was re-aimed: the floor now starts at row 3 instead of row 4,
# and that is a SHIFT ALONG THE ROW AXIS. A scalar multiplier cannot represent a shift,
# so the fit was pinned at TOF_FLOOR_PITCH_MIN = 0.80 against a true value of 0.735 and
# the model was simply wrong. Every ground row then read "shorter than the floor should
# be", which is the definition of an obstacle -- so the ToF reported the FLOOR as a wall
# 33 mm from the bumper and the reflex hard-stopped on 73% of ticks. She stood still for
# 847 of 1368 ticks and travelled 4.7 m in six minutes.
#
# This is the same bug class as the one that once stopped her driving at all, and
# re-measuring the constants would only postpone it to the next time a bracket moves.
# So the floor is now fitted PER FRAME from the frame's own row medians, which is
# invariant to any mounting angle:
#
#   * A floor RECEDES upward. Each row's median must exceed the row below it, and not
#     by so much that it has jumped past the horizon onto a wall. That walk finds the
#     ground rows without being told where they start.
#   * An object standing on the floor OCCLUDES, so it reads SHORTER than the floor. A
#     full-width object therefore breaks the recede test and the walk stops below it --
#     which correctly makes it an obstacle rather than a new floor.
#   * A cell is an obstacle when it is short against ITS OWN ROW's median, so no row
#     is ever compared against a distance meant for a different row.
#
# The constants above are kept as sanity bounds and as the fallback when a frame is too
# sparse to fit, never as the primary model.
TOF_FLOOR_ROW_QUORUM = 4        # cells a row must return before its median is trusted
TOF_FLOOR_MIN_GROWTH = 1.05     # a floor row must be at least this much further than
                                # the one below it, or it is not floor
TOF_FLOOR_MAX_GROWTH = 2.00     # ...and not this much further, or the beam has left
                                # the floor and landed on a wall
TOF_FLOOR_MAX_MM = 900          # absolute backstop on how far a ground row may read
TOF_FLOOR_MIN_ROWS = 2          # fewer ground rows than this = the floor is not visible
TOF_FLOOR_NEAR_ROWS = 3         # the bottom rows that carry the "path is clear" claim

# ---- cliffs (stairs, landings, the edge of a rug over a step) -----------------------
# How much FURTHER than its fitted floor a near cell must read before the floor counts
# as having fallen away. Above TOF_FLOOR_INTACT_MM (70), which is the "off-model" band,
# so a drop has to be a drop and not merely a floor the fit describes badly.
TOF_CLIFF_BEYOND_MM = 90
# How many near-row cells must agree before it is a cliff and not a dark patch. A real
# edge crosses the beam pattern and takes several cells at once; absorbing carpet takes
# one. Same argument and same order of magnitude as tof_min_valid_zones.
TOF_CLIFF_MIN_CELLS = 4
# THE FIT CANNOT SEE A DROP THAT TAKES THE WHOLE FIELD AT ONCE, and this is the bound
# that catches it. A floor receding uniformly is geometrically indistinguishable from a
# sensor aimed further out -- every row still grows on the same curve, every cell still
# matches its own row's reference, and `floor_intact` comes back True. Caught by a test
# only after the test was checked against broken code: the partial-edge case passed
# through the OCCLUSION path and hid it.
#
# So the bottom row gets an absolute ceiling, because it is the steepest look-down and
# the one with the least room for legitimate variation. MEASURED on this robot: 120 mm
# at the original mounting, 89 mm after the bumper was disturbed and the sensor
# re-aimed. 300 mm is 2.5x the first and 3.4x the second -- far outside anything a
# mounting change has ever produced here, and far inside the 520 mm a one-step drop
# gives. If it ever fires on real floor the sensor has moved enormously, and that is
# worth stopping for and saying out loud either way.
TOF_FLOOR_BOTTOM_MAX_MM = 300

# ---- relocalising into a map that is still mostly unknown ---------------------------
# How unambiguous the match must be to stand in for the score gate. `sharp` is peak
# over median across the whole search window, so it is scale-free: a sweep that fits
# nowhere in the window produces a flat surface and cannot reach this however wrong the
# map is. MEASURED on this robot: 29.7, 11.3, 7.1 and 4.2 on maps that were correct and
# placed within 2 cm of the parked pose. Set at 3.0 -- above the 1.4 floor used for
# confirming a pose you already trust, because this is overruling a gate rather than
# agreeing with one, and below every real match observed.
# How far the winning pose may sit from the saved one and still be believed on
# sharpness alone. She was parked there at the end of the last run; if nobody picked
# her up, the honest answer is a few centimetres. Deliberately much tighter than the
# search window, which is what makes this a corroboration and not a rubber stamp.
RELOC_TRUST_SHARPNESS = 3.0
RELOC_TRUST_M = 0.30
RELOC_TRUST_DEG = 15.0


def tof_floor_rows(frame) -> tuple[int, list]:
    """(index of the topmost GROUND row, per-row median mm or None).

    Ground rows are [horizon, TOF_ROWS). Fitted from the frame, so it follows the
    sensor wherever it is aimed -- see the note above for the measurement that made
    that necessary.
    """
    med: list = []
    for row in frame:
        vals = sorted(v for v in row if 0 < v < TOF_NO_RETURN_MM)
        med.append(float(statistics.median(vals))
                   if len(vals) >= TOF_FLOOR_ROW_QUORUM else None)
    horizon = TOF_ROWS
    prev = None
    for r in range(TOF_ROWS - 1, -1, -1):        # bottom row upward
        m = med[r]
        if m is None or m > TOF_FLOOR_MAX_MM:
            break
        if prev is not None and not (prev * TOF_FLOOR_MIN_GROWTH <= m
                                     <= prev * TOF_FLOOR_MAX_GROWTH):
            break
        prev = m
        horizon = r
    return horizon, med


def tof_obstacles_mm(frame: list[list[int]]) -> tuple[np.ndarray, bool]:
    """One raw 8x8 frame -> (per-cell obstacle range in mm, floor_intact).

    Cells that are not an obstacle come back as TOF_NO_RETURN_MM, so the array can be
    fed straight to ToFReflex without the floor tripping it. Ported from
    friday/sensors.py `_forward_from_frame`; every rule below cost a real debugging
    session and the reasons are recorded there.

    The floor is fitted PER FRAME (tof_floor_rows), not read off a fixed row-indexed
    profile -- see the note above tof_floor_rows for the mounting shift that made a
    fixed profile actively dangerous (it turned the floor itself into a phantom wall
    33 mm from the bumper). TOF_FLOOR_PROFILE_MM/PITCH remain only as the fallback for
    a frame too sparse to fit its own floor.
    """
    out = np.full((TOF_ROWS, TOF_COLS), float(TOF_NO_RETURN_MM), np.float32)

    horizon, _ = tof_floor_rows(frame)
    n_ground = TOF_ROWS - horizon
    expected: dict[int, float] = {}

    if n_ground >= TOF_FLOOR_MIN_ROWS:
        # Fitted: each ground row's reference is the FARTHEST valid return in IT. An
        # occluding object only ever reads SHORTER than the floor it sits on, so the
        # max survives any minority of occluded cells -- the row's own MEDIAN does
        # not (a 3-of-8 occlusion already drags the median past the 45 mm margin,
        # which hid the object in exactly this row during testing).
        for r in range(horizon, TOF_ROWS):
            valid = [v for v in frame[r] if 0 < v < TOF_NO_RETURN_MM]
            if valid:
                expected[r] = float(max(valid))
    else:
        # Too sparse to fit this frame's own floor -- fall back to the fixed,
        # pitch-scaled profile. Never the primary model; see the note above.
        horizon = TOF_FLOOR_FROM_ROW
        profile = TOF_FLOOR_PROFILE_MM
        ground_fallback = frame[horizon:horizon + len(profile)]
        ratios = [v / profile[off]
                  for off, row in enumerate(ground_fallback) for v in row
                  if 0 < v < TOF_NO_RETURN_MM]
        pitch = 1.0
        if len(ratios) >= 8:
            pitch = min(max(statistics.median(ratios), TOF_FLOOR_PITCH_MIN),
                        TOF_FLOOR_PITCH_MAX)
        for off in range(len(profile)):
            expected[horizon + off] = profile[off] * pitch

    # --- ground rows: only shortfalls against each row's own fitted reference ---
    # `floor_intact` is decided on the NEAR rows (the TOF_FLOOR_NEAR_ROWS rows closest
    # to the bumper, by absolute row index -- not by offset from horizon, since horizon
    # now moves) by quorum, not on all cells by unanimity -- see TOF_FLOOR_NEAR_QUORUM
    # for the measurement that forced this. Occlusion detection runs on every ground
    # cell, near or far: a cell that reads SHORT is an object whichever row it is in.
    near_rows = set(range(TOF_ROWS - TOF_FLOOR_NEAR_ROWS, TOF_ROWS))
    near_seen = near_ok = 0
    cliff: list[tuple[int, int, float]] = []
    for r in range(horizon, TOF_ROWS):
        exp = expected.get(r)
        near = r in near_rows
        for col, v in enumerate(frame[r]):
            if near:
                near_seen += 1
            if not 0 < v < TOF_NO_RETURN_MM:
                # NO RETURN WHERE THE FLOOR SHOULD BE IS A CLIFF -- but only in the
                # near rows, and this is the measurement that makes the distinction
                # legitimate rather than a guess: over 182 frames on ordinary floor,
                # rows 5/6/7 returned 8.00 of 8 cells in EVERY frame, while row 4
                # returned 5.59. Row 4's silence is normal and means nothing. A near
                # row going quiet means the beam went somewhere the floor is not.
                if near and exp is not None:
                    cliff.append((r, col, exp))
                continue
            if exp is None:
                continue                                 # no fitted reference: no claim
            if v < exp - TOF_FLOOR_OBSTACLE_MM:
                out[r, col] = float(v)                   # an occlusion IS the object
            elif v > exp + TOF_CLIFF_BEYOND_MM:
                # THE FLOOR FELL AWAY. Down a step the beam keeps going and comes back
                # long -- the same geometry that makes an obstacle read SHORT makes a
                # drop read LONG, so the two are distinguishable by sign, and this one
                # used to be discarded with a `continue`.
                if near:
                    cliff.append((r, col, exp))
            elif abs(v - exp) > TOF_FLOOR_INTACT_MM:
                continue                                 # off-model but not a drop
            elif near:
                near_ok += 1                             # clean floor, close in
    # A drop-off or a floor that genuinely stopped reading still fails this: it takes
    # the near rows out, not the far one.
    floor_intact = near_seen > 0 and near_ok >= TOF_FLOOR_NEAR_QUORUM * near_seen

    # --- the cliff, if enough of the near floor agrees it is gone -------------------
    #
    # Harsh, 2026-09-04: "I also want her to be able to identify cliff-like obstacle
    # such that she does not fall on the stairs."
    #
    # A drop is entered into `out` AS AN OBSTACLE, at the range where the floor should
    # have been. That is deliberate and it is why this needs no new machinery: the
    # navigator's near-field barrier already steers around everything in this array,
    # and the reflex already stops for it. "Do not drive there" is the same instruction
    # whether the thing there is a table leg or a staircase, and the distance to the
    # EDGE is exactly the distance to the last floor that was still there.
    #
    # The quorum is what keeps a dark rug or one absorbing pixel from becoming a
    # staircase: a real edge crosses the beam pattern and takes several cells at once,
    # a speckle takes one. It is the same argument, and the same order of number, as
    # tof_min_valid_zones -- and the 3-frame median in DFRobotToFThread has to agree
    # as well before any of this reaches the reflex.
    # A DROP THAT TAKES THE WHOLE NEAR FIELD re-fits as "the floor is further away":
    # every row still grows on the expected curve and every cell still matches its own
    # row, so nothing above notices. The bottom row's absolute range is the one thing
    # that cannot be explained away -- see TOF_FLOOR_BOTTOM_MAX_MM.
    bottom = expected.get(TOF_ROWS - 1)
    if bottom is not None and bottom > TOF_FLOOR_BOTTOM_MAX_MM:
        floor_intact = False
        # REPORTED AT WHERE THE FLOOR SHOULD HAVE BEEN, not at the bound that detected
        # it. When the whole near field has gone there is no edge left in view to
        # measure, so the honest range is the nearest place floor was ever seen on this
        # robot -- the measured profile's bottom row. Reporting it at the detection
        # bound instead put the hazard at 300 mm, outside tof_stop_distance, and the
        # reflex drove on through it.
        for col in range(TOF_COLS):
            cliff.append((TOF_ROWS - 1, col, float(TOF_FLOOR_PROFILE_MM[-1])))

    if len(cliff) >= TOF_CLIFF_MIN_CELLS:
        for r, col, exp in cliff:
            out[r, col] = min(float(out[r, col]), float(exp))

    # --- ahead rows: every return is a thing, minus the horizon-row echo ---
    echo = expected.get(horizon)
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
    # nearer than the horizon row's intercept would have occluded the ground rows
    # first. So an ahead-row return closer than the floor we can still see is
    # multipath.
    # THIS FILTER IS FOR THE AHEAD ROWS ONLY. Applying it to the ground rows deleted
    # every genuine near obstacle -- an occlusion reads SHORT, which is exactly what
    # the filter discarded -- and the robot drove into something until the reflex
    # caught it at 16 mm.
    if floor_intact and ahead and echo is not None:
        ahead = [(r, c, v) for r, c, v in ahead if v >= echo]
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
        self.colour_slot = None
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
        # ...and the COLOUR frame, when anyone asked for one. The object detector
        # needs RGB and this thread already has the decoded BGR in hand, so the
        # alternative was decoding the same JPEG twice. Costs a reference; the slot
        # only exists when a consumer created it.
        if self.colour_slot is not None:
            self.colour_slot.publish(frame)

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
        self._rear_mask: np.ndarray | None = None
        self.last_rear_min = float("inf")   # for telemetry; inf until a scan arrives
        self.rear_blocks = 0
        self.last_fwd_min = float("inf")    # k-th smallest ahead, as the law saw it
        self.last_closing = 0.0             # [m/s] rate the forward clearance is shrinking
        self.ttc_brakes = 0
        self._prev_fwd: float | None = None
        self._prev_t: float | None = None
        self._trig: tuple[np.ndarray, np.ndarray] | None = None
        self.last_scrape = float("inf")   # nearest thing in the front hemisphere
        self.last_band = 0                # beams inside the swept corridor
        self.scrape_brakes = 0

    def _beam_trig(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """(cos, sin) of every beam bearing, cached per beam count.

        Beam i looks along i * (360/n) degrees with beam 0 down the nose, so cos is
        how much of a return lies AHEAD and sin how much lies to the SIDE.
        """
        if self._trig is None or self._trig[0].size != n:
            bearing = np.arange(n, dtype=np.float64) * (2.0 * np.pi / n)
            self._trig = (np.cos(bearing), np.sin(bearing))
        return self._trig

    def forward_clearance(self, scan_m: np.ndarray, omega: float = 0.0) -> float:
        """How far she can drive straight before her body reaches something.

        Not "how close is the nearest thing in front" -- that is the question the
        90 deg arc asked, and a hallway answers it with its own walls. See
        cfg.corridor_half_width for the measurement that forced this.

        Two rules, and the second is why this is still safe:

          1. CORRIDOR. Keep only returns inside the strip the body sweeps,
             |d*sin(bearing)| <= half-width, and measure them along the direction of
             travel, d*cos(bearing). A wall 0.6 m to the side is 0.6 m to the side.
          2. SCRAPE GUARD. Anything inside her own footprint radius, at ANY forward
             bearing, still brakes. The corridor rule cannot see the thing she is
             about to catch with a shoulder as she turns; this can.
        """
        c = self.cfg
        scan = self._despeckle(scan_m)
        cos, sin = self._beam_trig(scan.size)
        x = scan * cos
        y = scan * sin
        widen = min(c.corridor_max_widen, abs(float(omega)) * c.corridor_turn_widen)
        half = c.corridor_half + widen
        band = x[(x > 0.0) & (np.abs(y) <= half)]
        self.last_band = int(band.size)
        # The plain minimum, because the scan is already de-speckled. A k-th smallest
        # here would be wrong twice over: far down a corridor the band holds only three
        # or four beams, so the 3rd smallest IS the maximum, and a thin obstacle dead
        # ahead would be averaged away.
        d_min = float(band.min()) if band.size else float(c.lidar_max_range)

        near = scan[x > 0.0]
        self.last_scrape = float(near.min()) if near.size else float("inf")
        if self.last_scrape <= c.scrape_radius and self.last_scrape < d_min:
            self.scrape_brakes += 1
            d_min = self.last_scrape
        return d_min

    def _despeckle(self, scan_m: np.ndarray) -> np.ndarray:
        """Rolling median along the beam axis: a lone short return is deleted.

        The speckle rule this project keeps rediscovering, moved from the STATISTIC to
        the SCAN, which is where it belongs. `k`-th smallest over a set only works when
        the set is big; the swept corridor is a handful of beams when the room is open,
        and there the 3rd smallest of 3 is the maximum. In ANGLE the rule is exact and
        size-independent instead: a real obstacle occupies adjacent beams, a speckle
        does not, and a 3-wide median keeps the first and erases the second.

        np.roll wraps, which is right -- the scan is a circle.
        """
        w = max(1, int(self.cfg.forward_min_valid_beams)) | 1     # force odd
        if w < 3 or scan_m.size < w:
            return scan_m
        r = w // 2
        return np.median(np.stack([np.roll(scan_m, i) for i in range(-r, r + 1)]),
                         axis=0)

    def _rear_mask_for(self, n: int) -> np.ndarray:
        """Beams within +/- governor_rear_arc_deg of the TAIL. Cached like the front."""
        if self._rear_mask is None or self._rear_mask.size != n:
            bearing = (np.arange(n) * (360.0 / n) + 180.0) % 360.0 - 180.0
            self._rear_mask = np.abs(np.abs(bearing) - 180.0) <= self.cfg.governor_rear_arc_deg
        return self._rear_mask

    def __call__(self, v: float, omega: float, scan_m: np.ndarray) -> tuple[float, float, bool]:
        c = self.cfg
        # The braking law gates FORWARD travel, so it must look where she is GOING --
        # down the corridor her body sweeps, not over an angular arc. This was
        # np.min() over the whole sweep first (a wall behind her vetoed driving
        # forward: 299 interventions in 410 ticks), then the front hemisphere (a wall
        # BESIDE her vetoed it: the 2026-09-03 zig-zag). forward_clearance() carries
        # the geometry and the scrape guard that keeps the hemisphere's real value.
        # Rotation and reverse are unaffected -- neither is gated by d_min.
        d_min = self.forward_clearance(scan_m, omega)
        self.last_fwd_min = d_min

        # ---- closing rate: the part a distance-only law cannot see -----------------
        d_eff = self.effective_clearance(d_min)

        v_out = float(np.clip(v, c.v_min, c.v_max))
        omega_out = float(np.clip(omega, -c.omega_max, c.omega_max))
        if d_eff <= c.hard_stop:
            v_out = min(v_out, 0.0)                  # rotate / reverse only
        elif d_eff < c.safe_distance:
            span = max(c.safe_distance - c.hard_stop, 1e-6)
            v_out = min(v_out, c.v_max * (d_eff - c.hard_stop) / span)

        # ---- REVERSE, which until now nothing checked at all -----------------------
        #
        # The clamp above only ever lowers v, so reverse passed through it untouched,
        # and the ToF is a forward-facing patch. A reverse command was therefore
        # executed with NO obstacle check behind the robot, by either sensor. Harsh
        # reported her backing into things; this is why.
        #
        # The gate is BINARY rather than the proportional law used going forward,
        # because proportional braking cannot work in reverse on this drivetrain: the
        # only executable reverse is v_min = -0.150 m/s against a 0.149 m/s stiction
        # floor, so any scaled-down reverse falls under the floor and executes as a
        # full stop anyway. Pretending to brake smoothly would be theatre.
        v_out = self.gate_reverse(v_out, scan_m)
        return v_out, omega_out, v_out != v

    def effective_clearance(self, d_min: float) -> float:
        """Forward clearance discounted by what the closing rate will consume.

        WHY THIS EXISTS: the braking law is a function of distance alone. That is exactly
        right for a wall, which only approaches as fast as the robot drives at it, and
        wrong for anything that moves on its own. Someone walking toward the robot at
        1 m/s crosses the entire 0.19 m band between safe_distance and hard_stop in
        0.19 s -- under two control ticks -- so a distance-only law does not start
        braking until it is already too late to brake gently.

        Subtracting `rate * horizon` makes the law act on where the gap WILL be, so the
        robot slows early and smoothly for something coming toward it and is unchanged
        for something standing still. Only CLOSING counts: a widening gap must never
        grant extra speed, so the discount is one-sided.
        """
        c = self.cfg
        now = time.monotonic()
        rate = 0.0
        if (c.ttc_enabled and self._prev_fwd is not None
                and self._prev_t is not None):
            dt = now - self._prev_t
            # Bound dt: a stalled loop makes a stale sample look like a huge closing
            # rate, which would brake hard for nothing. Bound the rate too -- beyond
            # ttc_max_rate the gap did not close, the beam jumped onto another surface.
            if 1e-3 < dt < 1.0:
                rate = float(np.clip((self._prev_fwd - d_min) / dt,
                                     0.0, c.ttc_max_rate))
        self._prev_fwd, self._prev_t = d_min, now
        self.last_closing = rate
        if rate <= 0.0:
            return d_min
        d_eff = max(d_min - rate * c.ttc_horizon_s, 0.0)
        # Counted only when the discount CHANGED the answer -- crossed the braking
        # threshold that the raw distance had not. "The rate was nonzero" is true on
        # nearly every moving tick and would say nothing.
        if d_eff < c.safe_distance <= d_min:
            self.ttc_brakes += 1
        return d_eff

    def gate_reverse(self, v: float, scan_m: np.ndarray) -> float:
        """Forbid reverse into a blocked rear. Idempotent; safe to apply repeatedly.

        Called from __call__, and AGAIN after the escape and recovery stages, which
        run downstream of the governor and issue reverse of their own -- recovery's
        `_maneuver` returns v_min to back off. Gating only inside __call__ would leave
        exactly the commands most likely to reverse into something ungated, which is
        the case Harsh actually hit.
        """
        rear = scan_m[self._rear_mask_for(scan_m.size)]
        if rear.size:
            # k-th smallest, NOT the minimum. One spurious beam must not veto the
            # robot's main escape maneuver. Measured over a 90 s run: 5 of 316 ticks
            # carried a return below 0.22 m -- geometrically impossible, since that is
            # inside her own body radius -- and a min-based gate turns each of those
            # into a refused reverse. This is the same reasoning ToFReflex applies to
            # its 8x8 patch, and the same speckle trap that made a min-based corridor
            # metric report 0.57 m of clearance where a 60-scan median saw 4.78 m.
            k = min(self.cfg.rear_min_valid_beams, rear.size) - 1
            self.last_rear_min = float(np.sort(rear)[k])
        else:
            self.last_rear_min = float("inf")
        if v < 0.0 and self.last_rear_min <= self.cfg.rear_stop:
            self.rear_blocks += 1
            return 0.0
        return v


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
            # Never rate-limit a stop. Forward is killed outright; a REVERSE the
            # caller has already sanctioned is passed through, because slewing an
            # escape is the same mistake as slewing the stop itself.
            self.v = min(v, 0.0)
            if not allow_rotation:
                self.omega = 0.0
                return self.v, 0.0
            # The caller has established that turning in place cannot reach the
            # obstacle (it lies outside robot_radius), so rotation is the escape and
            # must survive the stop. It is deliberately NOT slew limited either --
            # the same reasoning that makes a stop immediate makes turning AWAY from
            # something immediate. Forward speed stays vetoed regardless; self.v is
            # already clamped to <= 0 above, so a sanctioned reverse survives here too.
            self.omega = omega
            return self.v, omega
        dv = self.cfg.max_lin_accel * dt
        dw = self.cfg.max_ang_accel * dt
        # Slowing down is always allowed at full rate; only acceleration is limited.
        self.v = v if abs(v) < abs(self.v) else float(np.clip(v, self.v - dv, self.v + dv))
        # A PIVOT IN PLACE IS ISSUED AT ONCE. See RobotConfig.pivot_slew_free: the slew
        # limit protects dead reckoning against wheel slip under a step change in
        # TRANSLATION, and a robot rotating about its own centre translates nothing.
        # The limit's real effect here was to spend the first tick of every turn at
        # 0.40 rad/s, which this drivetrain executes as a full stop.
        #
        # Tested on |v| AFTER the linear slew, because that is the speed actually going
        # to the wheels this tick -- testing the request would exempt a turn whose
        # forward component the limiter is still ramping down, which is a translating
        # robot however the command reads.
        if self.cfg.pivot_slew_free and abs(self.v) <= self.cfg.pivot_slew_v_eps:
            w = float(np.clip(omega, -self.cfg.omega_max, self.cfg.omega_max))
            # ONE EXCEPTION: a sign flip goes through zero first. Reversing a spinning
            # motor at pivot duty is plugging -- the back-EMF adds to the supply and the
            # winding sees roughly twice stall current. Four motors on two L298N
            # channels is already about double the per-channel load this board is rated
            # for, and a brownout mid-drive resets the Pi, which is the one failure this
            # whole project exists to avoid. Costing one tick to cross zero is cheap;
            # the magnitude change that mattered for hesitancy is still immediate.
            if w * self.omega < 0.0:
                w = 0.0
            self.omega = w
        else:
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
    # Opening the port resets the board (DTR), so the identity query has to survive the
    # bootloader. Four tries at 0.75 s on top of the initial 2.0 s settle -- comfortably
    # past the ~2 s boot, and still a hard refusal if nothing ever identifies itself.
    HANDSHAKE_TRIES = 4
    HANDSHAKE_RETRY_S = 0.75

    def __init__(self, cfg: RobotConfig, port: str, baud: int = 115200,
                 max_duty: float = 0.60, watchdog_s: float = 0.5,
                 invert_left: bool = False, invert_right: bool = False,
                 min_duty: int = 0, kick_ticks: int = 2, kick_scale: float = 1.6,
                 creep_cutoff: float = 0.5, turn_min_duty: int | None = None,
                 turn_min_omega: float = 0.15,
                 turn_kick_duty: float | None = None,
                 turn_kick_cooldown_s: float = 1.0,
                 pulse_creep_duty: int = 0, pulse_burst_ticks: int = 1,
                 kick_rest_s: float = 0.0, plug_guard: bool = False,
                 steady_pivots: bool = False, pivot_hold_ticks: int = 3,
                 start_debounce_ticks: int = 1) -> None:
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
        # The smallest rotation worth executing, in the units the request is actually
        # made in. Expressed as rad/s and converted here, because a duty threshold for
        # a pivot is meaningless on its own -- it depends on the wheel base, the speed
        # constant and the duty ceiling, all of which have changed at least once.
        # Everything at or above it executes at turn_min_duty; below it is a heading
        # nudge too small to be worth a break-away lurch, and is dropped.
        self.turn_min_omega = max(0.0, turn_min_omega)
        self.turn_cutoff_duty = (255.0 * self.max_duty
                                 * (self.turn_min_omega * self.cfg.wheel_base / 2.0)
                                 / max(self.cfg.wheel_speed_max, 1e-6))
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
        # QUIETER MOTORS (Harsh, 2026-09-28: "the motors are like making too noise").
        # MEASURED in the 23:24 run (190 s): 252 motor starts, 159 of them after a
        # single stopped tick; 102 of those one-tick gaps were the creep pulse
        # coasting while the chain still asked to move. ~300 ticks ran at kick
        # duty (216-229) because the standstill kick fired after EVERY stop, however
        # short -- the pivot kick's 1 s cooldown never applied to it. 51 times a
        # powered wheel went straight from one direction to the other (plugging:
        # a clunk and a current spike on a battery that has cut the Pi twice).
        # Each fix only ever REMOVES energy from the wheels; a stop request is
        # honoured on the tick it arrives, exactly as before.
        #   pulse_burst_ticks -- a creep pulse lasts this many ticks, with the same
        #       average speed (1 = the old one-tick pulse)
        #   kick_rest_s -- the standstill kick needs this long stopped first; a
        #       robot still coasting has no static friction to break (0 = old)
        #   plug_guard -- a wheel about to reverse under power coasts one tick
        self.pulse_burst_ticks = max(1, int(pulse_burst_ticks))
        self._burst_left = 0
        self.kick_rest_s = max(0.0, float(kick_rest_s))
        self._stopped_at = -1e9                   # a fresh driver has been at rest
        self.plug_guard = bool(plug_guard)
        self._prev_duties = (0, 0)
        # STEADY TURNS (Harsh, 2026-10-04: "they make noise more than movement
        # ... very harsh movements, it should be smooth"). MEASURED in the 490 Hz
        # runs' black boxes: 71-144 stop<->move switches a minute, nearly all in
        # open space with forward cut by the governor and a TURN IN PLACE asked
        # for -- and the creep pulse below chopped that turn into on/off stabs at
        # the pivot floor (190+ duty for a tick, then a stop), because it compares
        # each wheel against the STRAIGHT floor before the pivot rule ("at or above
        # turn_min_omega executes at turn_min_duty, below it is dropped") is ever
        # reached. With steady_pivots:
        #   * a pivot is never pulsed: worth turning -> steady at the pivot floor;
        #     a nudge below turn_min_omega -> dropped, as the rule always said;
        #   * a pivot once started holds pivot_hold_ticks if the request dips
        #     just under the floor IN THE SAME DIRECTION -- never through a stop
        #     request (0, 0), never into the other direction, never into driving.
        # Forward creep keeps its pulses: that is the governor's braking near an
        # obstacle and it must not become a lurch at floor speed.
        self.steady_pivots = bool(steady_pivots)
        self.pivot_hold_ticks = max(0, int(pivot_hold_ticks))
        # THE NEXT LAYER (Harsh, 2026-10-04: "DO whats still left"): the rest of
        # the stop-go was in the COMMAND stream, measured in the same black boxes:
        #   (Tried and REMOVED: running a too-slow forward arc as a turn in place.
        #   Replayed, it made the stop/go WORSE -- 140 vs 133 switches a minute.)
        #   * one-tick stop/go blinks between stages. start_debounce_ticks: from a
        #     standstill, motion starts only once it has been asked for that many
        #     ticks running. A STOP is never delayed -- this only ever turns a start
        #     into a stop, i.e. removes energy, like every other fix here.
        self.start_debounce_ticks = max(1, int(start_debounce_ticks))
        self._go_ticks = 0
        self._pivot_hold = 0
        self._pivot_sign = 0
        self.noise_stats = {"starts": 0, "kicks": 0, "plug_coasts": 0,
                            "pulse_bursts": 0, "pivot_holds": 0,
                            "starts_debounced": 0}
        self._err_count = 0
        self._cmds = 0
        self._closed = False
        # GYRO (Harsh, 2026-10-07; firmware GYRO 2+): one "G" a tick, its reply
        # read on the NEXT tick's drain -- no round trip inside the budget. The
        # yaw is integrated ON THE BOARD at 100 Hz, so a reply a tick late
        # loses no rotation: the loop uses differences between replies.
        self._rx = b""
        self.last_imu: dict | None = None
        self.last_imu_t = 0.0
        self.imu_replies = 0
        self.tilt_replies = 0
        self._imu_parse = None

        # The Uno resets when DTR toggles on open; nothing sent during the bootloader
        # window is heard, so wait it out before the handshake.
        #
        # A DEAD ADAPTER MUST NOT LOOK LIKE A BUG IN THIS FILE. /dev/ttyUSB0 survives
        # the FTDI falling off the USB bus -- the node is created at boot and the
        # kernel does not tear it down -- so resolve_serial_port() finds a perfectly
        # good-looking symlink and the open then fails deep inside pyserial's DTR
        # handling with a bare OSError(5) and a traceback that points at serialposix.
        # MEASURED 2026-09-03: dmesg had "Unable to read latency timer: -32" seven
        # seconds after boot and "device not accepting address, error -71" repeatedly
        # after that, while lsusb showed the FTDI flapping in and out of existence.
        # Nothing in software can fix that, so the least it can do is say so.
        try:
            self._ser = serial.Serial(port, baud, timeout=1.0, write_timeout=0.2)
        except OSError as exc:
            raise RuntimeError(
                f"Could not open the motor controller on {port}: {exc}. The device "
                f"node exists but the adapter is not answering, which on this robot "
                f"means the FTDI has dropped off the USB bus rather than that "
                f"anything is wrong with this program -- the node outlives the "
                f"device. Confirm with `lsusb | grep FTDI` (absent = gone) and "
                f"`dmesg | tail` (errors -32/-71 = the adapter is flapping). It is a "
                f"cable, connector or hub-power fault: reseat the FTDI's USB "
                f"connection. The wheels stay stopped until it is back."
            ) from exc
        time.sleep(2.0)
        self._ser.reset_input_buffer()

        # ONE QUERY AT A FIXED DELAY IS A RACE AGAINST THE BOOTLOADER, NOT A CHECK.
        # Opening this port asserts DTR, which RESETS the Arduino; the bootloader then
        # takes ~1.5-2 s before the sketch answers anything at all. The fixed 2.0 s
        # above is right at that edge, so on a loaded Pi it is a coin flip.
        # MEASURED 2026-09-04: a run handshook fine at 04:44, an identical one at 05:07
        # got '' and refused to drive, and a manual probe moments later -- same port,
        # same board, 2.5 s of patience -- answered 'FRIDAY-MOTORS 1' on the first try.
        # A transient must not read as "the board is not Friday's".
        #
        # THE GATE IS NOT WEAKENED. An unidentified board still refuses; it is simply
        # asked more than once before that conclusion is drawn.
        reply = ""
        for attempt in range(self.HANDSHAKE_TRIES):
            reply = self._command("P")
            if reply.startswith(self.IDENT):
                if attempt:
                    LOGGER.info("Motor board answered on handshake attempt %d/%d "
                                "(it was still booting)", attempt + 1,
                                self.HANDSHAKE_TRIES)
                break
            if attempt + 1 < self.HANDSHAKE_TRIES:
                time.sleep(self.HANDSHAKE_RETRY_S)
                self._ser.reset_input_buffer()
        if not reply.startswith(self.IDENT):
            self._ser.close()
            raise RuntimeError(
                f"{port} answered {reply!r}, not {self.IDENT}, on {self.HANDSHAKE_TRIES} "
                f"attempts over {2.0 + self.HANDSHAKE_TRIES * self.HANDSHAKE_RETRY_S:.1f} s. "
                f"Refusing to drive an unidentified board -- flash friday_motors.ino, or "
                f"check you are on the Arduino's port (yours is the FTDI FT232R, not the "
                f"LiDAR's CP2102)."
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

    def _mps_to_duty(self, mps: float, floor: int | None = None,
                     cutoff_duty: float | None = None) -> int:
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
            #
            # A PIVOT GETS ITS OWN, LOWER CUTOFF (`cutoff_duty`). The argument above is
            # about TRANSLATION: the thing that must not become a lurch is the
            # governor's proportional braking as the robot closes on an obstacle. A
            # pivot has no proportional band to protect -- it does not translate, so it
            # approaches nothing -- and the straight cutoff of 40 was killing every
            # rotation under 0.425 rad/s outright. MEASURED over the 3675 ticks logged
            # on 2026-09-03: 797 (22%) commanded a pure turn that came out as a stop.
            cut = self.min_duty * self.creep_cutoff if cutoff_duty is None else cutoff_duty
            if abs(duty) < cut:
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
        if self.steady_pivots and self._is_pivot(left, right):
            self._burst_left = 0
            return left, right                      # steady turns: never pulsed
        if pulse <= 0.0 or peak <= 0.0 or peak >= self.min_speed_mps():
            self._burst_left = 0
            if peak <= 0.0:
                self._pulse_acc = 0.0               # a stop owes no pulses
            return left, right                      # fast enough to execute directly
        self._pulse_acc += min(peak / pulse, 1.0)
        # A burst of pulse_burst_ticks, started when one tick's worth has built up
        # (as responsive as the old pulse) and paid for afterwards: the accumulator
        # may go negative during a burst, and the coast that follows is longer. The
        # average is the request, with a third as many starts at 3.
        if self._burst_left > 0:
            self._burst_left -= 1
        elif self._pulse_acc >= 1.0:
            self._burst_left = self.pulse_burst_ticks - 1
            self.noise_stats["pulse_bursts"] += 1
        else:
            return 0.0, 0.0                         # coast this tick
        self._pulse_acc -= 1.0
        scale = pulse / peak                        # both wheels: geometry preserved
        return left * scale, right * scale

    @staticmethod
    def _is_pivot(left: float, right: float) -> bool:
        return (left > 0) != (right > 0) and abs(left) > 1e-6 and abs(right) > 1e-6

    def _apply(self, left: float, right: float) -> None:
        if self.start_debounce_ticks > 1:
            if left == 0.0 and right == 0.0:
                self._go_ticks = 0                  # a stop: always at once
            elif not self._moving:
                self._go_ticks += 1
                if self._go_ticks < self.start_debounce_ticks:
                    self.noise_stats["starts_debounced"] += 1
                    left = right = 0.0              # not yet: a start, held back
        asked_pivot_sign = (0 if not self._is_pivot(left, right)
                            else (1 if right > left else -1))
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
        cut = self.turn_cutoff_duty if pivot else None
        l_duty = self._mps_to_duty(left, floor, cut)
        r_duty = self._mps_to_duty(right, floor, cut)

        if self.steady_pivots:
            if pivot and (l_duty or r_duty):
                if (1 if r_duty > l_duty else -1) != self._pivot_sign:
                    self._pivot_hold = self.pivot_hold_ticks  # a new turn: commit
                self._pivot_sign = 1 if r_duty > l_duty else -1
            elif (self._pivot_hold > 0 and l_duty == 0 and r_duty == 0
                  and asked_pivot_sign == self._pivot_sign):
                # The same turn, dipped under the floor for a tick: keep turning
                # steadily rather than stop and kick back in. Only for a nonzero
                # same-direction turn request -- a stop (0, 0) never lands here.
                self._pivot_hold -= 1
                self.noise_stats["pivot_holds"] += 1
                d = min(self.turn_min_duty, int(255 * self.max_duty))
                l_duty, r_duty = -self._pivot_sign * d, self._pivot_sign * d
                pivot = True
            else:
                self._pivot_hold, self._pivot_sign = 0, 0

        # No plugging: a wheel that would go straight from one direction to the
        # other under power coasts for this tick instead. Only ever a stop.
        pl, pr = self._prev_duties
        if (self.plug_guard and (l_duty or r_duty)
                and (l_duty * pl < 0 or r_duty * pr < 0)):
            self.noise_stats["plug_coasts"] += 1
            l_duty = r_duty = 0
            self._burst_left = 0

        if l_duty == 0 and r_duty == 0:
            if self._moving:
                self._stopped_at = now
            self._moving = False
            self._was_pivot = False
            self._kick_left = 0
            self._prev_duties = (0, 0)
            self.last_executed = (0.0, 0.0)               # the creep cutoff bites here
            self.last_duties = (0, 0)
            self._write_raw("S")                          # firmware's dedicated stop
            return

        # Boost out of standstill, spread across ticks so the loop never blocks.
        # A REAL standstill: after a one-tick coast she is still rolling and there
        # is no static friction to break -- kicking there was most of the noise.
        if not self._moving:
            self.noise_stats["starts"] += 1
        rested = now - self._stopped_at >= self.kick_rest_s
        if (not self._moving and rested) or entering_pivot:
            self._kick_left = self.kick_ticks
            self.noise_stats["kicks"] += 1
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
        self._prev_duties = (l_duty, r_duty)

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
            self._drain(line)
        except Exception as exc:                          # noqa: BLE001
            LOGGER.error("Motor write failed: %s", exc)

    def request_imu(self) -> None:
        """Ask the board for its gyro + accelerometer (firmware GYRO 2+). The
        reply is picked up by the next drain. NOT a motor command: last_wire
        and the command count are untouched, and the firmware does not count
        G as a keepalive, so this can never keep a move alive."""
        with self._lock:
            if self._closed or self._ser is None or not self._ser.is_open:
                return
            try:
                self._ser.write(b"G\n")
                self._drain("G")
            except Exception as exc:                      # noqa: BLE001
                LOGGER.debug("gyro request failed: %s", exc)

    def _drain(self, line: str) -> None:
        """Read what the board has said, a whole line at a time."""
        pending = self._ser.in_waiting
        if not pending:
            return
        self._rx += self._ser.read(pending)
        *lines, self._rx = self._rx.split(b"\n")
        for raw in lines:
            s = raw.decode(errors="replace").strip()
            if s.startswith("G "):
                imu = self._parse_imu(s)
                if imu is not None:
                    self.last_imu, self.last_imu_t = imu, time.monotonic()
                    self.imu_replies += 1
            elif s == "TILT":
                self.tilt_replies += 1
            elif s == "ERR":
                self._err_count += 1
                if self._err_count in (1, 10) or self._err_count % 100 == 0:
                    LOGGER.warning("Motor firmware returned ERR (%d of %d commands) "
                                   "-- last sent: %r", self._err_count, self._cmds, line)

    def _parse_imu(self, reply: str) -> dict | None:
        """motor.parse_imu -- the ONE place config.GYRO_SIGN is applied."""
        if self._imu_parse is None:
            if str(HERE.parent) not in sys.path:
                sys.path.append(str(HERE.parent))
            import motor as _friday_motor
            self._imu_parse = _friday_motor.parse_imu
        return self._imu_parse(reply)

    def imu(self, max_age_s: float = 0.5) -> dict | None:
        """The latest reading if it is fresh, else None."""
        if self.last_imu is None or time.monotonic() - self.last_imu_t > max_age_s:
            return None
        return self.last_imu

    def close(self) -> None:
        """Idempotent: safe to call from several shutdown paths."""
        if self._closed:
            return
        LOGGER.info("Motor noise: %s", self.noise_stats)
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
              "infer_ms", "explored_m2", "recovery",
              # WHAT THE BRAKING LAW ACTUALLY SAW. `lidar_min` is the whole-360 raw
              # minimum and is kept unchanged so old files stay comparable -- but it is
              # NOT the number any decision was made on, and reading it as though it
              # were is how a run spent in a tight corner was misdiagnosed as a sensor
              # fault. `fwd_min` is the k-th smallest in the forward arc, which is the
              # law's input; `closing` is the rate that gap is shrinking, which is the
              # part a distance-only law is blind to.
              "fwd_min", "closing",
              # AND WHAT THE REAR GATE SAW. Its absence is why "rear=0" across 334
              # reversing ticks read as "nothing was ever behind her" instead of as
              # "the threshold was too tight to fire" -- there was no column that could
              # tell those apart. Reverse is single-sensor; it needs its own evidence.
              "rear_min",
              # WHAT THE CAMERAS SAW. The zig-zag analysis could not answer "was the
              # camera pointing her somewhere sensible" because nothing recorded it --
              # and for the first months of this project the honest answer was that
              # there was no camera at all (cv2 was missing from the venv, so every
              # frame was None). `cam_conf` is the column that makes that visible:
              # a run of zeros means she is blind, whatever the logs say.
              "cam_bearing", "cam_range", "cam_conf")

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
        # THE WAY BACK (Harsh, 2026-09-26). When the time is up she keeps going
        # EXACTLY as she explored -- policy, navigator, reflexes, recovery,
        # narration -- with the start as the planner's target instead of a
        # frontier, then stops there. See rl/way_back.py.
        self.way_back: "_way_back.WayBack | None" = None
        self.way_back_enabled = True
        self._home_target: "_way_back.HomeTarget | None" = None
        # UNDO THE LAST MOVE (Harsh, 2026-09-28). When the ToF reflex, its turn
        # freeze and the rear gate all say "no" at once, she retraces the move that
        # got her there -- space her own body has just left -- and after two tries
        # at one place asks to be moved. See rl/unstick.py.
        self.unstick_enabled = True
        self.unstick = _unstick.Unstick(cfg.robot_radius, cfg.v_min, cfg.omega_max,
                                        cfg.tof_pivot_min_clearance)
        self._undo_active = False
        self._avoid_seen = 0
        # TURN COMMITMENT (Harsh, 2026-10-04): see _commit_turn. 0 = off.
        self.turn_commit_s = 0.0
        self._turn_sign = 0
        self._turn_since = -1e9
        # FILE 21 HOLD (Harsh, 2026-09-29): the exploration mission may ask her to
        # stand still -- to ask what an area is called, for "go to X", for a pause.
        # Inert unless state/exploration_mission/hold.json is fresh. rl/way_back.py.
        self.hold = _way_back.Hold()
        # FILE 21 U1: the mission's drive order (a path, a turn, a back-up) while
        # it holds the run -- see rl/way_back.py MissionDrive. Never while exploring.
        self.mission_drive = _way_back.MissionDrive()
        # The gyro (stage 6) and her body (stage 4d): nothing until the board
        # answers a G -- NullMotorDriver and --simulate never do.
        self._gyro_prev: tuple[int, float] | None = None
        self.gyro_gaps = 0          # fallback ticks so far: the tracker's "turn known" test
        self.gyro_seen = False
        self._stall_ticks = 0
        self._body = None
        self._body_seq = 0
        self._body_publish = None
        # LOCAL NAVIGATION. The planner says WHERE (a frontier, at 0.5 Hz); this says
        # HOW to get there right now, by simulating candidate arcs against the live
        # scan at full control rate. It sits between the policy and the governor and,
        # like every stage there, may only ever reduce the command. See navigator.py
        # for the measurement that motivated it (515-1799 deg of turning per metre).
        self.navigator = Navigator(cfg, DynamicWindowPlanner(cfg))
        self.navigator_enabled = True
        # Broader stuck detection than the boxed-in check below. It sees oscillation,
        # spinning and no-progress, which are invisible in any single tick.
        self.recovery = RecoveryController(cfg, self.world)
        self.recovery_enabled = True
        self.governor = LidarGovernor(cfg)
        self.reflex = ToFReflex(cfg)
        self.visual = VisualHazardMonitor(cfg, baseline)   # None => fail-open, disabled
        # Where the space is, from the cameras. Advisory while the LiDAR is healthy;
        # the only thing steering when it is not. See friday_freespace.
        self.freespace = FreeSpaceNavigator()
        self.free_view = self.freespace.last
        self._cam_seen: dict[str, int] = {}
        # Up camera: measured image statistics, reported and never acted on. Explicitly
        # NOT object detection -- nothing here names what is in view, because nothing
        # that could is implemented.
        self.context = _cameras.ContextMonitor()
        # Set by main() so telemetry can report per-role device paths and reconnects.
        self.cam_threads: list = []
        self.detector = None
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
        # Ring buffer. 30 s at 10 Hz is 316 rows, and a longer run silently loses its
        # BEGINNING -- which is where a run that starts badly puts its evidence.
        # Sized from --blackbox-seconds so a long exploration can keep all of it.
        self.blackbox = BlackBox(getattr(cfg, 'blackbox_seconds', 30.0), cfg.control_hz)

        self.last_action = np.zeros(2, np.float32)
        self._obs = np.zeros(cfg.obs_dim, np.float32)
        self._stop = threading.Event()
        self._reflex_thread: threading.Thread | None = None
        self.stats = {"ticks": 0, "reflex_stops": 0, "visual_stops": 0,
                      "governor_limits": 0, "stale_stops": 0, "overruns": 0,
                      "no_reverse_turns": 0, "escape_turns": 0, "planner_biases": 0,
                      "reverse_floor_lifts": 0,
                      "recovery_ticks": 0,
                      # Local navigator. `nav_handoffs` is the honest one: ticks where
                      # nothing forward was drivable and the old behaviour kept the
                      # tick. In an open room it should be near zero.
                      "nav_decisions": 0, "nav_handoffs": 0,
                      "nav_open": 0, "nav_corridor": 0, "nav_cluttered": 0,
                      "nav_blocked": 0}
        self.loop_ms: deque[float] = deque(maxlen=200)
        # WHERE THE BUDGET ACTUALLY GOES, accumulated over the whole run.
        #
        # This loop has missed its 100 ms deadline on the majority of ticks in every
        # run tonight, and the cause was guessed at three times -- the policy, the
        # cameras, the arc search -- with a six-minute hardware run spent per guess and
        # every guess wrong (inference measured 4.5 ms mean against a 160 ms tick).
        # A perf_counter pair per stage costs microseconds and answers it directly.
        #
        # The number that matters most is the LAST one printed: stage total vs wall
        # clock. Everything here runs on ONE Python thread, so a large gap between the
        # two is not this loop being slow -- it is this loop being DESCHEDULED, by the
        # mapper, the frontier planner or the camera threads holding the GIL.
        self.stage_s: dict[str, float] = {}
        self.stage_wall_s = 0.0
        self._world_stats: dict | None = None
        self._world_stats_t = 0.0
        self._last_tick_t: float | None = None
        # Populated by every tick, including the fail-safe early returns. Exists from
        # construction so a status render before the first tick is empty, not a crash.
        self.last_record: tlm.DecisionRecord | None = None

    # ------------------------------------------------------------------ reflex ------
    def relocalise(self, timeout_s: float = 4.0) -> bool:
        """Find the robot in the map it just loaded. True if it succeeded.

        Reuses Friday's own SLAM matcher (`slam.core.match_pose` over a
        `LikelihoodField`) rather than a second implementation -- two matchers that
        could disagree about where the robot is would be worse than none.

        The prior is the pose the map was SAVED at, which is where she was when the
        last run ended. That is the honest assumption: she is usually left where she
        stopped. If she was picked up and carried, the match fails and the caller
        throws the map away, which is the correct outcome and not an error.

        The search window is deliberately far wider than SLAM's own tracking window
        (0.20 m / 12 deg): tracking corrects a tick of drift, this has to absorb
        being nudged between sessions.
        """
        try:
            from slam.core import LikelihoodField, match_pose
            # The repo-root config.py, which is what slam/core.py itself imports --
            # there is no slam.config. SLAM_MIN_SCORE lives there.
            import config as friday_config
        except Exception as exc:                      # noqa: BLE001
            LOGGER.warning("Relocalisation unavailable (%s); cannot place the robot "
                           "in the restored map.", exc)
            return False

        prior = self.world.loaded_pose
        if prior is None:
            LOGGER.warning("The restored map has no saved pose, so there is no prior "
                           "to relocalise from. Maps written before this existed "
                           "cannot be placed.")
            return False

        deadline = time.monotonic() + timeout_s
        scan = None
        while time.monotonic() < deadline:
            candidate, age = self.slots["lidar"].read()
            if candidate is not None and age < self.cfg.lidar_deadline:
                scan = np.asarray(candidate, np.float64)
                break
            time.sleep(0.05)
        if scan is None:
            LOGGER.warning("No LiDAR scan within %.1f s; cannot relocalise.", timeout_s)
            return False

        # A map worth matching against, and a sweep worth matching with. Both floors
        # are measured constants with their reasons recorded in config.py: below
        # MIN_CELLS "the room is still mostly a rumour", and without MIN_BEAMS "a pose
        # that puts almost the whole sweep into unmapped space can score 1.0 on the
        # three beams that happened to land on a known wall".
        min_cells = getattr(friday_config, "SLAM_MATCH_MIN_CELLS", 150)
        min_beams = getattr(friday_config, "SLAM_MATCH_MIN_BEAMS", 40)
        n_occupied = int(self.world.grid.occupied_mask().sum())
        if n_occupied < min_cells:
            LOGGER.warning("The restored map has only %d occupied cells (need %d); "
                           "too little structure to relocalise against.",
                           n_occupied, min_cells)
            return False

        n = scan.size
        bearing = np.arange(n) * (2.0 * math.pi / n)
        good = scan < self.cfg.lidar_max_range - 0.02
        if int(good.sum()) < min_beams:
            LOGGER.warning("Only %d usable beams (need %d); too few to relocalise.",
                           int(good.sum()), min_beams)
            return False
        pts = np.stack([scan[good] * np.cos(bearing[good]),
                        scan[good] * np.sin(bearing[good])], axis=1)

        field = LikelihoodField.from_grid(self.world.grid)
        (x, y, th), score, sharp = match_pose(
            field, pts, prior=(float(prior[0]), float(prior[1]), float(prior[2])),
            window_m=self.cfg.relocalise_window_m,
            window_deg=self.cfg.relocalise_window_deg)

        min_score = getattr(friday_config, "SLAM_MIN_SCORE", 0.15)
        min_sharp = getattr(friday_config, "SLAM_POSE_MIN_SHARPNESS", 1.4)
        moved_m = math.hypot(x - prior[0], y - prior[1])
        moved_deg = abs(math.degrees(th - prior[2]))
        ok = score >= min_score and sharp >= min_sharp
        why = ""

        # THE SCORE GATE ALONE IS A RATCHET THAT KEEPS THE MAP EMPTY FOREVER.
        #
        # `score` is how much of the sweep lands on cells the map already calls
        # occupied, so it measures ALIGNMENT TIMES MAP COMPLETENESS. `sharp` is peak
        # over median across the search window, which is alignment ALONE. On a young
        # map most of the sweep necessarily lands on cells nobody has written yet, so
        # the score is held down by how little has been mapped -- and requiring 0.15 of
        # it means a sparse map can never be relocalised, so it is discarded, so it
        # stays sparse. Every run starts from nothing and the map never gets to grow.
        #
        # MEASURED 2026-09-04 16:55 on the 113.9 m^2 / 207-cell map: score 0.044 --
        # REJECTED -- with sharpness 11.27 against a floor of 1.4, and the winning pose
        # 0.02 m and 0 deg from the saved one. The match was as unambiguous as this
        # matcher ever reports and it landed exactly where the robot was parked. That
        # is not a failed relocalisation; it is a correct one being judged by the
        # wrong statistic.
        #
        # So on a map too sparse for the score to mean anything, judge it on the two
        # things that do: an unambiguous peak, and a peak that agrees with where she
        # was left. Both are required, and the displacement bound is tight -- if she
        # has been carried anywhere, either the peak moves outside it or the window
        # does not contain the truth and the peak goes flat.
        # DENSITY WAS THE WRONG GATE ON THIS BRANCH, and one hardware run said so.
        # The first version only allowed it below 450 occupied cells, on the assumption
        # that `score` becomes meaningful once a map has some structure. It does not:
        # the map reached 753 cells and still scored 0.058, because score is the
        # fraction of THE SWEEP that lands on already-occupied cells, and an indoor map
        # is nearly all free space no matter how long you drive. The threshold merely
        # decided which runs got to keep their map, and the good one was rejected.
        #
        # What actually carries the argument is the pair below, and neither mentions
        # density: an UNAMBIGUOUS peak, and a peak that agrees with where she was
        # parked. If she has been carried anywhere, the sweep matches nothing in the
        # window and the surface goes flat -- sharpness is peak over median, so a
        # failed match cannot fake it. The bar is set above min_sharp deliberately:
        # 1.4 is the floor for trusting a pose you already believe, and this is being
        # used to overrule a scoring gate, which is a stronger claim.
        if not ok and sharp >= RELOC_TRUST_SHARPNESS \
                and moved_m <= RELOC_TRUST_M and moved_deg <= RELOC_TRUST_DEG:
            ok = True
            why = (f" [accepted on sharpness {sharp:.1f} at {moved_m:.2f} m from the "
                   f"saved pose: with {n_occupied} occupied cells the score measures "
                   f"how little is mapped, not how badly this fits]")

        LOGGER.info("Relocalisation: score %.3f (min %.2f), sharpness %.2f (min %.2f), "
                    "moved %.2f m / %.0f deg from the saved pose -- %s%s",
                    score, min_score, sharp, min_sharp, moved_m, moved_deg,
                    "ACCEPTED" if ok else "REJECTED", why)
        if not ok:
            return False
        self.world.set_pose(x, y, th)
        self.map.set_pose(x, y, th)
        return True

    def _archive_map(self) -> str:
        """Move the unplaceable map aside, with its pose and visit trace, and say where.

        Renamed rather than copied: the rebuild is about to write to the same path, and
        a copy that gets overwritten a second later is not a backup.
        """
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = self.world.map_file
        dest = base.parent / f"map-archive-{stamp}"
        moved = []
        try:
            dest.mkdir(parents=True, exist_ok=True)
            for path in (base, base.with_suffix(".pose.json"),
                         base.with_suffix(".visits.npz")):
                if path.exists():
                    path.rename(dest / path.name)
                    moved.append(path.name)
        except OSError as exc:
            # Never let bookkeeping stop the run. Worst case the rebuild overwrites it,
            # which is exactly what happened before this method existed.
            LOGGER.warning("Could not archive the old map (%s); it will be overwritten.",
                           exc)
            return "archiving FAILED -- the file will be overwritten"
        return f"{dest} ({', '.join(moved) or 'nothing to move'})"

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
                if emergency and self._fast_reflex_should_stop():
                    self.motors.emergency_stop()
                    self.limiter.reset()
            time.sleep(period)

    def _fast_reflex_should_stop(self) -> bool:
        """The 30 Hz reflex applies the TICK's emergency rule, not a harsher one.

        Found 2026-09-28. The tick's reflex was changed on 2026-09-02 (approved by
        Harsh) to let a sanctioned REVERSE through an emergency, and a PIVOT when the
        bumper is clear of tof_pivot_min_clearance -- but this thread was not, and it
        stopped the motors 30 times a second whatever the tick had decided. So every
        escape the chain allowed near an obstacle ran for 0-33 ms of each 100 ms
        tick. Measured in the black boxes: sanctioned pivots held for 1.8 s with the
        bumper reading 0.142-0.148 m (a robot turning at 0.8 rad/s would have swung
        ~80 deg); on 2026-09-26 839 sanctioned reverses left her 2-13 mm from what
        she was touching. The limiter had the same bug once and was fixed with it;
        this thread was missed.

        Forward is still killed at once -- that is what this thread is for.
        """
        v, w = self.motors.last_cmd
        if v > 1e-6:
            return True
        if (abs(w) > 1e-6 and not self._undo_active
                and self.reflex.last_min <= self.cfg.tof_pivot_min_clearance):
            return True
        return False

    # ------------------------------------------------------------ observation ------
    def _lidar_body_rejects(self) -> int:
        """How often a return inside the footprint was dropped, outside the self-mask.

        Reported rather than silently swallowed: the rejection is a safety net, and a
        safety net that is catching a lot is telling you the thing above it has moved.
        """
        for t in getattr(self, "_threads", ()) or ():
            n = getattr(t, "_body_rejects", None)
            if n is not None:
                return int(n)
        return 0

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

    # ------------------------------------------------------------ cameras ---------
    def _read_cameras(self) -> dict:
        """Both camera slots, with anything older than cam_deadline dropped."""
        frames: dict = {}
        if not self.cfg.cam_enabled:
            return frames
        for key in ("cam_up", "cam_down"):
            if key in self.slots:
                fr, age = self.slots[key].read()
                frames[key] = (fr if (fr is not None and age <= self.cfg.cam_deadline)
                               else None)
        return frames

    def _update_freespace(self, frames: dict):
        """Re-run the free-space estimate, but only on frames we have not seen.

        The cameras publish at 15 fps and this loop runs at 10 Hz, so roughly a third
        of ticks would otherwise re-analyse an image whose answer is already known.
        Each analysis is ~3.6 ms measured; two cameras every tick is 7 ms of a budget
        this loop already misses on 47% of ticks.
        """
        if not self.cfg.freespace_enabled:
            return self.free_view
        fresh = False
        for key in ("cam_up", "cam_down"):
            slot = self.slots.get(key)
            if slot is None or frames.get(key) is None:
                continue
            if self._cam_seen.get(key) != slot.updates:
                self._cam_seen[key] = slot.updates
                fresh = True
        if fresh:
            self.free_view = self.freespace.update(frames.get("cam_up"),
                                                   frames.get("cam_down"))
        elif not any(f is not None for f in frames.values()):
            # Both cameras gone. Do not keep steering on a remembered picture.
            self.free_view = self.freespace.update(None, None)
        return self.free_view

    def _vision_drive(self, gate: str, scan, scan_age, depth_age):
        """Drive on the cameras alone, because a critical sensor has gone stale.

        Harsh, 2026-09-03: "make sure that when the sensors are not working the device
        (YOU) uses the camera's vision to navigate through the environment like a
        self-driving vehicle."

        THE RULE THAT MAKES THIS SAFE TO DO AT ALL: it engages only when ONE of the
        two critical sensors is stale. The camera is blind inside 0.87 m -- that is
        geometry, not tuning (friday_freespace.CAM_HEIGHT_M) -- and that blind zone is
        precisely the ToF's patch. With both sensors down there would be nothing
        watching the metre she is about to drive through, so that case still stops,
        exactly as before.

        Capped at freespace_v_max, which is a third of v_max, because the distances
        here come from an estimated camera height and an assumed level mount. Driving
        on a number you cannot verify is the moment to go slowly.
        """
        c = self.cfg
        view = self._update_freespace(self._read_cameras())
        if not (c.freespace_fallback and view.usable):
            return None
        v = 0.0
        if view.centre_range > c.freespace_stop_range and not view.near_blocked:
            span = max(c.lidar_max_range - c.freespace_stop_range, 1e-6)
            reach = (view.centre_range - c.freespace_stop_range) / span
            v = float(np.clip(c.freespace_v_max * (0.35 + 0.65 * reach),
                              0.0, c.freespace_v_max))
        omega = float(np.clip(c.freespace_heading_gain * view.bearing,
                              -c.omega_max, c.omega_max))
        # Whichever sensor IS still alive keeps its veto. A stale LiDAR does not make
        # the ToF's opinion less true, and the reflex thread is running regardless.
        if scan is not None and scan_age <= c.lidar_deadline:
            v, omega, _ = self.governor(v, omega, scan)
        v, omega = self.limiter(v, omega, c.dt)
        self.motors.drive(v, omega)
        self.stats["vision_ticks"] = self.stats.get("vision_ticks", 0) + 1
        rec = tlm.DecisionRecord(tick=self.stats["ticks"], obs_dim=c.obs_dim,
                                 model_name=self.policy.model_name, model_loaded=True,
                                 gate=gate, lidar_age=scan_age, tof_age=depth_age)
        rec.add("vision_fallback", v, omega,
                f"{view.source} sees {view.centre_range:.2f} m ahead, open "
                f"{math.degrees(view.bearing):+.0f} deg (conf {view.confidence:.2f})")
        rec.wire = self.motors.last_wire
        self.last_record = rec
        lidar_min = float(np.min(scan)) if scan is not None else float("nan")
        return {"gate": gate, "v": v, "omega": omega,
                "lidar_min": lidar_min, "tof_min": float("nan"), "visual": 0.0,
                "v_policy": 0.0, "omega_policy": 0.0,
                "v_raw": float("nan"), "omega_raw": float("nan"),
                "authority": tlm.SRC_SAFETY, "loop_ms": 0.0}

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
        # REAL elapsed since the last command, not the nominal dt.
        #
        # The slew limiter converts an acceleration limit into a per-tick step, so it
        # needs the time that actually passed. It was handed c.dt unconditionally
        # while 33% of ticks overran the 100 ms budget (measured: mean 94 ms, p95 275,
        # max 465). On a 250 ms tick it therefore permitted 0.8 * 0.1 = 0.08 m/s of
        # change where 0.20 was due, so she accelerated 2.5x too slowly -- and by a
        # factor that varied with the jitter, which is what "not smooth" is.
        #
        # Clamped both ways. Below half nominal a near-zero interval would freeze
        # acceleration; above 3x, a long stall would bank enough allowance to make the
        # recovery a lurch. Only the SLEW rate is affected: governor, reflex and the
        # duty ceiling still bound absolute speed, so this cannot make her faster,
        # only less sluggish at reaching a speed already permitted.
        dt_real = c.dt
        if self._last_tick_t is not None:
            dt_real = min(max(t_start - self._last_tick_t, c.dt * 0.5), c.dt * 3.0)
        self._last_tick_t = t_start
        gate = "ok"

        # Stage accounting -- see ControlLoop.stage_s. `lap` charges everything since
        # the previous lap to one name; the names below are the section headings.
        _acc, _mark = self.stage_s, t_start

        def lap(name: str) -> None:
            nonlocal _mark
            now = time.perf_counter()
            _acc[name] = _acc.get(name, 0.0) + (now - _mark)
            _mark = now

        scan, scan_age = self.slots["lidar"].read()
        depth, depth_age = self.slots["tof"].read()

        # ---- FAIL-SAFE: a stale safety-critical sensor means STOP, never "clear" ----
        #
        # ...unless the OTHER critical sensor is still healthy and the cameras can see,
        # in which case she navigates on vision at a third of her top speed rather than
        # standing still until someone reseats a USB plug. _vision_drive() carries the
        # reasoning and returns None whenever it is not entitled to act, which puts the
        # original stop back exactly as it was.
        lidar_stale = scan is None or scan_age > c.lidar_deadline
        tof_stale = c.tof_enabled and (depth is None or depth_age > c.tof_deadline)
        # ---- FILE 21 HOLD: the exploration mission asked her to stand still -------
        # Motors zero and NOTHING ELSE DECIDES this tick: the policy, navigator,
        # recovery, unstick and the vision fallback below do not run, so a question
        # is never read as "I am stalled" and she never creeps on cameras while
        # held. Zero is zero whatever the sensors say. Lapses by itself
        # (rl/way_back.py Hold).
        # FILE 21 U1: ...unless the hold carries a DRIVE ORDER: then this tick
        # runs in full, as the way back's does, with the order as the request.
        if self.hold.active() and self.hold.driving() is None:
            return self._hold_row(scan, depth, t_start)
        if lidar_stale != tof_stale:
            row = self._vision_drive("vision_" + ("lidar" if lidar_stale else "tof"),
                                     scan, scan_age, depth_age)
            if row is not None:
                self.stats["ticks"] += 1
                return row
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
        lap("sensors+obs")
        action = self.policy.infer(obs)
        lap("policy")
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
                # The message has to distinguish two genuinely different states, because
                # it used to assert the worse one unconditionally. Since wheel_speed_max
                # was measured at 0.403 the floor is 0.149 m/s and v_min is -0.150, so
                # the fastest reverse the policy can ask for IS executable -- by 0.001
                # m/s. Claiming "no reverse is executable" there is simply false, and a
                # warning that overstates is one people learn to discount.
                if abs(c.v_min) < floor_mps:
                    LOGGER.warning(
                        "CAPABILITY MISMATCH: the policy asked to reverse and this "
                        "drivetrain cannot. v_min %+.3f m/s is itself below the %.3f m/s "
                        "stiction floor, so NO reverse in the trained action space is "
                        "executable. Substituting a pivot toward open space, labelled "
                        "CAPABILITY_ADAPTER in telemetry -- the raw policy output is "
                        "preserved unmodified. The real fix is retraining with the "
                        "measured floor. --no-reverse-fallback shows the raw behaviour.",
                        c.v_min, floor_mps)
                else:
                    LOGGER.warning(
                        "CAPABILITY MISMATCH: the policy asked to reverse at %+.3f m/s, "
                        "below the %.3f m/s stiction floor. Note v_min %+.3f m/s IS "
                        "executable, so this is a PARTIAL mismatch -- only reverses "
                        "slower than the floor are being substituted with a pivot "
                        "(CAPABILITY_ADAPTER in telemetry; raw policy output preserved). "
                        "The margin is %.3f m/s, which is thin enough that measurement "
                        "error in wheel_speed_max could close it. "
                        "--no-reverse-fallback shows the raw behaviour.",
                        v_raw, floor_mps, c.v_min, abs(c.v_min) - floor_mps)
            # A REVERSE THAT IS MERELY TOO GENTLE IS LIFTED, NOT REPLACED.
            #
            # The warning above already says v_min is executable when it is, and then
            # this branch used to discard the reverse anyway and pivot. Substituting a
            # pivot is right when the drivetrain genuinely cannot back up; it is wrong
            # when it can, because a pivot cannot free a robot that is WEDGED -- its
            # footprint does not translate, which is the same property that makes it
            # safe. MEASURED 2026-09-03: she drove into something and sat with
            # tof_min at 0.020 m -- twenty millimetres, unchanged across all 1233 ticks
            # of a 180 s run -- while the policy asked to reverse at -0.069 m/s on 297
            # of them and the chain turned every one into a pivot she could not use.
            #
            # So: if ANY reverse is executable, honour the direction the policy asked
            # for at the slowest speed the wheels can actually deliver. This cannot
            # increase authority -- it is still a reverse, still slower in magnitude
            # than nothing else in the chain grants -- and it stays gated by the rear
            # gate at stage 2d, which checks the rear arc before any reverse executes.
            reverse_possible = abs(c.v_min) >= floor_mps
            if v_raw < 0.0 and reverse_possible:
                v_pol = c.v_min
                self.stats["reverse_lifts"] = self.stats.get("reverse_lifts", 0) + 1
                gate = "reverse_lifted"
                rec.add("reverse_lift", v_pol, omega_pol,
                        f"policy asked {v_raw:+.3f} m/s, below the {floor_mps:.3f} m/s "
                        f"floor; lifted to the executable {c.v_min:+.3f} m/s rather "
                        f"than replaced with a pivot")
            else:
                v_pol = 0.0
                omega_pol = self._open_side(scan) * c.omega_max
                self.stats["no_reverse_turns"] += 1
                gate = "no_reverse_turn"
                rec.add("no_reverse", v_pol, omega_pol,
                        "; ".join(rec.capability["findings"])
                        or f"reverse {v_raw:+.3f} m/s is not executable")

        lap("adapter")
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
        target_bearing = None
        if self.planner is not None:
            err = self.planner.bearing_error(self.map.pose)
            target_bearing = err
            home_cmd = None
            if self.way_back is not None and self.way_back.phase == "returning":
                # The way back REPLACES the policy's request (it has no idea
                # where the start is); every stage below still has the last
                # word. See way_back.homing_command.
                home_cmd = _way_back.homing_command(
                    err, self.way_back.distance(self.map.pose), c.omega_max,
                    self.motors.min_speed_mps())
            if home_cmd is not None:
                v_pol, omega_pol, why = home_cmd
                bias = 0.0
                gate = "planner"
                rec.add("planner", v_pol, omega_pol, why)
            else:
                bias, why = steering_bias(err, omega_pol, c.omega_max)
            planner_ctx = {"bearing_error": err, "bias": bias, "reason": why}
            if bias != 0.0:
                omega_pol = float(np.clip(omega_pol + bias, -c.omega_max, c.omega_max))
                self.stats["planner_biases"] += 1
                gate = "planner"
                rec.add("planner", v_pol, omega_pol, why)
            self.planner.note_progress(self.map.pose)
        rec.planner = planner_ctx
        # ---- FILE 21 U1: THE MISSION'S DRIVE ORDER (only while it holds the run) --
        # Replaces the request exactly as the way back's homing_command does, and
        # after it, so it wins over both the planner's nudge and the way back.
        # Every stage below -- arc planner, governor, recovery, unstick, the rear
        # gate, the ToF reflex, the limiter -- still has the last word.
        order = self.hold.driving()
        if order is not None:
            v_pol, omega_pol, why = self.mission_drive.command(
                order, self.way_back.tracker if self.way_back is not None else None,
                self.map.pose, c.omega_max, self.motors.min_speed_mps(), c.v_min)
            gate = "mission_drive"
            rec.add("mission_drive", v_pol, omega_pol, why)
            # ...and the local navigator below dodges TOWARD the path, not toward
            # the exploring's old frontier (they fought: 2026-10-04 robot test).
            if self.mission_drive.last_err is not None:
                target_bearing = self.mission_drive.last_err
            self.stats["mission_drive_ticks"] = self.stats.get("mission_drive_ticks", 0) + 1

        lap("planner-bias")
        # ---- 1d. LOCAL NAVIGATOR (is the command she just asked for DRIVABLE?) -----
        #
        # Everything downstream of the policy can only say no, and the governor's clamp
        # is `v_out = min(v_out, 0.0)` -- forward is zeroed, omega passes through. So a
        # robot refused permission to drive still has full permission to turn, and
        # turns. That is the 515 deg/m (best run) to 1799 deg/m (worst) measured on
        # 2026-09-04, and it is why she looked hesitant rather than deliberate.
        #
        # This stage answers a different question from every other one: not "is this
        # command dangerous" but "of the commands available, which one actually goes
        # somewhere". It simulates ~105 candidate arcs against the live scan and keeps
        # the best survivor.
        #
        # THREE THINGS KEEP IT SAFE, and none of them is trust in this code:
        #   * it is capped at `v_pol` -- it can only slow her down, never speed her up;
        #   * it only samples FORWARD arcs, so reverse, escape turns and recovery are
        #     untouched and still own every case where backing up is the answer;
        #   * a None means "nothing forward is drivable", and the tick then runs
        #     exactly the code it ran before this stage existed.
        # The governor, visual gate, ToF reflex and rate limiter all still run below.
        nav_ctx = None
        if (self.navigator is not None and self.navigator_enabled
                and scan is not None and v_pol > 0.0):
            # SPEED AUTHORITY, and it is OFF (0.0) unless asked for.
            #
            # With authority 0 this stage is capped at `v_pol`, so it may only ever
            # slow her down -- the chain's invariant holds exactly as written, and the
            # navigator changes only WHICH WAY she goes. The cost is that she is only
            # ever as decisive as the policy, and this policy is timid: measured mean
            # v_raw +0.206 m/s, and it asks for 0.026 in an open room.
            #
            # Above 0, the navigator may command up to that fraction of v_max on an arc
            # it has SIMULATED CLEAR over LOOKAHEAD_M against the live scan -- which is
            # strictly more evidence about the next 0.6 m than the policy has. It is
            # still gated on the policy wanting to go forward at all, and the governor,
            # visual gate, ToF reflex and limiter all still run below it and can still
            # take it away. This is a deliberate, bounded exemption from "a planner can
            # never make this robot go faster", and it is Harsh's call, not a default.
            authority = float(getattr(c, "nav_speed_authority", 0.0))
            v_allow = max(v_pol, c.v_max * authority) if authority > 0.0 else v_pol
            decision = self.navigator.decide(
                scan, v_allow=v_allow, omega_want=omega_pol,
                desired_bearing=target_bearing, blind_beams=self.map.blind_beams,
                v_floor=max(self.motors.min_speed_mps(), 0.05),
                # The near field, so the planner and the reflex stop disagreeing about
                # what is in front of her. Without this the navigator plans through the
                # band below the LiDAR's 0.22 m floor and the reflex brakes for it.
                tof_depth=depth if not tof_stale else None)
            if decision is not None:
                v_pol, omega_pol = decision.v, decision.omega
                gate = "navigator"
                nav_ctx = {"situation": decision.situation,
                           "committed": decision.committed,
                           "clearance": decision.clearance}
                rec.add("navigator", v_pol, omega_pol, decision.why)
                self.stats["nav_decisions"] += 1
                self.stats[f"nav_{decision.situation}"] += 1
            else:
                self.stats["nav_handoffs"] += 1
        rec.navigator = nav_ctx

        lap("navigator")
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
        # THE FORWARD ARC, not the whole sweep, and the governor's own k-th smallest
        # rather than a fresh minimum. Taking np.min over 360 degrees meant a wall
        # BEHIND her counted as "forward is blocked" and armed the escape timer while
        # the path ahead was clear -- the same error the governor's braking law was
        # corrected for on 2026-09-02, still living one stage down. Reusing
        # last_fwd_min also guarantees the escape and the law that blocked her agree
        # about the clearance; computing it twice is how they drift apart.
        if (self.escape_turns and cannot_reverse
                and self.governor.last_fwd_min <= c.hard_stop and v <= 1e-6):
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
            # THE INTENT, NOT THE GATED COMMAND. `no_progress` asks "was she commanded
            # to move, and did she fail to?" -- and it used to be handed `v` from after
            # the governor and the rear gate. So the instant a safety gate refused her,
            # `asked` fell to zero and the detector concluded she was not trying, when
            # in fact being refused is the MOST stuck a robot can be. The gate doing its
            # job destroyed the evidence recovery needed to fire.
            #
            # Found by raising rear_stop to 0.36 m: with the rear now correctly blocked
            # she could no longer reverse, and recovery -- which exists for exactly that
            # situation -- went from 8 triggering ticks to 0.
            #
            # v_pol is the policy's output after the capability adapter and the planner
            # and before any safety stage, so it is what the robot WANTED. `ex` stays
            # the executed twist, so the comparison is still intent against reality.
            forced = self.recovery.update(
                scan, ex[0], ex[1], v_pol, boxed_in, self.map.pose,
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

        # ---- 2d. REAR GATE, applied to whatever 2b and 2c decided ------------------
        #
        # The governor already gated reverse, but stages 2b (escape) and 2c (recovery)
        # run downstream of it and issue reverse of their own -- `_maneuver` returns
        # v_min to back off. Re-applying here means no stage can reverse into a blocked
        # rear, whoever authored the command. It only ever raises v toward 0, so it
        # cannot grant authority, and it is idempotent when nothing is behind.
        v_before_rear = v
        v = self.governor.gate_reverse(v, scan)
        if v != v_before_rear:
            self.stats["rear_blocks"] = self.stats.get("rear_blocks", 0) + 1
            gate = "rear_blocked"
            rec.add("rear_gate", v, omega,
                    f"reverse refused: {self.governor.last_rear_min:.2f} m behind, "
                    f"inside the {c.rear_stop:.2f} m rear stop")

        # ---- 2e. MAKE THE REVERSE EXECUTABLE, whoever asked for it -----------------
        #
        # Harsh, 2026-09-04: "we dont want her to hesitate while making turns and
        # traversing in the reverse direction... move the lift below recovery".
        #
        # Stage 1b already lifts a too-gentle reverse to the slowest the wheels can
        # actually turn -- but only when THE POLICY asked to reverse. MEASURED in the
        # 12-minute run of 2026-09-04 17:21: she spent one 611-tick (65 s) stretch with
        # the ToF pinned at 0.14 m and `lidar_min` FROZEN at 0.38 m, commanded to pivot
        # and not actually rotating, while the rear was 1.26 m CLEAR. Across that run
        # 36% of all ticks were spin-in-place. The reverses that were issued averaged
        # -0.063 m/s against a 0.149 m/s stiction floor, so every one of them executed
        # as a stop -- and stage 1b never saw them, because the POLICY was asking to go
        # FORWARD (+0.42). The reverse came from recovery, which runs downstream of the
        # lift.
        #
        # So the lift runs again HERE, below escape (2b), recovery (2c) and the rear
        # gate (2d), where it can see the reverse whoever authored it.
        #
        # THIS CANNOT GRANT AUTHORITY TO REVERSE. It only ever changes the MAGNITUDE of
        # an already-negative v, and it runs AFTER the rear gate -- so a reverse the
        # rear gate refused has already been raised to 0.0 and is not negative any
        # more, which means there is nothing here for this to act on. It is bounded by
        # v_min, and the governor, visual gate, reflex and limiter all still run below.
        floor_mps = self.motors.min_speed_mps()
        if v < 0.0 and floor_mps > 0.0 and abs(v) < floor_mps:
            lifted = max(-floor_mps, c.v_min)
            if lifted < 0.0:
                rec.add("reverse_floor", lifted, omega,
                        f"reverse {v:+.3f} m/s is under the {floor_mps:.3f} m/s stiction "
                        f"floor and would execute as a stop; issued at {lifted:+.3f}")
                v = lifted
                self.stats["reverse_floor_lifts"] += 1

        lap("governor+escape+rear")
        # ---- 3. visual gate (advisory) --------------------------------------------
        frames = self._read_cameras()
        if c.cam_enabled:
            # UP camera: measured context only, never motion authority. Its statistics
            # are recorded and reported; nothing downstream reads them.
            self.context.update(frames.get("cam_up"))
            self._update_freespace(frames)
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

        # ---- 3b. free-space advisory (a nudge, never an override) -----------------
        #
        # Harsh asked for the cameras to be used to "move according to the space
        # available", and this is that -- at the authority it has earned. The camera
        # tells her which way is OPEN, which is a shape and survives the fact that its
        # height and tilt are estimated; the LiDAR remains the authority on where
        # things ARE. So it adds a bounded turn rate and can touch nothing else.
        #
        # ONLY WHILE SHE IS ACTUALLY NAVIGATING. During an escape, a recovery or a
        # reflex stop something more urgent has already decided where she is pointing,
        # and a camera nudge on top of that is the same mistake as steering by two
        # opinions at once -- which is a description of the zig-zag.
        view = self.free_view
        if (c.freespace_enabled and view.usable and v > 0.0
                and gate in ("ok", "planner", "visual_slow", "governor", "tof_slow")):
            nudge = float(np.clip(c.freespace_gain * view.bearing,
                                  -c.freespace_max_nudge, c.freespace_max_nudge))
            if abs(nudge) > 1e-3:
                omega = float(np.clip(omega + nudge, -c.omega_max, c.omega_max))
                self.stats["vision_nudges"] = self.stats.get("vision_nudges", 0) + 1
                rec.add("freespace", v, omega,
                        f"open {math.degrees(view.bearing):+.0f} deg ({view.source}, "
                        f"conf {view.confidence:.2f}) -> nudge {nudge:+.2f} rad/s")
        if c.freespace_enabled and view.near_blocked and v > 0.0:
            # The down camera sees the half-metre the ToF's 8x8 patch samples between
            # its zones. It slows her; it does not stop her -- that camera falls off
            # the USB bus repeatedly on this robot and must not be able to end a run.
            v = min(v, c.v_max * c.visual_min_speed_scale)
            rec.add("freespace_near", v, omega, "down camera: floor gone at the bumper")

        lap("cameras")
        # ---- 4. ToF reflex (hard veto) --------------------------------------------
        # Set by the emergency branch when the obstacle is far enough that turning in
        # place cannot reach it. It has to be carried down to the rate limiter, which
        # otherwise zeroes rotation along with speed and silently undoes the escape.
        pivot_allowed = False
        tof_scale, emergency = self.reflex.evaluate(depth)
        if emergency:
            # FORWARD motion is vetoed absolutely -- that is the whole point of the
            # reflex and it does not soften. Reverse is a different matter: the ToF
            # looks FORWARD, so an obstacle 0.2 m ahead is a reason to stop advancing
            # and precisely a reason to back off. Zeroing v outright also contradicted
            # LidarGovernor one layer up, which on the same evidence does
            #     v_out = min(v_out, 0.0)   # "rotate / reverse only"
            # This makes the reflex agree with the rule the chain already had rather
            # than inventing a new exception. Measured while she was wedged: the
            # policy asked for -0.150 m/s on nearly every stuck tick and the reflex
            # threw all of them away. Approved by Harsh 2026-09-02.
            v = min(v, 0.0)
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

        # ---- 4b. UNDO THE LAST MOVE: every rule above said "no" -------------------
        #
        # Harsh, 2026-09-28: "it gets stucked at one place sometimes rather than
        # trying to avoid it ... dont bump". MEASURED 2026-09-27 22:00:44: bumper
        # 0.09 m froze turning (<= tof_pivot_min_clearance) and forward, the rear
        # gate refused reverse at 0.33 m, and she output (0, 0) for 2 min 40 s. She
        # had turned INTO it. rl/unstick.py retraces the last move -- only through
        # space her body has just left, never forward, the rear LiDAR guarding any
        # reverse -- and after two tries at one place stops and asks to be moved.
        # This is the ONE stage allowed to issue motion the reflex refused, and only
        # that retrace; `--no-unstick` is the old run exactly.
        self._undo_active = False
        if self.unstick_enabled:
            frozen = bool(emergency and v >= 0.0
                          and self.reflex.last_min <= c.tof_pivot_min_clearance)
            undo = self.unstick.update(
                frozen=frozen,
                tof_min=self.reflex.last_min if depth is not None else None,
                rear_min=self.governor.last_rear_min, pose=self.map.pose)
            if undo is not None:
                v, omega, why_undo = undo
                v = min(v, 0.0)                     # never forward, whoever asked
                pivot_allowed = omega != 0.0
                self._undo_active = True
                self.stats["undo_ticks"] = self.stats.get("undo_ticks", 0) + 1
                gate = "undo"
                rec.add("undo", v, omega, why_undo)
            self._drain_unstick()

        # ---- 4c. TURN COMMITMENT: no dithering left/right while blocked ------------
        #
        # Harsh, 2026-10-04: "noise more than movement ... very harsh ... it should
        # be smooth", then "DO whats still left". MEASURED in the 490 Hz black boxes:
        # blocked (|v| < 0.15), the POLICY flips its turn between +1.6 and -1.6
        # rad/s 10-38 times a minute, 35-45% of them back again within a second,
        # and every flip passes through (0, 0) at the slew limiter -- the one-tick
        # stop "blips" (30 of 35 in the 23:03 run, 40 of 48 at 15:43). So a turn
        # taken while blocked is KEPT for turn_commit_s: a request to turn the
        # other way inside that window turns the committed way at the same rate.
        # Only the DIRECTION of a turn the chain already asked for -- never a turn
        # it did not ask for, never through a stop (a turn under 0.15 rad/s or a
        # zero passes as it is), never faster. The deliberate turns that pick their
        # side (escape, recovery, unstick's undo, the mission's path, the
        # open-side pivot) are exempt; driving (|v| >= 0.15) drops the commitment.
        # The ToF reflex already ran: a turn it froze is a zero here and stays one.
        if self.turn_commit_s > 0.0 and not self._undo_active:
            omega = self._commit_turn(v, omega, gate, rec)

        lap("reflex")
        # ---- 4d. PICKED UP: the wheels stay still until she is put down --------------
        # (body_sense.Detector on the motor board's IMU, read in stage 6b. The
        # board refuses M by itself past 35 deg of tilt; a level lift is this.)
        if self._body and self._body.lifted and (v != 0.0 or omega != 0.0):
            v, omega = 0.0, 0.0
            gate = "lifted"
            rec.add("lifted", v, omega, "picked up: wheels still until put down")
            self.stats["lifted_ticks"] = self.stats.get("lifted_ticks", 0) + 1

        # ---- 5. slew limit + actuate ----------------------------------------------
        v, omega = self.limiter(v, omega, dt_real, emergency=emergency,
                                allow_rotation=pivot_allowed)
        rec.add("limiter", v, omega, "slew limited")
        self.motors.drive(v, omega)

        # Read back what the driver actually put on the wire. Done AFTER drive() so
        # these are this tick's values, not the previous tick's.
        rec.final_v, rec.final_omega = v, omega
        rec.wire = self.motors.last_wire
        rec.duties = self.motors.last_duties
        rec.gate = gate

        lap("actuate")
        # ---- 6. propagate pose with what the WHEELS actually did -------------------
        # Not the commanded twist. The creep cutoff, the stiction floor and the duty
        # cap all make executed != commanded on this drivetrain, and dead reckoning
        # from the command puts the frontier channel's map somewhere the robot is not.
        # In --simulate the world moves exactly as commanded, so the sim is the truth
        # there and the driver's opinion is irrelevant.
        if self.sim is not None:
            self.sim.step(v, omega, c.dt)
            self.map.dead_reckon(v, omega, c.dt)
            if self.unstick_enabled:
                self.unstick.record(v, omega, c.dt)
        else:
            ex_v, ex_omega = self.motors.executed()
            # THE GYRO, when the board has one (Harsh, 2026-10-07): the turn she
            # MADE, not the turn she was told to make. ~22% of commanded turns
            # execute as stops on this drivetrain, and dead reckoning counted
            # every one -- the 2026-10-04 run ended ~98 deg out.
            dtheta = self._gyro_dtheta()
            omega_dr = ex_omega if dtheta is None else dtheta / c.dt
            if dtheta is not None:
                self._note_pivot_stall(ex_v, ex_omega, omega_dr)
            self.map.dead_reckon(ex_v, omega_dr, c.dt)
            if self.unstick_enabled:
                self.unstick.record(ex_v, omega_dr, c.dt)

        # ---- 6b. persistent memory ------------------------------------------------
        # Same pose as the observation map, so the two never disagree about where the
        # robot is. submit() copies and returns immediately; integration happens on the
        # mapper thread, because folding a sweep costs ~24 ms against a 100 ms budget
        # and that cost GROWS with the map.
        if self.mapper is not None:
            self.world.set_pose(*(float(x) for x in self.map.pose))
            self.mapper.submit(scan, self.map.pose, self.map.blind_beams)
        # The way back's tracker: the same pose, and the 1-degree sweep when the
        # LiDAR thread publishes one. Copies and returns at once.
        if self.way_back is not None:
            fine_slot = self.slots.get("lidar_fine")
            fine, fine_age = (fine_slot.read() if fine_slot is not None
                              else (None, float("inf")))
            self.way_back.tracker.submit(
                fine if (fine and fine_age <= c.lidar_deadline) else scan,
                self.map.pose, self.map.blind_beams,
                gyro_gaps=self.gyro_gaps if self.gyro_seen else None)

        # NOTE: loop_ms deliberately stops here, because the black box row below has
        # to carry it. The telemetry tail after this point is charged to the stage
        # accounting at the end of the function, so `budget by stage` can exceed
        # loop_ms -- that difference IS the tail, and it used to include a world.stats()
        # over every cell in the map.
        loop_ms = (time.perf_counter() - t_start) * 1000.0
        self.loop_ms.append(loop_ms)
        self.stats["ticks"] += 1
        if loop_ms > c.dt * 1000.0:
            self.stats["overruns"] += 1

        rec.loop_ms = loop_ms
        rec.explored_m2 = self.map.explored_m2
        if self.mapper is not None:
            # TELEMETRY, NOT CONTROL -- and it counts every cell in the grid, so its
            # cost GROWS WITH THE MAP. Measured 5.8 ms on a 41k-cell grid and ~12 ms by
            # the time the map reached 88k cells, charged to every single tick, for a
            # number nothing in the chain reads. Refreshed at 2 Hz instead: the black
            # box gets the same curve, and the control loop stops paying for the map's
            # success. (It also sat AFTER loop_ms was computed, so it never even
            # appeared in the loop time it was inflating.)
            now = time.monotonic()
            if self._world_stats is None or now - self._world_stats_t >= 0.5:
                w = self.world.stats()
                w["dropped"] = self.world.dropped
                self._world_stats, self._world_stats_t = w, now
            rec.world, rec.world_loaded = self._world_stats, self.world.loaded_from_disk
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
                              (rec.__dict__.get("_recovery_state") or ""),
                              round(self.governor.last_fwd_min, 3),
                              round(self.governor.last_closing, 3),
                              round(min(self.governor.last_rear_min, 99.9), 3),
                              round(math.degrees(self.free_view.bearing), 1),
                              round(self.free_view.centre_range, 2),
                              round(self.free_view.confidence, 2)))
        lap("pose+telemetry")
        self.stage_wall_s += time.perf_counter() - t_start
        return row

    def _hold_row(self, scan, depth, t_start: float) -> dict[str, Any]:
        """FILE 21: one held tick. Zero on the wire; a record says why."""
        c = self.cfg
        self.motors.drive(0.0, 0.0)
        self.limiter.reset()
        self.stats["mission_hold_ticks"] = self.stats.get("mission_hold_ticks", 0) + 1
        self.stats["ticks"] += 1
        loop_ms = (time.perf_counter() - t_start) * 1000.0
        lidar_min = float(np.min(scan)) if scan is not None else float("nan")
        rec = tlm.DecisionRecord(tick=self.stats["ticks"], obs_dim=c.obs_dim,
                                 model_name=self.policy.model_name, model_loaded=True,
                                 gate="mission_hold", lidar_min=lidar_min)
        rec.final_v, rec.final_omega = 0.0, 0.0
        rec.wire = self.motors.last_wire
        rec.loop_ms = loop_ms
        rec.pose = tuple(float(x) for x in self.map.pose)
        self.last_record = rec
        return {"gate": "mission_hold", "v": 0.0, "omega": 0.0,
                "lidar_min": lidar_min,
                "tof_min": self.reflex.last_min if depth is not None else float("nan"),
                "visual": 0.0, "v_policy": 0.0, "omega_policy": 0.0,
                "v_raw": float("nan"), "omega_raw": float("nan"),
                "authority": tlm.SRC_SAFETY, "loop_ms": loop_ms}

    # ------------------------------------------------------------------- run --------
    def run(self, duration: float | None = None, verbose: bool = True) -> None:
        if self.cfg.tof_enabled:
            self._reflex_thread = threading.Thread(target=self._reflex_worker,
                                                   name="reflex", daemon=True)
            self._reflex_thread.start()
            LOGGER.info("Reflex thread running at %.0f Hz", self.cfg.reflex_hz)

        # PERCEPTION, not control -- see friday_vision for why that line is drawn
        # where it is. Absent model, absent camera and absent cv2 all degrade to
        # "she has nothing to say about the room", which costs the run nothing.
        if getattr(self.cfg, "detect_enabled", False) and "cam_up_rgb" in self.slots:
            self.detector = ObjectDetector(
                DETECT_MODEL_PATH, self.slots["cam_up_rgb"],
                period_s=self.cfg.detect_period_s, threads=self.cfg.detect_threads)
            self.detector.start()

        if self.map_enabled:
            if self.world.load():
                s = self.world.stats()
                LOGGER.info("World map restored from %s: %.1f m^2 known, %d sweeps, "
                            "%.1f m^2 previously driven%s",
                            self.world.map_file.name, s["explored_m2"],
                            s["sweeps_total"], s["visited_m2"],
                            "" if self.world.visits_loaded
                            else "  (visit trace MISSING -- map only)")
                if not self.relocalise():
                    # A map you cannot place yourself in is WORSE than no map. The
                    # pose re-initialises to zero every run, so an unrelocalised map
                    # is a picture of the room taken from somewhere the robot is not,
                    # and the frontier planner will steer confidently toward places
                    # that are not there. Measured after three runs of this: 335
                    # disconnected occupied blobs, largest 17 cells, because one wall
                    # had been written at three different offsets. Discard and rebuild.
                    # SET ASIDE, NOT DELETED. Harsh, 2026-09-04: "DO NOT discard the
                    # maps." Starting fresh is still right -- an unplaceable map sends
                    # the frontier planner to addresses that do not exist -- but the
                    # file is 12 minutes of driving and there is no reason for the
                    # rebuild to consume it. It moves aside with its pose and visit
                    # trace so it can be inspected, merged or restored later.
                    kept = self._archive_map()
                    LOGGER.warning("Could not place the robot in the restored map, so "
                                   "this run rebuilds from scratch -- every target "
                                   "named from it would be at the wrong address. The "
                                   "old map is NOT deleted: %s", kept)
                    self.world = WorldMap(max_range=self.cfg.lidar_max_range)
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
                if self.way_back_enabled:
                    tracker = _way_back.StartTracker(self.cfg.lidar_max_range)
                    tracker.start()
                    self.way_back = _way_back.WayBack(tracker)
                    LOGGER.info("Way back ON: the start is tracked from the first "
                                "sweep; when the time is up she drives back to it "
                                "(within %.2f m, up to %.0f s).",
                                self.way_back.reached_m, self.way_back.budget_s)

        self.hold.start()                   # FILE 21: polls the mission's hold file
        if getattr(self, "voice", None) is not None:
            self.voice.muted = self.hold.active  # FILE 21: quiet while she asks
        held_left = None                    # FILE 21: run time left when a hold began

        period = self.cfg.dt
        next_tick = time.monotonic()
        t_start = time.monotonic()
        t_end = None if duration is None else t_start + duration
        LOGGER.info("Control loop starting at %.0f Hz. Ctrl+C to stop.", self.cfg.control_hz)

        try:
            while not self._stop.is_set():
                now = time.monotonic()
                # FILE 21 U3: paused by a person for longer than anyone would
                # mean -- the mission is gone or forgot her. She does not set
                # off again on her own: the run ENDS, standing still.
                if self.hold.paused_too_long():
                    LOGGER.warning("Paused for over %.0f min with nobody saying "
                                   "continue: ending the run where she stands.",
                                   _way_back.PAUSE_MAX_S / 60.0)
                    break
                # FILE 21: time standing still for the mission (a question, a pause)
                # does not use up the run's exploring time.
                if self.hold.active():
                    if held_left is None and t_end is not None:
                        held_left = max(t_end - now, 0.0)
                    if held_left is not None:
                        t_end = now + held_left
                else:
                    held_left = None
                wb = self.way_back
                if wb is not None and wb.phase == "exploring":
                    if t_end is not None and now >= t_end:
                        self._begin_way_back("time")
                    elif (self.planner is not None and not self.hold.active()
                          and wb.exhausted_long_enough(
                              self.planner.exhausted, now - t_start, now)):
                        self._begin_way_back("explored")
                    elif self.hold.return_requested():
                        # FILE 21: the mission ended exploring early ("go back",
                        # its time cap, nothing left). The same way back, sooner.
                        self._begin_way_back("mission")
                # FILE 21 U1: while the mission holds her (asking, or driving its
                # own way home) the run's way back waits: the mission decides
                # when she is home, and she must not end the run under it.
                if wb is not None and wb.phase == "returning":
                    if not self.hold.active():
                        self._steer_home()
                        outcome = wb.check(self.map.pose, now)
                        if outcome is not None:
                            self._finish_way_back(outcome)
                            break
                elif t_end is not None and now >= t_end:
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

    # ---------------------------------------------------------------- way back ---
    def _begin_way_back(self, why: str) -> None:
        wb = self.way_back
        wb.begin(why)
        home = wb.home()
        self._home_target = _way_back.HomeTarget(home[:2] if home else (0.0, 0.0))
        self._steer_home()
        d = wb.distance(self.map.pose)
        LOGGER.info("WAY BACK (%s): heading to where I started, %s away by the start "
                    "tracker (%s).", why,
                    "?" if d is None else f"{d:.2f} m", wb.tracker.snapshot())
        # Through the run's own voice. `self.narrator` is disabled in the localnav
        # run, so until 2026-09-28 none of these lines was ever heard on the robot.
        self._say("way_back", "Exploration completed. Returning to where I started.")

    def _say(self, key: str, text: str) -> bool:
        """One line out loud, through whichever voice this run has. The localnav
        run speaks through `self.voice` (friday_voice); pi_deploy's own narrator is
        the fallback. True if a voice took the line."""
        voice = getattr(self, "voice", None)
        if voice is not None:
            return bool(voice.say(key, text, priority=2, cooldown_s=0.0))
        return bool(self.narrator.say(key, text, P_RECOVERY))

    TURN_COMMIT_EXEMPT = frozenset({"escape_turn", "recovery", "undo",
                                    "mission_drive", "no_reverse_turn"})
    TURN_COMMIT_MAX_V = 0.15        # slower than this: blocked, turning (nearly) in place
    TURN_COMMIT_MIN_OMEGA = 0.15    # a smaller turn is a nudge, passed as it is

    # ---- the gyro (firmware GYRO 2+) -------------------------------------------------
    def _gyro_dtheta(self) -> float | None:
        """The heading change since the last tick, from the board's own yaw
        (rad, + = left), or None to fall back to the commanded turn.

        Differences of the board's integrated yaw, so a reply that arrives a
        tick late loses nothing: that tick counts 0 and the next one counts
        both. After ANY fallback the chain restarts (one more fallback tick),
        so a turn is never counted twice -- once commanded, once measured.
        Also feeds the body detector (stage 4d reads it next tick)."""
        m = self.motors
        if not hasattr(m, "request_imu"):
            return None
        imu = m.imu(max_age_s=0.5)
        m.request_imu()
        if imu is not None and "tilt_deg" in imu:
            if self._body is None:
                self._init_body()
            if self._body is not None:
                self._body_update(imu)
        if imu is None or not imu.get("ok"):
            if self._gyro_prev is not None or self.gyro_seen:
                self.gyro_gaps += 1
            self._gyro_prev = None
            self.stats["gyro_fallback_ticks"] = self.stats.get("gyro_fallback_ticks", 0) + 1
            return None
        self.gyro_seen = True
        n = m.imu_replies
        if self._gyro_prev is None:
            self._gyro_prev = (n, imu["yaw_deg"])
            self.gyro_gaps += 1               # this tick still falls back
            return None
        if n == self._gyro_prev[0]:
            self.stats["gyro_ticks"] = self.stats.get("gyro_ticks", 0) + 1
            return 0.0                        # no new reply yet: next tick has it
        dyaw = imu["yaw_deg"] - self._gyro_prev[1]
        self._gyro_prev = (n, imu["yaw_deg"])
        if abs(dyaw) > GYRO_MAX_STEP_DEG:     # a glitch, or she was carried round
            self.gyro_gaps += 1
            self._gyro_prev = None
            return None
        self.stats["gyro_ticks"] = self.stats.get("gyro_ticks", 0) + 1
        return math.radians(dyaw)

    def _note_pivot_stall(self, ex_v: float, ex_omega: float, omega_meas: float) -> None:
        """A turn commanded but not happening, measured: counted and logged once
        per stall. (Dead reckoning is already right -- it used the measured
        turn -- this is the evidence, and what the escape logic can read.)"""
        commanded = abs(ex_omega) >= PIVOT_STALL_CMD and abs(ex_v) < 0.05
        if commanded and abs(omega_meas) < PIVOT_STALL_MEAS:
            self._stall_ticks += 1
            if self._stall_ticks == PIVOT_STALL_TICKS:
                self.stats["pivot_stalls"] = self.stats.get("pivot_stalls", 0) + 1
                LOGGER.info("GYRO: a turn was commanded (%.2f rad/s) but she is not "
                            "turning (%.2f rad/s) -- stalled turn #%d", ex_omega,
                            omega_meas, self.stats["pivot_stalls"])
        else:
            self._stall_ticks = 0

    @property
    def pivot_stalled(self) -> bool:
        return self._stall_ticks >= PIVOT_STALL_TICKS

    def _init_body(self) -> None:
        try:
            if str(HERE.parent) not in sys.path:
                sys.path.append(str(HERE.parent))
            import body_sense as _bs
            self._body = _bs.Detector()
            self._body_publish = _bs.publish
            self._body_seq = int((_bs.read() or {}).get("seq", 0))
        except Exception as exc:                          # noqa: BLE001
            LOGGER.warning("body sense unavailable in the run: %s", exc)
            self._body = False              # not again

    def _body_update(self, imu: dict) -> None:
        """Picked up / put down / shaken... during a run: the same detector and
        the same file as friday-mcp's (body_sense.py), so friday-brain reacts
        while she explores too."""
        try:
            last = self.motors.last_cmd if hasattr(self.motors, "last_cmd") else (0.0, 0.0)
            driving = abs(last[0]) > 0.0 or abs(last[1]) > 0.0
            for event in self._body.update(imu, time.monotonic(), driving):
                self._body_seq += 1
                LOGGER.info("BODY: %s (tilt %.0f deg, shake %.1f deg/s)", event,
                            imu.get("tilt_deg", 0.0), imu.get("shake_dps", 0.0))
                self.stats["body_events"] = self.stats.get("body_events", 0) + 1
                if self._body_publish is not None:
                    self._body_publish(event, self._body.lifted, self._body_seq,
                                       tilt_deg=imu.get("tilt_deg"), during="run")
        except Exception as exc:                          # noqa: BLE001
            LOGGER.debug("body update failed: %s", exc)

    def _commit_turn(self, v: float, omega: float, gate: str, rec) -> float:
        now = time.monotonic()
        if gate in self.TURN_COMMIT_EXEMPT or abs(v) >= self.TURN_COMMIT_MAX_V:
            self._turn_sign = 0             # driving, or a deliberate turn: nothing kept
            return omega
        if abs(omega) < self.TURN_COMMIT_MIN_OMEGA:
            return omega                    # a stop / nudge passes; the commitment stays
        s = 1 if omega > 0 else -1
        if self._turn_sign == 0 or now - self._turn_since >= self.turn_commit_s:
            if s != self._turn_sign:
                self._turn_sign, self._turn_since = s, now
            return omega
        if s == self._turn_sign:
            return omega
        self.stats["turn_commits"] = self.stats.get("turn_commits", 0) + 1
        rec.add("turn_commit", v, -omega,
                f"kept turning {'left' if self._turn_sign > 0 else 'right'} "
                f"({now - self._turn_since:.1f} s of {self.turn_commit_s:.1f} s)")
        return -omega

    def _drain_unstick(self) -> None:
        """Say what unstick decided, and keep the planner away from where she got
        stuck. Cheap: nothing to do on almost every tick."""
        u = self.unstick
        while u.events:
            key, text = u.events.pop(0)
            LOGGER.info("UNSTICK %s: %s (%s)", key, text, u.snapshot())
            self._say(key, text)
        if len(u.avoid) > self._avoid_seen and self.planner is not None:
            keys = u.avoid_keys()
            with self.planner._lock:
                self.planner.blacklist |= keys
                t = self.planner.target
                if (t is not None and self.planner.home is None
                        and t.key() in keys):
                    self.planner.target = None
                    self.planner.targets_abandoned += 1
            LOGGER.info("UNSTICK: avoiding %d stuck spot(s) -- %d frontier keys "
                        "blacklisted", len(u.avoid), len(keys))
            self._avoid_seen = len(u.avoid)

    def _steer_home(self) -> None:
        home = self.way_back.home()
        if home is not None and self._home_target is not None:
            self._home_target.approach = (float(home[0]), float(home[1]))
        if self.planner is not None and self._home_target is not None:
            self.planner.go_home(self._home_target)

    def _finish_way_back(self, outcome: str) -> None:
        wb = self.way_back
        self.motors.drive(0.0, 0.0)
        d = wb.distance_m
        took = time.monotonic() - (wb.started_at or time.monotonic())
        LOGGER.info("WAY BACK %s after %.0f s, %s from the start (%s).",
                    outcome.upper(), took, "?" if d is None else f"{d:.2f} m",
                    wb.tracker.snapshot())
        voice = getattr(self, "voice", None) or self.narrator
        said = getattr(voice, "spoken", 0)
        taken = self._say("way_back_done",
                          "I am back where I started." if outcome == "arrived" else
                          (f"I could not get all the way back. I am about {d:.1f} "
                           "metres from where I started." if d is not None else
                           "I could not get all the way back."))
        # Let the line be heard before shutdown stops the voice. Both voices count
        # what they have finished saying in `spoken`.
        deadline = time.monotonic() + 12.0
        while (taken and getattr(voice, "spoken", said) == said
               and time.monotonic() < deadline):
            time.sleep(0.1)

    def shutdown(self) -> None:
        self._stop.set()
        self.motors.emergency_stop()
        if self._reflex_thread is not None:
            self._reflex_thread.join(timeout=1.0)
        if self.detector is not None:
            self.detector.stop()
            self.detector.join(timeout=2.0)
        try:
            self.narrator.finish(self.autonomy_state())
        except Exception:                              # noqa: BLE001
            pass
        self.narrator.stop()
        if self.way_back is not None:
            self.way_back.tracker.stop()
            LOGGER.info("Way back: %s", self.way_back.snapshot())
        if self.unstick_enabled:
            LOGGER.info("Unstick: %s", self.unstick.snapshot())
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
            objective=self._objective(), stats=self.stats) | (
                {"way_back": self.way_back.snapshot()}
                if self.way_back is not None else {}) | self._objects_state()

    def _objects_state(self) -> dict[str, Any]:
        """FILE 21 U4 (spec 4.7 point 9): what the run's OWN detector sees, for
        the mission's room-name suggestion -- read-only, and only while that
        detector is running and has looked at a picture."""
        det = self.detector
        if det is None or not getattr(det, "enabled", False) or not det.frames:
            return {}
        return {"OBJECTS": [{"label": d.label, "score": round(float(d.score), 3)}
                            for d in det.snapshot()],
                "OBJECTS_FRAMES": int(det.frames)}

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
        if self.way_back is not None and self.way_back.phase == "returning":
            d = self.way_back.distance_m
            return ("going back to where the run started"
                    + ("" if d is None else f", {d:.2f} m away by the start tracker"))
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
        # SAY WHICH TICKS. self.loop_ms is a deque(maxlen=200) -- twenty seconds of a
        # twelve-minute run -- and this line has been read as the run's mean all night.
        # It reported 215 ms for a run whose whole-run mean (from the black box) was
        # 160. Same ring-buffer-as-summary mistake as the black box itself.
        LOGGER.info("loop ms over the LAST %d ticks: mean %.2f | p95 %.2f | max %.2f "
                    "(budget %.0f) -- the black box has the whole run",
                    lm.size, lm.mean(), np.percentile(lm, 95), lm.max(),
                    self.cfg.dt * 1000)
        if self.stage_s and self.stats["ticks"]:
            n = self.stats["ticks"]
            acc = sorted(self.stage_s.items(), key=lambda kv: -kv[1])
            LOGGER.info("budget by stage, mean ms/tick over all %d ticks: %s", n,
                        "  ".join(f"{k} {1000.0 * v / n:.1f}" for k, v in acc))
            in_tick = 1000.0 * self.stage_wall_s / n
            wall = float(lm.mean())
            # THE GAP IS THE POINT. Everything timed above runs on this one thread, so
            # work cannot hide between the stages -- only DESCHEDULING can. A gap means
            # the mapper, the frontier planner or the camera threads are holding the
            # GIL, and no amount of optimising this loop will recover it.
            LOGGER.info("  measured inside tick(): %.1f ms/tick; "
                        "wall clock per tick (last %d): %.1f ms; "
                        "unaccounted %.1f ms = time this thread was NOT RUNNING "
                        "(other threads holding the GIL)",
                        in_tick, lm.size, wall, max(wall - in_tick, 0.0))
        s = self.stats
        if self.navigator is not None and (s["nav_decisions"] or s["nav_handoffs"]):
            total = max(s["nav_decisions"] + s["nav_handoffs"], 1)
            n = self.navigator
            LOGGER.info("navigator: %d ticks steered (%.0f%%), %d handed back "
                        "(%d blocked ahead, %d no drivable arc); "
                        "open %d / corridor %d / cluttered %d",
                        s["nav_decisions"], 100.0 * s["nav_decisions"] / total,
                        s["nav_handoffs"], n.handoff_blocked, n.handoff_no_arc,
                        s["nav_open"], s["nav_corridor"], s["nav_cluttered"])
            LOGGER.info("  near field: %d ticks with a ToF barrier in the arc search "
                        "(%d of them already inside the reflex's stop distance -- the "
                        "other %d are arcs steered around something the reflex could "
                        "only have braked for), %d ticks speed-capped by it",
                        n.tof_informed, n.tof_braked,
                        max(n.tof_informed - n.tof_braked, 0), n.tof_slowed)
            # SWITCHES are the number that matters -- a direction reversal a rival had
            # to earn. Expiries are just COMMIT_MAX_S elapsing and say nothing about
            # flip-flopping; counting them together read as "94 broken vs 28 held" and
            # made a working mechanism look broken.
            LOGGER.info("  commitments %d taken, %d reversed by a better option, "
                        "%d expired on age; arc choice %.2f ms mean "
                        "(a HIGH handback count means she is spending her time in "
                        "places with no drivable way forward)",
                        self.navigator.commitments, self.navigator.commit_switches,
                        self.navigator.commit_expiries, self.navigator.planner.mean_ms)
        LOGGER.info("escape turns: %d  (boxed in: forward blocked and not moving, so "
                    "she turned toward open space)", self.stats["escape_turns"])
        LOGGER.info("reverse lifts: %d  (policy asked for a reverse too gentle to "
                    "execute; honoured at the executable v_min instead of pivoting)",
                    self.stats.get("reverse_lifts", 0))
        LOGGER.info("reverse floor lifts: %d  (a reverse from ANY stage -- escape, "
                    "recovery, the policy -- that was under the stiction floor and "
                    "would have executed as a stop, reissued at the slowest speed the "
                    "wheels can actually turn)", self.stats["reverse_floor_lifts"])
        LOGGER.info("no-reverse turns: %d  (policy asked to back up; this drivetrain "
                    "cannot, so it turned toward open space instead)",
                    self.stats["no_reverse_turns"])
        LOGGER.info("interventions: reflex=%d visual=%d governor=%d rear=%d stale=%d "
                    "scrape=%d overruns=%d",
                    self.stats["reflex_stops"], self.stats["visual_stops"],
                    self.stats["governor_limits"], self.stats.get("rear_blocks", 0),
                    self.stats["stale_stops"], self.governor.scrape_brakes,
                    self.stats["overruns"])
        if self.detector is not None and self.detector.enabled:
            seen = self.detector.snapshot()
            LOGGER.info("object detection: %d frames, %.0f ms mean, %d failures; "
                        "currently sees %s", self.detector.frames,
                        self.detector.mean_ms, self.detector.failures,
                        ", ".join(f"{d.label}({d.score:.2f})" for d in seen) or "nothing")
        LOGGER.info("cameras: free-space updates=%d failures=%d, nudges=%d, "
                    "ticks driven on vision=%d",
                    self.freespace.updates, self.freespace.failures,
                    self.stats.get("vision_nudges", 0),
                    self.stats.get("vision_ticks", 0))
        LOGGER.info("lidar: %d sub-footprint returns rejected outside the self-mask "
                    "(a growing count means the chassis wedge moved -- re-measure "
                    "--lidar-self-mask); %d ticks braked for closing rate alone",
                    self._lidar_body_rejects(), self.governor.ttc_brakes)
        # Counted separately from `governor` because the rear gate answers a different
        # question and its label never survives to the blackbox: stage 4 overwrites
        # `gate` with REFLEX_STOP whenever the reflex also fires, which on this robot
        # is most of the time. Without this line there is no way to tell a reverse
        # refused for safety from a reverse never requested.
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
        # The up camera's frame in COLOUR, for the object detector only. Separate
        # from cam_up because everything else in this file wants grayscale and
        # would otherwise have to know which it was handed.
        "cam_up_rgb": SensorSlot("cam_up_rgb", cfg.cam_deadline, critical=False),
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
        # The way back's 1-degree sweep (rl/way_back.py). Not critical: without
        # it the way back matches the 72 bins instead. Not attached at all with
        # --no-way-back, which is then the old run exactly.
        if not getattr(args, "no_way_back", False):
            slots["lidar_fine"] = SensorSlot("lidar_fine", cfg.lidar_deadline,
                                             critical=False)
            threads[-1].fine_slot = slots["lidar_fine"]
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
            cam = CameraThread(slots[f"cam_{role}"], cfg, role,
                               serial=serial, path=override)
            if role == "up" and getattr(cfg, "detect_enabled", False):
                cam.colour_slot = slots["cam_up_rgb"]
            threads.append(cam)
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
                     start: int = 20, step: int = 8, hold: float = 1.4,
                     direction: int = 1, announce=None) -> None:
    """Ramp PWM duty upward so you can see where the wheels actually start turning.

    This measures the ONE number the m/s -> duty mapping cannot be honest without: the
    stiction threshold. Below it a brushed motor sits energised and humming, which is
    what you just heard. Note the duty at which each side begins to rotate steadily,
    and that becomes --motor-min-duty.

    `direction` is +1 for forward, -1 for REVERSE, and the reverse figure is not
    cosmetic. The floor was 72, measured driving forward, and the deployed code reused
    it for reverse. Measured consequence: wedged 0.09 m from a wall, recovery issued 81
    reverse commands at that floor and the robot did not move at all, ending the run
    0.02 m CLOSER than it started.

    RESOLVED 2026-09-02 by running this sweep in both directions with the motor
    inversion finally correct: break-away is 80 forward and 80 reverse. The old 72 sat
    BELOW the real threshold, so every reverse command energised the motors just under
    break-away and they hummed. Reverse turning out to match forward is a result, not
    an assumption -- a pivot still needs 190, because scrubbing sideways is a different
    load.
    """
    ceiling = int(255 * motors.max_duty)
    label = "FORWARD" if direction >= 0 else "REVERSE"
    LOGGER.warning("=" * 74)
    LOGGER.warning("DUTY SWEEP (%s) -- wheels WILL turn. Robot on a stand.", label)
    LOGGER.warning("Watch for the duty where the wheels start turning STEADILY (not humming).")
    LOGGER.warning("Cap is %d (max_duty=%.2f). Raise --motor-max-duty if it never moves.", ceiling, motors.max_duty)
    LOGGER.warning("=" * 74)
    time.sleep(2.0)

    sign = 1 if direction >= 0 else -1
    first_move = None
    try:
        for duty in range(start, ceiling + 1, step):
            LOGGER.info(">>> %s duty %3d  (%3.0f%% of cap)  %s",
                        label, duty, 100.0 * duty / max(ceiling, 1),
                        "<-- watch here" if first_move is None else "")
            # Said ALOUD, because the person who has to spot the break-away is
            # watching the wheels, not a terminal. Without this the measurement
            # depends on correlating a moving wheel with a scrolling log.
            if announce is not None:
                announce(duty)
            deadline = time.monotonic() + hold
            while time.monotonic() < deadline:
                # re-send: also pets the watchdog
                motors.drive_duty(sign * duty, sign * duty)
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


def build_motors(cfg: RobotConfig, args) -> MotorDriver:
    """The ONE place a motor driver is constructed from parsed arguments.

    There were two, and they had already drifted: friday_localnav_deploy -- which is
    the file every real exploration run actually uses -- passed max_duty, watchdog,
    the inversions, min_duty, creep_cutoff and turn_min_duty, and silently dropped
    turn_kick_duty, turn_kick_cooldown_s and pulse_creep_duty. So the pivot break-away
    kick and the creep pulse ran on their defaults no matter what was typed on the
    command line, and nothing said so. A flag that is accepted and ignored is worse
    than one that is rejected.

    Anything added to FridayMotorDriver from here on is passed by this function and
    therefore reaches both entry points, which is the whole point of it existing.
    """
    if args.motor == "friday":
        motors = FridayMotorDriver(
            cfg, resolve_serial_port("motor", args.motor_port),
            max_duty=args.motor_max_duty, watchdog_s=args.motor_watchdog,
            invert_left=args.motor_invert_left, invert_right=args.motor_invert_right,
            min_duty=args.motor_min_duty, creep_cutoff=args.motor_creep_cutoff,
            turn_min_duty=args.motor_turn_min_duty,
            turn_min_omega=args.motor_turn_min_omega,
            turn_kick_duty=args.motor_turn_kick_duty,
            turn_kick_cooldown_s=args.motor_turn_kick_cooldown,
            pulse_creep_duty=args.motor_pulse_duty,
            pulse_burst_ticks=args.motor_pulse_burst,
            kick_rest_s=args.motor_kick_rest,
            plug_guard=args.motor_plug_guard,
            steady_pivots=args.motor_steady_pivots,
            pivot_hold_ticks=args.motor_pivot_hold,
            start_debounce_ticks=args.motor_start_debounce,
        )
        LOGGER.warning("MOTORS ARE LIVE (max_duty=%.2f). Wheels can turn.",
                       args.motor_max_duty)
        return motors
    if args.motor == "serial":
        return SerialMotorDriver(cfg, args.motor_port)
    motors = NullMotorDriver(cfg, verbose=args.verbose)
    LOGGER.warning("Motor driver = NULL: commands are logged, wheels will NOT turn. "
                   "Use --motor friday once you have bench-tested.")
    return motors


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

    # 1. governor must forbid forward motion at/below hard_stop.
    # A REAL obstacle spans several beams -- at 72 beams each is 5 deg, and anything
    # solid at 0.3 m subtends far more than the three the law now requires. The single
    # hot beam this test used to present is not a wall, it is exactly the speckle the
    # k-th-smallest rule exists to ignore, which is asserted separately at 1a.
    loop.governor._prev_fwd = None          # no stale sample -> no closing-rate effect
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[:3] = cfg.hard_stop - 0.05
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("governor blocks forward inside hard_stop", v <= 0.0, f"v={v:.3f}")

    # 1a. ...and ONE spurious beam must not. This is the property the rear gate and the
    # ToF reflex already had and the forward law did not: a body return or a speckle
    # does not move when the robot pivots, so a min-based law that trips on one is
    # never cleared by anything the robot can do. Measured on 2026-09-03: the whole-scan
    # minimum was inside robot_radius on 63% and 65% of ticks in two runs.
    loop.governor._prev_fwd = None
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[0] = 0.05
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("one spurious forward beam does not block forward", v > 0.0, f"v={v:.3f}")

    # 1a-ii. A LiDAR return inside the robot's own footprint is not a thing to steer
    # around, and must never reach the scan at all. Asserted on the parser, because
    # that is where it is rejected -- five consumers read the scan and fixing it in
    # one leaves the other four believing it.
    check("sub-footprint ranges are rejected as no-return",
          cfg.lidar_floor >= cfg.robot_radius,
          f"floor={cfg.lidar_floor:.2f} m")

    # 1a-iii. THE CORRIDOR. This is the whole zig-zag fix, and it is the assertion
    #     that would have caught it: a hallway narrower than 2 * safe_distance put its
    #     own walls inside the old front-hemisphere arc, so the braking law sat in the
    #     braking band the entire length of a corridor with nothing in it. She braked,
    #     turned away, met the other wall, and turned back -- 419 degrees of turning
    #     per metre travelled over the 6-minute run of 2026-09-03 18:44.
    loop.governor._prev_fwd = None
    scan = np.full(cfg.n_beams, 5.0, np.float32)
    side = int(round(cfg.n_beams / 4))                  # 90 deg
    scan[side - 4:side + 4] = 0.55                      # walls 0.55 m either side
    scan[3 * side - 4:3 * side + 4] = 0.55
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("a corridor she fits down does not brake her", v >= cfg.v_max - 1e-6,
          f"v={v:.3f} clearance={loop.governor.last_fwd_min:.2f} m")

    # 1a-iv. ...and the guard that keeps the old arc's real value. The corridor rule
    #     alone cannot see the thing she is about to catch with a shoulder while she
    #     turns, so anything inside her own footprint at ANY forward bearing still
    #     brakes. Three beams, because one is speckle.
    loop.governor._prev_fwd = None
    scan = np.full(cfg.n_beams, 5.0, np.float32)
    scan[side - 1:side + 2] = cfg.scrape_radius - 0.02  # level with her, touching
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("something touching her shoulder still brakes", v <= 0.0,
          f"v={v:.3f} scrape={loop.governor.last_scrape:.2f} m")

    # 1a-v. A thin obstacle dead ahead spans only a few beams once it is far enough
    #     away that the corridor is narrow. Taking the k-th smallest of a 3-beam band
    #     would be taking its MAXIMUM, and she would drive straight through it.
    loop.governor._prev_fwd = None
    scan = np.full(cfg.n_beams, 5.0, np.float32)
    scan[0] = scan[1] = cfg.hard_stop - 0.05
    scan[-1] = 5.0
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("a narrow obstacle dead ahead is not averaged away", v <= 0.0,
          f"v={v:.3f} clearance={loop.governor.last_fwd_min:.2f} m")

    loop.governor._prev_fwd = None
    # 1b. ...but a wall BEHIND her must not. Driving forward cannot reach it, and
    # vetoing on it made her undriveable in any room with a wall in it.
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[cfg.n_beams // 2] = cfg.hard_stop - 0.05       # directly astern
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("a wall behind does not block forward", v > 0.0, f"v={v:.3f}")

    # 1c. A wall behind MUST block REVERSE. Nothing checked this before: the forward
    #     braking law only ever lowers v, so reverse passed through untouched, and the
    #     ToF is a forward-facing patch -- so a reverse command was executed with no
    #     obstacle check behind the robot by either sensor. She backed into things.
    rear_i = cfg.n_beams // 2
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    # A real obstacle spans several beams; the gate requires rear_min_valid_beams to
    # agree, so the test has to present one rather than a single hot beam.
    scan[rear_i - 1:rear_i + 2] = cfg.rear_stop - 0.05
    v, _, _ = loop.governor(cfg.v_min, 0.0, scan)
    check("a wall behind blocks reverse", v >= 0.0, f"v={v:.3f}")

    # ...and ONE spurious beam must NOT. 5 of 316 ticks in a 90 s run carried a return
    # below 0.22 m, which is inside her own body and therefore impossible; a min-based
    # gate turns each into a refused escape.
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[rear_i] = 0.04
    v, _, _ = loop.governor(cfg.v_min, 0.0, scan)
    check("one spurious rear beam does not veto reverse", v < 0.0, f"v={v:.3f}")

    # 1d. ...and a CLEAR rear must still allow it. Over-gating reverse is not a safe
    #     default: it is how she ended up able only to spin in a doorway.
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[0] = cfg.hard_stop - 0.05                      # blocked ahead, clear astern
    v, _, _ = loop.governor(cfg.v_min, 0.0, scan)
    check("a clear rear still permits reverse escape", v < 0.0, f"v={v:.3f}")

    # 1e. A wall behind must not start blocking FORWARD as a side effect of 1c.
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[rear_i - 1:rear_i + 2] = cfg.rear_stop - 0.05
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("the rear gate does not leak into forward", v > 0.0, f"v={v:.3f}")

    # 1d. THE WALL SHE REVERSED INTO. Harsh, 2026-09-03: "it wasn't able to detect a
    # wall behind it and if an obstacle was moved from behind it kept moving backwards
    # without noticing the wall". rear=0 across 334 reversing ticks was not "nothing
    # was behind her" -- it was a 0.26 m threshold on a robot that coasts, arriving at
    # the full v_min because reverse has no proportional band to slow her first.
    loop.governor._prev_fwd = None
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    rear_i = cfg.n_beams // 2
    scan[rear_i - 1:rear_i + 2] = 0.30           # a wall 0.30 m behind: inside the coast
    v, _, _ = loop.governor(cfg.v_min, 0.0, scan)
    check("a wall 0.30 m behind now blocks reverse (0.26 m gate let it through)",
          v >= 0.0, f"v={v:.3f} vs rear_stop {cfg.rear_stop:.2f}")

    # ...and the margin must actually cover the way she stops. She coasts 0.22 m from
    # 0.403 m/s; scaled to the 0.15 m/s reverse that is 0.031-0.082 m depending on
    # whether coast goes as v or v^2, plus 0.033 m for one more tick at loop p95.
    check("rear_stop covers the coast plus a control tick",
          cfg.rear_stop - cfg.robot_radius >= 0.115,
          f"margin {cfg.rear_stop - cfg.robot_radius:.3f} m vs 0.115 m budget")

    # 1e. REVERSE IS SINGLE-SENSOR, so it may not be less sensitive than forward. The
    # ToF is a forward-facing bumper patch; behind her the LiDAR is all there is.
    check("the rear needs fewer agreeing beams than the front, not more",
          cfg.rear_min_valid_beams <= cfg.forward_min_valid_beams,
          f"rear k={cfg.rear_min_valid_beams} front k={cfg.forward_min_valid_beams}")
    # A narrow obstacle -- a chair leg is 3-5 cm -- must still be seen at close range.
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[rear_i:rear_i + 2] = 0.30               # 2 beams: ~5 cm wide at 0.3 m
    v, _, _ = loop.governor(cfg.v_min, 0.0, scan)
    check("a narrow obstacle behind blocks reverse", v >= 0.0, f"v={v:.3f}")
    # ...but one lone beam still must not, or a speckle vetoes every escape she has.
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[rear_i] = 0.05
    v, _, _ = loop.governor(cfg.v_min, 0.0, scan)
    check("one speckled rear beam still does not veto reverse", v < 0.0, f"v={v:.3f}")

    # 2. governor must scale speed inside the bubble
    loop.governor._prev_fwd = None
    scan = np.full(cfg.n_beams, 3.0, np.float32)
    scan[:3] = (cfg.hard_stop + cfg.safe_distance) / 2.0
    v, _, _ = loop.governor(cfg.v_max, 0.0, scan)
    check("governor brakes inside safety bubble", 0.0 < v < cfg.v_max, f"v={v:.3f}")

    # 2b. CLOSING RATE. Two ticks with the same clearance must brake identically to the
    # distance-only law; a gap that is shrinking fast must brake harder than its
    # distance alone says. This is the only thing in the chain that can see something
    # walking toward the robot, and it must never work the other way -- a WIDENING gap
    # may not grant extra speed, or the law would accelerate toward a retreating
    # obstacle it has not yet resolved.
    far = np.full(cfg.n_beams, 3.0, np.float32)
    near = np.full(cfg.n_beams, 3.0, np.float32)
    near[:3] = cfg.safe_distance + 0.10          # still outside the bubble on distance
    loop.governor._prev_fwd = None
    loop.governor(cfg.v_max, 0.0, far)           # seed: 3.0 m ahead
    # Back-to-back calls are microseconds apart and the rate estimator correctly
    # ignores them (see the dt bound). Age the sample by one control tick so the test
    # exercises the arithmetic rather than the guard.
    loop.governor._prev_t -= cfg.dt
    v_closing, _, _ = loop.governor(cfg.v_max, 0.0, near)
    loop.governor._prev_fwd = None
    v_static, _, _ = loop.governor(cfg.v_max, 0.0, near)
    check("closing obstacle brakes earlier than distance alone",
          v_closing < v_static, f"closing={v_closing:.3f} static={v_static:.3f}")
    loop.governor._prev_fwd = None
    loop.governor(cfg.v_max, 0.0, near)          # seed: close
    loop.governor._prev_t -= cfg.dt
    v_opening, _, _ = loop.governor(cfg.v_max, 0.0, far)
    check("a widening gap never grants extra speed",
          v_opening <= cfg.v_max + 1e-9, f"v={v_opening:.3f}")

    # 2b-ii. A REVERSE TOO GENTLE TO EXECUTE IS LIFTED, NOT REPLACED BY A PIVOT.
    # She drove into something on 2026-09-03 and sat with tof_min at 0.020 m for all
    # 1233 ticks of a 180 s run while the policy asked to reverse on 297 of them; every
    # one became a pivot, and a pivot cannot free a wedged robot because its footprint
    # does not translate. Asserted on the arithmetic that decides it, because the branch
    # itself needs a whole ControlLoop and the thing that was wrong was the rule.
    floor_mps = cfg.robot_radius * 0.0 + 0.149        # measured stiction floor, m/s
    check("a reverse below the floor is still executable at v_min",
          abs(cfg.v_min) >= floor_mps,
          f"v_min {cfg.v_min:+.3f} vs floor {floor_mps:.3f}")
    check("...so the pivot substitution is reserved for drivetrains that truly cannot",
          not (abs(cfg.v_min) < floor_mps), f"margin {abs(cfg.v_min) - floor_mps:+.3f} m/s")

    # 2c. A PIVOT IS NOT SLEW LIMITED. The first tick of a turn from standstill used to
    # be issued at max_ang_accel * dt = 0.40 rad/s, which this drivetrain executes as a
    # full stop -- 22% of every tick logged on 2026-09-03. Rotating about its own centre
    # translates nowhere, so there is no travel for a slew limit to protect.
    lim = RateLimiter(cfg)
    v_p, w_p = lim(0.0, cfg.omega_max, cfg.dt)
    check("a standing pivot is issued in full on the first tick",
          abs(w_p - cfg.omega_max) < 1e-9, f"omega={w_p:.3f}")
    # ...but reversing a spinning pivot crosses zero first -- plugging four motors on
    # two L298N channels is the brownout this project cannot afford.
    _, w_rev = lim(0.0, -cfg.omega_max, cfg.dt)
    check("a pivot reversal passes through zero", abs(w_rev) < 1e-9, f"omega={w_rev:.3f}")
    _, w_after = lim(0.0, -cfg.omega_max, cfg.dt)
    check("...and reaches full rotation the very next tick",
          abs(w_after + cfg.omega_max) < 1e-9, f"omega={w_after:.3f}")
    # ...but a TRANSLATING turn still is: there the wheels can slip, and slip is what
    # corrupts the only odometry this robot has.
    lim2 = RateLimiter(cfg)
    lim2(cfg.v_max, 0.0, cfg.dt)                 # now translating
    _, w_t = lim2(cfg.v_max, cfg.omega_max, cfg.dt)
    check("a turn while translating stays slew limited",
          w_t <= cfg.max_ang_accel * cfg.dt + 1e-9, f"omega={w_t:.3f}")

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

    # 5c. FORWARD must never survive an emergency stop, whatever else is allowed.
    loop.limiter.v = 0.5
    v, _ = loop.limiter(0.6, 0.0, cfg.dt, emergency=True, allow_rotation=True)
    check("emergency stop never lets forward through", v == 0.0, f"v={v:.3f}")

    # 5d. ...but REVERSE must, because the ToF looks forward and backing off is the
    # escape. Same rule LidarGovernor already applies: "rotate / reverse only".
    loop.limiter.reset()
    v, _ = loop.limiter(-0.15, 0.0, cfg.dt, emergency=True)
    check("emergency stop permits reverse escape", v == -0.15, f"v={v:.3f}")

    # 6. rate limiter must cap acceleration
    loop.limiter.reset()
    v, _ = loop.limiter(cfg.v_max, 0.0, cfg.dt)
    check("acceleration is slew limited", v <= cfg.max_lin_accel * cfg.dt + 1e-6, f"v={v:.3f}")

    # 7. wheel speeds must stay legal for any command
    l, r = loop.motors.to_wheel_speeds(cfg.v_max, cfg.omega_max)
    check("wheel speeds within limit",
          max(abs(l), abs(r)) <= cfg.wheel_speed_max + 1e-6, f"L={l:.3f} R={r:.3f}")

    # 6b. The slew limit must scale with REAL elapsed time, and stay bounded.
    #     Feeding it the nominal dt on a 250 ms tick allowed 2.5x too little
    #     acceleration, so she crawled up to speed by an amount that varied with
    #     however long the tick happened to take.
    loop.limiter.reset()
    v_short, _ = loop.limiter(cfg.v_max, 0.0, cfg.dt)
    loop.limiter.reset()
    v_long, _ = loop.limiter(cfg.v_max, 0.0, cfg.dt * 3.0)
    check("slew allowance scales with elapsed time",
          v_long > v_short * 2.5, f"{v_short:.3f} at dt vs {v_long:.3f} at 3*dt")
    #     ...and tick() must clamp it, so a long stall cannot bank a lurch.
    loop.limiter.reset()
    v_huge, _ = loop.limiter(cfg.v_max, 0.0, cfg.dt * 50.0)
    check("an unclamped dt would be a lurch (hence the 3x cap in tick)",
          v_huge > cfg.v_max * 0.9, f"v={v_huge:.3f}")

    # 7b. THE WHOLE CHAIN, not just the governor: with the rear blocked, no stage may
    #     output reverse. 1c-1e test the governor in isolation, but escape (2b) and
    #     recovery (2c) run downstream of it and author reverse of their own, so the
    #     isolated tests would pass while the robot still backed into a wall.
    boxed = np.full(cfg.n_beams, 3.0, np.float32)
    boxed[0] = cfg.hard_stop - 0.05                     # blocked ahead -> wants reverse
    boxed[cfg.n_beams // 2 - 1:cfg.n_beams // 2 + 2] = cfg.rear_stop - 0.05   # behind too
    # Recovery is FORCED to demand reverse. Waiting for it to trigger naturally made
    # this test pass with the gate deleted -- it never reached a boxed_in trigger in
    # the tick budget, so it proved nothing. The point is the WIRING: that whatever
    # recovery returns is gated, not that recovery happens to fire.
    real_update = loop.recovery.update
    loop.recovery.update = lambda *a, **k: (cfg.v_min, 0.0, "forced reverse for selftest")
    worst_v = 0.0
    try:
        for _ in range(10):
            loop.slots["lidar"].publish(boxed)
            worst_v = min(worst_v, float(loop.tick()["v"]))
    finally:
        loop.recovery.update = real_update
    check("no stage reverses into a blocked rear", worst_v >= 0.0, f"min v={worst_v:.3f}")

    # 7c. THE REVERSE FLOOR LIFT (stage 2e), both halves.
    #
    # It exists because a reverse under the stiction floor executes as a STOP, and
    # stage 1b only ever saw the ones the POLICY authored. Measured in the 12-minute
    # run of 2026-09-04 17:21: recovery issued reverses averaging -0.063 m/s against a
    # 0.149 m/s floor, all of them silently nothing, while she spun in place for 65 s
    # with 1.26 m of clear space behind her.
    # NullMotorDriver reports no stiction floor (correctly -- it has no wheels), so
    # this whole check was dead code the first time it was written: the guard was false
    # and nothing ran. The floor is what makes the bug possible, so the sim is given
    # the REAL one measured on this drivetrain, 0.149 m/s at min_duty 80.
    floor_mps = 0.149
    real_floor = loop.motors.min_speed_mps
    loop.motors.min_speed_mps = lambda: floor_mps
    if abs(cfg.v_min) >= floor_mps:
        gentle = -floor_mps * 0.4                       # far too gentle to execute
        loop.recovery.update = lambda *a, **k: (gentle, 0.0, "sub-floor reverse")
        try:
            # Rear CLEAR: the lift must make it executable.
            clear_rear = np.full(cfg.n_beams, 3.0, np.float32)
            clear_rear[0] = cfg.hard_stop - 0.05        # blocked ahead, so she wants out
            best_v = 0.0
            for _ in range(10):
                loop.slots["lidar"].publish(clear_rear)
                best_v = min(best_v, float(loop.tick()["v"]))
            check("a sub-floor reverse from RECOVERY is lifted to an executable speed",
                  best_v <= -floor_mps + 1e-6,
                  f"min v={best_v:.3f} vs floor {-floor_mps:.3f}")

            # Rear BLOCKED: the rear gate has already raised it to 0, so there is
            # nothing negative left for the lift to act on. THIS IS THE PROPERTY THAT
            # MAKES THE LIFT SAFE -- it changes a magnitude, it never grants authority,
            # and it runs after the gate precisely so that ordering holds.
            worst_v = 0.0
            for _ in range(10):
                loop.slots["lidar"].publish(boxed)
                worst_v = min(worst_v, float(loop.tick()["v"]))
            check("...and it cannot resurrect a reverse the rear gate refused",
                  worst_v >= 0.0, f"min v={worst_v:.3f}")
        finally:
            loop.recovery.update = real_update
            loop.motors.min_speed_mps = real_floor

    # 8. stale LiDAR must stop the robot -- WHEN THERE IS NOTHING ELSE TO SEE WITH.
    #    The cameras are cleared first so this asserts the original fail-safe rather
    #    than whatever the free-space fallback happened to be holding.
    loop.freespace.last = loop.free_view = fs_mod.FreeSpaceView(
        np.zeros(9, np.float32), 0.0, 0.0, 0.0, "none")
    loop._cam_seen.clear()
    for key in ("cam_up", "cam_down"):
        if key in loop.slots:
            loop.slots[key]._value = None
    loop.slots["lidar"]._stamp = time.monotonic() - 10.0
    row = loop.tick()
    check("stale lidar with no camera still triggers stop",
          row["gate"] == "stale_lidar" and row["v"] == 0.0, str(row["gate"]))

    # 8a. ...but with the ToF healthy and a camera that can see, she navigates on
    #     vision instead of standing still until someone reseats a USB plug. Harsh
    #     asked for exactly this: "when the sensors are not working the device (YOU)
    #     uses the camera's vision to navigate through the environment".
    def _open_scene():
        img = np.full((240, 320), 170, np.uint8)
        img[:96, :] = 140                       # a wall well above the horizon
        return img

    # STALENESS CANNOT BE TESTED AGAINST A LIVE PUBLISHER. Under --simulate a
    # SimSensorThread republishes the ToF at 15 Hz and the LiDAR at 10, so winding a
    # slot's _stamp back 10 s is undone by the next publish a few milliseconds later.
    # These checks therefore raced: "both sensors stale is a stop" failed roughly one
    # run in five, on nothing but scheduling. A safety assertion that has to be re-run
    # to be believed is not an assertion. So the slot is FROZEN -- its publish becomes
    # a no-op -- for exactly as long as the check needs it to be stale.
    def _freeze(name):
        slot = loop.slots[name]
        real = slot.publish
        slot.publish = lambda *a, **k: None
        slot._stamp = time.monotonic() - 10.0
        return lambda: setattr(slot, "publish", real)

    loop.slots["tof"].publish(np.full((8, 8), cfg.tof_max_range, np.float32))
    if "cam_up" in loop.slots:
        loop.slots["cam_up"].publish(_open_scene())
        loop._cam_seen.clear()
        thaw_lidar = _freeze("lidar")
        row = loop.tick()
        check("a stale lidar falls back to camera vision, not to a stop",
              row["gate"] == "vision_lidar" and row["v"] > 0.0,
              f"gate={row['gate']} v={row['v']:.3f}")
        check("...and vision drives well under the normal speed limit",
              row["v"] <= cfg.freespace_v_max + 1e-6,
              f"v={row['v']:.3f} cap={cfg.freespace_v_max:.2f}")

        # 8b. THE RULE THAT MAKES 8a SAFE. The camera cannot see the ground inside
        #     0.87 m -- geometry, not tuning -- and that blind zone is the ToF's patch.
        #     With BOTH critical sensors stale there is nothing watching the metre she
        #     is about to drive through, so it must still be a stop.
        thaw_tof = _freeze("tof")
        loop.slots["lidar"]._stamp = time.monotonic() - 10.0
        loop._cam_seen.clear()
        loop.slots["cam_up"].publish(_open_scene())
        row = loop.tick()
        check("both sensors stale is a stop, however well the camera sees",
              row["v"] == 0.0 and row["gate"].startswith("stale"),
              f"gate={row['gate']} v={row['v']:.3f}")
        thaw_tof()
        thaw_lidar()

    # 8c. The advisory nudge is BOUNDED. It is a hint from a camera whose height and
    #     tilt are estimates; it may lean her toward the open side and may not take
    #     the wheel. An unbounded camera term fighting the LiDAR is a new zig-zag.
    hard_left = fs_mod.FreeSpaceView(np.ones(9, np.float32),
                                     math.radians(90.0), 3.0, 1.0, "up")
    nudge = float(np.clip(cfg.freespace_gain * hard_left.bearing,
                          -cfg.freespace_max_nudge, cfg.freespace_max_nudge))
    check("the camera advisory is capped well under omega_max",
          abs(nudge) <= cfg.freespace_max_nudge < cfg.omega_max,
          f"|nudge|={abs(nudge):.2f} cap={cfg.freespace_max_nudge:.2f} "
          f"omega_max={cfg.omega_max:.2f}")

    # 10. UNDO THE LAST MOVE (2026-09-28). First the 30 Hz reflex: it must apply
    #     the TICK's emergency rule. It used to stop the motors whatever the tick
    #     decided, so every escape the chain sanctioned near an obstacle ran for
    #     0-33 ms of each tick and she sat still "recovering".
    saved_cmd, saved_min = loop.motors.last_cmd, loop.reflex.last_min
    loop.reflex.last_min = cfg.tof_stop_distance - 0.04        # emergency, turn allowed
    loop.motors.last_cmd = (0.10, 0.0)
    check("fast reflex stops FORWARD in an emergency", loop._fast_reflex_should_stop())
    loop.motors.last_cmd = (0.10, 1.0)
    check("fast reflex stops a forward ARC in an emergency",
          loop._fast_reflex_should_stop())
    loop.motors.last_cmd = (cfg.v_min, 0.0)
    check("fast reflex lets a sanctioned reverse run", not loop._fast_reflex_should_stop())
    loop.motors.last_cmd = (0.0, cfg.omega_max)
    check("fast reflex lets a turn run outside the turn freeze",
          not loop._fast_reflex_should_stop())
    loop.reflex.last_min = cfg.tof_pivot_min_clearance - 0.02
    check("fast reflex stops a turn inside the turn freeze", loop._fast_reflex_should_stop())
    loop._undo_active = True
    check("...unless it is the undo retracing her last turn",
          not loop._fast_reflex_should_stop())
    loop.motors.last_cmd = (0.10, 0.0)
    check("...and never forward, undo or not", loop._fast_reflex_should_stop())
    loop._undo_active = False
    loop.motors.last_cmd, loop.reflex.last_min = saved_cmd, saved_min

    # 10b. THE 09-27 LOCK THROUGH THE WHOLE CHAIN. Bumper inside the turn freeze,
    #      blocked ahead, rear inside rear_stop; she had just turned into it. The
    #      chain must issue the undo turn (the other way), the limiter must let it
    #      out, and nothing may go forward.
    if loop.unstick_enabled:
        def _hold(name, value):
            slot = loop.slots[name]
            real = slot.publish
            real(value)
            slot.publish = lambda *a, **k: None
            return lambda: setattr(slot, "publish", real)

        stuck_depth = np.full((8, 8), 2.0, np.float32)
        stuck_depth[3:6, 3:6] = cfg.tof_pivot_min_clearance - 0.03
        stuck_scan = np.full(cfg.n_beams, 3.0, np.float32)
        stuck_scan[:2] = stuck_scan[-2:] = cfg.hard_stop - 0.05
        rear_i = cfg.n_beams // 2
        stuck_scan[rear_i - 2:rear_i + 3] = cfg.rear_stop - 0.03
        real_unstick = loop.unstick
        loop.unstick = _unstick.Unstick(cfg.robot_radius, cfg.v_min, cfg.omega_max,
                                        cfg.tof_pivot_min_clearance)
        for _ in range(8):
            loop.unstick.record(0.0, -1.5, cfg.dt)          # turned right into it
        thaw_t, thaw_l = _hold("tof", stuck_depth), _hold("lidar", stuck_scan)
        real_update = loop.recovery.update
        loop.recovery.update = lambda *a, **k: None         # the undo's own test
        worst_v, undo_w, undo_gate = 0.0, 0.0, ""
        try:
            for _ in range(_unstick.LOCK_TICKS + 2):
                row = loop.tick()
                worst_v = max(worst_v, float(row["v"]))
                if row["gate"] == "undo":
                    undo_gate, undo_w = "undo", loop.motors.last_cmd[1]
        finally:
            loop.recovery.update = real_update
            thaw_t()
            thaw_l()
        check("the triple lock is undone by turning back the other way",
              undo_gate == "undo" and undo_w > 0.0, f"omega={undo_w:+.2f}")
        check("...and the undo never goes forward", worst_v <= 0.0, f"max v={worst_v:+.3f}")
        check("...and the 30 Hz reflex lets the undo run",
              not loop._fast_reflex_should_stop() if loop._undo_active else False)
        loop.unstick, loop._undo_active = real_unstick, False
        loop.limiter.reset()
        loop.motors.emergency_stop()

    # 9. the LD19's FIRST revolution is partial and must never be published. The thread
    #    attaches mid-sweep, so the bins behind the first wrap cover only the arc the
    #    sensor happened to sweep after start-up; every other bin still holds inf and
    #    would go out as lidar_max_range. Measured on a cold start: 47 of 72 beams wrong
    #    by up to 4.64 m. Nothing downstream can catch it -- the control loop only tests
    #    whether a scan is FRESH, and MapperThread writes those phantom max-range beams
    #    into the PERSISTENT map as free space that was never observed.
    def _ld19_frame(start_deg: float, end_deg: float, dist_mm: int) -> bytes:
        body = bytearray(46)
        body[0], body[1] = LD19Thread.HEADER, LD19Thread.VERLEN
        struct.pack_into("<H", body, 2, 2000)                      # speed, unused here
        struct.pack_into("<H", body, 4, int(start_deg * 100) % 36000)
        for i in range(LD19Thread.POINTS):
            struct.pack_into("<H", body, 6 + 3 * i, dist_mm)
            body[8 + 3 * i] = 200                                  # intensity
        struct.pack_into("<H", body, 42, int(end_deg * 100) % 36000)
        struct.pack_into("<H", body, 44, 0)
        return bytes(body) + bytes([ld19_crc8(bytes(body))])

    probe_slot = SensorSlot("lidar_probe", cfg.lidar_deadline, critical=False)
    ld19 = LD19Thread(probe_slot, cfg, "/dev/null", 0.0, 230400, False, False, ())
    # Attach at 180 deg: a real thread starts wherever the sensor happens to be.
    for a in range(180, 360, 15):
        ld19._handle_frame(_ld19_frame(a, a + 15, 1000))
    ld19._handle_frame(_ld19_frame(0, 15, 1000))                   # first wrap
    first_scan, _ = probe_slot.read()
    check("LD19 discards its partial first revolution", first_scan is None,
          "published a half-swept scan" if first_scan is not None else "")

    #    ...and the SECOND wrap, which is a genuine full revolution, must publish.
    for a in range(15, 360, 15):
        ld19._handle_frame(_ld19_frame(a, a + 15, 1000))
    ld19._handle_frame(_ld19_frame(0, 15, 1000))                   # second wrap
    second_scan, _ = probe_slot.read()
    ok = second_scan is not None and float(np.max(second_scan)) < cfg.lidar_max_range - 1e-6
    check("LD19 publishes the first COMPLETE revolution", ok,
          "" if ok else ("no scan" if second_scan is None
                         else f"max={float(np.max(second_scan)):.3f} m -- unswept bins leaked"))

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
    p.add_argument("--navigator-speed-authority", type=float, default=0.0,
                   metavar="FRACTION",
                   help="Fraction of v_max the local navigator may drive at on an arc "
                        "it has simulated clear, even if the policy asked for less. "
                        "0 (default) = it may only ever slow her down, so she is only "
                        "as decisive as the network -- which measures a mean of "
                        "0.21 m/s. 0.6 is a reasonable first try. The governor, ToF "
                        "reflex and rate limiter all still run below it.")
    p.add_argument("--no-navigator", action="store_true",
                   help="Disable the local navigator (arc selection + situational "
                        "commitment) and let the policy's command go straight to the "
                        "governor, exactly as before. Use it to A/B the thing it was "
                        "built to fix: turning per metre, and what fraction of ticks "
                        "end in a safety override.")
    p.add_argument("--turn-commit", type=float, default=0.0,
                   help="Seconds a turn taken while blocked is kept against a request "
                        "to turn the other way (no left/right dithering). 0 = off.")
    p.add_argument("--unstick-keep-trying", action="store_true",
                   help="After the undo tries at one place, rest a moment and try "
                        "again instead of asking to be moved (rl/unstick.py "
                        "keep_trying; Harsh 2026-10-02).")
    p.add_argument("--no-unstick", action="store_true",
                   help="Do not undo the last move when the ToF reflex, its turn "
                        "freeze and the rear gate all refuse at once (rl/unstick.py). "
                        "The 30 Hz reflex still applies the tick's rule.")
    p.add_argument("--no-way-back", action="store_true",
                   help="Do not drive back to where the run started when it ends. "
                        "By default a run with the map and planner on keeps going "
                        "after --duration, exactly as it explored, with the start "
                        "as its target (rl/way_back.py). Ctrl+C / SIGINT always "
                        "stops her where she is.")
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
    p.add_argument("--blackbox-seconds", type=float, default=30.0,
                   help="How much history the black box keeps. It is a RING BUFFER: a "
                        "run longer than this loses its beginning, which is where a run "
                        "that started badly put its evidence. Set it to the run length.")
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
    # duty 80 -- below this the motors hum and do not turn."
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
    p.add_argument("--motor-min-duty", type=int, default=80,
                   help="Stiction floor: below this the motors hum and do not turn. "
                        "80 is measured on this robot, forward AND reverse. 0 DISABLES "
                        "the floor and the creep cutoff -- the motors buzz instead of "
                        "moving.")
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
    p.add_argument("--motor-turn-min-omega", type=float, default=0.15,
                   help="Smallest rotation [rad/s] that is executed at all. Below it a "
                        "heading nudge is dropped rather than lurching at the pivot "
                        "floor. Replaces the straight-line creep cutoff for pivots, "
                        "which was silently turning every rotation under 0.425 rad/s "
                        "into a full stop -- 22%% of all ticks logged on 2026-09-03.")
    p.add_argument("--lidar-min-range", type=float, default=0.0,
                   help="Reject LiDAR returns closer than this [m] as the robot's own "
                        "body. 0 uses robot_radius, which is the geometric answer: a "
                        "centre-mounted LiDAR cannot see past the chassis it stands on.")
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
    p.add_argument("--motor-pulse-burst", type=int, default=3,
                   help="Ticks per creep pulse (same average speed). 1 = the old "
                        "one-tick pulse, which started the motors ~3x as often.")
    p.add_argument("--motor-kick-rest", type=float, default=0.3,
                   help="Seconds stopped before the standstill kick fires again. "
                        "0 = kick after every stop, however short (the old, noisy way).")
    p.add_argument("--motor-plug-guard", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Coast one tick instead of reversing a wheel under power.")
    p.add_argument("--motor-steady-pivots", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Turns in place run steadily at the pivot floor instead of "
                        "being pulsed on and off (Harsh 2026-10-04: smooth, not harsh).")
    p.add_argument("--motor-start-debounce", type=int, default=1,
                   help="From a standstill, start only after motion has been asked "
                        "for this many ticks running (stops are never delayed).")
    p.add_argument("--motor-pivot-hold", type=int, default=3,
                   help="With steady pivots: ticks a started turn holds through a "
                        "same-direction dip under the floor (never through a stop).")
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
    cfg.blackbox_seconds = max(1.0, float(args.blackbox_seconds))
    cfg.lidar_min_valid_range = max(0.0, float(args.lidar_min_range))
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
            turn_min_omega=args.motor_turn_min_omega,
            turn_kick_duty=args.motor_turn_kick_duty,
            turn_kick_cooldown_s=args.motor_turn_kick_cooldown,
            pulse_creep_duty=args.motor_pulse_duty,
            pulse_burst_ticks=args.motor_pulse_burst,
            kick_rest_s=args.motor_kick_rest,
            plug_guard=args.motor_plug_guard,
            steady_pivots=args.motor_steady_pivots,
            pivot_hold_ticks=args.motor_pivot_hold,
            start_debounce_ticks=args.motor_start_debounce,
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
    motors = build_motors(cfg, args)

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
        loop.way_back_enabled = not args.no_way_back
        loop.unstick_enabled = not args.no_unstick
        loop.turn_commit_s = max(0.0, float(getattr(args, "turn_commit", 0.0)))
        loop.navigator_enabled = not args.no_navigator
        cfg.nav_speed_authority = float(args.navigator_speed_authority)
        loop.recovery_enabled = not args.no_recovery
        loop.cam_threads = [t for t in threads if isinstance(t, CameraThread)]
        loop._threads = threads          # for the LiDAR rejection counter

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



