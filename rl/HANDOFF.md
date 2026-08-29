# Handoff: get this robot navigating autonomously

You are picking up a differential-drive exploration robot on a Raspberry Pi 4B. The
software is written and tested; hardware bring-up is unfinished. Your job is to
diagnose the remaining hardware/driver problems and get it driving itself.

Everything below marked **measured** was observed on this actual robot. Everything
marked **unverified** is a guess that has not been checked — treat those with suspicion.

---

## 1. What exists

**`train.py`** (on the dev machine, not here) trained a PPO policy in a 2D Gymnasium
simulator: 72-beam LiDAR + 72 frontier rays + last action = 146-dim observation, output
`(v, omega)` in SI units. Exported to `exploration_model.onnx`. Final training run:
collision rate 0.07, success 0.43, mean clearance 0.99 m. It is competent, not polished.

**`pi_deploy.py`** (~2200 lines, here) is the runtime. Command chain, each stage able
only to *reduce* authority:

```
policy(obs) -> LidarGovernor -> VisualGate -> ToFReflex -> RateLimiter -> motors
                                                              |
                              Arduino firmware watchdog (independent, in hardware)
```

**`tests/run_all.py`** — 6 offline suites, 59 assertions, no hardware needed.
`pi_deploy.py --simulate --selftest` — 8 live safety assertions.
**Both pass right now. Keep them passing.**

---

## 2. Hardware (measured)

| Part | Detail |
|---|---|
| LiDAR | LDROBOT **STL-19P** — LD19 protocol, **230400 baud**, Silicon Labs CP2102 |
| Motors | **L298N + Arduino Uno** (FTDI FT232R) running `friday_motors.ino` |
| ToF | VL53L5CX 8x8, I2C |
| Cameras | 2x USB — **not** at `/dev/video0`; real indices unknown |

**Motor protocol** (ASCII, newline-terminated, 115200):
`P` → `FRIDAY-MOTORS...` · `W<ms>` → `OK` · `M<left> <right>` → `OK` (duties −255..255) ·
`S` → `OK` · empty line → keepalive, no reply.

**LD19 frame** (47 bytes): `0x54 0x2C | speed:u16 | start_angle:u16 |
12x{dist:u16 mm, intensity:u8} | end_angle:u16 | timestamp:u16 | crc8`.
Angles in 0.01°, CRC-8 poly `0x4D`. **Verified against the real device.**

**Measured on this robot:**
- Stiction floor: **duty 72** — below this the motors hum and do not turn.
- Pivots need more than straight driving (all four wheels scrub sideways).
  Currently 1.4x, i.e. 100. Not independently measured — verify it.
- Control loop timing: mean 5.98 ms, p95 16.15 ms, max 36.88 ms against a 100 ms
  budget, `overruns=0`. Comfortable.

---

## 3. Blockers, in priority order

### B1 — LiDAR stops publishing after 30–90 s  ← THE blocker

Symptom: `sensor lidar updates=90` then `updates=29` on a later run, then the slot goes
stale and the control loop correctly holds the robot at `v=0` forever.

Error seen: `SerialException: device reports readiness to read but returned no data
(device disconnected or multiple access on port?)` from `self._ser.read(512)`.

**The owner states the device does NOT physically disconnect and is on a dedicated
powered USB hub.** That points at the driver, not the hardware — but this has not been
confirmed with `dmesg`, and nobody has actually looked. **Do that first:**

```bash
dmesg -w | tee ~/usb.log &        # leave running
python3 pi_deploy.py --no-tof --no-cameras --duration 120
# when it dies:
dmesg | tail -40
vcgencmd get_throttled            # 0x0 = healthy
cat /sys/module/usbcore/parameters/autosuspend
```

Interpretation:
- `USB disconnect` / `error -71` / `-110` → hardware after all; power or cable.
- `get_throttled != 0x0` → under-voltage.
- **Nothing in dmesg** → the driver. Look at `LD19Thread._poll` / `_handle_frame`.

Recent mitigations already in place (they may be enough, or may be masking it):
`exclusive=True` on open; transient read errors recover with a 20 ms flush rather than
a full reopen; 20 soft retries before escalating; 10 reopens with growing backoff.
Check the counters — `_read_errors`, `_soft_retries`, `_reconnects` — to see which path
is actually firing.

Also rule out the boring cause: a stale process holding the port.
`pkill -f pi_deploy.py; sudo fuser -k /dev/ttyUSB*`

### B2 — ToF not up (I2C disabled)

`/dev/i2c-1` does not exist. Needed for reflex braking, which is the layer that in this
robot's own black box caught an obstacle at 0.213 m while the LiDAR read 2.27 m clear.

```bash
sudo raspi-config nonint do_i2c 0 && sudo reboot
ls /dev/i2c-*        # expect /dev/i2c-1
i2cdetect -y 1       # VL53L5CX expected at 0x29
```

If `0x29` never appears, it is wiring or 3.3 V power. Also confirm the Python package
name matches the breakout — `VL53L5CXThread._setup` imports `vl53l5cx_ctypes`, and
vendors ship different libraries.

### B3 — `wheel_speed_max = 0.85 m/s` is **unverified**

It came from the training simulator. Nobody has measured this robot. Every speed in the
system is expressed against it: governor braking, reflex stop distance, dead reckoning,
the policy's own `v_max`. If the truth is 0.3 m/s, everything is overstated ~3x.

```bash
python3 pi_deploy.py --motor friday --motor-calibrate 100 --i-am-watching \
  --motor-max-duty 0.7 --motor-min-duty 72
```

Drives straight 3 s at duty 100. Measure the distance, compute m/s, repeat at two or
three duties to see whether the relation is even linear (on a loaded brushed motor it
often is not). Then set `RobotConfig.wheel_speed_max`.

**Known consequence:** with floor 72, the slowest achievable speed is high relative to
the policy's range, and reverse (`v_min = -0.15`) may be unreachable — every reverse
command becomes a stop. The driver logs this at startup. Read that line.

### B4 — Cameras (optional)

`/dev/video0` fails to open; on a Pi that node is often an ISP device.
`v4l2-ctl --list-devices` to find real indices, then set `RobotConfig.cam_indices`.
The visual gate **fails open** when uncalibrated, so `--no-cameras` is a legitimate
configuration. Do not let this block navigation.

---

## 4. Safety invariants — do not remove these

These were built deliberately and cost real debugging. If a change requires breaking
one, say so explicitly rather than doing it quietly.

1. **Critical sensors fail CLOSED; the advisory camera layer fails OPEN.** Stale LiDAR
   or ToF stops the robot — a missing sensor is treated as an obstacle, never as
   "clear". An *uncalibrated* visual gate disables itself, because an uncalibrated
   heuristic that vetoes all motion just looks like a broken robot.
2. **The ToF reflex is a hard veto**, on its own thread, with a direct line to
   `MotorDriver.emergency_stop()`. It must not wait for the policy tick.
3. **The FRIDAY-MOTORS handshake gates all driving.** A board that does not identify is
   never sent a duty.
4. **Duty is capped at `max_duty`**, including the kickstart boost and the stiction
   floor. Clamp on the integer after rounding — clamping the float let `127.5` round to
   `128` and breach the cap.
5. **The creep cutoff.** A commanded speed far below the stiction floor becomes STOP,
   not floor speed. Without it the governor's proportional braking silently becomes a
   lurch at floor speed as the robot nears a wall. The cutoff is measured against the
   *straight* floor, never the higher pivot floor — scaling it with the pivot floor
   makes ordinary turns fall under the cutoff and the robot cannot turn at all.
6. **`--motor-test` / `--motor-sweep` / `--motor-calibrate` require `--i-am-watching`.**
7. **The Arduino watchdog is the last line.** If this process dies, the firmware stops
   the motors on its own. Keep the control loop sending commands faster than the window.

---

## 5. How to verify any change

```bash
python3 tests/run_all.py                      # 6 suites, 59 assertions, offline
python3 pi_deploy.py --simulate --selftest    # 8 live safety assertions
```

Both must pass. If you fix a driver bug, add a test for it — every suite in `tests/`
exists because something actually broke.

---

## 6. Definition of done

```bash
# 1. LiDAR alone, 120 s, stale=0
python3 pi_deploy.py --no-tof --no-cameras --duration 120

# 2. + ToF; wave a hand ~20 cm in front, expect REFLEX_STOP
python3 pi_deploy.py --no-cameras --duration 30

# 3. Autonomous, clear floor, hand on the keyboard
python3 pi_deploy.py --motor friday --i-am-watching \
  --motor-max-duty 0.5 --motor-min-duty 72 --no-cameras --duration 30
```

**Expect the first run to look bad.** The policy assumes continuous speed control from
−0.15 to 0.5 m/s; this drivetrain has a stiction floor and effectively no reverse, so
motion will be stop-start rather than smooth. That is the safety logic working, not a
bug. Log it, do not "fix" it by removing the creep cutoff.

`blackbox.csv` holds the last 30 s of every run — `t, lidar_min, tof_min, visual,
v_policy, omega_policy, v_out, omega_out, gate, loop_ms`. Read it before theorising.
