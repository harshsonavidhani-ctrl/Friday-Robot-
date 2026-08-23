# Friday

A Raspberry Pi companion robot. See `CLAUDE.md` for the full brief — read
**Part 1 (system safety)** before touching anything on this machine.

## Layout

```
CLAUDE.md        the brief; safety rules, working agreement, hardware, targets
personality.md   Friday's character — the brain's system prompt
config.py        SINGLE source of truth: keys, devices, thresholds, models
main.py          the orchestrator — wake -> listen -> think -> speak -> act

mcp_server/      friday-robot: the ONE doorway to the hardware
  tools/         one file per capability; dropping a file in here registers it
  registry.py    the @tool decorator (errors, decision log, threading)
  safety.py      the movement interrupt — in code, not in a prompt
  hardware.py    the single owner of every device

wake_word.py     the trigger (Enter, or the word "friday")
stt.py           faster-whisper + webrtcvad endpointing
brain.py         Claude + the MCP tool loop
tts.py           Piper through the USB soundcard
face_control.py  drives friday_face.py via a state file
friday_face.py   the 20-emotion face on the 7" panel
motor.py         L298N over an Arduino, with a firmware watchdog
sensors.py       LiDAR + 8x8 ToF, on background pollers
vision.py        both cameras, stitched into one image

dashboard/       the phone-facing web UI (live log, cameras, face, drive)
firmware/        the Arduino sketch
scripts/         hardware survey, tool tests, LiDAR calibration
systemd/         Friday's own units, kept under git as well as installed
```

## Run

```bash
./venv/bin/python main.py --check          # what is ready
./venv/bin/python main.py                  # the voice loop
./venv/bin/python main.py --say "hello"    # skip the mic, type it
```

Friday's own services (all **user** units — nothing in `/etc`):

```bash
systemctl --user status friday-mcp friday-face friday-dashboard
```

## Adding a new MCP tool — under 5 minutes

Every physical capability is an MCP tool, and adding one is **one decorated
function in one file**. There is no registration list to update and no server
file to edit: every module under `mcp_server/tools/` is imported at start-up,
so the file existing *is* the registration.

**1. Write the file.** `mcp_server/tools/arm.py`:

```python
"""Arm tools."""

from mcp_server.registry import tool
from mcp_server import hardware


@tool
def wave_arm(times: int = 2) -> str:
    """Wave Friday's arm. Use when greeting someone or saying goodbye.

    Args:
        times: how many waves, 1 to 5.
    """
    times = max(1, min(int(times), 5))
    hardware.arm().wave(times)          # your hardware, via hardware.py
    return f"waved {times} times"
```

**2. Restart the server.**

```bash
systemctl --user restart friday-mcp.service
./venv/bin/python -m mcp_server --list      # your tool is there
```

That's it. The decorator already gave you:

- **Error containment** — any exception becomes a readable string. Your tool
  cannot crash the server, and a broken arm cannot cost Friday its motors.
- **The decision log** — name, arguments, result, duration, timestamp, written
  whether it succeeded or failed. Read it with
  `./venv/bin/python -m mcp_server.decision_log`.
- **A worker thread** — blocking hardware calls stay off the event loop, which
  is what lets `stop()` interrupt a move that is already running.

### The rules

- **The docstring IS the prompt.** The model picks tools by reading the first
  line and the `Args:` section. Write them for the model: say *when to use it*,
  not just what it does.
- **Type-hint every argument.** The schema is generated from the signature.
- **Return a plain string** a person could read out loud. For an image, return
  `ToolResult(Image(...), "what happened")` — see `tools/perception.py`.
- **No hard-coded values.** Pins, device names, indices and thresholds live in
  `config.py`. If your tool needs a number, add it there.
- **Get hardware from `hardware.py`**, never by opening a port yourself. One
  process, one connection per device.
- **Anything that moves goes through `safety.guarded_move()`.** Never call
  `motor.py` from a tool. That is the rule the whole design rests on.

## The safety interrupt

The limits live in **code, in the MCP server** — not in a prompt. A prompt can
be misread or argued with; this cannot:

- forward clearance is checked **before** a move and again every 100 ms **during**
  it; inside `SAFETY_STOP_MM` the motors stop and the reason is returned
- a clearance of `null` means *no sensor could see* and is **never** treated as
  clear — a blind robot does not drive forward
- durations are capped; `stop()` sets an abort flag before touching serial, so
  it interrupts a move rather than queueing behind it
- a move that was cut short **says so**, so the brain can react honestly

This applies to the dashboard's manual drive too — it calls the same tools.
There is deliberately no override.

## Honest about what it cannot measure

- **No wheel encoders.** `move_distance` and `turn_degrees` are *timed*
  conversions from `state/calibration.json`. Every result quotes its ±20%.
- **No battery gauge.** `get_status` reports the battery as unknown rather than
  inventing a number.
- **LiDAR bearings are uncalibrated** until `LIDAR_FORWARD_DEG` is measured, so
  `scan_environment` reports left/right/backward as `null`. Forward still works
  — it trusts the bumper ToF. Fix it in 30 seconds:
  `./venv/bin/python scripts/calibrate_lidar.py`

## Hardware notes that cost time to learn

- **Audio goes through pipewire, not ALSA `plughw`.** pipewire holds the USB
  card open exclusively while the desktop runs, so `aplay`/`arecord` fail with
  *Device or resource busy*. Devices are pipewire node names.
- **Never cache a camera path.** The second camera drops off the USB bus and
  returns with a different descriptor, renaming its `by-id` path.
- **One reader per camera.** The dashboard's stream and an MCP capture cannot
  both hold a webcam; `vision.py` names the process holding it.
- **The motor Arduino is the FT232R** (serial `A5069RR4`) — an Uno clone. It is
  pinned by serial number, not VID:PID, because `0403:6001` is a generic chip.
- **Models load once.** Piper costs ~10 s and faster-whisper more; both are
  preloaded at start-up. Loading per utterance was the original 17 s voice bug.

## Build order

Spine first, then organs. One module per session: build it, test it on the real
robot, commit it, stop. Nothing is committed untested — and after every change,
SSH, WiFi and login are verified still working.
