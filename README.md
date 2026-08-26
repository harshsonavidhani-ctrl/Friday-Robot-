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
friday_face.py   the 20-emotion face on the 7" panel — also hosts the apps
app_control.py   drives the screen apps and holds the alarm/timer schedule
apps/            one file per screen app: clock, timer, alarm, weather
motor.py         L298N over an Arduino, with a firmware watchdog
sensors.py       LiDAR + 8x8 ToF, on background pollers
vision.py        both cameras, stitched into one image
slam/            the room map: scan matching and the occupancy grid
  core.py        points, the likelihood field, and the scan matcher

dashboard/       the phone-facing web UI (live log, cameras, face, drive)
firmware/        the Arduino sketch
scripts/         hardware survey, tool tests, LiDAR calibration
backup/          git bundles (gitignored) — NOT an off-card backup, see below
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

## Adding a screen app — one file

The 7" panel shows Friday's face by default. An app takes it while it has
something to say and hands it straight back. Drop a file in `apps/`:

```python
import apps
from apps.base import App

@apps.register
class Battery(App):
    NAME = "battery"

    def setup(self, percent: float = 100.0):
        self.percent = percent

    def draw(self, surface):
        surface.fill((0, 0, 0))
        self.paint.text(surface, f"{round(self.percent)}%",
                        self.paint.px(0.30),
                        (self.width // 2, self.height // 2))
```

The file existing is the registration — same as `mcp_server/tools/`. Launch it
with `app_control.launch("battery", percent=61)`, or add a `@tool` in
`mcp_server/tools/screen.py` so the brain can. `App.paint` carries the face's
finish (gradient, gloss, drop shadow) so a new app matches without copying any
of it; sizes go through `paint.px(fraction)` so they hold on any panel.

### The rules apps live under

- **The face is the resting state.** An app that has said its piece sets
  `self.done` and disappears; it does not sit there.
- **A safety expression outranks any app.** `scared` and `dizzy` are drawn
  over whatever is up. The single exception is an alarm actually ringing,
  which sets `URGENT = True` — it is already the thing demanding attention.
- **Touch anywhere means stop.** The default `handle_event` dismisses.
- **Nothing is drawn from another process.** Everything goes through
  `app_control.py`, which writes state files the face polls — the same
  pattern as `face_control.py`, and for the same reason: the face process
  owns the only window on the panel.

## Alarms and timers

```bash
./venv/bin/python app_control.py alarm "8 am" school
./venv/bin/python app_control.py timer "10 minutes" pasta
./venv/bin/python app_control.py --list
./venv/bin/python app_control.py --cancel alarm1
./venv/bin/python scripts/test_apps.py --render      # PNGs, no panel needed
```

They live in `state/schedule.json` and are watched by the **face** process, so
they ring after a restart of anything else and after a power cut. Something due
while the Pi was off still rings if it is less than `APP_ALARM_GRACE_S` late.
The same rows are set from the brain (`set_alarm`, `set_timer`), from the
dashboard's **Screen** panel on the phone, and from the CLI above — so any of
the three can cancel what another one set.

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

## The map (step 9, in progress)

`slam/` is pure numpy/scipy — ROS2 was evaluated and rejected because Debian 13
has no ROS2 packages at all, so it would need a third-party apt repo or a
source build, which CLAUDE.md Part 1 forbids outright.

```bash
./venv/bin/python scripts/test_slam.py               # matcher AND map vs known answers
./venv/bin/python scripts/test_slam.py --show        # print the map it built
./venv/bin/python scripts/calibrate_lidar.py --drive # measure LIDAR_FORWARD_DEG
```

`slam/core.py` answers *how did the robot move between these two sweeps*;
`slam/grid.py` answers *what is where*, and unlike the matcher it remembers —
one log-odds array in `state/map.npz` that survives a restart and a power cut.
Folding a 350-beam sweep in costs **24 ms**, and a full-size 24 m map is
0.9 MB, so mapping at 1 Hz is a few percent of one core.

Three decisions in the map are worth knowing before trusting it:

- **Log-odds, clamped.** Not for the arithmetic — so the map can *change its
  mind*. Unclamped, a cell seen as a wall two hundred times needs two hundred
  contradicting looks to come free, so a chair moved this morning is still a
  wall tonight. Clamped, it clears in twelve sweeps; the test measures exactly
  that by taking the pillar away and counting.
- **A missing bearing carves nothing.** An absent bearing is either "nothing
  within six metres" or "a black jacket that returned no signal", and the
  LiDAR does not say which. So an open doorway beyond range stays *unknown*
  rather than being mapped as floor.
- **Unknown is never driveable.** `free_mask` needs positive evidence of empty
  space — the same rule the safety interrupt already applies to a null
  clearance, and for the same reason.

### Exploring

```bash
./venv/bin/python scripts/test_explore.py            # explore a simulated flat
./venv/bin/python scripts/test_explore.py --show     # print the map it built
```

`slam/frontier.py` finds the boundary between mapped and unmapped space;
`slam/plan.py` plans a route to it (one Dijkstra gives the true driving
distance to *every* candidate at once, so ranking a dozen is array lookups);
`slam/explore.py` is the policy that picks one. It decides — it never drives.
Execution goes through the MCP server, where the safety interrupt lives.

Frontiers are chosen by **new area per metre travelled**. Nearest-first leaves
the robot nibbling round the room it is in; biggest-first sends it across the
flat for every scrap.

On a simulated four-room flat it finds the doorways, gets into the rooms, and
maps ~2× the floor per metre driven that random headings manage.

**Known open — pose drift.** Over ~60 m the map comes out ~120% of the real
floor area. The matcher is not failing: it reports a good score and applies a
0.002 m correction, because the map is built *at* the drifted pose, so every
new sweep fits it perfectly. Drift is only observable on returning to geometry
mapped before it accumulated — which frontier exploration, fanning outward,
rarely does. Loop closure (`Mapper.try_loop_closure`) is implemented for
exactly this and correctly does nothing on an outward-only run. Refining
against a finer field was tried twice and made it worse both times; the
reasoning is written down in `slam/core.py` so it is not tried a third time.

Both of the LiDAR's own quirks are already handled in `sensors.Lidar.scan()`
and `config.py`, and both cost hours to find:

- **Friday can see herself.** Raw bearings 90-97 read a rock-steady 210 mm.
  Masked via `LIDAR_SELF_MASK`, or it is permanently the nearest thing in every
  sweep and gets mapped as an obstacle that follows the robot.
- **The forward bearing must be measured by DRIVING**, not by putting an object
  in front. The LiDAR plane is at leg height: a box on the floor is below it
  and never appears, and a person appears as two returns twelve degrees apart,
  one per leg.

## Backups — read this

**There is no git remote.** CLAUDE.md requires one and the Pi cannot create one
alone (no `gh`, no push keys, no removable storage). Until that is fixed, the
whole project exists only on this SD card. To take a copy off the Pi:

```bash
git bundle create backup/friday-$(date +%Y%m%d).bundle --all   # on the Pi
scp harsh@Harsh.local:/home/harsh/friday/backup/*.bundle .     # from a laptop
git clone friday-20260826.bundle friday                        # to restore
```

`./scripts/rollback.sh` returns to the `known-good` tag; `--mark` declares the
current (tested, committed) state good.

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
