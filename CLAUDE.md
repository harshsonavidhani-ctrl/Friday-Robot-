# Friday — project brief for Claude Code

You (Claude Code) are building and maintaining the software for **Friday**, a
Raspberry Pi-based intelligent companion robot (Jarvis/Alexa-style), built by
Harsh. Read this file fully every session before doing anything.

---

# ⛔ PART 1 — SYSTEM SAFETY. READ THIS FIRST. NON-NEGOTIABLE.

## What actually happened before (this is not hypothetical)

A previous Claude Code session was asked to make Friday's programs start
reliably at boot. While doing that, system-level services were changed. The
result:

- `dbus.service` **failed to start at boot**, and kept failing on every retry.
- Because dbus was dead, everything that depends on it died too:
  `NetworkManager` (DEPEND failed) → **no WiFi**, `wpa_supplicant` failed,
  `avahi-daemon` failed, and `systemd-logind` failed.
- With `systemd-logind` dead, **login became impossible**. The correct password
  was rejected with "Login incorrect", because no session could be created.
- No WiFi meant **no SSH**, so there was no remote way in either.
- The Pi could not be repaired from itself. Recovery required physically
  **prying open a glued 3D-printed chassis**, pulling the SD card, editing
  `cmdline.txt` on another computer to boot into a raw root shell, and
  hand-repairing the system.
- Three days were lost. Most of the project's code could not be recovered.

**One "make it start at boot" request destroyed the whole machine.** Understand
this before you touch anything.

## The rule that would have prevented all of it

> **You may modify Friday's own project folder and create systemd units for
> Friday's own programs. Nothing else. Ever. Without exception.**
>
> **If a task seems to require anything outside that, STOP and ASK.**

## NEVER touch these — no exceptions, no "just this once"

Do not create, edit, move, delete, disable, mask, stop, restart, reinstall, or
"fix" any of the following:

- **D-Bus** — `dbus`, `dbus.service`, `dbus.socket`, `/etc/dbus-1/`,
  `/usr/share/dbus-1/`, any file under them. This is what killed the machine.
- **Login/session** — `systemd-logind`, `polkit`, `/etc/pam.d/`, `/etc/passwd`,
  `/etc/shadow`, `/etc/group`, `/etc/sudoers`, user accounts, passwords, groups.
- **Networking** — `NetworkManager`, `wpa_supplicant`, `systemd-networkd`,
  `dhcpcd`, `/etc/network/`, `/etc/wpa_supplicant/`, `/etc/resolv.conf`,
  firewall rules, `iptables`, `ufw`.
- **Remote access** — `ssh`, `sshd`, `/etc/ssh/`, its port, its config, its keys.
  SSH is the ONLY remote way into this Pi. If it breaks, the chassis has to be
  pried open again.
- **Boot** — `/boot/`, `/boot/firmware/`, `cmdline.txt`, `config.txt`,
  `systemctl set-default`, boot targets, the display manager, `initramfs`,
  the bootloader, EEPROM.
- **Core system** — `udev` rules, `systemd` itself, kernel modules,
  `/etc/fstab`, `/etc/systemd/system.conf`, timers or units you did not create.

## Also never

- `systemctl disable` / `mask` / `stop` / `edit` **any unit you did not create
  yourself in this project**. Not to test something. Not to fix a dependency.
- `apt install` / `remove` / `purge` / `reinstall` **any system package** —
  especially anything touching dbus, systemd, networking, login, audio routing,
  X/Wayland, or the desktop. Python packages inside Friday's venv are fine.
- Edit anything in `/etc/` that is not a systemd unit for Friday's own programs.
- "Fix" a boot ordering or dependency problem by editing a core system unit.
  If Friday's service won't start, the bug is in FRIDAY'S unit or code — fix
  that, never the system's.
- Change file ownership or permissions outside Friday's project folder.
- Run destructive commands (`rm -rf`, `chmod -R`, `chown -R`, `dd`, `mkfs`)
  anywhere outside Friday's project folder.
- Assume something is safe because it "should just work" or "is standard
  practice". The last failure looked routine too.

## What you MAY do

- Create, edit, and manage **systemd units for Friday's own programs only**
  (e.g. `friday-brain.service`, `friday-face.service`, `friday-dashboard.service`).
- In those units, **depend on** system targets — never modify them.
  `After=network-online.target` and `Wants=` are fine. Touching NetworkManager
  is not.
- Install Python packages into Friday's own virtualenv.
- Read anything. Reading is always safe. Diagnosing is always safe.
  **Changing is what's restricted.**

## When you hit a wall

If a task appears to require a system-level change, do this — in this order:

1. **STOP.** Do not proceed.
2. Explain to me: what you want to change, exactly why, what could break,
   and how it would be undone.
3. **Wait for my explicit approval.** Silence is not approval. "It's probably
   fine" is not approval. A previous refusal is not permission to retry later.
4. If I say no, find a way that stays inside Friday's project folder, or tell
   me the task can't be done safely.

## Recovery must always remain possible

After **every** change, this must still be true — verify it:

- SSH still works (I can still get in remotely).
- WiFi still connects.
- The Pi still boots to a working login.
- The last known-good state is committed in git and can be restored.

If any change could put ANY of these at risk: **don't do it. Ask.**

**If you are ever unsure whether something is safe: it isn't. Ask.**
A blocked task costs a few minutes. A dead Pi costs three days and a broken
chassis. That trade is never worth it.

---

# PART 2 — How we work

- **Spine first, then organs.** Get a thin end-to-end pipeline working, then
  thicken it module by module. Never build one giant feature in isolation.
- **One module per session.** Do the single task you're asked, test it, stop.
  Never "build everything at once."
- **Read before you write.** Inspect existing code and match its style. Verify
  hardware on the real system (`ls /dev/video*`, `i2cdetect -y 1`, `lsusb`,
  `arecord -l`, `aplay -l`) — never guess indices, pins, ports, or endpoints.
- **Config-driven.** All keys, camera indices, device names, thresholds, model
  names live in ONE `config.py` (or `config.yaml`). No scattered hard-coded values.
- **Modular + one orchestrator.** Each capability in its own file; a single main
  loop wires them. A change in one module must not break others.
- **Git is mandatory, from the very first step.** Commit every working state.
  Push to a remote (GitHub) so code survives even if the SD card dies — the last
  rebuild lost nearly everything because there was no remote copy.
- **Test on the real robot after each step, then commit.** Never stack untested
  changes.
- **Ask before risky changes** (motors, power, anything safety-related) — and
  before anything in Part 1.

## Cost safety (avoid surprise bills)
- **Building vs running are separate.** While BUILDING Friday, Claude Code runs
  on my Pro/Max subscription — so during build sessions `ANTHROPIC_API_KEY` must
  be UNSET (otherwise Claude Code silently bills my API instead of the
  subscription). While RUNNING Friday, the brain uses the API key.
- **Cap parallelism.** Never spawn large numbers of subagents; keep parallel
  agents to a small limit. Never leave Claude Code running unattended in a loop.
- **Cheapest model that fits the task.** Haiku (`claude-haiku-4-5-20251001`) for
  simple work; Sonnet 5 (`claude-sonnet-5`) only for real reasoning or new code.
- **Small context + caching.** Send only the relevant file(s) + the exact error,
  never the whole repo. Use prompt caching for repeated context.
- **Friday's self-improving features must respect a daily spend cap** and log
  rough token cost per call.

---

# PART 3 — The robot

## Hardware (all physically mounted; verify on the real system before use)
- Raspberry Pi (4B/5), USB 3.0 hub
- Up camera (USB, above screen) — forward vision
- Down camera (USB, above bumper) — floor / cliff / optical-flow
- LiDAR: LDROBOT D500 (USB, top dome) — 360° scan
- Depth: DFRobot 8x8 Matrix ToF (front bumper, I2C/UART)
- Mic: USB shotgun (top). Speaker: via USB soundcard.
- Drive: L298N, 4-motor differential drive
- 7" display (front) — Friday's face
- Chassis: custom 3D-printed, **glued shut**. Opening it risks destroying it.
  This is why the Pi must never become unbootable.

## Current software state — READ THIS
The SD card was rebuilt from scratch after the failure described in Part 1.
**Assume nothing is installed and nothing is configured** unless you verify it.

What I have to start from:
- `friday_face.py` — pygame face module, 20 emotions via `set_emotion(name)`:
  neutral happy sad angry surprised listening thinking speaking confused dizzy
  success excited wink love curious proud bored sleepy sleeping scared.
  Emotion *reactions* (when to show which) are NOT wired yet.
- `robot_server.py` — recovered from the old system. Read it before reusing;
  treat it as a reference, not as known-good.
- `claude.md` — Friday's personality/judgement (use as the brain's system prompt).

Everything else — brain, voice pipeline, dashboard, MCP tools, motor control —
must be built fresh, following the build kit steps.

## The brain
- Main model: **Claude Sonnet 5** (`claude-sonnet-5`) via the Anthropic API.
  Key in env: `ANTHROPIC_API_KEY`.
- Fast/reflex model: **Claude Haiku** (`claude-haiku-4-5-20251001`).
- Use `claude.md` as the system prompt. Keep spoken replies short (1-3 sentences).
- Enable the Anthropic **web_search tool** for factual questions, and
  **prompt caching** for the system prompt + tools.

## Voice pipeline (target)
- Wake word: **openWakeWord** with a custom `friday.onnx` model (free, no key,
  runs on the Pi CPU). NOT a record→transcribe→check loop (it misses detections).
- STT: **faster-whisper** local (base int8) + **VAD** endpointing.
- TTS: **Piper** through the USB soundcard, start speaking on the first sentence.

## Definition of done
Friday wakes instantly to "Hey Friday", listens, answers basic questions clearly
and correctly, controls motors, sees through its cameras, reacts with the right
face, remembers things, and keeps working (with basic offline fallback) without
crashing — **and the Pi still boots, logs in, and accepts SSH after every
single change.**

---

# PART 4 — What file 08 wired in (2026-09-19)

File 08 ("Full Brain Upgrade") was **wire-up only** — it connected modules that
already existed rather than building subsystems. This note exists so file 09
does not have to rediscover any of it. Steps 5 and 11 were never pasted, so
they were never built; everything else in the file is done except **music**,
which the spec's opening list names and nothing in the repo plays.

## Every brain turn ends in one object (steps 1-2)
`reply_contract.py` — the model returns `{"reply", "emotion", "memory_note"}`
after any tool calls. `brain.ask()` still returns the spoken words; the object
is `brain.last_response`. Switch: `config.BRAIN_STRUCTURED_REPLY`.
- It is a **prompt contract, not `output_config.format`** — structured outputs
  400 when citations are present, and web search answers carry citations.
- The brain runs on **Haiku only** (`config.BRAIN_MODEL`), a budget decision.
  Anything Haiku cannot do must be gated to `MODEL_MAIN`, not assumed.
- `emotion` comes from `reply_contract.EMOTION_TRIGGERS` (15 content emotions).
  The parser refuses the pipeline-owned five, so the model cannot claim to be
  listening, speaking, thinking, sleepy or sleeping.

## The face is arbitrated, never set directly (step 2)
`face_director.py` owns `state/emotion.txt`. Four bands: SAFETY > EMOTION >
PIPELINE > IDLE, with pipeline changes forced past a lingering emotion but
never past a live safety face. Anything writing the file from outside (the
dashboard, a script, an RL run) is adopted at IDLE.
- **Sleeping latches.** Once "sleeping" is up — told to, or drifted there —
  `rest()` and the idle drift hold it until a real state arrives (the wake word
  as "listening", an emotion, or safety). Before this, the turn that put her to
  sleep ended in `rest()` and she was showing **bored** 45 seconds later.

## Three things now act without being asked
Each on its own thread, never inside main.py's wake loop — a blocking generator
lives there and anything awaited beside it swallows a "hey Friday".
- `scene_bridge.py` (step 7) — a new object in view earns at most one short
  remark. A **wake source**, so a remark queues behind a real turn instead of
  landing on top of it. Gates: 2-tick confirm, 30-min forget, ignore person,
  600 s cooldown, someone there within 5 min.
- `idle_study.py` (step 8) — in the quiet she looks up an open question and
  saves one sentence. **Never speaks**; `situation.py` offers it to exactly one
  later turn. It may only save what a search returned.
- `physical_curiosity.py` (step 9) — the idle drive. **`config.PHYSICAL_CURIOSITY`
  is False and ships False**; `scripts/test_smoke.py` fails if that line is ever
  committed as True. It calls `explore_room()` through the MCP server and
  **writes no movement or stop-distance logic of its own**, so every move keeps
  `embodiment.check`, `guarded_move`, the duration cap and the
  did-the-turn-actually-happen test. `heard()` runs the moment the wake word
  fires, before the capture, and calls `stop()` on the live session.

## Money and wheels get a speed limit, not just a ceiling (step 10)
`DAILY_SPEND_CAP_USD` is the wall at the end of the corridor and every paying
path checks it. `rate_limit.py` + `config.RATE_LIMITS` is the speed limit, so
one tool in a loop cannot reach the wall in a minute. Web search has its own
`max_uses` per turn; voice turns, scene remarks and idle study have their own
gates. **Any new tool that costs money needs an entry in `RATE_LIMITS` or a
documented gate of its own.** It fails OPEN on a corrupt counter file and is
deliberately not a safety mechanism — safety lives in `mcp_server/safety.py`.

## Anything that makes sound goes through `tts.speak()`
`tts.speak()` writes `state/speaking.json` wherever it runs, and the wake
detector drops audio while that mark says her voice is in the room. Without it
she answers herself: exploration narration, `say_what_you_see` and the RL
scripts all speak from **other processes**, where main.py's pause does not
reach, and the turn log has her acting on her own narration as a command.
A new way of making noise that bypasses `tts.speak()` re-opens that hole.

## Do not edit these to make something else work
- **The committed exploration stack** — `exploration_mode.py`,
  `mcp_server/tools/explore.py`, `slam/`, `rl/`. Harsh, 2026-09-19: "the
  exploration mode thats committed will not be like editted." Call it, don't
  change it.
- `mcp_server/safety.py`, `reflex.py`, `devguard.py` and this file are in
  devguard's PROTECTED list: Friday cannot edit them through `dev_task`.

---

# PART 5 — What file 14 added (2026-09-21)

File 14 is NAVIGATION: return-to-home and target navigation. It is **additive
only**. New files: `path_planner.py`, `nav_follower.py`, `nav_state_machine.py`,
`landmarks.py`, `mcp_server/tools/nav_tools.py`, `dashboard/nav_view.py`, and
their tests. The only pre-existing files it touched are the dashboard's
(`dashboard/server.py` one GET route, `static/index.html`, `app.js`,
`style.css`), because step 11 says extend the dashboard rather than build a
second one.

## What it wired in

**One engine, four entry points.** `NavController._drive_to(goal, label, mode)`
is the whole of navigation. RETURN_TO_HOME and TARGET_NAV are the same code
and differ only in what triggers entry and what happens on arrival — if you
find yourself writing a second copy of it for a third destination, you have
taken a wrong turn.

    relocalise ──> path_planner ──> nav_follower ──> arrival
     (refuses)     despeckle          look-ahead      held, not
                   inflate            heading error   sampled
                   A* + gradient      ONE guarded
                   smooth             move per tick

- `path_planner.py` — steps 1-3. `preprocess()` despeckles (isolated 1-2 cell
  specks become free; inflating one makes a 60 cm disc of phantom wall out of
  25 cm² of nothing) then calls `slam/plan.py::passable_mask` for the
  inflation, which is NOT reimplemented. `astar()` is 8-connected with an
  octile heuristic and a soft clearance gradient near walls — a preference,
  never a veto, so a tight doorway is still taken when it is the only way.
  `smooth()` shortcuts the staircase by line of sight. Its LOS check enforces
  the diagonal gate as well as sampling, because a 45° segment rounds onto
  cells along its own diagonal and never onto the two it squeezes between.
- `nav_follower.py` — steps 4-5. Pure-pursuit geometry; the actuation is
  pivot-then-step because the guarded interface has no arc primitive. Every
  tick re-measures the pose from the scan matcher — there are no encoders, so
  a tick costs a LiDAR sweep and runs at about 1 Hz, not 20. `RateModel`
  learns cm/s and deg/s from measured pose because calibration reads 0.
  Arrival needs tolerance AND dwell AND no commanded motion, and relaxes to
  `GOTO_ARRIVE_M` after 25 s rather than hunting.
- `nav_state_machine.py` — steps 6, 9, 10. The six states, the home timer,
  the re-plan loop, `RuntimeObstacles`, and pause/resume.
- `landmarks.py` — step 7. `AreaWatch` reports a CONTIGUOUS new region ≥ 4 m²;
  scattered edge-trimming that adds up to the same area is not a room.
- `mcp_server/tools/nav_tools.py` — step 8. `navigate_to`, `return_home`,
  `nav_status`, plus the round-trip three: `set_round_trip`,
  `explore_and_return`, `come_back`. `go_to` is untouched and still the older,
  longer-serving path.
- `dashboard/nav_view.py` + `GET /api/nav` (read) and `POST /api/nav`
  (round-trip toggle, go, come back, stop) — step 11, plus the dashboard
  trigger Harsh asked for afterwards.

**Return-to-Home.** `HomeTimer` counts wall clock, not ticks, because a run
stalled against a wall is the one that most needs it to fire. Home is a row in
the `places` table called "home" — an ordinary landmark, so `list_places` and
`go_to` already see it. Set it with `remember_place("home")`.

**Target Navigation.** The brain parses the voice command into a name and an
optional action, calls `navigate_to`, and PERFORMS THE ACTION ITSELF on the
result's `action_pending`. The server cannot call the brain, and a tool that
spoke would bypass `reply_contract` and the face arbiter.

**Round trip — "go and come back" (2026-09-21, after step 14).** Harsh:
"friday go explore the home it explores and like come to the position where it
started and like then i commanded it go to the kitchen then it should go to the
kitchen and return to where it started."

⚠ "WHERE IT STARTED" IS NOT HOME. `home` is a place somebody deliberately
named and it stays put for weeks. The start of a round trip is wherever she
happened to be standing when the command arrived, and it is valid for that one
errand only — so it is captured per run from a RELOCALISED pose and never
written to the `places` table.

Two shapes, because the outbound legs are different:

  * TARGET NAV is driven here, in one call, so the return happens inline —
    `go_to_landmark(..., return_to_start=True)`, then `_return_to_start`.
  * EXPLORATION is a SEPARATE PROCESS holding the hardware lease, so the
    return is ARMED (`arm_return_to_start` writes `state/nav_round_trip.json`)
    and a watcher task in the MCP server fires it when `autonomy._running()`
    goes False. If the server restarts mid-run the task dies and the arming
    survives on disk — `nav_status` reports it and `come_back()` fires it by
    hand. That is a visible loose end rather than a silent one.

The mode is persisted in `state/nav_mode.json` so the voice toggle and the
dashboard checkbox are the same switch. `return_to_start=None` means "whatever
the mode says".

⚠ COMING BACK IS A SECOND DRIVE WITH ITS OWN REPORT, and that matters for what
she says afterwards: "I got to the kitchen but could not get back" is true and
useful, and merging the two legs into one result would report the whole errand
as a failure and lose the part that worked. An arrival monitor passed by a
caller is rebuilt for the start pose — reusing the outbound one would declare
her "back" the moment she reached the kitchen.

**Landmark labeling.** `LandmarkNamer.consider(grid, pose)` asks only when
somewhere big and new has appeared and she is not already inside a known
place. `ask` is INJECTED — no `ask`, no question and nothing saved. An answer
that is not a name ("no", "I don't know", a sentence) saves nothing, and a
name that already exists is confirmed before it is moved, because
`places.remember_here` moves silently by design.

**Pause/resume.** A target-nav command mid-exploration calls
`pause_exploration()` → snapshot + `stop_exploration()`, then
`resume_exploration()` → `start_exploration(minutes=what was left)`.
⚠ THE FRONTIER LIST IS NOT RESTORED AND MUST NOT BE. It is a pure function of
the occupancy grid and the pose (`slam/frontier.py::find`), and the grid
persists — so resume recomputes it, which is also more correct, since a
frontier snapshotted before the interruption may have been explored during it.
What is genuinely lost is the committed target and the blacklist; they are
captured as a hint that nothing yet consumes, and that is recorded rather than
hidden. Resuming with under `RESUME_MIN_MINUTES` left goes home instead.

⚠ `registry.tool` RETURNS THE PLAIN FUNCTION and registers a separate async
`invoke`. `autonomy.stop_exploration()` is callable directly and has no
`__wrapped__` — reaching for one raised straight into an `except` and reported
"could not stop exploration" while the run kept driving. A pause that does not
pause leaves two things steering the same robot.

## Running the tests

    ./venv/bin/python scripts/test_path_planner.py       # steps 1-3
    ./venv/bin/python scripts/test_nav_follower.py       # steps 4-5
    ./venv/bin/python scripts/test_nav_home.py           # steps 6-11
    ./venv/bin/python scripts/test_file14_acceptance.py  # step 13, end to end

No hardware, no motors. ⚠ ALL OF THEM MUST STAY SIDE-EFFECT FREE: places go to
a temporary sqlite file and `NAV_LOG` to a temporary directory. The suite
originally wrote forty imaginary journeys into `state/nav_log.jsonl`, and the
dashboard showed them as real history.

## The hard rules this phase adds

- **The exploration/mapping/pose-tracking files are a READ-ONLY dependency of
  this module.** `slam/`, `rl/`, `exploration_mode.py` and
  `mcp_server/tools/explore.py` are never modified to make navigation work —
  no exceptions, and no "just this accessor". Everything in file 14 calls
  them. Nothing in file 14 changes them. This is the same rule as PART 4's
  "call it, don't change it", restated because navigation is the module most
  tempted to break it.
- **Everything here stays inside the project folder.** No systemd unit, no
  system service, nothing in `/etc/`. Navigation never has a reason to go near
  dbus or NetworkManager, and PART 1 applies to it exactly as written.
- **No autonomous decision in this module bypasses the safety layer.**
  Re-planning, landmark selection and target navigation all move the robot the
  same way everything else does: `movement.move()` → `safety.guarded_move()`,
  which runs `embodiment.check`, the forward clearance check, the 100 ms
  re-check while moving and the duration cap. `nav_follower.py` does not
  import `motor.py` and does not call `hardware.motors()`, and
  `scripts/test_nav_follower.py` asserts that with `ast` rather than trusting
  it. **A follower that owns the wheels is a fast path, and there must never
  be one.**
- **Nothing in this module loops without a throttle.** Landmark-naming prompts
  need a big new contiguous area AND a pose outside every known place before
  they ask, the same shape of gate file 08 put on the scene bridge. A block is
  rechecked on a timer before anything re-plans, so a person walking past
  cannot cause a replanning storm. Re-planning is capped at `max_attempts`.
- **FAILSAFE halts and waits for a human.** It calls `halt()`, it does not
  retry beyond the replanning budget, and it surfaces to the dashboard. There
  is no autonomous recovery after FAILSAFE — that is what the state is for.

## Things that will bite you in this module

- **There are two maps and they are different frames.** `state/map.npz` is the
  scan-matched survey and is the frame every landmark, `go_to`, `navigate_to`
  and return-to-home works in. `state/rl_map.npz` is the RL stack's
  dead-reckoned map and is what the dashboard's Map **picture** shows. On
  2026-09-21 the survey covered 7.5 × 8.1 m and the RL map claimed 24 × 24 m
  for the same flat, with 81.5% of its "walls" being single-cell specks.
  Drawing one's coordinates on the other's picture is wrong and looks right.
- **A stored pose is never trusted.** The map persists across runs; her
  position in it does not. Every navigation run relocalises first
  (`places_tools._placed_mapper()`) and refuses if the scan match fails.
- **`state/calibration.json` still reads 0.** `move_distance()` and
  `turn_degrees()` refuse while it does, which is why `go_to` cannot currently
  drive a leg and why `nav_follower.py` commands in seconds and learns the
  rates from measured pose. Calibrating unblocks the distance tools too.
- **The microphone belongs to one process.** `main.py`/`wake_word.py` hold the
  `stt.AudioStream`; nothing under `mcp_server/` opens it — but exploration
  runs inside the MCP server. `landmarks.py` therefore takes an INJECTED
  `ask()` and never speaks or listens itself.
- **`navigate_to` does not fuzzy-match and must not learn to.** A near-miss
  string match sends the robot several metres to somewhere nobody asked for.
- **The Navigation panel has a POST now.** It was read-only until Harsh asked
  for the dashboard to trigger target nav too. The reason it was read-only has
  not gone away, so the POST answers it the way the Explore button does:
  everything goes out through the MCP server and its safety interrupt, never
  the motors, and the destination must be a place she already knows. Keep it
  that way.
