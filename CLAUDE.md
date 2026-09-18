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
