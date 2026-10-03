# File 21 — exploration mission: hardware checklist (H1–H7)

Harsh runs these on the robot. Claude Code only prepared them, and never
reboots the Pi or touches a network setting. Everything below stays inside
`~/friday` and Friday's own user units.

All commands from `cd ~/friday`.

## Before you start

- Mark the floor at the start spot with tape (H4 measures against it).
- H1 and H2 run with the mission **off** (as shipped). H3–H7 need it **on**:

```
# turn ON: in config.py set  EXPLORATION_MISSION_SERVICE = True , then
systemctl --user restart friday-mcp friday-brain friday-dashboard
tail -f logs/exploration_mission.log        # expect: "exploration mission started", then boot lines
```

```
# turn OFF again: set it back to False, then the same restart
systemctl --user restart friday-mcp friday-brain friday-dashboard
```

The dashboard's **Exploration mission** panel shows the state, the rooms, the
trail and the return attempts. Its only button is naming a room.

---

## H1 — probe idle, then a 5-minute run with the recorder

```
./venv/bin/python -m exploration_mission.adapters --probe     # never moves, never writes
./venv/bin/python -m exploration_mission.recorder             # leave running in a 2nd terminal
# then start a plain 5-minute run: dashboard Explore -> 5 min (mission still OFF)
watch -n 5 'head -c 160 state/exploration_mission/live.json; echo; uptime'
```

**Pass when:**
- The probe prints sane values: a LiDAR sweep, the map, the env id, `conversation`.
- `live.json` updates about once a second, and its reason is mostly `matched`.
- The load stays under about 4.
- The run log shows overruns no worse than the last run:
  `grep -a "overruns\|measured inside tick" "$(python3 -c "import json;print(json.load(open('state/explore_mode.json'))['log'])")"`
- No `stale lidar` in that log.

**Send me** the file the recorder wrote: `state/exploration_mission/recordings/run-*.jsonl`.
This is the one real recording that decides how the mission builds its map.

## H2 — the home anchor's signature

```
./venv/bin/python scripts/file21_anchor_check.py capture   # at the home spot
# turn her ~90 degrees by hand, same spot
./venv/bin/python scripts/file21_anchor_check.py compare
# push her ~1 m away
./venv/bin/python scripts/file21_anchor_check.py compare
```

Read-only: it uses friday-mcp's `lidar_sweep`, never moves her, and writes
only `/tmp/friday-anchor-check.json`.

**Pass when:**
- After the turn: `MATCH`, turned about ±90°.
- After the push: `NO MATCH`.
- Already measured on the Pi on 2026-10-02, idle at the same spot: 342 of 360
  bearings, `MATCH` at +0°, 1 cm.

## H3 — one-room mission (mission ON)

"Friday, explore." When she asks "What should I call it?", just answer
"kitchen". She listens for about 10 s without the wake word, and "Friday,
save it as kitchen" works too.

Before it (once): stand her on her home spot and say "Friday, this is home"
(U4, 2026-10-03). That saves your permanent home AND the mission's start
there, on the saved map. Start missions from that spot.

**Pass when:**
- She says "Saved as kitchen." and the panel lists it.
- "Friday, what have you saved?" names it.
- Started by her home: the panel shows the room in frame `map` straight away,
  and "Friday, go to the kitchen" works after the mission (File 14 drives).
- ⚠ Started anywhere else, the room stays in the mission's own frame unless
  the end-of-mission fit places it. In the hostel room that fit refuses (two
  spots look alike, 1.22 against 1.12): correct, not a failure.

## H4 — whole house (2–4 rooms)

During it, also try (U2/U3, 2026-10-03; the wake word now stays ON for the
whole mission):
- "Friday, stop" while she drives → "Stopping. Say continue when you're
  ready." She stays put however long you wait; "Friday, continue" →
  "Continuing exploration." and she carries on.
- "Friday, go to the hall" (a room she already named this mission) →
  "Heading to hall." … "I have arrived at hall.", then she explores again.
- If she circles without finding new space for a minute while unexplored
  places are left, she drives to the nearest one herself (frontier-assist,
  `frontier_assist` in the mission log), then the RL drives again.

**Pass when:**
- She asks about every room.
- The summary names them all.
- She stops within 0.35 m of the tape mark.
- She says "I've reached where I started. Exploration saved."
- The run's log shows no rise in "overruns" from the wake word being on
  (compare with the 2026-10-02 23:18 run: 94 overruns, 60.5 ms per tick).

## H5 — something in the way back

Put a box on her way back, then block the way fully, then clear it.

**Pass when:**
- She says "I'm stuck about N meters from where I started. Please help me or
  clear the way.", with a sensible N.
- After you clear it, "Friday, come back" tries again.
- With the box (not fully blocked): the mission log shows the ladder's rungs
  in order (U1, 2026-10-03: File 14's planner, the held run driving):
  wait 5 s → re-plan round it → a full turn ("Checking where I am.") → back
  up → retrace her path ("Retracing my path.") → nearest free spot ("Getting
  as close as I can."), and she gets home round the box.
- Fully blocked: HELP every 30 s, 20 rounds, then "I'll wait here. Say
  Friday, come back, to try again."

## H6 — power cut, and a reboot after a finished mission

Pull the power mid-exploration and power back on. Separately, reboot after a
finished mission.

**Pass when:**
- At boot she says she was exploring when she lost power, and she does not
  move until "continue" or "go back".
- "What have you saved?" still lists every room.
- After the clean reboot: no interruption line, rooms intact.

## H7 — offline (no internet, Wi-Fi left ON)

⚠ Changed from the spec: do NOT switch the router or hotspot off, because
that drops SSH too. Unplug the router's **internet** cable, or turn off the
phone's **mobile data** while its hotspot stays on. Then do a full mission by
voice. Never change any network setting on the Pi.

**Pass when:**
- Everything works.
- `grep -iE "network|dns|connect|http" logs/exploration_mission.log` shows nothing.
- SSH stayed up the whole time.

---

## Appendix — proposed CLAUDE.md PART 12 (NOT appended; only after you approve)

```
# PART 12 — What file 21 added (2026-10-02): the exploration mission

Harsh's "15_exploration_mission.md", numbered 21 (15 is vision). Additive,
100% offline: no model, no API key, no network code. Package
exploration_mission/ (FSM, its own map, frontiers, rooms, label dialog,
offline router, area store, home anchor, return ladder, boot restore,
curiosity, voice link, dashboard panel, sim). Tests:
scripts/test_file21_step1..12.py (pytest is not installed; network blocked).

⛔ EXPLORATION_MISSION_SERVICE ships False; Harsh turns it on after his
supervised trip (docs/file21_hardware_checklist.md). curiosity.mode ships
"off" (a robot proposing trips is his call, like AGENCY_MOTION).

What it changed outside its package (all approved):
- rl/way_back.py + rl/pi_deploy.py: the publisher (live.json, 1 Hz, child
  process), the HOLD (hold.json, lapses by itself) and RETURN NOW
  (return.json). The run still drives; the mission never sends a wheel
  command.
- mcp_server/server.py: one guarded launcher block (service.start()).
- main.py: the voice hook after is_stop (mission_voice on the RAW session --
  through _FreshOrders it would start the stale-order clock early),
  MissionWatch (its questions, no wake word), the wake word on while held.
- brain.py: config.BRAIN_HIDDEN_TOOLS (the five mission_* tools).
- dashboard: Explore button -> mission (not with thinking; cameras handed
  back), GET/POST /api/mission -- POST is NAMING ONLY {area_id, name}.

Rules:
- A mission never places its start on the saved map from one sweep (sure
  and wrong at 3 of 6 real spots). Home-spot missions share the home frame;
  the whole mission map is fitted at the END, and only a sure fit moves
  areas into frame "map" (and so into File 14's landmarks).
- Never delete or rewrite state/exploration_mission/envs/ or the areas /
  missions / home_anchor tables in memory.db (sqlite backup() before any
  schema change).
- "this is home" stays the permanent home (File 10); the mission's start is
  "set home here" / "mark the start here".
- Sim runs: ./venv/bin/python -m exploration_mission.sim (prints, never
  speaks). friday_localnav_deploy.py --simulate SPEAKS unless --no-voice.
```
