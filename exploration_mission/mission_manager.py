"""The mission: what Friday is doing, ten times a second (File 21 step 7, spec 3.3).

    IDLE -> MISSION_INIT -> EXPLORE <-> ASK_LABEL          (the run drives; she asks)
    EXPLORE / ASK_LABEL -> PAUSED -> back where it paused  ("stop" / "continue")
    EXPLORE -> SUMMARIZE -> RETURN_TO_START -> ARRIVED_START -> SAVE_MISSION -> IDLE
    RETURN_TO_START -> HELP_WAIT                           (could not get back)
    any -> FAILSAFE                                        (the run is lost)

WHO DRIVES. The RL run does, exactly as it always has. This class never sends
a wheel command. Its whole influence on motion is three files the run reads:
start the run, HOLD (stand still: a question, a pause), RETURN NOW (the run
starts its own way back). Everything else here is watching, asking, saving.

WHAT IT WATCHES. The run's start tracker publishes a pose and a sweep once a
second. The mission folds the sweeps into its own map (mission_map.py), and a
worker finds frontiers (<= 1 Hz) and rooms (<= every 10 s) on it. The 10 Hz
tick never waits for that work: it reads the latest result.

FRAMES. Everything a mission computes is in the START frame (the start is
(0, 0, 0)). If the start could be placed on the saved map with certainty, an
area is converted to the saved map's frame when it is saved and becomes a
File 14 landmark at once; if not (two spots look alike), it is kept in the
mission's own frame, named all the same, and never a landmark.

⚠ THE HOLD MUST BE REFRESHED. It lapses by itself in hold_ttl_s so a crashed
mission cannot freeze her -- which also means a question that blocked the
tick would let her drive off mid-sentence. So the dialog runs as a JOB (its
own thread on the robot) and the tick keeps refreshing the hold beside it.

⚠ SHE DOES NOT SAY IT TWICE. The run announces its own return and arrival, so
with the real run the mission keeps only what the run cannot say: the summary
with the areas, the questions, the help lines.

THE WAY BACK (step 8, rth_ladder.py). The run's own way back is the first
attempt. If it does not arrive, the run has ended and File 14 can have the
wheels: the ladder climbs its rungs through the nav adapter, then HELP, then
HELP_WAIT. Arrival is checked by position (tolerance + dwell) and then looked
at once with the LiDAR against the start's signature -- a mismatch is
`verified = false`, never a failure.

Not here yet: boot restore (step 9), curiosity (step 10), the voice hook (11).
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections import deque

from exploration_mission import (area_store, breadcrumbs, frontiers, home_anchor,
                                 intents, label_dialog, mission_map, rth_ladder,
                                 segmentation, speech)

log = logging.getLogger("friday.exploration_mission")

MOVING_STATES = frozenset({"EXPLORE", "RETURN_TO_START", "TARGET_NAV"})
RUN_START_TIMEOUT_S = 120.0      # the run loads a detector before its first tick
LOST_AFTER_S = 10.0              # no published fix for this long, run still out
STOP_CONFIRM_S = 3.0             # how long to wait for "motors are zero"
RETURN_REASK_S = 10.0
STATUS_PERIOD_S = 0.4
# Exploring is never called finished before this: the map is one sweep a second
# and a single sweep is not yet evidence of free floor (slam's log-odds need a
# second look). The same 20 s rl/way_back.py waits before it trusts "exhausted".
MIN_EXPLORE_S = 20.0
STATUS_MAX_AGE_S = 5.0


# --- blocking work beside the tick ---------------------------------------------------

class Job:
    def __init__(self):
        self.done = False
        self.result = None
        self.error: Exception | None = None

    def _run(self, fn) -> None:
        try:
            self.result = fn()
        except Exception as exc:                        # noqa: BLE001
            self.error = exc
            log.exception("mission job failed")
        self.done = True


class ThreadRunner:
    """The robot: each job on its own daemon thread."""

    def start(self, fn, name: str = "mission-job") -> Job:
        job = Job()
        threading.Thread(target=job._run, args=(fn,), name=name, daemon=True).start()
        return job


class InlineRunner:
    """The tests: the job has finished by the time start() returns."""

    def start(self, fn, name: str = "") -> Job:
        job = Job()
        job._run(fn)
        return job


# --- the worker: map, frontiers, rooms -------------------------------------------------

class Analysis:
    """Owns the mission's map. `step()` is called by the worker thread on the
    robot and straight from the tick in tests (`inline=True`)."""

    def __init__(self, adapters, cfg, policy: str = "matched", inline: bool = False):
        self.a, self.cfg, self.inline = adapters, cfg, inline
        self.policy = policy
        self.fparams = frontiers.Params.from_config(cfg)
        self.sparams = segmentation.Params.from_config(cfg)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self.map = mission_map.MissionMap(self.a.slam, self.policy)
            self._fixes: deque = deque()
            self._inputs = ((0.0, 0.0, 0.0), None, [], [])
            self.latest: dict | None = None
            self.tracker = segmentation.Tracker(self.sparams.same_area_iou)
            self._t_frontier = -1e9
            self._t_segment = -1e9
            self._seg = None

    def push(self, fix: dict) -> None:
        self._fixes.append(fix)

    def set_inputs(self, pose, blacklist, crumbs, areas) -> None:
        with self._lock:
            self._inputs = (tuple(pose), blacklist, list(crumbs), list(areas))

    def step(self, now: float) -> None:
        while self._fixes:
            self.map.add(self._fixes.popleft())
        if now - self._t_frontier < 1.0 / self.cfg.explore.frontier_hz:
            return
        grid = self.map.planner_grid()
        if grid is None:
            return
        with self._lock:
            pose, blacklist, crumbs, areas = self._inputs
        self._t_frontier = now
        fres = frontiers.find(grid, pose, self.a.slam.inflate_m(), self.fparams,
                              blacklist, crumbs)
        seg_ms = None
        if (self._seg is None or now - self._t_segment >= self.cfg.segmentation.period_s
                or self._seg.labels.shape != fres.coarse.shape):
            self._t_segment = now
            self._seg = segmentation.segment(fres.coarse, self.sparams)
            self.tracker.assign(self._seg)
            seg_ms = self._seg.ms["total"]
        segmentation.mark_complete(self._seg, fres, self.sparams.complete_margin_m)
        segmentation.match_areas(self._seg, areas, self.sparams.same_area_iou)
        if seg_ms is not None:
            log.info("frontiers %.0f ms, segmentation %.0f ms (%d rooms, %d frontiers "
                     "left)", fres.ms["total"], seg_ms, len(self._seg.segments),
                     len(fres.remaining))
        self.latest = {"t": now, "frontiers": fres, "seg": self._seg,
                       "explored_m2": grid.explored_m2(), "map": self.map.stats(),
                       "frontier_ms": fres.ms["total"], "segment_ms": seg_ms}

    # -- the robot's worker thread --
    def start_thread(self, clock) -> None:
        if self.inline or (self._thread and self._thread.is_alive()):
            return
        self._stop.clear()

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    self.step(clock.monotonic())
                except Exception:                       # noqa: BLE001
                    log.exception("analysis step failed")
                self._stop.wait(0.2)

        self._thread = threading.Thread(target=loop, name="mission-analysis",
                                        daemon=True)
        self._thread.start()

    def stop_thread(self) -> None:
        self._stop.set()


# --- the mission ------------------------------------------------------------------------

class MissionManager:
    def __init__(self, adapters, store: area_store.AreaStore, cfg,
                 runner=None, inline_analysis: bool = False,
                 map_policy: str = "matched"):
        self.a, self.store, self.cfg = adapters, store, cfg
        self.runner = runner or ThreadRunner()
        self.analysis = Analysis(adapters, cfg, map_policy, inline=inline_analysis)
        self.lparams = segmentation.LabelParams.from_config(cfg)
        self.dialog = label_dialog.LabelDialog(
            adapters, store, cfg.voice.listen_s, cfg.label.reask_once,
            cfg.label.max_name_words, cfg.label.max_name_chars)
        self._cmds: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.RLock()
        self.state = "IDLE"
        self.interrupted = False            # set by boot restore (step 9)
        self.tick_ms: deque = deque(maxlen=2000)
        self._reset_mission()

    def _reset_mission(self) -> None:
        self.mission: dict | None = None
        self.env_id = "home"
        self.map_id: str | None = None
        self.T = None                       # the start's pose in the saved map, if sure
        self.trail = breadcrumbs.Trail(self.cfg.breadcrumbs.spacing_m,
                                       self.cfg.breadcrumbs.loop_cut_m)
        self.blacklist = frontiers.Blacklist(self.cfg.explore.blacklist_after_fails,
                                             self.cfg.explore.blacklist_radius_m)
        self.completion = frontiers.Completion(self.cfg.explore.done_after_empty_checks)
        self.dwell = segmentation.Dwell()
        self.pose = (0.0, 0.0, 0.0)
        self.paused_from: str | None = None
        self.pending_label: dict | None = None
        self.stop_reason: str | None = None
        self.job: Job | None = None
        self.new_area_ids: list[str] = []
        self.distance_m = 0.0
        self.explored_m2 = 0.0
        self.estop = False
        self.questions = 0
        self._stage = ""
        self._after_ask = "EXPLORE"
        self._t_state = 0.0
        self._t_explore = None
        self._t_question = None
        self._t_last_fix = 0.0
        self._t_status = -1e9
        self._t_hold = -1e9
        self._t_checkpoint = -1e9
        self._t_return = 0.0
        self._last_fix_t = None
        self._matched = False
        self._status: dict | None = None
        self._minutes = None
        self._stag = (0.0, 0.0)             # (time, explored m2) at window start
        self._areas_cache: tuple[float, list] = (-1e9, [])
        self._preset = None
        self.ladder: rth_ladder.Ladder | None = None
        self.arrival: rth_ladder.ArrivalCheck | None = None
        self._arrive: dict | None = None
        self.verified: bool | None = None
        self._t_counted = None
        self._nav_back = "EXPLORE"
        self._pause_after_ask = False
        self._last_scan: dict | None = None

    # ======================================================================
    # what the voice side calls (any thread)
    # ======================================================================
    def context(self) -> intents.Context:
        return intents.Context(
            state=self.state, localized=True, interrupted=self.interrupted,
            moving=self.state in MOVING_STATES,
            dialog="label" if self.state == "ASK_LABEL" and self._stage == "dialog"
            else None)

    def halt(self) -> None:
        # Stand still NOW, not at the next tick: the hold is written here.
        if self.mission is not None and self.state not in ("HELP_WAIT", "FAILSAFE"):
            self._hold("halt", force=True)
        self._cmds.put(("halt",))

    def finish(self) -> None:
        self._cmds.put(("finish",))

    def resume(self) -> None:
        self._cmds.put(("resume",))

    def start(self, minutes=None, fresh: bool = False) -> None:
        self._cmds.put(("start", minutes, fresh))

    def label_now(self, name: str, display: str) -> None:
        self._cmds.put(("label_now", name, display))

    def go_to(self, name: str, target: dict) -> None:
        self._cmds.put(("go_to", name, target))

    def set_home(self) -> None:
        self._cmds.put(("set_home",))

    def dialog_answer(self, text: str) -> None:
        log.info("heard during a dialog, left to the dialog: %r", text)

    def say(self, key: str, **fields) -> str:
        return speech.say(self.a, key, **fields)

    def rename(self, area: dict, new_display: str) -> bool:
        try:
            renamed = self.store.rename(area["id"], new_display)
        except area_store.StoreError as exc:
            log.info("rename refused: %s", exc)
            return False
        self.say("renamed", old=area["display_name"], new=renamed["display_name"])
        self._event("renamed", area=area["id"], to=renamed["name"])
        return True

    def forget(self, area: dict) -> None:
        self.store.forget(area["id"])
        self._event("forgot", area=area["id"], name=area["name"])

    def say_list(self) -> str:
        areas = self.store.list_areas(self.a.env.env_id())
        new = [a for a in areas if a["id"] in self.new_area_ids]
        old = [a for a in areas if a["id"] not in self.new_area_ids]
        text, mood = speech.list_line([a["display_name"] for a in new + old])
        return speech.speak(self.a, text, mood)

    def say_status(self) -> str:
        n = len(self.store.list_areas(self.a.env.env_id()))
        text, mood = speech.status_line(self._map_m2(), n)
        return speech.speak(self.a, text, mood)

    # ======================================================================
    # small helpers
    # ======================================================================
    def _map_m2(self) -> float:
        if self.explored_m2:
            return self.explored_m2
        g = self.a.slam.grid()
        return g.explored_m2() if g is not None else 0.0

    def _event(self, kind: str, **data) -> None:
        log.info("mission %s %s", kind, data or "")
        if self.mission is not None:
            try:
                self.store.add_event(self.mission["id"], dict(data, kind=kind,
                                                              state=self.state))
            except Exception as exc:                    # noqa: BLE001
                log.warning("event not saved: %s", exc)

    def _to(self, state: str, now: float) -> None:
        if state != self.state:
            log.info("mission state %s -> %s", self.state, state)
            prev, self.state, self._t_state = self.state, state, now
            if state in ("EXPLORE", "RETURN_TO_START"):
                self._t_last_fix = now      # held: nothing was published meanwhile
            self._event("state", frm=prev, to=state)
            self._checkpoint(now)

    def _hold(self, why: str, now: float | None = None, force: bool = False) -> None:
        now = self.a.clock.monotonic() if now is None else now
        if force or now - self._t_hold >= self.cfg.run.hold_refresh_s:
            self._t_hold = now
            self.a.run.hold(why, self.cfg.run.hold_ttl_s)

    def _release(self) -> None:
        self._t_hold = -1e9
        self.a.run.release()

    def _read_status(self, now: float, force: bool = False) -> dict | None:
        """The run's published state, or None unless it is FRESH. The file
        outlives its run: a day-old one still says "returning", and a mission
        that believed it would think its new run was already heading home."""
        if force or now - self._t_status >= STATUS_PERIOD_S:
            self._t_status = now
            status = self.a.run.status()
            age = status.get("age_s", 0.0) if status else None
            self._status = status if (age is not None and age <= STATUS_MAX_AGE_S) \
                else None
        return self._status

    def _frame(self) -> str:
        return area_store.FRAME_MAP if self.T is not None \
            else f"mission:{self.mission['id']}"

    def _to_store(self, points):
        return mission_map.transform_points(points, self.T) if self.T is not None \
            else [[float(p[0]), float(p[1])] for p in points]

    def _areas_in_start_frame(self, now: float) -> list[dict]:
        """Saved areas this mission can compare its rooms with, as polygons in
        the start frame. Refreshed every segmentation period."""
        if now - self._areas_cache[0] < self.cfg.segmentation.period_s:
            return self._areas_cache[1]
        out, mine = [], self._frame()
        inv = mission_map.inverse(self.T) if self.T is not None else None
        for a in self.store.list_areas(self.env_id):
            if a["frame"] == area_store.FRAME_MAP and inv is not None:
                poly = mission_map.transform_points(a["polygon"], inv)
            elif a["frame"] == mine:
                poly = a["polygon"]
            else:
                continue
            out.append({"id": a["id"], "polygon": poly, "area_m2": a["area_m2"]})
        self._areas_cache = (now, out)
        return out

    def _seg_record(self, seg, complete: bool) -> dict:
        center = self._to_store([seg.center])[0]
        return {"env_id": self.env_id, "map_id": self.map_id, "frame": self._frame(),
                "polygon": self._to_store(seg.polygon), "center": tuple(center),
                "area_m2": seg.area_m2, "radius_m": seg.radius_m,
                "complete": bool(complete), "mission_id": self.mission["id"],
                "key": seg.key}

    def _checkpoint(self, now: float) -> None:
        if self.mission is None:
            return
        self._t_checkpoint = now
        try:
            self.store.save_checkpoint(self.env_id, {
                "mission_id": self.mission["id"], "state": self.state,
                "paused_from": self.paused_from, "stop_reason": self.stop_reason,
                "start": {"x": self.mission["start_x"], "y": self.mission["start_y"],
                          "theta": self.mission["start_theta"],
                          "signature": [None if not math.isfinite(v) else v
                                        for v in (self.mission.get("start_signature")
                                                  or [])]},
                "anchor": list(self.T) if self.T is not None else None,
                "map_id": self.map_id, "pose": list(self.pose),
                "breadcrumbs": self.trail.to_list(),
                "blacklist": self.blacklist.to_list(),
                "pending_label": self.pending_label,
                "counters": {"distance_m": round(self.distance_m, 2),
                             "explored_m2": round(self.explored_m2, 2),
                             "questions": self.questions,
                             "new_areas": list(self.new_area_ids)},
                "dwell": self.dwell.to_dict()})
        except Exception as exc:                        # noqa: BLE001
            log.warning("checkpoint not saved: %s", exc)

    # ======================================================================
    # the tick
    # ======================================================================
    def tick(self, now: float | None = None) -> str:
        now = self.a.clock.monotonic() if now is None else now
        t0 = time.perf_counter()
        spent_inline = 0.0
        with self._lock:
            self._commands(now)
            getattr(self, "_st_" + self.state.lower())(now)
            if self.analysis.inline and self.state in ("EXPLORE", "ASK_LABEL", "PAUSED"):
                t1 = time.perf_counter()
                self.analysis.step(now)         # the worker's job, done here in tests
                spent_inline = time.perf_counter() - t1
            if (self.mission is not None
                    and now - self._t_checkpoint >= self.cfg.checkpoint_s):
                self._checkpoint(now)
        self.tick_ms.append((time.perf_counter() - t0 - spent_inline) * 1000.0)
        return self.state

    def run_forever(self, stop: threading.Event | None = None) -> None:
        period = 1.0 / self.cfg.loop_hz
        self.analysis.start_thread(self.a.clock)
        nxt = self.a.clock.monotonic()
        while stop is None or not stop.is_set():
            try:
                self.tick()
            except Exception:                           # noqa: BLE001
                log.exception("mission tick failed")
            nxt += period
            wait = nxt - self.a.clock.monotonic()
            if wait > 0:
                self.a.clock.sleep(wait)
            else:
                nxt = self.a.clock.monotonic()

    # ---- commands -----------------------------------------------------------
    def _commands(self, now: float) -> None:
        while True:
            try:
                cmd = self._cmds.get_nowait()
            except queue.Empty:
                return
            name, st = cmd[0], self.state
            if name == "start":
                if st == "IDLE":
                    self._begin(cmd[1], now)
                else:
                    self.say("cant_now")
            elif name == "halt":
                if st in ("EXPLORE", "RETURN_TO_START", "TARGET_NAV"):
                    self._pause(now, st)
                elif st == "ASK_LABEL":
                    if self._stage == "dialog":
                        # The dialog is mid-sentence on its own thread: let it
                        # finish (the hold keeps her still), then pause.
                        self._pause_after_ask = True
                    else:
                        self._pause(now, "EXPLORE")     # the label stays pending
                elif st == "MISSION_INIT":
                    self._abort(now, "voice_stop")
                elif st == "HELP":
                    self._help_wait(now, self._ladder_distance(), speak=False)
                    self._close(now)                    # "Friday stop": no more tries
                elif st in ("HELP_WAIT", "FAILSAFE"):
                    self._close(now)
            elif name == "resume":
                if st == "PAUSED":
                    self._resume(now)
            elif name == "finish":
                if st in ("EXPLORE", "PAUSED", "ASK_LABEL"):
                    self.stop_reason = self.stop_reason or "voice_stop"
                    if st == "PAUSED" and self.paused_from == "RETURN_TO_START":
                        self.paused_from = None         # "go back" mid-return:
                        self._release()                 # carry on returning
                        self._to("RETURN_TO_START", now)
                    elif st == "ASK_LABEL" and self._stage == "dialog":
                        self._after_ask = "SUMMARIZE"   # after this question
                    else:
                        self._summarize(now)
                elif st == "HELP_WAIT":
                    self._return_again(now)
            elif name == "label_now":
                self._label_now(cmd[1], cmd[2], now)
            elif name == "go_to":
                self._go_to(cmd[2], now)
            elif name == "set_home":
                self._set_home(now)

    # ---- IDLE / MISSION_INIT ---------------------------------------------------
    def _st_idle(self, now: float) -> None:
        pass

    def _begin(self, minutes, now: float) -> None:
        self._reset_mission()
        self._minutes = minutes
        self.analysis.reset()
        self._stage = "capture"
        self.job = self.runner.start(self._capture, "mission-init")
        self.state, self._t_state = "MISSION_INIT", now
        log.info("mission state IDLE -> MISSION_INIT")

    def _capture(self) -> dict:
        """Standing still, before the run takes the LiDAR: the start signature
        and, if it can be had with certainty, where the start is on the saved
        map."""
        sig = self.cfg.signature
        scans = []
        for _ in range(sig.scans):
            sweep = self.a.lidar.sweep()
            if sweep:
                scans.append(sweep)
            self.a.clock.sleep(0.1)
        if len(scans) < max(3, sig.scans // 4):
            return {"ok": False, "why": f"only {len(scans)} sweeps"}
        anchor = home_anchor.capture(scans, [(0.0, 0.0, 0.0)] * len(scans),
                                     sig.max_range_m)
        return {"ok": True, "anchor": anchor, "map_pose": self.a.slam.localise(scans[-1])}

    def _st_mission_init(self, now: float) -> None:
        if self._stage == "capture":
            if not self.job.done:
                return
            res = self.job.result if self.job.error is None else {"ok": False}
            if not res or not res.get("ok"):
                log.warning("mission not started: %s", (res or {}).get("why"))
                self.say("cant_start")
                self.state = "IDLE"
                return
            self.env_id = self.a.env.env_id()
            self.map_id = self.a.slam.map_id()
            self.T = res["map_pose"]
            if self.store.mark_stale(self.env_id, self.map_id):
                self.say("map_reset")
            start = self.T if self.T is not None else (0.0, 0.0, 0.0)
            sig = res["anchor"].signature
            # ALL SAVED BEFORE THE FIRST MOTION (spec 4.9).
            self.mission = self.store.start_mission(self.env_id, self.map_id, start,
                                                    start_signature=sig)
            self.store.set_home_anchor(self.env_id, start[0], start[1], start[2], sig,
                                       self.map_id, "auto", self._frame())
            self._event("start", anchored=self.T is not None,
                        sweeps=res["anchor"].scans)
            self._checkpoint(now)
            self.say("mission_start")
            cap = self.cfg.explore.mission_cap_min
            minutes = min(float(self._minutes), cap) if self._minutes else cap
            self._minutes = minutes
            self.a.run.clear_return()
            started = self.a.run.start(minutes)
            if not (started or {}).get("started"):
                log.error("the run did not start: %s", started)
                self._abort(now, "failsafe")
                self.say("cant_now")
                return
            self._stage, self._t_state = "waiting", now
            return
        fix = self.a.run.live(self.cfg.run.live_max_age_s)
        if fix is not None:
            self._t_explore = now
            self._t_last_fix = now
            self._stag = (now, 0.0)
            self._to("EXPLORE", now)
        elif now - self._t_state > RUN_START_TIMEOUT_S:
            self._failsafe(now, "the run never published")

    # ---- EXPLORE ------------------------------------------------------------------
    def _take_fix(self, now: float) -> bool:
        fix = self.a.run.live(self.cfg.run.live_max_age_s)
        if fix is None:
            return False
        self._t_last_fix = now
        if fix.get("t") == self._last_fix_t:
            return True
        self._last_fix_t = fix.get("t")
        self._matched = str(fix.get("reason", "")).startswith("matched")
        pose = tuple(float(v) for v in fix["pose"])
        self.distance_m += math.hypot(pose[0] - self.pose[0], pose[1] - self.pose[1])
        self.pose = pose
        self._last_scan = fix.get("scan")
        self.trail.update(pose[0], pose[1], self.a.clock.wall())
        self.analysis.push(fix)
        return True

    def _st_explore(self, now: float) -> None:
        fresh = self._take_fix(now)
        status = self._read_status(now)
        self.analysis.set_inputs(self.pose, self.blacklist, self.trail.crumbs,
                                 self._areas_in_start_frame(now))
        if not fresh and now - self._t_last_fix > LOST_AFTER_S:
            if self.a.run.running():
                self._failsafe(now, "the run stopped publishing where she is")
            else:
                self._run_gone(now)
            return
        self.estop = self.a.run.estop(status)
        phase = (status or {}).get("way_back_phase")
        if phase in ("returning", "done"):
            # The run began its own way back: its timer, or it ran out of frontiers.
            self.stop_reason = self.stop_reason or (
                "time_cap" if self._minutes and now - self._t_explore
                >= self._minutes * 60.0 - 5.0 else "no_frontiers")
            self._end_exploring(now, run_returning=True)
            return
        res = self.analysis.latest
        if res is None:
            return
        self.explored_m2 = res["explored_m2"]
        here = res["seg"].at(self.pose[0], self.pose[1])
        self.dwell.update(here.key if here is not None else None, now)
        self._grow_saved(res, now)
        self._stagnation(res, now)
        if self.estop:
            return                       # her own reflex has her: start nothing new
        if self.a.power.low():
            self.stop_reason = "power_low"
            self.a.run.stop()
            self.say("battery")
            self._help_wait(now, speak=False)
            return
        if now - self._t_explore >= self.cfg.explore.mission_cap_min * 60.0:
            self.stop_reason = "time_cap"
            self._end_exploring(now)
            return
        # One count per frontier ANALYSIS (<= 1 Hz), not per 10 Hz tick: three
        # ticks on the same result are one look, not three.
        if res["t"] != self._t_counted and now - self._t_explore >= MIN_EXPLORE_S:
            self._t_counted = res["t"]
            if self.completion.update(len(res["frontiers"].remaining), self._matched):
                self.stop_reason = "no_frontiers"
                self._end_exploring(now)
                return
        choice = segmentation.choose_label_target(
            res["seg"], self.dwell, now, self._t_question, True, self.lparams,
            distance_from_saved_m=self._distance_from_saved(res))
        if choice is not None:
            self._begin_ask(choice.segment, choice.complete, now, after="EXPLORE")

    def _distance_from_saved(self, res) -> float:
        saved = [s.center for s in res["seg"].segments if s.area_id is not None]
        pts = saved or [(0.0, 0.0)]                    # nothing saved: from the start
        return min(math.hypot(self.pose[0] - x, self.pose[1] - y) for x, y in pts)

    def _grow_saved(self, res, now: float) -> None:
        """spec 3.5 growing map: same place, >= 20% bigger -> update silently."""
        if res.get("_grown_t") == res["t"]:
            return
        res["_grown_t"] = res["t"]
        areas = self._areas_in_start_frame(now)
        for area_id, seg in segmentation.growth_updates(
                res["seg"], areas, self.cfg.segmentation.grow_update_ratio):
            center = self._to_store([seg.center])[0]
            self.store.update_geometry(area_id, polygon=self._to_store(seg.polygon),
                                       center=center, area_m2=seg.area_m2,
                                       radius_m=seg.radius_m,
                                       mission_id=self.mission["id"])
            self._areas_cache = (-1e9, [])
            self._event("area_grew", area=area_id, area_m2=round(seg.area_m2, 1))

    def _stagnation(self, res, now: float) -> None:
        """Logged, never acted on: the run's own frontier planner does the
        driving (the spec's FRONTIER_ASSIST would be a second planner)."""
        t0, m0 = self._stag
        if now - t0 < self.cfg.explore.stagnation_window_s:
            return
        if (res["explored_m2"] - m0 < self.cfg.explore.stagnation_m2
                and res["frontiers"].remaining):
            self._event("stagnation", grew_m2=round(res["explored_m2"] - m0, 2),
                        reachable_frontiers=len(res["frontiers"].remaining))
        self._stag = (now, res["explored_m2"])

    def _end_exploring(self, now: float, run_returning: bool = False) -> None:
        """One last ask, cooldown ignored, for the room she is standing in
        (spec 4.6), then the summary."""
        res = self.analysis.latest
        if res is not None and not self.estop:
            choice = segmentation.choose_label_target(
                res["seg"], self.dwell, now, self._t_question, True, self.lparams,
                ignore_cooldown=True)
            if choice is not None:
                self._begin_ask(choice.segment, choice.complete, now, after="SUMMARIZE")
                return
        self._summarize(now)

    # ---- ASK_LABEL ----------------------------------------------------------------
    def _begin_ask(self, seg, complete: bool, now: float, after: str,
                   preset: label_dialog.Parsed | None = None) -> None:
        self.pending_label = self._seg_record(seg, complete)
        self._preset = preset
        self._after_ask = after
        self._stage = "stopping"
        self._hold("asking", now, force=True)
        self._to("ASK_LABEL", now)

    def _motors_stopped(self, status: dict | None) -> bool:
        return bool(status) and status.get("reason_code") == "mission_hold" \
            and not status.get("final_v") and not status.get("final_omega")

    def _st_ask_label(self, now: float) -> None:
        self._hold("asking", now)
        self._take_fix(now)
        if self._stage == "stopping":
            if self._motors_stopped(self._read_status(now, force=True)):
                seg, preset = dict(self.pending_label), self._preset
                self._stage = "dialog"
                self.questions += 1
                self.job = self.runner.start(
                    (lambda: self.dialog.named_now(seg, preset)) if preset is not None
                    else (lambda: self.dialog.run(seg)), "mission-dialog")
            elif now - self._t_state > STOP_CONFIRM_S:
                # Never ask while she may still be rolling. Try again later.
                self._event("ask_skipped", why="motors not confirmed stopped")
                self.pending_label = None
                self._t_question = now
                self._after(now)
            return
        if not self.job.done:
            return
        result = self.job.result if self.job.error is None else None
        self._t_question = now
        if result is None:                              # the dialog itself broke
            self.pending_label = None
            self._after(now)
            return
        for entry in result.transcript:
            self._event("dialog", **entry)
        if result.outcome == "stopped":
            # "Friday stop" mid-question: nothing saved, the label stays pending
            # and is asked again on "continue".
            self._pause_after_ask = False
            self._pause(now, "EXPLORE")
            return
        self.pending_label = None
        if result.area is not None:
            self.new_area_ids.append(result.area["id"])
            self._areas_cache = (-1e9, [])
            self._event("area_saved", area=result.area["id"], name=result.area["name"],
                        status=result.area["status"])
        if result.outcome == "finished":
            self.stop_reason = self.stop_reason or "voice_stop"
            self._after_ask = "SUMMARIZE"
        self._after(now)

    def _after(self, now: float) -> None:
        """Where a question leads: on exploring, back to paused, or the summary."""
        if self._pause_after_ask and self._after_ask != "SUMMARIZE":
            self._pause_after_ask = False
            self._pause(now, "EXPLORE")
        elif self._after_ask == "SUMMARIZE":
            self._summarize(now)
        elif self._after_ask == "PAUSED":
            self._hold("paused", now, force=True)
            self._to("PAUSED", now)
        else:
            self._release()
            self._to("EXPLORE", now)

    def _label_now(self, name: str, display: str, now: float) -> None:
        """"save this as X" while exploring or paused: the room she is in."""
        if self.state not in ("EXPLORE", "PAUSED"):
            self.say("cant_now")
            return
        res = self.analysis.latest
        seg = res["seg"].at(self.pose[0], self.pose[1]) if res else None
        if seg is None or seg.area_id is not None:
            self.say("cant_now")
            return
        after = "EXPLORE" if self.state == "EXPLORE" else "PAUSED"
        self._begin_ask(seg, seg.complete, now, after=after,
                        preset=label_dialog.Parsed("NAME", name, display))

    # ---- PAUSED -------------------------------------------------------------------
    def _pause(self, now: float, paused_from: str, already_said: bool = False) -> None:
        self.paused_from = paused_from
        if self.ladder is not None and self.ladder.phase not in ("external", "done"):
            self.ladder.e_stop_quietly()        # File 14 is driving: stop it too
        self._hold("paused", now, force=True)
        self._to("PAUSED", now)
        if not already_said:
            self.say("paused")

    def _st_paused(self, now: float) -> None:
        self._hold("paused", now)
        self._take_fix(now)

    def _resume(self, now: float) -> None:
        back = self.paused_from or "EXPLORE"
        self.paused_from = None
        self.say("resumed")
        if self.pending_label is not None and back == "EXPLORE":
            # The label she was asking about when he said stop: asked again.
            self._after_ask, self._stage, self._preset = "EXPLORE", "stopping", None
            self._to("ASK_LABEL", now)
            return
        self._release()
        self._t_last_fix = now
        self._to(back, now)

    # ---- TARGET_NAV ---------------------------------------------------------------
    def _go_to(self, target: dict, now: float) -> None:
        if self.state == "IDLE":
            if not self.a.nav.go(target):
                self.say("cant_now")
            return
        if self.state not in ("EXPLORE", "PAUSED") or not self.a.nav.available_during_run():
            self.say("cant_now")
            return
        self._nav_back = "EXPLORE"
        self._hold("go-to", now, force=True)
        if not self.a.nav.go(target):
            self._release()
            self.say("cant_now")
            return
        self._event("go_to", name=target.get("name"))
        self._to("TARGET_NAV", now)

    def _st_target_nav(self, now: float) -> None:
        self._hold("go-to", now)
        if self.a.nav.busy():
            return
        self._event("go_to_done", outcome=self.a.nav.outcome())
        self._release()
        self._t_last_fix = now
        self._to(self._nav_back, now)       # trail, blacklist, dwell: all kept

    # ---- SUMMARIZE -> RETURN_TO_START ------------------------------------------------
    def _summarize(self, now: float) -> None:
        self.state = "SUMMARIZE"
        res = self.analysis.latest
        unentered = 0
        if res is not None:
            for seg in segmentation.never_entered(res["seg"], self.dwell,
                                                  self.lparams.min_area_m2):
                rec = self._seg_record(seg, True)
                area = self.store.add_area(
                    rec["env_id"], rec["map_id"], polygon=rec["polygon"],
                    center=rec["center"], area_m2=rec["area_m2"],
                    radius_m=rec["radius_m"], name=None, source="auto",
                    mission_id=rec["mission_id"], frame=rec["frame"])
                self.new_area_ids.append(area["id"])
                unentered += 1
        named = [self.store.get(i) for i in self.new_area_ids]
        named = [a for a in named if a is not None]
        names = [a["display_name"] for a in named[:len(named) - unentered]]
        total = len(self.store.list_areas(self.env_id))
        text, mood = speech.summary(self.explored_m2, names, total,
                                    time_cap=self.stop_reason == "time_cap",
                                    not_entered=unentered)
        speech.speak(self.a, text, mood)
        self._event("summary", text=text, stop_reason=self.stop_reason)
        self.store.update_mission(self.mission["id"], stop_reason=self.stop_reason,
                                  explored_m2_end=self.explored_m2,
                                  distance_m=self.distance_m)
        self._start_return(now)

    def _start_return(self, now: float) -> None:
        status = self._read_status(now, force=True)
        if (status or {}).get("way_back_phase") not in ("returning", "done"):
            self.a.run.request_return(self.mission["id"])
        self._release()
        if not getattr(self.a.run, "announces_return", False):
            self.say("returning")
        self._t_return = now
        self._t_last_fix = now
        r = self.cfg["return"]
        grid = None
        try:
            grid = self.analysis.map.planner_grid()
        except Exception as exc:                        # noqa: BLE001
            log.warning("no mission map for the way back: %s", exc)
        # The ladder works in the START frame: the start is (0, 0).
        self.ladder = rth_ladder.Ladder(
            self.a.nav, rth_ladder.LadderConfig.from_config(self.cfg), self.say,
            {"goal": (0.0, 0.0), "trail": self.trail.crumbs, "grid": grid,
             "inflate_m": self.a.slam.inflate_m(),
             "body_radius_m": self.a.slam.body_radius_m(), "obstacles": None})
        self.ladder.begin(now)
        self.arrival = rth_ladder.ArrivalCheck(r.arrive_tolerance_m,
                                               r.arrive_hysteresis_m, r.arrive_dwell_s)
        self._arrive = None
        self._to("RETURN_TO_START", now)

    def _ladder_distance(self):
        if self.ladder is None:
            return None
        if self.ladder.attempts and self.ladder.attempts[-1].distance_m is not None:
            return self.ladder.attempts[-1].distance_m
        return math.hypot(self.pose[0], self.pose[1])

    def _st_return_to_start(self, now: float) -> None:
        if self._arrive is not None:
            self._arrival_steps(now)
            return
        out = None
        if self.ladder.phase == "external":
            # The first attempt: the run's own way back.
            fresh = self._take_fix(now)
            status = self._read_status(now)
            phase = (status or {}).get("way_back_phase")
            tracked = math.hypot(self.pose[0], self.pose[1])
            settled = fresh and self.arrival.update(tracked, now)
            if phase == "done":
                arrived = status.get("way_back_outcome") == "arrived" or settled
                out = self._run_way_back_over(arrived, status.get(
                    "way_back_distance_m", tracked), now)
            elif phase != "returning" and now - self._t_return > RETURN_REASK_S:
                if not self.a.run.running():
                    # It ended without saying how: where the tracker last had her.
                    out = self._run_way_back_over(
                        tracked <= self.arrival.tolerance_m and self.distance_m > 0,
                        tracked, now)
                else:
                    self.a.run.request_return(self.mission["id"])
                    self._t_return = now
        else:
            out = self.ladder.tick(now)
        self._ladder_outcome(out, now)

    def _run_way_back_over(self, arrived: bool, distance, now: float):
        """The run has ended. Give the hardware back to friday-mcp (a run that
        ends on its own leaves its lease behind -- File 14's lease trap), and
        let the ladder take it from there."""
        try:
            self.a.run.stop()
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not reclaim the hardware: %s", exc)
        return self.ladder.external_result(arrived, distance, now)

    def _ladder_outcome(self, out, now: float) -> None:
        if out == "arrived":
            self._arrive = {"stage": "look", "job": None, "spun": False,
                            "distance": self._ladder_distance()}
        elif out == "help_wait":
            self._help_wait(now, self._ladder_distance(), speak=False)
            self.say("help_wait")
        elif self.ladder.in_help and self.state != "HELP":
            self._to("HELP", now)

    def _st_help(self, now: float) -> None:
        """Stuck, and saying so: rungs 1-6 again every help_retry_s."""
        if self._arrive is not None:
            self._arrival_steps(now)
            return
        self._ladder_outcome(self.ladder.tick(now), now)

    def _return_again(self, now: float) -> None:
        """"Friday, come back" in HELP_WAIT: the whole ladder again."""
        if self.ladder is None:
            self.say("cant_now")
            return
        self.store.update_mission(self.mission["id"], status="running")
        self._to("RETURN_TO_START", now)
        self._ladder_outcome(self.ladder.restart(now), now)

    # ---- the look at arrival (spec 4.12) -----------------------------------------
    def _look(self) -> dict:
        sig = self.cfg.signature
        sweeps = []
        for _ in range(sig.arrival_scans):
            sweep = self.a.lidar.sweep()
            if sweep:
                sweeps.append(sweep)
            self.a.clock.sleep(0.1)
        return rth_ladder.verify(sweeps, (self.mission or {}).get("start_signature"),
                                 sig.max_range_m, sig.match_mad_m, sig.min_overlap)

    def _arrival_steps(self, now: float) -> None:
        """Position says she is there. Look once; if the view does not match,
        one spin + relocalise and look again; then arrive EITHER WAY -- the
        second mismatch is `verified = false`, never a third try."""
        a = self._arrive
        if a["stage"] == "look":
            if a["job"] is None:
                a["job"] = self.runner.start(self._look, "mission-arrival")
                return
            if not a["job"].done:
                return
            seen = a["job"].result if a["job"].error is None else {"verified": False}
            a["job"] = None
            if seen.get("verified") or a["spun"]:
                self._event("arrival_check", **seen)
                self._arrive = None
                self._arrived(now, a["distance"], bool(seen.get("verified")))
                return
            self._event("arrival_check", **dict(seen, retry=True))
            a["spun"] = True
            if self.a.nav.available() and self.a.nav.spin_relocalise():
                self.say("rung3")
                a["stage"], a["t"] = "spin", now
            # no engine to spin with: the second look is the same view, so it
            # settles at once as not verified.
        elif a["stage"] == "spin":
            if not self.a.nav.busy() or now - a["t"] > rth_ladder.MANEUVER_TIMEOUT_S:
                a["stage"] = "look"

    # ---- ARRIVED_START -> SAVE_MISSION -> IDLE ---------------------------------------
    def _arrived(self, now: float, distance, verified: bool = False) -> None:
        self.state = "ARRIVED_START"
        self.verified = verified
        self._finish_mission(now, "returned", home_error_m=distance)
        self.say("arrived" if not getattr(self.a.run, "announces_return", False)
                 else "saved_only")
        self._close(now)

    def _finish_mission(self, now: float, status: str, **extra) -> None:
        """SAVE_MISSION first, then she speaks (spec 4.12)."""
        self.state = "SAVE_MISSION"
        m = self.mission
        end = self._to_store([self.pose[:2]])[0]
        rungs = self.ladder.rungs_used() if self.ladder is not None else []
        if self.verified is not None:
            rungs = rungs + [{"rung": "arrival", "verified": self.verified}]
        self.store.update_mission(
            m["id"], status=status, ended_at=area_store.now_iso(), end_x=end[0],
            end_y=end[1], distance_m=self.distance_m,
            explored_m2_end=self.explored_m2, stop_reason=self.stop_reason,
            rth_attempts=len(self.ladder.attempts) if self.ladder is not None else 0,
            rungs_used=rungs, **{k: v for k, v in extra.items() if v is not None})
        self.store.save_trail(self.env_id, m["id"], self.trail.to_list())
        self._event("mission_saved", status=status)

    def _help_wait(self, now: float, distance=None, speak: bool = True) -> None:
        self._finish_mission(now, "help_wait", home_error_m=distance)
        if speak:
            self.say("help", d=distance if distance is not None else 0.0)
            self.say("help_wait")
        self._to("HELP_WAIT", now)

    def _st_help_wait(self, now: float) -> None:
        pass                                 # motors off, no autonomy: waits for him

    def _failsafe(self, now: float, why: str) -> None:
        log.error("FAILSAFE: %s", why)
        try:
            self.a.run.stop()
        except Exception as exc:                        # noqa: BLE001
            log.error("could not stop the run: %s", exc)
        self._release()
        if self.mission is not None:
            self.stop_reason = "failsafe"
            self._finish_mission(now, "failsafe")
        self.say("failsafe")
        self._to("FAILSAFE", now)

    def _st_failsafe(self, now: float) -> None:
        pass                                 # stop, speak, dashboard, no autonomy

    def _run_gone(self, now: float) -> None:
        """The run ended under the mission (the dashboard's Stop, a person).
        A person's stop is not followed by a drive back."""
        self.stop_reason = self.stop_reason or "voice_stop"
        self._release()
        self._finish_mission(now, "aborted")
        self._close(now)

    def _abort(self, now: float, reason: str) -> None:
        try:
            self.a.run.stop()
        except Exception as exc:                        # noqa: BLE001
            log.error("could not stop the run: %s", exc)
        self._release()
        if self.mission is not None:
            self.stop_reason = reason
            self._finish_mission(now, "aborted")
        self._close(now)

    def _close(self, now: float) -> None:
        """Back to IDLE. The mission row and its areas are already saved."""
        self.a.run.clear_return()
        self._release()
        self.state = "IDLE"
        self._checkpoint(now)
        log.info("mission closed -> IDLE")
        self.mission = None

    # ---- the transient states are never ticked, but must exist -----------------------
    def _st_summarize(self, now: float) -> None:
        pass

    def _st_arrived_start(self, now: float) -> None:
        pass

    def _st_save_mission(self, now: float) -> None:
        pass

    # ---- SET_HOME ---------------------------------------------------------------
    def _set_home(self, now: float) -> None:
        if self.state not in ("IDLE", "PAUSED"):
            self.say("cant_now")
            return
        env = self.a.env.env_id()
        if self.state == "PAUSED":
            # Held: the run owns the LiDAR and publishes nothing while she
            # stands still, so the anchor is its last sweep, taken where she
            # is standing now, at the tracked pose.
            if not self._last_scan:
                self.say("cant_now")
                return
            sig = home_anchor.signature([self._last_scan], self.cfg.signature.max_range_m)
            pose = self.T and mission_map.compose(self.T, self.pose) or self.pose
            self.store.set_home_anchor(env, pose[0], pose[1], pose[2], sig,
                                       self.map_id, "voice", self._frame())
            self.say("home_set")
            return
        res = self._capture()
        if not res.get("ok"):
            self.say("cant_start")
            return
        pose = res["map_pose"] or (0.0, 0.0, 0.0)
        frame = area_store.FRAME_MAP if res["map_pose"] is not None else "unplaced"
        self.store.set_home_anchor(env, pose[0], pose[1], pose[2],
                                   res["anchor"].signature, self.a.slam.map_id(),
                                   "voice", frame)
        self.say("home_set")

    # ---- for the dashboard ---------------------------------------------------------
    def snapshot(self) -> dict:
        res = self.analysis.latest
        return {"state": self.state, "paused_from": self.paused_from,
                "mission_id": self.mission["id"] if self.mission else None,
                "stop_reason": self.stop_reason, "pose": list(self.pose),
                "anchored": self.T is not None, "explored_m2": round(self.explored_m2, 1),
                "distance_m": round(self.distance_m, 1), "estop": self.estop,
                "questions": self.questions, "new_areas": list(self.new_area_ids),
                "crumbs": len(self.trail),
                "return": None if self.ladder is None else {
                    "rung": self.ladder.rung, "phase": self.ladder.phase,
                    "cycle": self.ladder.cycle, "in_help": self.ladder.in_help,
                    "attempts": self.ladder.rungs_used()},
                "verified": self.verified,
                "frontiers_left": len(res["frontiers"].remaining) if res else None,
                "rooms": len(res["seg"].segments) if res else None,
                "map": res["map"] if res else None}
