"""Event-driven voice for the autonomy stack. Grounded, throttled, and never blocking.

THE HARD CONSTRAINT
-------------------
`friday/tts.py speak()` is BLOCKING -- it returns when Friday has finished talking,
which for a sentence is a second or more. The control loop has a 100 ms budget and
currently uses about 7 ms. One `speak()` on that thread would blow the deadline by an
order of magnitude, the LiDAR slot would go stale, and the fail-safe would stop the
robot. Speaking would look exactly like a sensor fault.

So the control loop only ever ENQUEUES. A worker thread does the talking, and if it
falls behind, the queue drops rather than growing -- stale narration is worse than
silence, because a robot describing a wall it cleared four seconds ago is actively
misleading.

WHAT IT IS ALLOWED TO SAY
-------------------------
Only things derived from the autonomy state that has just been measured. Every line
here is built from a real field: a clearance in metres, a target's bearing, the
recovery trigger that actually fired. There are no canned observations, and nothing
that names an object -- no object detection is implemented, so a sentence like "I see
a chair" would be an invention. The rule is the same one the telemetry follows: if it
cannot be grounded in a field, it is not said.

THROTTLING
----------
Three layers, because a robot that narrates every tick is unusable and gets muted,
which costs all the value of having it speak at all:

  1. Per-event cooldown -- the same kind of event cannot repeat for N seconds.
  2. A global minimum gap, so two different events cannot stack into a monologue.
  3. Deduplication of identical text, so a rephrasing of the same fact is dropped.

Priority breaks ties: safety and recovery events jump the queue, exploration
commentary is dropped first when the queue is full.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

LOGGER = logging.getLogger("pi_deploy.narrator")

# Priorities. Higher wins a full queue.
P_SAFETY = 30       # something stopped the robot
P_RECOVERY = 20     # stuck, or got out of it
P_EXPLORE = 10      # targets, regions, progress
P_CHATTER = 5       # everything else

# Per-event-kind cooldowns, seconds. Tuned so a run is narrated, not dictated.
COOLDOWNS = {
    "start": 0.0,
    "target": 12.0,
    "region": 20.0,
    "obstacle": 8.0,
    "safety_stop": 6.0,
    "stuck": 5.0,
    "recovered": 5.0,
    "exhausted": 60.0,
    "camera": 30.0,
    "done": 0.0,
}
MIN_GAP_S = 3.0             # global floor between any two utterances
QUEUE_MAX = 4               # small on purpose: a backlog means stale narration


@dataclass(order=True)
class _Utterance:
    neg_priority: int
    seq: int
    kind: str = ""
    text: str = ""


class Narrator:
    """Turns autonomy-state transitions into speech, off the control thread."""

    def __init__(self, speak: Callable[[str], Any] | None = None,
                 enabled: bool = False, verbosity: str = "normal") -> None:
        self.enabled = enabled
        self.verbosity = verbosity
        self._speak = speak
        self._q: queue.PriorityQueue = queue.PriorityQueue(maxsize=QUEUE_MAX)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._last_said: dict[str, float] = {}
        self._last_any = 0.0
        self._last_text = ""
        self.spoken = 0
        self.dropped = 0
        self.suppressed = 0
        self.log: list[tuple[float, str, str]] = []

        # Transition state. All initialised to sentinels that cannot equal a real
        # value, so the FIRST observation is not mistaken for a change.
        self._prev_target: Any = "<none>"
        self._prev_gate = "<none>"
        self._prev_recovery_state = "<none>"
        self._prev_region_m2 = -1.0
        self._prev_cams: dict[str, bool] = {}
        self._announced_start = False

    # ---------------------------------------------------------------- lifecycle --
    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="narrator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                u = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                if self._speak is not None:
                    self._speak(u.text)
                else:
                    LOGGER.info("[voice] %s", u.text)
                self.spoken += 1
            except Exception as exc:                   # noqa: BLE001
                # Speech failing must never affect driving. Log once and carry on.
                LOGGER.warning("narration failed (%s): %s", type(exc).__name__, exc)

    # ------------------------------------------------------------------- queue ---
    def say(self, kind: str, text: str, priority: int = P_CHATTER) -> bool:
        """Enqueue. Returns whether it was accepted. NEVER blocks."""
        if not self.enabled or not text:
            return False
        now = time.monotonic()
        if now - self._last_said.get(kind, -1e9) < COOLDOWNS.get(kind, 10.0):
            self.suppressed += 1
            return False
        # The global gap exists to stop EXPLORATION COMMENTARY stacking into a
        # monologue. It must not silence a rarer, more important event that happens to
        # land shortly after a chatty one: measured in the smoke test, a stop at t=0
        # swallowed the recovery, the new target AND the map milestone that followed
        # within three seconds -- three distinct facts lost to one timer. Safety and
        # recovery bypass it; their own per-kind cooldowns are the real limiter, and
        # those events are rare by construction.
        if now - self._last_any < MIN_GAP_S and priority < P_RECOVERY:
            self.suppressed += 1
            return False
        if text == self._last_text:
            self.suppressed += 1
            return False

        self._seq += 1
        u = _Utterance(-priority, self._seq, kind, text)
        try:
            self._q.put_nowait(u)
        except queue.Full:
            # Drop the LOWEST-priority item to make room, and drop this one if it is
            # the lowest. A full queue means we are already behind; letting it grow
            # would only make the narration staler.
            try:
                worst = max(self._q.queue)
                if worst.neg_priority > u.neg_priority:
                    self._q.queue.remove(worst)
                    self._q.put_nowait(u)
                else:
                    self.dropped += 1
                    return False
            except (ValueError, queue.Full):
                self.dropped += 1
                return False
        self._last_said[kind] = now
        self._last_any = now
        self._last_text = text
        self.log.append((time.time(), kind, text))
        return True

    # ------------------------------------------------------------ state -> speech -
    def observe(self, state: dict[str, Any]) -> None:
        """Called from the control loop with the freshly built autonomy state.

        Pure transition detection plus string building -- microseconds, no I/O. The
        expensive part happens on the worker thread.
        """
        if not self.enabled or not state:
            return

        if not self._announced_start:
            self._announced_start = True
            mode = state.get("CURRENT_MODE", "autonomous mode")
            self.say("start", f"Starting {mode.lower()}.", P_EXPLORE)

        # --- safety: a NEW stop, not a continuing one -----------------------------
        gate = state.get("REASON_CODE", "")
        sensors = state.get("SENSOR_STATE", {}) or {}
        if gate != self._prev_gate:
            if gate == "REFLEX_STOP":
                d = (sensors.get("tof") or {}).get("nearest")
                self.say("safety_stop",
                         f"Stopping. Obstacle {d:.2f} metres ahead." if d is not None
                         else "Stopping. Obstacle too close.", P_SAFETY)
            elif gate == "governor":
                d = (sensors.get("lidar") or {}).get("min")
                if d is not None:
                    self.say("obstacle",
                             f"Slowing down, {d:.1f} metres of clearance.", P_EXPLORE)
            elif gate in ("stale_lidar", "stale_tof"):
                self.say("safety_stop",
                         f"Stopping. My {'lidar' if 'lidar' in gate else 'depth sensor'} "
                         f"stopped reporting.", P_SAFETY)
            elif gate == "no_reverse_turn":
                self.say("obstacle",
                         "I need to back up but I cannot, so I am turning instead.",
                         P_EXPLORE)
        self._prev_gate = gate

        # --- recovery -------------------------------------------------------------
        rec = state.get("RECOVERY_STATUS") or {}
        rstate = rec.get("state", "<none>")
        if rstate != self._prev_recovery_state:
            if rstate == "RECOVERING":
                active = rec.get("active") or {}
                trig = (active.get("trigger") or "").replace("_", " ")
                self.say("stuck",
                         f"I am not making progress: {trig}. Trying a new heading."
                         if trig else "I am not making progress. Trying a new heading.",
                         P_RECOVERY)
            elif rstate == "MONITOR" and self._prev_recovery_state == "VERIFY":
                self.say("recovered", "I am moving freely again.", P_RECOVERY)
        self._prev_recovery_state = rstate

        # --- exploration target ---------------------------------------------------
        tgt = state.get("EXPLORATION_TARGET") or {}
        if tgt.get("exhausted"):
            self.say("exhausted",
                     "I have no unexplored areas left that I can reach.", P_EXPLORE)
        key = tgt.get("target_xy")
        if key is not None and key != self._prev_target:
            plan = state.get("PLANNER_CONTRIBUTION") or {}
            err = plan.get("bearing_error")
            dist = tgt.get("target_distance")
            if err is not None and dist is not None:
                side = "left" if err > 0 else "right"
                self.say("target",
                         f"New area to explore, {dist:.1f} metres to my {side}.",
                         P_EXPLORE)
            elif dist is not None:
                self.say("target", f"New area to explore, {dist:.1f} metres away.",
                         P_EXPLORE)
        self._prev_target = key

        # --- map growth -----------------------------------------------------------
        m = state.get("MAP_STATE") or {}
        area = m.get("explored_m2")
        if area is not None:
            if self._prev_region_m2 < 0:
                self._prev_region_m2 = area
            elif area - self._prev_region_m2 >= 10.0:
                self._prev_region_m2 = area
                self.say("region",
                         f"I have mapped {area:.0f} square metres so far.", P_EXPLORE)

        # --- cameras coming and going --------------------------------------------
        for role, st in (state.get("SENSOR_STATE", {}).get("cameras") or {}).items():
            present = bool(st.get("device"))
            was = self._prev_cams.get(role)
            if was is not None and present != was:
                self.say("camera",
                         f"My {role} camera is back." if present
                         else f"I have lost my {role} camera.", P_CHATTER)
            self._prev_cams[role] = present

    def finish(self, state: dict[str, Any] | None) -> None:
        """One closing line, built from what the run actually achieved."""
        if not self.enabled:
            return
        m = (state or {}).get("MAP_STATE") or {}
        area = m.get("explored_m2")
        self.say("done",
                 f"Exploration finished. I mapped {area:.0f} square metres."
                 if area else "Exploration finished.", P_EXPLORE)

    def snapshot(self) -> dict[str, Any]:
        return {"enabled": self.enabled, "spoken": self.spoken,
                "dropped": self.dropped, "suppressed": self.suppressed,
                "queued": self._q.qsize()}
