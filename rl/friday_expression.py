"""What Friday's face should show while she is exploring.

Harsh, 2026-09-03: "the facial expressions are not matching the tasks it is
performing."

They were not matching because NOTHING IN THE EXPLORATION STACK EVER SET THEM.
`tts.speak()` switches to "speaking" and puts back whatever was there before, and
that was the only writer in the whole autonomous path -- so through a three-minute
run that drove 5.4 m^2, found frontiers, braked for obstacles and recovered from
being stuck, her face showed whatever it happened to be showing at boot.

TWO RULES, both learned the hard way in this project:

  * NEVER set "speaking". tts owns it. `tts._Mouth.start()` saves the current
    emotion so it can restore it afterwards, so a second writer setting "speaking"
    makes tts save "speaking" and restore into it -- Friday then talks forever with
    nothing to say. main.py carries the same warning for the same reason.

  * The mapping is a PURE FUNCTION of state the supervisor already has, so it can
    be tested without a face, a robot, or a display. The writer is the thin part.

The vocabulary is fixed by face_control.EMOTIONS and friday_face.py's geometry
table; the two refuse to start if they disagree, so nothing here may invent a name.
"""

from __future__ import annotations

import logging
import time

LOGGER = logging.getLogger("friday.expression")

# Phase names are duplicated rather than imported to keep this module free of any
# dependency on the supervisor -- it is imported BY it. They are asserted against
# friday_exploration.Phase in tests/test_expression.py, so drift cannot go unnoticed.
BOOTSTRAP, SELECT, NAVIGATE = "bootstrap", "select", "navigate"
OBSERVE, RECOVER, COMPLETE = "observe", "recover", "complete"

# How close something has to be, in front, before "surprised" becomes "scared".
# 0.25 m is inside the 0.36 m hard stop, so it means the safety chain has already
# refused to move and she is nearly touching -- not merely braking.
SCARED_DISTANCE_M = 0.25

# Re-assert the current intent this often even when it has not changed. tts restores
# the emotion it saved when it started speaking, which is by then out of date, so
# without this she would keep whatever she was feeling before the sentence.
REASSERT_S = 4.0

# Minimum time an expression stays on her face before another may replace it.
#
# Not a performance tweak -- a legibility one, though it helps both. The gate column
# changes several times a second in a tight space (REFLEX_STOP, governor, recovery,
# planner, ok), and mapping that straight through made the face strobe between
# surprised, thinking and dizzy faster than its own 280 ms cross-fade could finish.
# A face that never completes a transition is not showing an expression at all, which
# is the complaint this module exists to answer. It also costs a state-file write and
# a rename each time, on a Pi already missing its 100 ms control budget.
#
# 0.6 s is just over two cross-fades, so every expression is fully drawn and readable.
# ALARM IS EXEMPT: something appearing in front of her must reach her face at once.
MIN_DWELL_S = 0.6
URGENT = frozenset({"surprised", "scared"})


def expression_for(phase: str, gate: str = "ok", *,
                   fwd_min: float | None = None,
                   just_reached: bool = False,
                   just_found: bool = False) -> str:
    """The face for this moment. Pure; returns a name from face_control.EMOTIONS.

    Ordered most urgent first, because these overlap constantly -- she is very often
    navigating AND braking AND holding a target, and the face can only say one thing.
    What wins is what a person watching would most need to understand.
    """
    # 1. Danger, and how close it is. The safety chain has already acted by the time
    #    these gates appear; the face is reporting, not deciding.
    if gate in ("REFLEX_STOP", "visual_stop"):
        if fwd_min is not None and fwd_min <= SCARED_DISTANCE_M:
            return "scared"
        return "surprised"
    if gate in ("stale_lidar", "stale_tof"):
        return "confused"          # she cannot see; that is confusion, not fear

    # 2. Being stuck. "dizzy" for the spin-in-place recovery specifically, because
    #    that is what it looks like from outside and the face should agree with the
    #    wheels; "confused" for the rest of being unable to make progress.
    if gate == "recovery":
        return "dizzy"
    if phase == RECOVER:
        return "confused"

    # 3. Events worth a reaction, checked before the steady states they interrupt.
    if just_reached:
        return "success"
    if just_found:
        return "excited"

    # 4. Steady states.
    if phase == COMPLETE:
        return "proud"             # the room is mapped; that is an accomplishment
    if phase == NAVIGATE:
        # Braking for something is not an emergency, but it is not business as usual
        # either -- it is the moment she has noticed something and is deciding.
        return "thinking" if gate in ("governor", "tof_slow", "escape_turn",
                                      "no_reverse_turn", "reverse_lifted",
                                      "rear_blocked") else "curious"
    if phase in (SELECT, OBSERVE):
        return "thinking"          # choosing where to go / folding in what she saw
    if phase == BOOTSTRAP:
        return "listening"         # sensors spinning up, taking a first look
    return "neutral"


class FaceExpression:
    """Writes `expression_for` to the face. Best-effort and never in the way.

    The face is a separate process on a CPU-bound Pi. Losing an expression must
    never cost a control tick, so every write is guarded and failures are logged
    once rather than raised -- exactly the contract tts._Mouth works to.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.current: str | None = None
        self.changes = 0
        self.suppressed = 0
        self._last_write = 0.0
        self._last_change = 0.0
        self._warned = False

    def update(self, phase: str, gate: str = "ok", **kw) -> str | None:
        """Called once per tick. Returns what it set, or None if it left it alone."""
        if not self.enabled:
            return None
        want = expression_for(phase, gate, **kw)
        now = time.monotonic()
        # Write on a change, or periodically to undo tts's stale restore. Writing
        # every tick would be 10 file writes a second for no benefit.
        if want == self.current and (now - self._last_write) < REASSERT_S:
            return None
        # Hold the current expression long enough to actually be seen -- unless the new
        # one is alarm, which must never wait behind a calmer face.
        if (want != self.current and want not in URGENT
                and (now - self._last_change) < MIN_DWELL_S):
            self.suppressed += 1
            return None
        try:
            import face_control
            face_control.set_emotion(want)
        except Exception as exc:                        # noqa: BLE001
            if not self._warned:
                self._warned = True
                LOGGER.debug("face unavailable, expressions disabled for this run: %s",
                             exc)
            self.enabled = False
            return None
        if want != self.current:
            self.changes += 1
            self._last_change = now
        self.current, self._last_write = want, now
        return want
