"""embodiment.py — "look before you move", enforced rather than requested.

A docstring that says "call observe() first" is advice. The model can forget
it, run out of attention for it, or reason its way past it while sounding
entirely sensible. This file is what makes it structural: movement tools ask
here first, and if the robot has not actually looked at the world recently,
the move does not happen. No argument, no override argument, no way through.

WHAT IS ENFORCED

  * A movement tool refuses unless an observation was taken within the TTL.
    observe() and scan_environment() are what record one.
  * The TTL is SHORTER once the robot has moved since that observation
    (config.OBSERVATION_TTL_MOVING_S). Driving invalidates a view faster than
    the world drifting on its own does, so a move spends the freshness it was
    granted -- and the next move needs a new look. That is what makes the loop
    observe -> move -> observe -> move instead of one look and a blind
    sequence.
  * Steps are CLIPPED to config.MAX_INCREMENT_CM / MAX_INCREMENT_DEGREES
    rather than refused. Refusing a 2 m request teaches the model to argue;
    clipping it to 40 cm and saying "partial, 160 cm still to go" teaches it
    to re-observe and continue, which is the behaviour actually wanted.

WHAT IS DELIBERATELY NOT GATED

  stop(). Never. An emergency stop that could be refused because the robot has
  not looked recently would be an emergency stop that fails exactly when it is
  needed most. Reversing and turning are gated on freshness like anything else
  but are not blocked by forward clearance -- see safety.py for why.

UNKNOWN IS NOT CLEAR. If the last observation could not measure a direction,
asking to move that way is refused with an instruction to ASK Harsh rather
than guess. A null reading means no sensor covered it; it never means "fine".
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import config

log = logging.getLogger("friday.mcp.embodiment")

_LOCK = threading.Lock()


@dataclass
class Observation:
    """One moment of actually having looked, and what it showed."""
    at: float                                   # time.monotonic()
    clearance_mm: dict = field(default_factory=dict)
    source: str = "observe"
    saw_image: bool = False
    moved_since: bool = False                   # set when a move consumes it

    def age_s(self) -> float:
        return time.monotonic() - self.at

    def ttl_s(self) -> float:
        """How long this observation stays trustworthy.

        Shorter once it has been spent on a move, because the robot itself is
        now the thing that changed the scene.
        """
        return (config.OBSERVATION_TTL_MOVING_S if self.moved_since
                else config.OBSERVATION_TTL_S)

    def is_fresh(self) -> bool:
        return self.age_s() <= self.ttl_s()


_last: Observation | None = None


def record(clearance_mm: dict | None = None, source: str = "observe",
           saw_image: bool = False) -> None:
    """Note that Friday just looked. Called by observe() / scan_environment()."""
    global _last
    with _LOCK:
        _last = Observation(at=time.monotonic(),
                            clearance_mm=dict(clearance_mm or {}),
                            source=source, saw_image=saw_image)
    log.debug("observation recorded from %s (image=%s)", source, saw_image)


def note_moved() -> None:
    """Called after a move. Spends the current observation's freshness."""
    with _LOCK:
        if _last is not None:
            _last.moved_since = True


def last() -> Observation | None:
    with _LOCK:
        return _last


def status() -> dict:
    """For get_status and the tests: is Friday currently cleared to move."""
    obs = last()
    if obs is None:
        return {"has_observed": False, "cleared_to_move": False,
                "reason": "nothing observed since startup"}
    return {
        "has_observed": True,
        "age_s": round(obs.age_s(), 1),
        "ttl_s": obs.ttl_s(),
        "moved_since_observing": obs.moved_since,
        "cleared_to_move": obs.is_fresh(),
        "source": obs.source,
    }


# Directions whose clearance we can meaningfully check against the ledger.
_DIRECTION_KEYS = {
    "forward": "forward", "ahead": "forward", "front": "forward",
    "backward": "backward", "back": "backward", "backwards": "backward",
    "reverse": "backward",
    "left": "left", "right": "right",
}

# Directions that carry Friday somewhere new, as opposed to pivoting in place.
# Only these are refused for an unmeasured direction -- see check().
_TRANSLATING = {"forward", "backward"}


def check(direction: str) -> str | None:
    """Refusal text, or None to allow the move.

    This is the gate. Every movement tool except stop() calls it first.
    """
    if not config.REQUIRE_OBSERVATION_BEFORE_MOVING:
        return None

    obs = last()
    if obs is None:
        return ("REFUSED — Friday has not looked at the room yet. Call "
                "observe() first, say what you can actually see, then move in "
                "one small step. Moving blind is the one thing that is never "
                "allowed.")

    if not obs.is_fresh():
        return (f"REFUSED — the last observation is {obs.age_s():.0f}s old and "
                f"is only trusted for {obs.ttl_s():.0f}s"
                + (" once Friday has moved" if obs.moved_since else "")
                + ". The room may not look like that any more. Call observe() "
                  "again and check the scene changed the way you expected "
                  "before moving further.")

    # Fresh, but did it actually cover the direction being asked for?
    #
    # ONLY FOR DIRECTIONS THAT TRANSLATE THE ROBOT. Driving forward or backward
    # carries Friday into space it may not have measured, so an unmeasured
    # direction there is a refusal. A pivot in place does not: it sweeps the
    # robot's own footprint, which it is already standing in. Gating turns on
    # unknown side clearance would also be self-defeating -- with the LiDAR
    # unplugged, left and right read null permanently, so Friday could never
    # turn, and a robot stopped facing a wall would have no way out of the
    # room. safety.py refuses to block turns for exactly this reason; this
    # gate must not quietly reintroduce the problem one layer up.
    key = _DIRECTION_KEYS.get(direction.strip().lower())
    if key in _TRANSLATING and obs.clearance_mm:
        if key in obs.clearance_mm and obs.clearance_mm[key] is None:
            return (f"REFUSED — nothing could measure {key} at the last "
                    f"observation, so that direction is UNKNOWN, not clear. "
                    f"Do not guess. Say you cannot see that way and ASK Harsh "
                    f"whether it is safe before moving {key}.")
    return None


def clip_increment(value: float, limit: float, unit: str) -> tuple[float, str | None]:
    """Clamp one step to a single increment. Returns (value, note or None).

    CLIPPED, NOT REFUSED, and the difference matters. Refusing "go 2 metres"
    teaches the model to argue with the tool or give up on the errand. Clipping
    it to 40 cm and saying so -- with the remainder named -- teaches it to take
    the step, look again, and continue, which is the behaviour actually wanted.
    The sign is preserved, so backward and left-hand turns clip the same way.
    """
    if abs(value) <= limit:
        return value, None
    clipped = limit if value > 0 else -limit
    remaining = abs(value) - limit
    return clipped, (
        f"clipped to a single {limit:g} {unit} step — {remaining:g} {unit} of "
        f"the {abs(value):g} asked for is still to go. Observe again and check "
        f"the scene changed as expected, then continue if it still looks right")
