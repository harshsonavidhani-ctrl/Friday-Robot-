"""Deciding where to go next, when nothing has told Friday where to go.

This is the exploration POLICY and it does not drive anything. It looks at the
map, picks a frontier, plans a route to it, and hands back the turns and
distances; the caller executes them through the MCP server, where the safety
interrupt lives and cannot be argued with. Keeping the decision separate from
the driving is what lets the whole policy be tested against a simulated flat
with no robot attached (scripts/test_explore.py).

THE LOOP IS: decide -> drive a couple of legs -> decide again. Not: plan a
route and follow it to the end. The map changes as it is driven through, and a
route computed from a five-second-old map is wrong exactly where it matters,
at the far end -- which is also the part that was planned through cells nobody
had looked at yet.

⚠ "NO FRONTIER LEFT" HAS TO MEAN NO FRONTIER LEFT. The first version of this
gave up on a frontier permanently whenever two legs of driving towards it
revealed less than a few hundredths of a square metre -- which is the NORMAL
experience of heading for a doorway, because nothing behind it is visible
until you are standing in it. The run then announced that the flat was fully
mapped, from the corridor, with thirty-five real frontiers outstanding. So
patience here is per-cycle and RESETS ON PROGRESS, and `done` is only ever
reported after checking every frontier rather than the filtered list.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _field

import numpy as np

import config
from slam.frontier import Frontier, describe, find
from slam.mapper import Mapper
from slam.plan import DistanceField, path_to_moves

# How many of the ranked frontiers to try planning to before giving the cycle
# up. The best one is sometimes the cell the robot is standing on, and the
# second is sometimes behind a chair leg the inflation has sealed.
_CANDIDATES = 12


@dataclass
class Decision:
    """What to do next, and something short to say about it."""

    done: bool = False
    reason: str = ""
    say: str = ""
    target: Frontier | None = None
    moves: list = _field(default_factory=list)
    path: np.ndarray | None = None
    field: DistanceField | None = None


class Explorer:
    """Frontier exploration, one decision at a time.

    Usage is a loop: `decide`, execute the moves it returns, then `note` what
    the map did while you were driving so it knows whether that was progress.
    """

    def __init__(self, legs: int = 2, stall_limit: int = 6) -> None:
        self.legs = legs
        self.stall_limit = stall_limit
        self.unreachable: set = set()
        self.stalled = 0
        self.visited: list = []

    def decide(self, mapper: Mapper, now: float | None = None) -> Decision:
        """Pick the next frontier and the route to it."""
        field = DistanceField(mapper.grid, mapper.pose[:2],
                              extra_blocked=mapper.unmapped_obstacles(now))

        options = find(mapper.grid, field, ignore=self.unreachable)
        if not options:
            # Only genuinely finished if nothing is left even counting the
            # ones this run has given up on.
            everything = find(mapper.grid, field)
            if not everything:
                return Decision(done=True, reason="no frontiers left",
                                say="that is everywhere I can get to, mapped",
                                field=field)
            return Decision(done=True, reason="nothing reachable left",
                            say="I have got as far as I can from here",
                            field=field)

        for candidate in options[:_CANDIDATES]:
            route = field.path_to(candidate.approach)
            if len(route) >= 2:
                return Decision(reason="heading for a frontier",
                                say=describe(candidate, mapper.pose),
                                target=candidate,
                                moves=path_to_moves(route, mapper.pose[2]
                                                    )[:self.legs],
                                path=route, field=field)
            # Standing on it, or sealed off behind inflation. Not permanent:
            # note() clears this the moment anything is learned.
            self.unreachable.add(candidate.key())

        self.stalled += 1
        return Decision(done=self.stalled >= 3,
                        reason="nothing plannable this cycle",
                        say="", field=field)

    def note(self, mapper: Mapper, gained_m2: float) -> None:
        """Tell the policy how much the map grew while the moves were driven.

        Progress clears every frontier this run had written off. A frontier
        that was unreachable from the corridor is very often reachable from
        two metres further along it, and a policy that cannot change its mind
        about that stops exploring long before the flat is mapped.
        """
        if gained_m2 >= 0.05:
            self.unreachable.clear()
            self.stalled = 0
        else:
            self.stalled += 1

    @property
    def stuck(self) -> bool:
        return self.stalled >= self.stall_limit

    def status(self) -> dict:
        return {"stalled": self.stalled, "written_off": len(self.unreachable),
                "stuck": self.stuck}
