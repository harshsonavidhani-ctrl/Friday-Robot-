"""Does the autonomous loop explore, and does it stop when the wheels lie?

    ./venv/bin/python scripts/test_mission.py

NO MOTORS AND NO HARDWARE. Mission talks to a small robot interface, so this
drives a simulated one through the same two-room flat scripts/test_explore.py
uses. What is being checked is not "does frontier exploration work" -- that
has its own suite -- but the loop around it: budgets, stopping, narration, and
above all whether a robot whose turns do not happen is caught or believed.

THAT LAST ONE IS THE POINT. On the real chassis a commanded pivot sometimes
does not break traction. Odometry still reports it, the scan matcher cannot
see the error because the truth is outside its search window, and every plan
after that is built on a heading that is a lie. Harsh caught it by watching
the robot; nothing in software noticed. A test that only ever drives a robot
whose wheels work would not notice either.
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import config                                                    # noqa: E402
from slam.mapper import Mapper                                   # noqa: E402
from slam.mission import Mission, scan_shift_mm                  # noqa: E402
from test_explore import START, TWO_ROOMS, clearance             # noqa: E402
from test_slam import drive, report, robot_sweep                 # noqa: E402


class FakeRobot:
    """A robot in the simulated flat. Optionally one with dead wheels."""

    def __init__(self, seed: int = 7, turns_work: bool = True,
                 stop_after: int | None = None) -> None:
        self.truth = START
        self.rng = np.random.default_rng(seed)
        self.turns_work = turns_work
        self.stop_after = stop_after
        self.calls = 0
        self.said: list[str] = []
        self.refusals = 0
        self.trips = 0
        self.trip_every = None      # trip the reflex every Nth forward
        self.forwards = 0

    async def sweep(self):
        return robot_sweep(self.truth, self.rng, room=TWO_ROOMS)

    async def observe(self):
        return None

    async def turn(self, degrees: float) -> str:
        self.calls += 1
        if self.turns_work:
            self.truth = drive(self.truth, 0.0, degrees)
            return f"right for 0.3s. Turned about {abs(degrees):.0f} degrees."
        # Energised, buzzing, going nowhere -- and still reporting success,
        # which is exactly what the real one does.
        return f"right for 0.3s. Turned about {abs(degrees):.0f} degrees."

    async def forward(self, cm: float) -> str:
        self.calls += 1
        ahead = drive(self.truth, cm / 100.0, 0.0)
        if clearance(ahead[:2]) < config.NAV_ROBOT_RADIUS_M:
            self.refusals += 1
            return "REFUSED: something is close ahead"
        self.forwards += 1
        if self.trip_every and self.forwards % self.trip_every == 0:
            # The reflex fired: she moved part of the way and was cut short.
            self.trips += 1
            self.truth = drive(self.truth, cm / 200.0, 0.0)
            return f"forward STOPPED after 0.4s — something is close."
        self.truth = ahead
        return f"forward for 1.0s. Travelled about {cm:.0f} cm."

    def say(self, text: str) -> None:
        self.said.append(text)

    def reflex_trips(self) -> int:
        return self.trips

    def stopped(self) -> bool:
        return (self.stop_after is not None
                and self.calls >= self.stop_after)


async def main() -> int:
    failures = 0

    print("an autonomous run in the two-room flat, wheels working\n")
    robot = FakeRobot()
    mapper = Mapper(pose=START, forward_deg=0.0)
    out = await Mission(robot).run(mapper, budget_m=6.0, budget_s=120.0)
    print(f"        {out['cycles']} cycles, drove {out['driven_m']} m, "
          f"mapped {out['floor_m2']} m2, stopped on {out['stopped_because']!r}")

    failures += not report("it actually drove",
                           out["driven_m"] > 1.0,
                           f"{out['driven_m']} m")
    failures += not report("it built a map while doing it",
                           out["floor_m2"] > 12.0,
                           f"{out['floor_m2']} m2 of floor")
    failures += not report("it respected the distance budget",
                           out["driven_m"] <= 6.0 + 0.5,
                           f"{out['driven_m']} m against a 6.0 m budget")
    failures += not report("it narrated, without chattering",
                           2 <= len(out["said"]) <= out["cycles"],
                           f"{len(out['said'])} lines over {out['cycles']} cycles")
    failures += not report("every turn was checked against the sweeps",
                           all(t["scan_shift_mm"] is not None
                               for t in out["turns"]) and out["turns"],
                           f"{len(out['turns'])} turns, all verified")

    print("\nthe same run, but the wheels do not turn")
    dead = FakeRobot(turns_work=False)
    mapper2 = Mapper(pose=START, forward_deg=0.0)
    out2 = await Mission(dead).run(mapper2, budget_m=6.0, budget_s=120.0)
    print(f"        stopped on {out2['stopped_because']!r} after "
          f"{out2['cycles']} cycles")
    failures += not report("a turn that did not happen stops the run",
                           "did not turn" in out2["stopped_because"],
                           out2["stopped_because"])
    failures += not report("and it stops EARLY, not after wandering blind",
                           out2["driven_m"] <= 2.0,
                           f"drove {out2['driven_m']} m before noticing")
    failures += not report("and it says so out loud",
                           any("turning" in s for s in out2["said"]),
                           f"{out2['said'][-1]!r}" if out2["said"] else "(silent)")

    print("\nthe first turn attempt stalls, the retry works")
    class Sticky(FakeRobot):
        def __init__(self):
            super().__init__(seed=5)
            self.attempts = 0
        async def turn(self, degrees):
            self.attempts += 1
            if self.attempts % 2 == 1:
                self.calls += 1
                return "right for 0.3s. Turned about 30 degrees."   # a lie
            return await FakeRobot.turn(self, degrees)
    sticky = Sticky()
    mapper5 = Mapper(pose=START, forward_deg=0.0)
    out5 = await Mission(sticky).run(mapper5, budget_m=3.0, budget_s=120.0)
    failures += not report("a stalled turn is retried once and recovered",
                           out5["driven_m"] > 0.5
                           and "did not turn" not in out5["stopped_because"],
                           f"drove {out5['driven_m']} m, stopped on "
                           f"{out5['stopped_because']!r}")
    failures += not report("and the retry is recorded, not hidden",
                           any(t.get("retried") for t in out5["turns"]),
                           f"{sum(1 for t in out5['turns'] if t.get('retried'))}"
                           f" of {len(out5['turns'])} turns retried")

    print("\nthe reflex keeps firing -- she should recover, not give up")
    tripping = FakeRobot(seed=3)
    tripping.trip_every = 2
    mapper4 = Mapper(pose=START, forward_deg=0.0)
    out4 = await Mission(tripping).run(mapper4, budget_m=6.0, budget_s=120.0)
    print(f"        {out4['cycles']} cycles, {out4['recoveries']} recoveries, "
          f"stopped on {out4['stopped_because']!r}")
    failures += not report("a safety trip is recovered from, not fatal",
                           out4["recoveries"] >= 1,
                           f"{out4['recoveries']} recoveries")
    failures += not report("but it gives up rather than loop forever",
                           out4["recoveries"] <= config.MISSION_MAX_RECOVERIES + 1,
                           f"capped at {config.MISSION_MAX_RECOVERIES}")
    failures += not report("and a trip is never reported as a human stop",
                           out4["stopped_because"] != "stop was called",
                           out4["stopped_because"])

    print("\nand stop() ends it wherever it is")
    halted = FakeRobot(stop_after=3)
    mapper3 = Mapper(pose=START, forward_deg=0.0)
    out3 = await Mission(halted).run(mapper3, budget_m=6.0, budget_s=120.0)
    failures += not report("stop is honoured",
                           out3["stopped_because"] == "stop was called",
                           out3["stopped_because"])

    print(f"\n{'all good' if not failures else f'{failures} failed'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
