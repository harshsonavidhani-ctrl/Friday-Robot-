"""The simulated world the mission state-machine tests run in (File 21).

    w = World(["save it as hall", "kitchen"])       # scripted voice replies
    w.mgr.start(); w.run()                           # tick until IDLE again

Real: the slam grid, File 14's despeckle, frontiers, rooms, the area store
(temp database), the dialog, the speech lines. Fake: the clock, the voice, the
RL run (simworld.SimRun), File 14's engine (adapters.FakeNav).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from exploration_mission import adapters as ad
from exploration_mission import area_store as st
from exploration_mission import config as mcfg
from exploration_mission import mission_manager as M
import make_maps as mm
import simworld as sw

TMP = Path(tempfile.mkdtemp(prefix="friday-file21-world-"))
_n = [0]


class DeferredRunner:
    """Jobs that finish only when the test says so (a dialog 'in progress')."""

    def __init__(self):
        self.waiting: list = []

    def start(self, fn, name: str = ""):
        job = M.Job()
        self.waiting.append((job, fn))
        return job

    def finish_all(self) -> None:
        waiting, self.waiting = self.waiting, []
        for job, fn in waiting:
            job._run(fn)


class World:
    def __init__(self, replies=(), world=None, path=None, cfg=None, runner=None,
                 nav=None, anchor=None, matched_within_m=None, policy="matched",
                 sweeps=True, fail_return_at_m=None):
        _n[0] += 1
        d = TMP / f"w{_n[0]}"
        self.clock = ad.FakeClock()
        self.sim = sw.SimRun(self.clock, world or mm.three_rooms(),
                             path or sw.THREE_ROOM_TOUR,
                             matched_within_m=matched_within_m,
                             fail_return_at_m=fail_return_at_m)
        self.voice = ad.FakeVoice(list(replies))
        slam = ad.SlamAdapter(map_file=d / "none.npz", session_file=d / "s.json")
        slam.map_id = lambda read_only=False: "map-test"
        slam.localise = lambda scan: anchor
        self.a = ad.fake_adapters(
            clock=self.clock, run=self.sim, voice=self.voice, slam=slam,
            lidar=self.sim.lidar() if sweeps else ad.FakeLidar(),
            nav=nav or ad.FakeNav(during_run=False, clock=self.clock,
                                  engine=False))
        self.store = st.AreaStore(db_path=d / "m.db", data_dir=d / "data",
                                  places=self.a.places)
        self.store.ensure_schema()
        self.mgr = M.MissionManager(self.a, self.store, mcfg.load(cfg or {}),
                                    runner=runner or M.InlineRunner(),
                                    inline_analysis=True, map_policy=policy)
        self.states: list[tuple[float, str]] = []
        self.t = 0.0
        self.explore_ms: list[float] = []

    def step(self) -> str:
        self.clock.advance(0.1)
        self.sim.step(0.1)
        before = self.mgr.state
        s = self.mgr.tick()
        self.t += 0.1
        if before == "EXPLORE" and s == "EXPLORE":
            self.explore_ms.append(self.mgr.tick_ms[-1])
        if not self.states or self.states[-1][1] != s:
            self.states.append((round(self.t, 1), s))
        return s

    def run(self, until=None, max_s: float = 400.0) -> str:
        """Tick until `until(self)` is true, or the mission is back in IDLE."""
        started = self.t
        while self.t - started < max_s:
            s = self.step()
            if until is not None:
                if until(self):
                    return s
            elif s == "IDLE" and self.t - started > 3.0:
                return s
        return self.mgr.state

    def seen(self) -> list[str]:
        return [s for _, s in self.states]

    def areas(self) -> list[dict]:
        return self.store.list_areas("home")

    def mission(self) -> dict:
        return self.store.missions("home")[0]

    def orders(self, *names) -> list:
        return [o for o in self.sim.orders if o[0] in names]
