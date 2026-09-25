"""trip_tracker.py — where she is during "explore and come back", on the saved map.

    Mission(ListeningRobot(robot, tracker)).run(tracker.explorer_mapper(), ...)
    await tracker.drain()
    tracker.best()  ->  (mapper, start, certain)   # drive home on this

WHY THIS EXISTS. The explorer (slam/mission.py) keeps a map of its own, and
that is the right map for choosing where to explore. It is not a safe map to
drive home on: when a turn happens, it tells its matcher the COMMANDED angle.
It does check that the wheels turned at all (the sweep before against the sweep
after), but a turn that happened only halfway is still reported as a full one,
and the matcher only searches 20% of the claim around it. The truth sits
outside that window, and after that the explorer's pose is wrong without
anything saying so. On 2026-09-24 23:55 the first sweep of the way home did
not match at all. In simulation on the real map, with some turns slipping
halfway, the explorer's own pose ended 1.3-2.5 m and 65-85 degrees out; this
tracker, fed the same sweeps, 5-12 cm.

The explorer is not changed to fix that. Harsh, 2026-09-25: exploring stays as
it is, the saved map is used, and only the coming back is added. So this
LISTENS instead. Every turn, drive and sweep the explorer makes passes through
`ListeningRobot` unchanged, and the tracker folds the sweeps with the prior the
follower uses, `nav_follower.motion_fraction`. That is the map's own answer to
"did that move happen in full, halfway, or not at all?" (60/60 on the real map).
It never commands anything itself.

⚠ IT MUST NOT SLOW THE EXPLORER. A fold costs 0.4 s on the Pi, 1.4 s after a
move (three matches per look-alike), measured 2026-09-25. Done inline, every
sweep the explorer asked for would come back that much later, and a two-minute
trip would explore a third less. So sweeps are handed to ONE worker thread, in
order, and the explorer gets its sweep straight back; `drain()` waits for the
worker before anyone drives on the answer.

TWO SPOTS THAT LOOK ALIKE ARE BOTH FOLLOWED. From parts of the hostel room the
whole-map search cannot tell two places apart (2026-09-25: 1.10 against 1.07,
3.4 m and 84 degrees apart), and the tracker's "matched" is no help: it said
matched and trusted at both. Harsh: "it just sees, it should explore". So she
does not refuse and does not shuffle about first. One tracker starts at each
candidate, and every sweep taken AFTER A MOVE is scored against the saved map
from each one, counting every beam (the fair score place_by_moving uses).
Walking about is what tells look-alikes apart, and exploring IS walking about.

⚠ ONLY SWEEPS AFTER A MOVE ARE EVIDENCE. Standing still, ten sweeps are the
same look ten times; summing them turned one small difference into a "clear
lead" with no new information at all (seen on the real sweeps, 2026-09-25).

For coming back to the start, the frame does not even have to be the right one:
each tracker drives back to ITS OWN start, so a room that genuinely looks the
same from both still brings her back to where she really set off. The certainty
matters for the map. `certain` is False unless the start was clear or the trip
settled it, and the caller must not grow the saved map or save home from an
uncertain trip.
"""

from __future__ import annotations

import copy
import logging
import math
import queue
import threading

import config
import nav_follower

log = logging.getLogger("friday.trip_tracker")

# The trip settles a look-alike start only on a clear lead in summed fair
# score, AND only when the most recent evidence agrees with it. The same rule
# as place_by_moving, gathered from the explorer's moves instead of three
# dedicated ones.
LEAD_MARGIN = 0.20
# How many of the latest after-a-move sweeps must ALL favour the leader --
# and there must be at least this many. One step alone picked wrong 2/20.
AGREE_LAST = 3


class _Hypothesis:
    """One place she might have started from, followed on its own map copy."""

    def __init__(self, grid, pose, name: str) -> None:
        from slam.mapper import Mapper

        self.name = name
        self.mapper = Mapper(grid=copy.deepcopy(grid),
                             pose=tuple(float(v) for v in pose))
        self.start: tuple[float, float, float] | None = None
        self.total = 0.0
        self.recent: list[float] = []
        self.unmatched = 0


class TripTracker:
    """Follows her on the saved map from every candidate start. Moves nothing."""

    def __init__(self, grid, hypotheses, certain: bool) -> None:
        from slam.core import LikelihoodField

        # The SAVED map, as it was when the trip started, for judging the
        # candidates fairly. Their own maps grow as they see, and a
        # hypothesis must not be judged against walls it drew itself.
        field = LikelihoodField.from_grid(grid)
        self._fair = copy.copy(field)
        self._fair.known = None
        self._grid = grid
        names = ("first", "second", "third")
        self.hyps = [_Hypothesis(grid, h, names[i])
                     for i, h in enumerate(hypotheses[:3])]
        self.started_certain = bool(certain) and len(self.hyps) == 1
        self._moved = 0.0
        self._turned = 0.0
        self._any_move = False
        self._still_folded = 0
        self.sweeps = 0
        self.skipped = 0
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None

    def explorer_mapper(self):
        """The explorer's own map: a copy of the saved one, at the best guess.

        Exactly what it was given before the tracker existed -- its choices
        of where to explore are its own business and are not changed here.
        """
        from slam.mapper import Mapper

        return Mapper(grid=copy.deepcopy(self._grid),
                      pose=tuple(float(v) for v in self.hyps[0].mapper.pose))

    # --- what the explorer did (event loop side; never blocks) ------------

    def commanded(self, moved_m: float = 0.0, turned_rad: float = 0.0) -> None:
        """A move went out. It is the prior for the NEXT sweep, checked."""
        if not self._any_move:
            # The start is where she was when the first move went out, after
            # the explorer's standing-still sweeps settled it: the worker
            # must finish those before the mark is read.
            self._queue.put(("mark", None, 0.0, 0.0))
            self._ensure_worker()
        self._any_move = True
        self._moved += float(moved_m)
        self._turned += float(turned_rad)

    def note_sweep(self, scan: dict[int, int]) -> None:
        """A sweep the explorer took. Queued for the worker; returns at once.

        Every sweep after a move is kept. Standing still, the settling sweeps
        before the first move are kept (they place the start), and after that
        one per pause -- the rest are the same look again and cost 0.4 s each.
        """
        if not scan:
            return
        moved, turned = self._moved, self._turned
        self._moved = self._turned = 0.0
        if moved or turned:
            self._still_folded = 0
        elif self._any_move:
            if self._still_folded >= 1:
                self.skipped += 1
                return
            self._still_folded += 1
        self._queue.put(("sweep", scan, moved, turned))
        self._ensure_worker()

    async def drain(self) -> None:
        """Wait until every queued sweep has been folded."""
        import anyio.to_thread

        await anyio.to_thread.run_sync(self._queue.join)

    def close(self) -> None:
        """Let the worker go once it has finished what is queued."""
        if self._worker is not None and self._worker.is_alive():
            self._queue.put(("stop", None, 0.0, 0.0))

    # --- the worker thread ------------------------------------------------

    def _ensure_worker(self) -> None:
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._work, daemon=True,
                                            name="trip-tracker")
            self._worker.start()

    def _work(self) -> None:
        # It leaves only when told to (close). Leaving on a quiet spell
        # could race a sweep queued at that instant, and drain() would then
        # wait for ever on a sweep nobody is left to fold.
        while True:
            kind, scan, moved, turned = self._queue.get()
            try:
                if kind == "stop":
                    return
                if kind == "mark":
                    for h in self.hyps:
                        if h.start is None:
                            h.start = tuple(float(v) for v in h.mapper.pose)
                else:
                    self.fold(scan, moved, turned)
            except Exception as exc:     # noqa: BLE001
                # Listening must never be what ends a run that is otherwise
                # going fine; one lost sweep is a smaller hole than a crash.
                log.warning("trip tracker could not fold a sweep: %s", exc)
            finally:
                self._queue.task_done()

    def fold(self, scan: dict[int, int], moved: float = 0.0,
             turned: float = 0.0) -> None:
        """Fold one sweep into every hypothesis. Blocking."""
        from slam.core import points_from_scan, transform

        self.sweeps += 1
        evidence = bool(moved or turned)
        for h in self.hyps:
            fraction = 1.0
            if evidence:
                fraction = nav_follower.motion_fraction(h.mapper, scan, moved,
                                                        turned)
            info = h.mapper.update(scan, moved * fraction, turned * fraction)
            if not h.mapper.pose_trusted:
                h.unmatched += 1
                log.info("trip tracker (%s): sweep did not match (%s)",
                         h.name, info.get("reason", ""))
            if evidence and len(self.hyps) > 1:
                local = points_from_scan(scan,
                                         forward_deg=h.mapper.forward_deg)
                pose = h.mapper.pose
                score = float(self._fair.score(
                    transform(local, pose[0], pose[1], pose[2])))
                h.total += score
                h.recent = (h.recent + [score])[-AGREE_LAST:]

    # --- the answer (after drain) -----------------------------------------

    def ranked(self) -> list[_Hypothesis]:
        # Stable: with no evidence yet, the search's own order stands.
        return sorted(self.hyps, key=lambda h: h.total, reverse=True)

    def settled(self) -> bool:
        """Is the start known: clear from the first look, or made clear since?"""
        if self.started_certain:
            return True
        if len(self.hyps) < 2:
            return False
        lead, second = self.ranked()[:2]
        if len(lead.recent) < AGREE_LAST:
            return False
        agrees = all(a > b for a, b in zip(lead.recent, second.recent))
        return lead.total - second.total >= LEAD_MARGIN and agrees

    def best(self):
        """(mapper, start, certain): the map to drive home on, and the mark."""
        lead = self.ranked()[0]
        start = lead.start or tuple(float(v) for v in lead.mapper.pose)
        return lead.mapper, start, self.settled()

    def as_dict(self) -> dict:
        return {
            "sweeps": self.sweeps,
            "skipped": self.skipped,
            "certain": self.settled(),
            "hypotheses": [{
                "name": h.name,
                "start": ([round(v, 2) for v in h.start] if h.start else None),
                "pose": [round(float(v), 2) for v in h.mapper.pose],
                "total": round(h.total, 2),
                "unmatched": h.unmatched,
            } for h in self.ranked()],
        }


def start_hypotheses(grid, scan) -> tuple[list, bool, str]:
    """Where she might be standing, from one look: (poses, certain, why).

    One pose when the whole-map search is sure; the two it could not choose
    between when it is not; none when nothing on the map fits at all.
    """
    import global_localise
    import path_planner

    try:
        plannable = path_planner.preprocess(grid).plannable
    except Exception:                            # noqa: BLE001
        plannable = None
    fix = global_localise.localise(grid, scan, restrict_to=plannable)
    if fix.pose is not None:
        return [fix.pose], True, ""
    if fix.candidate is not None and fix.runner_up is not None:
        return [fix.candidate, fix.runner_up], False, fix.why
    return [], False, fix.why


class ListeningRobot:
    """The explorer's robot, untouched, with the trip tracker listening in.

    Every call goes to the real robot exactly as before and returns exactly
    what it returned, as soon as it returned it. The only addition: a move
    that went out is noted, and a sweep that came back is queued for the
    tracker.
    """

    def __init__(self, inner, tracker) -> None:
        self._inner = inner
        self._tracker = tracker

    def __getattr__(self, name: str):
        return getattr(self._inner, name)

    async def turn(self, degrees: float) -> str:
        said = await self._inner.turn(degrees)
        if "REFUSED" not in (said or "").upper():
            self._tracker.commanded(turned_rad=math.radians(float(degrees)))
        return said

    async def forward(self, cm: float) -> str:
        said = await self._inner.forward(cm)
        if "REFUSED" not in (said or "").upper():
            self._tracker.commanded(moved_m=float(cm) / 100.0)
        return said

    async def sweep(self) -> dict[int, int]:
        scan = await self._inner.sweep()
        try:
            self._tracker.note_sweep(scan)
        except Exception as exc:                 # noqa: BLE001
            log.warning("trip tracker could not take a sweep: %s", exc)
        return scan
