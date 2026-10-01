"""The mission manager as a thread inside friday-mcp (File 21 step 9, spec 4.14.6).

    # mcp_server/server.py, beside the scene and environment watchers:
    from exploration_mission import service
    service.start()

No systemd unit, no new process: friday-mcp is the process that owns the
LiDAR and the motors and that File 14's engine runs in, so the mission lives
there as one daemon thread (plus its worker), the way environment_watch does.

⚠ `config.EXPLORATION_MISSION_SERVICE` SHIPS False. Turning it on starts a
thread in the process that holds the wheels and adds three tables to
memory.db (after a backup). Nothing here drives by itself, but it has not yet
run on the robot, so -- like AGENCY_MOTION and PHYSICAL_CURIOSITY -- switching
it on is Harsh's call, together with the voice hook (step 11).

start() never raises into the server: a mission that cannot start costs the
mission, never friday-mcp.

STEP 11 -- what the MCP tools (mcp_server/tools/mission_tools.py) call:

    voice(text, age_s)      every transcript, from main.handle(), before the
                            brain: intents.Router decides; True = handled
    answer(ask_id, text)    the words for a question she asked (MissionWatch)
    status()                the dashboard's panel
    start_mission(minutes)  the dashboard's Explore button
    name_area(id, name)     the dashboard's ONE write: name / rename an area

With the switch off (no manager) every one of them says so and does nothing:
voice() returns False, so speech takes exactly today's path.
"""

from __future__ import annotations

import logging
import logging.handlers
import threading

log = logging.getLogger("friday.exploration_mission")

_manager = None
_thread: threading.Thread | None = None
_stop = threading.Event()
_log_handler = None
_router = None
_asker = None
_voice_lock = threading.Lock()
# What would set her driving: refused when the order is stale, like main.py's
# COMMAND_FRESH_TOOLS. Stopping, pausing and answering never are.
MOTION_ACTIONS = ("start", "resume", "finish", "go_to")


def enabled() -> bool:
    import config
    return bool(getattr(config, "EXPLORATION_MISSION_SERVICE", False))


def _file_log() -> None:
    """logs/exploration_mission.log, rotating (spec 4.16): every transition,
    save, rung, question + answer, and the frontier / segmentation timings."""
    global _log_handler
    if _log_handler is not None:
        return
    import config
    folder = config.ROOT / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    _log_handler = logging.handlers.RotatingFileHandler(
        folder / "exploration_mission.log", maxBytes=1_000_000, backupCount=3)
    _log_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    log.addHandler(_log_handler)
    log.setLevel(logging.INFO)


def build(adapters=None, store=None, cfg=None, runner=None):
    """A manager on the real adapters (or the ones handed in, for tests)."""
    from exploration_mission import adapters as ad
    from exploration_mission import area_store
    from exploration_mission import config as mcfg
    from exploration_mission import mission_manager
    global _asker
    if adapters is None:
        adapters = ad.Adapters()
        # Inside friday-mcp the LiDAR is ours to read (never a second reader).
        adapters.lidar = ad.LidarAdapter(run=adapters.run, in_server=True)
        # ...and the ears are friday-brain's, through the voice link.
        from exploration_mission import voice_link
        _asker = voice_link.Asker()
        adapters.voice = ad.VoiceAdapter(asker=_asker)
    if store is None:
        store = area_store.AreaStore(places=adapters.places)
    store.ensure_schema()
    return mission_manager.MissionManager(adapters, store, cfg or mcfg.load(),
                                          runner=runner)


def start(adapters=None, store=None, cfg=None, runner=None, force: bool = False):
    """Start the mission thread (once). Returns the manager, or None if it is
    switched off or could not start."""
    global _manager, _thread
    if not force and not enabled():
        log.info("exploration mission off (EXPLORATION_MISSION_SERVICE = False)")
        return None
    if _thread is not None and _thread.is_alive():
        return _manager
    try:
        if adapters is None:
            _file_log()
        _manager = build(adapters, store, cfg, runner)
        _manager.boot()
        _stop.clear()
        _thread = threading.Thread(target=_manager.run_forever, args=(_stop,),
                                   name="friday-mission", daemon=True)
        _thread.start()
        log.info("exploration mission started (boot restore running)")
        return _manager
    except Exception as exc:                            # noqa: BLE001
        log.error("exploration mission did not start: %s: %s",
                  type(exc).__name__, exc)
        _manager = None
        return None


def get():
    """The running manager, or None."""
    return _manager


# --- step 11: the voice hook and the dashboard -------------------------------------

class _AgeGuard:
    """The manager, as the router's `actions`, refusing to set her driving on
    an order that is too old (main.py's COMMAND_MAX_AGE_S rule)."""

    def __init__(self, mgr, age_s: float, limit_s: float):
        self._mgr, self._age, self._limit = mgr, float(age_s), float(limit_s)
        self.refused: str | None = None

    def __getattr__(self, name):
        attr = getattr(self._mgr, name)
        if name in MOTION_ACTIONS and self._limit and self._age > self._limit:
            def refuse(*_a, **_k):
                self.refused = name
                log.warning("REFUSED mission %s: the order is %.1f s old (limit %g s)",
                            name, self._age, self._limit)
                self._mgr.say("stale_order")
            return refuse
        return attr


def router():
    """One router for the life of the manager (it keeps its own yes/no
    question, "Should I forget X?", between two transcripts)."""
    global _router
    mgr = _manager
    if mgr is None:
        return None
    if _router is None or _router.actions is None:
        from exploration_mission import intents
        _router = intents.Router(
            None,
            resolve_area=lambda name: mgr.store.by_name(mgr.a.env.env_id(), name),
            resolve_place=mgr.a.places.find_exact,
            monotonic=mgr.a.clock.monotonic)
    return _router


def voice(text: str, age_s: float = 0.0, limit_s: float | None = None) -> dict:
    """One transcript. {"handled": bool, ...}. Never raises: an error is "not
    handled", and the existing pipeline carries on."""
    mgr = _manager
    if mgr is None:
        return {"handled": False, "why": "mission off"}
    if limit_s is None:
        import config
        limit_s = float(getattr(config, "COMMAND_MAX_AGE_S", 0.0) or 0.0)
    try:
        with _voice_lock:
            r = router()
            guard = _AgeGuard(mgr, age_s, limit_s)
            r.actions = guard
            try:
                handled = bool(r.handle(text, mgr.context()))
            finally:
                r.actions = mgr
        if handled:
            log.info("voice: %r handled by the mission (state %s%s)", text,
                     mgr.state, f", REFUSED {guard.refused}: stale" if guard.refused
                     else "")
        return {"handled": handled, "state": mgr.state, "refused": guard.refused}
    except Exception as exc:                            # noqa: BLE001
        log.error("mission voice failed on %r: %s: %s", text, type(exc).__name__, exc)
        return {"handled": False, "why": f"error: {type(exc).__name__}"}


def answer(ask_id: str, text: str | None) -> dict:
    if _asker is None:
        return {"ok": False, "why": "mission off"}
    ok = _asker.answer(ask_id, text)
    log.info("answer to %s: %r (%s)", ask_id, text,
             "taken" if ok else "nobody was waiting")
    return {"ok": ok}


def status(areas: bool = True) -> dict:
    """Everything the dashboard's panel shows (spec 4.16), read-only."""
    mgr = _manager
    if mgr is None:
        return {"enabled": enabled(), "running": False}
    from exploration_mission import dashboard_panels
    return dashboard_panels.panel(mgr, areas=areas)


def start_mission(minutes: float | None = None) -> dict:
    """The dashboard's Explore button: a NEW mission, as "explore" would."""
    mgr = _manager
    if mgr is None:
        return {"ok": False, "why": "mission off"}
    if mgr.state != "IDLE" or mgr.mission is not None:
        return {"ok": False, "why": f"busy ({mgr.state})"}
    if not mgr.localized:
        return {"ok": False, "why": "no LiDAR sweeps"}
    mgr.start(minutes, True)
    return {"ok": True, "state": "MISSION_INIT"}


def name_area(area_id: str, name: str) -> dict:
    """The dashboard's ONE write action (spec 4.16): name or rename an area,
    through area_store, with the same landmark sync as the voice."""
    mgr = _manager
    if mgr is None:
        return {"ok": False, "why": "mission off"}
    from exploration_mission import area_store
    try:
        area = mgr.store.rename(str(area_id), str(name), source="dashboard")
    except area_store.DuplicateName as exc:
        return {"ok": False, "why": f"{exc.name!r} is already somewhere else",
                "suggestion": exc.suggestion}
    except area_store.StoreError as exc:
        return {"ok": False, "why": str(exc)}
    log.info("dashboard named area %s -> %s", area_id, area["name"])
    return {"ok": True, "area": {"id": area["id"], "name": area["name"],
                                 "display_name": area["display_name"],
                                 "status": area["status"]}}


def stop() -> None:
    global _manager, _thread, _router
    _router = None
    _stop.set()
    if _manager is not None:
        _manager.analysis.stop_thread()
    if _thread is not None:
        _thread.join(3.0)
    _manager, _thread = None, None
