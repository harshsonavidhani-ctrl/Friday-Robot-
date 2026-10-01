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
    if adapters is None:
        adapters = ad.Adapters()
        # Inside friday-mcp the LiDAR is ours to read (never a second reader).
        adapters.lidar = ad.LidarAdapter(run=adapters.run, in_server=True)
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


def stop() -> None:
    global _manager, _thread
    _stop.set()
    if _manager is not None:
        _manager.analysis.stop_thread()
    if _thread is not None:
        _thread.join(3.0)
    _manager, _thread = None, None
