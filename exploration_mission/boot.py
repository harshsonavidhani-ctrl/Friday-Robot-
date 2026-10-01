"""Coming back after a reboot or a power cut (File 21 step 9, spec 4.14).

Friday's crashes are power cuts, so this is not an edge case: the mission's
areas, its start anchor and its last checkpoint are on disk (area_store), and
this module is what reads them back and decides where she is -- WITHOUT
DRIVING. Nothing here starts a run, holds one, or moves a wheel.

    loaded = load(store, env_id)                 # 1. areas, anchor, checkpoint
    sweeps = wait_for_lidar(adapters, cfg)       # 2. the LiDAR has to be up
    placed = relocalise(sweeps, loaded, adapters.slam, cfg)     # 4. the chain

THE CHAIN stops at the first success:
    a  HOME CHECK     the view now against the home anchor's signature
    b  CHECKPOINT     File 14's global search agrees (0.5 m) with where the
                      last checkpoint left her
    c  GLOBAL         File 14's global search is sure, somewhere else
    d  none of them   not placed

⚠ "NOT PLACED" DOES NOT REFUSE A MISSION HERE, and that is a deliberate
departure from the spec's (d). On this robot a mission runs in its own start
frame and needs no saved-map pose, and Harsh's standing rule (2026-09-25) is
that two look-alike spots must never stop her exploring. Not placed means
only that areas named on the next mission stay in that mission's frame and
are not File 14 landmarks. What DOES refuse "explore" is having no LiDAR
sweeps at all (`localized = False`): then she honestly cannot mark a start.

(b) and (c) are one call: File 14's global search is the expensive part
(a second or two of CPU), so it runs once and its answer is read both ways.
"""

from __future__ import annotations

import logging
import math

from exploration_mission import area_store, mission_map, rth_ladder

log = logging.getLogger("friday.exploration_mission.boot")

CHECKPOINT_AGREE_M = 0.5
# A checkpoint in one of these states was a mission in progress.
LIVE_STATES = frozenset({"MISSION_INIT", "EXPLORE", "ASK_LABEL", "TARGET_NAV", "PAUSED",
                         "SUMMARIZE", "RETURN_TO_START", "HELP"})
RETURNING_STATES = frozenset({"SUMMARIZE", "RETURN_TO_START", "HELP"})


def load(store: area_store.AreaStore, env_id: str) -> dict:
    """Step 1. Never raises: a corrupt or too-new checkpoint is logged and
    treated as no checkpoint (the .bak fallback is inside the store)."""
    areas = store.list_areas(env_id)
    anchor = store.home_anchor(env_id)
    try:
        checkpoint, source = store.load_checkpoint(env_id)
    except area_store.StoreError as exc:
        log.error("checkpoint not usable: %s", exc)
        checkpoint, source = None, None
    log.info("boot: %d area(s) (%d unnamed, %d stale), home anchor %s, checkpoint %s",
             len(areas), sum(a["status"] == "unnamed" for a in areas),
             sum(a["status"] == "stale" for a in areas),
             "yes" if anchor else "no",
             f"{source}: {checkpoint.get('state')}" if checkpoint else "none")
    return {"areas": areas, "anchor": anchor, "checkpoint": checkpoint,
            "checkpoint_source": source}


def interrupted_mission(store: area_store.AreaStore, checkpoint: dict | None
                        ) -> dict | None:
    """The mission a power cut interrupted, if the checkpoint shows one."""
    if not checkpoint or checkpoint.get("state") not in LIVE_STATES:
        return None
    mission = store.mission(checkpoint.get("mission_id") or "")
    if mission is None or mission["status"] not in ("running", "interrupted"):
        return None
    return mission


def wait_for_lidar(adapters, want: int, wait_s: float) -> list[dict]:
    """Step 2. Up to `wait_s` for the LiDAR to give `want` sweeps. Standing
    still; returns whatever it got (possibly nothing)."""
    sweeps: list[dict] = []
    waited = 0.0
    while len(sweeps) < want and waited <= wait_s:
        sweep = None
        try:
            sweep = adapters.lidar.sweep()
        except Exception as exc:                        # noqa: BLE001
            log.debug("no sweep yet: %s", exc)
        if sweep:
            sweeps.append(sweep)
            pause = 0.1
        else:
            pause = 1.0
        adapters.clock.sleep(pause)
        waited += pause
    return sweeps


def home_check(sweeps: list[dict], anchor: dict | None, cfg) -> dict:
    """(a): does it look like the home anchor from here?"""
    if not sweeps or not anchor or anchor.get("signature") is None:
        return {"verified": False, "why": "no anchor" if sweeps else "no sweeps"}
    sig = cfg.signature
    return rth_ladder.verify(sweeps, anchor["signature"], sig.max_range_m,
                             sig.match_mad_m, sig.min_overlap)


def checkpoint_pose_on_map(checkpoint: dict | None):
    """Where the last checkpoint left her, in the saved map's frame -- only
    known if that mission's start had been placed on the saved map."""
    if not checkpoint or not checkpoint.get("anchor") or not checkpoint.get("pose"):
        return None
    return mission_map.compose(tuple(checkpoint["anchor"]), tuple(checkpoint["pose"]))


def relocalise(sweeps: list[dict], loaded: dict, slam, cfg) -> dict:
    """Step 4, the chain. No motion in any branch."""
    if not sweeps:
        return {"placed": None, "how": "no sweeps", "localized": False}
    look = home_check(sweeps, loaded.get("anchor"), cfg)
    if look.get("verified"):
        a = loaded["anchor"]
        heading = a["theta"] + math.radians(look.get("shift_deg", 0.0))
        return {"placed": "home", "how": "the view matches the home anchor",
                "localized": True, "pose": (a["x"], a["y"], heading),
                "frame": a.get("frame"), "look": look}
    fix = None
    try:
        fix = slam.localise(sweeps[-1])
    except Exception as exc:                            # noqa: BLE001
        log.warning("global localise failed: %s", exc)
    if fix is not None:
        was = checkpoint_pose_on_map(loaded.get("checkpoint"))
        if was is not None and math.hypot(fix[0] - was[0],
                                          fix[1] - was[1]) <= CHECKPOINT_AGREE_M:
            return {"placed": "checkpoint", "localized": True, "pose": tuple(fix),
                    "how": "File 14's search agrees with the last checkpoint",
                    "frame": area_store.FRAME_MAP, "look": look}
        return {"placed": "global", "localized": True, "pose": tuple(fix),
                "how": "File 14's global search is sure",
                "frame": area_store.FRAME_MAP, "look": look}
    return {"placed": None, "localized": True, "look": look,
            "how": "the view is not home and File 14's search is not sure"}
