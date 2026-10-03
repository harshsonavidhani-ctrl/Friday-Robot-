"""The exploration mission's settings (File 21, spec 4.17): load + validate.

THE NUMBERS LIVE IN FRIDAY'S ONE config.py, as `config.EXPLORATION_MISSION`
(CLAUDE.md PART 2: "all thresholds live in ONE config.py"), not in a YAML file
as the spec proposed. This module holds no defaults of its own -- only the
SHAPE each value must have -- so there is exactly one place a number can be
changed, and a typo there fails loudly at start-up instead of silently
becoming a default.

    cfg = load()                  # config.EXPLORATION_MISSION, validated
    cfg.explore.mission_cap_min   # attribute access, read-only
    cfg = load({"explore": {"mission_cap_min": 5}})   # overrides (tests, sim)

Unknown keys: a warning, then ignored (spec 4.17). Missing keys, wrong types
and out-of-range values: ConfigError, because a mission that starts on a bad
number is a robot driving on one.
"""

from __future__ import annotations

import copy
import logging
import re
from types import MappingProxyType
from typing import Any, Mapping

log = logging.getLogger("friday.exploration_mission.config")


class ConfigError(ValueError):
    pass


# (type, low, high) -- both bounds inclusive, None = unbounded. `float` accepts
# ints; nothing accepts a bool unless the type IS bool.
_F, _I, _S, _B, _L = float, int, str, bool, list

SCHEMA: dict[str, Any] = {
    "env_id": (_S, None, None),
    "loop_hz": (_F, 1.0, 50.0),
    "checkpoint_s": (_F, 1.0, 600.0),
    "events_cap": (_I, 10, 100000),
    "grid": {
        "free_below": (_F, 0.0, 1.0),
        "occupied_above": (_F, 0.0, 1.0),
        "seg_resolution_m": (_F, 0.01, 1.0),
        "robot_radius_m": (_F, 0.01, 1.0),
        "safety_margin_m": (_F, 0.0, 1.0),
    },
    "explore": {
        "mission_cap_min": (_F, 0.5, 240.0),
        "stagnation_m2": (_F, 0.0, 100.0),
        "stagnation_window_s": (_F, 1.0, 3600.0),
        "frontier_hz": (_F, 0.01, 5.0),
        "min_frontier_size_m": (_F, 0.0, 10.0),
        # Not in the spec: an unknown pocket smaller than this makes no
        # frontier (frontiers.py). 0 turns the rule off.
        "min_unknown_m2": (_F, 0.0, 10.0),
        "gain_scale": (_F, 0.0, 100.0),
        "potential_scale": (_F, 0.0, 100.0),
        "heading_scale": (_F, 0.0, 100.0),
        "revisit_scale": (_F, 0.0, 100.0),
        "progress_timeout_s": (_F, 1.0, 600.0),
        "progress_min_m": (_F, 0.0, 10.0),
        "blacklist_after_fails": (_I, 1, 100),
        "blacklist_radius_m": (_F, 0.0, 10.0),
        "done_after_empty_checks": (_I, 1, 100),
        "assist_arrive_m": (_F, 0.0, 5.0),
    },
    "segmentation": {
        "period_s": (_F, 0.5, 600.0),
        "door_half_width_m": (_F, 0.05, 2.0),
        "min_seed_m2": (_F, 0.0, 100.0),
        "min_room_m2": (_F, 0.0, 100.0),
        "polygon_epsilon_m": (_F, 0.001, 2.0),
        "max_polygon_vertices": (_I, 3, 1000),
        "same_area_iou": (_F, 0.0, 1.0),
        "grow_update_ratio": (_F, 1.0, 100.0),
    },
    "label": {
        "trigger": (_S, None, None),
        "min_area_m2": (_F, 0.0, 1000.0),
        "min_dwell_s": (_F, 0.0, 3600.0),
        "recent_exit_s": (_F, 0.0, 3600.0),
        "cooldown_s": (_F, 0.0, 3600.0),
        "distance_prompt_m": (_F, 0.0, 100.0),
        "reask_once": (_B, None, None),
        "max_name_words": (_I, 1, 20),
        "max_name_chars": (_I, 1, 200),
        "suggest_from_objects": (_B, None, None),
        "suggestion_min_conf": (_F, 0.0, 1.0),
        "suggestion_min_sightings": (_I, 1, 1000),
    },
    "voice": {
        "listen_s": (_F, 1.0, 60.0),
        "listen_after_tts_s": (_F, 0.0, 10.0),
        "whisper_language": (_S, None, None),
    },
    "signature": {
        "scans": (_I, 1, 200),
        "arrival_scans": (_I, 1, 200),
        "max_range_m": (_F, 0.5, 40.0),
        "match_mad_m": (_F, 0.0, 10.0),
        "min_overlap": (_F, 0.0, 1.0),
    },
    "breadcrumbs": {
        "spacing_m": (_F, 0.01, 10.0),
        "checkpoint_keep": (_I, 1, 100000),
        "loop_cut_m": (_F, 0.0, 10.0),
    },
    "return": {
        "arrive_tolerance_m": (_F, 0.01, 5.0),
        "arrive_hysteresis_m": (_F, 0.0, 5.0),
        "arrive_dwell_s": (_F, 0.0, 60.0),
        "wait_s": (_F, 0.0, 600.0),
        "backup_m": (_F, 0.0, 2.0),
        "backup_rear_clear_m": (_F, 0.0, 5.0),
        "ring_step_m": (_F, 0.005, 1.0),
        "help_retry_s": (_F, 1.0, 3600.0),
        "help_max_cycles": (_I, 1, 1000),
        "help_speak_every": (_I, 1, 1000),
        "mode": (_S, None, None),
    },
    "boot": {
        "slam_wait_s": (_F, 0.0, 3600.0),
        "home_check_period_s": (_F, 1.0, 3600.0),
    },
    "curiosity": {
        "mode": (_S, None, None),
        "threshold": (_F, 0.0, 1.0),
        "weights": (_L, None, None),
        "min_idle_min": (_F, 0.0, 10000.0),
        "min_quiet_conversation_min": (_F, 0.0, 10000.0),
        "quiet_hours": (_L, None, None),
        "no_cooldown_h": (_F, 0.0, 1000.0),
        "silence_cooldown_min": (_F, 0.0, 10000.0),
        "check_period_s": (_F, 1.0, 86400.0),
    },
    # Not in the spec: the seam with the RL run (rl/way_back.py publisher +
    # in-run hold, approved 2026-09-29).
    "run": {
        "live_max_age_s": (_F, 0.5, 60.0),
        "hold_ttl_s": (_F, 0.5, 30.0),
        "hold_refresh_s": (_F, 0.1, 30.0),
    },
}

_CHOICES = {
    ("label", "trigger"): {"segmentation", "distance", "both"},
    ("curiosity", "mode"): {"off", "ask", "auto"},
    ("return", "mode"): {"path", "map_first", "run_first"},
}
_HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class Section(Mapping):
    """Read-only nested settings with attribute access (`cfg.explore.x`)."""

    def __init__(self, data: dict[str, Any]) -> None:
        wrapped = {k: Section(v) if isinstance(v, dict) else
                   (tuple(v) if isinstance(v, list) else v)
                   for k, v in data.items()}
        object.__setattr__(self, "_data", MappingProxyType(wrapped))

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("mission settings are read-only; pass overrides "
                             "to load()")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def to_dict(self) -> dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Section) else
                    (list(v) if isinstance(v, tuple) else v))
                for k, v in self._data.items()}


def _merge(base: dict, extra: Mapping) -> dict:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _check_value(path: str, value: Any, rule: tuple) -> Any:
    kind, low, high = rule
    if kind is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path} must be true/false, got {value!r}")
        return value
    if isinstance(value, bool):
        raise ConfigError(f"{path} must be a {kind.__name__}, got a bool")
    if kind is float:
        if not isinstance(value, (int, float)):
            raise ConfigError(f"{path} must be a number, got {value!r}")
        value = float(value)
    elif kind is int:
        if not isinstance(value, int):
            raise ConfigError(f"{path} must be a whole number, got {value!r}")
    elif kind is str:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{path} must be a non-empty string, got {value!r}")
    elif kind is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path} must be a list, got {value!r}")
        return list(value)
    if low is not None and value < low:
        raise ConfigError(f"{path} = {value} is below {low}")
    if high is not None and value > high:
        raise ConfigError(f"{path} = {value} is above {high}")
    return value


def _validate(data: Mapping, schema: dict, prefix: str = "") -> dict:
    out: dict[str, Any] = {}
    for key in data:
        if key not in schema:
            log.warning("exploration mission config: unknown key %s%s ignored",
                        prefix, key)
    for key, rule in schema.items():
        path = f"{prefix}{key}"
        if key not in data:
            raise ConfigError(f"{path} is missing from config.EXPLORATION_MISSION")
        if isinstance(rule, dict):
            if not isinstance(data[key], Mapping):
                raise ConfigError(f"{path} must be a section")
            out[key] = _validate(data[key], rule, path + ".")
        else:
            out[key] = _check_value(path, data[key], rule)
    return out


def _cross_checks(d: dict) -> None:
    for (section, key), allowed in _CHOICES.items():
        if d[section][key] not in allowed:
            raise ConfigError(f"{section}.{key} = {d[section][key]!r}, "
                              f"expected one of {sorted(allowed)}")
    g = d["grid"]
    if not g["free_below"] < g["occupied_above"]:
        raise ConfigError("grid.free_below must be below grid.occupied_above")
    w = d["curiosity"]["weights"]
    if len(w) != 3 or any(isinstance(x, bool) or not isinstance(x, (int, float))
                          or x < 0 for x in w) or sum(w) <= 0:
        raise ConfigError("curiosity.weights must be three non-negative "
                          "numbers with a positive sum")
    q = d["curiosity"]["quiet_hours"]
    if len(q) != 2 or not all(isinstance(t, str) and _HHMM.match(t) for t in q):
        raise ConfigError('curiosity.quiet_hours must be ["HH:MM", "HH:MM"]')
    if d["signature"]["arrival_scans"] > d["signature"]["scans"]:
        raise ConfigError("signature.arrival_scans must not exceed signature.scans")
    r = d["run"]
    if r["hold_refresh_s"] >= r["hold_ttl_s"]:
        raise ConfigError("run.hold_refresh_s must be shorter than run.hold_ttl_s "
                          "or the hold lapses between refreshes")


def load(overrides: Mapping | None = None, base: Mapping | None = None) -> Section:
    """Validated settings. `base` defaults to config.EXPLORATION_MISSION."""
    if base is None:
        import config as friday_config
        base = getattr(friday_config, "EXPLORATION_MISSION", None)
        if base is None:
            raise ConfigError("config.EXPLORATION_MISSION is not defined")
    data = _merge(dict(base), overrides or {})
    checked = _validate(data, SCHEMA)
    _cross_checks(checked)
    return Section(checked)
