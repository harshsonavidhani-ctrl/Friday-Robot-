"""Friday's exploration map, drawn for the dashboard. READ-ONLY.

Harsh, 2026-09-13: "show the map live on the dashboard" -- and, the same day,
"dont make changes in the map". Nothing here writes to state/: the map, the
visit trace and the pose are opened with np.load / read_text and never saved.

What it draws is the map the exploration run loads (state/rl_map.npz, the
persistent WorldMap): free space, obstacles, unknown, the area she has driven,
and her position. The run SAVES that file only when it ends, so during a run
the picture is the map as of the last run; the position marker is the saved
pose. The metadata says which, so the dashboard never presents an old map as
live.

Rendering is cached on the files' mtimes, so a phone polling every few seconds
costs one stat() per request until something actually changes.
"""
from __future__ import annotations

import io
import json
import math
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAP_FILE = ROOT / "state" / "rl_map.npz"
VISITS_FILE = ROOT / "state" / "rl_map.visits.npz"
POSE_FILE = ROOT / "state" / "rl_map.pose.json"

PX_PER_CELL = 3
OCCUPIED_LOGODDS = 0.5
FREE_LOGODDS = -0.5

_lock = threading.Lock()
_cache: dict = {"key": None, "png": b"", "meta": {}}


def _stamp(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _render() -> tuple[bytes, dict]:
    import numpy as np
    from PIL import Image, ImageDraw

    with np.load(MAP_FILE, allow_pickle=False) as z:
        logodds = z["logodds"].copy()
        origin = z["origin"].astype(float).copy()
        cell = float(z["cell"])
        sweeps = int(z["scans"])
        updated = float(z["updated_at"])
    visits = np.zeros((0, 2), np.int64)
    vcell = 0.10
    try:
        with np.load(VISITS_FILE, allow_pickle=False) as v:
            visits = v["cells"].copy()
            vcell = float(v["cell_m"])
    except (OSError, KeyError, ValueError):
        pass
    pose = None
    try:
        p = json.loads(POSE_FILE.read_text())
        pose = (float(p["x"]), float(p["y"]), float(p["theta"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        pass

    nx, ny = logodds.shape
    occ = logodds > OCCUPIED_LOGODDS
    free = logodds < FREE_LOGODDS
    s = PX_PER_CELL
    rgb = np.full((ny, nx, 3), (46, 52, 64), np.uint8)       # unknown: dark slate
    rgb[free.T] = (226, 232, 240)                            # free: light
    rgb[occ.T] = (10, 12, 16)                                # obstacle: near-black
    rgb = rgb[::-1]                                          # +y up
    im = Image.fromarray(rgb).resize((nx * s, ny * s), Image.NEAREST)
    draw = ImageDraw.Draw(im, "RGBA")

    def px(x: float, y: float) -> tuple[float, float]:
        return ((x - origin[0]) / cell * s, (ny - (y - origin[1]) / cell) * s)

    for a, b in visits:
        x0, y1 = px(a * vcell, b * vcell)
        x1, y0 = px((a + 1) * vcell, (b + 1) * vcell)
        draw.rectangle([x0, y0, x1, y1], fill=(255, 106, 0, 95))
    for gx in range(math.ceil(origin[0]), math.floor(origin[0] + nx * cell) + 1):
        x, _ = px(gx, 0.0)
        draw.line([(x, 0), (x, ny * s)], fill=(77, 163, 255, 40))
    for gy in range(math.ceil(origin[1]), math.floor(origin[1] + ny * cell) + 1):
        _, y = px(0.0, gy)
        draw.line([(0, y), (nx * s, y)], fill=(77, 163, 255, 40))
    if pose is not None:
        x, y = px(pose[0], pose[1])
        r = 0.22 / cell * s
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 59, 48, 255), width=3)
        draw.line([(x, y), (x + 2.2 * r * math.cos(pose[2]),
                            y - 2.2 * r * math.sin(pose[2]))],
                  fill=(255, 59, 48, 255), width=4)

    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    meta = {
        "ok": True,
        "file": "state/rl_map.npz",
        "saved_at": updated,
        "saved": time.strftime("%Y-%m-%d %H:%M", time.localtime(updated)),
        "size_m": [round(nx * cell, 1), round(ny * cell, 1)],
        "cell_cm": round(cell * 100),
        "known_m2": round(float((occ | free).sum()) * cell * cell, 1),
        "driven_m2": round(len(visits) * vcell * vcell, 1),
        "sweeps": sweeps,
        "pose": None if pose is None else [round(v, 2) for v in pose],
    }
    return buf.getvalue(), meta


def current() -> tuple[bytes, dict]:
    """(png bytes, metadata), re-rendered only when a source file changed."""
    key = (_stamp(MAP_FILE), _stamp(VISITS_FILE), _stamp(POSE_FILE))
    with _lock:
        if _cache["key"] == key and _cache["png"]:
            return _cache["png"], _cache["meta"]
    if key[0] == 0.0:
        return b"", {"ok": False, "why": "no map saved yet (state/rl_map.npz)"}
    try:
        png, meta = _render()
    except Exception as exc:                          # noqa: BLE001
        return b"", {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    meta["version"] = int(key[0] * 1000) ^ int(key[1] * 1000) ^ int(key[2] * 1000)
    with _lock:
        _cache.update(key=key, png=png, meta=meta)
    return png, meta
