"""Friday's exploration map, drawn for the dashboard. READ-ONLY.

Harsh, 2026-09-13: "show the map live on the dashboard" -- and, the same day,
"dont make changes in the map". Nothing here writes to state/: the map, the
visit trace and the pose are opened with np.load / read_text and never saved.

What it draws is the map the exploration run loads (state/rl_map.npz, the
persistent WorldMap): free space, obstacles, unknown, the area she has driven,
and her position. The run SAVES that file only when it ends, so while one is out
the picture comes from state/rl_map.live.npz (rl/live_map.py), rewritten every
second: the map grows and her position moves, like a GPS. A live file that has
stopped changing is not believed, and the metadata says which one this is, so the
dashboard never presents an old map as live.

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
LIVE_FILE = ROOT / "state" / "rl_map.live.npz"
LIVE_FRESH_S = 6.0

# THE RUN'S WORLD IS A MIRROR IMAGE (see config.TURN_WORDS_MIRRORED: it reads the
# LiDAR's clockwise bearings as counter-clockwise). It steers fine inside it, but
# drawn as it stands the room is backwards and, live, she turns left on the floor
# and right on the phone. So the picture alone flips: +y runs DOWN the image. The
# map, the pose and every file stay exactly as the run wrote them.
MIRRORED = True

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


def _live_now() -> bool:
    stamp = _stamp(LIVE_FILE)
    return stamp > 0.0 and time.time() - stamp <= LIVE_FRESH_S


def _load_live():
    import numpy as np
    with np.load(LIVE_FILE, allow_pickle=False) as z:
        pose = z["pose"].astype(float)
        return (z["logodds"].copy(), z["origin"].astype(float).copy(), float(z["cell"]),
                int(z["scans"]), float(z["updated_at"]), z["cells"].copy(),
                float(z["cell_m"]), (float(pose[0]), float(pose[1]), float(pose[2])))


def _load_saved():
    import numpy as np
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
    return logodds, origin, cell, sweeps, updated, visits, vcell, pose


def _render(live: bool = False) -> tuple[bytes, dict]:
    import numpy as np
    from PIL import Image, ImageDraw

    logodds, origin, cell, sweeps, updated, visits, vcell, pose = (
        _load_live() if live else _load_saved())

    nx, ny = logodds.shape
    occ = logodds > OCCUPIED_LOGODDS
    free = logodds < FREE_LOGODDS
    s = PX_PER_CELL
    rgb = np.full((ny, nx, 3), (46, 52, 64), np.uint8)       # unknown: dark slate
    rgb[free.T] = (226, 232, 240)                            # free: light
    rgb[occ.T] = (10, 12, 16)                                # obstacle: near-black
    if not MIRRORED:
        rgb = rgb[::-1]                                      # +y up
    im = Image.fromarray(rgb).resize((nx * s, ny * s), Image.NEAREST)
    draw = ImageDraw.Draw(im, "RGBA")

    def px(x: float, y: float) -> tuple[float, float]:
        row = (y - origin[1]) / cell
        return ((x - origin[0]) / cell * s, (row if MIRRORED else ny - row) * s)

    for a, b in visits:
        x0, y0 = px(a * vcell, b * vcell)
        x1, y1 = px((a + 1) * vcell, (b + 1) * vcell)
        draw.rectangle([x0, min(y0, y1), x1, max(y0, y1)], fill=(255, 106, 0, 95))
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
        tip = px(pose[0] + 2.2 * 0.22 * math.cos(pose[2]),
                 pose[1] + 2.2 * 0.22 * math.sin(pose[2]))
        draw.line([(x, y), tip],
                  fill=(255, 59, 48, 255), width=4)

    buf = io.BytesIO()
    # A live frame is replaced in a second, and this runs every second somebody watches.
    im.save(buf, "PNG", compress_level=1 if live else 6, optimize=not live)
    meta = {
        "ok": True,
        "live": live,
        "file": "state/rl_map.live.npz" if live else "state/rl_map.npz",
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
    live = _live_now()
    key = ((_stamp(LIVE_FILE), 0.0, 0.0) if live
           else (_stamp(MAP_FILE), _stamp(VISITS_FILE), _stamp(POSE_FILE)))
    with _lock:
        if _cache["key"] == key and _cache["png"]:
            return _cache["png"], _cache["meta"]
    if key[0] == 0.0:
        return b"", {"ok": False, "why": "no map saved yet (state/rl_map.npz)"}
    try:
        png, meta = _render(live)
    except Exception as exc:                          # noqa: BLE001
        return b"", {"ok": False, "why": f"{type(exc).__name__}: {exc}"}
    meta["version"] = int(key[0] * 1000) ^ int(key[1] * 1000) ^ int(key[2] * 1000)
    with _lock:
        _cache.update(key=key, png=png, meta=meta)
    return png, meta
