"""Every navigation map Friday has, drawn for the dashboard. READ-ONLY.

Harsh, 2026-09-26: "all the map generated also will be showed in dashboard
like the old map and all the map made will be displayed in dashboard".

Lists, newest first within each group:
  * the live navigation map          state/map.npz
  * every version the bundle kept    state/map-permanent/navigation/history/*/map.npz
  * every survey, stitched and spots state/map-permanent/surveys/*/{stitched,spot_*}.npz

Nothing here writes to state/. Pictures are rendered on demand and cached on
each file's mtime, so a phone polling the list costs a directory walk.
Only files the listing produced can be drawn: a request names an id from the
listing, never a path.
"""
from __future__ import annotations

import io
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIVE = ROOT / "state" / "map.npz"
HISTORY = ROOT / "state" / "map-permanent" / "navigation" / "history"
SURVEYS = ROOT / "state" / "map-permanent" / "surveys"

PX_PER_CELL = 3
MAX_SIDE_PX = 900

_lock = threading.Lock()
_png_cache: dict[str, tuple[float, bytes]] = {}
_stat_cache: dict[str, tuple[float, dict]] = {}


def _entries() -> list[dict]:
    out = []
    if LIVE.exists():
        out.append({"path": LIVE, "group": "Navigation map (in use)",
                    "label": "The map she navigates on"})
    if HISTORY.exists():
        for d in sorted(HISTORY.iterdir(), reverse=True):
            f = d / "map.npz"
            if f.exists():
                why = ""
                for name in ("why.txt",):
                    try:
                        why = (d / name).read_text().strip().splitlines()[0][:140]
                    except (OSError, IndexError):
                        pass
                out.append({"path": f, "group": "Earlier versions",
                            "label": d.name + (f" — {why}" if why else "")})
    if SURVEYS.exists():
        for d in sorted(SURVEYS.iterdir(), reverse=True):
            if not d.is_dir():
                continue
            files = [d / "stitched.npz"] + sorted(d.glob("spot_*.npz"))
            for f in files:
                if f.exists():
                    out.append({"path": f, "group": f"Survey {d.name}",
                                "label": ("Stitched from the spots"
                                          if f.stem == "stitched"
                                          else f"Standing spot {f.stem[5:]}")})
    for e in out:
        e["id"] = str(e["path"].relative_to(ROOT))
    return out


def _grid(path: Path):
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from slam.grid import OccupancyGrid
    return OccupancyGrid.load(path)


def _stats(path: Path) -> dict:
    mtime = path.stat().st_mtime
    key = str(path)
    hit = _stat_cache.get(key)
    if hit and hit[0] == mtime:
        return hit[1]
    try:
        g = _grid(path)
        c2 = g.cell ** 2
        nx, ny = g.shape
        stats = {"free_m2": round(float(g.free_mask().sum() * c2), 1),
                 "walls_m2": round(float(g.occupied_mask().sum() * c2), 2),
                 "size_m": [round(nx * g.cell, 1), round(ny * g.cell, 1)]}
    except Exception as exc:                     # noqa: BLE001
        stats = {"error": str(exc)[:120]}
    _stat_cache[key] = (mtime, stats)
    return stats


def listing() -> list[dict]:
    with _lock:
        out = []
        for e in _entries():
            try:
                mtime = e["path"].stat().st_mtime
            except OSError:
                continue
            out.append({"id": e["id"], "group": e["group"],
                        "label": e["label"], "saved_at": mtime,
                        "version": int(mtime),
                        **_stats(e["path"])})
        return out


def png(map_id: str) -> bytes | None:
    """The picture of one listed map, or None if the id is not listed."""
    import numpy as np
    from PIL import Image

    with _lock:
        match = next((e for e in _entries() if e["id"] == map_id), None)
        if match is None:
            return None
        path = match["path"]
        mtime = path.stat().st_mtime
        hit = _png_cache.get(map_id)
        if hit and hit[0] == mtime:
            return hit[1]
        g = _grid(path)
        img = np.full(g.occupied_mask().shape, 150, np.uint8)   # unknown: slate
        img[g.free_mask()] = 255
        img[g.occupied_mask()] = 0
        pic = Image.fromarray(np.flipud(img.T))
        scale = max(1, min(PX_PER_CELL, MAX_SIDE_PX // max(pic.size)))
        pic = pic.resize((pic.size[0] * scale, pic.size[1] * scale),
                         Image.NEAREST)
        buf = io.BytesIO()
        pic.save(buf, "PNG")
        data = buf.getvalue()
        _png_cache[map_id] = (mtime, data)
        return data
