#!/usr/bin/env python3
"""Teach Friday to drive from Harsh's recordings (helper 17, 2026-10-21).

Behaviour cloning, kept small enough to run on the LAPTOP in plain Python:
  input  = the game picture as a 32x18 grey thumbnail (576 numbers) plus where
           the GPS route goes on the minimap (3 numbers)
  output = steer (left / straight / right) and pedal (gas / coast / brake),
           two small MLPs (sklearn), exported as plain JSON weights.

    ./venv/bin/python scripts/train_driver.py                  # all sessions
    ./venv/bin/python scripts/train_driver.py --report-only    # data summary

Frames showing the pause menu / big map are dropped.
The last 20% of each session is held out (time order, not shuffled: the
frames next to each other are near-identical, so a random split would only
measure memory). Writes state/driver_model.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import friday_laptop_agent as fa                          # noqa: E402

REC = ROOT / "state" / "recordings"
OUT = ROOT / "state" / "driver_model.json"
THUMB = (32, 18)


def thumb_grey(w: int, h: int, bgra: bytes) -> np.ndarray:
    """The picture as THUMB grey levels in 0..1 (box average, row-major)."""
    a = np.frombuffer(bgra, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].astype(np.float32)
    grey = a.mean(axis=2)
    tw, th = THUMB
    ys = np.linspace(0, h, th + 1).astype(int)
    xs = np.linspace(0, w, tw + 1).astype(int)
    out = np.empty((th, tw), dtype=np.float32)
    for j in range(th):
        for i in range(tw):
            out[j, i] = grey[ys[j]:ys[j + 1], xs[i]:xs[i + 1]].mean()
    return (out / 255.0).ravel()


def menu_on_screen(w: int, h: int, bgra: bytes) -> bool:
    """GTA's pause menu / big map: the white 'MAP' tab with the dark tab bar
    beside it (same test as the drive supervisor)."""
    a = np.frombuffer(bgra, dtype=np.uint8).reshape(h, w, 4)[:, :, :3].mean(axis=2)
    tab = a[int(0.17 * h):int(0.20 * h) + 1, int(0.17 * w):int(0.26 * w)]
    bar = a[int(0.17 * h):int(0.20 * h) + 1, int(0.40 * w):int(0.48 * w)]
    return bool((tab > 200).mean() > 0.6 and (bar < 70).mean() > 0.6)


def route_features(w: int, h: int, bgra: bytes) -> list[float]:
    """[route seen?, sin, cos] of the GPS route's direction on the minimap,
    relative to up (the map turns with the car, its arrow sits at ARROW_AT).
    The recorded minimap is only 64 px wide -- too small to FIND the arrow (a
    few pixels), so its fixed spot is used."""
    t = fa.tune_defaults()
    cx, cy = t["arrow_x"] * w, t["arrow_y"] * h
    ang = fa._route_ring(w, h, bgra, cx, cy, 0.0, 0.14 * w, h * fa.MAP_BAR_FROM, t)
    if ang is None:
        return [0.0, 0.0, 1.0]
    r = math.radians(ang)
    return [1.0, math.sin(r), math.cos(r)]


def labels(keys: str) -> tuple[int, int]:
    steer = (1 if "d" in keys else 0) - (1 if "a" in keys else 0)       # -1, 0, 1
    pedal = 1 if "w" in keys and "s" not in keys else (-1 if "s" in keys and "w" not in keys else 0)
    return steer, pedal


def load(session: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    X, S, P = [], [], []
    seen = dropped = 0
    for lab in sorted(session.glob("labels_*.jsonl")):
        blob = (session / lab.name.replace("labels_", "frames_").replace(".jsonl", ".bin")).read_bytes()
        for line in lab.read_text().splitlines():
            row = json.loads(line)
            seen += 1
            v, m = row["view"], row["map"]
            view = zlib.decompress(blob[v["off"]:v["off"] + v["len"]])
            small = zlib.decompress(blob[m["off"]:m["off"] + m["len"]])
            if menu_on_screen(v["w"], v["h"], view):
                dropped += 1
                continue
            route = route_features(m["w"], m["h"], small)
            X.append(np.concatenate([thumb_grey(v["w"], v["h"], view), np.array(route, np.float32)]))
            s, p = labels(row["keys"])
            S.append(s)
            P.append(p)
    info = {"session": session.name, "frames": seen, "dropped_menu": dropped, "kept": len(X)}
    if not X:
        return np.zeros((0, THUMB[0] * THUMB[1] + 3)), np.zeros(0), np.zeros(0), info
    return np.stack(X), np.array(S), np.array(P), info


def export(clf, classes: list[int]) -> dict:
    return {"coefs": [c.T.round(5).tolist() for c in clf.coefs_],      # [out][in] per layer
            "intercepts": [b.round(5).tolist() for b in clf.intercepts_],
            "activation": clf.activation, "classes": [int(c) for c in classes]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--upload", action="store_true",
                    help="send state/driver_model.json to the laptop helper (18+) and stop")
    args = ap.parse_args()
    if args.upload:
        sys.path.insert(0, str(ROOT))
        import laptop_research as lr
        got = lr._request("/model", {"model": json.loads(OUT.read_text())}, 30.0)
        print("laptop:", got)
        return 0 if got.get("ok") else 1
    from sklearn.neural_network import MLPClassifier
    trX, trS, trP, teX, teS, teP = [], [], [], [], [], []
    for session in sorted(p for p in REC.iterdir() if p.is_dir()):
        t0 = time.time()
        X, S, P, info = load(session)
        print(json.dumps(info), f"({time.time() - t0:.0f} s)")
        if len(X) < 50:
            continue
        cut = int(len(X) * 0.8)
        trX.append(X[:cut]); trS.append(S[:cut]); trP.append(P[:cut])
        teX.append(X[cut:]); teS.append(S[cut:]); teP.append(P[cut:])
    if not trX:
        print("not enough driving recorded yet")
        return 1
    trX, trS, trP = np.concatenate(trX), np.concatenate(trS), np.concatenate(trP)
    teX, teS, teP = np.concatenate(teX), np.concatenate(teS), np.concatenate(teP)
    for name, y in (("steer", trS), ("pedal", trP)):
        vals, counts = np.unique(y, return_counts=True)
        print(name, "train mix:", dict(zip(vals.tolist(), (counts / counts.sum()).round(3).tolist())))
    if args.report_only:
        return 0
    model = {"thumb": list(THUMB), "made": time.strftime("%Y-%m-%d %H:%M"), "train": int(len(trX)),
             "test": int(len(teX))}
    for name, ytr, yte in (("steer", trS, teS), ("pedal", trP, teP)):
        clf = MLPClassifier(hidden_layer_sizes=(args.hidden,), alpha=1e-3, max_iter=300,
                            early_stopping=True, random_state=0)
        clf.fit(trX, ytr)
        acc = float((clf.predict(teX) == yte).mean())
        base = float(max((yte == c).mean() for c in np.unique(yte)))
        print(f"{name}: held-out accuracy {acc:.3f} (always-the-commonest would get {base:.3f})")
        model[name] = export(clf, list(clf.classes_))
        model[name + "_acc"] = round(acc, 3)
        model[name + "_baseline"] = round(base, 3)
    OUT.write_text(json.dumps(model))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
