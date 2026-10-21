#!/usr/bin/env python3
"""Driving from a trained network (helper 18, 2026-10-21): the laptop's plain
Python forward pass must give the SAME answers as the sklearn model the Pi
trained, on the SAME thumbnail numbers. No laptop, no game.

    ./venv/bin/python scripts/test_driver_model.py
"""
from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))
sys.path.insert(0, str(ROOT / "scripts"))

import friday_laptop_agent as fa                          # noqa: E402
import train_driver as td                                 # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


rng = np.random.default_rng(0)
W, H = 160, 90
frame = rng.integers(0, 256, size=(H, W, 4), dtype=np.uint8)
frame[:, :, 3] = 255
bgra = frame.tobytes()

print("the same input numbers on the Pi and on the laptop")
a = td.thumb_grey(W, H, bgra)
b = np.array(fa.thumb_grey(W, H, bgra, *td.THUMB), dtype=np.float32)
check("thumbnail: helper == trainer (32x18 grey)", a.shape == b.shape and np.allclose(a, b, atol=1e-5),
      f"max diff {np.abs(a - b).max() if a.shape == b.shape else 'shape'}")
mini = rng.integers(40, 90, size=(36, 64, 4), dtype=np.uint8)
mini[:, :, 3] = 255
check("route features: helper == trainer",
      np.allclose(td.route_features(64, 36, mini.tobytes()), fa.route_features(64, 36, mini.tobytes())))

print("the same answers")
from sklearn.neural_network import MLPClassifier                  # noqa: E402
X = rng.random((400, td.THUMB[0] * td.THUMB[1] + 3)).astype(np.float32)
y3 = rng.integers(-1, 2, size=400)
y2 = rng.integers(0, 2, size=400)
for name, y in (("three classes", y3), ("two classes", y2)):
    clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=200, random_state=0).fit(X, y)
    net = td.export(clf, list(clf.classes_))
    net = json.loads(json.dumps(net))
    same = all(fa.mlp_predict(net, X[i].tolist())[0] == int(clf.predict(X[i:i + 1])[0])
               for i in range(100))
    check(f"forward pass == sklearn predict on 100 inputs ({name})", same)

model = {"thumb": list(td.THUMB), "made": "test", "train": 400,
         "steer": td.export(MLPClassifier(hidden_layer_sizes=(8,), max_iter=50, random_state=0)
                            .fit(X, y3), [-1, 0, 1]),
         "pedal": td.export(MLPClassifier(hidden_layer_sizes=(8,), max_iter=50, random_state=0)
                            .fit(X, y3), [-1, 0, 1])}
model = json.loads(json.dumps(model))
check("check_model accepts a real export", fa.check_model(model) == "", fa.check_model(model))
bad = json.loads(json.dumps(model))
bad["steer"]["coefs"][0][0] = bad["steer"]["coefs"][0][0][:-1]
check("...and refuses a layer with the wrong number of inputs", fa.check_model(bad) != "")
bad = json.loads(json.dumps(model))
bad["pedal"]["classes"] = [-1, 0, 7]
check("...and classes that are not -1/0/1", fa.check_model(bad) != "")
check("...and anything else", fa.check_model({"exec": "x"}) != "" and fa.check_model(None) != "")

print("driving with it")
PRESSED: list = []


def grab(box, max_w):
    if max_w == fa.REC_MAP_W:
        return 64, 36, mini.tobytes()
    return W, H, bgra


ap = fa.Autopilot(lambda: "Grand Theft Auto V", grabber=grab,
                  presser=lambda k, up: PRESSED.append((k, up)))
ap.model = model
want_steer, _ = fa.mlp_predict(model["steer"], fa.thumb_grey(W, H, bgra, 32, 18) + fa.route_features(64, 36, mini.tobytes()))
t0 = time.perf_counter()
ap.set({"game": "Grand Theft Auto V Enhanced", "mode": "model", "keys": dict(fa.AUTO_KEYS),
        "box": fa.MAP_BOX, "speed": 1.0, "for_s": 30})
time.sleep(0.8)
rep = ap.report()
ap.stop("test")
ticks = rep.get("ticks", 0)
check("model mode ticks along", ticks >= 3, str(rep))
last = rep.get("last", {})
check("...pressing what the network says", last.get("steer") == want_steer, str(last))
check("...and every key is up when it stops", not ap._down, str(ap._down))
ap2 = fa.Autopilot(lambda: "Grand Theft Auto V", grabber=grab, presser=lambda k, up: None)
ap2.set({"game": "Grand Theft Auto V Enhanced", "mode": "model", "keys": dict(fa.AUTO_KEYS),
         "box": fa.MAP_BOX, "for_s": 30})
time.sleep(0.3)
check("no network loaded -> it stops and says so",
      not ap2.running() and "network" in ap2.report().get("why", ""), str(ap2.report()))
t = time.perf_counter()
for _ in range(5):
    fa.mlp_predict(model["steer"], fa.thumb_grey(W, H, bgra, 32, 18) + [0.0, 0.0, 1.0])
print(f"        (one tick's thumbnail + network on the Pi: {(time.perf_counter() - t) / 5 * 1000:.0f} ms)")

print("through the HTTP routes")
tmp = Path(tempfile.mkdtemp())
agent = fa.Agent({"token": "t" * 40, "port": 0, "bind": "127.0.0.1"},
                 titler=lambda: "Grand Theft Auto V", grabber=grab, rec_root=tmp,
                 autokey=lambda k, up: None)
srv = ThreadingHTTPServer(("127.0.0.1", 0), agent.handler())
threading.Thread(target=srv.serve_forever, daemon=True).start()


def call(method: str, path: str, body: dict | None = None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=10)
    data = json.dumps(body) if body is not None else None
    hdr = {"X-Friday-Token": "t" * 40}
    if data:
        hdr["Content-Length"] = str(len(data))
    c.request(method, path, data, hdr)
    r = c.getresponse()
    return r.status, json.loads(r.read() or b"{}")


st, got = call("GET", "/model")
check("GET /model before -> none loaded", st == 200 and got.get("loaded") is False, str(got))
st, got = call("POST", "/model", {"model": model})
check("POST /model -> loaded and saved beside the helper",
      st == 200 and (tmp / fa.MODEL_FILE).exists(), str(got))
st, got = call("POST", "/model", {"model": {"thumb": [32, 18]}})
check("POST /model with junk -> 400", st == 400, str(got))
agent2 = fa.Agent({"token": "t" * 40, "port": 0, "bind": "127.0.0.1"},
                  titler=lambda: "x", grabber=grab, rec_root=tmp)
check("a restarted helper loads the saved network", agent2.autopilot.model is not None)

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
