#!/usr/bin/env python3
"""Driving from an image network (helper 19, 2026-10-25): mode "net" must feed
the network the pictures it was trained on (RGB in the right order, two
pictures NET_GAP ticks apart, the minimap) and press exactly that vehicle's
keys for as long as each axis says. A tiny hand-made ONNX network stands in
for train_driver_net.py's (no PyTorch on the Pi): steer = red - green of the
picture, pedal = +0.6 always, pitch/roll from the minimap. No laptop, no game.

    ./venv/bin/python scripts/test_driver_net.py
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
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import friday_laptop_agent as fa                          # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


def make_net(path: Path, weights: np.ndarray, bias: np.ndarray) -> None:
    """axes = tanh([mean of each view channel (6), mean of each map channel (3)] @ W + b)."""
    n_out = weights.shape[1]
    nodes = [
        helper.make_node("GlobalAveragePool", ["view"], ["vp"]),
        helper.make_node("GlobalAveragePool", ["map"], ["mp"]),
        helper.make_node("Flatten", ["vp"], ["vf"]),
        helper.make_node("Flatten", ["mp"], ["mf"]),
        helper.make_node("Concat", ["vf", "mf"], ["f"], axis=1),
        helper.make_node("MatMul", ["f", "W"], ["z"]),
        helper.make_node("Add", ["z", "b"], ["zb"]),
        helper.make_node("Tanh", ["zb"], ["axes"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_driver",
        [helper.make_tensor_value_info("view", TensorProto.FLOAT, [1, 6, 90, 160]),
         helper.make_tensor_value_info("map", TensorProto.FLOAT, [1, 3, 45, 64])],
        [helper.make_tensor_value_info("axes", TensorProto.FLOAT, [1, n_out])],
        [numpy_helper.from_array(weights.astype(np.float32), "W"),
         numpy_helper.from_array(bias.astype(np.float32), "b")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def write_vehicle(folder: Path, vehicle: str, weights, bias, **meta_extra) -> None:
    onnx_path = folder / fa.NET_FILE.format(vehicle)
    make_net(onnx_path, np.array(weights), np.array(bias))
    meta = {"vehicle": vehicle, "axes": [list(a) for a in fa.VEHICLES[vehicle]],
            "view": [160, 90], "map": [64, 45], "gap": 2, "lag": 2, "smooth": 9,
            "made": "test", "frames": {"train": 1}, "scores": {}}
    meta.update(meta_extra)
    onnx_path.with_suffix(".json").write_text(json.dumps(meta))


SCENE = {"bgr": (0, 0, 255), "map_bgr": (0, 0, 0)}     # pure red picture: steer right


def grab(box, max_w):
    if max_w == 64:
        b, g, r = SCENE["map_bgr"]
        return 64, 45, bytes([b, g, r, 255]) * (64 * 45)
    b, g, r = SCENE["bgr"]
    return 160, 90, bytes([b, g, r, 255]) * (160 * 90)


tmp = Path(tempfile.mkdtemp())
# car: steer = 3*(red now - green now), pedal = +0.6 (atanh 0.6 = 0.693)
car_w = np.zeros((9, 2))
car_w[0, 0], car_w[1, 0] = 3.0, -3.0
write_vehicle(tmp, "car", car_w, [0.0, 0.6931])
# plane: yaw 0, throttle 0, pitch = +3*map red, roll = -3*map blue
plane_w = np.zeros((9, 4))
plane_w[6, 2], plane_w[8, 3] = 3.0, -3.0
write_vehicle(tmp, "plane", plane_w, [0.0, 0.0, 0.0, 0.0])
# heli: made for other keys -> refused
write_vehicle(tmp, "heli", np.zeros((9, 4)), [0, 0, 0, 0],
              axes=[["yaw", "q", "e"], ["throttle", "s", "w"], ["pitch", "num5", "num8"],
                    ["roll", "num4", "num6"]])

print("pictures")
pic = fa.net_picture(160, 90, bytes([10, 20, 30, 255]) * (160 * 90), (160, 90))
check("net_picture is [3, h, w] uint8, RGB from BGRA",
      pic.shape == (3, 90, 160) and pic.dtype == np.uint8 and tuple(pic[:, 0, 0]) == (30, 20, 10),
      f"{pic.shape} {pic[:, 0, 0]}")
small = fa.net_picture(320, 180, bytes(range(256)) * (320 * 180 * 4 // 256), (160, 90))
check("...a bigger window is scaled to the trained size", small.shape == (3, 90, 160), str(small.shape))

print("the networks beside the helper")
nets = fa.Nets(tmp)
info = nets.info()
check("info lists car, plane, heli", set(info) == {"car", "plane", "heli"}, str(info))
check("a vehicle with no network says what to record", "record some boat" in nets.get("boat"),
      str(nets.get("boat")))
check("a network made for other keys is refused", "other keys" in str(nets.get("heli")),
      str(nets.get("heli")))
got = nets.get("car")
check("the car network loads", not isinstance(got, str), str(got))


def drive(vehicle: str, seconds: float, **cfg):
    pressed: list = []
    ap = fa.Autopilot(lambda: "Grand Theft Auto V", grabber=grab,
                      presser=lambda k, up: pressed.append((k, up, time.monotonic())))
    ap.nets = nets
    ap.set({"game": "Grand Theft Auto V Enhanced", "mode": "net", "vehicle": vehicle,
            "keys": dict(fa.AUTO_KEYS), "box": fa.MAP_BOX, "speed": cfg.get("speed", 1.0),
            "bias": cfg.get("bias", 0.0), "for_s": 30})
    time.sleep(seconds)
    rep = ap.report()
    ap.stop("test")
    return ap, rep, pressed


def held_share(pressed, key: str) -> float:
    downs = [p[2] for p in pressed if p[0] == key and not p[1]]
    ups = [p[2] for p in pressed if p[0] == key and p[1]]
    if not downs:
        return 0.0
    return sum(u - d for d, u in zip(downs, ups)) / len(downs) / fa.AUTO_TICK_S


print("a car")
ap, rep, pressed = drive("car", 1.0)
last = rep.get("last", {})
check("it ticks along", rep.get("ticks", 0) >= 5, str(rep))
check("a red picture (R=1, G=0) steers right at full", last.get("steer") == 1.0 and "d" in last.get("keys", ""),
      str(last))
check("pedal +0.6", abs(last.get("pedal", 0) - 0.6) < 0.01, str(last))
share = held_share(pressed, "w")
check(f"W held about 60% of each tick ({share:.0%})", 0.4 < share < 0.8, str(share))
check("only the car's keys were ever pressed", {p[0] for p in pressed} <= {"a", "d", "w", "s"},
      str({p[0] for p in pressed}))
check("every key is up when it stops", not ap._down, str(ap._down))
SCENE["bgr"] = (0, 255, 0)
_, rep, _ = drive("car", 0.5)
check("a green picture steers left", rep.get("last", {}).get("steer") == -1.0
      and "a" in rep["last"].get("keys", ""), str(rep.get("last")))
SCENE["bgr"] = (0, 128, 128)
_, rep, _ = drive("car", 0.5, speed=0.3, bias=0.6)
last = rep.get("last", {})
check("speed caps the gas (0.3) and bias leans the steering (+0.3)",
      abs(last.get("pedal", 0) - 0.3) < 0.01 and abs(last.get("steer", 0) - 0.3) < 0.02, str(last))
SCENE["bgr"] = (0, 0, 255)

print("a plane")
SCENE["map_bgr"] = (255, 0, 255)                      # map red=1, blue=1: pitch +1, roll -1
ap, rep, pressed = drive("plane", 0.6)
last = rep.get("last", {})
check("pitch and roll from the minimap -> numpad 8 and numpad 4",
      last.get("pitch") == 1.0 and last.get("roll") == -1.0
      and set(last.get("keys", "").split("+")) == {"num8", "num4"}, str(last))
check("only the plane's keys were pressed", {p[0] for p in pressed} <= {"num8", "num4"},
      str({p[0] for p in pressed}))
check("numpad keys are game keys the helper may press",
      all(k in fa.GAME_KEYS for _, neg, pos in fa.VEHICLES["plane"] for k in (neg, pos)))
SCENE["map_bgr"] = (0, 0, 0)

print("refusals")
ap, rep, _ = drive("boat", 0.3)
check("no boat network -> it stops and says so", not ap.running() and "boat" in rep.get("why", ""),
      str(rep))
ap = fa.Autopilot(lambda: "Notepad", grabber=grab, presser=lambda k, up: None)
ap.nets = nets
ap.set({"game": "Grand Theft Auto V Enhanced", "mode": "net", "vehicle": "car",
        "keys": dict(fa.AUTO_KEYS), "box": fa.MAP_BOX, "for_s": 30})
time.sleep(0.3)
check("the game not in front -> it stops", not ap.running()
      and "front" in ap.report().get("why", ""), str(ap.report()))

print("through the HTTP routes")
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
    return r.status, json.loads(r.read())


st, got = call("GET", "/model")
check("GET /model lists the image networks", st == 200 and set(got.get("nets", {})) == {"car", "plane", "heli"},
      str(got)[:200])
st, got = call("POST", "/model", {"reload": True})
check("POST /model reload", st == 200 and got.get("ok") and "car" in got.get("nets", {}), str(got)[:200])
st, got = call("POST", "/autopilot", {"game": "Grand Theft Auto V", "mode": "net", "vehicle": "tank"})
check("an unknown vehicle is refused", st == 400, str(got))
st, got = call("POST", "/autopilot", {"game": "Grand Theft Auto V", "mode": "net", "vehicle": "car",
                                      "for_s": 5})
time.sleep(0.5)
st2, rep = call("GET", "/autopilot")
check("POST /autopilot mode net drives", st == 200 and rep.get("ticks", 0) >= 2 and rep.get("mode") == "net",
      str(rep))
call("POST", "/autopilot", {"mode": "off"})

t = time.perf_counter()
sess, meta = nets.get("car")
x = np.zeros((1, 6, 90, 160), np.float32)
m = np.zeros((1, 3, 45, 64), np.float32)
for _ in range(20):
    fa.net_picture(*grab(None, 160), (160, 90))
    sess.run(None, {"view": x, "map": m})
print(f"        (pictures + a fake network on the Pi: {(time.perf_counter() - t) / 20 * 1000:.1f} ms a tick)")

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
