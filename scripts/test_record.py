#!/usr/bin/env python3
"""The laptop helper's drive recorder (helper 17, 2026-10-21): what Harsh
agreed to -- only W/A/S/D, only while the game is in front, only between start
and stop -- checked with a fake game window, fake keys and a temp folder.

    ./venv/bin/python scripts/test_record.py      # ~10 s
"""
from __future__ import annotations

import http.client
import json
import sys
import tempfile
import threading
import time
import zlib
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

import friday_laptop_agent as fa                          # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if not ok and detail:
        print(f"        {detail}")


GAME = "Grand Theft Auto V Enhanced"
TITLE = {"t": "Grand Theft Auto V"}
READS = {"n": 0}
HELD = {"k": "wd"}
TABLES: list = []


def keys(table=None):
    READS["n"] += 1
    TABLES.append(table)
    return HELD["k"]


def grab(box, max_w):
    w = max_w
    h = max(1, int(w * 9 / 16))
    return w, h, bytes([90, 90, 90, 255]) * (w * h)


print("source: only W, A, S, D are ever read (and the 4 numpad flying keys for a heli/plane)")
src = (ROOT / "laptop" / "friday_laptop_agent.py").read_text()
check("REC_KEYS is exactly W A S D", fa.REC_KEYS == {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44})
check("REC_AIR_KEYS is W A S D + numpad 8 5 4 6, nothing else",
      fa.REC_AIR_KEYS == {"w": 0x57, "a": 0x41, "s": 0x53, "d": 0x44,
                          "num8": 0x68, "num5": 0x65, "num4": 0x64, "num6": 0x66})
check("cars, bikes, boats read REC_KEYS; only heli/plane read the numpad",
      all(fa.rec_keys_for(v) is fa.REC_KEYS for v in ("car", "bike", "boat"))
      and all(fa.rec_keys_for(v) is fa.REC_AIR_KEYS for v in ("heli", "plane")))
check("GetAsyncKeyState appears once, inside keys_held, over its table only",
      src.count("GetAsyncKeyState") == 1
      and "held = [k for k, vk in table.items() if user32.GetAsyncKeyState(vk)" in src)
check("every key a vehicle's network presses is a key its recording reads",
      all({k for _, neg, pos in axes for k in (neg, pos)} == set(fa.rec_keys_for(v))
          for v, axes in fa.VEHICLES.items() if v in ("heli", "plane"))
      and all({k for _, neg, pos in axes for k in (neg, pos)} <= set(fa.REC_KEYS)
              for v, axes in fa.VEHICLES.items() if v not in ("heli", "plane")))
check("no keyboard hook anywhere (SetWindowsHookEx / GetKeyboardState)",
      "SetWindowsHookEx" not in src and "GetKeyboardState" not in src)

print("recording")
tmp = Path(tempfile.mkdtemp())
rec = fa.Recorder(lambda: TITLE["t"], grabber=grab, keys=keys, root=tmp)
fa.REC_HZ = 20.0
got = rec.start(GAME, 60, fa.MAP_BOX)
check("start -> running, a session folder", got["running"] and (tmp / got["session"]).is_dir(), str(got))
time.sleep(1.0)
TITLE["t"] = "Command Prompt"
time.sleep(0.2)
reads_before = READS["n"]
time.sleep(0.6)
check("another window in front -> the keys are NOT read at all",
      READS["n"] == reads_before and rec.report()["skipped"] > 0, str(rec.report()))
TITLE["t"] = "Grand Theft Auto V"
HELD["k"] = "a"
time.sleep(0.5)
rep = rec.stop("test")
check("stop -> stopped, says why", not rep["running"] and rep["why"] == "test", str(rep))
frames = rep["frames"]
check("~20 frames a second while the game was in front", 20 <= frames <= 40, str(frames))
folder = tmp / rep["session"]
info = json.loads((folder / "info.json").read_text())
check("info.json says the game, size, keys", info["game"] == GAME and info["keys"] == list("wasd"))
lines = [json.loads(x) for x in (folder / "labels_00000.jsonl").read_text().splitlines()]
check("one label line per frame", len(lines) == frames, f"{len(lines)} vs {frames}")
check("labels carry the held keys (wd, then a)", lines[0]["keys"] == "wd" and lines[-1]["keys"] == "a",
      f"{lines[0]['keys']} .. {lines[-1]['keys']}")
blob = (folder / "frames_00000.bin").read_bytes()
v = lines[0]["view"]
raw = zlib.decompress(blob[v["off"]:v["off"] + v["len"]])
check("each frame's picture decodes to w*h*4 bytes", len(raw) == v["w"] * v["h"] * 4
      and v["w"] == fa.REC_SIZE[0], str(v))
m = lines[0]["map"]
check("...and so does its minimap", len(zlib.decompress(blob[m["off"]:m["off"] + m["len"]]))
      == m["w"] * m["h"] * 4 and m["w"] == fa.REC_MAP_W, str(m))

print("time limit")
rec2 = fa.Recorder(lambda: "Grand Theft Auto V", grabber=grab, keys=keys, root=tmp)
rec2.start(GAME, 0.4, fa.MAP_BOX)
time.sleep(1.0)
check("max_s reached -> it stops by itself", not rec2.running() and rec2.report()["why"] == "time is up",
      str(rec2.report()))

print("a plane: the flying keys too")
time.sleep(1.1)                                  # sessions are named by the second
rec3 = fa.Recorder(lambda: "Grand Theft Auto V", grabber=grab, keys=keys, root=tmp)
TABLES.clear()
HELD["k"] = ["w", "num8"]
got = rec3.start(GAME, 0.4, fa.MAP_BOX, "plane")
time.sleep(1.0)
info3 = json.loads((tmp / got["session"] / "info.json").read_text())
lines3 = [json.loads(x) for x in (tmp / got["session"] / "labels_00000.jsonl").read_text().splitlines()]
check("info.json says plane and its keys", info3.get("vehicle") == "plane"
      and info3["keys"] == list(fa.REC_AIR_KEYS), str(info3))
check("the keys were read from REC_AIR_KEYS", TABLES and all(t is fa.REC_AIR_KEYS for t in TABLES))
check("labels carry them as a list", lines3 and lines3[0]["keys"] == ["w", "num8"], str(lines3[:1]))
HELD["k"] = "wd"

print("through the HTTP routes")
agent = fa.Agent({"token": "t" * 40, "port": 0, "bind": "127.0.0.1"},
                 titler=lambda: TITLE["t"], grabber=grab, keyreader=keys, rec_root=tmp,
                 autokey=lambda k, up: None)
srv = ThreadingHTTPServer(("127.0.0.1", 0), agent.handler())
threading.Thread(target=srv.serve_forever, daemon=True).start()


def call(method: str, path: str, body: dict | None = None):
    c = http.client.HTTPConnection("127.0.0.1", srv.server_port, timeout=5)
    data = json.dumps(body) if body is not None else None
    hdr = {"X-Friday-Token": "t" * 40}
    if data:
        hdr["Content-Length"] = str(len(data))
    c.request(method, path, data, hdr)
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw)
    except ValueError:
        return r.status, raw


st, got = call("POST", "/record", {"action": "start", "game": GAME, "vehicle": "tank"})
check("an unknown vehicle is refused", st == 400 and not agent.recorder.running(), str(got))
st, got = call("POST", "/record", {"action": "start", "game": GAME, "max_s": 30})
check("POST /record start (a car by default)", st == 200 and got.get("running")
      and got.get("vehicle") == "car", str(got))
st, got = call("POST", "/autopilot", {"game": GAME, "mode": "road"})
check("the autopilot refuses while he is being recorded (labels must be HIS keys)", st == 409, str(got))
time.sleep(0.4)
st, got = call("GET", "/record")
check("GET /record reports frames", st == 200 and got.get("frames", 0) > 0, str(got))
st, got = call("POST", "/record", {"action": "stop"})
check("POST /record stop", st == 200 and not got.get("running"), str(got))
session = got["session"]
st, got = call("GET", "/record/list")
check("GET /record/list names the sessions and files",
      st == 200 and any(s["session"] == session and "info.json" in s["files"] for s in got["sessions"]),
      str(got)[:200])
st, got = call("GET", f"/record/file?session={session}&name=labels_00000.jsonl")
check("GET /record/file fetches one of its files", st == 200 and got and b'"keys"' in got)
for bad in (f"session={session}&name=../../friday_agent.json", "session=..&name=info.json",
            f"session={session}&name=notes.md"):
    st, _ = call("GET", "/record/file?" + bad)
    check(f"refused: {bad.split('&')[1]}", st == 404)
agent.autopilot.set({"game": GAME, "mode": "cruise", "keys": dict(fa.AUTO_KEYS), "hold": ["w"],
                     "box": fa.MAP_BOX, "for_s": 30})
st, got = call("POST", "/record", {"action": "start", "game": GAME})
check("recording refuses while the autopilot drives", st == 409, str(got))
agent.autopilot.stop("test")

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
