#!/usr/bin/env python3
"""She says when her CPU is hot (cpu_heat.py; Harsh, 2026-10-03: "if pi temp
reaches above 60 degree so friday will say my cpu is hot pls turn on the fan").

    ./venv/bin/python scripts/test_cpu_heat.py

No sound, no network, no real temperature: the readings and the clock are
faked; one read of the real sensor file only checks it parses.
"""

from __future__ import annotations

import ast
import socket
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402
import cpu_heat as ch                                   # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


print("--- the settings ---")
check("60 C, his number; a fixed line asking for the fan",
      config.CPU_HOT_SAY_C == 60.0 and config.CPU_HEAT_WARN is True
      and config.CPU_HOT_LINE == "My CPU is hot. Please turn on the fan.")
check("hysteresis below it, a sustain, a repeat that does not nag",
      config.CPU_HOT_CLEAR_C < config.CPU_HOT_SAY_C
      and config.CPU_HOT_SUSTAIN_S > 0 and config.CPU_HOT_REPEAT_S >= 300)

print("--- the rule (Policy) ---")


def policy():
    return ch.Policy(60.0, 57.0, sustain_s=15.0, repeat_s=600.0, retry_s=30.0)


p = policy()
check("cool: never", not any(p.update(t, s) for s, t in enumerate([45, 55, 59.9] * 20)))
p = policy()
said = [s for s in range(0, 40) if p.update(70.0 if s < 5 else 50.0, float(s))]
check("one hot burst shorter than the sustain: nothing", said == [])
p = policy()
said = []
for s in range(0, 200):
    if p.update(65.0, float(s)):
        said.append(s)
        p.said_it(float(s))
check("hot and staying hot: once at 15 s, then not again for 10 min",
      said == [15], str(said))
for s in range(200, 700):
    if p.update(65.0, float(s)):
        said.append(s)
        p.said_it(float(s))
check("...still hot 10 min later: said again", said == [15, 615], str(said))
p = policy()
for s in range(0, 20):
    if p.update(62.0, float(s)):
        p.said_it(float(s))
check("wobbling between 57 and 60 is the same hot spell, not a new one",
      not any(p.update(58.5 if s % 2 else 60.5, float(s)) for s in range(20, 300)))
check("...below 57 it has cooled", not p.update(56.0, 300.0) and p.hot_since is None)
check("...and the next crossing is a new spell (sustain again)",
      not p.update(61.0, 301.0) and p.update(61.0, 316.0))
p = policy()
p.update(70.0, 0.0)
check("due, but he was talking: offered again after 30 s, not dropped",
      p.update(70.0, 15.0) and (p.missed(15.0) or True) and not p.update(70.0, 30.0)
      and p.update(70.0, 45.0))
check("an unreadable sensor says nothing", not policy().update(None, 100.0))
try:
    ch.Policy(60.0, 61.0, 1, 1, 1)
    check("clear must be below hot", False)
except ValueError:
    check("clear must be below hot", True)

print("--- 70 C: said again at once (Harsh: 'if I forgot, at 70 the alarm again') ---")


def policy2():
    return ch.Policy(60.0, 57.0, sustain_s=15.0, repeat_s=600.0, retry_s=30.0,
                     again_c=70.0, again_clear_c=67.0)


def feed(p, temps_by_s, start=0, end=None):
    out = []
    for s in range(start, end):
        lvl = p.update(temps_by_s(s), float(s))
        if lvl:
            p.handed(lvl)
            p.said_it(float(s))
            out.append((s, lvl))
    return out


check("the file's settings: 70 C, cooled below 67 C, a stronger line",
      config.CPU_HOT_AGAIN_C == 70.0 and config.CPU_HOT_AGAIN_CLEAR_C == 67.0
      and config.CPU_HOT_LINE_AGAIN == "My CPU is very hot now. Please turn on the fan.")
p = policy2()
said = feed(p, lambda s: 63.0 if s < 100 else 71.0, 0, 200)
check("63 C: 'hot' once; he forgets; it climbs to 71 C -> 'very hot' 15 s later, NOT "
      "after the 10-minute wait", said == [(15, 1), (115, 2)], str(said))
said = feed(p, lambda s: 71.0, 200, 900)
check("...still 71 C: then every 10 min, at the very-hot level", said == [(715, 2)],
      str(said))
said = feed(p, lambda s: 65.0 if s < 960 else 71.0, 900, 1000)
check("down to 65 C (below 67: cooled from very hot, still hot) then up to 71 C "
      "again: said again at once", said == [(975, 2)], str(said))
p = policy2()
said = feed(p, lambda s: 68.5 if s % 2 else 70.5, 0, 300)
check("wobbling round 70 (never below 67) is one very-hot alarm, not a new one each "
      "time", [lvl for _, lvl in said].count(2) == 1, str(said))
p = policy2()
said = feed(p, lambda s: 72.0, 0, 100)
check("straight from cool to 72 C: ONE line (very hot), not 'hot' then 'very hot'",
      said == [(15, 2)], str(said))
p = policy2()
p.update(64.0, 0.0)
lvl = p.update(64.0, 15.0)
p.handed(lvl)
p.said_it(15.0)
p.update(72.0, 20.0)
p.update(72.0, 35.0)
lvl = p.update(72.0, 36.0)
p.handed(lvl)
p.missed(36.0)
check("very hot but he was talking: offered again after 30 s, not dropped",
      lvl == 2 and not p.update(72.0, 50.0) and p.update(72.0, 66.0) == 2)
try:
    ch.Policy(60.0, 57.0, 1, 1, 1, again_c=59.0)
    check("the very-hot level must be above the hot one", False)
except ValueError:
    check("the very-hot level must be above the hot one", True)

print("--- the wake source (HeatWatch) ---")


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def watch(temps, exploring=False, quiet=False):
    clock = Clock()
    it = iter(temps)
    w = ch.HeatWatch(read=lambda: next(it, None), clock=clock,
                     exploring=lambda: exploring, quiet=lambda: quiet, poll_s=0.01)
    return w, clock


def run(w, clock, n, step_s=5.0):
    out = []
    for _ in range(n):
        trig = w.step()
        if trig:
            out.append((clock.t, trig))
        clock.t += step_s
    return out


w, clock = watch([72.4] * 10)
fired = run(w, clock, 10)
check("hot (72 C, straight from cool): ONE trigger, the very-hot level, after the "
      "sustain", len(fired) == 1 and fired[0][1] == "72 2"
      and fired[0][0] - 1000.0 >= 15.0, str(fired))
check("...nothing more until main says it was said (no queue of repeats)",
      run(w, clock, 3) == [])
w.delivered()
check("...said: the next one is 10 min away", run(w, clock, 20) == [])
w, clock = watch([75.0] * 20)
w.pause()
check("she is talking: held", run(w, clock, 6) == [])
w.resume()
check("...then said", len(run(w, clock, 2)) == 1)
w, clock = watch([75.0] * 40)
w.sleep(3600)
check("'Friday, go to sleep' (not exploring): held", run(w, clock, 10) == [])
w, clock = watch([75.0] * 10, exploring=True)
w.sleep(3600)
check("...but a RUN's hush (same call): said anyway -- a run is when it is hottest",
      len(run(w, clock, 10)) == 1)
check("the watch is never counted as asleep (the run hush keeps polling the rest)",
      w.sleeping_for() == 0.0 and w.sleep(10) == 0.0)
w, clock = watch([75.0] * 40, quiet=True)
check("quiet mode: held", run(w, clock, 10) == [])
w, clock = watch([None] * 10)
check("no sensor: silent, no crash", run(w, clock, 10) == [])
check("triggers parse back; others are not ours",
      ch.parse_trigger("cpu_heat: 72 2") == 72.0 and ch.trigger_level("cpu_heat: 72 2") == 2
      and ch.parse_trigger("cpu_heat: 64") == 64.0 and ch.trigger_level("cpu_heat: 64") == 1
      and ch.parse_trigger("reminders: x") is None and ch.parse_trigger("cpu_heat: x") is None)
check("level 1 says the hot line, level 2 the very-hot line",
      ch.line_for(1) == "My CPU is hot. Please turn on the fan."
      and ch.line_for(2) == "My CPU is very hot now. Please turn on the fan.")

print("--- reading the sensor ---")
tmp = Path(tempfile.mkdtemp(prefix="friday-heat-")) / "temp"
tmp.write_text("61234\n")
check("millidegrees -> degrees", ch.read_c(tmp) == 61.234)
check("missing file -> None", ch.read_c(tmp.parent / "nope") is None)
real = ch.read_c()
check("the real sensor parses (Pi only)", real is None or 20.0 < real < 100.0, str(real))

print("--- offline, and the one way she speaks ---")
src = (ROOT / "cpu_heat.py").read_text()
imports = {n.names[0].name.split(".")[0] if isinstance(n, ast.Import) else (n.module or "")
           .split(".")[0] for n in ast.walk(ast.parse(src))
           if isinstance(n, (ast.Import, ast.ImportFrom))}
check("cpu_heat imports no network, model or sound library",
      not imports & {"socket", "requests", "httpx", "urllib", "anthropic", "brain",
                     "tts", "pygame", "sounddevice", "subprocess"}, str(imports))
main = (ROOT / "main.py").read_text()
check("main.py: the watch is a wake source and its turn says the line for its level",
      "cpu_heat.HeatWatch()" in main and 'trigger.startswith("cpu_heat: ")' in main
      and "cpu_heat.line_for(cpu_heat.trigger_level(trigger))" in main
      and "voice.say(line)" in main and "watch.missed()" in main)
real_connect = socket.socket.connect


def refuse(*a, **k):
    raise AssertionError("network used")


socket.socket.connect = refuse
try:
    w, clock = watch([70.0] * 6)
    run(w, clock, 6)
    check("a whole hot spell runs with the network refused", True)
finally:
    socket.socket.connect = real_connect

print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
