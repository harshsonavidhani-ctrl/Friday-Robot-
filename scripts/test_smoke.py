#!/usr/bin/env python3
"""test_smoke.py — does Friday still work? The gate every self-made change must pass.

Run after ANY change, and automatically after every change Friday makes to her
own code. If this fails, the change is reverted before it ever goes live.

    ./venv/bin/python scripts/test_smoke.py          # ~20s, no hardware needed
    ./venv/bin/python scripts/test_smoke.py --full   # + the live tool suite

FAST ON PURPOSE. A gate that takes five minutes gets skipped, and a skipped
gate is not a gate. Everything here is import-and-inspect rather than
drive-the-robot, so it runs in seconds and needs no clear floor in front of
the wheels. --full adds the real hardware suite for when that matters.

WHAT IT REFUSES TO LET THROUGH, beyond "the code parses":

  * The safety invariants. Not "safety.py exists" but "guarded_move still
    consults the embodiment gate", "the reflex still has no off switch",
    "stop() is still ungated". A change that quietly removed one of those
    would compile perfectly and pass any ordinary test.
  * The machine's own reachability: ssh, wifi, logind. CLAUDE.md requires
    these after every change, so they are part of the gate rather than
    something to remember.
"""

from __future__ import annotations

import argparse
import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, WARN = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"
_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append(ok)
    print(f"  {PASS if ok else FAIL}  {name}")
    if detail and not ok:
        print(f"        {detail}")
    return ok


def warn(name: str, detail: str = "") -> None:
    print(f"  {WARN}  {name}")
    if detail:
        print(f"        {detail}")


def source(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8-sig")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also run the live MCP tool suite")
    args = ap.parse_args()

    print("SMOKE — can Friday still run?\n")

    print("CODE")
    bad = []
    for path in sorted(ROOT.glob("**/*.py")):
        if any(p in path.parts for p in ("venv", "tools", "__pycache__")):
            continue
        try:
            # utf-8-sig, BECAUSE PYTHON ITSELF ACCEPTS A BOM AND THIS GATE DID
            # NOT. rl/pi_deploy.py, rl/tests/test_creep.py and
            # rl/tests/test_ld19.py were committed with a UTF-8 BOM on
            # 2026-09-17 (6f77380). The interpreter strips it, py_compile is
            # happy and every one of those files imports and runs -- but
            # read_text() hands the U+FEFF to ast.parse, which calls it an
            # "invalid non-printable character". So the gate every change has
            # to pass was failing on three files that work, and CLAUDE.md says
            # a failing gate means the change does not go live. The files are
            # the committed exploration stack, which is not to be edited to
            # make something else work; the reader was the thing that was
            # wrong.
            ast.parse(path.read_text(encoding="utf-8-sig"))
        except SyntaxError as exc:
            bad.append(f"{path.relative_to(ROOT)}:{exc.lineno}: {exc.msg}")
    check("every python file parses", not bad, "; ".join(bad[:3]))

    print("\nIMPORTS")
    for module in ("config", "tts", "sensors", "motor", "memory", "offline",
                   "face_director", "vision"):
        try:
            __import__(module)
            check(f"import {module}", True)
        except Exception as exc:                # noqa: BLE001
            check(f"import {module}", False, f"{type(exc).__name__}: {exc}")

    print("\nTHE MCP SERVER")
    try:
        from mcp_server import registry
        modules = registry.load_tool_modules()
        tools = registry.all_tools() if hasattr(registry, "all_tools") else []
        check("tool modules load", bool(modules), str(modules))
        names = {getattr(t, "name", getattr(t, "__name__", "")) for t in tools}
        for essential in ("stop", "observe", "move"):
            if names:
                check(f"the {essential} tool still exists", essential in names,
                      f"have: {sorted(names)[:8]}")
    except Exception as exc:                    # noqa: BLE001
        check("tool modules load", False, f"{type(exc).__name__}: {exc}")

    print("\nSAFETY INVARIANTS — the things a change must never quietly remove")
    safety = source("mcp_server/safety.py")
    check("guarded_move still consults the observation gate",
          "embodiment.check(" in safety,
          "look-before-you-move was removed from the movement path")
    check("guarded_move still re-checks clearance while running",
          "check_forward_clearance()" in safety and "while True" in safety)
    check("a move still always ends in stop()",
          "finally:" in safety and "drive.stop()" in safety)
    check("emergency_stop still sets ABORT before touching serial",
          "hardware.ABORT.set()" in safety)

    reflex = source("mcp_server/reflex.py")
    check("the reflex still watches for a cliff", "cliff" in reflex)
    check("the reflex still has no off switch",
          "REFLEX_ENABLED" not in reflex and "def disable" not in reflex,
          "something added a way to turn the reflex off")

    movement = source("mcp_server/tools/movement.py")
    check("stop() is still ungated",
          "Never gated" in movement or "emergency_stop()" in movement)

    # FILE 08 STEP 9/10 — the autonomous drive and the money.
    curiosity = source("config.py")
    check("physical curiosity still ships OFF",
          "\nPHYSICAL_CURIOSITY = False" in curiosity,
          "config.PHYSICAL_CURIOSITY is not False in the committed source — "
          "step 9 must never default to on (file 08 step 10)")
    step9 = source("physical_curiosity.py")
    # What it CALLS, not what it talks about: the docstring names
    # safety.guarded_move on purpose, to say what it is trusting.
    check("the idle drive still goes through explore_room, not the motors",
          'call_tool(\n                "explore_room"' in step9
          or '"explore_room"' in step9,
          "step 9 no longer drives through the server's explore_room")
    check("the idle drive still touches no motor of its own",
          "import motor" not in step9 and "import serial" not in step9
          and 'call_tool("move' not in step9
          and 'call_tool("turn' not in step9,
          "step 9 must not drive the motors or judge clearance itself")
    check("the idle drive still stops when he speaks",
          "def heard(" in step9 and 'call_tool("stop"' in step9)
    check("a wake still stops the idle drive",
          "_curiosity_heard()" in source("main.py"),
          "main.py no longer stops an outing when the wake word fires")
    check("vision-triggered Haiku calls are still rate limited",
          'rate_limit.allow("vision_describe")' in source(
              "mcp_server/tools/narrate.py"),
          "say_what_you_see could be called in a loop (file 08 step 10)")
    check("web search still has a per-turn ceiling",
          "max_uses" in source("brain.py"))

    # FILE 09 STEP 1 — corrections are a separate category, not ordinary notes.
    fixes = source("corrections.py")
    check("corrections are still kept under their own kind",
          "LEARNED_PREFERENCE_KIND" in fixes
          and "LEARNED_PREFERENCE_KIND" in source("config.py"),
          "a correction stored as an ordinary fact cannot be queried apart "
          "from everything else Friday has been told")
    check("the reply object still carries the correction field",
          '"correction"' in source("reply_contract.py"))
    check("a near-duplicate no longer drops the kind",
          "UPDATE facts SET text=?, kind=?" in source("memory.py"),
          "a correction that lands on an existing fact would be declassified")

    # FILE 09 STEPS 2-3 — the learning has to reach the model, and gaps must
    # never become facts.
    mem = source("memory.py")
    check("corrections are still injected as instructions, not trivia",
          "told you to behave" in mem and "_preferences_for" in mem,
          "being corrected would change the log and nothing else")
    check("...and are still kept out of the ordinary memory list",
          "exclude_kinds=(preference_kind,)" in mem)
    check("an uncertain answer is still logged WITHOUT an outcome",
          "outcome" not in source("uncertainty.py").split(
              "memory.log_event(")[1].split(")")[0],
          "an outcome would make a gap resurface as a lesson that worked")
    check("recall_experience still returns only events with an outcome",
          "outcome IS NOT NULL" in mem,
          "a logged guess could be read back to her as experience")

    guard = source("mcp_server/devguard.py")
    for essential in ("mcp_server/safety.py", "mcp_server/reflex.py",
                      "mcp_server/devguard.py", "scripts/rollback.sh"):
        check(f"devguard still protects {essential}", f'"{essential}"' in guard)

    print("\nTHE RAILS THEMSELVES")
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "test_devguard.py")],
                            capture_output=True, text=True, timeout=120)
    check("devguard's own tests pass", result.returncode == 0,
          result.stdout.strip().splitlines()[-1] if result.stdout else "")

    print("\nTHE MACHINE IS STILL REACHABLE (CLAUDE.md requires this)")
    ssh = subprocess.run(["systemctl", "is-active", "ssh"],
                         capture_output=True, text=True).stdout.strip()
    check("ssh is active", ssh == "active", f"systemctl says {ssh!r}")
    listening = subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout
    check("ssh is listening on 22", ":22 " in listening)
    logind = subprocess.run(["systemctl", "is-active", "systemd-logind"],
                            capture_output=True, text=True).stdout.strip()
    check("systemd-logind is active (login still possible)", logind == "active")
    net = subprocess.run(["nmcli", "-t", "-f", "STATE", "general"],
                         capture_output=True, text=True).stdout.strip()
    check("the network is connected", net.startswith("connected"), f"nmcli: {net!r}")

    if args.full:
        print("\nTHE LIVE ROBOT")
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "test_tools.py"), "--http"],
            capture_output=True, text=True, timeout=400)
        tail = result.stdout.strip().splitlines()[-1] if result.stdout else ""
        check("the live tool suite passes", result.returncode == 0, tail)

    passed = sum(_results)
    failed = len(_results) - passed
    print(f"\n{passed} passed, {failed} failed")
    if failed:
        print("\nSMOKE TEST FAILED — a change that does this must not go live.")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
