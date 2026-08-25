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
    return (ROOT / rel).read_text()


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
            ast.parse(path.read_text())
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
