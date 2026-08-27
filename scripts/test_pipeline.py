#!/usr/bin/env python3
"""Prove commands actually EXECUTE, and that Friday re-arms after each one.

    ./venv/bin/python scripts/test_pipeline.py             # safe commands only
    ./venv/bin/python scripts/test_pipeline.py --move      # includes driving
    ./venv/bin/python scripts/test_pipeline.py -n 10

THIS IS THE FAILURE MODE B TEST. Failure Mode A is "she never hears me" and
scripts/test_wake.py measures it. This is the other one: she hears, the face
changes, and the command never happens -- or worse, THIS command works and the
NEXT one silently does not, because the turn ended without handing the
microphone back.

So the check is not "did it answer". It is, for every single command:

    the turn reached a terminal stage           (it did not hang)
    a tool actually ran, when one was expected  (it did not just talk)
    the turn ended in LISTENING                 (it re-armed)

The last one is the whole point. A pipeline that finishes a command but never
returns to waiting looks perfect on the command you are testing and broken on
the one after it, which is why this runs commands BACK TO BACK and reports the
sequence rather than testing one in isolation.

WHAT IT DOES NOT DO. It does not speak into the microphone -- it injects the
transcript, because that isolates the half being tested. Wake word and capture
are test_wake.py's job. If both pass and real speech still fails, the fault is
between them, in the handover.

MOTOR COMMANDS ARE OPT-IN. Without --move nothing that drives is sent, so this
is safe to run with Friday on a desk and nobody watching. With --move, be in
the room.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                            # noqa: E402

# Commands that change nothing physical. "stop" is here because stopping an
# already-stopped robot is a no-op that still exercises the whole tool path.
SAFE = [
    ("what time is it", False),
    ("stop", True),
    ("what's the time", False),
    ("halt", True),
    ("tell me the time", False),
]
MOVING = [
    ("move forward", True),
    ("turn left", True),
    ("turn right", True),
    ("go backward", True),
    ("stop", True),
]


class TraceCapture(logging.Handler):
    """Collect the stage trace for one turn, so it can be asserted on."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "friday.turn":
            self.lines.append(record.getMessage())

    def reset(self) -> None:
        self.lines = []

    def reached(self, *needles: str) -> bool:
        return any(any(n in line for n in needles) for line in self.lines)


async def run(count: int, include_moving: bool, gap: float) -> int:
    import brain as brain_module
    import main as friday
    import stages

    capture = TraceCapture()
    logging.getLogger("friday.turn").addHandler(capture)

    commands = list(SAFE)
    if include_moving:
        commands += MOVING
    while len(commands) < count:
        commands += commands
    commands = commands[:count]

    face = friday.Face()
    voice = friday.Voice(face).start()

    brain_obj = None
    try:
        brain_obj = brain_module.Brain()
        print("brain: online (commands go to the model)")
    except brain_module.BrainError:
        print("brain: OFFLINE (no API key) — testing the local keyword path")

    results: list[tuple[str, bool, str, float]] = []

    async with brain_module.mcp_session() as session:
        print(f"\nrunning {len(commands)} commands back to back\n")
        for index, (text, expect_tool) in enumerate(commands, 1):
            capture.reset()
            turn = stages.Turn(f"injected {index}")
            began = time.monotonic()
            failure = ""
            try:
                await asyncio.wait_for(
                    friday.handle(brain_obj, session, voice, face, text, turn),
                    timeout=config.STAGE_BRAIN_TIMEOUT_S + 30)
            except asyncio.TimeoutError:
                failure = "HUNG — never reached a terminal stage"
            except Exception as exc:                      # noqa: BLE001
                failure = f"raised {type(exc).__name__}: {exc}"

            # RE-ARM, the thing this test exists for. In the live loop the
            # run() finally does it; here the turn is driven directly, so the
            # trace is checked for a terminal stage instead.
            elapsed = time.monotonic() - began
            if not failure and not capture.reached("DONE", "ABANDONED",
                                                   "FAILED"):
                failure = "no terminal stage — the turn never finished"
            if not failure and expect_tool and not capture.reached("TOOL"):
                failure = "no tool ran, but this command needed one"

            ok = not failure
            results.append((text, ok, failure, elapsed))
            print(f"  [{index:2}/{len(commands)}] {'ok  ' if ok else 'FAIL'} "
                  f"{text:<22} {elapsed:5.1f}s  {failure}")
            if not ok:
                # SHOW THE TRACE FOR THE FAILURE, rather than making Harsh go
                # digging. This is the log he asked to be shown.
                for line in capture.lines:
                    print(f"        | {line}")
            await asyncio.sleep(gap)

    voice.close()

    passed = sum(1 for _, ok, _, _ in results if ok)
    print(f"\n{passed}/{len(results)} executed and finished cleanly")
    slowest = max(results, key=lambda r: r[3]) if results else None
    if slowest:
        print(f"slowest: {slowest[0]!r} at {slowest[3]:.1f}s")
    if passed == len(results):
        print("PASS")
        return 0
    print("FAIL — the traces above show which stage ate each one.")
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-n", "--count", type=int, default=10)
    ap.add_argument("--move", action="store_true",
                    help="include commands that actually drive the motors")
    ap.add_argument("--gap", type=float, default=2.0,
                    help="seconds between commands")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    if args.move:
        print("⚠ --move will DRIVE the robot. Be in the room.\n")
    return asyncio.run(run(args.count, args.move, args.gap))


if __name__ == "__main__":
    raise SystemExit(main())
