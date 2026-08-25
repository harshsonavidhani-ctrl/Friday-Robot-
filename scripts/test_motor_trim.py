#!/usr/bin/env python3
"""test_motor_trim.py — prove the per-side trim without a real Arduino.

Motors._send() needs an open serial link, so this stubs _command instead of
connecting to hardware and points config.CALIBRATION_FILE at a scratch file
for the duration of the test, restoring it afterwards either way.

    ./venv/bin/python scripts/test_motor_trim.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                                # noqa: E402
from motor import Motors                                     # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(ok)
    print(f"  {PASS if ok else FAIL}  {name}")
    if detail:
        print(f"        {detail}")


def make_motor() -> Motors:
    """A Motors instance with the wire stubbed out: no serial, no Arduino."""
    m = Motors(port="/dev/null")
    sent = []
    m._command = lambda line: (sent.append(line), "OK")[1]    # type: ignore[method-assign]
    m.sent = sent                                              # type: ignore[attr-defined]
    return m


def main() -> int:
    print("MOTOR TRIM — per-side balance, tested without hardware\n")

    with tempfile.TemporaryDirectory() as tmp:
        real_file = config.CALIBRATION_FILE
        config.CALIBRATION_FILE = Path(tmp) / "calibration.json"
        try:
            # --- 1. default trim leaves an equal drive command unchanged ---
            config.CALIBRATION_FILE.write_text(json.dumps({}))
            m = make_motor()
            m.drive(100, 100)
            left, right = m.sent[-1][1:].split()
            # MOTOR_INVERT_RIGHT is True on this robot, so a straight drive
            # command is asymmetric on the wire even with trim at 1.0.
            check("untrimmed left is the commanded speed",
                  int(left) == 100, m.sent[-1])
            check("untrimmed right is inverted, same magnitude",
                  int(right) == -100, m.sent[-1])

            # --- 2. a left trim boosts the left side only ------------------
            config.CALIBRATION_FILE.write_text(
                json.dumps({"trim_left": 1.2, "trim_right": 1.0}))
            m = make_motor()
            m.drive(100, 100)
            left, right = m.sent[-1][1:].split()
            check("trim_left=1.2 scales the left side up",
                  int(left) == 120, m.sent[-1])
            check("trim_right=1.0 leaves the right side alone",
                  int(right) == -100, m.sent[-1])

            # --- 3. trim cannot be used to sneak past the duty cap ---------
            config.CALIBRATION_FILE.write_text(json.dumps({"trim_left": 1.2}))
            m = make_motor()
            m.drive(200, 200)
            left, _ = m.sent[-1][1:].split()
            limit = int(255 * config.MOTOR_MAX_DUTY)
            check(f"trimmed 200*1.2 is still clamped to the {limit} cap",
                  int(left) == limit, m.sent[-1])
        finally:
            config.CALIBRATION_FILE = real_file

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
