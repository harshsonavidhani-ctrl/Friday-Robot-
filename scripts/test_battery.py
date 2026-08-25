#!/usr/bin/env python3
"""test_battery.py — prove the throttle-bit parsing without a real Pi.

`vcgencmd` only exists on the actual hardware, so this tests _parse_throttled
directly against real example outputs rather than shelling out.

    ./venv/bin/python scripts/test_battery.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_server.tools import battery                        # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(ok)
    print(f"  {PASS if ok else FAIL}  {name}")
    if detail:
        print(f"        {detail}")


def main() -> int:
    print("BATTERY — throttle-word parsing, tested without hardware\n")

    # --- 1. everything healthy ----------------------------------------------
    flags = battery._parse_throttled("throttled=0x0\n")
    check("0x0 means nothing is wrong",
          not any(v for k, v in flags.items() if k != "raw"), flags)

    # --- 2. under-voltage happening right now --------------------------------
    flags = battery._parse_throttled("throttled=0x1\n")
    check("bit 0 is under_voltage_now", flags["under_voltage_now"], flags)
    check("and nothing else is set",
          not any(v for k, v in flags.items()
                  if k not in ("raw", "under_voltage_now")), flags)

    # --- 3. a brownout that has already recovered ----------------------------
    # Real example: under-voltage occurred at some point since boot, but the
    # low bits (right now) are clear -- exactly the case that would otherwise
    # leave no trace.
    flags = battery._parse_throttled("throttled=0x50000\n")
    check("bit 16 is under_voltage_since_boot",
          flags["under_voltage_since_boot"], flags)
    check("bit 18 is throttled_since_boot", flags["throttled_since_boot"], flags)
    check("but *_now flags are clear",
          not flags["under_voltage_now"] and not flags["throttled_now"], flags)

    # --- 4. currently under-voltage AND throttled, all history set -----------
    flags = battery._parse_throttled("throttled=0x50005\n")
    check("bit 0 and bit 2 both set for a live brownout",
          flags["under_voltage_now"] and flags["throttled_now"], flags)
    check("raw value is echoed back as hex", flags["raw"] == "0x50005", flags)

    # --- 5. garbage input does not raise --------------------------------------
    flags = battery._parse_throttled("not vcgencmd output at all")
    check("unparseable output is reported, not raised", "error" in flags, flags)

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
