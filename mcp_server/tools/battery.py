"""Battery: honest about the gauge that does not exist.

No battery fuel gauge is wired on this robot (config.BATTERY_SENSOR is None),
so there is no percentage to report and inventing one would be worse than
saying nothing. The closest real signal is the Pi's own power health:
`vcgencmd get_throttled` reports under-voltage and throttling straight from
the SoC, which is exactly what a sagging battery or a thin USB cable looks
like from software.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess

import config
from mcp_server.registry import tool

_THROTTLED_RE = re.compile(r"0x([0-9a-fA-F]+)")

# Bit meanings per `vcgencmd get_throttled` — the low bits are the state RIGHT
# NOW, the high bits are "has this happened since boot". Both matter: "now" is
# what to act on, "since boot" catches a brownout that has already recovered
# and would otherwise leave no trace.
_BITS = {
    0: "under_voltage_now",
    1: "frequency_capped_now",
    2: "throttled_now",
    3: "soft_temp_limit_now",
    16: "under_voltage_since_boot",
    17: "frequency_capped_since_boot",
    18: "throttled_since_boot",
    19: "soft_temp_limit_since_boot",
}


def _parse_throttled(raw: str) -> dict:
    """Turn vcgencmd's `throttled=0x50005` into named flags.

    Split out from the tool so the bit parsing can be tested without
    vcgencmd, which only exists on the real Pi.
    """
    match = _THROTTLED_RE.search(raw)
    if not match:
        return {"error": f"unrecognised vcgencmd output: {raw.strip()!r}"}

    value = int(match.group(1), 16)
    flags = {name: bool(value & (1 << bit)) for bit, name in _BITS.items()}
    flags["raw"] = hex(value)
    return flags


@tool
def get_battery() -> str:
    """Friday's power health.

    THERE IS NO BATTERY GAUGE on this robot — config.BATTERY_SENSOR is None,
    so `percent` is always "unknown" and must be reported to Harsh as such,
    never guessed. What this DOES give you is real: `power` comes straight
    from the Pi's SoC and tells you whether it is currently under-voltage or
    throttled, or has been since the last boot. under_voltage_now / true
    means the power supply cannot keep up right now — treat that as the robot
    running low on power even without a percentage. *_since_boot flags catch
    a brief brownout that has already recovered, which matters because it can
    explain an earlier glitch (a reboot, a dropped camera) after the fact.
    """
    percent = ("unknown — no battery gauge is wired on this robot, so there "
               "is no percentage to report"
               if config.BATTERY_SENSOR is None
               else f"configured at {config.BATTERY_SENSOR} but not implemented yet")

    if not shutil.which("vcgencmd"):
        power = {"error": "vcgencmd not available — not running on a Pi?"}
    else:
        try:
            result = subprocess.run(
                ["vcgencmd", "get_throttled"], capture_output=True,
                text=True, timeout=5, check=True)
            power = _parse_throttled(result.stdout)
        except (subprocess.SubprocessError, OSError) as exc:
            power = {"error": f"{type(exc).__name__}: {exc}"}

    return json.dumps({"percent": percent, "power": power}, indent=2)
