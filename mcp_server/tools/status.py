"""Status: how Friday itself is doing.

Every number here is read from the real machine. Where something genuinely
cannot be measured — the battery, on this robot — it says so rather than
inventing a figure. A robot that reports a confident 87% from a sensor it does
not have is worse than one that admits it has no fuel gauge.
"""

from __future__ import annotations

import json
import time

import psutil

import config
import face_control
import tts
import vision
from mcp_server import hardware
from mcp_server.registry import tool

_STARTED = time.monotonic()


def _cpu_temperature_c() -> float | None:
    try:
        with open(config.CPU_TEMP_PATH) as handle:
            return int(handle.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


def _uptime(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {secs}s" if minutes else f"{secs}s"


def _battery() -> str:
    """Honest about the missing fuel gauge."""
    if config.BATTERY_SENSOR is None:
        return ("unknown — no battery gauge is wired on this robot, so there "
                "is no percentage to report")
    return f"configured at {config.BATTERY_SENSOR} but not implemented yet"


@tool
def get_status() -> str:
    """How Friday is doing: power, temperature, load, uptime, and what is
    plugged in.

    Worth checking if something is behaving oddly, if a tool has just failed,
    or if Harsh asks how you are. The `hardware` section says which devices
    actually answered just now — that is the thing to look at when a camera or
    the motors are not responding.
    """
    temperature = _cpu_temperature_c()
    ranges = hardware.ranges()
    cameras = vision.resolve_cameras()
    drive = hardware.motors_if_present()

    status = {
        "battery": _battery(),
        "cpu_percent": psutil.cpu_percent(interval=0.3),
        "cpu_temperature_c": (round(temperature, 1)
                              if temperature is not None else None),
        "cpu_temperature_note": (
            "hot — Friday may be throttling"
            if temperature is not None and temperature >= config.CPU_TEMP_WARN_C
            else "normal"),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage(str(config.ROOT)).percent,
        "current_emotion": face_control.get_emotion(),
        "mcp_server_uptime": _uptime(time.monotonic() - _STARTED),
        "pi_uptime": _uptime(time.time() - psutil.boot_time()),
        "speed_percent": hardware.speed_percent(),
        "voice": tts.engine_name(),
        "debug_silent_speech": config.DEBUG_SILENT_SPEECH,
        "hardware": {
            "motors": (f"{drive.firmware} on {drive.port}" if drive
                       else "not connected"),
            "up_camera": cameras["up"] or "not on the USB bus",
            "down_camera": cameras["down"] or "not on the USB bus",
            **ranges.status(),
        },
    }
    return json.dumps(status, indent=2)
