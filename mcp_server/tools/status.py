"""Status: how Friday itself is doing.

Every number here is read from the real machine. Where something genuinely
cannot be measured — the battery, on this robot — it says so rather than
inventing a figure. A robot that reports a confident 87% from a sensor it does
not have is worse than one that admits it has no fuel gauge.
"""

from __future__ import annotations

import json
import socket
import time

import psutil

import config
import face_control
import tts
import vision
from mcp_server import embodiment, hardware
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


def _network() -> dict:
    """Friday's own address on the LAN, and where the dashboard is.

    WHY THIS IS IN STATUS. The Pi is on DHCP with a 6 hour lease and the campus
    network registers no name for it, so its address genuinely moves during a
    day. The dashboard is then unreachable from the phone and nothing on the
    robot says why. mDNS (Harsh.local) is the textbook answer and does not help
    here: Android's browser will not resolve .local reliably.

    So the robot that knows the address is the one that should be asked for it.
    "Friday, what's your IP?" is the intended path back in.
    """
    addresses = []
    try:
        for name, entries in psutil.net_if_addrs().items():
            if name == "lo":
                continue
            for entry in entries:
                if entry.family == socket.AF_INET and entry.address:
                    addresses.append((name, entry.address))
    except Exception:                           # noqa: BLE001
        pass

    primary = next((ip for _, ip in addresses), None)
    return {
        "addresses": {name: ip for name, ip in addresses} or "no LAN address",
        "dashboard_url": (f"https://{primary}:{config.DASHBOARD_PORT}/"
                          if primary else "unavailable — no LAN address"),
        "dashboard_note": (
            "Give Harsh the numeric address, not Harsh.local — his phone's "
            "browser does not resolve .local. The address changes because the "
            "lease is short, so read it out fresh rather than from memory."),
    }


@tool
def get_status() -> str:
    """How Friday is doing: power, temperature, load, uptime, and what is
    plugged in.

    Worth checking if something is behaving oddly, if a tool has just failed,
    or if Harsh asks how you are. The `hardware` section says which devices
    actually answered just now — that is the thing to look at when a camera or
    the motors are not responding.

    ALSO CALL THIS when Harsh asks for your IP address, or says he cannot reach
    the dashboard. `network.dashboard_url` is the answer, and it must be read
    fresh every time: the address is on a short DHCP lease and it moves.
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
        # Whether Friday is currently allowed to move, and why not if it is
        # not. Surfaced here so "why won't you move?" is answerable without
        # having to trip the refusal first.
        "observation": embodiment.status(),
        "network": _network(),
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
