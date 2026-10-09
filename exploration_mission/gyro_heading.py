"""Which way she faces in the START's frame, by the gyro, after the run.

The run tracks her heading with the gyro (pi_deploy._gyro_dtheta): on
2026-10-07 it slid only 4-7 deg in 3 minutes, where it used to slide ~100.
The arrival looks and the approach onto the start then searched EVERY
heading, so a room that matched itself turned round was "two places" and the
approach refused (2026-10-07 21:34), and the look said "turned 137 deg".

The board's yaw cannot simply be carried over: opening the serial port resets
the Uno (DTR), so its yaw starts again at 0 in every process that opens it.
So the heading is LATCHED once, standing still after the run has parked:
the run's last heading (gyro-tracked) plus the board's yaw at that moment.
After that, heading = latched + (yaw now - yaw then), through every turn the
engine makes. A reconnect (a new drive object = the board reset), a reading
that is not "ok", or a run without a gyro -> None, and the callers search
every heading as before. Never a guess.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Callable

log = logging.getLogger("friday.exploration_mission.gyro_heading")

READY_WAIT_S = 4.0         # after a reconnect the board measures its bias (~1 s still)


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _hardware_imu() -> tuple[object, dict | None]:
    """(the drive object, its G reading) from friday-mcp's own link. The
    object is the identity check: a new one means the board was reset.

    ⚠ NEVER OPENS THE LINK: opening the port resets the Uno. Only a link
    friday-mcp already holds (body_sense's thread opens it once the run's
    lease is gone) is read; outside friday-mcp -- a test -- there is none."""
    import sys
    hardware = sys.modules.get("mcp_server.hardware")
    if hardware is None:
        return None, None
    try:
        if hardware.lease_active() > 0.0:
            return None, None                           # the run still holds it
        drive = getattr(hardware, "_motors", None)
        if drive is None or getattr(drive, "serial", None) is None:
            return None, None
    except Exception:                                   # noqa: BLE001
        return None, None
    reader = getattr(drive, "imu", None)
    if reader is None:
        return drive, None
    try:
        return drive, reader()
    except Exception:                                   # noqa: BLE001
        return drive, None


class GyroHeading:
    def __init__(self, read: Callable[[], tuple[object, dict | None]] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 monotonic: Callable[[], float] = time.monotonic):
        self._read = read or _hardware_imu
        self._sleep, self._mono = sleep, monotonic
        self._latched: tuple[float, float, int] | None = None   # theta, yaw0 deg, drive id
        self.why = "not latched"

    def reset(self) -> None:
        self._latched, self.why = None, "not latched"

    @property
    def latched(self) -> bool:
        return self._latched is not None

    def _ok_reading(self):
        """An "ok" reading, waiting up to READY_WAIT_S for the link to be
        reopened after the run and for the board's bias ("cal")."""
        drive, imu = self._read()
        t0 = self._mono()
        while (not imu or not imu.get("ok")) and self._mono() - t0 < READY_WAIT_S:
            if imu and imu.get("status") == "nogyro":
                break
            self._sleep(0.2)
            drive, imu = self._read()
        return drive, (imu if imu and imu.get("ok") else None)

    def latch(self, theta_rad: float | None) -> bool:
        """Standing still: her heading in the start's frame now is
        `theta_rad` (the run's, gyro-tracked; None = the run had no gyro).
        True if the board's yaw was there to carry it on."""
        self._latched = None
        if theta_rad is None:
            self.why = "the run had no gyro"
            return False
        drive, imu = self._ok_reading()
        if imu is None:
            self.why = "no gyro reading"
            return False
        self._latched = (_wrap(float(theta_rad)), float(imu["yaw_deg"]), id(drive))
        self.why = "latched"
        log.info("heading latched: %+.0f deg in the start's frame (board yaw %.1f)",
                 math.degrees(self._latched[0]), self._latched[1])
        return True

    def now(self) -> float | None:
        """Her heading in the start's frame (radians), or None when the
        gyro cannot vouch for it."""
        if self._latched is None:
            return None
        theta, yaw0, drive_id = self._latched
        drive, imu = self._ok_reading()
        if imu is None:
            self.why = "no gyro reading"
            return None                                  # maybe a busy link: keep the latch
        if id(drive) != drive_id:
            self._latched, self.why = None, "the motor board was reconnected (its yaw restarted)"
            log.info("heading dropped: %s", self.why)
            return None
        return _wrap(theta + math.radians(float(imu["yaw_deg"]) - yaw0))
