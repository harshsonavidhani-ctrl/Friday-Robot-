"""sensors.py — range sensing: the 360° LiDAR and the 8x8 ToF.

Both are read here and nowhere else. The MCP server's scan_environment tool
fuses them into the structured answer the brain navigates by; nothing else in
the project should open these ports.

  * LiDAR — LDROBOT D500 over a CP2102 USB bridge, LD19-family binary protocol
    (config.LIDAR_PORT / LIDAR_BAUD). 360° coverage at ~10 Hz.
  * ToF — DFRobot 8x8 matrix, bridged through a Pi Pico over USB CDC
    (config.TOF_PORT). NOT on I2C: /dev/i2c-1 is disabled on this Pi and
    enabling it is off-limits (CLAUDE.md Part 1). The Pico makes that
    unnecessary. Lines are "yN:d0,...,d7," in mm, 4000 = no return.

WHY BACKGROUND THREADS. The safety interrupt has to re-check clearance every
100 ms *while the robot is moving*. Opening a port and waiting for a fresh ToF
frame takes ~70 ms and a LiDAR revolution takes ~100 ms, so reading on demand
would either stall the move or check stale data. Instead each sensor holds one
long-lived connection and keeps the newest reading in memory; a clearance check
is then a dictionary lookup.

Every reader degrades rather than raising. A sensor that is unplugged reports
`available=False` and the caller says so out loud — navigating on a sensor you
only think you have is how a robot drives into things.

    with Rangefinders() as r:
        r.clearance_mm()       # {'forward': 421, 'left': None, ...}
        r.tof.latest()         # (frame, age_s) or None
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Iterable

import serial

import config

# --- LD19 binary protocol ------------------------------------------------
# 47-byte packet: header, verlen, speed, start angle, 12x(dist,intensity),
# end angle, timestamp, crc. Angles are hundredths of a degree.
LD19_HEADER = 0x54
LD19_VERLEN = 0x2C
LD19_POINTS = 12
LD19_PACKET_LEN = 47


def _crc8_table(poly: int = 0x4D) -> list[int]:
    """LD19's CRC8: polynomial 0x4D, init 0, no reflection.

    Built rather than pasted so it is checkable. If it ever stops matching
    real packets, Lidar falls back to structural validation on its own and
    says so, rather than silently dropping every scan.
    """
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC_TABLE = _crc8_table()


def _crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc = _CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


class _Poller:
    """A serial device read continuously on its own thread.

    Reconnects on its own. Both of Friday's range sensors have been observed
    disappearing from the USB bus mid-session, so "gone" is a normal state to
    be in, not an error to crash on.
    """

    name = "sensor"
    port = ""
    baud = 9600

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._serial: serial.Serial | None = None
        self._last_error: str | None = None
        self._connected = False

    # --- lifecycle ---------------------------------------------------

    def start(self) -> "_Poller":
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"{self.name}-poller", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._close()

    def _close(self) -> None:
        with self._lock:
            if self._serial is not None:
                try:
                    self._serial.close()
                except Exception:       # noqa: BLE001 - shutting down anyway
                    pass
                self._serial = None
            self._connected = False

    def __enter__(self) -> "_Poller":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # --- the loop ----------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with self._lock:
                    self._serial = serial.Serial(self.port, self.baud,
                                                 timeout=1.0)
                    self._connected = True
                    self._last_error = None
                self._read_forever(self._serial)
            except Exception as exc:        # noqa: BLE001 - stay alive
                self._last_error = f"{type(exc).__name__}: {exc}"
            self._close()
            # Unplugged is normal. Retry slowly rather than spinning the CPU
            # on open() failures for a device that is genuinely not there.
            self._stop.wait(1.5)

    def _read_forever(self, port: serial.Serial) -> None:
        raise NotImplementedError

    # --- status ------------------------------------------------------

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_error(self) -> str | None:
        return self._last_error


class ToF(_Poller):
    """DFRobot 8x8 matrix ToF, forward-facing on the front bumper.

    This is the sensor the safety interrupt trusts for "what is straight
    ahead", because unlike the LiDAR its orientation is not in question — it
    is bolted to the bumper pointing forward.
    """

    name = "tof"

    def __init__(self) -> None:
        super().__init__()
        self.port = config.TOF_PORT
        self.baud = config.TOF_BAUD
        self._frame: list[list[int]] | None = None
        self._frame_at = 0.0
        self._partial: dict[int, list[int]] = {}

    def _read_forever(self, port: serial.Serial) -> None:
        port.reset_input_buffer()
        while not self._stop.is_set():
            raw = port.readline()
            if not raw:
                continue
            line = raw.decode(errors="replace").strip()
            # "yN:d0,d1,...,d7,"  — the first line after a connect is usually
            # a fragment, and has no "yN:" prefix. Dropping it is correct.
            if not line.startswith("y") or ":" not in line:
                continue
            head, _, body = line.partition(":")
            try:
                row = int(head[1:])
            except ValueError:
                continue
            values = [int(v) for v in body.split(",") if v.strip().isdigit()]
            if row < 0 or row >= config.TOF_ROWS or len(values) != config.TOF_COLS:
                continue
            self._partial[row] = values
            if len(self._partial) == config.TOF_ROWS:
                frame = [self._partial[r] for r in range(config.TOF_ROWS)]
                self._frame = frame
                self._frame_at = time.monotonic()
                self._partial = {}

    def latest(self, max_age_s: float = 1.0
               ) -> tuple[list[list[int]], float] | None:
        """The newest complete 8x8 frame and its age, or None if too old.

        Age matters more than presence: a frame from four seconds ago
        describes a room the robot has since driven through.
        """
        frame, at = self._frame, self._frame_at
        if frame is None:
            return None
        age = time.monotonic() - at
        return (frame, age) if age <= max_age_s else None

    def forward_mm(self, max_age_s: float = 1.0) -> int | None:
        """Nearest real return anywhere in the frame, in mm.

        Nearest rather than centre-average on purpose: a chair leg that fills
        one cell is exactly the thing worth stopping for, and averaging it
        against seven cells of open air hides it.
        """
        got = self.latest(max_age_s)
        if got is None:
            return None
        frame, _ = got
        real = [v for row in frame for v in row
                if 0 < v < config.TOF_NO_RETURN_MM]
        return min(real) if real else None

    def floor_gap_mm(self, max_age_s: float = 1.0) -> int | None:
        """Nearest return in the BOTTOM rows — the floor just ahead.

        Used for the cliff check: if the bottom rows suddenly report no
        return where they had been seeing floor, the floor stopped.
        """
        got = self.latest(max_age_s)
        if got is None:
            return None
        frame, _ = got
        bottom = frame[config.TOF_ROWS - 2:]
        real = [v for row in bottom for v in row
                if 0 < v < config.TOF_NO_RETURN_MM]
        return min(real) if real else None


class Lidar(_Poller):
    """LDROBOT D500, 360°, on the top dome.

    ⚠ BEARINGS ARE UNCALIBRATED. Which raw angle points at the robot's front
    is not established — nobody has measured it. config.LIDAR_FORWARD_DEG is
    the offset, and until it is measured the sector helpers below refuse to
    name a direction rather than guessing one. A confidently wrong "left is
    clear" is worse than no answer.
    """

    name = "lidar"

    def __init__(self) -> None:
        super().__init__()
        self.port = config.LIDAR_PORT
        self.baud = config.LIDAR_BAUD
        # bearing (whole degrees, raw sensor frame) -> (distance_mm, when)
        self._points: dict[int, tuple[int, float]] = {}
        self._points_lock = threading.Lock()
        self._crc_ok = 0
        self._crc_bad = 0

    def _read_forever(self, port: serial.Serial) -> None:
        port.reset_input_buffer()
        buf = bytearray()
        while not self._stop.is_set():
            chunk = port.read(256)
            if chunk:
                buf.extend(chunk)
            # Resync on the header pair rather than assuming alignment: the
            # stream is joined mid-packet on every connect.
            while len(buf) >= LD19_PACKET_LEN:
                if buf[0] != LD19_HEADER or buf[1] != LD19_VERLEN:
                    del buf[0]
                    continue
                packet = bytes(buf[:LD19_PACKET_LEN])
                if _crc8(packet[:-1]) != packet[-1]:
                    self._crc_bad += 1
                    del buf[0]
                    continue
                self._crc_ok += 1
                del buf[:LD19_PACKET_LEN]
                self._ingest(packet)
            if len(buf) > 4096:             # never let a bad stream grow
                del buf[:-LD19_PACKET_LEN]

    def _ingest(self, packet: bytes) -> None:
        start = struct.unpack_from("<H", packet, 4)[0] / 100.0
        end = struct.unpack_from("<H", packet, 42)[0] / 100.0
        span = (end - start) % 360.0
        step = span / (LD19_POINTS - 1) if LD19_POINTS > 1 else 0.0
        now = time.monotonic()
        fresh: list[tuple[int, tuple[int, float]]] = []
        for i in range(LD19_POINTS):
            dist, _intensity = struct.unpack_from("<HB", packet, 6 + i * 3)
            if dist <= 0:                   # 0 means no return
                continue
            angle = int((start + step * i) % 360.0)
            fresh.append((angle, (dist, now)))
        if fresh:
            with self._points_lock:
                self._points.update(fresh)

    def scan(self, max_age_s: float = 1.0) -> dict[int, int]:
        """Bearing (raw degrees) -> distance mm, for recent points only.

        Stale bearings are dropped rather than kept: a rotating sensor that
        has stopped rotating would otherwise keep reporting the last sweep
        forever, which reads as a perfectly clear room.
        """
        cutoff = time.monotonic() - max_age_s
        with self._points_lock:
            return {a: d for a, (d, at) in self._points.items() if at >= cutoff}

    def nearest_mm(self, max_age_s: float = 1.0) -> int | None:
        points = self.scan(max_age_s)
        return min(points.values()) if points else None

    def sector_mm(self, centre_deg: float, width_deg: float = 60.0,
                  max_age_s: float = 1.0) -> int | None:
        """Nearest return in a sector around a ROBOT-RELATIVE bearing.

        0 is forward, 90 is right, 270 is left — but only once
        config.LIDAR_FORWARD_DEG has been measured. Until then this returns
        None, because the mapping from robot-forward to sensor-zero is
        genuinely unknown and inventing it would be worse than saying so.
        """
        if config.LIDAR_FORWARD_DEG is None:
            return None
        points = self.scan(max_age_s)
        if not points:
            return None
        half = width_deg / 2.0
        best: int | None = None
        for raw_angle, dist in points.items():
            relative = (raw_angle - config.LIDAR_FORWARD_DEG) % 360.0
            delta = abs((relative - centre_deg + 180.0) % 360.0 - 180.0)
            if delta <= half and (best is None or dist < best):
                best = dist
        return best

    @property
    def health(self) -> str:
        total = self._crc_ok + self._crc_bad
        if total == 0:
            return "no packets"
        return f"{self._crc_ok}/{total} packets passed CRC"


# Robot-relative bearings the brain gets clearances for. Names match the
# scan_environment contract exactly, so the tool never renames them.
SECTORS: dict[str, float] = {
    "forward": 0.0,
    "forward_right": 45.0,
    "right": 90.0,
    "backward_right": 135.0,
    "backward": 180.0,
    "backward_left": 225.0,
    "left": 270.0,
    "forward_left": 315.0,
}


class Rangefinders:
    """Both range sensors, fused. This is what the MCP server holds."""

    def __init__(self) -> None:
        self.tof = ToF()
        self.lidar = Lidar()

    def start(self) -> "Rangefinders":
        self.tof.start()
        self.lidar.start()
        return self

    def stop(self) -> None:
        self.tof.stop()
        self.lidar.stop()

    def wait_ready(self, timeout: float = 2.5) -> "Rangefinders":
        """Give the poller threads a moment to open their ports.

        Without this, anything that checks status() immediately after start()
        sees "not connected" for a sensor that is merely still connecting, and
        reports a present device as absent.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.tof.connected and self.lidar.connected:
                break
            time.sleep(0.1)
        return self

    def __enter__(self) -> "Rangefinders":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def clearance_mm(self) -> dict[str, int | None]:
        """Nearest obstacle per direction, in mm. None means "cannot see".

        None is a real answer and must be passed through, not defaulted to a
        big number. A missing sensor reading as "10 metres clear" is the
        single most dangerous bug this file could have.
        """
        out: dict[str, int | None] = {
            name: self.lidar.sector_mm(bearing)
            for name, bearing in SECTORS.items()
        }
        # The ToF is bolted to the front bumper facing forward, so its bearing
        # is not in doubt the way the LiDAR's is. Where both see forward, the
        # nearer wins.
        bumper = self.tof.forward_mm()
        out["bumper"] = bumper
        if bumper is not None:
            ahead = out.get("forward")
            out["forward"] = bumper if ahead is None else min(ahead, bumper)
        out["nearest_any"] = _nearest(
            [self.lidar.nearest_mm(), bumper])
        return out

    def forward_clearance_mm(self) -> tuple[int | None, str]:
        """What is straight ahead, and which sensor said so.

        The safety interrupt calls this. It returns the source too, so a stop
        can be reported as "the bumper saw something" rather than an
        anonymous number.
        """
        bumper = self.tof.forward_mm()
        lidar_ahead = self.lidar.sector_mm(0.0)
        if bumper is not None and lidar_ahead is not None:
            return ((min(bumper, lidar_ahead), "ToF + LiDAR")
                    if bumper <= lidar_ahead else (lidar_ahead, "LiDAR"))
        if bumper is not None:
            return bumper, "ToF"
        if lidar_ahead is not None:
            return lidar_ahead, "LiDAR"
        return None, "no range sensor"

    def status(self) -> dict[str, str]:
        return {
            "tof": _describe(self.tof),
            "lidar": (f"connected, {self.lidar.health}"
                      if self.lidar.connected else _describe(self.lidar)),
            "lidar_bearings": ("calibrated"
                               if config.LIDAR_FORWARD_DEG is not None
                               else "UNCALIBRATED — directions unavailable"),
        }


def _describe(poller: _Poller) -> str:
    """A poller's state, distinguishing absent from merely not-yet-open."""
    if poller.connected:
        return "connected"
    if poller.last_error:
        return f"not connected ({poller.last_error})"
    return "not connected (absent, or still opening)"


def _nearest(values: Iterable[int | None]) -> int | None:
    real = [v for v in values if v is not None]
    return min(real) if real else None


def main() -> int:
    """Live read-only dump, for checking the sensors by eye."""
    import argparse
    ap = argparse.ArgumentParser(description="Read Friday's range sensors.")
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    with Rangefinders() as r:
        time.sleep(1.5)                     # let both connect and fill
        for key, value in r.status().items():
            print(f"{key:16} {value}")
        print()
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            ahead, source = r.forward_clearance_mm()
            scan = r.lidar.scan()
            print(f"forward {str(ahead) + ' mm' if ahead else '—':>10}"
                  f"  ({source})   lidar bearings: {len(scan):3}"
                  f"   nearest any: {r.clearance_mm()['nearest_any']}")
            time.sleep(0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
