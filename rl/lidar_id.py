#!/usr/bin/env python3
"""Identify an unknown 360-degree LiDAR: which baud, and which protocol.

We established the device streams continuously but does not speak RPLIDAR. That means
either the baud is wrong (garbage in, garbage out) or it is a different family. This
sweeps every common baud and scores the raw bytes against the known frame headers.

Run:  python3 lidar_id.py [/dev/ttyUSB1]
"""
import collections
import sys
import time

try:
    import serial
except ImportError:
    sys.exit("FATAL: pyserial missing. pip install pyserial")

PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyUSB1"

# Baud rates used by the common hobby 360 LiDARs.
BAUDS = [115200, 128000, 150000, 153600, 230400, 256000, 460800, 512000]

# (name, header bytes, expected frame length) -- frame length None = variable.
SIGNATURES = [
    ("RPLIDAR A-series (descriptor)", b"\xa5\x5a", None),
    ("LD06 / LD19 / STL-19P",         b"\x54\x2c", 47),
    ("YDLIDAR X2/X4",                 b"\xaa\x55", None),
    ("Delta-2 series",                b"\xaa\xaa\xaa\xaa", None),
    ("3irobotix / Camsense X1",       b"\x55\xaa\x03\x08", None),
]


def score(buf: bytes) -> list[tuple[int, str, int]]:
    """Count header occurrences and how evenly they are spaced (real framing is regular)."""
    out = []
    for name, hdr, flen in SIGNATURES:
        positions = []
        start = 0
        while True:
            i = buf.find(hdr, start)
            if i < 0:
                break
            positions.append(i)
            start = i + 1
        if len(positions) < 3:
            continue
        gaps = [b - a for a, b in zip(positions, positions[1:])]
        common_gap, count = collections.Counter(gaps).most_common(1)[0]
        # Regularity is the real signal: random data also contains any 2-byte pattern,
        # but not at a consistent stride.
        regularity = count / max(len(gaps), 1)
        out.append((int(regularity * 100), name, common_gap))
    return sorted(out, reverse=True)


print(f"port: {PORT}")
best = []
for baud in BAUDS:
    try:
        s = serial.Serial(PORT, baud, timeout=1.0)
        s.reset_input_buffer()
        time.sleep(0.35)                       # let a burst accumulate
        buf = s.read(4096)
        s.close()
    except Exception as exc:                    # noqa: BLE001
        print(f"  @{baud:6d}: ERROR {type(exc).__name__}: {exc}")
        continue

    if not buf:
        print(f"  @{baud:6d}: NO DATA")
        continue

    uniq = len(set(buf))
    hits = score(buf)
    head = buf[:24].hex()
    print(f"  @{baud:6d}: {len(buf):5d} B, {uniq:3d} distinct  {head}")
    for reg, name, gap in hits[:2]:
        print(f"              -> {name}: header every {gap} B, {reg}% regular")
        best.append((reg, baud, name, gap))

print()
if best:
    best.sort(reverse=True)
    reg, baud, name, gap = best[0]
    print(f"BEST GUESS: {name} at {baud} baud "
          f"(header stride {gap} B, {reg}% regular)")
    if "RPLIDAR" in name:
        print(f"  -> pi_deploy.py --lidar-port {PORT} --lidar-baud {baud}")
    else:
        print("  -> NOT an RPLIDAR. pi_deploy.py's RPLidarThread will not drive this;")
        print("     it needs a driver for the family above (see notes I sent).")
else:
    print("No known framing matched at any baud.")
    print("  * Confirm the disc is physically SPINNING (unpowered motor = no frames).")
    print("  * Read the model number printed on the unit.")
    print("  * If the LiDAR is wired to the Pi's GPIO UART rather than USB, the device")
    print("    is /dev/serial0 -- re-run:  python3 lidar_id.py /dev/serial0")
