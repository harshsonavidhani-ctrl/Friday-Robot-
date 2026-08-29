"""Test port auto-detection, including the empty-variable case that just bit us."""
import sys

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import resolve_serial_port

fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


# Exactly what is on this robot.
REAL = [
    "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A5069RR4-if00-port0",
    "/dev/serial/by-id/usb-Silicon_Labs_CP2102_USB_to_UART_Bridge_Controller_0001-if00-port0",
]

print("--- serial port resolution ---")

check("finds the LiDAR (CP2102)",
      "CP2102" in resolve_serial_port("lidar", "auto", REAL))
check("finds the motors (FTDI)",
      "FTDI" in resolve_serial_port("motor", "auto", REAL))
check("lidar and motor resolve to different devices",
      resolve_serial_port("lidar", "auto", REAL) != resolve_serial_port("motor", "auto", REAL))

# The bug we just hit: an unset shell variable becomes "" and must not be
# treated as a literal port name.
check("empty string is treated as auto",
      "CP2102" in resolve_serial_port("lidar", "", REAL))
check("None is treated as auto",
      "CP2102" in resolve_serial_port("lidar", None, REAL))

check("explicit path is passed through untouched",
      resolve_serial_port("lidar", "/dev/ttyUSB7", REAL) == "/dev/ttyUSB7")

# Nothing plugged in -> actionable error listing what IS present.
try:
    resolve_serial_port("motor", "auto", [])
    check("errors when nothing matches", False, "it returned a port")
except RuntimeError as exc:
    check("errors when nothing matches", "Could not auto-detect" in str(exc))
    check("error lists devices present", "(none present)" in str(exc))

try:
    resolve_serial_port("motor", "auto", ["/dev/serial/by-id/usb-Some_Random_Thing"])
    check("errors on non-matching device", False, "it returned a port")
except RuntimeError as exc:
    check("error shows the unmatched device", "Some_Random_Thing" in str(exc))

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)

