#!/usr/bin/env bash
# Read-only hardware survey. Changes nothing, installs nothing.
# Run this on the real robot, then copy the real values into config.py.

echo "=== Cameras (/dev/video*) ==="
ls -l /dev/video* 2>/dev/null || echo "  none"
echo
echo "--- stable camera paths (prefer these over indices) ---"
ls -l /dev/v4l/by-id/ 2>/dev/null || echo "  none"

echo
echo "=== USB devices ==="
lsusb 2>/dev/null || echo "  lsusb not available"

echo
echo "=== Serial devices (LiDAR lives here) ==="
ls -l /dev/serial/by-id/ 2>/dev/null || echo "  none"

echo
echo "=== I2C bus 1 (ToF matrix) ==="
if command -v i2cdetect >/dev/null 2>&1; then
    i2cdetect -y 1
else
    echo "  i2cdetect not installed (package: i2c-tools) — ask Harsh before installing"
fi

echo
echo "=== Capture devices (mic) ==="
arecord -l 2>/dev/null || echo "  arecord not available"

echo
echo "=== Playback devices (speaker) ==="
aplay -l 2>/dev/null || echo "  aplay not available"

echo
echo "Done. Nothing was modified."
