"""sensors.py — range sensing: the 360° LiDAR and the 8x8 ToF.

Both are read here and nowhere else. The MCP server's scan_environment tool
fuses them into the structured answer the brain navigates by; nothing else in
the project should open these ports.

  * LiDAR — LDROBOT D500 over a CP2102 USB bridge, LD19-family binary
    protocol (config.LIDAR_PORT / LIDAR_BAUD). Gives 360° coverage at ~10 Hz.
  * ToF — DFRobot 8x8 matrix, bridged through a Pi Pico over USB CDC
    (config.TOF_PORT). NOT on I2C: /dev/i2c-1 is disabled on this Pi and
    enabling it is off-limits (CLAUDE.md Part 1). The Pico makes that
    unnecessary. Lines are "yN:d0,...,d7," in mm, 4000 = no return.

Either sensor may be absent. Every reader here degrades to None rather than
raising, and the caller reports honestly which sensors it actually had.

Not built yet — Step 0.5 fills this in.
"""

from __future__ import annotations
