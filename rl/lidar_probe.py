#!/usr/bin/env python3
"""Standalone RPLIDAR probe -- bypasses pi_deploy.py entirely.

Answers three questions with no ambiguity:
  1. Can we even OPEN the serial port (permissions / device present)?
  2. Does an RPLIDAR answer on it, and at which baud rate?
  3. What does it report about its own health?

Run:  python3 lidar_probe.py
"""
import glob
import sys

print(f"python: {sys.version.split()[0]}")

try:
    from rplidar import RPLidar, RPLidarException
except ImportError as exc:
    sys.exit(f"FATAL: rplidar package missing ({exc}). pip install rplidar-roboticia")

# --- 0. permissions / presence -------------------------------------------------
import grp
import os
import stat

user_groups = {g.gr_name for g in grp.getgrall() if os.getlogin() in g.gr_mem}
print(f"your groups: {sorted(user_groups) or '(none found)'}")

ports = sorted(glob.glob("/dev/ttyUSB*") + glob.glob("/dev/ttyACM*"))
if not ports:
    sys.exit("FATAL: no /dev/ttyUSB* or /dev/ttyACM* devices at all -- is it plugged in?")

for p in ports:
    st = os.stat(p)
    grp_name = grp.getgrgid(st.st_gid).gr_name
    readable = os.access(p, os.R_OK | os.W_OK)
    print(f"  {p}  group={grp_name:10s} mode={stat.filemode(st.st_mode)}  "
          f"rw_access={'YES' if readable else 'NO  <-- permission problem'}")

# --- 1. try every port at every plausible baud ---------------------------------
# A1/A1M8 = 115200, A2 = 115200 or 256000, A3/S1 = 256000.
for port in ports:
    for baud in (115200, 256000):
        tag = f"{port} @ {baud}"
        lidar = None
        try:
            lidar = RPLidar(port, baudrate=baud, timeout=3)
            info = lidar.get_info()
            health = lidar.get_health()
            print(f"  OK   {tag}: info={info} health={health}")
            lidar.start_motor()
            for i, scan in enumerate(lidar.iter_scans(max_buf_meas=2000)):
                print(f"       -> got scan with {len(scan)} points, "
                      f"first={scan[0] if scan else None}")
                break
            lidar.stop()
            lidar.stop_motor()
            print(f"  >>>> USE: --lidar-port {port} --lidar-baud {baud}")
        except RPLidarException as exc:
            print(f"  fail {tag}: RPLidarException: {exc}")
        except Exception as exc:                      # noqa: BLE001
            print(f"  fail {tag}: {type(exc).__name__}: {exc}")
        finally:
            if lidar is not None:
                try:
                    lidar.disconnect()
                except Exception:                     # noqa: BLE001
                    pass

print("\nIf every line says 'fail':")
print("  * permission denied      -> sudo usermod -aG dialout,plugdev $USER, then re-login")
print("  * timeout / no response  -> check 5V power to the LiDAR; is the motor spinning?")
print("  * descriptor errors      -> likely not an RPLIDAR-protocol device")
