#!/usr/bin/env python3
"""Run every pi_deploy.py test. Use this to prove a change did not break anything.

    python3 tests/run_all.py

Exit code 0 means all suites passed. These are offline tests -- no hardware, no
serial ports, no LiDAR. They cover the parts that must stay correct regardless of
what the hardware is doing:

    test_ld19       LD19/STL-19P frame decode, CRC, angle mapping, 0 mm handling
    test_reconnect  transient vs. sustained serial faults, retry budget
    test_motors     FRIDAY-MOTORS handshake, duty cap, inversion, kickstart
    test_ports      /dev/serial/by-id auto-detection, empty-value handling
    test_creep      stiction floor vs. creep cutoff (slow request must STOP)
    test_pivot      pivots use a higher stiction floor than straight driving
    test_tof_floor  the ToF's bottom rows are the FLOOR, not obstacles
    test_frontier   the frontier channel varies (it was a dead constant)
    test_odometry   dead reckoning uses what the wheels DID, not what was asked
    test_wiring     the RIGHT motor leads are reversed; the default must correct it
    test_creep_pulse  a robot with no creep executing a creep command (invariant 5)
    test_escape     boxed in: forward blocked, no reverse -> turn toward open space
    test_telemetry  the raw policy output survives every stage that follows it
    test_world      persistent occupancy + visit memory, and the observation it must
                    NOT disturb
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITES = ["test_ld19.py", "test_reconnect.py", "test_motors.py",
          "test_ports.py", "test_creep.py", "test_pivot.py", "test_tof_floor.py",
          "test_frontier.py", "test_odometry.py", "test_wiring.py",
          "test_creep_pulse.py", "test_escape.py", "test_telemetry.py",
          "test_world.py"]

failed = []
for suite in SUITES:
    proc = subprocess.run([sys.executable, str(HERE / suite)],
                          capture_output=True, text=True)
    verdict = "PASS" if proc.returncode == 0 else "FAIL"
    print(f"[{verdict}] {suite}")
    if proc.returncode != 0:
        failed.append(suite)
        for line in (proc.stdout + proc.stderr).splitlines():
            if "FAIL" in line or "Error" in line or "error" in line:
                print(f"        {line.strip()}")

print()
if failed:
    print(f"{len(failed)} suite(s) FAILED: {', '.join(failed)}")
    print("Do not deploy. Re-run an individual suite for full output:")
    print(f"    python3 {HERE / failed[0]}")
    sys.exit(1)

print(f"All {len(SUITES)} suites passed.")
print("Also run the live safety assertions:")
print("    python3 pi_deploy.py --simulate --selftest")
sys.exit(0)
