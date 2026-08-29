"""Verify LD19 reconnect behaviour: recovers from transient faults, gives up on real ones."""
import sys

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pi_deploy import LD19Thread, RobotConfig, SensorSlot

cfg = RobotConfig()
fails = 0


def check(name, ok, detail=""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        fails += 1


class FlakySerial:
    """Raises on the first `n_bad` reads, then returns real bytes.

    Returning DATA (not b"") on recovery is deliberate: the driver only clears its
    failure streak when bytes actually flow. An empty read is just a timeout and proves
    nothing about the port's health, so it must not count as recovery.
    """

    def __init__(self, n_bad):
        self.n_bad, self.reads = n_bad, 0

    def read(self, n):
        self.reads += 1
        if self.reads <= self.n_bad:
            raise OSError("device reports readiness to read but returned no data")
        return b"\x00" * 16          # non-empty: proof the port is alive

    def close(self):
        pass


print("--- LD19 reconnect tests ---")

# 1. A permanently unopenable port must give up after MAX_RECONNECTS, not on attempt 1,
#    and must fail with an actionable RuntimeError rather than a raw SerialException.
th = LD19Thread(SensorSlot("lidar", 1.0, True), cfg, "/dev/definitely_not_here")
th._ser = FlakySerial(n_bad=9999)
err = None
# Budget: MAX_SOFT_RETRIES cheap recoveries first, then MAX_RECONNECTS real reopens.
for _ in range(th.MAX_SOFT_RETRIES + th.MAX_RECONNECTS + 10):
    try:
        th._poll()
    except RuntimeError as exc:
        err = exc
        break
check("gives up with RuntimeError", isinstance(err, RuntimeError))
check("consumed full retry budget", th._reconnects == th.MAX_RECONNECTS + 1,
      f"reconnects={th._reconnects} (max={th.MAX_RECONNECTS})")
check("error names the fix", err is not None and "fuser" in str(err))

# 2. A transient fault must be survived, and the failure streak must reset.
th2 = LD19Thread(SensorSlot("lidar", 1.0, True), cfg, "/dev/whatever")
flaky = FlakySerial(n_bad=1)
th2._ser = flaky


def fake_reconnect():
    th2._reconnects += 1
    th2._ser = flaky              # "reopen" succeeds


th2._reconnect = fake_reconnect
th2._poll()                        # one transient error
# A single fault must NOT tear the port down: it is recovered cheaply (flush + retry),
# because a full close/reopen costs seconds of scans for what is usually a blip.
check("single fault does not reopen the port", th2._reconnects == 0,
      f"reconnects={th2._reconnects}")
check("but the error is counted", th2._read_errors == 1, f"read_errors={th2._read_errors}")
th2._poll()                        # real bytes arrive -> soft streak resets
check("soft streak resets once data flows", th2._soft_retries == 0,
      f"soft_retries={th2._soft_retries}")

# Sustained faults must eventually escalate to a real reopen.
th5 = LD19Thread(SensorSlot("lidar", 1.0, True), cfg, "/dev/whatever")
th5._ser = FlakySerial(n_bad=9999)
th5._reconnect = lambda: setattr(th5, "_reconnects", th5._reconnects + 1)
for _ in range(th5.MAX_SOFT_RETRIES + 1):
    th5._poll()
check("sustained faults escalate to reopen", th5._reconnects == 1,
      f"after {th5.MAX_SOFT_RETRIES + 1} errors, reconnects={th5._reconnects}")

# An empty read must NOT clear the streak -- a timeout is not proof of recovery.
th3 = LD19Thread(SensorSlot("lidar", 1.0, True), cfg, "/dev/whatever")


class SilentSerial:
    def read(self, n):
        return b""

    def close(self):
        pass


th3._ser = SilentSerial()
th3._reconnects = 3
th3._poll()
check("empty read does not clear the streak", th3._reconnects == 3,
      f"reconnects={th3._reconnects}")

print(f"--- {'ALL PASSED' if not fails else f'{fails} FAILURE(S)'} ---")
sys.exit(1 if fails else 0)

