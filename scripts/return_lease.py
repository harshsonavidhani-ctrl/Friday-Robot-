#!/usr/bin/env python3
"""Hand the serial devices back to friday-mcp. Run when an exploration run ends.

    scripts/return_lease.py

WHY THIS EXISTS AS ITS OWN SCRIPT. While a run is out, friday-mcp stays UP but
holds no serial port: `hardware.lease_out()` closes the motors and the range
sensors and refuses to reopen them until the lease is gone. Something has to
put that lease back, and the one place that always runs -- on a clean finish, a
crash, a SIGINT from the stop button, or systemd giving up -- is the unit's own
ExecStopPost. That is a separate process from the MCP server, so it cannot call
into the server's memory; it removes the lease FILE, and the server reopens the
devices lazily the next time a tool asks for them.

stop_exploration() also returns the lease, and so does the lease's own expiry.
This is the third of those three, and the only one that covers a run nobody
stopped and that outlived whatever started it. All three are just an unlink, so
running more than one of them is harmless.

NOT a bare `rm state/hardware.lease`: the path belongs to mcp_server.hardware,
and a second copy of it here is a second thing to get wrong the day it moves.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server import hardware                     # noqa: E402


def main() -> int:
    left = hardware.lease_active()
    hardware.lease_return()
    if left > 0.0:
        print(f"hardware lease returned ({left:.0f}s early); "
              f"friday-mcp reopens the motors and range sensors on next use.")
    else:
        print("no hardware lease was out; nothing to return.")
    return 0


if __name__ == "__main__":
    # NEVER fail the unit over this. ExecStopPost runs on the failure path too,
    # and a run that crashed must not also be reported as a failed teardown --
    # the lease expires on its own regardless.
    try:
        sys.exit(main())
    except Exception as exc:                        # noqa: BLE001
        print(f"could not return the hardware lease: "
              f"{type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(0)
