#!/usr/bin/env python3
"""Put the current laptop helper on Harsh's laptop and restart it, from the Pi
(2026-10-25: "make the laptop helper work on the first try" -- no more
"copy the new helper" for him). Over SSH (he turned on Windows' SSH server
for this, key-only):

  1. copies laptop/friday_laptop_agent.py, train_driver_net.py and
     restart_helper.ps1 into C:/Users/<him>/Friday/
  2. runs restart_helper.ps1 -RestartOnly: stops every running copy and starts
     the new one in HIS desktop session through the scheduled task he made
     (it creates and changes nothing on the laptop by itself)
  3. waits until /health answers with this copy's version

    ./venv/bin/python scripts/laptop_update.py           # copy + restart + check
    ./venv/bin/python scripts/laptop_update.py --check   # only: which version answers?
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import laptop_research as lr                              # noqa: E402

LAPTOP_USER = "Harsh Vidhani"
FILES = ("friday_laptop_agent.py", "train_driver_net.py", "restart_helper.ps1")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", "-o", "HostKeyAlias=friday-laptop"]


def wanted_version() -> int:
    src = (ROOT / "laptop" / "friday_laptop_agent.py").read_text()
    return int(re.search(r'server_version = "FridayLaptopHelper/(\d+)"', src).group(1))


def answering_version(base: str) -> int | None:
    """The helper's version from its Server header, or None if it does not answer."""
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=4) as reply:
            server = reply.headers.get("Server", "")
    except urllib.error.HTTPError as exc:            # 401 without the token: still a header
        server = exc.headers.get("Server", "")
    except Exception:                                # noqa: BLE001
        return None
    m = re.search(r"FridayLaptopHelper/(\d+)", server)
    return int(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    up, why = lr.available()                         # finds it again if its address moved
    if not up:
        print(f"the laptop is not answering ({why})")
        return 1
    base = json.loads((ROOT / "secrets" / "laptop_agent.json").read_text())["url"]
    host = urllib.parse.urlparse(base).hostname
    want, have = wanted_version(), answering_version(base)
    print(f"laptop {host}: helper {have} answering, {want} here")
    if args.check:
        return 0 if have == want else 1
    target = f"{LAPTOP_USER}@{host}"
    got = subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "HostKeyAlias=friday-laptop",
                          *[str(ROOT / "laptop" / f) for f in FILES], f"{target}:Friday/"],
                         capture_output=True, text=True, timeout=60)
    if got.returncode:
        print("copying failed:", (got.stderr or got.stdout).strip()[:300])
        return 1
    print("copied", ", ".join(FILES))
    got = subprocess.run(SSH + [target, 'powershell -NoProfile -ExecutionPolicy Bypass -File '
                                '"%USERPROFILE%\\Friday\\restart_helper.ps1" -RestartOnly'],
                         capture_output=True, text=True, timeout=90)
    print((got.stdout or "").strip())
    if got.returncode:
        print("restart failed:", (got.stderr or "").strip()[:300])
        return 1
    for _ in range(15):
        have = answering_version(base)
        if have == want:
            print(f"helper {want} is answering")
            return 0
        time.sleep(1.0)
    print(f"restarted, but the helper answering is {have}, not {want}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
