#!/usr/bin/env python3
"""Copy Harsh's driving recordings from the laptop to the Pi
(state/recordings/<session>/, gitignored) -- only files not already here.

    ./venv/bin/python scripts/fetch_recordings.py          # all sessions
    ./venv/bin/python scripts/fetch_recordings.py --list   # what the laptop has
"""
from __future__ import annotations

import argparse
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config                                            # noqa: E402
import laptop_research as lr                             # noqa: E402

DEST = ROOT / "state" / "recordings"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    got = lr._request("/record/list", None, float(config.LAPTOP_OPEN_TIMEOUT_S))
    sessions = got.get("sessions", [])
    if args.list:
        for s in sessions:
            print(s["session"], len(s["files"]), "files")
        return 0
    st = lr._settings()
    for s in sessions:
        folder = DEST / s["session"]
        folder.mkdir(parents=True, exist_ok=True)
        new = [f for f in s["files"] if not (folder / f).exists()]
        for name in new:
            url = st["url"].rstrip("/") + "/record/file?" + urllib.parse.urlencode(
                {"session": s["session"], "name": name})
            req = urllib.request.Request(url, headers={"X-Friday-Token": st["token"]})
            with urllib.request.urlopen(req, timeout=60) as reply:
                data = reply.read()
            tmp = folder / f".{name}.part"
            tmp.write_bytes(data)
            tmp.replace(folder / name)
        print(f"{s['session']}: {len(new)} new file(s), {len(s['files'])} in all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
