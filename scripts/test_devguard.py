#!/usr/bin/env python3
"""test_devguard.py — try to get past the rails on Friday's self-modification.

Written adversarially on purpose. A guard tested only with the cases it was
designed for tells you nothing; these are the attempts someone would actually
make, including the ones a confused model would make by accident.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp_server import devguard                             # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_r = []


def blocked(label: str, verdict, expect_reason: str = "") -> None:
    ok = not verdict.ok
    _r.append(ok)
    print(f"  {PASS if ok else FAIL}  blocked: {label}")
    if not ok:
        print(f"        LET THROUGH — {verdict.reason or 'no reason'}")
    elif expect_reason and expect_reason.lower() not in verdict.reason.lower():
        print(f"        (reason: {verdict.reason[:70]})")


def allowed(label: str, verdict) -> None:
    ok = bool(verdict.ok)
    _r.append(ok)
    print(f"  {PASS if ok else FAIL}  allowed: {label}")
    if not ok:
        print(f"        WRONGLY BLOCKED — {verdict.reason}")


def approval(label: str, verdict) -> None:
    ok = verdict.ok and verdict.needs_approval
    _r.append(ok)
    print(f"  {PASS if ok else FAIL}  needs Harsh: {label}")
    if not ok:
        print(f"        got ok={verdict.ok} approval={verdict.needs_approval}")


print("PATHS — everything outside the repo, and the untouchables inside it\n")
blocked("/etc/passwd", devguard.check_path("/etc/passwd"))
blocked("/boot/config.txt", devguard.check_path("/boot/config.txt"))
blocked("/usr/lib/systemd/system/dbus.service",
        devguard.check_path("/usr/lib/systemd/system/dbus.service"))
blocked("escaping with ..", devguard.check_path("../../etc/shadow"))
blocked("escaping with .. from inside",
        devguard.check_path("mcp_server/../../../etc/hosts"))
blocked("the home directory above the repo",
        devguard.check_path("/home/harsh/.bashrc"))
blocked("the git database", devguard.check_path(".git/config"))
blocked("safety.py — the interrupt", devguard.check_path("mcp_server/safety.py"))
blocked("reflex.py — the veto", devguard.check_path("mcp_server/reflex.py"))
blocked("devguard.py — itself", devguard.check_path("mcp_server/devguard.py"))
blocked("rollback.sh — the way back", devguard.check_path("scripts/rollback.sh"))
blocked("CLAUDE.md — the rules", devguard.check_path("CLAUDE.md"))
blocked("absolute path to safety.py",
        devguard.check_path(str(ROOT / "mcp_server" / "safety.py")))

allowed("a new skill file", devguard.check_path("modules/battery.py"))
allowed("the dashboard", devguard.check_path("dashboard/server.py"))
allowed("a new test", devguard.check_path("scripts/test_new.py"))

approval("motor.py", devguard.check_path("motor.py"))
approval("config.py — safety limits live here", devguard.check_path("config.py"))
approval("the Arduino firmware",
         devguard.check_path("firmware/friday_motors/friday_motors.ino"))

print("\nCOMMANDS — the ones that killed this Pi, and their disguises\n")
for cmd in ("sudo apt install ros-humble",
            "apt-get remove --purge pipewire",
            "systemctl restart dbus",
            "sudo systemctl disable systemd-logind",
            "usermod -aG sudo friday",
            "passwd harsh",
            "nmcli con down wlan0",
            "sudo systemctl stop ssh",
            "echo x >> /boot/firmware/config.txt",
            "rm -rf /var/log",
            "chown -R friday:friday /usr",
            "modprobe -r brcmfmac",
            "sudo reboot",
            "dd if=/dev/zero of=/dev/mmcblk0",
            "iptables -F",
            "git push origin main",
            "crontab -e"):
    blocked(cmd, devguard.check_command(cmd))

print("\n  systemctl — allowed narrowly, refused otherwise\n")
allowed("systemctl --user restart friday-mcp",
        devguard.check_command("systemctl --user restart friday-mcp.service"))
allowed("systemctl --user status friday-face",
        devguard.check_command("systemctl --user status friday-face.service"))
blocked("systemctl restart friday-mcp (no --user, hits the system)",
        devguard.check_command("systemctl restart friday-mcp.service"))
blocked("systemctl --user mask friday-mcp (verb not allowed)",
        devguard.check_command("systemctl --user mask friday-mcp.service"))
blocked("systemctl --user restart bluetooth (not Friday's)",
        devguard.check_command("systemctl --user restart bluetooth.service"))

print("\n  ordinary work must still be possible\n")
for cmd in ("./venv/bin/python -m pytest scripts/",
            "git add -A && git commit -m 'add battery skill'",
            "./venv/bin/python scripts/test_tools.py",
            "grep -rn 'battery' modules/",
            "./venv/bin/pip install requests"):
    allowed(cmd, devguard.check_command(cmd))

print("\nDIFF REVIEW — the part that runs after the agent has finished\n")
blocked("a diff that edited safety.py",
        devguard.review_diff(["modules/battery.py", "mcp_server/safety.py"]))
blocked("a diff that wrote outside the repo",
        devguard.review_diff(["/etc/systemd/system/evil.service"]))
allowed("a clean diff", devguard.review_diff(["modules/battery.py",
                                              "scripts/test_battery.py"]))
approval("a diff touching motor.py",
         devguard.review_diff(["motor.py"]))

print("\nPORCELAIN PARSING — the bug that would have let safety.py through\n")
# `git status --porcelain` puts the path at column 3, so an unstaged change
# reads " M path" WITH A LEADING SPACE. Stripping git's output before slicing
# eats that space and cuts a character off the filename -- which silently
# turned "mcp_server/safety.py" into "p_server/safety.py", a path that matches
# nothing protected and sails through review_diff. Guard the parse itself.
sys.path.insert(0, str(ROOT))
from mcp_server.tools import dev                            # noqa: E402

_PORCELAIN = (
    (" M mcp_server/safety.py", "mcp_server/safety.py"),
    ("M  config.py", "config.py"),
    ("?? modules/new.py", "modules/new.py"),
    ("A  scripts/test_x.py", "scripts/test_x.py"),
    ("MM mcp_server/reflex.py", "mcp_server/reflex.py"),
)
for line, expected in _PORCELAIN:
    got = line[3:].strip().strip('"')
    ok = got == expected
    _r.append(ok)
    print(f"  {PASS if ok else FAIL}  {line!r} parses to {expected}")
    if not ok:
        print(f"        got {got!r}")

# And the whole path: a dirty protected file must be refused.
blocked("an unstaged edit to safety.py, parsed from real porcelain",
        devguard.review_diff([" M mcp_server/safety.py"[3:].strip()]))

passed = sum(_r)
print(f"\n{passed} passed, {len(_r) - passed} failed")
raise SystemExit(0 if all(_r) else 1)
