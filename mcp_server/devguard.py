"""devguard.py — the rails on Friday editing her own code.

Self-modification is the highest-risk feature in this project, and the reason
is written in CLAUDE.md Part 1: a previous session made this Pi unbootable and
the glued chassis had to be pried open to recover it. That happened WITHOUT any
self-modification. This file exists so that adding it cannot make that possible
again.

EVERYTHING HERE IS ENFORCED IN CODE, NOT ASKED FOR IN A PROMPT. A system prompt
saying "only edit files in the project" is a wish. The model can misread it,
be argued out of it, or simply be wrong -- and the whole point of Friday being
able to change herself is that nobody is reading the diff first. So:

  1. PATHS. Every path a change touches is resolved (symlinks and all) and must
     land inside the repository. /etc, /boot, /usr, /lib, /var and everything
     else outside is refused by check_path, not by instruction.
  2. COMMANDS. An explicit blocklist, checked by pattern, refuses apt,
     systemctl beyond status/restart of Friday's OWN units, and anything
     touching dbus, logind, PAM, users, networking, ssh or boot config.
  3. PROTECTED FILES. Friday may never edit the safety machinery, the approval
     gate, this file, or the rollback -- not with approval, not ever. A system
     that can edit its own limits has no limits.
  4. THE DIFF IS INSPECTED AFTERWARDS. This is the part that actually holds.
     The first three are things we ask of a subprocess we do not control. This
     one reads what it ACTUALLY changed, after it has finished, and reverts the
     whole branch if it went anywhere it should not have. An agent that ignores
     every instruction still cannot land a protected edit.

WHY BOTH A BLOCKLIST AND AN INSPECTION. Neither is sufficient alone. A
blocklist cannot anticipate every dangerous command, and inspection cannot undo
a command that already ran. Together: the blocklist stops the obvious damage
before it happens, the inspection catches everything else before it goes live.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

import config

REPO = Path(__file__).resolve().parent.parent


@dataclass
class Verdict:
    """Allowed, or refused with a reason Friday can say out loud."""
    ok: bool
    reason: str = ""
    needs_approval: bool = False

    def __bool__(self) -> bool:
        return self.ok


# --- 1. what may never be edited, by anyone, through this path -------------
#
# The safety machinery, the gate that guards it, and the way back. If Friday
# could edit these she could remove the limits and then remove the evidence,
# and no other rule here would matter.
PROTECTED = (
    "mcp_server/safety.py",         # the movement interrupt
    "mcp_server/reflex.py",         # the fast loop's veto
    "mcp_server/embodiment.py",     # look-before-you-move
    "mcp_server/devguard.py",       # this file
    "mcp_server/tools/dev.py",      # the tool that runs Claude Code
    "scripts/rollback.sh",          # the way back
    "scripts/test_smoke.py",        # the gate the change must pass
    "CLAUDE.md",                    # the rules themselves
)

# Changes allowed only after Harsh says yes. These can hurt the hardware or the
# person near it, but they are legitimately things Friday may need to change.
NEEDS_APPROVAL = (
    "motor.py",
    "config.py",                    # every safety threshold lives here
    "firmware/",                    # what the Arduino runs
    "mcp_server/hardware.py",
    "mcp_server/tools/movement.py",
    "sensors.py",
)

# --- 2. commands that must never run, whatever the instruction said --------
#
# Patterns, not exact strings: "sudo apt-get install" and "apt install" are the
# same intent. Matched against the whole command line, case-insensitively.
BLOCKED_COMMANDS = (
    (r"\bapt(-get)?\b", "installs or removes system packages"),
    (r"\bdpkg\b", "installs or removes system packages"),
    (r"\bsnap\b", "installs system packages"),
    (r"\bpip\s+install\b(?!.*--user)(?=.*(?:sudo|/usr|/lib))",
     "installs outside Friday's virtualenv"),
    (r"\bdbus[\w-]*\b", "D-Bus is what killed this Pi once already"),
    (r"\bsystemd-logind\b|\bloginctl\b", "the login session manager"),
    (r"\bpam[\w_-]*\b|/etc/pam", "authentication"),
    (r"\buseradd\b|\buserdel\b|\busermod\b|\bpasswd\b|\bchpasswd\b|\bgpasswd\b",
     "user accounts and passwords"),
    (r"\bvisudo\b|/etc/sudoers", "sudo configuration"),
    (r"\bnmcli\b|\bnetworkmanager\b|\bwpa_supplicant\b|\bdhcpcd\b|\biwconfig\b",
     "networking — losing wifi means losing ssh"),
    (r"\bip\s+(link|addr|route)\s+(set|add|del)", "network interfaces"),
    (r"\biptables\b|\bufw\b|\bnft\b", "firewall rules"),
    (r"\bsshd?\b|/etc/ssh", "ssh is the only remote way into this Pi"),
    (r"/boot\b|\bcmdline\.txt\b|\bconfig\.txt\b|\brpi-eeprom", "boot configuration"),
    (r"\bmkfs\b|\bfdisk\b|\bparted\b|\bdd\s+if=", "disks and filesystems"),
    (r"\bmount\b|\bumount\b|/etc/fstab", "filesystem mounts"),
    (r"\budevadm\b|/etc/udev", "device rules"),
    (r"\bmodprobe\b|\brmmod\b|\binsmod\b", "kernel modules"),
    (r"\breboot\b|\bshutdown\b|\bhalt\b|\bpoweroff\b|\bsystemctl\s+(isolate|set-default)",
     "restarting or powering off the machine"),
    (r"\bchown\b\s+-[rR]|\bchmod\b\s+-[rR]",
     "recursive ownership or permission changes"),
    (r"\brm\b\s+-[rf]{1,2}\s+/(?!home/harsh/friday)",
     "recursive deletion outside the project"),
    (r"\bgit\s+push\b", "pushing code off this machine without Harsh saying so"),
    (r"\bcrontab\b|\bat\s+now\b", "scheduled jobs"),
)

# systemctl is allowed ONLY for these verbs, and ONLY on Friday's own units.
_SYSTEMCTL_VERBS = ("status", "restart", "start", "stop", "is-active", "show",
                    "cat", "list-units")
_FRIDAY_UNITS = re.compile(r"\bfriday-[\w.-]+", re.I)


def _rel(path: str | Path) -> str | None:
    """Path relative to the repo, or None if it escapes it."""
    try:
        resolved = Path(path).expanduser()
        if not resolved.is_absolute():
            resolved = REPO / resolved
        # resolve() follows symlinks, which is the point: a symlink out of the
        # repo is exactly how you would try to get around a path check.
        resolved = resolved.resolve()
        return str(resolved.relative_to(REPO))
    except (ValueError, OSError):
        return None


def check_path(path: str | Path) -> Verdict:
    """May a change touch this path?"""
    rel = _rel(path)
    if rel is None:
        return Verdict(False, f"{path} is outside Friday's project folder. "
                              f"Changes are confined to {REPO}.")
    if rel.startswith(".git/"):
        return Verdict(False, "the git database itself is not editable — the "
                              "history is the way back")
    for protected in PROTECTED:
        if rel == protected or rel.startswith(protected.rstrip("/") + "/"):
            return Verdict(False, f"{rel} is safety machinery and can never be "
                                  f"changed this way, with or without approval. "
                                  f"Ask Harsh to edit it by hand.")
    for guarded in NEEDS_APPROVAL:
        if rel == guarded or rel.startswith(guarded.rstrip("/")):
            return Verdict(True, f"{rel} affects motors, power or safety limits",
                           needs_approval=True)
    return Verdict(True)


def check_command(command: str) -> Verdict:
    """May this shell command run?"""
    text = " ".join(command.split())
    # ⚠ MATCH CASE-INSENSITIVELY, do not lowercase the text first. Lowercasing
    # silently breaks any pattern containing an uppercase flag: "chown -R /usr"
    # stopped matching a pattern written with -R and walked straight through
    # this guard. Caught by test_devguard.py, which is the only reason it was
    # caught at all.
    lowered = text

    for pattern, why in BLOCKED_COMMANDS:
        if re.search(pattern, lowered, re.IGNORECASE):
            return Verdict(False, f"refused: that command touches {why}. "
                                  f"This is a hard limit from CLAUDE.md and it "
                                  f"is not negotiable.")

    if re.search(r"\bsystemctl\b", lowered, re.IGNORECASE):
        # Allowed narrowly: Friday may look at and bounce her OWN services.
        try:
            parts = shlex.split(text)
        except ValueError:
            return Verdict(False, "refused: could not parse that systemctl line")
        verbs = [p for p in parts if p in _SYSTEMCTL_VERBS]
        if not verbs:
            return Verdict(False, "refused: systemctl is limited to "
                                  f"{', '.join(_SYSTEMCTL_VERBS)} on Friday's "
                                  f"own units")
        if not _FRIDAY_UNITS.search(text) and "list-units" not in verbs:
            return Verdict(False, "refused: systemctl may only name Friday's "
                                  "own friday-* units, never a system unit")
        if "--user" not in parts and "list-units" not in verbs:
            return Verdict(False, "refused: Friday's units are user units — "
                                  "a systemctl without --user targets the "
                                  "system, which is out of bounds")

    return Verdict(True)


def review_diff(changed: list[str]) -> Verdict:
    """Inspect what a finished change ACTUALLY touched.

    The teeth. Everything above is asked of a subprocess we do not control;
    this reads the result. An agent that ignored every instruction still cannot
    land an edit to the safety machinery, because this runs afterwards and the
    branch is thrown away if it did.
    """
    if not changed:
        return Verdict(True, "no files changed")

    outside, protected, approval = [], [], []
    for path in changed:
        verdict = check_path(path)
        if not verdict.ok:
            (protected if _rel(path) is not None else outside).append(path)
        elif verdict.needs_approval:
            approval.append(path)

    if outside:
        return Verdict(False, "REVERTED — the change wrote outside Friday's "
                              f"project folder: {', '.join(outside)}")
    if protected:
        return Verdict(False, "REVERTED — the change edited safety machinery: "
                              f"{', '.join(protected)}. Friday may not modify "
                              f"her own limits.")
    if approval:
        return Verdict(True, "touches motors, power or safety limits: "
                             f"{', '.join(approval)}", needs_approval=True)
    return Verdict(True, f"{len(changed)} file(s), all in safe territory")


def describe() -> dict:
    """What the rails are, for get_status and for saying out loud."""
    return {
        "repo": str(REPO),
        "protected_files": list(PROTECTED),
        "needs_harsh_approval": list(NEEDS_APPROVAL),
        "blocked_command_classes": [why for _, why in BLOCKED_COMMANDS],
        "systemctl": f"only {', '.join(_SYSTEMCTL_VERBS)} on friday-* user units",
        "enforced": "in code — path resolution, command patterns, and a diff "
                    "review that reverts the branch after the fact",
    }
