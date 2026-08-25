"""dev.py — Friday changing her own code, on rails.

    dev_task("add a skill that reports battery percentage")

Friday describes a change in words; Claude Code makes it; the change is proved
before it is allowed to matter. Nothing here trusts the agent it invokes. The
rails are in mcp_server/devguard.py and scripts/test_smoke.py, and the sequence
below is deliberately paranoid at every step:

  1. ASK FIRST if it touches motors, power or safety limits. Not afterwards.
  2. Work on a BRANCH, never on main. Main is what Friday is running.
  3. Run Claude Code with a restricted tool surface and the rails restated.
  4. READ WHAT IT ACTUALLY DID. devguard.review_diff on the real file list --
     this is the step that holds when the instructions did not.
  5. SMOKE TEST it. Safety invariants, imports, and whether the machine is
     still reachable.
  6. Merge only if all of that passed. Otherwise revert the branch entirely
     and say so.
  7. Verify ssh, wifi and login AFTER merging. If any broke, roll back at once
     and shout.

WHY A BRANCH AND A REVIEW AND A TEST, rather than any one of them. The branch
means a bad change is never what Friday is running. The review catches a change
that went somewhere it should not have, which no test would notice because the
code would work fine. The test catches a change that stayed in bounds and still
broke the robot. They fail in different directions.

FRIDAY CANNOT WIDEN THESE RAILS. devguard protects itself, safety.py, reflex.py,
the smoke test and the rollback script by name, and review_diff refuses any
diff that touches them. The approval gate is a parameter this tool controls,
not something the instruction can set.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import config
from mcp_server import devguard
from mcp_server.registry import tool

log = logging.getLogger("friday.mcp.dev")

REPO = devguard.REPO
CLAUDE = Path.home() / ".local" / "bin" / "claude"
BRANCH_PREFIX = "friday/dev"

# Restated for the agent itself. This is the WEAKEST of the protections -- it
# is a prompt, and a prompt is a wish -- which is exactly why it is not the
# only one. Everything it asks for is also enforced after the fact.
RAILS = f"""
You are making a change to Friday, a robot, from inside her own project at
{REPO}. Hard rules, all of which are checked in code after you finish:

* Work ONLY inside {REPO}. Never write to /etc, /boot, /usr, /lib, /var, or
  anywhere else outside it. A change that does is thrown away entirely.
* NEVER edit: mcp_server/safety.py, mcp_server/reflex.py,
  mcp_server/embodiment.py, mcp_server/devguard.py, mcp_server/tools/dev.py,
  scripts/test_smoke.py, scripts/rollback.sh, CLAUDE.md. These are the safety
  machinery and the rules. Touching any of them voids the whole change.
* Never run apt, dpkg, or any package manager. Never touch dbus, systemd-logind,
  PAM, user accounts, networking, ssh, or boot configuration. systemctl only as
  `systemctl --user status|restart` on friday-* units.
* Match the surrounding code: this project comments the WHY, not the what.
* Add or update a test for what you change.

Keep the change as small as it can be and still be right.
""".strip()


def _run_git(*args: str, check: bool = True) -> str:
    result = subprocess.run(("git",) + args, cwd=REPO, capture_output=True,
                            text=True)
    if check and result.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def _changed_files() -> list[str]:
    """Every path touched relative to the branch point, staged or not."""
    out = _run_git("status", "--porcelain")
    files = []
    for line in out.splitlines():
        path = line[3:].strip()
        if " -> " in path:                      # a rename reports both sides
            files.extend(p.strip() for p in path.split(" -> "))
        elif path:
            files.append(path)
    return files


def _spent_today() -> float:
    """Roughly what Friday's self-development has cost today."""
    total = 0.0
    today = datetime.now(timezone.utc).date().isoformat()
    try:
        with open(config.COST_LOG) as handle:
            for line in handle:
                entry = json.loads(line)
                if entry.get("date") == today and entry.get("kind") == "dev_task":
                    total += float(entry.get("usd", 0.0))
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return total


def _log_spend(usd: float, instruction: str) -> None:
    try:
        config.COST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(config.COST_LOG, "a") as handle:
            handle.write(json.dumps({
                "date": datetime.now(timezone.utc).date().isoformat(),
                "at": datetime.now(timezone.utc).isoformat(),
                "kind": "dev_task", "usd": usd,
                "instruction": instruction[:200]}) + "\n")
    except OSError as exc:
        log.warning("could not log spend: %s", exc)


def _verify_machine() -> tuple[bool, str]:
    """ssh, wifi and login — the three CLAUDE.md demands after every change."""
    problems = []
    ssh = subprocess.run(["systemctl", "is-active", "ssh"],
                         capture_output=True, text=True).stdout.strip()
    if ssh != "active":
        problems.append(f"ssh is {ssh}")
    listening = subprocess.run(["ss", "-ltn"], capture_output=True,
                               text=True).stdout
    if ":22 " not in listening:
        problems.append("nothing is listening on port 22")
    logind = subprocess.run(["systemctl", "is-active", "systemd-logind"],
                            capture_output=True, text=True).stdout.strip()
    if logind != "active":
        problems.append(f"systemd-logind is {logind} — login would fail")
    net = subprocess.run(["nmcli", "-t", "-f", "STATE", "general"],
                         capture_output=True, text=True).stdout.strip()
    if not net.startswith("connected"):
        problems.append(f"the network is {net or 'unknown'}")
    return (not problems), "; ".join(problems)


def _abandon(branch: str, base: str, why: str) -> str:
    """Throw the branch away and put Friday back on what she was running."""
    try:
        _run_git("checkout", "-f", base, check=False)
        _run_git("branch", "-D", branch, check=False)
    except Exception as exc:                    # noqa: BLE001
        log.error("could not abandon %s: %s", branch, exc)
        return (f"{why}\n\n⚠ AND THE BRANCH COULD NOT BE CLEANED UP ({exc}). "
                f"Tell Harsh to run ./scripts/rollback.sh")
    return f"{why}\n\nNothing changed — Friday is still running {base}."


@tool
def dev_task(instruction: str, approved: bool = False) -> str:
    """Change Friday's own code. Describe the change in plain words.

    Friday's self-development. The change is made on a branch by Claude Code,
    then reviewed, smoke-tested, and merged ONLY if it passes. If anything
    fails it is reverted completely and this says why.

    ASK HARSH FIRST for anything touching motors, power, or safety limits.
    This tool will refuse those unless approved=True, and you may only set
    that after Harsh has actually said yes out loud — never on your own
    judgement, and never because the change seems small.

    Friday can NEVER change her own safety limits, this tool, the approval
    gate, or the rollback, whatever the instruction says. Those are refused in
    code, not by convention.

    Args:
        instruction: what to change, in plain words. Be specific about the
            behaviour wanted, not the implementation.
        approved: set only when Harsh has explicitly approved a change that
            touches motors, power or safety.
    """
    instruction = (instruction or "").strip()
    if not instruction:
        return "dev_task needs an instruction — what should change?"

    if not CLAUDE.exists():
        return (f"cannot self-develop: no Claude Code at {CLAUDE}. "
                f"Ask Harsh to install it.")

    spent = _spent_today()
    if spent >= config.DAILY_SPEND_CAP_USD:
        return (f"refusing — self-development has already used about "
                f"${spent:.2f} today, at the ${config.DAILY_SPEND_CAP_USD:.2f} "
                f"cap. It resets tomorrow, or Harsh can raise the cap.")

    # The instruction itself may name something out of bounds. Cheap to check
    # before spending anything, though it is not the real defence.
    for word, path in (("safety", "mcp_server/safety.py"),
                       ("reflex", "mcp_server/reflex.py"),
                       ("rollback", "scripts/rollback.sh")):
        if word in instruction.lower() and "limit" in instruction.lower():
            return (f"refused before starting — that reads like a change to "
                    f"Friday's own safety limits ({path}). Those can only be "
                    f"changed by Harsh, by hand. This is not something I can "
                    f"be talked into.")

    try:
        base = _run_git("rev-parse", "--abbrev-ref", "HEAD")
        if _changed_files():
            return ("refusing — the working tree already has uncommitted "
                    "changes. Self-development starts from a clean state so "
                    "that what it changed is unambiguous. Ask Harsh to commit "
                    "or stash first.")
    except RuntimeError as exc:
        return f"cannot start: {exc}"

    branch = f"{BRANCH_PREFIX}-{time.strftime('%Y%m%d-%H%M%S')}"
    started = time.monotonic()
    try:
        _run_git("checkout", "-q", "-b", branch)
    except RuntimeError as exc:
        return f"could not create a working branch: {exc}"

    # --- run Claude Code, headless, scoped to the repo --------------------
    environment = dict(os.environ)
    environment["CLAUDE_PROJECT_DIR"] = str(REPO)
    prompt = f"{RAILS}\n\nTHE CHANGE TO MAKE:\n{instruction}"

    try:
        result = subprocess.run(
            [str(CLAUDE), "-p", prompt,
             "--permission-mode", "acceptEdits",
             "--add-dir", str(REPO)],
            cwd=REPO, capture_output=True, text=True,
            env=environment, timeout=config.DEV_TASK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _abandon(branch, base,
                        f"the change took longer than "
                        f"{config.DEV_TASK_TIMEOUT_S}s and was stopped.")
    except Exception as exc:                    # noqa: BLE001
        return _abandon(branch, base, f"could not run Claude Code: {exc}")

    elapsed = time.monotonic() - started
    _log_spend(config.DEV_TASK_EST_USD, instruction)
    transcript = (result.stdout or "").strip()

    # --- what did it ACTUALLY touch? --------------------------------------
    changed = _changed_files()
    if not changed:
        return _abandon(branch, base,
                        f"nothing changed after {elapsed:.0f}s. Claude Code "
                        f"said:\n{transcript[-600:]}")

    verdict = devguard.review_diff(changed)
    if not verdict.ok:
        return _abandon(branch, base,
                        f"{verdict.reason}\n\nFiles it touched: "
                        f"{', '.join(changed)}")

    if verdict.needs_approval and not approved:
        # Keep the branch this time: Harsh may well say yes, and redoing the
        # work would cost money for no reason.
        _run_git("add", "-A", check=False)
        _run_git("commit", "-q", "-m",
                 f"PROPOSED (needs Harsh): {instruction[:60]}", check=False)
        _run_git("checkout", "-q", base, check=False)
        return (f"STOPPED AND ASKING FIRST — {verdict.reason}.\n\n"
                f"I have written the change but NOT applied it. It is on "
                f"branch {branch}.\n"
                f"Files: {', '.join(changed)}\n\n"
                f"Harsh, do you want me to apply this? If you say yes I will "
                f"run it again with your approval. Review it with:\n"
                f"  git diff {base}..{branch}")

    # --- prove it before it matters ---------------------------------------
    _run_git("add", "-A", check=False)
    _run_git("commit", "-q", "-m", f"Friday: {instruction[:60]}", check=False)

    smoke = subprocess.run(
        [str(REPO / "venv" / "bin" / "python"),
         str(REPO / "scripts" / "test_smoke.py")],
        cwd=REPO, capture_output=True, text=True, timeout=300)
    if smoke.returncode != 0:
        failures = [l.strip() for l in smoke.stdout.splitlines() if "FAIL" in l]
        return _abandon(branch, base,
                        "the smoke test FAILED, so the change was reverted "
                        "rather than going live:\n  "
                        + "\n  ".join(failures[:6]))

    # --- go live ----------------------------------------------------------
    try:
        _run_git("checkout", "-q", base)
        _run_git("merge", "--no-ff", "-q", branch, "-m",
                 f"Friday's own change: {instruction[:60]}")
    except RuntimeError as exc:
        return _abandon(branch, base, f"could not merge the change: {exc}")

    ok, problems = _verify_machine()
    if not ok:
        subprocess.run([str(REPO / "scripts" / "rollback.sh")], cwd=REPO,
                       capture_output=True, timeout=180)
        return (f"⚠ THE CHANGE BROKE THE MACHINE ({problems}) — I rolled back "
                f"to the last known-good state immediately. HARSH, CHECK THIS.")

    return (f"Done, and it went live.\n"
            f"  changed: {', '.join(changed)}\n"
            f"  {verdict.reason}\n"
            f"  smoke test passed, ssh/wifi/login all still fine\n"
            f"  took {elapsed:.0f}s, branch {branch} merged into {base}\n"
            f"  undo it with: ./scripts/rollback.sh")


@tool
def dev_status() -> str:
    """What Friday is allowed to change about herself, and what she has spent.

    Worth reading out loud when asked "can you change your own code" — the
    answer is yes, within limits that are worth being specific about.
    """
    rails = devguard.describe()
    report = {
        "can_change": "anything inside " + rails["repo"],
        "can_never_change": rails["protected_files"],
        "needs_harsh_to_approve": rails["needs_harsh_approval"],
        "enforced": rails["enforced"],
        "every_change": "made on a branch, reviewed, smoke-tested, and "
                        "reverted automatically if it fails",
        "spent_today_usd": round(_spent_today(), 2),
        "daily_cap_usd": config.DAILY_SPEND_CAP_USD,
        "rollback": "./scripts/rollback.sh",
    }
    return json.dumps(report, indent=2)
