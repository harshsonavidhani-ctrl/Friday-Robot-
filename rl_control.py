"""rl_control.py — tell a running exploration run what to drive at.

    import rl_control
    rl_control.target("chair")        # she nudges toward the nearest chair
    rl_control.clear()                # back to the frontier planner
    rl_control.seen()                 # what she can see right now

    python3 rl_control.py --seen
    python3 rl_control.py chair

THE WRITER HALF OF rl/brain_target.py. The run reads state/rl_command.json; this
is the only thing that writes it.

⚠ DEPENDENCY-FREE ON PURPOSE, exactly like face_control.py and app_control.py:
"The face is a separate process. It watches a single text file; anything that
can write a file can drive it. No sockets, no server." The same is true here.
This module imports no robot code, opens no device and starts no thread, so the
brain, an MCP tool, the dashboard or a shell can all drive a run without any of
them being able to break it.

WHY A FILE AND NOT AN MCP CALL. The run is a separate process that does not talk
to the MCP server -- it cannot, because friday-mcp has leased it the serial ports
and stepped back (mcp_server/hardware.py). A file in state/ is the one channel
both processes already share, and it is the channel every other cross-process
command in this project uses.

THE COMMAND IS A STANDING ORDER, NOT A ONE-SHOT. Unlike app_control's "put this
on the screen now", a target stays live until it expires, is replaced, or is
cleared -- so the token exists to mark a CHANGE of order, and re-reading the same
token is a no-op. That is what makes the run's per-tick poll free.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE_DIR = ROOT / "state"
COMMAND_FILE = STATE_DIR / "rl_command.json"
# Written by the run; read here so "what can she see" needs no robot access.
AUTONOMY_FILE = STATE_DIR / "rl_autonomy.json"

DEFAULT_TTL_S = 90.0


def _write_atomic(path: Path, payload: dict, prefix: str) -> None:
    """Temp file in the SAME directory, then rename -- the rename is atomic on
    one filesystem, so the run never reads half a JSON object."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _read_json(path: Path, default: dict) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError, UnicodeDecodeError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


# =====================================================================
# the command channel
# =====================================================================

def target(label: str, ttl_s: float = DEFAULT_TTL_S) -> dict:
    """Ask the running exploration to nudge toward the nearest `label`.

    `label` must be one the on-board detector can actually emit -- the 80 COCO
    classes in rl/friday_vision.py. An unknown label is not an error here and is
    not worth making one: the run simply never sees a match, keeps exploring on
    its frontier planner, and the order expires. Failing loudly in this process
    would tell nobody anything, because the thing that knows is the run.
    """
    label = (label or "").strip().lower()
    if not label:
        raise ValueError("a target needs a label")
    message = {
        "label": label,
        "token": time.monotonic_ns(),
        "ttl_s": float(ttl_s),
        "issued_at": time.time(),
    }
    _write_atomic(COMMAND_FILE, message, ".rlcmd.")
    return message


def heading(turn_right_deg: float, ttl_s: float = 15.0) -> dict:
    """Ask the running exploration to turn this many degrees from where it faces.

    ⚠ THE PARAMETER IS NAMED FOR ITS SIGN ON PURPOSE. `turn_right_deg` is
    POSITIVE TO THE RIGHT -- the convention of the cognitive planner's
    `target_heading_offset` and of ordinary speech ("turn right 90"). The wire
    format and rl/brain_target.py are the OPPOSITE way round, POSITIVE IS LEFT,
    matching omega and bearing_error(). This function is the ONE place that
    negation happens; do it anywhere else as well and she turns away from what
    the planner reasoned about, which is the exact failure
    scripts/test_brain_target.py exists to catch.

    THE ORDER IS RELATIVE TO HER HEADING WHEN THE RUN ADOPTS IT, and the run
    turns it into an absolute heading on the first pose it sees. It then decays
    as she turns and goes quiet within 8 degrees, handing the wheels back to the
    frontier planner. So this is a nudge that ENDS -- it is not a bias that has
    to be cancelled.

    THE TTL IS SHORT BECAUSE THE PLANNER RE-ISSUES. A Tier-2 planner that dies
    mid-turn must not leave a standing order steering her; 15 s is a few planner
    ticks, so a live planner always refreshes it and a dead one stops mattering.
    """
    try:
        deg = float(turn_right_deg)
    except (TypeError, ValueError) as exc:
        raise ValueError("a heading order needs a number of degrees") from exc
    if not -180.0 <= deg <= 180.0:
        raise ValueError(f"heading {deg} is outside -180..180")
    message = {
        # negated HERE, exactly once -- see the warning above
        "heading_deg": -deg,
        "requested_turn_right_deg": deg,     # what the caller asked, unnegated
        "token": time.monotonic_ns(),
        "ttl_s": float(ttl_s),
        "issued_at": time.time(),
    }
    _write_atomic(COMMAND_FILE, message, ".rlcmd.")
    return message


def clear() -> dict:
    """Stop steering at anything; go back to exploring on the frontier planner.

    Writes an EMPTY label rather than deleting the file: a missing file and a
    cancelled order are the same to the run either way, but leaving the message
    behind means `status()` can still say what happened and when.
    """
    # ⚠ THIS DOES NOT CANCEL A HEADING ORDER INSTANTLY, and the difference
    # matters. An empty label is read as NO ORDER (read_command returns None),
    # which is the long-standing contract the run and the tests both rely on --
    # so a standing order is not replaced, it simply stops being refreshed and
    # LAPSES. For a label order that is the 90 s TTL; for a heading order it is
    # 15 s, and in practice much less, because a heading order goes quiet by
    # itself as soon as she is within 8 degrees of it. To stop her turning NOW,
    # stop the planner that is re-issuing the order -- Tier 1 is what actually
    # holds the brakes either way.
    message = {"label": "", "token": time.monotonic_ns(),
               "ttl_s": 0.0, "issued_at": time.time()}
    _write_atomic(COMMAND_FILE, message, ".rlcmd.")
    return message


def status() -> dict:
    """The standing order as last written. Not proof the run has read it."""
    return _read_json(COMMAND_FILE, {"label": "", "token": 0})


# =====================================================================
# what she can see
# =====================================================================

def seen() -> list[dict]:
    """The run's current detections: label, score, `where`, area.

    `where` is the box centre as a FRACTION of the frame, +1 hard left to -1
    hard right. It is not an angle and must not be turned into one -- see the
    warning at the top of rl/brain_target.py.

    Empty when no run is out, when the detector has seen nothing, or when the
    published state predates this feature. All three mean the same thing to a
    caller: there is nothing to aim at.
    """
    state = _read_json(AUTONOMY_FILE, {})
    found = state.get("seen")
    return found if isinstance(found, list) else []


def describe(limit: int = 3) -> str:
    """One line a person could read aloud, or "" when she sees nothing."""
    items = sorted(seen(), key=lambda d: -float(d.get("score", 0.0)))[:limit]
    if not items:
        return ""
    parts = []
    for item in items:
        where = float(item.get("where", 0.0))
        side = ("on my left" if where > 0.15 else
                "on my right" if where < -0.15 else "straight ahead")
        parts.append(f"{item.get('label', '?')} {side}")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("label", nargs="?", help="what to drive at")
    parser.add_argument("--ttl", type=float, default=DEFAULT_TTL_S,
                        help=f"seconds the order stays live (default {DEFAULT_TTL_S:g})")
    parser.add_argument("--clear", action="store_true", help="cancel the order")
    parser.add_argument("--seen", action="store_true",
                        help="what the run can see right now")
    parser.add_argument("--status", action="store_true",
                        help="the standing order as last written")
    args = parser.parse_args(argv)

    if args.seen:
        for item in sorted(seen(), key=lambda d: -float(d.get("score", 0.0))):
            print(f"{item.get('label','?'):16s} {float(item.get('score',0)):.2f}"
                  f"  where={float(item.get('where',0)):+.2f}"
                  f"  area={float(item.get('area',0)):.3f}")
        print(describe() or "(nothing in view -- is a run out, with a camera?)")
        return 0
    if args.status:
        print(json.dumps(status(), indent=2, sort_keys=True))
        return 0
    if args.clear:
        clear()
        print("order cleared -- she explores on her own again")
        return 0
    if not args.label:
        parser.print_help()
        return 2
    message = target(args.label, args.ttl)
    print(f"driving at {message['label']!r} for up to {message['ttl_s']:g}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
