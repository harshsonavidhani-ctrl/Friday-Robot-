"""Measure how fast this chassis actually drives and pivots.

    ./venv/bin/python scripts/calibrate_drive.py               # dry run
    ./venv/bin/python scripts/calibrate_drive.py --turn        # PIVOTS
    ./venv/bin/python scripts/calibrate_drive.py --forward     # DRIVES
    ./venv/bin/python scripts/calibrate_drive.py --turn --forward --save

⚠ MOVES THE ROBOT. Ask Harsh first. Every move goes out through the MCP
server's `move(direction, seconds)`, so the clearance interrupt, the duration
cap and the look-before-you-move gate all apply exactly as they do to a
spoken command. Nothing here touches motor.py.

WHY THIS EXISTS. `state/calibration.json` has read forward_cm_per_s = 0 and
turn_deg_per_s = 0 since the rebuild, and `movement.move_distance()` and
`turn_degrees()` REFUSE while it does -- correctly, because an uncalibrated
distance is a guess. That refusal is why `go_to` cannot currently drive a
single leg. This measures the two numbers instead of guessing them.

HOW EACH ONE IS MEASURED, and they are measured differently on purpose.

  TURN -- by CORRELATING THE SWEEPS, not by asking the estimator.
  A pure pivot rotates the whole LiDAR picture about the robot, so the scan
  after a turn is the scan before it, shifted in bearing by the angle turned.
  Find the shift that best lines the two up and you have measured the turn
  with nothing but the sensor. This matters here more than anywhere: the
  scan matcher's search window is only SLAM_ODOM_ERROR (±20%) of whatever
  odometry CLAIMED, so when the claim is a guess -- which during calibration
  it is, that being the entire problem -- the truth can sit outside the
  window and the matcher hands the guess back as if it had confirmed it.
  scripts/pivot_test.py reported six good turns Harsh watched not happen,
  for exactly that reason. Correlation has no window and no prior.

  FORWARD -- with the range sensor as a tape measure.
  Drive at a surface and watch the distance to it shrink: before minus after
  is how far she went, in millimetres, from the one instrument on this robot
  that measures distance directly.

  ⚠ THE SCAN MATCHER WAS TRIED FIRST AND IT LIED. Asked to measure a 1 s hop
  with a 0.15 m prior it returned 0.150 m -- the prior, to the millimetre, at
  a score of 0.224 -- while the forward clearance fell from 781 mm to 288 mm.
  She had gone half a metre. The matcher may only move the pose ±20% from
  what odometry declared (SLAM_ODOM_ERROR), the truth was three times outside
  that window, so it returned the declaration and called it a match. An
  estimator cannot be used to calibrate the thing its own prior comes from.

Both are cross-checked against `scan_moved_mm` -- the median range change,
the one measure that comes from neither the estimator nor the motors -- so a
commanded move that did not physically happen is caught rather than averaged
in. On this chassis 22% of turns execute as stops.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession                                    # noqa: E402
from mcp.client.streamable_http import streamable_http_client    # noqa: E402

import config                                                    # noqa: E402
from slam.mapper import Mapper                                   # noqa: E402

# Starting guesses ONLY. The turn figure is the one scripts/calibrate_turn.py
# was written around; both are corrected by measurement within a few trials,
# and neither is ever saved unmeasured.
GUESS_TURN_DEG_S = 92.0
GUESS_FORWARD_CM_S = 15.0

# A move must shift the world by at least this much to count as having
# happened at all. Sensor noise between two stationary sweeps is a few
# millimetres; this is well clear of it and well under any real move.
REAL_MOVE_MM = 25.0

# Forward trials stop once two successive estimates agree this closely.
FORWARD_SETTLE_FRAC = 0.12
FORWARD_MAX_ROUNDS = 4


@asynccontextmanager
async def session():
    async with streamable_http_client(config.MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as client:
            await client.initialize()
            yield client


def text_of(result) -> str:
    for block in result.content:
        if getattr(block, "text", None):
            return block.text
    return ""


async def sweep(client) -> dict[int, int]:
    data = json.loads(text_of(await client.call_tool("lidar_sweep",
                                                     {"full": True})))
    return {int(b): int(mm) for b, mm in data.get("bearings_mm", {}).items()}


def scan_moved_mm(before: dict[int, int], after: dict[int, int]) -> float | None:
    """Median absolute range change. Did the world move past the LiDAR at all?"""
    shared = [abs(after[b] - before[b]) for b in before if b in after]
    if len(shared) < 40:
        return None
    shared.sort()
    return float(shared[len(shared) // 2])


def rotation_deg(before: dict[int, int], after: dict[int, int]
                 ) -> tuple[float | None, float]:
    """The bearing shift that best lines `after` up with `before`.

    Returns (degrees, how much better than the runner-up). Positive is the
    direction of increasing bearing; the caller decides what that means in
    left/right terms by looking at which way it commanded.

    Searched over every whole degree, so there is no window to fall outside
    of. The score is a MEDIAN absolute difference rather than a mean, so a
    person walking through one sector of the sweep cannot drag the answer --
    the same reason `scan_moved_mm` uses one.
    """
    best: list[tuple[float, int]] = []
    for shift in range(-180, 181):
        diffs = [abs(after[(b + shift) % 360] - before[b])
                 for b in before if (b + shift) % 360 in after]
        if len(diffs) < 40:
            continue
        diffs.sort()
        best.append((diffs[len(diffs) // 2], shift))
    if len(best) < 10:
        return None, 0.0
    best.sort()
    winner = best[0]
    # How decisive it is: the best score against the median of all of them.
    spread = statistics.median(s for s, _ in best)
    margin = (spread - winner[0]) / spread if spread > 0 else 0.0
    return float(winner[1]), float(margin)


async def settle(client, seconds: float = None) -> None:
    """Let the rolling sweep window clear after a move.

    A scan is a rolling revolution, so the instant a move ends the newest
    window still holds bearings measured while the robot was moving. Folding
    or correlating those compares two different moments and smears the
    answer. One full revolution and every point postdates the stop.
    """
    await asyncio.sleep(seconds or max(config.LIDAR_SWEEP_MAX_AGE_S * 3, 0.5))


async def drive(client, direction: str, seconds: float) -> str:
    """One guarded move. Observes first, because the gate requires it."""
    await client.call_tool("scan_environment", {})
    said = text_of(await client.call_tool(
        "move", {"direction": direction, "duration_seconds": seconds}))
    await settle(client)
    return said


async def clearance(client) -> dict:
    return json.loads(text_of(await client.call_tool("scan_environment", {})))


# --------------------------------------------------------------------------

async def measure_turn(client, seconds: float, direction: str) -> dict:
    """One pivot, measured by correlating the sweeps either side of it."""
    before = await sweep(client)
    said = await drive(client, direction, seconds)
    after = await sweep(client)

    moved = scan_moved_mm(before, after)
    shift, margin = rotation_deg(before, after)
    out = {"direction": direction, "seconds": seconds, "said": said,
           "moved_mm": moved, "shift_deg": shift, "margin": round(margin, 3)}
    if moved is None or moved < REAL_MOVE_MM:
        out["verdict"] = "THE TURN DID NOT HAPPEN"
        return out
    if shift is None:
        out["verdict"] = "could not correlate the sweeps"
        return out
    # A right pivot turns the robot clockwise, so the world appears to rotate
    # the other way. Only the magnitude is used for a rate.
    out["deg_s"] = abs(shift) / seconds
    out["verdict"] = "ok"
    return out


async def measure_forward(client, seconds: float, _guess=None) -> dict:
    """One forward hop, measured by how much closer the wall ahead got.

    Returns cm/s, the millimetres travelled, and whether the guard cut the
    move short -- which it will, at SAFETY_STOP_MM, and which must be
    honoured or the rate comes out low. `move` says so in its own words and
    those words are parsed rather than assumed.
    """
    before = (await clearance(client))["clearance_mm"]["forward"]
    said = await drive(client, "forward", seconds)
    after = (await clearance(client))["clearance_mm"]["forward"]

    # "STOPPED after 0.4s ..." when the interrupt fired.
    actual = seconds
    cut = "STOPPED" in said.upper()
    if cut:
        import re
        found = re.search(r"after\s+([0-9.]+)\s*s", said)
        if found:
            actual = float(found.group(1))

    travelled_mm = before - after
    out = {"seconds": seconds, "actual_s": actual, "said": said.strip()[:90],
           "before_mm": before, "after_mm": after,
           "travelled_mm": travelled_mm, "cut_short": cut}
    if travelled_mm < REAL_MOVE_MM:
        out["verdict"] = "THE MOVE DID NOT HAPPEN"
        return out
    if actual <= 0.05:
        out["verdict"] = "stopped before it started"
        return out
    out["cm_s"] = travelled_mm / 10.0 / actual
    out["verdict"] = "ok"
    return out


# --------------------------------------------------------------------------

async def run_turns(client) -> float | None:
    """Pivot both ways at several durations; return degrees per second."""
    print("\n— TURN —  pivots in place, each one undone by the next")
    rates: list[float] = []
    # Paired left/right so she ends where she began and a one-off scuff on
    # the carpet cannot pass as a result.
    for seconds in (0.5, 0.8, 1.2):
        for direction in ("right", "left"):
            out = await measure_turn(client, seconds, direction)
            got = out.get("deg_s")
            print(f"  {direction:>5} {seconds:>4.1f}s  "
                  f"world moved {out['moved_mm'] or 0:>5.0f} mm  "
                  f"shift {str(out['shift_deg']):>5}°  "
                  f"margin {out['margin']:.2f}  "
                  + (f"→ {got:.0f} °/s" if got else f"→ {out['verdict']}"))
            if got and out["margin"] > 0.10:
                rates.append(got)
    if not rates:
        print("  no usable turn — nothing measured, nothing saved")
        return None
    rate = statistics.median(rates)
    print(f"  {len(rates)} good turns, median {rate:.1f} °/s "
          f"(spread {min(rates):.0f}–{max(rates):.0f})")
    return rate


async def run_forward(client) -> float | None:
    """Back off to make room, then hop at the wall and watch it come closer."""
    print("\n— FORWARD —  the wall ahead is the tape measure")
    rates: list[float] = []

    for seconds in (0.6, 0.6, 1.0, 1.0):
        room = (await clearance(client))["clearance_mm"]["forward"]
        # Enough space that the guard does not cut the hop short: what she is
        # expected to travel, plus the stop distance, plus a margin.
        want = (rates[-1] if rates else GUESS_FORWARD_CM_S) * seconds * 10
        if room < want + config.SAFETY_STOP_MM + 120:
            back = (await clearance(client))["clearance_mm"]["backward"]
            if back < 500:
                print(f"  only {room} mm ahead and {back} mm behind — "
                      "no room to measure in; move her somewhere open")
                break
            print(f"  {room} mm ahead is not enough; backing off")
            await drive(client, "backward", 1.2)
            continue

        out = await measure_forward(client, seconds)
        got = out.get("cm_s")
        print(f"  {seconds:.1f}s: {out['before_mm']} → {out['after_mm']} mm "
              f"= {out['travelled_mm']} mm in {out['actual_s']:.2f}s"
              + ("  [cut short]" if out["cut_short"] else "")
              + (f"  → {got:.1f} cm/s" if got else f"  → {out['verdict']}"))
        if got:
            rates.append(got)

    if not rates:
        print("  nothing measured — nothing saved")
        return None
    rate = statistics.median(rates)
    print(f"  {len(rates)} good hops, median {rate:.1f} cm/s "
          f"(spread {min(rates):.0f}–{max(rates):.0f})")
    return rate


def save(forward_cm_s: float | None, turn_deg_s: float | None) -> None:
    path = ROOT / "state" / "calibration.json"
    cal = json.loads(path.read_text())
    before = dict(cal)
    if forward_cm_s:
        cal["forward_cm_per_s"] = round(forward_cm_s, 1)
    if turn_deg_s:
        cal["turn_deg_per_s"] = round(turn_deg_s, 1)
    cal["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    cal["calibrated_at_speed_percent"] = cal.get("speed_percent")
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(cal, indent=2) + "\n")
    tmp.replace(path)
    print(f"\nsaved {path.relative_to(ROOT)}")
    for key in ("forward_cm_per_s", "turn_deg_per_s"):
        print(f"  {key}: {before[key]} → {cal[key]}")
    print("  ⚠ these are true AT speed_percent "
          f"{cal.get('speed_percent')}. Change the speed and they are wrong.")


async def main() -> int:
    do_turn = "--turn" in sys.argv
    do_forward = "--forward" in sys.argv
    do_save = "--save" in sys.argv

    print(f"connecting to {config.MCP_HTTP_URL}")
    async with session() as client:
        first = await sweep(client)
        if not first:
            print("no sweep — the LiDAR is not answering; nothing to measure")
            return 1
        room = await clearance(client)
        print(f"sensor ok, {len(first)} bearings")
        print(f"clearance: {json.dumps(room['clearance_mm'])}")
        print(f"speed_percent: {config.load_calibration().get('speed_percent')}")

        if not (do_turn or do_forward):
            # A DRY RUN THAT ACTUALLY PROVES SOMETHING. Two sweeps with the
            # robot standing still: the world must not have moved and the
            # correlation must say zero. If it does not, the measurement is
            # broken and no amount of driving will fix it.
            await asyncio.sleep(0.4)
            second = await sweep(client)
            moved = scan_moved_mm(first, second)
            shift, margin = rotation_deg(first, second)
            print(f"\ndry run — standing still:")
            print(f"  world moved {moved:.0f} mm (expect a few)")
            print(f"  correlation says {shift}° (expect 0), "
                  f"margin {margin:.2f}")
            ok = moved is not None and moved < REAL_MOVE_MM and shift == 0
            print("  measurement is sane" if ok else
                  "  ⚠ MEASUREMENT IS NOT SANE — do not drive on this")
            print("\npass --turn and/or --forward to measure. Both move her.")
            return 0 if ok else 1

        turn_rate = await run_turns(client) if do_turn else None
        forward_rate = await run_forward(client) if do_forward else None

        if do_save and (turn_rate or forward_rate):
            save(forward_rate, turn_rate)
        elif turn_rate or forward_rate:
            print("\n(not saved — pass --save)")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
