"""Exercise every friday-robot tool over the real MCP protocol.

Not an import-and-call test: this launches the server exactly the way the
brain will (stdio subprocess), speaks MCP to it, and checks what comes back.
That way it also proves the argument schemas, the transport and the start-up
sequence, not just the Python functions.

    ./venv/bin/python scripts/test_tools.py            # everything but moving
    ./venv/bin/python scripts/test_tools.py --move     # also turns the wheels
    ./venv/bin/python scripts/test_tools.py --http     # test the RUNNING service

MOVEMENT IS OPT-IN. Without --move nothing spins, so this is safe to run with
the robot sitting on a desk. With --move, lift the wheels clear first.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters       # noqa: E402
from mcp.client.stdio import stdio_client                  # noqa: E402
from mcp.client.streamable_http import streamable_http_client  # noqa: E402

import config                                              # noqa: E402

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"


def render(result) -> str:
    """Flatten an MCP tool result into something printable."""
    parts = []
    for item in result.content:
        kind = getattr(item, "type", "?")
        if kind == "text":
            parts.append(item.text)
        elif kind == "image":
            data = getattr(item, "data", "") or ""
            parts.append(f"<image {getattr(item, 'mimeType', '?')}, "
                         f"{len(data)} b64 chars>")
        else:
            parts.append(f"<{kind}>")
    return "\n".join(parts)


class Report:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def check(self, name: str, ok: bool, detail: str, note: str = "") -> None:
        if ok:
            self.passed += 1
            mark = PASS
        else:
            self.failed += 1
            mark = FAIL
        head = f"  {mark}  {name}"
        if note:
            head += f"  ({note})"
        print(head)
        for line in detail.strip().splitlines()[:6]:
            print(f"        {line[:110]}")

    def skip(self, name: str, why: str) -> None:
        self.skipped += 1
        print(f"  {SKIP}  {name}  ({why})")


async def _wait_for_http(url: str, timeout_s: float = 90.0) -> None:
    """Block until something is listening on the server's port."""
    from urllib.parse import urlparse
    parsed = urlparse(url)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            _, writer = await asyncio.open_connection(host, port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            await asyncio.sleep(1.0)
    raise SystemExit(
        f"nothing is listening on {host}:{port} after {timeout_s:g}s — is "
        f"friday-mcp.service running? (systemctl --user status friday-mcp)")


@asynccontextmanager
async def connect(over_http: bool):
    """Talk to the server, either spawning it or using the running service.

    --http is the one that proves friday-mcp.service is genuinely usable, not
    merely alive: a process can sit there listening and still be broken.
    """
    if over_http:
        # Wait for the port rather than assuming it is up. Startup is not
        # instant -- Piper's voice load alone is ~10s, and closer to 25s when
        # the Pi is thermally throttling -- so connecting straight away is a
        # race that fails as "All connection attempts failed" and looks like a
        # broken server rather than a slow one.
        await _wait_for_http(config.MCP_HTTP_URL)
        async with streamable_http_client(config.MCP_HTTP_URL) as streams:
            read, write = streams[0], streams[1]
            yield read, write
    else:
        params = StdioServerParameters(
            command=str(ROOT / "venv" / "bin" / "python"),
            args=["-m", "mcp_server"],
            cwd=str(ROOT),
        )
        async with stdio_client(params) as (read, write):
            yield read, write


async def run(move: bool, over_http: bool = False) -> int:
    report = Report()

    async with connect(over_http) as (read, write):
        if True:
            session_cm = ClientSession(read, write)
        async with session_cm as session:
            await session.initialize()
            tools = await session.list_tools()
            names = sorted(t.name for t in tools.tools)
            print(f"\nconnected — {len(names)} tools: {', '.join(names)}\n")

            async def call(tool: str, **arguments):
                result = await session.call_tool(tool, arguments)
                return render(result)

            print("STATUS")
            text = await call("get_status")
            report.check("get_status", "cpu_percent" in text
                         and "battery" in text, text)

            print("\nEXPRESSION")
            text = await call("set_face", emotion="happy")
            report.check("set_face(happy)", "happy" in text, text)
            text = await call("get_face")
            # "neutral" is an accepted answer: face_director decays an
            # emotion back to neutral after its hold time, and on a throttled
            # Pi the round trip can outlast it. Reading back something in the
            # vocabulary is the real assertion -- pinning it to "happy" tests
            # the scheduler's timing, not get_face.
            report.check("get_face", text.strip() in ("happy", "neutral"),
                         text, "happy, or neutral if it already decayed")
            text = await call("set_face", emotion="not_a_feeling")
            # The refusal wording is "unknown expression"; accept either noun
            # so a phrasing change is not a test failure. What matters is that
            # a bad name is refused in words rather than raising.
            report.check("set_face rejects nonsense",
                         "unknown" in text.lower()
                         and ("emotion" in text.lower()
                              or "expression" in text.lower()), text,
                         "should refuse, not crash")
            await call("set_face", emotion="neutral")

            text = await call("speak", text="Friday online.")
            report.check("speak", "said" in text or "debug silence" in text,
                         text, "listen for it")

            print("\nEMBODIED REASONING")
            # The gate is only worth anything if it holds from a cold start,
            # so this runs before anything has observed. Ordering matters:
            # move the block and the first assertion becomes meaningless.
            text = await call("move", direction="forward", duration_seconds=0.3)
            report.check("move refused before looking", "REFUSED" in text,
                         text, "must not drive on an unseen room")
            text = await call("stop")
            report.check("stop works even ungated", "stop" in text.lower(),
                         text, "an emergency stop is never refused")
            text = await call("observe")
            report.check("observe returns a bundled view",
                         "clearance_mm" in text or "image" in text.lower(),
                         text, "camera and ranges from one moment")
            text = await call("get_status")
            report.check("status reports observation freshness",
                         "observation" in text and "has_observed" in text, text)
            text = await call("move_distance", cm=500)
            report.check("long move is clipped to one step",
                         "clipped" in text.lower(), text,
                         "small increments, not one blind commitment")

            print("\nPERCEPTION")
            text = await call("get_distance")
            report.check("get_distance", "cm ahead" in text or "cannot tell"
                         in text, text)
            text = await call("scan_environment")
            report.check("scan_environment",
                         "safe_directions" in text and "clearance_mm" in text,
                         text)
            text = await call("look", camera="up")
            report.check("look(up)", "<image" in text, text)
            text = await call("look")
            report.check("look(both)", "<image" in text, text,
                         "stitched dual view")
            text = await call("capture_image", camera="up")
            report.check("capture_image", "saved" in text, text)

            print("\nMOVEMENT")
            text = await call("set_speed", percent=50)
            report.check("set_speed(50)", "50%" in text, text)
            text = await call("set_speed", percent=500)
            report.check("set_speed clamps 500", "outside the allowed" in text,
                         text, "hard limit, not advice")
            text = await call("stop")
            report.check("stop", "stop" in text.lower(), text,
                         "must always answer")
            text = await call("move", direction="sideways",
                              duration_seconds=1)
            report.check("move rejects bad direction",
                         "not a direction" in text, text)
            # Calibrated at 18.5 cm/s and 92 deg/s, so these convert to a
            # duration rather than refusing. Either outcome is correct here:
            # a real move, a safety stop, or an honest motor-board error.
            text = await call("move_distance", cm=30)
            report.check("move_distance", "cannot do distances" in text
                         or "STOPPED" in text or "Travelled" in text
                         or "FAILED" in text, text, "quotes an error margin")
            text = await call("turn_degrees", degrees=90)
            report.check("turn_degrees", "cannot do angles" in text
                         or "Turned" in text or "STOPPED" in text
                         or "FAILED" in text, text, "quotes an error margin")

            if move:
                text = await call("move", direction="forward",
                                  duration_seconds=1.0)
                report.check("move(forward, 1s)",
                             "forward" in text.lower(), text,
                             "watch the wheels")
                text = await call("move", direction="left",
                                  duration_seconds=0.5)
                report.check("move(left, 0.5s)", "left" in text.lower(), text)
            else:
                report.skip("move(forward)", "pass --move to turn the wheels")
                report.skip("move(left)", "pass --move to turn the wheels")

            print("\nSAFETY")
            # Look first, or the observation gate refuses the move before the
            # duration cap is ever reached -- correct behaviour, but it would
            # test the gate a second time instead of testing the cap.
            # scan_environment, not observe: it records an observation the same
            # way but takes 0.1s instead of 4.6s, and observe()'s camera work is
            # slow enough under load that the reading it just took can age out
            # of the gate before the next call is made.
            await call("scan_environment")
            text = await call("move", direction="forward",
                              duration_seconds=999)
            report.check("duration is capped",
                         "capped at the" in text or "STOPPED" in text
                         or "FAILED" in text, text,
                         "999s must never be honoured")
            report.check("the cap is not a refusal", "REFUSED" not in text,
                         text, "the gate must not mask the duration cap")

    print(f"\n{report.passed} passed, {report.failed} failed, "
          f"{report.skipped} skipped\n")
    return 1 if report.failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--move", action="store_true",
                    help="also run tools that turn the wheels — lift the "
                         "robot off the ground first")
    ap.add_argument("--http", action="store_true",
                    help="test the running friday-mcp.service over HTTP "
                         "instead of spawning a fresh server over stdio")
    args = ap.parse_args()
    if args.move:
        print("\n⚠  --move will turn the wheels. Wheels clear of the ground?")
    return asyncio.run(run(args.move, args.http))


if __name__ == "__main__":
    raise SystemExit(main())
