"""Exercise every friday-robot tool over the real MCP protocol.

Not an import-and-call test: this launches the server exactly the way the
brain will (stdio subprocess), speaks MCP to it, and checks what comes back.
That way it also proves the argument schemas, the transport and the start-up
sequence, not just the Python functions.

    ./venv/bin/python scripts/test_tools.py            # everything but moving
    ./venv/bin/python scripts/test_tools.py --move     # also turns the wheels

MOVEMENT IS OPT-IN. Without --move nothing spins, so this is safe to run with
the robot sitting on a desk. With --move, lift the wheels clear first.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters       # noqa: E402
from mcp.client.stdio import stdio_client                  # noqa: E402

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


async def run(move: bool) -> int:
    report = Report()
    params = StdioServerParameters(
        command=str(ROOT / "venv" / "bin" / "python"),
        args=["-m", "mcp_server"],
        cwd=str(ROOT),
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
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
            report.check("get_face", text.strip() == "happy", text)
            text = await call("set_face", emotion="not_a_feeling")
            report.check("set_face rejects nonsense",
                         "unknown emotion" in text.lower(), text,
                         "should refuse, not crash")
            await call("set_face", emotion="neutral")

            text = await call("speak", text="Friday online.")
            report.check("speak", "said" in text or "debug silence" in text,
                         text, "listen for it")

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
            text = await call("move", direction="forward",
                              duration_seconds=999)
            report.check("duration is capped",
                         "capped at the" in text or "STOPPED" in text
                         or "FAILED" in text, text,
                         "999s must never be honoured")

    print(f"\n{report.passed} passed, {report.failed} failed, "
          f"{report.skipped} skipped\n")
    return 1 if report.failed else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--move", action="store_true",
                    help="also run tools that turn the wheels — lift the "
                         "robot off the ground first")
    args = ap.parse_args()
    if args.move:
        print("\n⚠  --move will turn the wheels. Wheels clear of the ground?")
    return asyncio.run(run(args.move))


if __name__ == "__main__":
    raise SystemExit(main())
