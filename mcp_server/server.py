"""server.py — the friday-robot MCP server.

The single doorway to Friday's body. Every physical capability is exposed here
and nowhere else; the brain calls these tools and never touches a device. That
is what makes safety.py's interrupt impossible to route around — there is no
other path to the motors.

Run it:
    python -m mcp_server                 # stdio — the brain spawns it
    python -m mcp_server --http          # long-running, what the service runs
    python -m mcp_server --list          # what tools exist, and exit
    python -m mcp_server --check         # what hardware answered, and exit

Two transports on purpose. stdio is the normal MCP shape, where the client
starts the server as a subprocess. But a stdio server run as a systemd daemon
has nothing attached to its stdin, so friday-mcp.service runs --http instead
and the brain connects to config.MCP_HTTP_URL.

Adding a tool: see mcp_server/registry.py, or the README. It is one decorated
function in one file in mcp_server/tools/ — nothing here needs editing.
"""

from __future__ import annotations

import logging
import sys

from mcp.server.mcpserver import MCPServer

import config
from mcp_server import hardware, reflex, registry

log = logging.getLogger("friday.mcp")

INSTRUCTIONS = """
Friday's body. These tools are the ONLY way to move, see, sense or speak.

Before any movement, call scan_environment and read safe_directions. A value
of null there means "no sensor could measure that direction" — it does not
mean clear, and it is not safe to drive into.

Movement is guarded by the server itself: forward moves stop automatically if
something comes within the stop distance, durations are capped, and stop()
interrupts a move already in progress. These limits cannot be disabled from
here, and a move that was cut short says so in its result — report that
honestly rather than assuming the move succeeded.

Distances and angles are TIMED APPROXIMATIONS. These motors have no encoders.
Say "about" and mean it.
""".strip()


def build_server() -> tuple[MCPServer, list[str]]:
    """Create the server and register everything under mcp_server/tools/."""
    modules = registry.load_tool_modules()
    server = MCPServer(
        name=config.MCP_SERVER_NAME,
        instructions=INSTRUCTIONS,
        version="0.1.0",
    )
    for invoke, name, description in registry.REGISTRY:
        server.tool(
            name=name,
            description=description,
            # Unstructured on purpose: every tool can return either its normal
            # result or a plain error string, and a declared output schema
            # would reject the error case exactly when it is most needed.
            structured_output=False,
        )(invoke)
    log.info("registered %d tools from %d modules: %s",
             len(registry.REGISTRY), len(modules), ", ".join(sorted(modules)))
    return server, modules


def _setup_logging() -> None:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        handlers=[
            # stderr, NEVER stdout: stdout is the MCP protocol channel and a
            # stray log line there corrupts the session.
            logging.StreamHandler(sys.stderr),
            logging.FileHandler(config.LOG_DIR / "mcp_server.log"),
        ],
    )


def _check() -> int:
    """Report what hardware answered.

    Asks the RUNNING service first. The serial devices allow one reader, so a
    second process opening them reports "multiple access on port" and makes
    present, working sensors look absent — which is exactly the wrong answer
    from a diagnostic. Only when nothing is already serving does this open the
    devices itself.
    """
    import asyncio

    async def ask_running() -> dict[str, str] | None:
        try:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamable_http_client
            async with streamable_http_client(config.MCP_HTTP_URL) as streams:
                async with ClientSession(streams[0], streams[1]) as session:
                    await session.initialize()
                    result = await session.call_tool("get_status", {})
                    import json
                    status = json.loads(result.content[0].text)
                    report = dict(status.get("hardware", {}))
                    report["voice"] = status.get("voice", "?")
                    return report
        except Exception:                   # noqa: BLE001 - nothing running
            return None

    live = asyncio.run(ask_running())
    if live is not None:
        print(f"(from the running server at {config.MCP_HTTP_URL})\n")
        for key, value in live.items():
            print(f"{key:16} {value}")
        return 0

    print("(no server running — opening the devices directly)\n")
    for key, value in hardware.startup().items():
        print(f"{key:16} {value}")
    hardware.shutdown()
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Friday's robot MCP server.")
    ap.add_argument("--list", action="store_true",
                    help="list the registered tools and exit")
    ap.add_argument("--check", action="store_true",
                    help="report what hardware answered and exit")
    ap.add_argument("--http", action="store_true",
                    help=f"serve over HTTP on {config.MCP_HTTP_URL} instead "
                         f"of stdio — this is what friday-mcp.service runs")
    args = ap.parse_args(argv)

    _setup_logging()
    server, modules = build_server()

    if args.list:
        print(f"{config.MCP_SERVER_NAME}: {len(registry.REGISTRY)} tools "
              f"from {', '.join(sorted(modules))}\n")
        for _, name, description in sorted(registry.REGISTRY,
                                           key=lambda item: item[1]):
            print(f"  {name:16} {description.splitlines()[0]}")
        return 0

    if args.check:
        return _check()

    log.info("starting %s (%s)", config.MCP_SERVER_NAME,
             config.MCP_HTTP_URL if args.http else "stdio")
    for key, value in hardware.startup().items():
        log.info("  %s: %s", key, value)

    # The fast loop comes up with the server and stays up for its whole life.
    # Started here rather than lazily on the first move: a reflex that only
    # exists once something has already gone right is not much of a reflex.
    reflex.start()

    # WHAT SHE CAN SEE, kept fresh in the background -- see scene.py. Guarded:
    # a watcher that cannot start costs her a description, never the server.
    try:
        import scene
        scene.start()
    except Exception as exc:                # noqa: BLE001
        log.warning("scene watcher did not start: %s: %s",
                    type(exc).__name__, exc)

    # WHICH PLACE THIS IS, and what the maps and runs amounted to (file 19):
    # a boot check with one sweep and no movement, then file polls only.
    # Guarded the same way -- it can cost a summary, never the server.
    try:
        import environment_watch
        environment_watch.start()
    except Exception as exc:                # noqa: BLE001
        log.warning("environment watch did not start: %s: %s",
                    type(exc).__name__, exc)

    # WHERE HIS LAPTOP IS (file 20): its helper says hello over UDP whenever
    # its address changes, and this keeps the saved one right. Guarded too.
    try:
        import laptop_research
        laptop_research.start_listener()
    except Exception as exc:                # noqa: BLE001
        log.warning("laptop hello listener did not start: %s: %s",
                    type(exc).__name__, exc)

    try:
        if args.http:
            server.run("streamable-http",
                       host=config.MCP_HTTP_HOST,
                       port=config.MCP_HTTP_PORT,
                       streamable_http_path=config.MCP_HTTP_PATH)
        else:
            server.run()                   # stdio, the brain spawns it
    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        # The motors are the part that can hurt. Stop them no matter how the
        # server is going down.
        reflex.stop()
        hardware.shutdown()
        log.info("stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
