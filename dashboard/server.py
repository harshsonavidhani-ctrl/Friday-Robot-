"""Friday's dashboard server.

    python3 -m dashboard.server              https + http
    python3 -m dashboard.server --http-only  plain http (no cert needed)
    python3 -m dashboard.server --demo       replay a scripted turn, to see
                                             the log UI without a brain

Serves:
    GET  /                     the UI
    GET  /api/stream           Server-Sent Events -- the live log
    POST /api/events           ingest one event (this is how the brain logs)
    GET  /api/state            mood + camera + connection status
    GET  /api/emotions         the 20 emotion names
    POST /api/mood             {"name": "happy"} -> drives the face
    GET  /api/calibration      current motor calibration
    POST /api/calibration      save motor calibration
    GET  /cam/up  /cam/down    multipart MJPEG
    GET  /ca.crt               the local CA, to install on a phone once
    POST /api/camera/{which}   {"on": false} -> switch a camera off
    POST /api/motor            {"direction": "forward", "seconds": 0.5}
    GET  /api/screen           what is on the 7" panel, and what is scheduled
    POST /api/screen           {"action": "alarm"|"timer"|"launch"|"dismiss"|
                                          "cancel", ...}
    GET  /ca.crt               Friday's CA, to install on the phone once

Nothing here talks to the robot directly except through face_control, which
is the same file everything else uses to drive the face. MOTORS in particular
go through the friday-robot MCP server (dashboard/robot.py), never through a
serial port opened here -- see that file for why.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aiohttp import web

from dashboard import robot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app_control                                     # noqa: E402
import config                                          # noqa: E402
import face_control                                    # noqa: E402
from dashboard.cameras import BOUNDARY, CameraSet      # noqa: E402
from dashboard.events import EVENT_TYPES, EventBus, sse_format  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
STARTED = time.time()


# =====================================================================
# API
# =====================================================================

async def api_stream(request: web.Request) -> web.StreamResponse:
    """Live log over SSE. Recent history is replayed first so a browser
    that connects mid-session is not staring at an empty column."""
    bus: EventBus = request.app["bus"]
    response = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    })
    await response.prepare(request)

    queue = bus.subscribe()
    try:
        for event in bus.history():
            await response.write(sse_format(event))
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20)
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")   # hold the connection
                continue
            await response.write(sse_format(event))
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    finally:
        bus.unsubscribe(queue)
    return response


async def api_events(request: web.Request) -> web.Response:
    """Ingest. Friday's brain POSTs here; see dashboard_log.py."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    events = payload if isinstance(payload, list) else [payload]
    stored = []
    for event in events:
        if not isinstance(event, dict):
            raise web.HTTPBadRequest(text="each event must be an object")
        try:
            stored.append(request.app["bus"].emit(event))
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))
    return web.json_response({"ok": True, "accepted": len(stored)})


async def api_clear(request: web.Request) -> web.Response:
    request.app["bus"].clear()
    return web.json_response({"ok": True})


def robot_online() -> bool:
    """Is friday-mcp accepting connections?

    A bare TCP connect, not an MCP call: this is asked on every heartbeat and
    a real tool round-trip would be far too heavy for that. The dashboard has
    to load and stay usable whether or not the robot is running -- the whole
    point of it being a separate service.
    """
    try:
        with socket.create_connection(
                (config.MCP_HTTP_HOST, config.MCP_HTTP_PORT), timeout=0.4):
            return True
    except OSError:
        return False


async def api_state(request: web.Request) -> web.Response:
    cams: CameraSet = request.app["cams"]
    return web.json_response({
        "mood": face_control.get_emotion(),
        "emotions": list(face_control.EMOTIONS),
        "cameras": cams.status(),
        "viewers": request.app["bus"].subscriber_count,
        "uptime_s": time.time() - STARTED,
        "face_running": _face_is_running(),
        "robot_online": robot_online(),
        "screen": screen_state(),
    })


async def api_emotions(request: web.Request) -> web.Response:
    return web.json_response({
        "emotions": list(face_control.EMOTIONS),
        "yawn": face_control.YAWN_NAME,
    })


async def api_mood(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    # Accept either key -- "name" matches face_control's own vocabulary,
    # "mood" matches what the UI calls the control. Being strict here buys
    # nothing and makes the endpoint annoying to drive by hand.
    raw = payload.get("name", payload.get("mood", ""))
    name = str(raw).strip().lower()
    try:
        written = face_control.set_emotion(name)
    except ValueError as exc:
        raise web.HTTPBadRequest(text=str(exc))
    return web.json_response({"ok": True, "mood": face_control.get_emotion(),
                              "played": written})


# --- motors ----------------------------------------------------------
#
# Deliberately thin. Every limit -- speed cap, duration cap, the clearance
# interrupt that stops a forward move a few centimetres short of something --
# lives in the MCP server, so manual control from a phone is governed by
# exactly the same rules as the brain. Nothing is re-implemented here, because
# a second copy of a safety rule is a second chance to get it wrong.

MOTOR_DIRECTIONS = ("forward", "backward", "left", "right", "stop")


async def api_motor(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")

    direction = str(payload.get("direction", "")).strip().lower()
    if direction not in MOTOR_DIRECTIONS:
        raise web.HTTPBadRequest(
            text=f"direction must be one of {', '.join(MOTOR_DIRECTIONS)}")

    try:
        if direction == "stop":
            result = await robot.call("stop")
        else:
            # LOOK FIRST. mcp_server/embodiment.py refuses to move at all
            # unless Friday has actually observed the world inside the last
            # OBSERVATION_TTL_S -- 12 s, and only 6 s once a move has spent
            # that look. The dashboard never took one, so every press after
            # the first few seconds came back REFUSED and manual control
            # looked broken. It was the gate doing its job with nobody
            # feeding it.
            #
            # This is NOT a bypass, and must never become one: it satisfies
            # the gate the way the gate wants to be satisfied, by taking a
            # real observation immediately before the move. scan_environment
            # is the cheap one -- range sensors only, no camera -- so a held
            # button still feels responsive.
            await robot.call("scan_environment")
            seconds = float(payload.get("seconds", 0.5))
            result = await robot.call("move", {
                "direction": direction,
                "duration_seconds": seconds,
            })
    except robot.RobotUnavailable as exc:
        # 503, not 500: the dashboard is fine, the robot's server is not, and
        # the UI says so rather than showing a generic failure.
        return web.json_response({"ok": False, "result": str(exc)}, status=503)
    except (TypeError, ValueError) as exc:
        raise web.HTTPBadRequest(text=str(exc))

    # A refusal ("something is 117 mm ahead") is a SUCCESSFUL call with an
    # honest answer, not an HTTP error. The UI shows the sentence either way.
    return web.json_response({"ok": True, "result": result})


async def api_motor_speed(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
        percent = int(payload.get("percent"))
    except (json.JSONDecodeError, ValueError, TypeError):
        raise web.HTTPBadRequest(text='body must be {"percent": 1-100}')
    try:
        result = await robot.call("set_speed", {"percent": percent})
    except robot.RobotUnavailable as exc:
        return web.json_response({"ok": False, "result": str(exc)}, status=503)
    return web.json_response({"ok": True, "result": result})


# --- calibration -----------------------------------------------------

async def api_calibration_get(request: web.Request) -> web.Response:
    return web.json_response(config.load_calibration())


async def api_calibration_post(request: web.Request) -> web.Response:
    """Write calibration atomically so a half-written file can never be read
    by motor code mid-save."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")

    values = config.load_calibration()
    for key, default in config.CALIBRATION_DEFAULTS.items():
        if key not in payload:
            continue
        try:
            number = float(payload[key])
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text=f"{key} must be a number")
        if key in ("move_speed", "turn_speed"):
            if not 0 <= number <= 255:
                raise web.HTTPBadRequest(text=f"{key} must be 0-255")
            number = int(number)
        elif key == "speed_percent":
            if not 1 <= number <= config.MAX_SPEED_PERCENT:
                raise web.HTTPBadRequest(
                    text=f"speed_percent must be 1-{config.MAX_SPEED_PERCENT}")
            number = int(number)
        elif number < 0:
            raise web.HTTPBadRequest(text=f"{key} must not be negative")
        values[key] = number

    # One atomic writer, shared with set_speed, so the two cannot disagree
    # about how fast Friday drives.
    values = config.save_calibration(values)
    return web.json_response({"ok": True, **values})


# --- the 7" panel ----------------------------------------------------
#
# Same shape as the mood control: this writes app_control's state files and
# the face process picks them up. The dashboard never draws on the panel and
# never keeps its own copy of the schedule -- an alarm set from the phone and
# one set by voice are the same row in the same file, which is why one can
# cancel the other.

def screen_state() -> dict:
    """What is on the panel and what is still to come."""
    status = app_control.status()
    now = time.time()
    pending = [{
        "id": item["id"],
        "kind": item["kind"],
        "at": item["at"],
        "in_s": max(0.0, item["at"] - now),
        "when": app_control.say_time(item["at"]),
        "label": item.get("label", ""),
    } for item in app_control.pending()]
    return {"app": status.get("app"), "urgent": bool(status.get("urgent")),
            "apps": list(app_control.APP_NAMES), "pending": pending}


async def api_screen_get(request: web.Request) -> web.Response:
    return web.json_response(screen_state())


async def api_screen_post(request: web.Request) -> web.Response:
    """One endpoint, five actions. Every one of them is a file write, so it
    answers immediately whether or not the face process is up."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")

    action = str(payload.get("action", "")).strip().lower()
    label = str(payload.get("label", ""))
    try:
        if action == "alarm":
            alarm = app_control.add_alarm(str(payload.get("time", "")), label)
            said = f"alarm set for {app_control.say_time(alarm['at'])}"
        elif action == "timer":
            timer = app_control.add_timer(str(payload.get("duration", "")),
                                          label)
            app_control.launch("timer", timer_id=timer["id"])
            said = f"timer set for {app_control.say_duration(timer['seconds'])}"
        elif action == "launch":
            said = app_control.launch(str(payload.get("app", "")))
        elif action == "dismiss":
            said = app_control.dismiss()
        elif action == "cancel":
            said = app_control.cancel(str(payload.get("id", "")))
        else:
            raise web.HTTPBadRequest(
                text="action must be alarm, timer, launch, dismiss or cancel")
    except app_control.ScheduleError as exc:
        # A time Friday could not read is the phone's fault, not a fault in
        # the robot -- and the message is written to be shown as-is.
        raise web.HTTPBadRequest(text=str(exc))

    return web.json_response({"ok": True, "result": said, **screen_state()})


# --- cameras ---------------------------------------------------------

async def api_camera(request: web.Request) -> web.Response:
    """Manual on/off. Off releases ffmpeg and the V4L2 handle straight away,
    so a camera switched off here really is free -- not just hidden."""
    which = request.match_info["which"]
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    cams: CameraSet = request.app["cams"]
    try:
        on = await cams.set_enabled(which, bool(payload.get("on", True)))
    except KeyError:
        raise web.HTTPNotFound(text=f"no camera {which!r}")
    return web.json_response({"ok": True, "which": which, "on": on,
                              "cameras": cams.status()})


async def cam_stream(request: web.Request) -> web.StreamResponse:
    which = request.match_info["which"]
    cams: CameraSet = request.app["cams"]
    camera = cams.get(which)
    if camera is None:
        raise web.HTTPNotFound(text=f"no camera {which!r}")
    if not camera.enabled:
        raise web.HTTPConflict(text=f"{which} camera is switched off")
    if not camera.available:
        raise web.HTTPServiceUnavailable(
            text=f"{which} camera not present at {camera.device}")

    response = web.StreamResponse(headers={
        "Content-Type": f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
    })
    await response.prepare(request)
    try:
        async with camera.subscribe() as queue:
            while True:
                frame = await queue.get()
                await response.write(
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(frame)).encode() +
                    b"\r\n\r\n" + frame + b"\r\n")
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    except FileNotFoundError as exc:
        return web.Response(status=503, text=str(exc))
    return response


def _face_is_running() -> bool:
    """Whether friday_face.py is actually up -- the header dot reflects the
    face process, not just this server."""
    try:
        import subprocess
        out = subprocess.run(
            ["pgrep", "-f", "friday_face.py"],
            capture_output=True, text=True, timeout=2)
        # pgrep -f also matches the shell that launched it, so require a
        # line that looks like the python process itself.
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:                                   # noqa: BLE001
        return False


# =====================================================================
# App
# =====================================================================

def build_app() -> web.Application:
    app = web.Application(client_max_size=1024 * 1024)
    app["bus"] = EventBus()
    app["cams"] = CameraSet()

    app.router.add_get("/api/stream", api_stream)
    app.router.add_post("/api/events", api_events)
    app.router.add_post("/api/clear", api_clear)
    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/emotions", api_emotions)
    app.router.add_post("/api/mood", api_mood)
    app.router.add_get("/api/calibration", api_calibration_get)
    app.router.add_post("/api/calibration", api_calibration_post)
    app.router.add_get("/cam/{which}", cam_stream)
    app.router.add_post("/api/camera/{which}", api_camera)
    app.router.add_post("/api/motor", api_motor)
    app.router.add_post("/api/motor/speed", api_motor_speed)
    app.router.add_get("/api/screen", api_screen_get)
    app.router.add_post("/api/screen", api_screen_post)

    async def index(_request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / "index.html")

    app.router.add_get("/", index)
    app.router.add_static("/static/", STATIC, name="static")

    # The service worker must be served from the root to control the whole
    # scope; the manifest is fetched from the root by convention.
    async def root_file(request: web.Request) -> web.StreamResponse:
        return web.FileResponse(STATIC / Path(request.path).name)

    app.router.add_get("/sw.js", root_file)
    app.router.add_get("/manifest.webmanifest", root_file)

    async def ca_cert(request: web.Request) -> web.StreamResponse:
        """Hand out the local CA so a phone can trust the dashboard.

        Served over plain HTTP too, on purpose -- the phone has to be able to
        fetch this BEFORE it trusts the HTTPS port, otherwise there is no way
        in without clicking through the warning first.
        """
        path = Path(config.DASHBOARD_CA_CERT)
        if not path.exists():
            raise web.HTTPNotFound(
                text="no CA yet -- run ./scripts/make_dashboard_cert.sh")
        return web.FileResponse(path, headers={
            # The MIME type Android and iOS both recognise as an installable
            # certificate. Anything else and they just show it as text.
            "Content-Type": "application/x-x509-ca-cert",
            "Content-Disposition": 'attachment; filename="friday-ca.crt"',
        })

    app.router.add_get("/ca.crt", ca_cert)

    async def on_cleanup(app: web.Application) -> None:
        await app["cams"].stop_all()

    app.on_cleanup.append(on_cleanup)
    return app


def current_ips() -> set[str]:
    """Every IPv4 address the Pi currently answers on."""
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True,
                             text=True, timeout=5).stdout
    except Exception:                           # noqa: BLE001
        return set()
    return {word for word in out.split()
            if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", word)}


def cert_ips(cert: Path) -> set[str]:
    """The IP addresses listed in a certificate's SAN."""
    try:
        text = subprocess.run(
            ["openssl", "x509", "-in", str(cert), "-noout", "-text"],
            capture_output=True, text=True, timeout=5).stdout
    except Exception:                           # noqa: BLE001
        return set()
    return set(re.findall(r"IP Address:(\d+\.\d+\.\d+\.\d+)", text))


def refresh_cert_if_moved() -> bool:
    """Re-issue the server cert if the Pi's IP is not in it.

    WHY. The cert pins the addresses it is valid for, and this Pi gets its
    address from DHCP -- so a lease change silently breaks HTTPS on the phone
    with a certificate error, while everything else keeps working. That looks
    like a dashboard bug and is not one.

    Re-issuing REUSES THE CA, so the phone does not need the certificate
    installed a second time. Only --new-ca would require that, and this never
    passes it.

    Best-effort by design: if regeneration fails, the old cert is still there
    and localhost still works. Never let this stop the dashboard starting.
    """
    cert = Path(config.DASHBOARD_CERT)
    script = config.ROOT / "scripts" / "make_dashboard_cert.sh"
    if not cert.exists() or not script.exists():
        return False
    live = current_ips()
    if not live or live <= cert_ips(cert):
        return False                            # nothing new to cover
    missing = ", ".join(sorted(live - cert_ips(cert)))
    print(f"[dashboard] the Pi's address ({missing}) is not in the "
          f"certificate -- re-issuing it. The CA is unchanged, so the phone "
          f"does not need anything re-installed.", file=sys.stderr)
    try:
        done = subprocess.run([str(script)], capture_output=True, text=True,
                              timeout=60, cwd=str(config.ROOT))
        if done.returncode != 0:
            print(f"[dashboard] could not re-issue the certificate: "
                  f"{done.stderr.strip()[:200]}", file=sys.stderr)
            return False
    except Exception as exc:                    # noqa: BLE001
        print(f"[dashboard] could not re-issue the certificate: {exc}",
              file=sys.stderr)
        return False
    return True


def ssl_context() -> ssl.SSLContext | None:
    cert, key = Path(config.DASHBOARD_CERT), Path(config.DASHBOARD_KEY)
    if not (cert.exists() and key.exists()):
        return None
    try:
        refresh_cert_if_moved()
    except Exception as exc:                    # noqa: BLE001
        print(f"[dashboard] certificate check failed: {exc}", file=sys.stderr)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert, key)
    return ctx


async def run(http_only: bool = False, demo: bool = False) -> None:
    app = build_app()
    runner = web.AppRunner(app)
    await runner.setup()

    sites = []
    plain = web.TCPSite(runner, config.DASHBOARD_HOST,
                        config.DASHBOARD_HTTP_PORT)
    await plain.start()
    sites.append(f"http://{config.DASHBOARD_HOST}:{config.DASHBOARD_HTTP_PORT}")

    if not http_only:
        ctx = ssl_context()
        if ctx is None:
            print("[dashboard] no cert yet -- run scripts/make_dashboard_cert.sh "
                  "for HTTPS (needed to install as a PWA). HTTP only for now.",
                  file=sys.stderr)
        else:
            secure = web.TCPSite(runner, config.DASHBOARD_HOST,
                                 config.DASHBOARD_PORT, ssl_context=ctx)
            await secure.start()
            sites.append(
                f"https://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")

    for url in sites:
        print(f"[dashboard] listening on {url}")

    if demo:
        asyncio.create_task(_demo_loop(app["bus"]))

    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.Event().wait()
    await runner.cleanup()


async def _demo_loop(bus: EventBus) -> None:
    """Replay a realistic turn so the log UI can be checked before Friday's
    brain exists. Not used in normal operation."""
    from dashboard.events import Turn
    await asyncio.sleep(1.0)
    while True:
        turn = Turn(bus, config.MODEL_MAIN)
        turn.session_init([
            {"name": "friday-robot", "status": "connected"},
        ])
        await asyncio.sleep(1.2)
        turn.thinking("Harsh asked me to look ahead and describe what I see. "
                      "I'll grab a frame from the up camera first, then "
                      "answer briefly.")
        await asyncio.sleep(1.4)
        turn.tool_call("mcp__friday-robot__capture_image", {})
        await asyncio.sleep(0.9)
        turn.tool_result("mcp__friday-robot__capture_image",
                         {"path": "/tmp/frame.jpg", "width": 640,
                          "height": 480},
                         refs=["mcp__friday-robot__see_environment"])
        await asyncio.sleep(1.1)
        turn.tool_call("mcp__friday-robot__set_face_expression",
                       {"mood": "thinking"})
        await asyncio.sleep(0.8)
        turn.tool_result("mcp__friday-robot__set_face_expression", "OK thinking")
        await asyncio.sleep(1.0)
        turn.text("Looking now…")
        await asyncio.sleep(1.2)
        turn.done(
            "I can see a **desk** ahead with a monitor on it.\n"
            "- \U0001f4f7 clear path for about a metre\n"
            "- \U0001f9f1 wall on the left\n",
            input_tokens=2480, output_tokens=310,
            cache_read_tokens=1800)
        await asyncio.sleep(6.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Friday's dashboard")
    ap.add_argument("--http-only", action="store_true",
                    help="skip HTTPS even if a cert exists")
    ap.add_argument("--demo", action="store_true",
                    help="replay a scripted turn into the log")
    args = ap.parse_args()
    try:
        asyncio.run(run(http_only=args.http_only, demo=args.demo))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
