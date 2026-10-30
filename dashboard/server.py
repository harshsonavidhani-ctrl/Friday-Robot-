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
    POST /api/client           a browser saying whether it can install the app
                               (plain-HTTP page loads from other devices are
                               redirected to HTTPS -- only that can install)
    POST /api/camera/{which}   {"on": false} -> switch a camera off
    POST /api/motor            {"direction": "forward", "seconds": 0.5}
    GET  /api/explore          is she out exploring, and what the vault holds
    POST /api/explore          {"action": "start", "minutes": 10} | {"action":
                                          "stop"}
    GET  /api/screen           what is on the 7" panel, and what is scheduled
    POST /api/screen           {"action": "alarm"|"timer"|"launch"|"dismiss"|
                                          "cancel", ...}
    GET  /api/laptop           the laptop: on?, what is in front, her game
    GET  /api/laptop/screen.png  the game window (only a game of his, in front)
    POST /api/laptop           {"action": "play"|"stop"|"open", ...} via MCP
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
import logging
import os
import re
import socket
import ssl
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from aiohttp import web

from dashboard import map_view
from dashboard import robot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app_control                                     # noqa: E402
import config                                          # noqa: E402
import face_control                                    # noqa: E402
from dashboard.cameras import BOUNDARY, CameraSet      # noqa: E402
from dashboard.events import EVENT_TYPES, EventBus, sse_format  # noqa: E402

log = logging.getLogger("friday.dashboard")

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


def ui_version() -> str:
    """A short stamp that changes when the UI files change.

    ⚠ THE PAGE IN HIS HAND IS THE OLD ONE UNTIL IT RELOADS. Harsh, 2026-09-20,
    still seeing ROBOT OFF after the fix was committed and the service
    restarted: a dashboard tab holds the JavaScript it loaded when it opened,
    and nothing on the server can change that. The service worker is
    network-first so a RELOAD gets the new file — but he has no reason to
    reload, and no way to know he should. So the page watches this stamp and
    reloads itself once when it moves.

    Mtimes, not a content hash: it costs three stat() calls on a poll that
    already happens every four seconds, and an edit that does not change the
    mtime does not exist.
    """
    stamp = 0.0
    for name in ("app.js", "style.css", "index.html"):
        try:
            stamp = max(stamp, (STATIC / name).stat().st_mtime)
        except OSError:
            continue
    return f"{int(stamp)}"


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
        "narration": narration_state(),
        "day": day_state(),
        "environments": environments_state(),
        "ui": ui_version(),
        # FILE 09 STEP 7: "Reuse the dashboard's existing data-fetching
        # pattern instead of adding a separate one." This is that pattern --
        # one poll, every panel, the way the Screen panel already rides it.
        # growth.report() is cached for GROWTH_CACHE_S so a 4 s heartbeat is
        # not nine COUNT(*) queries a beat.
        **brain_growth(),
    })


def brain_growth() -> dict:
    """The Brain Growth panel's whole payload: stats, note, preferences."""
    import growth
    import reflection
    note = reflection.last() or {}
    try:
        import corrections
        learned = corrections.preferences(
            int(getattr(config, "DASHBOARD_PREFERENCES", 8)))
    except Exception:                           # noqa: BLE001
        learned = []
    import study
    try:
        study_panel = study.panel()
    except Exception:                           # noqa: BLE001
        study_panel = {}
    import people
    try:
        lane = people.pages()
    except Exception:                           # noqa: BLE001
        lane = []
    lane = _lane_with_controls(lane)
    return {"growth": growth.report(),
            "study": study_panel,
            "people": lane,
            "reflection": {
                "note": {k: note.get(k) for k in
                         ("summary", "went_well", "to_adjust", "themes",
                          "date", "reason")} if note else None,
                "flags": reflection.flags(),
                # THE DIARY IS THIS LIST. File 12 step 2 asks for "a date and a
            # few natural sentences" rather than raw stats — which is exactly
            # what went_well/to_adjust already are, so the diary needs no
            # model call and costs nothing to render.
            "log": reflection.log_entries(
                int(getattr(config, "DASHBOARD_DIARY_ENTRIES", 20))),
                "preferences": [{"id": row.get("id"), "text": row.get("text"),
                                 "date": row.get("date")} for row in learned],
                "waiting": float(reflection._too_soon())}}


# =========================================================================
# What she has learned (file 09 step 4)
# =========================================================================
# READ-ONLY except for two buttons, and they are the whole point: reflection
# FLAGS a preference it thinks is stale or contradictory and NEVER deletes it.
# A preference is something Harsh told her to do, so the only thing allowed to
# drop one is Harsh -- see reflection.resolve().

def reflection_state() -> dict:
    """The same payload the heartbeat carries — ONE shape, not two.

    The POST handler answers with it so a button press repaints from the same
    fields the next heartbeat will bring, and /api/reflection stays for
    anything that wants just this panel.
    """
    return brain_growth()


async def api_reflection_get(request: web.Request) -> web.Response:
    return web.json_response(reflection_state())


async def api_reflection_post(request: web.Request) -> web.Response:
    """run | keep | remove. `remove` is the ONLY path that forgets anything."""
    import reflection
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    action = str(payload.get("action", "")).strip().lower()

    if action == "run":
        note = reflection.run_self_reflection("the dashboard asked",
                                              force=True)
        said = (note["summary"] if note else
                "nothing to reflect on yet — no corrections and no gaps")
    elif action in ("keep", "remove"):
        flag = reflection.resolve(str(payload.get("id", "")),
                                  keep=action == "keep")
        if flag is None:
            raise web.HTTPBadRequest(text="no such flag")
        said = ("kept" if action == "keep"
                else ("removed" if flag.get("forgotten") else "could not remove"))
    else:
        raise web.HTTPBadRequest(text="action must be run, keep or remove")
    return web.json_response({"ok": True, "result": said, **reflection_state()})


# =========================================================================
# Study mode (file 12)
# =========================================================================
# The same shape as the Screen panel's endpoint: one POST, a few actions,
# and the answer carries the whole panel back so the page repaints from the
# same fields the next heartbeat will bring.

async def api_study_post(request: web.Request) -> web.Response:
    """review | add | load — the three things a study session does."""
    import study
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    action = str(payload.get("action", "")).strip().lower()
    try:
        if action == "review":
            out = study.review(int(payload.get("card_id") or 0),
                               str(payload.get("rating", "")))
            if out is None:
                raise web.HTTPBadRequest(text="no such card")
            said = f"next in {out['interval']}"
        elif action == "add":
            row = study.add_card(str(payload.get("question", "")),
                                 str(payload.get("answer", "")),
                                 topic=str(payload.get("topic", "")))
            said = f"card {row['id']} added, due now"
        elif action == "load":
            stored = study.load_named(str(payload.get("name", "")))
            said = (f"{stored['title']}: {stored['chars']} characters in "
                    f"{stored['chunks']} chunks")
        else:
            raise web.HTTPBadRequest(text="action must be review, add or load")
    except web.HTTPException:
        raise
    except ValueError as exc:
        # A card with no answer, a file that is not in notes/, a bad rating:
        # his mistake, and the message is written to be shown as it is.
        raise web.HTTPBadRequest(text=str(exc))
    study.clear_cache()
    return web.json_response({"ok": True, "result": said,
                              "study": study.panel(max_age_s=0)})


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
            app = str(payload.get("app", ""))
            # ⚠ THE WEATHER BUTTON HAS TO GO AND GET THE WEATHER FIRST.
            # `launch("weather")` draws the card with nothing in it -- an icon
            # and a "--" where the temperature goes -- which is what Harsh saw
            # on 2026-09-19 and reported as the weather screen not showing at
            # all. set_weather() with no dict fetches through weather_source
            # (wttr.in, no key) and RAISES if it cannot, so a phone with no
            # internet gets "no weather to show: ..." in the toast instead of
            # a card that reads as broken.
            said = (app_control.set_weather() if app == "weather"
                    else app_control.launch(app))
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


# --- describing what she sees (file 15) -------------------------------
# The button is the SAME switch as "keep describing what you see": both write
# state/narration.json through narration.set_mode, and friday-brain's watch
# does the rest. Nothing here looks, calls a model or speaks -- a file write,
# answered at once whether or not the brain is up.

def narration_state() -> dict:
    try:
        import narration
        return narration.panel()
    except Exception as exc:                    # noqa: BLE001
        return {"on": False, "status": f"unavailable ({exc})"}


async def api_narration_post(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    if not isinstance(payload.get("on"), bool):
        raise web.HTTPBadRequest(text='send {"on": true} or {"on": false}')
    import narration
    narration.set_mode(payload["on"], by="dashboard")
    return web.json_response({"ok": True, **narration_state()})


# --- Laptop (file 20 + laptop games, 2026-10-21) -------------------------------
# Harsh: "can you add this feature in dashboard as well". READ: is the laptop
# on, what is in front, which game, what she is playing (laptop_games writes
# state/laptop_game.json from friday-mcp, where she plays), and -- only while
# one of HIS GAMES is the window in front -- a picture of that game window,
# the same picture she gets. WRITE: exactly three actions, all through the
# MCP tools her voice uses, so voice and dashboard steer one game:
#   {"action": "play", "game": ""} | {"action": "stop"} | {"action": "open", "name": "..."}
# Never a key, click, URL or command from here.

_LAPTOP_VERSION = {"v": 0, "at": 0.0}


def laptop_state() -> dict:
    """Blocking (network, a few seconds at worst): run in a thread."""
    import laptop_games
    import laptop_play
    import laptop_research as lr
    if not lr.configured():
        return {"configured": False}
    on = laptop_play.laptop_on(time.time())
    info = lr.idle() if on else None
    if info is not None and time.time() - _LAPTOP_VERSION["at"] > 60.0:
        _LAPTOP_VERSION.update(v=lr.helper_version(), at=time.time())
    apps = lr.known_apps()
    return {"configured": True, "on": bool(on and info is not None),
            "version": _LAPTOP_VERSION["v"] if info is not None else 0,
            "front": {k: (info or {}).get(k) for k in ("app", "title", "idle_s")},
            "game": laptop_games.game_in_front(info, apps) if info else None,
            "play": laptop_games.state(),
            "games": sorted(a["name"] for a in apps if lr.is_game(a)),
            "apps": sorted(a["name"] for a in apps)}


async def api_laptop_get(request: web.Request) -> web.Response:
    try:
        return web.json_response(await asyncio.to_thread(laptop_state))
    except Exception as exc:                        # noqa: BLE001
        return web.json_response({"configured": True, "on": False, "error": str(exc)})


def _laptop_game_picture() -> tuple[bytes | None, str]:
    import laptop_games
    import laptop_research as lr
    game = laptop_games.game_in_front(lr.idle(), lr.known_apps())
    if game is None:
        return None, "no game of yours is in front"
    return lr.screen(game)


async def api_laptop_screen(request: web.Request) -> web.Response:
    png, why = await asyncio.to_thread(_laptop_game_picture)
    if not png:
        return web.json_response({"ok": False, "why": why}, status=409)
    return web.Response(body=png, content_type="image/png",
                        headers={"Cache-Control": "no-store"})


async def api_laptop_post(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    action = str(payload.get("action", "")) if isinstance(payload, dict) else ""
    if action == "play":
        tool, args = "laptop_game_play", {"action": "start",
                                          "game": str(payload.get("game", ""))[:60]}
    elif action == "stop":
        tool, args = "laptop_game_play", {"action": "stop"}
    elif action == "open":
        name = str(payload.get("name", "")).strip()[:60]
        if not name:
            raise web.HTTPBadRequest(text="which app?")
        tool, args = "laptop_open_app", {"name": name}
    else:
        raise web.HTTPBadRequest(text="action must be play, stop or open")
    try:
        said = await robot.call(tool, args, timeout=30.0)
    except robot.RobotUnavailable as exc:
        return web.json_response({"ok": False, "why": f"her body is not answering ({exc})"})
    return web.json_response({"ok": True, "result": said})


# --- Environments (file 19) ------------------------------------------------
# READ: the environments, which one is active, the boot check, areas waiting
# for a name, per-place obstacles, the bounded run-summary log.
# WRITE: exactly ONE thing -- a name for a pending area (POST /api/landmarks).
# ⛔ This route must never grow another control (no drive, no start/stop of
# exploration, no environment switch/create -- those are his spoken words)
# without its own safety review. See CLAUDE.md PART 9.

def environments_state() -> dict:
    try:
        import environments
        import run_summary
        return {**environments.panel(), "summaries": run_summary.panel()}
    except Exception as exc:                    # noqa: BLE001
        return {"error": str(exc)}


async def api_landmarks_post(request: web.Request) -> web.Response:
    """{"pending_id": int, "name": str} -- and nothing else is accepted."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    if not isinstance(payload, dict) or set(payload) - {"pending_id", "name"}:
        raise web.HTTPBadRequest(text="only pending_id and name are accepted")
    try:
        pending_id = int(payload.get("pending_id"))
    except (TypeError, ValueError):
        raise web.HTTPBadRequest(text="pending_id must be a number")
    name = payload.get("name")
    if not isinstance(name, str):
        raise web.HTTPBadRequest(text="name must be text")
    import environments
    clean, why = environments.validate_name(name)
    if clean is None:
        raise web.HTTPBadRequest(text=f"not saved: {why}")
    said = environments.name_pending(pending_id, clean)
    if not said.startswith("Saved"):
        raise web.HTTPConflict(text=said)
    return web.json_response({"ok": True, "result": said,
                              **environments_state()})


# --- Memory Lane's two controls (file 17) --------------------------------
# THE SECOND NARROW WRITE-PATH (the first: file 14's landmark naming). It can
# do exactly two things -- ask to delete one memory (then confirm it) and
# toggle one memory's protected flag -- and nothing else. Deleting goes
# through memory_governance's pending request + token, the same function the
# voice "forget ..." uses: the first click deletes nothing.

def _lane_with_controls(pages: list) -> list:
    """Each fact gets its `protected` flag; an "All" page lists recent ones,
    because a protected memory need not be about a person."""
    try:
        import memory
        import memory_governance
        guarded = memory_governance.protected_ids()
        pages = [{**page, "facts": [{**f, "protected": f.get("id") in guarded}
                                    for f in page.get("facts") or []]}
                 for page in pages]
        recent = [{"id": f.id, "text": f.text, "kind": f.kind,
                   "created_at": f.created_at, "protected": f.id in guarded}
                  for f in memory.all_facts(limit=40)]
        if recent:
            pages.append({"name": "All", "facts": recent, "count": len(recent),
                          "last_said": "", "all": True})
    except Exception as exc:                    # noqa: BLE001
        log.debug("memory lane controls unavailable: %s", exc)
    return pages


async def api_memory_post(request: web.Request) -> web.Response:
    """delete_request {id} | delete_confirm {id, token} | protect {id, on}."""
    try:
        payload = await request.json()
        fact_id = int(payload.get("id"))
    except (json.JSONDecodeError, ValueError, TypeError):
        raise web.HTTPBadRequest(text="send JSON with an id")
    import memory_governance
    action = str(payload.get("action", "")).strip().lower()
    if action == "delete_request":
        pending = memory_governance.request_delete(fact_id, via="dashboard")
        if pending is None:
            raise web.HTTPNotFound(text="no such memory")
        return web.json_response({"ok": True, **pending})
    if action == "delete_confirm":
        gone = memory_governance.confirm_delete(str(payload.get("token", "")),
                                                fact_id)
        if not gone:
            raise web.HTTPConflict(text="not deleted — the confirmation did "
                                        "not match or expired; try again")
        return web.json_response({"ok": True, "result": "forgotten"})
    if action == "protect":
        on = bool(payload.get("on"))
        if not memory_governance.set_protected(fact_id, on):
            raise web.HTTPNotFound(text="no such memory")
        return web.json_response({"ok": True, "protected": on,
                                  "result": "protected" if on else "unprotected"})
    raise web.HTTPBadRequest(text="action must be delete_request, "
                                  "delete_confirm or protect")


# --- reminders, to-do, routines, calendar (file 16) --------------------
# The SAME rows her voice writes (reminders.py, memory.db): anything added or
# ticked off here is what she reads out, and the reverse.

def day_state() -> dict:
    try:
        import reminders
        import routines
        items = reminders.open_items()
        import day_calendar
        calendar = {"connected": True, "status": "on the robot",
                    "events": [{"id": e.get("id"), "title": e["title"],
                                "start_ts": e["start_ts"],
                                "all_day": e.get("all_day", False),
                                "local": e.get("kind") == "calendar"}
                               for e in day_calendar.merged_day()]}
        import briefing
        return {"reminders": [r for r in items if r["due_at"] is not None],
                "todos": [r for r in items if r["due_at"] is None],
                "routines": routines.describe(),
                "calendar": calendar,
                "briefing_done_today": briefing.done_today()}
    except Exception as exc:                    # noqa: BLE001
        return {"error": str(exc)}


async def api_reminders_post(request: web.Request) -> web.Response:
    """add {text, when?} | done {id} | delete {id}. File writes, answered at
    once whether or not the brain is up."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    import reminders
    action = str(payload.get("action", "")).strip().lower()
    if action == "add":
        when = str(payload.get("when") or "").strip()
        due = reminders.spoken_time(when) if when else None
        if when and due is None:
            raise web.HTTPBadRequest(
                text=f"couldn't read the time '{when}' — try 5 pm, "
                     f"tomorrow morning, in 20 minutes")
        try:
            item = reminders.add(str(payload.get("text", "")), due_at=due,
                                 source="dashboard")
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))
        said = (f"reminder set for {reminders.say_when(due)}" if due
                else "added to the to-do list")
        result = f"{item['text']} — {said}"
    elif action == "event":
        import day_calendar
        when = str(payload.get("when") or "").strip()
        start = reminders.spoken_time(when) if when else None
        if start is None:
            raise web.HTTPBadRequest(
                text="an event needs a time — e.g. tomorrow at 10")
        try:
            event = day_calendar.add(str(payload.get("text", "")), start,
                                     source="dashboard")
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc))
        result = f"{event['title']} — on the calendar {reminders.say_when(start)}"
    elif action == "event_delete":
        import day_calendar
        found = day_calendar.remove(str(payload.get("id", "")))
        result = f"removed {found['title']}" if found else "not found"
    elif action in ("done", "delete"):
        try:
            item_id = int(payload.get("id"))
        except (TypeError, ValueError):
            raise web.HTTPBadRequest(text="send an id")
        if action == "done":
            found = reminders.complete(str(item_id))
            result = f"ticked off {found['text']}" if found else "not found"
        else:
            result = "deleted" if reminders.delete(item_id) else "not found"
    else:
        raise web.HTTPBadRequest(text="action must be add, done, delete, event or event_delete")
    return web.json_response({"ok": True, "result": result, **day_state()})


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


async def cam_latest(request: web.Request) -> web.Response:
    """One fresh frame for explore_eyes, from the stream this process owns, so
    nothing ever opens a camera a second time. Loopback only."""
    if request.remote not in ("127.0.0.1", "::1"):
        raise web.HTTPForbidden()
    camera = request.app["cams"].get(request.match_info["which"])
    if camera is None or not camera.available:
        raise web.HTTPServiceUnavailable(text="no such camera right now")
    try:
        async with camera.subscribe() as queue:
            # The newest frame if it is fresh: waiting for the next one cost up
            # to 0.4 s a look. Otherwise the first item may be the last frame of
            # a stream long since stopped, and the second is new.
            if time.monotonic() - camera._latest_at <= config.DASHBOARD_FRAME_FRESH_S:
                return web.Response(body=camera._latest, content_type="image/jpeg")
            frame = await asyncio.wait_for(queue.get(), config.DASHBOARD_FRAME_WAIT_S)
            frame = await asyncio.wait_for(queue.get(), config.DASHBOARD_FRAME_WAIT_S)
    except (asyncio.TimeoutError, TimeoutError):
        # A running ffmpeg with no pictures coming out: it does not recover by
        # itself after a USB reset. Stop it, and the next look starts it afresh.
        log.warning("%s camera gave no frame in %.0f s: restarting its stream",
                    camera.name, config.DASHBOARD_FRAME_WAIT_S)
        await camera.stop()
        raise web.HTTPServiceUnavailable(text="no frame: stream restarted")
    except (OSError, RuntimeError) as exc:
        raise web.HTTPServiceUnavailable(text=f"no frame: {exc}")
    return web.Response(body=frame, content_type="image/jpeg")


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
    # THE ROBOT OWNS THE CAMERAS WHILE SHE IS OUT. Releasing them when the run
    # starts is not enough on its own: the UI remounts a dropped stream on its
    # 4 s heartbeat, so ffmpeg would take the device straight back and the run
    # would be camera-blind anyway, a few seconds later and harder to see.
    # in a thread: this is a systemctl call, and the event loop is also serving
    # her live log.
    # ...only if the run reads them. MODE_CAMERAS=0 (since 09-17): it does not,
    # so his phone can watch her explore, sharing this ffmpeg with explore_eyes.
    if (config.EXPLORE_RUN_READS_CAMERAS
            and (await asyncio.to_thread(_explore_active))[0] is not None):
        raise web.HTTPConflict(
            text=f"{which} camera belongs to the exploration run until she parks")

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


# --- exploration mode ------------------------------------------------
#
# THE MODE ITSELF IS NOT TOUCHED FROM HERE. Harsh, 2026-09-08: "no changes in
# exploration mode pls it will be untoched only it will be wired up". Every flag
# still comes from rl/modes/confident_explore.env; this only chooses how many
# minutes and presses start.

EXPLORE_DEFAULT_UNIT = "friday-explore.service"        # the frozen 20-minute run
EXPLORE_MINUTES_MAX = 120
EXPLORE_STATUS_FILE = config.ROOT / "state" / "explore_mode.json"
EXPLORE_VAULT_META = config.ROOT / "state" / "map-permanent" / "vault.json"


def _systemctl(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["systemctl", "--user", *args],
                          capture_output=True, text=True, timeout=10)


COGNITIVE_UNIT = "friday-cognitive.service"


def _explore_active() -> tuple[str | None, int | None]:
    """The exploration unit running right now, and how many minutes it is for.

    BOTH the plain unit and any @minutes instance are looked for, because a run
    can be started from here, from the CLI or by the brain -- and stopping the
    wrong one would leave a robot driving with nothing watching it.

    `activating` matters as much as `active`: these are Type=oneshot units, so
    systemd reports a run IN PROGRESS as "activating (start)". Checking only for
    "active" would report every live run as stopped.
    """
    try:
        out = _systemctl("list-units", "--no-legend", "--no-pager",
                         "--state=active", "--state=activating",
                         "friday-explore*")
    except Exception:                                   # noqa: BLE001
        return None, None
    for line in out.stdout.splitlines():
        unit = line.replace("\u25cf", " ").replace("*", " ").split()
        if not unit or not unit[0].startswith("friday-explore"):
            continue
        name = unit[0]
        found = re.search(r"@(\d+)\.service$", name)
        return name, (int(found.group(1)) if found else None)
    return None, None


def _cognitive_active() -> bool:
    """Is the Tier-2 planner up? `activating` counts -- it takes a moment."""
    try:
        out = _systemctl("is-active", COGNITIVE_UNIT)
    except Exception:                                   # noqa: BLE001
        return False
    return out.stdout.strip() in ("active", "activating")


def _explore_progress(mode: dict) -> dict:
    """How the run is actually going, read from its own log.

    "STARTED" AND "DRIVING" ARE NOT THE SAME THING, and the panel used to show
    only the first. A run takes ~45 s to reach its control loop -- policy load,
    Piper, the map, relocalisation -- and every failure so far has happened
    inside that window, so the badge said "out" for a robot that had not moved
    and was never going to. Worse, a run CAN reach the loop and still not
    explore: measured 19:12 on 2026-09-08, 550 ticks with the reflex firing on
    every one of them and the navigator handing back 548 for "no drivable arc",
    which from the sofa looks exactly like a run that never started.

    FOR THE RUN THAT IS OUT RIGHT NOW -- `_explore_state()` only calls this
    when a unit is active, and the log is resolved as "the newest one written
    since this run started", which is only unambiguous while it is live.

    Only the tail is read (the log is megabytes by the end) and every failure
    is silent: this is decoration on a panel, and it must never be able to
    break the panel it decorates.
    """
    out: dict = {"phase": "starting", "ticks": None, "moving": None,
                 "gate": None, "held_pct": None}
    # THE LOG NAMED IN explore_mode.json IS THE PREVIOUS RUN'S FOR THE WHOLE
    # RUN. explore.sh writes the status file ONCE at the start and resolves
    # "log" as the newest file in state/runs at that instant -- but
    # run_localnav.sh does not create this run's log for another ~13 s, so the
    # name recorded is the last run's, and it is never corrected until the run
    # ends. Measured 2026-09-08 20:06: a live run pointed at a 19:33 log.
    #
    # So the name is a fallback, not the answer: find the newest run log that
    # was written at or after this run started. Rejecting the stale one is not
    # enough on its own -- that only trades a wrong answer for a stuck one,
    # which is what the panel did on its first live run.
    started = mode.get("started")
    begin = None
    if started:
        try:
            begin = datetime.fromisoformat(started).timestamp()
        except (TypeError, ValueError):
            begin = None

    candidates: list[Path] = []
    try:
        runs = sorted((config.ROOT / "state" / "runs").glob("run-*.log"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        # 5 s of slack: `started` is the shell's clock and the log's mtime is
        # the kernel's, and a boundary miss here shows a live run as stuck.
        candidates = [p for p in runs
                      if begin is None or p.stat().st_mtime >= begin - 5.0]
    except OSError:
        candidates = []
    if not candidates and mode.get("log"):
        candidates = [Path(mode["log"])]
    if not candidates:
        return out

    try:
        path = candidates[0]
        stat = path.stat()
        if begin is not None and stat.st_mtime < begin - 5.0:
            return out                              # still the previous run's
        size = stat.st_size
        with path.open("rb") as handle:
            if size > 8192:
                handle.seek(-8192, os.SEEK_END)
            tail = handle.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return out
    # The per-tick line, e.g.
    #   t=  550 | lidar 0.23 | tof 0.15 | ... -> out(+0.00,-0.80) | REFLEX_STOP
    ticks = re.findall(
        r"t=\s*(\d+).*?out\(([-+][\d.]+),\s*([-+][\d.]+)\)\s*\|\s*(\S+)", tail)
    if ticks:
        # JUDGE A WINDOW, NOT ONE TICK. She pivots on a single tick while going
        # nowhere at all -- the 19:12 run commanded a turn on its last tick
        # after 550 consecutive reflex stops -- so a one-tick read calls a
        # wedged robot "moving". The last 20 logged ticks are ~100 s of real
        # behaviour and settle it.
        window = ticks[-20:]
        out["phase"] = "driving"
        out["ticks"] = int(window[-1][0])
        out["gate"] = window[-1][3]
        out["moving"] = any(float(t[1]) != 0.0 or float(t[2]) != 0.0
                            for t in window)
        held = sum(1 for t in window if t[3] != "-" and "STOP" in t[3].upper())
        out["held_pct"] = round(100 * held / len(window))
    elif "Control loop starting" in tail:
        out["phase"] = "driving"
    return out


def _explore_state() -> dict:
    """What the dashboard needs to paint the panel: is she out, for how long,
    and -- because this is the thing Harsh asked to be sure of -- how much map
    is safely in the vault."""
    unit, minutes = _explore_active()
    mode: dict = {}
    try:
        mode = json.loads(EXPLORE_STATUS_FILE.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    vault: dict = {}
    try:
        vault = json.loads(EXPLORE_VAULT_META.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {
        "running": unit is not None,
        "unit": unit,
        "minutes": minutes,
        "max_minutes": EXPLORE_MINUTES_MAX,
        # WHICH MODE SHE IS OUT IN. "thinking" is the Tier-2 planner steering
        # by heading order; without it the run is the frozen RL + frontier
        # behaviour, which is what every run before 2026-09-12 was.
        "thinking": _cognitive_active(),
        # Only meaningful while she is out; painted from the run's own log.
        "progress": _explore_progress(mode) if unit else {},
        # The mode file is written on every exit path including a kill, so the
        # last run's verdict is always readable even after a crash.
        "last": {
            "state": mode.get("state"),
            "started": mode.get("started"),
            "ended": mode.get("ended"),
            "exit_code": mode.get("exit_code"),
            "log": mode.get("log"),
        },
        "map": {
            "explored_m2": vault.get("explored_m2"),
            "saved_at": vault.get("saved_at"),
        },
    }


async def api_map_png(request: web.Request) -> web.Response:
    """Friday's exploration map as a PNG -- read-only, see dashboard/map_view.py."""
    png, meta = await asyncio.to_thread(map_view.current)
    if not png:
        raise web.HTTPNotFound(text=meta.get("why", "no map"))
    return web.Response(body=png, content_type="image/png",
                        headers={"Cache-Control": "no-store"})


async def api_map_meta(request: web.Request) -> web.Response:
    """When the map was saved, how much is known and driven, and her pose."""
    _, meta = await asyncio.to_thread(map_view.current)
    meta = dict(meta)
    # A live map IS a run: no systemctl call once a second, on the event loop.
    meta["running"] = bool(meta.get("live")) or (
        await asyncio.to_thread(_explore_active))[0] is not None
    return web.json_response(meta)


async def api_maps(request: web.Request) -> web.Response:
    """Every navigation map she has: in use, earlier versions, surveys. READ-ONLY."""
    from dashboard import maps_gallery
    try:
        return web.json_response({"ok": True, "maps":
                                  await asyncio.to_thread(maps_gallery.listing)})
    except Exception as exc:                     # noqa: BLE001
        return web.json_response({"ok": False, "why": str(exc)})


async def api_maps_png(request: web.Request) -> web.Response:
    """One listed map's picture. Takes an id FROM THE LISTING, never a path."""
    from dashboard import maps_gallery
    png = await asyncio.to_thread(maps_gallery.png, request.query.get("id", ""))
    if not png:
        raise web.HTTPNotFound(text="no such map")
    return web.Response(body=png, content_type="image/png",
                        headers={"Cache-Control": "no-store"})


async def api_explore_get(request: web.Request) -> web.Response:
    return web.json_response(_explore_state())


async def api_nav(request: web.Request) -> web.Response:
    """File 14 step 11: landmarks, navigation attempts, return-home status.

    READ-ONLY and GET-only, deliberately: there is no matching POST anywhere
    in this file for navigation. Everything else on this dashboard that moves
    the robot needs a deliberate press and says so; starting an autonomous
    drive across the flat from a phone in a pocket is not a thing worth being
    one tap away.
    """
    from dashboard import nav_view
    try:
        return web.json_response(nav_view.current())
    except Exception as exc:                     # noqa: BLE001
        log.warning("nav panel: %s", exc)
        return web.json_response({"ok": False, "why": str(exc)})


async def api_nav_post(request: web.Request) -> web.Response:
    """File 14 — send her somewhere, or set round-trip mode, from the phone.

    ⚠ THIS PANEL WAS READ-ONLY UNTIL HARSH ASKED FOR IT (2026-09-21: "via
    voice triggered and the dashboard triggered"). The reason it was read-only
    was that an autonomous drive across the flat should not be one stray tap
    away in a pocket. That concern does not go away because a button exists,
    so it is answered the way the Explore button already answers it:

      * EVERYTHING GOES THROUGH THE MCP SERVER, never the motors. The same
        clearance interrupt, duration cap and look-before-you-move gate apply
        to a tap as to a spoken command — `dashboard/robot.py` exists for
        exactly this reason.
      * The destination must be a place she already knows. The dashboard
        cannot invent a coordinate and send her to it.
      * `stop` is always available and is never gated.
    """
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")

    action = str(payload.get("action", "")).strip().lower()

    if action == "round_trip":
        import nav_state_machine as nsm
        state = nsm.set_round_trip(bool(payload.get("on")))
        return web.json_response({"ok": bool(state.get("saved")), **state})

    tools = {"go": "navigate_to", "home": "return_home",
             "come_back": "come_back", "stop": "stop"}
    if action not in tools:
        raise web.HTTPBadRequest(
            text=f"action must be one of: round_trip, {', '.join(tools)}")

    arguments: dict = {}
    if action == "go":
        name = str(payload.get("landmark", "")).strip()
        if not name:
            raise web.HTTPBadRequest(text="which place?")
        arguments["landmark_name"] = name
        if "return_to_start" in payload:
            arguments["return_to_start"] = bool(payload["return_to_start"])

    try:
        said = await robot.call(tools[action], arguments,
                                timeout=config.GOTO_BUDGET_S + 60.0)
    except robot.RobotUnavailable as exc:
        return web.json_response(
            {"ok": False, "why": f"her body is not answering ({exc})"})
    return web.json_response({"ok": True, "result": said})


# --- the exploration mission's panel (file 21 step 11, spec 4.16) ---------------
#
# GET reads everything through friday-mcp (where the mission runs); POST is
# NAMING ONLY -- exactly {area_id, name}, anything else is a 400 -- through
# the same area_store rename and landmark sync as her voice. ⛔ Never add a
# drive, start or stop to this route: the Explore panel already has those.

MISSION_NAME_KEYS = {"area_id", "name"}


async def api_mission_get(request: web.Request) -> web.Response:
    try:
        data = json.loads(await robot.call("mission_status", {}, timeout=5.0))
    except robot.RobotUnavailable as exc:
        data = {"enabled": bool(getattr(config, "EXPLORATION_MISSION_SERVICE",
                                        False)),
                "running": False, "why": f"friday-mcp is not answering ({exc})"}
    except (json.JSONDecodeError, ValueError, TypeError):
        data = {"running": False, "why": "the mission's answer was unreadable"}
    return web.json_response(data)


async def api_mission_post(request: web.Request) -> web.Response:
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")
    if not isinstance(payload, dict) or set(payload) != MISSION_NAME_KEYS:
        raise web.HTTPBadRequest(text="naming only: send exactly {area_id, name}")
    area_id, name = str(payload["area_id"]).strip(), str(payload["name"]).strip()
    if not area_id or not name:
        raise web.HTTPBadRequest(text="area_id and name must not be empty")
    try:
        answer = json.loads(await robot.call(
            "mission_name_area", {"area_id": area_id, "name": name}))
    except robot.RobotUnavailable as exc:
        raise web.HTTPServiceUnavailable(text=f"friday-mcp is not answering ({exc})")
    except (json.JSONDecodeError, ValueError, TypeError):
        raise web.HTTPBadRequest(text="the mission's answer was unreadable")
    if not answer.get("ok"):
        why = str(answer.get("why", "not saved"))
        if answer.get("suggestion"):
            why += f" — try {answer['suggestion']!r}"
        raise web.HTTPBadRequest(text=why[:300])
    return web.json_response({"ok": True, "area": answer.get("area")})


async def api_explore_post(request: web.Request) -> web.Response:
    """Start or stop a run. Two actions, both of them one systemctl call."""
    try:
        payload = await request.json()
    except (json.JSONDecodeError, ValueError):
        raise web.HTTPBadRequest(text="body must be JSON")

    action = str(payload.get("action", "")).strip().lower()
    unit, _ = _explore_active()

    if action == "stop":
        if unit is None:
            return web.json_response({"ok": True, "result": "she is not out",
                                      **_explore_state()})
        # THROUGH THE MCP SERVER WHEN IT IS THERE, because it holds the lease.
        # stop_exploration() stops the unit AND gives the serial devices back,
        # so her wheels are answerable again the moment she parks rather than
        # when the lease expires several minutes later. The unit's own
        # ExecStopPost returns the lease too -- this is the fast path, that is
        # the backstop, and both are just an unlink.
        try:
            said = await robot.call("stop_exploration")
            return web.json_response({"ok": True, "result": said,
                                      **_explore_state()})
        except robot.RobotUnavailable:
            pass                    # she is out and the server is down: stop it here
        # --no-block: the EXIT trap in run_localnav.sh has up to 90 s to put
        # her services back, and a request that waited for it would time out
        # long before.
        _systemctl("stop", "--no-block", COGNITIVE_UNIT)
        done = _systemctl("stop", "--no-block", unit)
        if done.returncode != 0:
            raise web.HTTPBadRequest(text=done.stderr.strip()[:300] or "stop failed")
        return web.json_response({"ok": True, "result": "stopping — she parks "
                                  "and her services come back",
                                  **_explore_state()})

    if action != "start":
        raise web.HTTPBadRequest(text="action must be start or stop")

    if unit is not None:
        raise web.HTTPBadRequest(text=f"already exploring ({unit})")

    try:
        minutes = int(payload.get("minutes", 0))
    except (TypeError, ValueError):
        raise web.HTTPBadRequest(text="minutes must be a whole number")
    if not 1 <= minutes <= EXPLORE_MINUTES_MAX:
        raise web.HTTPBadRequest(
            text=f"minutes must be between 1 and {EXPLORE_MINUTES_MAX}")

    # TWO EXPLORATION MODES, ONE RUN. `thinking` adds the Tier-2 planner on top
    # of the identical run -- it does not replace or reconfigure it, so the
    # plain mode stays byte-for-byte the frozen confident_explore behaviour.
    thinking = bool(payload.get("thinking", False))

    # START THROUGH THE MCP SERVER, SO IT CAN LEASE THE HARDWARE TO THE RUN.
    # Harsh, 2026-09-08: "yes wire up the lease so mcp stays up during run".
    #
    # THE LEASE ONLY WORKS FROM INSIDE friday-mcp. hardware.lease_out() writes
    # the lease file AND closes this server's own motor and range-sensor
    # handles; writing that file from here would leave friday-mcp holding
    # /dev/ttyUSB* regardless, and rl/run_localnav.sh would abort the run on
    # "serial ports are still held". So the tool call is the mechanism, not a
    # convenience -- see mcp_server/tools/autonomy.start_exploration.
    #
    # With the lease out, run_localnav.sh takes its LEASED branch and leaves
    # friday-mcp UP: the brain keeps its hands, the dashboard keeps the manual
    # controls, and the face keeps its voice, while the run owns the ports.
    # THE EXPLORATION MISSION (file 21 step 11). Harsh, 2026-09-29: "Friday,
    # explore" AND this button both start the mission (mark the start, ask the
    # rooms' names, come back). Only with config.EXPLORATION_MISSION_SERVICE
    # on, and not for `thinking` (the mission runs the plain run); otherwise
    # -- or with friday-mcp down, where the mission lives -- the button does
    # exactly what it did before.
    if getattr(config, "EXPLORATION_MISSION_SERVICE", False) and not thinking:
        try:
            answer = json.loads(await robot.call("mission_start",
                                                 {"minutes": minutes}))
            if not answer.get("ok"):
                raise web.HTTPBadRequest(
                    text=str(answer.get("why", "the mission refused"))[:300])
            # The mission's run needs the cameras exactly like a plain one
            # (see HAND THE CAMERAS BACK below): a camera-blind run looks
            # like a working one in the log.
            with contextlib.suppress(Exception):    # noqa: BLE001
                await request.app["cams"].stop_all()
            return web.json_response({"ok": True, "mission": True,
                                      "result": "exploration mission started",
                                      **_explore_state()})
        except robot.RobotUnavailable as exc:
            log.warning("friday-mcp is not answering, so no mission: starting "
                        "the plain run: %s", exc)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            log.error("mission_start returned unreadable JSON: %s", exc)
            raise web.HTTPBadRequest(text="the mission did not answer clearly")

    leased = False
    try:
        answer = json.loads(await robot.call(
            "start_exploration", {"minutes": minutes, "thinking": thinking}))
        if not answer.get("started"):
            # A refusal is an ANSWER, not a failure to start -- "I am already
            # exploring" must not fall through and start a second run on top
            # of the first, fighting it for the ports.
            raise web.HTTPBadRequest(
                text=str(answer.get("why", "the robot refused"))[:300])
        leased = True
    except robot.RobotUnavailable as exc:
        # FALL BACK RATHER THAN REFUSE. friday-mcp being down is exactly when
        # the button still has to work, and an unleased run is the OLD
        # behaviour, not a broken one: run_localnav.sh simply stops friday-mcp
        # for the duration as it always did. What is lost is the brain's hands
        # during the run, which is worth saying but not worth refusing over.
        log.warning("friday-mcp is not answering, starting the run unleased "
                    "(the brain will have no hands while she drives): %s", exc)
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        # The tool answered something this cannot read. Do not guess whether
        # the run started -- _explore_active() below is the authority.
        log.error("start_exploration returned unreadable JSON: %s", exc)

    if not leased and _explore_active()[0] is None:
        # --no-block IS NOT OPTIONAL. These are Type=oneshot units, so a plain
        # `systemctl start` does not return until the whole run has finished --
        # the request would hang for the entire exploration and this handler
        # would block the event loop with it.
        done = _systemctl("start", "--no-block",
                          f"friday-explore@{minutes}.service")
        if done.returncode != 0:
            raise web.HTTPBadRequest(
                text=done.stderr.strip()[:300] or "start failed")
        # The leased path starts the planner inside start_exploration; this is
        # the unleased fallback, which has to do it itself or "thinking" would
        # silently mean nothing whenever friday-mcp is down.
        if thinking:
            _systemctl("start", "--no-block", COGNITIVE_UNIT)

    # HAND THE CAMERAS BACK BEFORE SHE NEEDS THEM. This dashboard used to be
    # STOPPED for the duration, so its ffmpeg children released /dev/video*
    # on the way down and the robot got them by default. Keeping it up (so the
    # stop button stays reachable) removed that, and a V4L2 device does not
    # survive two streaming readers: with the camera panel open, the run comes
    # up camera-blind -- measured 2026-09-04, ZERO frames to the robot, and
    # confirmed again in the 18:41 run, whose log says "camera up not present
    # at startup" while run_localnav.sh names this process as the holder.
    #
    # A camera-blind run looks EXACTLY like a working one in the log, which is
    # why this cannot be left to a warning nobody reads. The streams come back
    # on the UI's own heartbeat once she parks.
    with contextlib.suppress(Exception):            # noqa: BLE001
        await request.app["cams"].stop_all()

    return web.json_response({
        "ok": True,
        "result": f"exploring for {minutes} min" +
                  ("" if leased else " — friday-mcp is down, so this run has "
                                     "her ports the old way and the brain has "
                                     "no hands until she parks"),
        "leased": leased,
        **_explore_state(),
    })


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
    app["tls"] = {"on": False}      # run() flips it once the HTTPS port is up

    app.router.add_get("/api/stream", api_stream)
    app.router.add_post("/api/events", api_events)
    app.router.add_post("/api/clear", api_clear)
    app.router.add_get("/api/state", api_state)
    app.router.add_get("/api/emotions", api_emotions)
    app.router.add_post("/api/mood", api_mood)
    app.router.add_get("/api/calibration", api_calibration_get)
    app.router.add_post("/api/calibration", api_calibration_post)
    app.router.add_get("/cam/{which}", cam_stream)
    app.router.add_get("/api/cam/{which}/latest.jpg", cam_latest)
    app.router.add_post("/api/camera/{which}", api_camera)
    app.router.add_post("/api/motor", api_motor)
    app.router.add_post("/api/motor/speed", api_motor_speed)
    app.router.add_get("/api/screen", api_screen_get)
    app.router.add_post("/api/screen", api_screen_post)
    app.router.add_post("/api/narration", api_narration_post)
    app.router.add_get("/api/laptop", api_laptop_get)
    app.router.add_get("/api/laptop/screen.png", api_laptop_screen)
    app.router.add_post("/api/laptop", api_laptop_post)
    app.router.add_post("/api/reminders", api_reminders_post)
    app.router.add_post("/api/memory", api_memory_post)
    app.router.add_post("/api/landmarks", api_landmarks_post)
    app.router.add_post("/api/study", api_study_post)
    app.router.add_get("/api/reflection", api_reflection_get)
    app.router.add_post("/api/reflection", api_reflection_post)
    app.router.add_get("/api/explore", api_explore_get)
    app.router.add_get("/api/nav", api_nav)
    app.router.add_post("/api/nav", api_nav_post)
    app.router.add_get("/api/map.png", api_map_png)
    app.router.add_get("/api/map", api_map_meta)
    app.router.add_get("/api/maps", api_maps)
    app.router.add_get("/api/maps/png", api_maps_png)
    app.router.add_post("/api/explore", api_explore_post)
    app.router.add_get("/api/mission", api_mission_get)
    app.router.add_post("/api/mission", api_mission_post)

    async def index(request: web.Request) -> web.StreamResponse:
        # A PHONE ON THE PLAIN-HTTP PORT CAN ONLY EVER MAKE A SHORTCUT, because
        # installing the app needs a secure page. So another device asking for
        # the page over HTTP is sent to the HTTPS one, on whatever name or
        # address it used. Loopback is left alone: that is the Pi itself.
        if (request.app["tls"]["on"] and not request.secure
                and request.remote not in ("127.0.0.1", "::1")):
            raise web.HTTPFound(
                f"https://{request.url.host}:{config.DASHBOARD_PORT}/")
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

    async def api_client(request: web.Request) -> web.Response:
        """A browser reporting whether it can install the app, and why not.

        "It only makes a shortcut" has one cause on a server that is serving
        the app correctly: the phone does not trust the page, so the browser
        refuses the service worker -- and that refusal is the only place the
        reason is ever written down. The page reports it here, one line per
        load, so the answer lands in logs/dashboard_clients.log instead of on
        a phone across the room. The reply carries the two addresses the page
        needs, built from config rather than ports written into app.js.
        """
        try:
            said = await request.json()
        except Exception:                               # noqa: BLE001
            said = {}
        if not isinstance(said, dict):
            said = {}
        row = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "from": request.remote,
               "agent": request.headers.get("User-Agent", "")[:160]}
        row.update({key: said[key] for key in
                    ("href", "secure", "standalone", "sw", "error", "prompt",
                     "installed") if key in said})
        path = config.LOG_DIR / "dashboard_clients.log"
        with contextlib.suppress(OSError):
            # One line per page load; nobody reads line five thousand.
            if path.exists() and path.stat().st_size > 256 * 1024:
                path.write_text(
                    "".join(path.read_text().splitlines(True)[-500:]))
            with path.open("a") as out:
                out.write(json.dumps(row) + "\n")
        host = request.url.host
        return web.json_response({
            "secure_url": f"https://{host}:{config.DASHBOARD_PORT}/",
            "ca_url": f"http://{host}:{config.DASHBOARD_HTTP_PORT}/ca.crt",
        })

    app.router.add_post("/api/client", api_client)

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
            app["tls"]["on"] = True
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
