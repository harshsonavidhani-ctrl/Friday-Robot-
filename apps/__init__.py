"""apps — small full-screen views Friday can show instead of her face.

The face is the default view and the resting state. An app takes the screen
for as long as it has something to say -- a clock, a countdown, an alarm going
off -- and hands it straight back when it is done.

    manager = apps.Manager(screen_size)
    manager.launch("timer", seconds=600)
    manager.update(dt); manager.draw(surface)     # returns False if no app is up

ADDING AN APP IS ONE FILE. Drop a module in this package with a subclass of
apps.base.App and a NAME, and it is registered by existing -- the same shape as
mcp_server/tools/, and for the same reason: the interesting part of adding a
feature should be the feature.

THE FACE OUTRANKS EVERY APP WHEN IT MATTERS. A safety expression -- scared,
dizzy -- is not decoration and must not be hidden behind a weather card. The
manager is asked to yield for those, and does, whatever app is running and
however much it would rather not. See friday_face.py's draw().
"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil
import time
from pathlib import Path

import app_control
import config

from apps.base import App, AppError                         # noqa: F401

log = logging.getLogger("friday.apps")

_REGISTRY: dict[str, type[App]] = {}


def register(app_class: type[App]) -> type[App]:
    """Called by each module at import. Also usable as a decorator."""
    name = getattr(app_class, "NAME", "").strip().lower()
    if not name:
        raise AppError(f"{app_class.__name__} has no NAME")
    _REGISTRY[name] = app_class
    return app_class


def load() -> dict[str, type[App]]:
    """Import every module in this package, so each one registers itself."""
    if _REGISTRY:
        return dict(_REGISTRY)
    package = Path(__file__).parent
    for info in pkgutil.iter_modules([str(package)]):
        # base is the toolkit and ring is the speaker -- neither is a view.
        if info.name in ("base", "ring", "__init__"):
            continue
        try:
            importlib.import_module(f"apps.{info.name}")
        except Exception as exc:                # noqa: BLE001
            # One broken app must never cost Friday her whole screen.
            log.error("could not load app %r: %s", info.name, exc)
    log.info("apps available: %s", ", ".join(sorted(_REGISTRY)) or "none")
    return dict(_REGISTRY)


def names() -> list[str]:
    return sorted(load())


class Manager:
    """Holds at most one running app and gets out of the way otherwise."""

    def __init__(self, size: tuple[int, int]) -> None:
        self.size = size
        self.current: App | None = None
        self._cmd_stamp = _stamp(app_control.COMMAND_FILE)   # primed, not read
        self._cmd_next = 0.0
        self._sched_next = 0.0
        self._status = None
        load()
        self._write_status()

    # --- lifecycle ---------------------------------------------------

    def launch(self, name: str, **kwargs) -> str:
        name = (name or "").strip().lower()
        registry = load()
        if name not in registry:
            return (f"there is no {name!r} app. Friday has: "
                    f"{', '.join(sorted(registry))}")
        try:
            app = registry[name](self.size, **kwargs)
        except TypeError as exc:
            return f"cannot start {name}: {exc}"
        except Exception as exc:                # noqa: BLE001
            return f"{name} failed to start: {exc}"

        if self.current is not None:
            self.close()
        self.current = app
        log.info("app %s is up", name)
        self._write_status()
        return app.launched_message()

    def close(self) -> None:
        if self.current is None:
            return
        try:
            self.current.close()
        except Exception as exc:                # noqa: BLE001
            log.warning("%s did not close cleanly: %s", self.current.NAME, exc)
        log.info("app %s closed", self.current.NAME)
        self.current = None
        self._write_status()

    # --- the frame ---------------------------------------------------

    def update(self, dt: float) -> None:
        if self.current is None:
            return
        try:
            self.current.update(dt)
            if self.current.done:
                self.close()
        except Exception as exc:                # noqa: BLE001
            log.error("%s crashed while updating: %s", self.current.NAME, exc)
            self.close()

    def draw(self, surface) -> bool:
        """Draw the running app. False means nothing was drawn -- show the face."""
        if self.current is None:
            return False
        try:
            self.current.draw(surface)
            return True
        except Exception as exc:                # noqa: BLE001
            log.error("%s crashed while drawing: %s", self.current.NAME, exc)
            self.close()
            return False

    def handle_event(self, event) -> bool:
        if self.current is None:
            return False
        try:
            return bool(self.current.handle_event(event))
        except Exception as exc:                # noqa: BLE001
            log.error("%s crashed handling an event: %s", self.current.NAME, exc)
            self.close()
            return False

    def dismiss(self) -> str:
        """A spoken or touched 'stop'. What that means is the app's business."""
        if self.current is None:
            return "nothing is on the screen"
        name = self.current.NAME
        try:
            if self.current.dismiss():
                self.close()
                return f"{name} dismissed"
            return f"{name} is still going"
        except Exception:                       # noqa: BLE001
            self.close()
            return f"{name} dismissed"

    @property
    def running(self) -> str | None:
        return self.current.NAME if self.current else None

    @property
    def urgent(self) -> bool:
        """True while what is on screen outranks even a safety face. Only an
        alarm actually going off does -- see App.URGENT."""
        return bool(self.current is not None and self.current.URGENT)

    # --- the two files this watches ----------------------------------

    def poll(self, now: float) -> None:
        """One call per frame. Cheap: two os.stat calls at APP_POLL_S, and a
        schedule read once a second."""
        self._poll_command(now)
        self._poll_schedule(now)

    def _poll_command(self, now: float) -> None:
        if now < self._cmd_next:
            return
        self._cmd_next = now + config.APP_POLL_S
        stamp = _stamp(app_control.COMMAND_FILE)
        if stamp is None or stamp == self._cmd_stamp:
            return
        self._cmd_stamp = stamp
        message = app_control._read_json(app_control.COMMAND_FILE, {})
        action = message.get("action")
        if action == "launch":
            self.launch(message.get("app", ""),
                        **(message.get("kwargs") or {}))
        elif action == "dismiss":
            self.dismiss()

    def _poll_schedule(self, now: float) -> None:
        """An alarm or a timer coming due is what turns a stored promise into
        something on the screen. Checked once a second -- a second late on a
        7am alarm is not late."""
        if now < self._sched_next:
            return
        self._sched_next = now + 1.0
        if self.urgent:
            return                      # already ringing; do not stack rings
        try:
            ready = app_control.due()
        except Exception as exc:                # noqa: BLE001
            log.error("could not read the schedule: %s", exc)
            return
        if not ready:
            return
        item = ready[0]
        # Marked BEFORE the ring starts. If the ring or the app throws, the
        # alarm is spent rather than re-firing every second forever.
        app_control.mark_fired(item["id"])
        self.launch("alarm", label=item.get("label", ""), at=item.get("at", 0),
                    kind=item.get("kind", "alarm"), item_id=item["id"])

    # --- what the rest of the robot can see --------------------------

    def _write_status(self) -> None:
        """So a tool can answer "what is on the screen" without guessing."""
        state = {"app": self.running, "urgent": self.urgent,
                 "since": time.time() if self.running else 0.0}
        if state["app"] == (self._status or {}).get("app"):
            return
        self._status = state
        try:
            app_control._write_atomic(app_control.STATUS_FILE, state,
                                      ".appstat.")
        except OSError as exc:
            log.warning("could not write the app status: %s", exc)


def _stamp(path) -> tuple[int, int] | None:
    """A file's identity for change detection -- the same os.stat trick the
    face uses on emotion.txt, and just as nearly free."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)
