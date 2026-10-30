"""Friday's running picture of the room.

Harsh, 2026-09-13: "the ai should also object detection and also tell what it
sees ... it sees and describe all the enviroment around it continously updates
itself".

Every SCENE_PERIOD_S this takes one frame from the UP camera, names what is in it
with the SSD-MobileNet detector the exploration stack already ships
(rl/friday_vision.py), and writes both down:

    state/scene.json          when, what was named, and where in the frame
    state/scene_latest.jpg    the frame itself, shrunk

Two readers, in other processes, and both only read files:
  * situation.py puts "your camera saw a person ahead, 6 s ago" into every turn,
    so the brain always knows roughly what is around her without asking.
  * brain.py answers "what do you see" from scene_latest.jpg while it is fresh,
    instead of spending ~2 s on observe() and a second trip to the model.

IT RUNS INSIDE THE MCP SERVER because that process owns the cameras: vision.py's
capture lock keeps this and observe()/look() from opening one camera at once.

WHAT IT MUST NEVER DO: hold up a tool, take the camera from the dashboard stream
Harsh is watching on his phone, or fight an exploration run for it. Each of those
is a skip, logged once, never an error. And like rl/friday_vision.py, this is
PERCEPTION ONLY -- nothing steers on it.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

import config

log = logging.getLogger("friday.scene")

_thread: threading.Thread | None = None
_stop = threading.Event()
_shared = None
_detector_ready = threading.Event()


def start() -> None:
    """Begin watching. Idempotent, and a no-op when SCENE_WATCH is False."""
    global _thread
    if not getattr(config, "SCENE_WATCH", True):
        log.info("scene watcher off (SCENE_WATCH = False)")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(target=_run, name="friday-scene", daemon=True)
    _thread.start()


def stop() -> None:
    _stop.set()


def shared_detector(wait_s: float = 0.0):
    """The loaded detector, for explore_eyes: one session, not two 29 MB ones.
    Waits for it because creating one takes ~20 s and scene.py does it first."""
    until = time.monotonic() + wait_s
    while not _detector_ready.wait(0.5):
        if time.monotonic() >= until:
            return None
    return _shared


def _detector():
    """The object detector, or None -- a scene with no names still has a frame."""
    if str(config.ROOT) not in sys.path:
        sys.path.insert(0, str(config.ROOT))
    from rl.friday_vision import ObjectDetector
    detector = ObjectDetector(config.SCENE_MODEL, slot=None,
                              threads=config.SCENE_THREADS)
    return detector if detector._load() else None


def _why_not_now() -> str | None:
    """A reason to skip this tick, or None to go ahead."""
    from mcp_server import hardware
    import vision
    if hardware.lease_active() > 0:
        return "an autonomous run has the robot"
    if config.EYES and config.EYES_ALWAYS:
        # explore_eyes keeps one stream open and writes scene.json itself.
        # Opening the camera here every 8 s is what reset the C270 on the USB bus.
        return "explore_eyes is watching the up camera"
    device = vision.resolve_cameras().get("up")
    if device is None:
        return "the up camera is not on the USB bus"
    if vision._dashboard_holds(device):
        # Borrowing it every few seconds would stutter the stream on his phone.
        return "the dashboard is streaming the up camera"
    return None


def _atomic_write(path: Path, data: bytes) -> None:
    # Readers are other processes polling the file; they must never see half.
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _tick(detector) -> None:
    import io
    import numpy as np
    import vision

    device = vision.resolve_cameras().get("up")
    started = time.monotonic()
    # OPENCV, NOT vision.capture()'s ffmpeg. Through ffmpeg one tick measured
    # 1.5-2.1 s of CPU (process start, probing, decoding frames nobody keeps)
    # -- a fifth of a core every 8 s on a Pi whose whisper is the stage Harsh
    # waits on. cv2 opening the same camera and reading 5 frames measured 0.12 s
    # of CPU; the ~3 s it takes is the camera waking, not work. Same lock as
    # observe()/look(), so the two never open the camera at once.
    import cv2
    from PIL import Image
    width_px, height_px = config.SCENE_CAPTURE_SIZE
    frame = None
    with vision._CAPTURE_LOCK:
        cap = cv2.VideoCapture(str(device), cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width_px)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height_px)
            # The last settled frame; the first ones are before auto-exposure.
            for _ in range(int(config.SCENE_WARMUP_FRAMES)):
                ok, got = cap.read()
                if ok:
                    frame = got
        finally:
            cap.release()
    if frame is None:
        raise vision.CameraError(f"{device} gave no frames")
    image = Image.fromarray(frame[:, :, ::-1])
    capture_s = time.monotonic() - started
    width = int(config.SCENE_IMAGE_WIDTH)
    if image.width > width:
        image = image.resize((width, round(image.height * width / image.width)))

    objects: list[dict] = []
    detect_ms = None
    if detector is not None:
        found = detector._infer(np.asarray(image)[:, :, ::-1].copy())
        detect_ms = round(detector.last_ms)
        # Stricter than the detector's own 0.55: the first live tick named an
        # "aeroplane" at exactly 0.55 filling half the frame.
        found = [d for d in found if d.score >= config.SCENE_MIN_SCORE]
        # Biggest first: with no depth, the thing filling most of the frame is
        # the best available guess at the thing nearest her.
        for d in sorted(found, key=lambda d: -d.area):
            objects.append({"label": d.label, "score": round(d.score, 2),
                            "side": d.side(), "bearing_deg": round(d.bearing_deg),
                            "area": round(d.area, 3)})

    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=int(config.SCENE_IMAGE_QUALITY))
    # The image FIRST, then the JSON that dates it: a reader can then only ever
    # find a frame newer than the timestamp says, never older.
    _atomic_write(Path(config.SCENE_IMAGE), buffer.getvalue())
    _atomic_write(Path(config.SCENE_JSON), json.dumps({
        "taken_at": time.time(),
        "camera": "up",
        "objects": objects,
        "capture_s": round(capture_s, 2),
        "detect_ms": detect_ms,
        "image": str(config.SCENE_IMAGE),
    }).encode())


def _run() -> None:
    global _shared
    detector = None
    try:
        detector = _detector()
    except Exception as exc:                    # noqa: BLE001
        log.warning("scene: object detection OFF (%s: %s)",
                    type(exc).__name__, exc)
    _shared = detector
    _detector_ready.set()
    log.info("scene watcher on: up camera every %.0fs, object detection %s",
             config.SCENE_PERIOD_S, "on" if detector is not None else "OFF")
    paused: str | None = None
    while not _stop.is_set():
        reason = None
        try:
            reason = _why_not_now()
            if reason is None:
                _tick(detector)
        except Exception as exc:                # noqa: BLE001
            reason = f"{type(exc).__name__}: {exc}"
        if reason != paused:
            # Logged on CHANGE only: a dashboard left streaming for an hour is
            # one line, not four hundred.
            if reason:
                log.info("scene watcher paused: %s", reason[:160])
            else:
                log.info("scene watcher running")
            paused = reason
        _stop.wait(float(config.SCENE_PERIOD_S))
