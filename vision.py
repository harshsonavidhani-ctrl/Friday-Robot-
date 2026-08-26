"""vision.py — the two cameras, and the stitched dual view.

The up cam sees the horizon; the down cam sees the floor directly ahead.
look() stacks them into ONE image (up on top, down underneath) so a single
frame tells the brain both what is in the room and what is underfoot.

WHY ffmpeg AND NOT OpenCV. Both cameras output MJPEG natively and ffmpeg is
already on this Pi. Pulling in OpenCV to decode a JPEG the camera already
encoded would cost a large ARM wheel and a lot of CPU for nothing.

WHY THE FIRST FRAMES ARE THROWN AWAY. A USB webcam's first frame is captured
before auto-exposure and auto-white-balance have settled, so it comes back
near-black or blown out. config.CAM_WARMUP_FRAMES are grabbed and discarded;
the last one is kept. This was not a guess — a single-frame capture on this
C270 came back unusable.

WHY IDENTITY IS RESOLVED EVERY CALL. The second camera drops off this Pi's USB
bus and comes back with a different USB descriptor, which renames its
/dev/v4l/by-id path. Anything that caches that path breaks the next time it
reconnects. See config.CAM_SECOND.

A missing camera is normal, not an error. The stitch falls back to whichever
camera answered and says which one was missing.

    img, note = look()                 # PIL image + what it actually got
    path, note = capture_to_file()
"""

from __future__ import annotations

import contextlib
import glob
import logging
import os
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config

log = logging.getLogger("friday.vision")

BY_ID_DIR = "/dev/v4l/by-id"
PROC = "/proc"

# Label strip drawn across each half, so the brain can tell which view is
# which even though they arrive as one image.
LABEL_HEIGHT = 34
LABEL_BG = (0, 0, 0)
LABEL_FG = (255, 106, 0)        # Friday's orange, same as the face
LABEL_MISSING_FG = (235, 70, 70)


def _font(size: int = 20) -> "ImageFont.ImageFont":
    """A real font if the Pi has one, else PIL's bitmap default.

    Worth the lookup: the default font renders these labels so tightly that
    "DOWN CAM" reads as "DOWNCAM", and the brain reads this image.
    """
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


class CameraError(RuntimeError):
    pass


def _holder_of(device: str) -> str | None:
    """Which process currently has this camera open, if any.

    A USB webcam is single-access: while the dashboard is streaming it, the
    MCP server cannot capture from it, and v4l2 reports only a bare "Error
    opening input". Naming the holder turns a baffling error into an
    actionable one — usually "turn the dashboard stream off".

    Scans /proc directly rather than using fuser or lsof: neither is
    guaranteed installed, and pgrep-style matching on this Pi has a habit of
    matching the calling shell instead.
    """
    try:
        target = os.path.realpath(device)
    except OSError:
        return None
    for entry in glob.glob(os.path.join(PROC, "[0-9]*")):
        pid = os.path.basename(entry)
        try:
            for handle in glob.glob(os.path.join(entry, "fd", "*")):
                if os.path.realpath(handle) != target:
                    continue
                with open(os.path.join(entry, "cmdline"), "rb") as cmdline:
                    command = cmdline.read().replace(b"\0", b" ").decode(
                        errors="replace").strip()
                return f"pid {pid} ({command[:70]})"
        except OSError:
            continue                    # process exited mid-scan; normal
    return None


def _capture_nodes() -> list[str]:
    """Every by-id capture node present right now, newest listing each call.

    Only `-video-index0` is a capture node; index1 on these cameras is a
    metadata node that returns no frames.
    """
    return sorted(glob.glob(os.path.join(BY_ID_DIR, "*-video-index0")))


def _serial_of(node: str) -> str:
    """The serial out of a by-id capture node's name, or "".

    udev names these `usb-<descriptor strings>_<serial>-video-index0`. The
    descriptor strings are the unstable part; the serial is not.
    """
    name = os.path.basename(node)
    if name.startswith("usb-"):
        name = name[4:]
    name = name.split("-video-index")[0]
    return name.rsplit("_", 1)[-1] if "_" in name else name


def resolve_cameras() -> dict[str, str | None]:
    """Which /dev path is the up cam and which is the down cam, right now.

    Matched on SERIAL, re-read every call. This used to pin the up cam to
    config.CAM_UP and hand the down cam "whichever other capture node
    exists", which has a silent failure mode: when the C270 itself
    re-enumerates under different descriptor strings its by-id path stops
    matching CAM_UP, so it lands in "whichever other node" and becomes the
    DOWN camera. Nothing raises. look("down") simply returns the ceiling.
    That is worse than being blind, and with only one camera on the bus --
    which is the normal state while the down cam is off it -- it is one
    rename away at all times.

    A node whose serial matches neither camera is left unassigned. Guessing
    is what caused the bug.
    """
    nodes = _capture_nodes()
    by_serial = {_serial_of(n): n for n in nodes}
    up = by_serial.get(config.CAM_UP_SERIAL)
    down = by_serial.get(config.CAM_DOWN_SERIAL)

    # Fall back to the configured paths only for a camera the serials did not
    # place, and never onto a node already claimed by the other role.
    if up is None and config.CAM_UP in nodes and config.CAM_UP != down:
        up = config.CAM_UP
    if down is None and config.CAM_SECOND in nodes and config.CAM_SECOND != up:
        down = config.CAM_SECOND

    if up is not None and up == down:       # cannot happen; costs nothing
        down = None
    if config.CAM_SWAP_UP_DOWN:
        up, down = down, up
    return {"up": up, "down": down}


def _which_camera(device: str) -> str | None:
    """Is this device the up camera or the down one?"""
    for name, resolved in resolve_cameras().items():
        if resolved and resolved == device:
            return name
    return None


def _dashboard_holds(device: str) -> bool:
    """Is the process holding this camera the dashboard's stream?"""
    holder = _holder_of(device) or ""
    return "ffmpeg" in holder and "v4l2" in holder


@contextlib.contextmanager
def _borrowed_from_dashboard(device: str):
    """Turn the dashboard's stream off for this camera, then back on.

    WHY THIS EXISTS. A USB webcam allows exactly ONE reader. While the
    dashboard streams a camera, look() cannot open it — which showed up as
    Friday being blind to the very camera Harsh could see live on his phone.
    Telling him to go and flick a switch first is not a fix; the robot should
    just work.

    So the capture borrows the device through the dashboard's own authoritative
    on/off endpoint, which releases the V4L2 handle immediately, and restores
    the previous state afterwards. The stream blinks for well under a second.

    Yields True if the camera was actually released and is ours to read.

    Best-effort by design: if the dashboard is not running, or does not answer,
    this yields False and the caller reports the original busy error. And the
    restore runs in a finally, so a failed capture can never leave Harsh's
    camera switched off behind his back.
    """
    which = _which_camera(device)
    if which is None:
        yield False
        return

    url = f"{config.DASHBOARD_LOCAL_URL}/api/camera/{which}"
    try:
        _post_json(url, {"on": False})
    except Exception as exc:                    # noqa: BLE001
        log.debug("could not borrow the %s camera: %s", which, exc)
        yield False
        return

    # The handle is closed on return, but v4l2 needs a breath before another
    # process can claim it.
    time.sleep(config.CAM_BORROW_SETTLE_S)
    try:
        yield True
    finally:
        try:
            _post_json(url, {"on": True})
        except Exception as exc:                # noqa: BLE001
            log.warning("could not hand the %s camera back to the dashboard: "
                        "%s — switch it on again in the dashboard", which, exc)


def _post_json(url: str, payload: dict, timeout: float = 5.0) -> dict:
    """Small JSON POST, so vision.py needs no HTTP dependency."""
    import json as _json
    import urllib.request

    request = urllib.request.Request(
        url, data=_json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return _json.loads(response.read().decode())


def capture(device: str, size: tuple[int, int],
            warmup: int | None = None) -> Image.Image:
    """One settled frame from one camera, as a PIL image."""
    warmup = config.CAM_WARMUP_FRAMES if warmup is None else warmup
    config.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = f"{time.monotonic_ns()}"
    pattern = str(config.CAPTURE_DIR / f".grab_{stamp}_%03d.jpg")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-f", "v4l2", "-input_format", "mjpeg",
        "-video_size", f"{size[0]}x{size[1]}",
        "-i", device, "-frames:v", str(warmup), "-y", pattern,
    ]
    # Retry briefly: the dashboard's stream and this capture want the same
    # single-access device, and a stream that is starting or stopping frees it
    # within a moment. Only a PERSISTENT conflict is worth reporting.
    last: Exception | None = None
    for attempt in range(config.CAM_OPEN_ATTEMPTS):
        try:
            subprocess.run(cmd, check=True, capture_output=True,
                           timeout=config.CAM_TIMEOUT_S)
            last = None
            break
        except subprocess.TimeoutExpired as exc:
            raise CameraError(f"{device} timed out after "
                              f"{config.CAM_TIMEOUT_S}s") from exc
        except subprocess.CalledProcessError as exc:
            last = exc
            if attempt + 1 < config.CAM_OPEN_ATTEMPTS:
                time.sleep(config.CAM_OPEN_RETRY_S)
    if last is not None and _dashboard_holds(device):
        # The dashboard is streaming this camera. Borrow it: ask the dashboard
        # to release, grab the frame, then hand it straight back.
        with _borrowed_from_dashboard(device) as borrowed:
            if borrowed:
                try:
                    subprocess.run(cmd, check=True, capture_output=True,
                                   timeout=config.CAM_TIMEOUT_S)
                    last = None
                except (subprocess.CalledProcessError,
                        subprocess.TimeoutExpired) as exc:
                    last = exc

    if last is not None:
        detail = (getattr(last, "stderr", b"") or b"").decode(
            errors="replace").strip()
        message = detail.splitlines()[-1] if detail else str(last)
        holder = _holder_of(device)
        if holder:
            message += (f" — the camera is already open by {holder}. A USB "
                        f"webcam allows one reader at a time; stop that "
                        f"stream (the dashboard's camera switch) and retry.")
        raise CameraError(f"{device}: {message}") from last

    frames = sorted(config.CAPTURE_DIR.glob(f".grab_{stamp}_*.jpg"))
    if not frames:
        raise CameraError(f"{device} produced no frames")
    try:
        # The LAST frame is the settled one; everything before it is warmup.
        image = Image.open(frames[-1]).convert("RGB")
        image.load()
        return image
    finally:
        for path in frames:
            try:
                path.unlink()
            except OSError:
                pass


def _label(image: Image.Image, text: str) -> Image.Image:
    """Caption a frame so the two halves are tellable apart in one image."""
    out = Image.new("RGB", (image.width, image.height + LABEL_HEIGHT), LABEL_BG)
    out.paste(image, (0, LABEL_HEIGHT))
    draw = ImageDraw.Draw(out)
    draw.text((10, 7), text, fill=LABEL_FG, font=_font(20))
    return out


def _placeholder(width: int, text: str) -> Image.Image:
    """Stand-in for a camera that did not answer.

    Drawn rather than skipped so the stitch keeps its shape and the brain can
    SEE that a view is missing instead of quietly getting a one-camera image
    it believes is two.
    """
    image = Image.new("RGB", (width, 120), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.text((10, 48), text, fill=LABEL_MISSING_FG, font=_font(22))
    return image


def look(camera: str = "both") -> tuple[Image.Image, str]:
    """One image showing what Friday can see, plus a note on what it got.

    camera: "up" | "down" | "both" (default). "both" stacks up on top.
    """
    camera = (camera or "both").strip().lower()
    if camera not in ("up", "down", "both"):
        raise ValueError(f"camera must be up, down or both — got {camera!r}")

    devices = resolve_cameras()
    wanted = ("up", "down") if camera == "both" else (camera,)
    sizes = {"up": config.CAM_C270_MAX, "down": config.CAM_SECOND_MAX}

    panels: list[Image.Image] = []
    notes: list[str] = []
    for which in wanted:
        device = devices[which]
        if device is None:
            notes.append(f"{which} cam not on the USB bus")
            panels.append(None)             # sized once the width is known
            continue
        try:
            frame = capture(device, sizes[which])
            panels.append(_label(frame, f"{which.upper()} CAM"))
            notes.append(f"{which} cam ok")
        except CameraError as exc:
            notes.append(f"{which} cam failed: {exc}")
            panels.append(None)

    real = [p for p in panels if p is not None]
    if not real:
        raise CameraError("; ".join(notes) or "no cameras available")

    # Match widths before stacking — the two cameras have different native
    # resolutions, so an unscaled stack would come out ragged.
    width = max(p.width for p in real)
    stacked: list[Image.Image] = []
    for which, panel in zip(wanted, panels):
        if panel is None:
            stacked.append(_placeholder(width, f"{which.upper()} CAM — no signal"))
            continue
        if panel.width != width:
            height = round(panel.height * width / panel.width)
            panel = panel.resize((width, height), Image.LANCZOS)
        stacked.append(panel)

    if len(stacked) == 1:
        return stacked[0], "; ".join(notes)

    total = sum(p.height for p in stacked)
    canvas = Image.new("RGB", (width, total), LABEL_BG)
    y = 0
    for panel in stacked:
        canvas.paste(panel, (0, y))
        y += panel.height
    return canvas, "; ".join(notes)


def _prune_captures() -> None:
    """Keep only the newest config.CAPTURE_KEEP images.

    The SD card is the thing that dies on this robot; an unbounded capture
    directory is a slow way to kill it.
    """
    files = sorted(config.CAPTURE_DIR.glob("look_*.jpg"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in files[config.CAPTURE_KEEP:]:
        try:
            stale.unlink()
        except OSError:
            pass


def capture_to_file(camera: str = "both") -> tuple[Path, str]:
    """look(), saved to disk. Returns the path and the same note."""
    image, note = look(camera)
    config.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CAPTURE_DIR / f"look_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
    image.save(path, "JPEG", quality=config.CAPTURE_QUALITY)
    _prune_captures()
    return path, note


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Capture Friday's dual view.")
    ap.add_argument("--camera", default="both", choices=["up", "down", "both"])
    args = ap.parse_args()

    print("resolved:", resolve_cameras())
    started = time.monotonic()
    try:
        path, note = capture_to_file(args.camera)
    except (CameraError, ValueError) as exc:
        print(f"error: {exc}")
        return 1
    print(f"{path}  ({time.monotonic() - started:.1f}s)")
    print(note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
