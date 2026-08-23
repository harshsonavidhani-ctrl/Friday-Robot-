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

import glob
import os
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config

BY_ID_DIR = "/dev/v4l/by-id"

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


def _capture_nodes() -> list[str]:
    """Every by-id capture node present right now, newest listing each call.

    Only `-video-index0` is a capture node; index1 on these cameras is a
    metadata node that returns no frames.
    """
    return sorted(glob.glob(os.path.join(BY_ID_DIR, "*-video-index0")))


def resolve_cameras() -> dict[str, str | None]:
    """Which /dev path is the up cam and which is the down cam, right now.

    The up cam is pinned (config.CAM_UP, the C270 — confirmed by looking
    through it). The down cam is "whichever other capture node exists",
    which is what makes it survive the descriptor changing.
    """
    nodes = _capture_nodes()
    up = config.CAM_UP if config.CAM_UP in nodes else None
    others = [n for n in nodes if n != config.CAM_UP]
    down = config.CAM_SECOND if config.CAM_SECOND in nodes else (
        others[0] if others else None)
    if config.CAM_SWAP_UP_DOWN:
        up, down = down, up
    return {"up": up, "down": down}


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
    try:
        subprocess.run(cmd, check=True, capture_output=True,
                       timeout=config.CAM_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise CameraError(f"{device} timed out after "
                          f"{config.CAM_TIMEOUT_S}s") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or b"").decode(errors="replace").strip()
        raise CameraError(f"{device}: {detail.splitlines()[-1] if detail else exc}"
                          ) from exc

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
