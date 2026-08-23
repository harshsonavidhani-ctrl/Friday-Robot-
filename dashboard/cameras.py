"""MJPEG camera streaming for the dashboard.

Both of Friday's USB cameras emit MJPEG natively, so ffmpeg is asked to
COPY the stream rather than re-encode it. That is the whole trick: the Pi
does almost no work, because the JPEGs the browser receives are the ones
the camera's own hardware produced.

One ffmpeg process is shared by every browser watching a given camera --
opening a second handle on a V4L2 device fails, and re-encoding per viewer
would swamp the Pi.
"""

from __future__ import annotations

import asyncio
import contextlib
import glob
import os
from pathlib import Path

import config

JPEG_SOI = b"\xff\xd8"          # start of image
JPEG_EOI = b"\xff\xd9"          # end of image
BOUNDARY = "fridayframe"


class CameraStream:
    """One camera, one ffmpeg process, many subscribers."""

    def __init__(self, device: str, name: str,
                 size=None, fps: int | None = None,
                 exclude: str | None = None) -> None:
        self.configured = device
        self.device = device
        self.name = name
        # The other camera's path, so auto-resolution never picks it twice.
        self.exclude = exclude
        self.enabled = True          # manual on/off from the dashboard
        self.size = size or config.DASHBOARD_CAM_SIZE
        self.fps = fps or config.DASHBOARD_CAM_FPS
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._latest: bytes | None = None
        self._lock = asyncio.Lock()
        self.error: str | None = None
        self.passthrough = True

    # --- which /dev node is this camera, right now -------------------

    def resolve(self) -> str | None:
        """Find this camera's device node, re-checking every time.

        The configured by-id path is preferred, but it is not stable: the
        second camera re-enumerated with different USB descriptor strings
        ("HD camera" instead of "Generic HD camera"), which renames its
        by-id symlink and makes a hardcoded path point at nothing. So when
        the configured path is missing, fall back to any other by-id capture
        node that is not the other camera. That also means a camera that is
        unplugged and plugged back in is found again without editing config.
        """
        if self.configured and Path(self.configured).exists():
            self.device = self.configured
            return self.device

        other = os.path.realpath(self.exclude) if self.exclude else None
        for path in sorted(glob.glob("/dev/v4l/by-id/*-video-index0")):
            if os.path.realpath(path) == other:
                continue
            if self.configured and os.path.realpath(path) == \
                    os.path.realpath(self.configured):
                continue            # already tried above and it is gone
            self.device = path
            return path

        self.device = self.configured
        return None

    @property
    def available(self) -> bool:
        return self.enabled and self.resolve() is not None

    # --- lifecycle ---------------------------------------------------

    def _ffmpeg_cmd(self, passthrough: bool) -> list[str]:
        """Passthrough is the cheap path -- the camera's own JPEGs are
        forwarded untouched. Not every camera offers MJPEG at the size we
        ask for though, so the fallback takes whatever the camera gives and
        encodes it here. That costs CPU, which is why it is second."""
        w, h = self.size
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-f", "v4l2"]
        if passthrough:
            cmd += ["-input_format", "mjpeg"]
        cmd += ["-video_size", f"{w}x{h}", "-framerate", str(self.fps),
                "-i", self.device]
        if passthrough:
            cmd += ["-c:v", "copy"]
        else:
            cmd += ["-c:v", "mjpeg", "-q:v", str(config.DASHBOARD_CAM_QUALITY)]
        return cmd + ["-f", "mjpeg", "pipe:1"]

    async def _spawn(self) -> None:
        device = self.resolve()
        if device is None:
            raise FileNotFoundError(f"{self.name}: no capture device found")

        # Try passthrough; if ffmpeg dies straight away, the camera cannot
        # give us MJPEG at this size, so re-encode instead of showing nothing.
        for passthrough in (True, False):
            proc = await asyncio.create_subprocess_exec(
                *self._ffmpeg_cmd(passthrough),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.5)
            except asyncio.TimeoutError:
                self._proc = proc                 # still alive: it is working
                self.passthrough = passthrough
                self._task = asyncio.create_task(self._pump())
                return
            err = (await proc.stderr.read()).decode(errors="replace").strip()
            self.error = err[:400] or f"ffmpeg exited {proc.returncode}"
        raise RuntimeError(f"{self.name}: {self.error}")

    async def _pump(self) -> None:
        """Split ffmpeg's byte stream back into whole JPEG frames."""
        assert self._proc and self._proc.stdout
        buf = bytearray()
        try:
            while True:
                chunk = await self._proc.stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                # A frame is everything from SOI to EOI inclusive.
                while True:
                    start = buf.find(JPEG_SOI)
                    if start < 0:
                        break
                    end = buf.find(JPEG_EOI, start + 2)
                    if end < 0:
                        break
                    frame = bytes(buf[start:end + 2])
                    del buf[:end + 2]
                    self._latest = frame
                    for queue in list(self._subscribers):
                        if queue.full():
                            with contextlib.suppress(asyncio.QueueEmpty):
                                queue.get_nowait()   # drop the stale frame
                        with contextlib.suppress(asyncio.QueueFull):
                            queue.put_nowait(frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                     # noqa: BLE001
            self.error = str(exc)
        finally:
            if self._proc and self._proc.stderr and self.error is None:
                err = await self._proc.stderr.read()
                if err:
                    self.error = err.decode(errors="replace").strip()[:400]

    async def _ensure_running(self) -> None:
        async with self._lock:
            if self._proc is not None and self._proc.returncode is None:
                return
            self.error = None
            await self._spawn()

    async def stop(self) -> None:
        async with self._lock:
            if self._task:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
                self._task = None
            if self._proc and self._proc.returncode is None:
                self._proc.terminate()
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(self._proc.wait(), timeout=3)
            self._proc = None

    # --- consuming ---------------------------------------------------

    @contextlib.asynccontextmanager
    async def subscribe(self):
        if not self.enabled:
            raise PermissionError(f"{self.name} camera is switched off")
        if self.resolve() is None:
            raise FileNotFoundError(f"{self.name}: no capture device found")
        await self._ensure_running()
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        if self._latest:
            queue.put_nowait(self._latest)          # show something instantly
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)
            # Last viewer left: release the camera rather than idling ffmpeg.
            if not self._subscribers:
                await self.stop()


class CameraSet:
    """The up and down cameras, resolved from config."""

    def __init__(self) -> None:
        up_device = config.CAM_C270
        down_device = config.CAM_SECOND
        if config.CAM_SWAP_UP_DOWN:
            up_device, down_device = down_device, up_device
        self.up = CameraStream(up_device, "up", exclude=down_device)
        self.down = CameraStream(down_device, "down", exclude=up_device)

    def get(self, which: str) -> CameraStream | None:
        return {"up": self.up, "down": self.down}.get(which)

    async def stop_all(self) -> None:
        await self.up.stop()
        await self.down.stop()

    async def set_enabled(self, which: str, on: bool) -> bool:
        """Manual switch from the dashboard. Turning a camera off releases
        ffmpeg and the V4L2 handle immediately."""
        cam = self.get(which)
        if cam is None:
            raise KeyError(which)
        cam.enabled = bool(on)
        if not cam.enabled:
            await cam.stop()
        return cam.enabled

    def _one(self, cam: CameraStream) -> dict:
        present = cam.resolve() is not None
        return {
            "available": cam.enabled and present,
            "present": present,          # plugged in, regardless of the switch
            "enabled": cam.enabled,      # the dashboard's on/off switch
            "device": cam.device,
            "streaming": cam._proc is not None and cam._proc.returncode is None,
            "passthrough": cam.passthrough,
            "error": cam.error,
        }

    def status(self) -> dict:
        return {"up": self._one(self.up), "down": self._one(self.down)}
