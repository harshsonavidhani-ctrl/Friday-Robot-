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
import os
from pathlib import Path

import config

JPEG_SOI = b"\xff\xd8"          # start of image
JPEG_EOI = b"\xff\xd9"          # end of image
BOUNDARY = "fridayframe"


class CameraStream:
    """One camera, one ffmpeg process, many subscribers."""

    def __init__(self, device: str, name: str,
                 size=None, fps: int | None = None) -> None:
        self.device = device
        self.name = name
        self.size = size or config.DASHBOARD_CAM_SIZE
        self.fps = fps or config.DASHBOARD_CAM_FPS
        self._proc: asyncio.subprocess.Process | None = None
        self._task: asyncio.Task | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._latest: bytes | None = None
        self._lock = asyncio.Lock()
        self.error: str | None = None

    @property
    def available(self) -> bool:
        return bool(self.device) and Path(self.device).exists()

    # --- lifecycle ---------------------------------------------------

    async def _spawn(self) -> None:
        w, h = self.size
        cmd = [
            "ffmpeg", "-nostdin", "-loglevel", "error",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-video_size", f"{w}x{h}",
            "-framerate", str(self.fps),
            "-i", self.device,
            # copy: the camera already produced JPEGs, so no re-encode
            "-c:v", "copy",
            "-f", "mjpeg", "pipe:1",
        ]
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self._task = asyncio.create_task(self._pump())

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
        if not self.available:
            raise FileNotFoundError(f"{self.name}: no device at {self.device}")
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
        self.up = CameraStream(up_device, "up")
        self.down = CameraStream(down_device, "down")

    def get(self, which: str) -> CameraStream | None:
        return {"up": self.up, "down": self.down}.get(which)

    async def stop_all(self) -> None:
        await self.up.stop()
        await self.down.stop()

    def status(self) -> dict:
        return {
            "up": {"available": self.up.available, "device": self.up.device,
                   "error": self.up.error},
            "down": {"available": self.down.available,
                     "device": self.down.device, "error": self.down.error},
        }
