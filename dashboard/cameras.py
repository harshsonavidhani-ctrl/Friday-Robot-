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


def _serial_of(node: str) -> str:
    """The serial out of a by-id capture node's name, or "".

    Mirrors vision._serial_of; kept here so the dashboard does not import the
    MCP server's vision module just for one string split.
    """
    name = os.path.basename(node)
    if name.startswith("usb-"):
        name = name[4:]
    name = name.split("-video-index")[0]
    return name.rsplit("_", 1)[-1] if "_" in name else name

JPEG_SOI = b"\xff\xd8"          # start of image
JPEG_EOI = b"\xff\xd9"          # end of image
BOUNDARY = "fridayframe"
MAX_FRAME_BYTES = 2_000_000      # no 640x480 JPEG is this big


class CameraStream:
    """One camera, one ffmpeg process, many subscribers."""

    def __init__(self, device: str, name: str,
                 size=None, fps: int | None = None,
                 exclude: str | None = None,
                 serial: str | None = None) -> None:
        self.configured = device
        self.device = device
        self.name = name
        # The other camera's path, so auto-resolution never picks it twice.
        self.exclude = exclude
        # The stable half of the by-id name; see resolve().
        self.serial = serial
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
        self._linger: asyncio.Task | None = None

    # --- which /dev node is this camera, right now -------------------

    def resolve(self) -> str | None:
        """Find this camera's device node, re-checking every time.

        MATCHED ON SERIAL. The configured by-id path is not stable -- the
        second camera re-enumerated with different USB descriptor strings
        ("HD camera" instead of "Generic HD camera"), which renames its
        symlink and makes a hardcoded path point at nothing. udev builds
        those names from the descriptor strings; only the trailing serial
        survives, so that is what is matched.

        ⚠ The old fallback was "any other by-id node that is not the other
        camera", and it could hand this stream the WRONG camera: when the
        other camera re-enumerates, `exclude` stops matching it, so it stops
        being excluded and gets adopted here instead. The dashboard would
        show one camera in both panes, or the down pane showing the ceiling,
        with nothing logged. An unrecognised node is now left alone.
        """
        if self.serial:
            for path in sorted(glob.glob("/dev/v4l/by-id/*-video-index0")):
                if _serial_of(path) == self.serial:
                    self.device = path
                    return path

        if self.configured and Path(self.configured).exists():
            self.device = self.configured
            return self.device

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
                self.error = None                 # a working stream has none
                self._task = asyncio.create_task(self._pump())
                return
            err = (await proc.stderr.read()).decode(errors="replace").strip()
            self.error = err[:400] or f"ffmpeg exited {proc.returncode}"
        raise RuntimeError(f"{self.name}: {self.error}")

    async def _pump(self) -> None:
        """Split ffmpeg's byte stream back into whole JPEG frames.

        ⚠ BIND THE PIPE ONCE. This used to read self._proc.stdout on every
        iteration, and stop() sets self._proc = None -- so an ordinary
        shutdown raced the loop and raised "'NoneType' object has no attribute
        'stdout'", which the handler below then stored as this camera's error.
        A healthy camera sat in /api/state wearing a permanent failure message
        it had earned by being switched off normally.
        """
        proc = self._proc
        assert proc and proc.stdout
        stdout = proc.stdout
        buf = bytearray()
        try:
            while True:
                chunk = await stdout.read(65536)
                if not chunk:
                    break
                buf.extend(chunk)
                # A frame is everything from SOI to EOI inclusive.
                while True:
                    start = buf.find(JPEG_SOI)
                    if start < 0:
                        # ⚠ NOT A PICTURE AT ALL (2026-10-30: bytes with no SOI,
                        # after the C270 reset on the USB bus). Keeping them grew
                        # the buffer without limit and rescanned all of it on
                        # every read: 92% of a core, the event loop starved and
                        # no frame ever came out. Keep one byte: half a marker.
                        del buf[:-1]
                        break
                    end = buf.find(JPEG_EOI, start + 2)
                    if end < 0:
                        if len(buf) - start > MAX_FRAME_BYTES:
                            del buf[:]      # a start with no end: a broken frame
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
            # Only blame the camera if this pump is still the live one. If
            # stop() has already moved on, whatever went wrong here is the
            # teardown, not the hardware, and must not be reported as it.
            if self._proc is proc:
                self.error = str(exc)
        finally:
            # BOUNDED. This read used to be open-ended, and if ffmpeg was still
            # alive it never returned -- deadlocking anyone awaiting this task.
            # stop() no longer creates that situation, but the read is capped
            # anyway: draining a dead process's stderr is a nicety, and it must
            # never be able to wedge the camera again.
            if self._proc and self._proc.stderr and self.error is None:
                with contextlib.suppress(asyncio.TimeoutError, TimeoutError,
                                         asyncio.CancelledError, Exception):
                    err = await asyncio.wait_for(self._proc.stderr.read(),
                                                 timeout=1.0)
                    if err:
                        self.error = err.decode(
                            errors="replace").strip()[:400]

    async def _ensure_running(self) -> None:
        async with self._lock:
            if self._proc is not None and self._proc.returncode is None:
                return
            self.error = None
            await self._spawn()

    async def stop(self) -> None:
        """Release ffmpeg and the V4L2 handle.

        ⚠ ORDER MATTERS: ffmpeg is ended BEFORE the pump task is cancelled.
        Doing it the other way round deadlocks, and did: _pump's finally awaits
        a read on ffmpeg's stderr, and that read cannot reach EOF while ffmpeg
        is still running. stop() then awaited the cancelled task forever while
        holding the lock, so the camera could never be stopped OR started
        again, the HTTP request never returned, and ffmpeg kept the camera --
        which is why look() could not open a camera the dashboard was showing.

        The process is KILLED if it will not terminate. Giving up and setting
        _proc = None while ffmpeg still lives is what orphaned it holding the
        device.
        """
        async with self._lock:
            proc, self._proc = self._proc, None
            if proc is not None and proc.returncode is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=3)
                except (asyncio.TimeoutError, TimeoutError):
                    proc.kill()
                    with contextlib.suppress(asyncio.TimeoutError,
                                             TimeoutError):
                        await asyncio.wait_for(proc.wait(), timeout=2)

            if self._task:
                self._task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._task
                self._task = None

    # --- consuming ---------------------------------------------------

    @contextlib.asynccontextmanager
    async def subscribe(self):
        # Someone is back before the linger expired: keep the stream we have
        # rather than letting it be torn down under them.
        if self._linger and not self._linger.done():
            self._linger.cancel()
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
            # Last viewer left. DO NOT release the device immediately: a page
            # refresh is a viewer leaving and another arriving a moment later,
            # and each of those used to be a full open/close on the camera.
            # The down camera dies during exactly that -- it left the USB bus
            # the instant its stream was released. Linger instead, so a
            # refresh finds the stream still running. See config.
            if not self._subscribers:
                self._schedule_release()

    def _schedule_release(self) -> None:
        """Stop ffmpeg after the linger, unless someone comes back first."""
        if not config.DASHBOARD_CAM_LINGER_S:
            self._linger = asyncio.create_task(self.stop())
            return
        if self._linger and not self._linger.done():
            return                          # already counting down
        self._linger = asyncio.create_task(self._release_later())

    async def _release_later(self) -> None:
        try:
            await asyncio.sleep(config.DASHBOARD_CAM_LINGER_S)
        except asyncio.CancelledError:
            return
        if not self._subscribers:           # nobody came back
            await self.stop()


class CameraSet:
    """The up and down cameras, resolved from config."""

    def __init__(self) -> None:
        up_device = config.CAM_C270
        down_device = config.CAM_SECOND
        if config.CAM_SWAP_UP_DOWN:
            up_device, down_device = down_device, up_device
        up_serial, down_serial = config.CAM_UP_SERIAL, config.CAM_DOWN_SERIAL
        if config.CAM_SWAP_UP_DOWN:
            up_serial, down_serial = down_serial, up_serial
        self.up = CameraStream(up_device, "up", exclude=down_device,
                               serial=up_serial)
        # The down camera offers only 25 and 30 fps; asking it for the shared
        # default of 10 is asking for an interval it does not have. See config.
        self.down = CameraStream(down_device, "down", exclude=up_device,
                                 serial=down_serial,
                                 fps=config.CAM_SECOND_FPS)

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
