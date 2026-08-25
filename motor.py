"""Friday's drive: 4 motors, L298N, differential, driven through an Arduino.

The Pi does not touch the motor pins itself. An Arduino runs
`firmware/friday_motors/friday_motors.ino` and holds a watchdog; the Pi sends
it short commands over USB serial. That split is deliberate -- if this program
dies, or the USB cable is pulled, the Arduino stops the motors on its own
rather than leaving the robot driving.

    from motor import Motors

    with Motors() as m:              # stops the motors on the way out, always
        m.forward(seconds=1.0)
        m.turn_left(degrees=90)
        m.stop()

SAFETY, and none of it is optional:

* Speeds are capped at config.MOTOR_MAX_DUTY. The cap is low on purpose.
* Every timed move is fed keepalives, and always ends in an explicit stop --
  including when it raises.
* The port is never guessed. If the board cannot be positively identified,
  this refuses to send anything. This Uno is an FT232R clone, so it is pinned
  by its unique serial number in config.MOTOR_PORT rather than by VID:PID --
  0403:6001 is a generic chip and matching on it would let any FTDI cable be
  picked as "the motors". The firmware handshake below is the second gate:
  a board that does not answer FRIDAY-MOTORS is never driven.
* Distances and angles come from state/calibration.json, which the dashboard
  edits. Until those are measured they are 0, and any distance-based move
  raises instead of guessing.
"""

from __future__ import annotations

import threading
import time

import serial
from serial.tools import list_ports

import config

BAUD = 115200
IDENT = "FRIDAY-MOTORS"

# USB IDs of boards that could be running the motor firmware. Genuine Uno,
# Uno R3 clones on the same VID, and the two common USB-serial chips found on
# cheap clones. Anything not on this list is not driven without an explicit
# port, because the wrong guess moves the robot.
ARDUINO_IDS = {
    (0x2341, 0x0043),   # Arduino Uno R3
    (0x2341, 0x0001),   # Arduino Uno (older bootloader)
    (0x2A03, 0x0043),   # Arduino.org Uno
    (0x1A86, 0x7523),   # CH340, the usual clone bridge
    (0x1A86, 0x7522),   # CH340 variant
    (0x0403, 0x6015),   # FT231X, some clones
}

# The Arduino's watchdog window. The keepalive interval must be comfortably
# shorter, or a long move stutters as the watchdog trips and is re-armed.
WATCHDOG_MS = int(config.MOTOR_WATCHDOG_S * 1000)
KEEPALIVE_S = config.MOTOR_WATCHDOG_S / 3.0

# The Uno resets when the serial port is opened (DTR toggles). Nothing sent
# during the bootloader window is heard, so wait it out.
RESET_SETTLE_S = 2.0


class MotorError(RuntimeError):
    pass


def find_port() -> str | None:
    """The Arduino's serial port, or None if no known board is present.

    Deliberately strict. Returning None and making the caller pass a port
    explicitly is the safe failure; picking "probably that one" is not.
    """
    for port in list_ports.comports():
        if port.vid is None:
            continue
        if (port.vid, port.pid) in ARDUINO_IDS:
            return port.device
    return None


def describe_serial() -> list[str]:
    """Every serial device present, for when find_port() comes up empty."""
    out = []
    for p in list_ports.comports():
        ident = f"{p.vid:04x}:{p.pid:04x}" if p.vid is not None else "?"
        out.append(f"{p.device}  {ident}  {p.product or p.description}")
    return out


class Motors:
    """The drive. One instance owns the serial link; not thread safe."""

    def __init__(self, port: str | None = None, baud: int = BAUD,
                 max_duty: float | None = None) -> None:
        self.port = port or config.MOTOR_PORT or find_port()
        self.baud = baud
        self.max_duty = config.MOTOR_MAX_DUTY if max_duty is None else max_duty
        self.serial: serial.Serial | None = None
        self.firmware: str | None = None
        self._lock = threading.Lock()
        self._keepalive: threading.Thread | None = None
        self._stop_keepalive = threading.Event()
        # Whether the wheels are already turning, so a move that starts from
        # rest can be given a kickstart and one that is already going is not.
        self._moving = False

    # --- connection --------------------------------------------------

    def connect(self) -> str:
        if self.port is None:
            raise MotorError(
                "no Arduino found on USB. Serial devices present:\n  "
                + "\n  ".join(describe_serial())
                + "\nPlug the Arduino in, or pass port= explicitly once you "
                  "have confirmed which device drives the motors.")

        # write_timeout matters as much as timeout: without it a write to a
        # port that has stopped draining blocks forever, and that port is how
        # the robot is told to stop. See config.MOTOR_WRITE_TIMEOUT_S.
        self.serial = serial.Serial(
            self.port, self.baud, timeout=1.0,
            write_timeout=config.MOTOR_WRITE_TIMEOUT_S)
        time.sleep(RESET_SETTLE_S)
        self.serial.reset_input_buffer()

        reply = self._command("P")
        if not reply.startswith(IDENT):
            self.close()
            raise MotorError(
                f"{self.port} answered {reply!r}, not {IDENT}. That is not "
                f"Friday's motor firmware -- refusing to drive it. Flash "
                f"firmware/friday_motors/friday_motors.ino first.")
        self.firmware = reply
        self._command(f"W{WATCHDOG_MS}")
        return reply

    def close(self) -> None:
        self._stop_keepalive.set()
        if self.serial and self.serial.is_open:
            try:
                self._write("S")            # never leave the motors running
            except Exception:               # noqa: BLE001 - closing anyway
                pass
            self.serial.close()
        self.serial = None

    def __enter__(self) -> "Motors":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # --- the wire ----------------------------------------------------

    def _write(self, line: str) -> None:
        """Put a line on the wire. Never blocks indefinitely.

        ⚠ THERE IS NO flush() HERE ON PURPOSE. serial.flush() calls tcdrain(),
        which waits for the UART to physically empty and honours NO timeout at
        all -- not write_timeout, not anything. When the FTDI stopped draining
        after a USB re-enumeration, that one call hung every `stop` on the
        robot indefinitely (found with py-spy; five threads stacked behind it).
        The write itself is enough: it hands the bytes to the kernel, which
        sends them, and write_timeout bounds even that. For _command the
        readline that follows is the real synchronisation anyway.
        """
        if self.serial is None or not self.serial.is_open:
            raise MotorError("not connected -- call connect() first")
        try:
            self.serial.write((line + "\n").encode())
        except serial.SerialTimeoutException as exc:
            raise MotorError(
                f"the motor link would not accept a write within "
                f"{config.MOTOR_WRITE_TIMEOUT_S}s ({exc}). The Arduino "
                f"watchdog cuts the motors after "
                f"{config.MOTOR_WATCHDOG_S}s of silence.") from exc

    def _command(self, line: str) -> str:
        """Send a line and read the reply, under a lock that can be lost.

        The lock is acquired WITH A TIMEOUT rather than unconditionally: one
        wedged command must not be able to hold the port against an emergency
        stop for the rest of the process's life. Failing fast is strictly
        better here, because the Arduino cuts the motors on its own after
        MOTOR_WATCHDOG_S of silence -- so losing the port fails safe, while
        waiting on it does not.
        """
        if not self._lock.acquire(timeout=config.MOTOR_LOCK_TIMEOUT_S):
            raise MotorError(
                f"the motor link is busy and did not free up within "
                f"{config.MOTOR_LOCK_TIMEOUT_S}s. The Arduino watchdog cuts "
                f"the motors after {config.MOTOR_WATCHDOG_S}s of silence.")
        try:
            self._write(line)
            return self.serial.readline().decode(errors="replace").strip()
        finally:
            self._lock.release()

    # --- raw drive ---------------------------------------------------

    def _clamp(self, speed: float) -> int:
        limit = int(255 * self.max_duty)
        return max(-limit, min(limit, int(speed)))

    def drive(self, left: float, right: float) -> None:
        """Set both sides directly. Positive is forward.

        Both sides are capped at max_duty, so this cannot be used to sneak
        past the speed limit.

        THE INVERSION IS APPLIED HERE, at the last point before the wire, so
        every caller above -- named moves, the MCP tools, the dashboard's
        manual drive -- is corrected by it and none of them has to know. Fixing
        it any higher up would mean fixing it in several places and missing one.
        """
        self._kickstart(left, right)
        self._send(left, right)

    def _send(self, left: float, right: float) -> None:
        """Put one command on the wire, inversion applied."""
        l = self._clamp(left) * (-1 if config.MOTOR_INVERT_LEFT else 1)
        r = self._clamp(right) * (-1 if config.MOTOR_INVERT_RIGHT else 1)
        reply = self._command(f"M{l} {r}")
        if reply != "OK":
            raise MotorError(f"drive({l}, {r}) -> {reply!r}")
        self._moving = bool(l or r)

    def _kickstart(self, left: float, right: float) -> None:
        """Break static friction with a brief hard pulse before settling.

        WHY. Starting from rest, the motors need noticeably more torque to get
        moving than to keep moving. Below that threshold they sit and buzz --
        energised, audibly humming, going nowhere. Pivoting in place is the
        worst case on this robot, because all four wheels have to scrub
        sideways at once.

        So a move that starts from a standstill gets a short pulse at the duty
        cap first, then drops to the requested speed. It is the same trick as
        blipping a stalled fan: past the stiction, far less is needed to keep
        it turning.

        Capped by _clamp like everything else, so this cannot exceed
        MOTOR_MAX_DUTY -- it borrows the headroom that is already allowed, it
        does not invent any.
        """
        if not config.MOTOR_KICKSTART_MS or self._moving:
            return
        if not (left or right):
            return
        ceiling = 255 * self.max_duty
        # Only worth doing when the target is below the pulse level; above it
        # the move already starts hard enough.
        if max(abs(left), abs(right)) >= ceiling * config.MOTOR_KICKSTART_FLOOR:
            return
        scale = ceiling * config.MOTOR_KICKSTART_FLOOR
        peak = max(abs(left), abs(right)) or 1.0
        self._send(left / peak * scale, right / peak * scale)
        time.sleep(config.MOTOR_KICKSTART_MS / 1000.0)

    def stop(self) -> None:
        self._stop_keepalive.set()
        # Cleared before the write, not after: if the command throws, the
        # wheels are in an unknown state and the next move must assume it is
        # starting from rest and kickstart accordingly.
        self._moving = False
        reply = self._command("S")
        if reply != "OK":
            raise MotorError(f"stop -> {reply!r}")

    def keepalive(self) -> None:
        """Pet the Arduino's watchdog without changing the command.

        A bare newline. Deliberately write-only: the firmware sends no reply
        to a keepalive, so reading one back would block for the full serial
        timeout and turn a 100 ms safety poll into a 1 s one.
        """
        if not self._lock.acquire(timeout=config.MOTOR_LOCK_TIMEOUT_S):
            raise MotorError("the motor link is busy; skipping a keepalive")
        try:
            self._write("")
        finally:
            self._lock.release()

    # --- timed moves -------------------------------------------------

    def _run_for(self, left: float, right: float, seconds: float) -> None:
        """Hold a drive command for a while, then stop -- even on Ctrl-C.

        The keepalive matters: the Arduino stops on its own if it hears
        nothing for MOTOR_WATCHDOG_S, so a move longer than that window has
        to keep talking or it stutters.
        """
        if seconds <= 0:
            return
        self.drive(left, right)
        deadline = time.monotonic() + seconds
        try:
            while time.monotonic() < deadline:
                time.sleep(min(KEEPALIVE_S, max(0.0,
                                                deadline - time.monotonic())))
                self.keepalive()
        finally:
            self.stop()

    # --- named moves -------------------------------------------------

    def _speed(self, given: float | None, key: str) -> float:
        if given is not None:
            return given
        return config.load_calibration()[key]

    def forward(self, speed: float | None = None, seconds: float | None = None,
                cm: float | None = None) -> None:
        self._straight(+1, speed, seconds, cm)

    def backward(self, speed: float | None = None,
                 seconds: float | None = None, cm: float | None = None) -> None:
        self._straight(-1, speed, seconds, cm)

    def _straight(self, sign: int, speed, seconds, cm) -> None:
        speed = self._speed(speed, "move_speed") * sign
        if cm is not None:
            rate = config.load_calibration()["forward_cm_per_s"]
            if rate <= 0:
                raise MotorError(
                    "forward_cm_per_s is not calibrated yet (it is 0). "
                    "Drive a measured distance for a known time, then set it "
                    "in the dashboard's Calibration panel. Until then use "
                    "seconds= instead of cm=.")
            seconds = abs(cm) / rate
        if seconds is None:
            self.drive(speed, speed)         # runs until stop() or watchdog
        else:
            self._run_for(speed, speed, seconds)

    def turn_left(self, speed: float | None = None,
                  seconds: float | None = None,
                  degrees: float | None = None) -> None:
        self._pivot(-1, speed, seconds, degrees)

    def turn_right(self, speed: float | None = None,
                   seconds: float | None = None,
                   degrees: float | None = None) -> None:
        self._pivot(+1, speed, seconds, degrees)

    def _pivot(self, sign: int, speed, seconds, degrees) -> None:
        """Spin in place: the two sides run opposite ways."""
        speed = self._speed(speed, "turn_speed")
        if degrees is not None:
            rate = config.load_calibration()["turn_deg_per_s"]
            if rate <= 0:
                raise MotorError(
                    "turn_deg_per_s is not calibrated yet (it is 0). Time a "
                    "full 360 in place, then set it in the dashboard's "
                    "Calibration panel. Until then use seconds= instead of "
                    "degrees=.")
            seconds = abs(degrees) / rate
        left, right = (speed, -speed) if sign > 0 else (-speed, speed)
        if seconds is None:
            self.drive(left, right)
        else:
            self._run_for(left, right, seconds)

    # --- the shape the brain's MCP tool will call --------------------

    def move(self, direction: str, speed: float | None = None,
             seconds: float | None = None, distance_cm: float | None = None,
             degrees: float | None = None) -> str:
        """One entry point: move("forward", seconds=1.5).

        Matches the `move` MCP tool so the brain has a single verb.
        """
        direction = direction.strip().lower()
        if direction in ("forward", "ahead", "front"):
            self.forward(speed, seconds, distance_cm)
        elif direction in ("back", "backward", "backwards", "reverse"):
            self.backward(speed, seconds, distance_cm)
        elif direction == "left":
            self.turn_left(speed, seconds, degrees)
        elif direction == "right":
            self.turn_right(speed, seconds, degrees)
        elif direction == "stop":
            self.stop()
        else:
            raise MotorError(
                f"unknown direction {direction!r} -- use forward, back, "
                f"left, right or stop")
        return f"{direction} ok"


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Friday's motors. Detection and identification are safe; "
                    "anything that turns a wheel needs --i-am-watching.")
    ap.add_argument("--port", help="serial port (default: auto-detect)")
    ap.add_argument("--detect", action="store_true",
                    help="list serial devices and exit -- moves nothing")
    ap.add_argument("--identify", action="store_true",
                    help="open the port and ask the board who it is")
    ap.add_argument("--move", metavar="DIRECTION",
                    help="forward | back | left | right")
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--speed", type=float)
    ap.add_argument("--i-am-watching", action="store_true",
                    help="required for --move: confirms the robot is lifted "
                         "off the ground or has clear space around it")
    args = ap.parse_args()

    if args.detect or not (args.identify or args.move):
        print("serial devices present:")
        for line in describe_serial():
            print("  " + line)
        found = args.port or config.MOTOR_PORT or find_port()
        print(f"\nmotor port: {found or 'NOT FOUND — no known Arduino on USB'}")
        return 0

    if args.move and not args.i_am_watching:
        print("refusing to move without --i-am-watching. Lift the robot onto "
              "a stand or clear the floor first.")
        return 2

    m = Motors(port=args.port)
    try:
        print(m.connect())
        if args.move:
            m.move(args.move, speed=args.speed, seconds=args.seconds)
            print(f"{args.move} for {args.seconds}s, stopped")
    except MotorError as exc:
        print(f"error: {exc}")
        return 1
    finally:
        m.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
