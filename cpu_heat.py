"""Her CPU is hot: she says so, and asks for the fan.

Harsh, 2026-10-03: "build an offline detector like if pi temp reaches above 60
degree so friday will say my cpu is hot pls turn on the fan".

OFFLINE, NO MODEL: the temperature is the kernel's own reading
(config.CPU_TEMP_PATH, the same file get_status uses) and the words are one
fixed line (config.CPU_HOT_LINE). Nothing here imports a network library or a
model, and nothing here makes a sound: `HeatWatch` is a WAKE SOURCE like
reminders.ReminderWatch, so main.py speaks the line from its own loop, between
turns, through tts.speak() (CLAUDE.md PART 4) -- never over Harsh.

WHEN SHE SAYS IT (`Policy`, pure, so the tests need no clock and no Pi):
  * at or above CPU_HOT_SAY_C (60) for CPU_HOT_SUSTAIN_S -- a single hot
    reading (a burst of work for a second) is not "hot";
  * then not again while it STAYS hot until CPU_HOT_REPEAT_S has passed --
    the Pi idles near 60 with the face and the brain up, and she must not nag;
  * it counts as cooled only below CPU_HOT_CLEAR_C (hysteresis): a reading
    wobbling round 60 is one hot spell, not twenty;
  * a line that could not be said (he was talking) is offered again after
    CPU_HOT_RETRY_S, not dropped;
  * VERY HOT (Harsh, 2026-10-03: "if she says it at 60 degrees and I forgot,
    and the temp reaches 70, then again the alarm"): at CPU_HOT_AGAIN_C (70)
    for CPU_HOT_SUSTAIN_S she says CPU_HOT_LINE_AGAIN AT ONCE, whatever the
    10-minute repeat clock says; cooled from it only below
    CPU_HOT_AGAIN_CLEAR_C (67), so a new climb past 70 is said again. A jump
    straight from cool to 72 is one line (the 70 one), not two.

WHEN SHE HOLDS IT: quiet mode, and while she is asleep ("Friday, go to sleep")
-- EXCEPT during an exploration run. A run hushes every wake source with the
same sleep() call that "go to sleep" uses, and a run is when the Pi is hottest
(81 C measured), so during a run the line is said anyway.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path

import config

log = logging.getLogger("friday.cpu_heat")

NAME = "cpu_heat"


def read_c(path: str | Path | None = None) -> float | None:
    """The CPU temperature in degrees C, or None if it cannot be read."""
    try:
        with open(path or config.CPU_TEMP_PATH) as handle:
            return int(handle.read().strip()) / 1000.0
    except (OSError, ValueError):
        return None


class Policy:
    """Feed it readings; `update` says when to speak and at which level
    (0 = nothing, 1 = hot, 2 = very hot). No I/O, no clock."""

    def __init__(self, hot_c: float, clear_c: float, sustain_s: float,
                 repeat_s: float, retry_s: float, again_c: float | None = None,
                 again_clear_c: float | None = None):
        if clear_c >= hot_c:
            raise ValueError("clear_c must be below hot_c (hysteresis)")
        if again_c is not None and not (hot_c < (again_clear_c or again_c) <= again_c):
            raise ValueError("need hot_c < again_clear_c <= again_c")
        self.hot_c, self.clear_c = float(hot_c), float(clear_c)
        self.again_c = None if again_c is None else float(again_c)
        self.again_clear_c = (self.again_c if again_clear_c is None
                              else float(again_clear_c)) if again_c is not None else None
        self.sustain_s, self.repeat_s = float(sustain_s), float(repeat_s)
        self.retry_s = float(retry_s)
        self.hot_since: float | None = None     # this hot spell began
        self.again_since: float | None = None   # ...and has been at "again" since
        self.next_say: float | None = None      # when the line is due again
        self.level_said = 0                     # the highest level said this spell
        self.hold_until = -float("inf")         # a missed line waits for this
        self.pending = 0                        # a line handed out, not yet said
        self.said = 0
        self.peak: float | None = None

    def update(self, temp_c: float | None, now: float) -> int:
        """The level to say now, or 0 (the caller then calls handed, and
        later said_it or missed)."""
        if temp_c is None:
            return 0                             # unreadable: nothing to say
        if self.hot_since is None:
            if temp_c < self.hot_c:
                return 0
            self.hot_since, self.peak = now, temp_c
            self.next_say = now + self.sustain_s
            self.again_since, self.level_said = None, 0
        elif temp_c < self.clear_c:
            log.info("CPU cooled to %.1f C (peak %.1f C this spell)", temp_c,
                     self.peak or temp_c)
            self.hot_since = self.next_say = self.peak = self.again_since = None
            self.level_said = 0
            return 0
        self.peak = max(self.peak or temp_c, temp_c)
        if self.again_c is not None:
            if temp_c >= self.again_c:
                if self.again_since is None:
                    self.again_since = now
            elif temp_c < self.again_clear_c:
                self.again_since = None
                if self.level_said >= 2:
                    self.level_said = 1          # cooled from "very hot": a new climb is new
        if self.pending or now < self.hold_until:
            return 0
        # Harsh, 2026-10-03: "if she says it at 60 and I forgot, at 70 the alarm
        # again" -- at once, whatever the 10-minute repeat clock says.
        if (self.again_since is not None and self.level_said < 2
                and now - self.again_since >= self.sustain_s):
            return 2
        if temp_c >= self.hot_c and self.next_say is not None and now >= self.next_say:
            return 2 if (self.again_since is not None and self.level_said >= 2) else 1
        return 0

    def handed(self, level: int) -> None:
        self.pending = int(level)

    def said_it(self, now: float) -> None:
        self.said += 1
        self.level_said = max(self.level_said, self.pending or 1)
        self.pending = 0
        self.next_say = now + self.repeat_s

    def missed(self, now: float) -> None:
        self.pending = 0
        self.hold_until = now + self.retry_s


def _exploring(root: Path | None = None) -> bool:
    """An exploration run is out (the file main._exploring reads)."""
    try:
        data = json.loads(((root or Path(config.ROOT)) / "state"
                           / "explore_mode.json").read_text())
        return str(data.get("state")) == "running"
    except (OSError, ValueError, AttributeError):
        return False


def _quiet() -> bool:
    try:
        import mode
        return bool(mode.quiet())
    except Exception:                                   # noqa: BLE001
        return False


class HeatWatch:
    """A wake source (duck-typed like voice_link.MissionWatch): one trigger,
    "cpu_heat: <degrees>", each time the line is due."""

    name = NAME
    stream = None

    def __init__(self, read=read_c, clock=time.monotonic, exploring=_exploring,
                 quiet=_quiet, poll_s: float | None = None):
        self.read, self.clock = read, clock
        self.exploring, self.quiet = exploring, quiet
        self.poll_s = float(poll_s if poll_s is not None else config.CPU_HOT_POLL_S)
        self.policy = Policy(config.CPU_HOT_SAY_C, config.CPU_HOT_CLEAR_C,
                             config.CPU_HOT_SUSTAIN_S, config.CPU_HOT_REPEAT_S,
                             config.CPU_HOT_RETRY_S,
                             getattr(config, "CPU_HOT_AGAIN_C", None),
                             getattr(config, "CPU_HOT_AGAIN_CLEAR_C", None))
        self._closed = threading.Event()
        self._paused = False
        self._asleep_until = 0.0
        self.last_c: float | None = None

    def step(self) -> str | None:
        """One poll: the trigger to yield, or None."""
        now = self.clock()
        temp = self.read()
        self.last_c = temp
        level = self.policy.update(temp, now)
        if not level:
            return None
        if self._paused:
            return None                          # she is talking: next poll
        asleep = now < self._asleep_until and not self.exploring()
        if asleep or self.quiet():
            self.policy.missed(now)
            return None
        # Out until main says it was said (delivered) or not (missed): nothing
        # else is handed out meanwhile.
        self.policy.handed(level)
        return f"{temp:.0f} {level}"

    # -- what main.py tells it --------------------------------------------------
    def delivered(self) -> None:
        self.policy.said_it(self.clock())

    def missed(self) -> None:
        self.policy.missed(self.clock())

    # -- WakeSource ---------------------------------------------------------------
    def listen(self):
        log.info("CPU heat watch ready: says it at %.0f C and above (now %s)",
                 config.CPU_HOT_SAY_C,
                 "unreadable" if self.read() is None else f"{self.read():.1f} C")
        while not self._closed.wait(self.poll_s):
            try:
                trig = self.step()
            except Exception as exc:                    # noqa: BLE001
                log.warning("CPU heat poll failed: %s", exc)
                trig = None
            if trig:
                yield trig

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def sleep(self, seconds: float) -> float:
        # Remembered, not obeyed blindly: a run's hush and "go to sleep" are
        # the same call, and during a run she must still say it.
        self._asleep_until = self.clock() + max(float(seconds), 0.0)
        return 0.0

    def wake_up(self) -> None:
        self._asleep_until = 0.0

    def sleeping_for(self) -> float:
        return 0.0

    def close(self) -> None:
        self._closed.set()


def parse_trigger(trigger: str) -> float | None:
    """"cpu_heat: 72 2" -> 72.0 (None if it is not one of ours)."""
    if not trigger.startswith(NAME + ": "):
        return None
    try:
        return float(trigger.split(": ", 1)[1].split()[0])
    except (ValueError, IndexError):
        return None


def trigger_level(trigger: str) -> int:
    """"cpu_heat: 72 2" -> 2 ("very hot"); 1 when no level is given."""
    try:
        return int(trigger.split(": ", 1)[1].split()[1])
    except (ValueError, IndexError):
        return 1


def line_for(level: int) -> str:
    return (getattr(config, "CPU_HOT_LINE_AGAIN", config.CPU_HOT_LINE)
            if level >= 2 else config.CPU_HOT_LINE)
