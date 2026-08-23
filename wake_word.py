"""wake_word.py — the trigger that starts a turn.

One interface, several implementations, so Step 2 can swap the real one in
without touching main.py:

    for trigger in wake_word.listen():      # yields once per wake
        run_a_turn()

WHAT IS HERE NOW (Step 1):

  * EnterKey — press Enter. Always works, needs no model, and is what you want
    while testing the rest of the pipeline.
  * SpokenWord — record a short clip, transcribe it, look for "friday".

WHAT IS COMING (Step 2): openWakeWord running a custom friday.onnx continuously
on the mic.

⚠ SpokenWord IS A STOPGAP AND IT IS THE WEAK ONE. It is deaf while it is
transcribing: each pass records a window, then spends time in whisper, and
anything said during that gap is simply not heard. That is precisely the
record→transcribe→check loop CLAUDE.md warns against, and it is why
openWakeWord is the target rather than a nice-to-have. It is included because a
stopgap you can talk to beats no voice path at all — not because it is good.

Both implementations are cheap to construct and yield the same thing, so
main.py does not care which one it got.
"""

from __future__ import annotations

import logging
import sys
import threading
from typing import Iterator

import config

log = logging.getLogger("friday.wake")

WAKE_WORDS = ("friday", "फ्राइडे", "fryday", "frday")


class WakeSource:
    """Something that yields once per wake."""

    name = "wake"

    def listen(self) -> Iterator[str]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class EnterKey(WakeSource):
    """Press Enter to wake Friday. The reliable one."""

    name = "enter-key"

    def __init__(self, prompt: str = "\n[press Enter to talk to Friday] ") -> None:
        self.prompt = prompt

    def listen(self) -> Iterator[str]:
        while True:
            try:
                sys.stdin.readline()
            except (KeyboardInterrupt, EOFError):
                return
            yield "enter"


class SpokenWord(WakeSource):
    """Listen for the word "friday" in a short transcript.

    A stopgap. See the module docstring for why this is the weak path — it is
    deaf while transcribing, so it drops wakes.
    """

    name = "spoken-word"

    def __init__(self, words: tuple[str, ...] = WAKE_WORDS) -> None:
        self.words = words

    def listen(self) -> Iterator[str]:
        import stt

        stt.model()                      # load before the first pass, not during
        with stt.Microphone() as mic:
            while True:
                try:
                    audio = mic.listen(
                        max_seconds=config.WAKE_WINDOW_S,
                        start_timeout=config.WAKE_POLL_TIMEOUT_S)
                except stt.Silence:
                    continue             # nobody spoke; go round again
                except stt.STTError as exc:
                    log.error("wake listener failed: %s", exc)
                    return
                heard = stt.transcribe(audio).lower()
                if not heard:
                    continue
                if any(word in heard for word in self.words):
                    log.info("woken by %r", heard)
                    yield heard


class AnyOf(WakeSource):
    """Several sources at once — the first to fire wakes Friday.

    Lets Harsh say "friday" OR press Enter, which matters while the spoken
    path is still the unreliable stopgap: there is always a way in.
    """

    name = "any"

    def __init__(self, *sources: WakeSource) -> None:
        self.sources = sources

    def listen(self) -> Iterator[str]:
        import queue

        fired: queue.Queue = queue.Queue()
        stop = threading.Event()

        def pump(source: WakeSource) -> None:
            try:
                for trigger in source.listen():
                    if stop.is_set():
                        return
                    fired.put(f"{source.name}: {trigger}")
            except Exception as exc:            # noqa: BLE001
                log.error("wake source %s died: %s", source.name, exc)

        for source in self.sources:
            threading.Thread(target=pump, args=(source,),
                             name=f"wake-{source.name}", daemon=True).start()
        try:
            while True:
                yield fired.get()
        finally:
            stop.set()


def make(kind: str | None = None) -> WakeSource:
    """The wake source named in config, or the one asked for."""
    kind = (kind or config.WAKE_SOURCE).strip().lower()
    if kind == "enter":
        return EnterKey()
    if kind == "voice":
        return SpokenWord()
    if kind == "both":
        return AnyOf(EnterKey(), SpokenWord())
    raise ValueError(f"unknown wake source {kind!r} — use enter, voice or both")


def listen(kind: str | None = None) -> Iterator[str]:
    """Convenience: yield once per wake, using the configured source."""
    yield from make(kind).listen()


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Wait for a wake and report it.")
    ap.add_argument("--source", default=None,
                    choices=["enter", "voice", "both"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    source = make(args.source)
    print(f"wake source: {source.name}. Ctrl-C to stop.")
    try:
        for trigger in source.listen():
            print(f"  WOKEN — {trigger}")
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
