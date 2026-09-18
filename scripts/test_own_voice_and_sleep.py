#!/usr/bin/env python3
"""Two fixes from 2026-09-19, so neither comes back. No mic, no panel, no money.

    ./venv/bin/python scripts/test_own_voice_and_sleep.py

1. SHE MUST NOT HEAR HERSELF. main.py pauses the detector for a voice turn, but
   exploration narration, pi_deploy, the RL scripts and say_what_you_see all
   speak from OTHER processes with the detector wide open -- and the turn log
   has her answering her own narration as a command ("...Stop! Stop! I'd
   detected an obstacle 0.7 meters ahead."). tts.speak() now marks
   state/speaking.json wherever it runs and the detector honours it.

   ⚠ The failure direction is deafness, so the cases that matter most here are
   the broken ones: a stale mark, a corrupt mark, a missing file.

2. THE SLEEPING FACE MUST STICK. Harsh: "when sleeping face is triggered it
   will remain sleeping face until wake word spoken". The turn that put her to
   sleep used to end in rest(), which reset the idle clock and put neutral
   back, and 45 s later the drift showed bored.

Nothing here touches the real panel: face_control.set_emotion is replaced, so
this is safe to run while Friday is live.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                           # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def own_voice() -> None:
    import tts
    import wake_word

    config.SPEAKING_FILE = ROOT / "state" / "speaking.test.json"
    config.SPEAKING_FILE.unlink(missing_ok=True)
    det = wake_word.OpenWakeWord.__new__(wake_word.OpenWakeWord)
    det._own_voice_until = 0.0
    det._own_voice_checked = 0.0
    det._heard_herself = False

    def deaf() -> bool:
        det._own_voice_checked = 0.0            # force a re-read
        return det._hearing_herself(time.monotonic())

    print("SHE MUST NOT HEAR HERSELF")
    check("no mark at all: she listens", not deaf())
    with tts._audible():
        check("another process is speaking: she does not", deaf())
    check("the instant she stops, the tail is still in the room", deaf())
    time.sleep(config.AUDIO_DRAIN_S + config.WAKE_RESUME_SETTLE_S + 0.05)
    check("after the drain and the settle, she listens again", not deaf())

    config.SPEAKING_FILE.write_text(json.dumps(
        {"state": "speaking", "at": time.time() - 999, "until": 0, "pid": 1}))
    det._own_voice_until = 0.0
    check("a mark left by a process that DIED does not deafen her", not deaf(),
          "a stale mark must never mute the wake word")

    config.SPEAKING_FILE.write_text("{ not json")
    det._own_voice_until = 0.0
    check("a corrupt mark does not deafen her", not deaf())
    config.SPEAKING_FILE.unlink(missing_ok=True)


def sleeping_face() -> None:
    import face_control
    import face_director

    written: list[str] = []
    face_control.set_emotion = lambda name: written.append(name)
    face_director._emotion_stamp = lambda: None          # no foreign writes

    config.FACE_BORED_AFTER_S = 0.3
    config.FACE_SLEEPY_AFTER_S = 0.6
    config.FACE_SLEEPING_AFTER_S = 0.9
    config.FACE_TICK_S = 0.05
    config.FACE_MIN_DWELL_S = 0.0

    d = face_director.FaceDirector().start()
    try:
        def face(label: str, want: str) -> None:
            check(label, d.state()["face"] == want,
                  f"{d.state()['face']} (wanted {want})")

        print("\nTHE SLEEPING FACE MUST STICK")
        d.pipeline("sleeping")
        face("told to sleep", "sleeping")
        d.rest()
        face("the turn that put her to sleep ends", "sleeping")
        time.sleep(1.2)
        face("a second later — this is where bored used to appear", "sleeping")
        d.rest()
        time.sleep(0.6)
        face("another rest and another wait", "sleeping")
        d.pipeline("listening")
        face("the wake word takes her out of it", "listening")
        d.rest()
        time.sleep(0.5)
        face("and bored still works once she is awake", "bored")

        time.sleep(0.7)
        face("she drifts off on her own", "sleeping")
        d.rest()
        time.sleep(0.6)
        face("resting does not restart the drift", "sleeping")
        d.emotion("happy")
        face("but the brain can still change her face", "happy")
    finally:
        d.close()


def main() -> int:
    own_voice()
    sleeping_face()
    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    return 0 if all(_results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
