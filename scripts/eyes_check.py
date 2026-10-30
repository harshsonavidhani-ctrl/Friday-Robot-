"""Show her things and hear what she says. No run, no motors, no internet.

    ./venv/bin/python scripts/eyes_check.py [seconds] [--quiet]

The same looks and the same rules as explore_eyes during a run, so what she names
here she names while exploring. Ctrl+C ends it.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                               # noqa: E402
import explore_eyes as eyes                                 # noqa: E402
import scene                                                # noqa: E402

seconds = float(next((a for a in sys.argv[1:] if not a.startswith("-")), 60))
quiet = "--quiet" in sys.argv
detector, classifier = scene._detector(), eyes.Classifier.load()
print(f"detector {'on' if detector else 'OFF'}, classifier {'on' if classifier else 'OFF'}",
      flush=True)
if not quiet:
    import tts
    tts.speak("Ready.")                 # Piper's ~10 s load happens here, not mid-test
    print("READY", flush=True)
sights = eyes.Sightings()
end = time.monotonic() + seconds
got = {"up": 0, "down": 0}
while time.monotonic() < end:
    for which in ("up", "down"):
        frame = eyes._frame(which)
        if frame is None:
            continue
        got[which] += 1
        t0 = time.monotonic()
        seen = eyes.looked_at(which, frame, detector, classifier)
        now = time.monotonic()
        fresh = sights.update(which, seen, now)[:2]
        print(f"{which:4} {now - t0:4.2f}s {({k: round(v, 2) for k, v in seen.items()})}"
              f"{'  >>> ' + eyes.sentence(fresh) if fresh else ''}", flush=True)
        if fresh:
            sights.spoken(fresh, now)
            if not quiet:
                import tts
                tts.speak(eyes.sentence(fresh))
print("frames looked at:", got)
