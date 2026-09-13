#!/usr/bin/env python3
"""Old verifier vs new, on the clips that actually matter tonight.

    ./venv/bin/python scripts/verifier_before_after.py OLD.npz [NEW.npz]

ship_verifier's own gate checks recorded takes, other voices and unseen
conversation. It does NOT report the thing Harsh complained about: live
"hey Friday" said mid-conversation being turned away (2026-09-13, "Hey Friday,
good morning." rejected at 0.09). So this scores every harvested fire an
independent transcript says contains her name, plus the ones it says do not,
through both models -- the same features, the same decision the detector makes.

⚠ The name-clips were also TRAINING data for the new model, so its numbers on
them are optimistic. Read the conversation column as the honest one, and the
real test as Harsh saying it live.
"""
import re
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                                # noqa: E402
import wake_verify                                           # noqa: E402

LABELS = ROOT / "assets/models/wake/whisper_heard.tsv"
NAME = re.compile(r"\bfri ?day\b")
CHUNK = 1280


def labelled():
    """(path, has_name) for every labelled clip still on disk."""
    rows = []
    for line in LABELS.read_text(errors="replace").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3 or not Path(parts[1]).exists():
            continue
        rows.append((parts[1], bool(NAME.search(parts[2].lower()))))
    return rows


def best_score(path, oww, ver):
    """Highest verifier probability at any fire the detector would confirm."""
    try:
        with wave.open(path) as w:
            s = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    except Exception:
        return None
    oww.reset()
    above, best = 0, None
    for i in range(0, len(s) - CHUNK + 1, CHUNK):
        sc = oww.predict(s[i:i + CHUNK])
        v = float(max(sc.values()) if sc else 0.0)
        above = above + 1 if v >= config.WAKE_THRESHOLD else 0
        if above >= max(1, config.WAKE_PATIENCE):
            above = 0
            frames = max(oww.model_inputs.values())
            _, p = ver.accepts(oww.preprocessor.get_features(frames))
            if p is not None:
                best = p if best is None else max(best, p)
    return best


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    old = sys.argv[1]
    new = sys.argv[2] if len(sys.argv) > 2 else str(config.WAKE_VERIFIER_MODEL)
    from openwakeword.model import Model
    oww = Model(wakeword_model_paths=[str(config.WAKE_WORD_MODEL)],
                vad_threshold=config.WAKE_VAD_THRESHOLD)

    rows = labelled()
    print(f"{sum(h for _, h in rows)} clips with her name, "
          f"{sum(not h for _, h in rows)} without\n", flush=True)

    scores = {}
    for tag, path in (("old", old), ("new", new)):
        ver = wake_verify.Verifier(path=Path(path), threshold=0.0)
        ver.load()
        scores[tag] = [best_score(p, oww, ver) for p, _ in rows]
        print(f"scored with {tag}: {path}", flush=True)

    thresholds = {"old": 0.3, "new": config.WAKE_VERIFY_THRESHOLD}
    print(f"\n{'':8s}{'her name, WOKE':>18s}{'conversation, WOKE':>22s}"
          f"   (old at v>=0.30, new at v>={thresholds['new']})")
    for tag in ("old", "new"):
        t = thresholds[tag]
        name_n = name_hit = conv_n = conv_hit = 0
        for (p, has), sc in zip(rows, scores[tag]):
            if sc is None:
                continue                    # openWakeWord never confirmed it
            if has:
                name_n += 1
                name_hit += sc >= t
            else:
                conv_n += 1
                conv_hit += sc >= t
        print(f"{tag:8s}{name_hit:>12d}/{name_n:<5d}{conv_hit:>16d}/{conv_n:<5d}")

    print("\nher-name clips, old -> new verifier score:")
    for (p, has), so, sn in zip(rows, scores["old"], scores["new"]):
        if has and so is not None:
            flag = "RESCUED" if so < 0.3 <= (sn or 0) + 1e-9 and \
                (sn or 0) >= thresholds["new"] else ""
            print(f"  {Path(p).name:34s} {so:5.2f} -> {sn:5.2f}  {flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
