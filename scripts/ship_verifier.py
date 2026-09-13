#!/usr/bin/env python3
"""Fit and ship the wake verifier -- the one thing that stops conversations waking Friday.

    nice -n 19 ./venv/bin/python scripts/ship_verifier.py

Streams only the clips the fit needs, checks itself on clips it never trained
on, and writes assets/models/wake/friday_verifier.npz + config's
WAKE_VERIFY_THRESHOLD ONLY IF real wakes survive. See scripts/train_verifier.py
for why a verifier exists at all.

TWO THINGS THE FIRST ATTEMPT GOT WRONG, both fixed here:
  * LIVE wakes were never positives -- only clean recorded takes -- so it
    learned "silence before the phrase" and rejected 3 of 4 real live wakes.
    The fires a command followed, and the fires Vosk heard "friday" in, are
    positives now, checked two-fold so the report is still honest.
  * Only Harsh's voice was a positive. A few Piper voices go in too, one held
    out, so it is not tuned to him alone -- the judges are not him.
"""
import sys
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                                # noqa: E402
import train_verifier as tv                                  # noqa: E402

WAKE = config.ROOT / "assets/models/wake"
SYNTH = config.ROOT / "state" / "verifier_synth"
VOICES = ["en_US-lessac-medium", "en_US-amy-low", "en_GB-jenny_dioco-medium"]
PHRASES = ["hey friday", "Hey, Friday."]


def make_synth() -> list[str]:
    """A few other voices saying it, so the fit is not only Harsh's voice."""
    SYNTH.mkdir(parents=True, exist_ok=True)
    want = [SYNTH / f"{v}-{i}{s}.wav" for v in VOICES
            for i in range(len(PHRASES)) for s in ("", "-far")]
    if not all(p.exists() for p in want):
        import train_wake as tw
        from piper import PiperVoice
        rng = np.random.default_rng(7)
        for v in VOICES:
            voice = PiperVoice.load(str(config.MODEL_DIR / "piper" / f"{v}.onnx"))
            for i, text in enumerate(PHRASES):
                clip = tw.synth(voice, text, 1.0)
                for name, audio in ((f"{v}-{i}.wav", clip),
                                    (f"{v}-{i}-far.wav",
                                     tw.augment(tw.place(clip, rng), rng))):
                    with wave.open(str(SYNTH / name), "wb") as f:
                        f.setnchannels(1)
                        f.setsampwidth(2)
                        f.setframerate(16000)
                        f.writeframes(np.asarray(audio, np.int16).tobytes())
            print(f"  synthesised {v}", flush=True)
    return sorted(str(p) for p in SYNTH.glob("*.wav"))


def main() -> int:
    # BOTH LABELLERS. Vosk did 68 clips at 3.6 s each; Whisper is already on
    # this Pi and catches up with the pile (scripts/whisper_label_fires.py).
    # Vosk wins a disagreement only because it was the one checked against
    # Harsh's ears; in practice they rarely overlap.
    # ⚠ LOWERCASED, OR NONE OF THEM COUNT. train_verifier's FRIDAY regex is
    # r"\bfri ?day\b" with no IGNORECASE -- written for Vosk, which emits
    # lowercase. Whisper writes "Hey Friday.", so without this every Whisper
    # positive fails the match and the retrain quietly gets zero new wakes.
    vosk = {k: v.lower()
            for k, v in tv.read_vosk(WAKE / "whisper_heard.tsv").items()}
    vosk.update(tv.read_vosk(WAKE / "vosk_heard.tsv"))
    print(f"{len(vosk)} clips have an independent transcript", flush=True)
    rows = tv.clip_table(None, "20260911-2350", vosk, set())
    for p in make_synth():
        held = Path(p).stem.startswith(VOICES[0])
        rows.append((p, "other voices", 1, "test" if held else "train"))
    print(f"{len(rows)} clips ({sum(r[2] for r in rows)} wake, "
          f"{sum(1 - r[2] for r in rows)} not)", flush=True)

    cache = tv.build_cache([r[0] for r in rows], 1)
    rows = [r for r in rows if r[0] in cache]
    clips, feats = tv.assemble(rows, cache)

    live = [c for c in clips if c["group"] == "fire then command"]
    for n, c in enumerate(live):
        c["fold"] = n % 2
    T, P = config.WAKE_THRESHOLD, config.WAKE_PATIENCE
    levels = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    groups = ["harsh takes", "other voices", "unseen conversation",
              "labelled false fires"]

    def rows_for(sel):
        x, y = [], []
        for c in sel:
            for k, r in c["feat"].items():
                if c["scores"][k] >= 0.3:
                    x.append(feats[r])
                    y.append(c["label"])
        return np.array(x, np.float32), np.array(y)

    def woke(c, model, v):
        return tv.live_decision(c["scores"], c["feat"], feats, model, T, P, v)

    test = {g: [c for c in clips if c["group"] == g and c["split"] == "test"]
            for g in groups}
    base = {g: sum(woke(c, None, None) for c in cs) for g, cs in test.items()}
    base_live = sum(woke(c, None, None) for c in live)
    got = {v: dict.fromkeys(groups, 0) | {"live": 0} for v in levels}
    for fold in (0, 1):
        train = [c for c in clips if c["split"] == "train"
                 and c["group"] != "fire then command"]
        train += [c for c in live if c["fold"] != fold]
        model = tv.fit(*rows_for(train), 0.001)
        for v in levels:
            got[v]["live"] += sum(woke(c, model, v) for c in live
                                  if c["fold"] == fold)
            if fold == 0:
                for g in groups:
                    got[v][g] = sum(woke(c, model, v) for c in test[g])

    print(f"\nheld-out, threshold {T}, patience {P} (how many WAKE her)")
    print(f"{'':34s}{'now':>8s}" + "".join(f"{'v>=' + str(v):>9s}" for v in levels))
    def line(name, want, b, n, per):
        print(f"  {name:20s} {want:11s}{b:>4d}/{n:<3d}"
              + "".join(f"{per[v]:>6d}/{n:<3d}" for v in levels))
    line("live wakes", "SHOULD wake", base_live, len(live),
         {v: got[v]["live"] for v in levels})
    for g in groups:
        want = "SHOULD wake" if (test[g] and test[g][0]["label"]) else "should NOT"
        line(g, want, base[g], len(test[g]), {v: got[v][g] for v in levels})

    ok = [v for v in levels
          if got[v]["live"] >= base_live - 2
          and got[v]["harsh takes"] >= base["harsh takes"]
          and got[v]["other voices"] >= base["other voices"] - 1]
    if not ok:
        print("\nNO SAFE THRESHOLD — real wakes would be lost. Nothing written.")
        return 1
    chosen = max(ok)
    quiet = base["unseen conversation"] - got[chosen]["unseen conversation"]
    print(f"\nchosen v>={chosen}: conversation wakes "
          f"{base['unseen conversation']} -> {got[chosen]['unseen conversation']} "
          f"of {len(test['unseen conversation'])} ({quiet} fewer), live wakes "
          f"{got[chosen]['live']}/{len(live)} kept")

    model = tv.fit(*rows_for(clips), 0.001)
    out = Path(config.WAKE_VERIFIER_MODEL)
    np.savez(out, **model, C=np.array(0.001, np.float32),
             train_score=np.array(0.3, np.float32))
    src = Path(config.ROOT / "config.py").read_text()
    old = f"WAKE_VERIFY_THRESHOLD = {config.WAKE_VERIFY_THRESHOLD}\n"
    if src.count(old) != 1:
        print(f"could not set the threshold in config.py ({old.strip()!r})")
        return 1
    (config.ROOT / "config.py").write_text(
        src.replace(old, f"WAKE_VERIFY_THRESHOLD = {chosen}\n"))
    print(f"wrote {out} ({out.stat().st_size // 1024} kB) and set "
          f"WAKE_VERIFY_THRESHOLD = {chosen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
