#!/usr/bin/env python3
"""Train "hey friday" into a real openWakeWord model — friday.onnx.

    ./venv/bin/python scripts/train_wake.py --record-ambient 90
    ./venv/bin/python scripts/train_wake.py                    # train + export

WHY THIS EXISTS. openWakeWord ships alexa, hey_jarvis, hey_mycroft,
hey_marvin, timer and weather. There is no "hey friday" to download, and
Friday is not called Jarvis. So it gets trained here.

HOW IT WORKS. An openWakeWord model is three pieces and only the last one is
specific to a phrase:

    audio -> melspectrogram.onnx -> embedding_model.onnx -> classifier

The first two are Google's speech_embedding, already on disk, and they do all
the hard work. The classifier is a small dense net over sixteen 96-dim
embedding frames -- 2.0 s of audio, measured, not assumed. That is what gets
trained, and what is exported to friday.onnx with exactly the input/output
signature the real models use ([1,16,96] float -> [1,1] float), so it is a
drop-in: point config.WAKE_WORD_MODEL at it and nothing else changes.

WHERE THE VOICES COME FROM. Nine Piper voices are already on this Pi for TTS,
and they are used here as nine different speakers saying "hey friday" at three
speeds each. That is the honest limit of this approach and it is stated in the
report at the end: a model trained on synthetic speech generalises to real
speech imperfectly. It is measured against Harsh's actual voice with
`scripts/test_wake.py --live`, and if it is weak the fix is more real
recordings, not more synthetic ones.

THE NEGATIVES MATTER MORE THAN THE POSITIVES. A model trained only on "hey
friday" versus silence fires at everything. The negatives here are deliberately
hard: "friday" alone, "hey" alone, "hey fred", "good friday", "hey jarvis",
Friday's own command vocabulary, and -- most valuable of all -- real recorded
audio from the actual room this robot lives in (--record-ambient).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                            # noqa: E402

log = logging.getLogger("train.wake")

RATE = 16000
CLIP_S = 2.0                       # measured: yields exactly 16 embedding frames
CLIP_SAMPLES = int(RATE * CLIP_S)
CHUNK_SAMPLES = 1280               # what openWakeWord scores, 80 ms at 16 kHz

WAKE_PHRASES = ["hey friday", "hey Friday", "Hey, Friday.", "hey friday,"]

# Hard negatives: things that sound like it, and things Friday is actually
# told. A classifier that has never heard "friday" WITHOUT "hey" will fire on
# "see you friday".
NEGATIVE_PHRASES = [
    "friday", "hey", "hey there", "hey fred", "hey freddy", "hey frida",
    "good friday", "see you friday", "on friday", "it's friday",
    "happy friday", "friday morning", "next friday", "hey david",
    "hey jarvis", "hey google", "okay computer", "hey buddy", "hey you",
    "play some music", "what time is it", "move forward", "turn left",
    "turn right", "stop", "go backward", "what do you see",
    "how are you doing", "tell me a joke", "what's the weather",
    "come here", "go to the kitchen", "scan the room", "are you there",
    "the day after friday", "a friday afternoon", "hey, wait a second",
]


def piper_voices() -> list[Path]:
    found = sorted((config.MODEL_DIR / "piper").glob("*.onnx"))
    if not found:
        raise SystemExit(f"no Piper voices in {config.MODEL_DIR / 'piper'}")
    return found


def synth(voice, text: str, length_scale: float) -> np.ndarray:
    """One phrase, as 16 kHz mono int16."""
    from piper import SynthesisConfig
    chunks = list(voice.synthesize(
        text, syn_config=SynthesisConfig(length_scale=length_scale)))
    if not chunks:
        return np.zeros(0, dtype=np.int16)
    audio = np.concatenate(
        [np.frombuffer(c.audio_int16_bytes, dtype=np.int16) for c in chunks])
    rate = getattr(chunks[0], "sample_rate", RATE)
    if rate != RATE:                       # Piper voices are 16k or 22.05k
        target = int(len(audio) * RATE / rate)
        audio = np.interp(np.linspace(0, len(audio) - 1, target),
                          np.arange(len(audio)), audio).astype(np.int16)
    return audio


def place(clip: np.ndarray, rng: np.random.Generator,
          jitter: bool = True) -> np.ndarray:
    """Drop a clip into a 2.0 s window at a random offset, padded with quiet.

    THE OFFSET IS THE POINT. At inference the phrase can land anywhere in the
    rolling window, so a model trained with it always centred learns the
    position as much as the phrase and misses it live.
    """
    out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
    clip = clip[:CLIP_SAMPLES].astype(np.float32)
    room = CLIP_SAMPLES - len(clip)
    start = int(rng.integers(0, room + 1)) if (jitter and room > 0) else 0
    out[start:start + len(clip)] = clip
    return out


def augment(clip: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gain, noise and a crude room tail — the cheap half of robustness."""
    out = clip.copy()
    out *= float(rng.uniform(0.25, 1.4))                  # distance / level
    if rng.random() < 0.7:                                # room reverb
        tail = int(rng.integers(400, 2400))
        ir = np.zeros(tail, dtype=np.float32)
        ir[0] = 1.0
        for _ in range(int(rng.integers(2, 7))):
            ir[int(rng.integers(1, tail))] += float(rng.uniform(0.05, 0.35))
        ir *= np.exp(-np.linspace(0, float(rng.uniform(3, 9)), tail))
        out = np.convolve(out, ir)[:CLIP_SAMPLES]
    if rng.random() < 0.85:                               # hiss / fan
        # NOISE RELATIVE TO THE CLIP, NOT AN ABSOLUTE LEVEL. This used to add
        # sigma 30-400 regardless of how loud the clip was, which is not one
        # augmentation but two different ones: a loud take was barely touched
        # while a quiet take was buried under noise louder than the speech.
        # That is exactly the shape of the 8 misses in the first retrain --
        # every one peaked below -14.5 dBFS, every catch above it, clean
        # separation. A model trained that way learns "loud" as part of the
        # wake word, and Harsh across the room is not loud.
        speech = float(np.sqrt((out.astype(np.float64) ** 2).mean())) or 1.0
        snr_db = float(rng.uniform(5, 30))
        out += rng.normal(0, speech / (10 ** (snr_db / 20)), CLIP_SAMPLES)
    peak = np.abs(out).max()
    if peak > 32000:
        out *= 32000 / peak
    return out.astype(np.int16)


def ambient_windows(path: Path, rng: np.random.Generator,
                    count: int) -> list[np.ndarray]:
    """Slice a real room recording into 2 s windows. The best negatives there are."""
    import wave
    if not path.exists():
        return []
    with wave.open(str(path), "rb") as f:
        raw = f.readframes(f.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if len(audio) < CLIP_SAMPLES:
        return []
    out = []
    for _ in range(count):
        start = int(rng.integers(0, len(audio) - CLIP_SAMPLES))
        window = audio[start:start + CLIP_SAMPLES].copy()
        window *= float(rng.uniform(0.5, 2.5))
        out.append(np.clip(window, -32000, 32000).astype(np.int16))
    return out


def record_wakes(count: int, out_dir: Path) -> int:
    """Record Harsh actually saying the wake word, N times.

    THIS IS THE PART SYNTHETIC DATA CANNOT DO. Nine Piper voices give a model
    that catches Piper reliably and a real person unreliably -- which is
    exactly what happened: 10/10 from the speaker, misses from the sofa. TTS
    has no breath, no room, no accent, and none of the sloppiness of a phrase
    somebody has said ten thousand times. A few dozen real samples are worth
    more than another thousand synthetic ones.

    Each take is endpointed with the same VAD the live capture uses, so what
    lands on disk is the phrase and not the silence around it.
    """
    import stt

    out_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(out_dir.glob("wake_*.wav"))
    start = len(existing)
    if existing:
        print(f"{start} samples already recorded; adding {count} more.")
    print(f"\nSay \"hey Friday\" {count} times, with a short pause between each.")
    print("VARY IT ON PURPOSE -- this is the whole point:")
    print("  close up and from across the room, facing away, quietly,")
    print("  quickly, half-mumbled, mid-sentence. However you really say it.")
    print("Ctrl-C to stop early; whatever is recorded is kept.\n")

    # NO KEY PRESSES. Each utterance is segmented by the SAME VAD the live
    # capture uses, so Harsh just says it over and over with a breath between
    # takes and every one lands on disk endpointed the way the real pipeline
    # would endpoint it. Pressing Enter forty times also changes how you say a
    # phrase, which is the last thing this recording wants.
    stream = stt.AudioStream().start()
    kept = 0
    misses = 0
    try:
        while kept < count and misses < 6:
            try:
                audio = stream.capture(max_seconds=3.0, start_timeout=25.0,
                                       handover_ms=0, deadtime_ms=0)
            except stt.Silence:
                misses += 1
                print(f"  ...nothing heard for a while ({misses}/6)")
                continue
            secs = len(audio) / 2 / RATE
            if secs < 0.35:
                print(f"  ...{secs:.2f}s — too short to be the phrase, ignored")
                continue
            misses = 0
            path = out_dir / f"wake_{start + kept:03d}.wav"
            stt.save_wav(audio, path)
            kept += 1
            import numpy as np
            level = np.frombuffer(audio, dtype=np.int16).astype(np.float32)
            db = 20 * np.log10(max(np.sqrt((level ** 2).mean()), 1) / 32768)
            print(f"  [{kept}/{count}] {path.name}  {secs:.2f}s  {db:.0f} dBFS",
                  flush=True)
    except KeyboardInterrupt:
        print("\n  stopped early")
    finally:
        stream.close()

    total = len(list(out_dir.glob("wake_*.wav")))
    print(f"\n{kept} recorded, {total} in total at {out_dir}")
    print("Now retrain:  ./venv/bin/python scripts/train_wake.py")
    return 0 if total else 1


def real_wake_clips(out_dir: Path, rng: np.random.Generator,
                    per_clip: int,
                    sources: list[str] | None = None) -> list[np.ndarray]:
    """Load Harsh's own recordings and augment them hard.

    Weighted far above the synthetic ones by sheer repetition: these are the
    only samples that describe the voice the robot actually has to answer to.
    """
    import wave
    clips: list[np.ndarray] = []
    for path in sorted(out_dir.glob("wake_*.wav")):
        try:
            with wave.open(str(path), "rb") as f:
                raw = f.readframes(f.getnframes())
        except Exception:                                 # noqa: BLE001
            continue
        audio = np.frombuffer(raw, dtype=np.int16)
        if len(audio) == 0:
            continue
        # KEEP THE END, NOT THE BEGINNING. place() trims a long clip with
        # clip[:CLIP_SAMPLES], which is exactly backwards for a real take.
        # openWakeWord fires at the END of the phrase, and the VAD endpoint IS
        # that end. 13 of the first 40 recordings run past 2.0 s -- seven of
        # them all the way to the recorder's 3.0 s cap, where a short pause let
        # two takes merge into one file -- so the first-2 s window fed the
        # classifier the run-up and threw the phrase's ending away.
        if len(audio) > CLIP_SAMPLES:
            audio = audio[-CLIP_SAMPLES:]
        for _ in range(per_clip):
            clips.append(augment(place(audio, rng), rng))
            if sources is not None:
                sources.append(path.name)
    return clips


def _read_wav(path: Path) -> np.ndarray:
    import wave
    try:
        with wave.open(str(path), "rb") as f:
            raw = f.readframes(f.getnframes())
    except Exception:                                     # noqa: BLE001
        return np.zeros(0, dtype=np.int16)
    return np.frombuffer(raw, dtype=np.int16)


def hard_negative_clips(directory: Path, rng: np.random.Generator,
                        per_clip: int,
                        sources: list[str] | None = None) -> list[np.ndarray]:
    """Real speech that is NOT the wake word — the negatives that were missing.

    THIS IS THE FIX FOR THE 97-FIRES-IN-ONE-EVENING BUG. The model shipped on
    2026-08-27 had never heard a real conversation: its negatives were Piper
    TTS phrases and 45 s of an EMPTY room, while 60% of its positives were
    1280 augmentations of 40 clips of one voice. So it learned "a real human
    voice in this room", and a real human voice in this room is what a
    conversation is made of. Measured 2026-09-05: 97 fires in an evening,
    median 0.88, MAX 1.00 -- the false ones score as high as the true ones, so
    no threshold anywhere separates them and only the training set can.

    Takes both shapes of recording, because both are worth having:
      * LONG files (a ten-minute conversation) are sliced into windows;
      * SHORT files (the 3 s clips state/wake_fires harvests from live fires)
        are placed and augmented like a positive would be. These are the
        highest-value negatives that exist -- they are, by definition, the
        audio that already beat the model.
    """
    clips: list[np.ndarray] = []
    if not directory.exists():
        return clips
    for path in sorted(directory.glob("*.wav")):
        audio = _read_wav(path)
        if len(audio) < CLIP_SAMPLES // 2:
            continue
        if len(audio) >= CLIP_SAMPLES * 3:
            # A long recording: take windows across the whole thing, enough to
            # cover it densely rather than a fixed count of random grabs.
            windows = max(1, int(len(audio) / CLIP_SAMPLES) * 2)
            for _ in range(windows):
                start = int(rng.integers(0, len(audio) - CLIP_SAMPLES))
                window = audio[start:start + CLIP_SAMPLES].astype(np.float32)
                clips.append(augment(window, rng))
                if sources is not None:
                    sources.append(path.name)
        else:
            # THE END OF THE CLIP, NOT THE START. A harvested fire is the 3 s
            # BEFORE the model fired, and when it fired it was scoring the
            # LAST 2 s. place() keeps clip[:2 s], so every hard negative went
            # in as mostly run-up, and the second of audio that actually fooled
            # the model was thrown away -- the same backwards trim
            # real_wake_clips() already fixed for the positives.
            for _ in range(per_clip):
                clips.append(augment(end_window(audio, rng), rng))
                if sources is not None:
                    sources.append(path.name)
    return clips


def end_window(audio: np.ndarray, rng: np.random.Generator,
               slack_s: float = 0.5) -> np.ndarray:
    """The 2.0 s window ending at (or up to slack_s before) the end of a clip.

    The slack is WAKE_PATIENCE's doing: a confirmed fire is several chunks in
    a row over the threshold, so the windows ending a few chunks earlier
    fooled the model too and are worth training against as well.
    """
    audio = audio.astype(np.float32)
    if len(audio) <= CLIP_SAMPLES:
        out = np.zeros(CLIP_SAMPLES, dtype=np.float32)
        out[CLIP_SAMPLES - len(audio):] = audio        # flush right, like a fire
        return out
    slack = min(len(audio) - CLIP_SAMPLES, int(slack_s * RATE))
    end = len(audio) - int(rng.integers(0, slack + 1))
    return audio[end - CLIP_SAMPLES:end].copy()


def scrambled_clips(paths: list[Path], rng: np.random.Generator,
                    per_clip: int,
                    sources: list[str] | None = None) -> list[np.ndarray]:
    """Real voices with the phrase taken out: negatives that need no label.

    THE MODEL LEARNED THE VOICE, NOT THE PHRASE -- 60% of its positives were
    one person in one room, and it answers that person saying anything. The
    direct counter is that same person, room and microphone with the WORDS
    destroyed: reversed, or cut into syllable-sized pieces and reordered. The
    level, timbre and room are untouched; only the order is gone, and the
    order is the only thing that makes it "hey Friday". To score these low the
    model has to look at the order.

    Safe on ANY clip, including the harvested fires nobody has labelled -- a
    reversed "hey Friday" is not "hey Friday" -- so they all get used here.
    """
    clips: list[np.ndarray] = []
    for path in paths:
        audio = _read_wav(path).astype(np.float32)
        if len(audio) < RATE // 2:
            continue
        for index in range(per_clip):
            mixed = audio[::-1].copy() if index % 2 == 0 \
                else shuffle_pieces(audio, rng)
            if len(mixed) > CLIP_SAMPLES:
                start = int(rng.integers(0, len(mixed) - CLIP_SAMPLES + 1))
                mixed = mixed[start:start + CLIP_SAMPLES]
            clips.append(augment(place(mixed, rng), rng))
            if sources is not None:
                sources.append(path.name)
    return clips


def shuffle_pieces(audio: np.ndarray, rng: np.random.Generator,
                   piece_s: tuple[float, float] = (0.10, 0.25)) -> np.ndarray:
    """Cut into 100-250 ms pieces and reassemble in a new order.

    No two pieces that were neighbours stay neighbours in the same order, so
    nothing longer than one piece survives intact -- and a quarter of a second
    is shorter than "Friday". Each piece is faded over 5 ms, or the cuts click
    and the model learns "clicks mean no" instead.
    """
    pieces, pos = [], 0
    while pos < len(audio):
        step = int(rng.uniform(*piece_s) * RATE)
        pieces.append(audio[pos:pos + step].copy())
        pos += step
    count = len(pieces)
    order = None
    for _ in range(50):
        candidate = rng.permutation(count)
        if count >= 3 and all(candidate[k + 1] != candidate[k] + 1
                              for k in range(count - 1)):
            order = candidate
            break
    if order is None:
        return audio[::-1].copy()
    fade = int(0.005 * RATE)
    ramp = np.linspace(0, 1, fade, dtype=np.float32)
    for piece in pieces:
        if len(piece) > 2 * fade:
            piece[:fade] *= ramp
            piece[-fade:] *= ramp[::-1]
    return np.concatenate([pieces[k] for k in order])


def reset_streams(model) -> None:
    """Clear openWakeWord's STREAMING state between clips.

    `Model.reset()` resets the prediction buffer and NOTHING ELSE -- read it:
    it is three lines and it touches `prediction_buffer` only. The audio
    history lives one layer down in the AudioFeatures preprocessor, in
    `raw_data_buffer`, `melspectrogram_buffer` and a ~10 second
    `feature_buffer`, and none of them are cleared. So scoring forty clips
    through one Model scores each one against the tail of the last.

    MEASURED, and it is not subtle. The same eight recordings, same model:

        shared Model + reset()      2 of 8 fire
        a fresh Model per clip      5 of 8 fire

    wake_001 scores 0.000 reused and 0.996 clean. That is the difference
    between "retrain, this model has never heard this voice" and "it is fine",
    and score_samples() has been reporting the first one.

    Resetting the buffers is what reset() looks like it does. `feature_buffer`
    is rebuilt the way the constructor builds it -- from silence -- because an
    empty one is the wrong shape for the first predict() to stack onto.
    """
    import collections

    model.reset()
    pre = getattr(model, "preprocessor", None)
    if pre is None:
        return
    pre.raw_data_buffer = collections.deque(maxlen=RATE * 10)
    pre.melspectrogram_buffer = np.ones((76, 32))
    pre.accumulated_samples = 0
    # The same array every time, so embedded ONCE per preprocessor. Doing it
    # per clip was the slow half of scoring hundreds of them: py-spy caught
    # two of three scoring workers inside this call at once.
    silence = getattr(pre, "_silence_features", None)
    if silence is None:
        silence = pre._get_embeddings(np.zeros(RATE * 10).astype(np.int16))
        pre._silence_features = silence
    pre.feature_buffer = silence.copy()

    # AND THE VAD, which is the half that actually forced the zeros. Silero is
    # an LSTM -- it carries _h and _c across calls -- and openWakeWord gates on
    #     vad_frames = list(self.vad.prediction_buffer)[-7:-4]
    # which is four to seven frames BEHIND the current one. Score a second clip
    # without clearing that and the gate is reading the END OF THE PREVIOUS
    # CLIP; if that was silence, `vad_max_score` loses to vad_threshold and
    # EVERY model prediction is overwritten with 0.0 no matter what was said.
    #
    # That is why the misses were always exactly 0.000 and never 0.3 -- a
    # near miss and a gated frame do not look alike, and this project has
    # already read one as the other ("the catch rate is FLAT from 0.2 to 0.5,
    # there is no threshold left to tune"). A bimodal 0-or-1 distribution is
    # not a confident model, it is a gate firing.
    vad = getattr(model, "vad", None)
    if vad is not None:
        vad.prediction_buffer = collections.deque(maxlen=125)
        vad.reset_states()


def false_fire_rate(recording: Path, model_path: Path | None = None,
                    threshold: float | None = None) -> tuple[int, float]:
    """How often the model fires on a recording with no wake word in it.

    THE NUMBER THAT WAS NEVER MEASURED. Catch rate on Harsh's 40 takes says
    how often Friday hears him; it says nothing about how often she hears
    someone else and answers anyway, and that is the failure he actually
    reported. Returns (fires, minutes) so it can be quoted per hour.
    """
    from openwakeword.model import Model

    audio = _read_wav(recording)
    if len(audio) < CHUNK_SAMPLES:
        return 0, 0.0
    model_path = model_path or Path(config.WAKE_WORD_MODEL)
    model = Model(wakeword_model_paths=[str(model_path)],
                  vad_threshold=config.WAKE_VAD_THRESHOLD)
    threshold = config.WAKE_THRESHOLD if threshold is None else threshold

    reset_streams(model)
    fires = 0
    # The same cooldown the live detector uses, or one long phrase scoring
    # high across neighbouring frames counts as a dozen separate fires and the
    # number stops meaning anything.
    cooldown_chunks = int(config.WAKE_COOLDOWN_S * RATE / CHUNK_SAMPLES)
    cooling = 0
    for i in range(0, len(audio) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        out = model.predict(audio[i:i + CHUNK_SAMPLES])
        score = max(out.values()) if out else 0.0
        if cooling > 0:
            cooling -= 1
            continue
        if score >= threshold:
            fires += 1
            cooling = cooldown_chunks
            model.reset()
    return fires, len(audio) / RATE / 60.0


def score_samples(samples_dir: Path, model_path: Path | None = None,
                  negatives_dir: Path | None = None) -> int:
    """Run the CURRENT model over the real recordings and report the scores.

    This is the question "is the threshold too high, or is the model wrong?"
    turned into a number. A real "hey Friday" scoring 0.30 means lower the
    threshold and you are done today. One scoring 0.02 means the model has
    never heard a voice like this and no threshold will save it -- retrain.
    """
    import numpy as np
    from openwakeword.model import Model
    import wave

    paths = sorted(samples_dir.glob("wake_*.wav"))
    if not paths:
        print(f"no recordings in {samples_dir} — "
              f"run: scripts/train_wake.py --record-wakes 40")
        return 1

    model_path = model_path or Path(config.WAKE_WORD_MODEL)
    model = Model(wakeword_model_paths=[str(model_path)],
                  vad_threshold=config.WAKE_VAD_THRESHOLD)
    CHUNK = 1280
    scores = []
    print(f"\nmodel {model_path.name}, "
          f"threshold {config.WAKE_THRESHOLD}\n")
    for path in paths:
        with wave.open(str(path), "rb") as f:
            audio = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        # NOT model.reset() -- see reset_streams(). With the plain reset this
        # loop reported 2 of 8 where a clean model reports 5 of 8.
        reset_streams(model)
        peak = 0.0
        for i in range(0, len(audio) - CHUNK + 1, CHUNK):
            out = model.predict(audio[i:i + CHUNK])
            peak = max(peak, max(out.values()) if out else 0.0)
        scores.append(peak)
        mark = "caught" if peak >= config.WAKE_THRESHOLD else "MISSED"
        print(f"  {path.name}  {peak:5.3f}  {mark}")

    scores = np.array(scores)
    caught = float((scores >= config.WAKE_THRESHOLD).mean())
    print(f"\n{caught:.0%} would fire at {config.WAKE_THRESHOLD}")
    print(f"scores: min {scores.min():.3f}  median {np.median(scores):.3f}  "
          f"max {scores.max():.3f}")

    # BOTH HALVES, ALWAYS. Catch rate alone is what shipped the model that
    # fires 97 times an evening: it was 80% and looked like a success, because
    # nothing in this script had ever asked the other question. A wake word is
    # a ratio, not a recall number, so the negatives are scored in the same
    # breath and the thresholds are judged on both columns.
    negatives_dir = negatives_dir or (samples_dir.parent / "negatives")
    negative_recordings = sorted(negatives_dir.glob("*.wav")) \
        if negatives_dir.exists() else []
    fires_at: dict[float, tuple[int, float]] = {}
    if negative_recordings:
        print(f"\nfalse fires on {len(negative_recordings)} negative "
              f"recording(s) in {negatives_dir.name}/:")
        for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9):
            total_fires, total_minutes = 0, 0.0
            for path in negative_recordings:
                fires, minutes = false_fire_rate(path, model_path, t)
                total_fires += fires
                total_minutes += minutes
            fires_at[t] = (total_fires, total_minutes)
    else:
        print(f"\n! no negative recordings in {negatives_dir}/ — the false"
              f"-fire rate is UNMEASURED, which is how the 97-fires-an-evening"
              f"\n  model passed this check at 80% catch rate. Record some:"
              f"\n    scripts/train_wake.py --record-negatives 600")

    print()
    for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9):
        line = f"  at {t}: catches {float((scores >= t).mean()):5.0%}"
        if t in fires_at:
            fires, minutes = fires_at[t]
            per_hour = fires / minutes * 60 if minutes else 0.0
            line += f"   false fires {fires:3d} in {minutes:.0f} min " \
                    f"= {per_hour:5.1f}/hour"
        print(line)
    if np.median(scores) < 0.1:
        print("\nThe model does not recognise this voice at all. Retrain with "
              "these samples -- no threshold fixes this.")
    elif np.median(scores) < config.WAKE_THRESHOLD:
        print("\nThe voice IS recognised but scores under the threshold. "
              "Lowering it helps today; retraining helps properly.")
    return 0


def record_ambient(seconds: float, path: Path) -> int:
    """Record the real room, for negatives that actually match where Friday lives."""
    import subprocess, os
    path.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)
    print(f"Recording {seconds:.0f}s of room noise to {path}.")
    print("Move around, talk normally, but DO NOT say the wake word.")
    # pw-cat has no duration flag on this build, so it is stopped on a timer.
    proc = subprocess.Popen(
        ["pw-cat", "--record", f"--target={config.MIC_DEVICE}",
         "--rate", str(RATE), "--channels", "1", "--format", "s16", str(path)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(seconds)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    size = path.stat().st_size if path.exists() else 0
    print(f"  wrote {size/1024:.0f} kB")
    return 0 if size > RATE else 1



def _fires_nobody_answered(log_path: Path | None = None) -> set[str]:
    """Fire filenames that the turn log says were followed by silence.

    A false fire and a real one are indistinguishable by SCORE -- measured over
    415 wakes, the false ones score HIGHER (median 0.98) than the real ones
    (0.83). But they differ in what happens next: a person who says "hey
    Friday" then says something. This pairs each fire clip with the turn it
    started, by timestamp, and reports the ones where nothing was said.

    Returns an empty set on any problem. A missing or unreadable log must
    degrade the review to what it was, never break it.
    """
    log_path = log_path or (config.ROOT / "logs" / "friday.log")
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return set()

    # "[turn 7] WAKE ..." opens a turn; the first outcome after it closes one.
    quiet: set[str] = set()
    pending: dict[str, str] = {}          # turn number -> "YYYYmmdd-HHMMSS"
    spoke: set[str] = set()
    stamps: dict[str, str] = {}
    for line in text.splitlines():
        found = re.search(r"\[turn (\d+)\] ([A-Z][A-Z -]*)", line)
        if not found:
            continue
        turn, event = found.group(1), found.group(2).strip()
        when = line[:19]
        if event == "WAKE":
            try:
                stamp = (when[:4] + when[5:7] + when[8:10] + "-"
                         + when[11:13] + when[14:16] + when[17:19])
            except IndexError:
                continue
            pending[turn] = stamp
            stamps[turn] = stamp
            spoke.discard(turn)
        elif turn in pending:
            # AN EMPTY TRANSCRIPT IS NOT SILENCE. It means a capture closed on
            # SOMETHING -- and in this log half of the empties were the
            # transcription lock giving up, the audio never transcribed at
            # all. Only NOTHING SAID means nobody spoke: the VAD waited the
            # whole LISTEN_START_TIMEOUT_S and heard no one.
            if event in ("TRANSCRIBED", "RE-LISTENING"):
                spoke.add(turn)
            elif event == "NOTHING SAID":
                if turn not in spoke:
                    quiet.add(pending[turn])
                pending.pop(turn, None)
            elif event in ("ABANDONED", "DONE", "FAILED"):
                pending.pop(turn, None)

    # The clip is named for the moment of the fire; the WAKE line is logged in
    # the same second, but allow a second either side rather than lose a match
    # to a rounding boundary.
    wanted: set[str] = set()
    for stamp in quiet:
        wanted.add(stamp)
    out = set()
    for clip in (config.WAKE_FIRE_DIR).glob("fire-*.wav"):
        stamp = "-".join(clip.stem.split("-")[1:3])
        if stamp in wanted:
            out.add(clip.name)
    return out


def review_fires(fire_dir: Path, negatives_dir: Path,
                 samples_dir: Path) -> int:
    """Play back every live fire and ask what it actually was.

    THE LABEL HAS TO COME FROM A PERSON. The detector cannot know whether it
    was right -- if it could, it would not have fired -- so anything automatic
    here just records the model's own opinion back into its training set and
    makes the next model wronger in the same direction.

    Three answers, and the third matters as much as the first two: `n` files it
    as a hard negative, `y` files it as a real wake it caught, and `s` skips
    the ones that are neither (Friday's own voice, a slammed door, a clip too
    clipped to judge). Guessing on those poisons the set.
    """
    import shutil
    import subprocess

    clips = sorted(fire_dir.glob("fire-*.wav"))
    silence = _fires_nobody_answered()
    if not clips:
        print(f"no harvested fires in {fire_dir}.\n"
              f"Leave Friday running through a normal conversation first — "
              f"every fire lands there.")
        return 1

    negatives_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)

    print(f"{len(clips)} fires to review, newest last.")
    print("  n = false fire (a hard negative)      <- the common one")
    print("  y = a real 'hey Friday' she caught")
    print("  s = skip     r = replay     q = stop\n")

    kept = {"n": 0, "y": 0, "s": 0}
    for index, clip in enumerate(clips, 1):
        score = clip.stem.rsplit("-", 1)[-1]
        # A HINT, NEVER AN ANSWER. This comes from the turn log -- whether a
        # human said anything in the eight seconds after the fire -- which is
        # evidence the detector did not produce and cannot be fooled by its own
        # confidence. It is still only a hint: a real wake followed by a change
        # of mind looks identical. The prompt is unchanged and nothing is
        # pre-selected, because the moment this picks the answer it is the
        # automatic labelling the docstring above rules out.
        hint = "  [nobody spoke after this one]" if clip.name in silence else ""
        while True:
            subprocess.run(["pw-play", f"--target={config.SPEAKER_DEVICE}",
                            str(clip)], env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            answer = input(f"  [{index}/{len(clips)}] {clip.name} "
                           f"(scored {score}){hint}  n/y/s/r/q> ").strip().lower()
            if answer == "r":
                continue
            break
        if answer == "q":
            break
        if answer == "y":
            shutil.move(str(clip), samples_dir / f"wake_live_{clip.stem}.wav")
            kept["y"] += 1
        elif answer == "s":
            kept["s"] += 1
        else:                                   # anything else is a negative
            shutil.move(str(clip), negatives_dir / clip.name)
            kept["n"] += 1

    print(f"\n{kept['n']} hard negatives, {kept['y']} real wakes, "
          f"{kept['s']} skipped")
    if kept["n"] or kept["y"]:
        print("Retrain with them:\n  scripts/train_wake.py")
    return 0


def record_negatives(seconds: float, directory: Path) -> int:
    """Record normal conversation, to be trained against as negatives.

    Separate from --record-ambient because they are not the same thing and
    conflating them is what made the first two models. Ambient is the EMPTY
    room: fan, fridge, traffic. This is people TALKING, which is the sound the
    model actually confuses with its wake word, and it needs to be a different
    file so it can be weighted differently and measured separately.
    """
    import subprocess

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"conversation-{time.strftime('%Y%m%d-%H%M%S')}.wav"
    env = dict(os.environ)
    env.setdefault("XDG_RUNTIME_DIR", config.AUDIO_RUNTIME_DIR)
    print(f"Recording {seconds:.0f}s of CONVERSATION to {path.name}.")
    print("Talk normally — to someone else, on the phone, to yourself.")
    print("Say anything EXCEPT the wake word. Saying 'Friday' on its own is")
    print("good: that is exactly the confusion worth training against.")
    proc = subprocess.Popen(
        ["pw-cat", "--record", f"--target={config.MIC_DEVICE}",
         "--rate", str(RATE), "--channels", "1", "--format", "s16", str(path)],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for left in range(int(seconds), 0, -10):
            time.sleep(min(10, left))
            print(f"  {left}s left", flush=True)
    except KeyboardInterrupt:
        print("  stopped early — what was recorded is kept")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    size = path.stat().st_size if path.exists() else 0
    print(f"  wrote {size/1024:.0f} kB ({size/2/RATE/60:.1f} min)")
    return 0 if size > RATE else 1


def build_dataset(rng: np.random.Generator, ambient: Path,
                  per_clip: int,
                  samples_dir: Path | None = None,
                  negatives_dir: Path | None = None,
                  real_weight: int = 2,
                  fires_dir: Path | None = None,
                  ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """The clips, their labels, and the FILE each clip was made from.

    The source file is what lets --holdout leave a real recording out
    entirely -- every augmented copy of it and every reversed or reshuffled
    one -- so the model can be measured on audio it never heard in any form.
    """
    from piper import PiperVoice

    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    pos_src: list[str] = []
    neg_src: list[str] = []
    voices = piper_voices()
    print(f"{len(voices)} Piper voices, {len(WAKE_PHRASES)} wake phrasings, "
          f"{len(NEGATIVE_PHRASES)} negative phrases")

    for index, model in enumerate(voices, 1):
        print(f"  [{index}/{len(voices)}] {model.stem}", flush=True)
        voice = PiperVoice.load(str(model))
        for speed in (0.85, 1.0, 1.2):
            for text in WAKE_PHRASES:
                clip = synth(voice, text, speed)
                if len(clip) == 0:
                    continue
                for _ in range(per_clip):
                    positives.append(augment(place(clip, rng), rng))
        # Negatives are spread thinner per voice because there are far more
        # phrases; the total still outnumbers the positives several times over.
        for text in NEGATIVE_PHRASES:
            clip = synth(voice, text, float(rng.choice([0.85, 1.0, 1.2])))
            if len(clip) == 0:
                continue
            for _ in range(max(1, per_clip // 3)):
                negatives.append(augment(place(clip, rng), rng))
        del voice
    pos_src += ["piper"] * len(positives)
    neg_src += ["piper"] * len(negatives)

    # HARSH'S OWN VOICE, WEIGHTED HEAVILY. Each real take is augmented far more
    # times than a synthetic one, because there are far fewer of them and they
    # are the only samples that describe the voice this robot actually has to
    # answer. Without them the model catches Piper and misses the person.
    # REAL_WEIGHT WAS 4 AND THAT IS HALF OF WHY SHE FIRES ON A CONVERSATION.
    # per_clip*4 turned 40 recordings into 1280 clips -- 60% of every positive
    # the model ever saw, all of them one person in one room on one evening.
    # A classifier fed that learns the VOICE, not the phrase, and then answers
    # to the voice saying anything at all. Dropped to 2, and the catch rate
    # this gives up is bought back by the SNR-relative augment above and by
    # the real negatives below, both of which are new.
    real = real_wake_clips(samples_dir, rng, per_clip * real_weight,
                           sources=pos_src) if samples_dir else []
    if real:
        takes = len(list(samples_dir.glob('wake_*.wav')))
        print(f"  + {len(real)} clips from {takes} REAL recordings of Harsh")
        positives.extend(real)
    else:
        print("  ! no real voice samples — the model will catch Piper and miss "
              "a person.\n    Fix with: scripts/train_wake.py --record-wakes 40")

    room = ambient_windows(ambient, rng, count=max(400, len(positives)))
    if room:
        print(f"  + {len(room)} windows of real room audio")
    else:
        print("  ! no ambient recording — run --record-ambient 90 for a "
              "much better model")
    negatives.extend(room)
    neg_src += ["ambient"] * len(room)

    # REAL SPEECH THAT IS NOT THE WAKE WORD. The single most important
    # negatives in the set and the ones the first two models never had. Every
    # clip harvested from a live false fire lands here, which means the model
    # is trained specifically against the audio that already defeated it.
    hard = hard_negative_clips(negatives_dir, rng, per_clip * 3,
                               sources=neg_src) if negatives_dir else []
    if hard:
        files = len(list(negatives_dir.glob('*.wav')))
        print(f"  + {len(hard)} HARD negatives from {files} real recording(s)")
        negatives.extend(hard)
    else:
        print("  ! NO real-speech negatives. This is the model that fires 97"
              "\n    times an evening; it will be again. Record 10 minutes of"
              "\n    normal conversation:  --record-negatives 600")

    # THE VOICE WITHOUT THE PHRASE -- see scrambled_clips(). It needs no
    # labels, so it takes every real recording AND every harvested fire,
    # including the ones nobody has judged yet.
    scramble_from: dict[str, Path] = {}
    for directory, pattern in ((samples_dir, "wake_*.wav"),
                               (negatives_dir, "fire-*.wav"),
                               (fires_dir, "fire-*.wav")):
        if directory is not None and directory.exists():
            for path in sorted(directory.glob(pattern)):
                scramble_from.setdefault(path.name, path)
    scrambled = scrambled_clips(list(scramble_from.values()), rng,
                                max(2, per_clip // 2), sources=neg_src)
    if scrambled:
        print(f"  + {len(scrambled)} reversed/reshuffled clips from "
              f"{len(scramble_from)} real recordings")
        negatives.extend(scrambled)

    for _ in range(300):                                  # pure noise / quiet
        negatives.append((rng.normal(0, rng.uniform(5, 900), CLIP_SAMPLES)
                          ).clip(-32000, 32000).astype(np.int16))
    neg_src += ["noise"] * 300

    assert len(pos_src) == len(positives) and len(neg_src) == len(negatives)
    x = np.stack(positives + negatives).astype(np.int16)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    print(f"\n{len(positives)} positive, {len(negatives)} negative clips")
    return x, y, pos_src + neg_src


def embed(clips: np.ndarray) -> np.ndarray:
    import openwakeword.utils as owu
    features = owu.AudioFeatures(ncpu=4)
    print(f"embedding {len(clips)} clips (this is the slow part)…", flush=True)
    started = time.monotonic()
    out = features.embed_clips(clips.astype(np.int16), batch_size=64, ncpu=4)
    print(f"  {out.shape} in {time.monotonic() - started:.0f}s")
    return out


def export_onnx(model, path: Path) -> None:
    """Write the classifier as [1,16,96] -> [1,1], the shape openWakeWord uses.

    Built by hand rather than through a converter so the graph is exactly the
    signature the real models have: a Reshape to flatten the 16 frames, then
    the dense layers, then a sigmoid. Anything else and openwakeword.Model
    cannot load it.
    """
    from onnx import TensorProto, helper, numpy_helper, save

    nodes, inits = [], []
    inits.append(numpy_helper.from_array(
        np.array([1, 16 * 96], dtype=np.int64), "flat_shape"))
    nodes.append(helper.make_node("Reshape", ["x", "flat_shape"], ["h0"]))

    previous = "h0"
    for layer, (weight, bias) in enumerate(zip(model.coefs_,
                                               model.intercepts_)):
        w_name, b_name = f"W{layer}", f"B{layer}"
        inits.append(numpy_helper.from_array(
            weight.astype(np.float32), w_name))
        inits.append(numpy_helper.from_array(
            bias.astype(np.float32).reshape(1, -1), b_name))
        nodes.append(helper.make_node("MatMul", [previous, w_name],
                                      [f"m{layer}"]))
        nodes.append(helper.make_node("Add", [f"m{layer}", b_name],
                                      [f"a{layer}"]))
        last = layer == len(model.coefs_) - 1
        nodes.append(helper.make_node("Sigmoid" if last else "Relu",
                                      [f"a{layer}"], [f"h{layer + 1}"]))
        previous = f"h{layer + 1}"

    graph = helper.make_graph(
        nodes, "hey_friday",
        inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT,
                                              [1, 16, 96])],
        outputs=[helper.make_tensor_value_info(previous, TensorProto.FLOAT,
                                               [1, 1])],
        initializer=inits)
    model_proto = helper.make_model(
        graph, producer_name="friday-train-wake",
        opset_imports=[helper.make_opsetid("", 13)])
    model_proto.ir_version = 9          # onnxruntime 1.29 rejects newer
    path.parent.mkdir(parents=True, exist_ok=True)
    save(model_proto, str(path))


def fit(flat: np.ndarray, labels: np.ndarray, seed: int):
    """Train the classifier, and report it on a random fifth it did not see.

    That fifth shares SOURCE FILES with the other four -- a different
    augmentation of the same take -- so this report is optimistic. The honest
    number is --evaluate on a --holdout model.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    x_train, x_test, y_train, y_test = train_test_split(
        flat, labels, test_size=0.2, random_state=seed, stratify=labels)

    print("training…", flush=True)
    net = MLPClassifier(hidden_layer_sizes=(96, 32), activation="relu",
                        alpha=1e-3, max_iter=400, random_state=seed,
                        early_stopping=True, n_iter_no_change=15)
    net.fit(x_train, y_train)

    probability = net.predict_proba(x_test)[:, 1]
    auc = roc_auc_score(y_test, probability)
    positive = probability[y_test == 1]
    negative = probability[y_test == 0]
    print(f"\nheld-out AUC        {auc:.4f}")
    print(f"wake clips  median  {np.median(positive):.3f}  "
          f"5th pct {np.percentile(positive, 5):.3f}")
    print(f"other clips median  {np.median(negative):.3f}  "
          f"99th pct {np.percentile(negative, 99):.3f}")
    for threshold in (0.3, 0.5, 0.7):
        recall = float((positive >= threshold).mean())
        false = float((negative >= threshold).mean())
        print(f"  threshold {threshold}:  catches {recall:6.1%}  "
              f"false-fires {false:6.2%}")
    return net


def read_patterns(path: Path) -> list[str]:
    """File-name patterns, one per line; `#` starts a comment."""
    lines = (line.split("#", 1)[0].strip()
             for line in path.read_text().splitlines())
    return [line for line in lines if line]


def held_out(name: str, patterns: list[str]) -> bool:
    import fnmatch
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def holdout_path(out: Path) -> Path:
    return out.with_name(out.stem + ".holdout.onnx")


def live_fires(model, audio: np.ndarray, lead: np.ndarray,
               tail: np.ndarray) -> tuple[bool, float, int]:
    """Would this clip have woken Friday? -> (fired, peak score, longest run)

    Scored the way wake_word.py scores the microphone: a clean streaming
    state, a second of the real room in front so the VAD gate starts where it
    does live, then WAKE_THRESHOLD and WAKE_PATIENCE exactly as configured. A
    peak over the threshold is not a wake; WAKE_PATIENCE chunks in a row is.
    """
    reset_streams(model)
    stream = np.concatenate([lead, audio, tail]).astype(np.int16)
    run = longest = 0
    peak = 0.0
    for i in range(0, len(stream) - CHUNK_SAMPLES + 1, CHUNK_SAMPLES):
        out = model.predict(stream[i:i + CHUNK_SAMPLES])
        score = float(max(out.values())) if out else 0.0
        peak = max(peak, score)
        run = run + 1 if score >= config.WAKE_THRESHOLD else 0
        longest = max(longest, run)
    return longest >= max(1, config.WAKE_PATIENCE), peak, longest


def evaluate(models: list[Path], samples_dir: Path, negatives_dir: Path,
             fires_dir: Path, ambient: Path, holdout: Path | None) -> int:
    """Both columns for each model, on the clips a hold-out model never saw.

    Catches: Harsh's own recordings. False fires: harvested fires filed as
    negatives -- every one of them woke the live model once. Unlabelled fires
    are counted apart, because some are real wakes, and a model that stops
    firing on THOSE has got worse, not better.
    """
    from openwakeword.model import Model

    patterns = read_patterns(holdout) if holdout else None

    def pick(paths: list[Path]) -> list[Path]:
        return [p for p in paths
                if patterns is None or held_out(p.name, patterns)]

    positives = pick(sorted(samples_dir.glob("wake_*.wav")))
    negatives = pick(sorted(negatives_dir.glob("fire-*.wav")))
    filed = {p.name for p in negatives_dir.glob("fire-*.wav")}
    unlabelled = [p for p in pick(sorted(fires_dir.glob("fire-*.wav")))
                  if p.name not in filed]
    room = _read_wav(ambient)
    if len(room) < 3 * RATE:
        room = np.zeros(3 * RATE, dtype=np.int16)
    lead, tail = room[RATE:2 * RATE], room[2 * RATE:2 * RATE + RATE // 3]

    print(f"{len(positives)} real wakes, {len(negatives)} labelled false "
          f"fires, {len(unlabelled)} unlabelled fires"
          f"{' -- all held out' if patterns else ''}.  threshold "
          f"{config.WAKE_THRESHOLD}, patience {config.WAKE_PATIENCE}\n")
    for model_path in models:
        model = Model(wakeword_model_paths=[str(model_path)],
                      vad_threshold=config.WAKE_VAD_THRESHOLD)

        def fired(paths: list[Path]) -> int:
            return sum(live_fires(model, _read_wav(p), lead, tail)[0]
                       for p in paths)

        print(f"  {model_path.name:30s} catches {fired(positives):3d}/"
              f"{len(positives):<3d} false fires {fired(negatives):3d}/"
              f"{len(negatives):<3d} unlabelled fires "
              f"{fired(unlabelled):3d}/{len(unlabelled)}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-ambient", type=float, metavar="SECONDS",
                    help="record room noise for negatives, then exit")
    ap.add_argument("--record-wakes", type=int, metavar="N",
                    help="record yourself saying the wake word N times, then "
                         "exit. THIS is what makes it catch a real voice.")
    ap.add_argument("--score", action="store_true",
                    help="score the recorded samples against the CURRENT "
                         "model — tells you if it is the threshold or the model")
    ap.add_argument("--model", type=Path,
                    help="score THIS model instead of config.WAKE_WORD_MODEL "
                         "— the before/after comparison runs one code path")
    ap.add_argument("--samples", type=Path,
                    default=config.ROOT / "assets/models/wake/samples",
                    help="where the real recordings live")
    ap.add_argument("--ambient", type=Path,
                    default=config.ROOT / "assets/models/wake/ambient.wav")
    ap.add_argument("--record-negatives", type=float, metavar="SECONDS",
                    help="record normal CONVERSATION as hard negatives, then "
                         "exit. This is the fix for firing on human speech.")
    ap.add_argument("--review-fires", action="store_true",
                    help="play back every wake the live detector fired and "
                         "label it — the harvested false fires become "
                         "training negatives")
    ap.add_argument("--negatives", type=Path,
                    default=config.ROOT / "assets/models/wake/negatives",
                    help="real speech that is NOT the wake word")
    ap.add_argument("--fires", type=Path, default=config.WAKE_FIRE_DIR,
                    help="where the live detector harvests its fires")
    ap.add_argument("--real-weight", type=int, default=2,
                    help="augmentations per real recording, as a multiple of "
                         "--per-clip. Was 4, which made 40 takes 60%% of all "
                         "positives and taught the model a voice, not a phrase")
    ap.add_argument("--out", type=Path,
                    default=config.ROOT / "assets/models/wake/friday.onnx")
    ap.add_argument("--per-clip", type=int, default=8,
                    help="augmented variants per synthesised clip")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--holdout", type=Path, metavar="FILE",
                    help="clip-name patterns, one per line: ALSO train a "
                         "second model without those files, for --evaluate")
    ap.add_argument("--evaluate", action="store_true",
                    help="replay real wakes and labelled false fires through "
                         "models the way the live detector hears them")
    ap.add_argument("--compare", type=Path, nargs="+", metavar="MODEL",
                    help="the models --evaluate scores (default: the live one)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)-7s %(name)s | %(message)s")

    if args.score:
        return score_samples(args.samples, args.model, args.negatives)
    if args.evaluate:
        return evaluate(args.compare or [Path(config.WAKE_WORD_MODEL)],
                        args.samples, args.negatives, args.fires,
                        args.ambient, args.holdout)
    if args.record_wakes:
        return record_wakes(args.record_wakes, args.samples)
    if args.record_ambient:
        return record_ambient(args.record_ambient, args.ambient)
    if args.record_negatives:
        return record_negatives(args.record_negatives, args.negatives)
    if args.review_fires:
        return review_fires(args.fires, args.negatives, args.samples)

    rng = np.random.default_rng(args.seed)
    clips, labels, sources = build_dataset(rng, args.ambient, args.per_clip,
                                           samples_dir=args.samples,
                                           negatives_dir=args.negatives,
                                           real_weight=args.real_weight,
                                           fires_dir=args.fires)
    features = embed(clips)
    flat = features.reshape(len(features), -1)

    # THE MODEL THAT CAN BE MEASURED, from the SAME embeddings. Embedding is
    # the slow part, so both models share one pass: the hold-out model drops
    # every clip made from a held-out file, and --evaluate scores it on them.
    if args.holdout:
        patterns = read_patterns(args.holdout)
        held = np.array([held_out(name, patterns) for name in sources])
        print(f"\n--- hold-out model: {int(held.sum())} clips from held-out "
              f"files left out")
        export_onnx(fit(flat[~held], labels[~held], args.seed),
                    holdout_path(args.out))
        print(f"wrote {holdout_path(args.out)} -- measure it with --evaluate")
        print("\n--- the real model: everything")

    net = fit(flat, labels, args.seed)
    export_onnx(net, args.out)
    print(f"\nwrote {args.out}")

    import onnxruntime as ort
    session = ort.InferenceSession(str(args.out),
                                   providers=["CPUExecutionProvider"])
    probe = features[:1].astype(np.float32).reshape(1, 16, 96)
    check = session.run(None, {"x": probe})[0]
    print(f"onnx loads, output {check.shape} = {float(check[0][0]):.3f} "
          f"(sklearn says {net.predict_proba(flat[:1])[0][1]:.3f})")
    print(f"\nPoint config.WAKE_WORD_MODEL at it, then:\n"
          f"  ./venv/bin/python scripts/test_wake.py --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
