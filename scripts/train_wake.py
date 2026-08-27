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
        out += rng.normal(0, float(rng.uniform(30, 400)), CLIP_SAMPLES)
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


def build_dataset(rng: np.random.Generator, ambient: Path,
                  per_clip: int) -> tuple[np.ndarray, np.ndarray]:
    from piper import PiperVoice

    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
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

    room = ambient_windows(ambient, rng, count=max(400, len(positives)))
    if room:
        print(f"  + {len(room)} windows of real room audio")
    else:
        print("  ! no ambient recording — run --record-ambient 90 for a "
              "much better model")
    negatives.extend(room)

    for _ in range(300):                                  # pure noise / quiet
        negatives.append((rng.normal(0, rng.uniform(5, 900), CLIP_SAMPLES)
                          ).clip(-32000, 32000).astype(np.int16))

    x = np.stack(positives + negatives).astype(np.int16)
    y = np.concatenate([np.ones(len(positives)), np.zeros(len(negatives))])
    print(f"\n{len(positives)} positive, {len(negatives)} negative clips")
    return x, y


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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record-ambient", type=float, metavar="SECONDS",
                    help="record room noise for negatives, then exit")
    ap.add_argument("--ambient", type=Path,
                    default=config.ROOT / "assets/models/wake/ambient.wav")
    ap.add_argument("--out", type=Path,
                    default=config.ROOT / "assets/models/wake/friday.onnx")
    ap.add_argument("--per-clip", type=int, default=8,
                    help="augmented variants per synthesised clip")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING,
                        format="%(levelname)-7s %(name)s | %(message)s")

    if args.record_ambient:
        return record_ambient(args.record_ambient, args.ambient)

    rng = np.random.default_rng(args.seed)
    clips, labels = build_dataset(rng, args.ambient, args.per_clip)
    features = embed(clips)

    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score

    flat = features.reshape(len(features), -1)
    x_train, x_test, y_train, y_test = train_test_split(
        flat, labels, test_size=0.2, random_state=args.seed, stratify=labels)

    print("training…", flush=True)
    net = MLPClassifier(hidden_layer_sizes=(96, 32), activation="relu",
                        alpha=1e-3, max_iter=400, random_state=args.seed,
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
