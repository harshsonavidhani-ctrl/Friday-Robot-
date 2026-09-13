#!/usr/bin/env python3
"""Train the wake VERIFIER -- the second opinion that stops conversations waking Friday.

    ./venv/bin/python scripts/train_verifier.py --vosk FILE      # features, fit, report
    ./venv/bin/python scripts/train_verifier.py --vosk FILE --write --C 0.001

WHY THIS EXISTS. openWakeWord scores a conversation as high as "hey Friday":
over 415 turns the false fires scored a median 0.98, the real ones 0.83, so
there is no WAKE_THRESHOLD and no WAKE_PATIENCE that separates them. Its
classifier never saw a real conversation while training -- only Piper phrases
and an empty room -- so it learned "a voice in this room".

The features underneath it are fine; they are Google's general speech
embedding, and they do carry the words. So the verifier is a second
classifier on THE SAME 16x96 features, trained on exactly the audio that
fooled the first one: the harvested live fires. It is asked only when
openWakeWord has already fired, so it can only ever REJECT -- a wake that
openWakeWord missed stays missed. It costs one dot product, because the
features are already sitting in openWakeWord's buffer at the moment it fires.

WHY NOT A SPEECH RECOGNISER. Measured 2026-09-12 on Harsh's 40 recordings:
sherpa-onnx's keyword spotter caught 15/40 at its loosest setting, and Vosk's
Indian-English model 25/40 at 3.6 s a clip. Both rejected every conversation,
and both would have ignored Harsh most of the time.

THE FIRST VERIFIER LEARNED THE WRONG THING, AND THE FIX IS IN THE DATA. Its
positives were all "silence, then hey Friday" -- clean takes -- and its
negatives were all speech from end to end. So it partly learned "was somebody
talking before the phrase": Harsh's takes scored 0.70-0.96 while four LIVE
wakes that Vosk independently heard "friday" in scored 0.22-0.51, and a
judge saying "...this is our robot. Hey Friday!" is exactly that shape. Two
things answer it:
  * every positive is ALSO trained with conversation in front of it
    (--after-speech), so only the phrase can decide;
  * fires in which Vosk heard "friday" are REAL LIVE WAKES -- positives, and
    pulled out of the negatives if they were filed there. Vosk hears "friday"
    in 25 of Harsh's 40 takes and in 0 of 50 clips of a conversation, so its
    "friday" is strong evidence that did not come from either wake model; its
    silence proves nothing and is not used.

HOW IT IS MEASURED. Every clip is streamed through openWakeWord exactly the way
wake_word.py streams the microphone -- clean state, a second of the real room
in front, 80 ms chunks, WAKE_THRESHOLD and WAKE_PATIENCE -- and the verifier is
asked at every chunk where the live detector would have fired. Nothing is
scored on a clip it was trained on: the hold-out takes and evenings of
holdout.txt, one whole Piper voice, and the most recent conversation are kept
out of the fit entirely, and so are the conversations used as prefixes.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                            # noqa: E402

RATE = 16000
CHUNK = 1280
WAKE_DIR = config.ROOT / "assets/models/wake"
CACHE = config.ROOT / "state" / "verifier_features.npz"
AUG_DIR = config.ROOT / "state" / "verifier_aug"
KEEP_SCORE = 0.2          # features are kept for chunks scoring at least this
FRIDAY = re.compile(r"\bfri ?day\b")
# Fires from here on are this room, this hum, and the verifier switched off --
# see "recent false fires" in clip_table.
RECENT_FALSE_SINCE = "20260913-064000"


def _turns_that_acted(log_path: Path | None = None) -> set[str]:
    """Fire stamps (YYYYMMDD-HHMMSS) whose turn ran a tool or got an answer.

    Read from the turn log, which records what HAPPENED after a wake -- a move,
    a stop, a brain reply -- not what the wake model thought. The clip is saved
    just after the wake is logged, so each wake second and the two after it
    are all marked.
    """
    import datetime as dt
    path = log_path or config.ROOT / "logs" / "friday.log"
    if not path.exists():
        return set()
    stamp = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ ")
    acted: set[str] = set()
    woke = None
    for line in path.open(encoding="utf-8", errors="replace"):
        m = stamp.match(line)
        if not m:
            continue
        if "wake word detected" in line:
            woke = dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        elif woke and ("] TOOL " in line or "answered by" in line):
            for s in range(3):
                acted.add((woke + dt.timedelta(seconds=s)).strftime("%Y%m%d-%H%M%S"))
            woke = None
    return acted


# --- the clips ----------------------------------------------------------------

def read_vosk(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    heard = {}
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            heard[parts[0]] = parts[2]
    return heard


def clip_table(synth_dir: Path | None, test_since: str, vosk: dict[str, str],
               cached: "set[str] | None" = None) -> list[tuple[str, str, int, str]]:
    """(path, group, label, split) for every clip. split: train | test."""
    import train_wake as tw

    holdout = tw.read_patterns(WAKE_DIR / "holdout.txt")
    fires = Path(config.WAKE_FIRE_DIR)
    rows: list[tuple[str, str, int, str]] = []

    def split_for(name: str) -> str:
        return "test" if any(fnmatch.fnmatch(name, p) for p in holdout) else "train"

    def heard_friday(name: str) -> bool:
        return bool(FRIDAY.search(vosk.get(name, "")))

    for p in sorted((WAKE_DIR / "samples").glob("wake_*.wav")):
        rows.append((str(p), "harsh takes", 1, split_for(p.name)))

    # HIS VOICE FROM WHERE HE ACTUALLY STANDS, recorded 2026-09-13 in the
    # humming room (train_wake.py --record-wakes N --samples .../samples_far).
    # The verifier was switched off that morning because it had only heard him
    # close up: live, it rejected ~57 of his attempts at 0.01-0.15. Every third
    # take is held out so the gate below can ask about them honestly.
    for n, p in enumerate(sorted((WAKE_DIR / "samples_far").glob("wake_*.wav"))):
        rows.append((str(p), "harsh far takes", 1,
                     "test" if n % 3 == 2 else "train"))

    # Fires that a real command followed. Very probably a real "hey Friday",
    # but nobody has listened, so they are scored and never trained on.
    command_followed = set()
    for line in (WAKE_DIR / "labels.tsv").read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3 and parts[1] == "unsure" and parts[2] == "a command followed":
            for d in (WAKE_DIR / "unsure", fires):
                if (d / parts[0]).exists():
                    rows.append((str(d / parts[0]), "fire then command", 1, "test"))
                    command_followed.add(parts[0])
                    break

    filed = set()
    for p in sorted((WAKE_DIR / "negatives").glob("fire-*.wav")):
        filed.add(p.name)
        if heard_friday(p.name):
            rows.append((str(p), "live wakes (vosk)", 1, split_for(p.name)))
        else:
            rows.append((str(p), "labelled false fires", 0, split_for(p.name)))

    # LIVE WAKES from the unlabelled piles: only where a recogniser heard
    # "friday". The REJECTED pile is included too -- a verifier that turned a
    # real "hey Friday" away is precisely the clip it most needs to train on.
    # ⚠ BUT ONLY IF THE CLIP IS CONTIGUOUS. Until wake_word.resume() cleared
    # its clip ring, a fire just after a turn saved the PREVIOUS turn's wake
    # phrase glued onto <1 s of new audio. "Hey Friday, good morning." (fire
    # 20260913-003611, rejected at 0.09) was exactly that -- 0.7 s fresh --
    # and so were 8 of the 11 rejected name-clips that night. Clips saved
    # before that fix are not trustworthy positives.
    #
    # ⚠ NO DATE CUTOFF HERE ANY MORE, and that matters. test_since used to
    # exclude recent clips from this pool while the loop below filed every
    # recent fire as conversation -- so the newest real wakes, the ones from
    # the conditions that are actually failing, were being trained on as
    # NEGATIVES. Holdout is still honoured through split_for().
    for d in (WAKE_DIR / "unsure", fires, Path(config.WAKE_REJECT_DIR)):
        if not d.exists():
            continue
        for p in sorted(d.glob("fire-*.wav")):
            if (p.name in filed or p.name in command_followed
                    or not heard_friday(p.name)):
                continue
            filed.add(p.name)
            rows.append((str(p), "live wakes (vosk)", 1, split_for(p.name)))

    # THE VIDEO AND CONVERSATION THAT WOKE HER ON 2026-09-13, TRAINED ON. With
    # the verifier off, 40 of 49 wakes after 06:40 were a video or talk
    # ("Thanks for watching.", "please subscribe to my channel"). These clips
    # are contiguous (after the resume() fix) and from this room's hum, which
    # no older negative has. A fire is filed as false only if the turn it
    # started did NOTHING -- no tool ran and no brain answered, per the turn
    # log -- an outcome the wake model never saw. Half train, half test.
    acted = _turns_that_acted()
    recent = [p for p in sorted(fires.glob("fire-*.wav"))
              if "-".join(p.stem.split("-")[1:3]) >= RECENT_FALSE_SINCE
              and p.name not in filed and p.name[5:20] not in acted]
    for n, p in enumerate(recent):
        filed.add(p.name)
        rows.append((str(p), "recent false fires", 0,
                     "test" if n % 2 else "train"))

    # THE MOST RECENT CONVERSATION, NEVER TRAINED ON. The question that
    # matters at a hackathon is how it does on talk it has never heard, and
    # this is the only honest way to ask it.
    for p in sorted(fires.glob("fire-*.wav")):
        stamp = "-".join(p.stem.split("-")[1:3])
        # `filed` is checked so a clip a recogniser heard her name in is never
        # ALSO counted as conversation -- it would be a positive and a negative
        # at once, and the negative is simply wrong.
        if stamp >= test_since and p.name not in filed:
            rows.append((str(p), "unseen conversation", 0, "test"))

    # Other voices -- a stand-in for the judges. One whole voice is held out,
    # so the report says how it does on a speaker it has never heard.
    def synth_paths(sub: str) -> list[str]:
        if synth_dir and (synth_dir / sub).exists():
            return sorted(str(p) for p in (synth_dir / sub).glob("*.wav"))
        # THE PIPER CLIPS LIVED IN /tmp and a reboot took them. Their
        # features are in the cache, and features are all the fit needs.
        return sorted(p for p in (cached or ()) if f"/synth/{sub}/" in p)

    voices = sorted({Path(p).stem.rsplit("-", 2)[0] for p in synth_paths("pos")})
    test_voice = voices[-1] if voices else ""
    for sub, group, label in (("pos", "other voices", 1),
                              ("pos_aug", "other voices far/noisy", 1),
                              ("neg", "near-miss phrases", 0)):
        for p in synth_paths(sub):
            split = "test" if Path(p).stem.startswith(test_voice) else "train"
            rows.append((p, group, label, split))
    return rows


def _read(path: str) -> np.ndarray:
    try:
        with wave.open(path, "rb") as f:
            return np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    except Exception:                                       # noqa: BLE001
        return np.zeros(0, np.int16)


def after_speech(rows, copies: int, seed: int = 5) -> list[tuple[str, str, int, str]]:
    """Every positive again, with somebody talking in front of it.

    The prefix is the START of a conversation clip -- 0.8 to 1.8 s of it,
    the part that did not fool anything -- then a gap of 0 to 300 ms, then the
    phrase, levelled to within a few dB of each other. Prefixes come ONLY from
    TRAIN-split false fires, so no held-out conversation leaks into the fit.
    """
    rng = np.random.default_rng(seed)
    talk = [r[0] for r in rows if r[1] == "labelled false fires" and r[3] == "train"]
    talk = [a for a in (_read(p) for p in talk) if len(a) >= RATE * 2]
    if not talk:
        return []
    out = []
    groups = {"harsh takes": "harsh takes after speech",
              "other voices": "other voices after speech",
              "live wakes (vosk)": None, "fire then command": None}
    for path, group, label, split in rows:
        new_group = groups.get(group)
        if label != 1 or not new_group:
            continue
        phrase = None
        for n in range(copies):
            target = AUG_DIR / new_group.replace(" ", "_") / f"{Path(path).stem}-{n}.wav"
            # ALREADY MADE, KEEP IT -- even if its source is gone. The Piper
            # clips lived in /tmp and a reboot took them; what was generated
            # from them is on the SD card and is exactly as good.
            if not target.exists():
                if phrase is None:
                    phrase = _read(path).astype(np.float32)[-2 * RATE:]
                if len(phrase) == 0:
                    break
                level = float(np.sqrt((phrase ** 2).mean())) or 1.0
                prefix = talk[int(rng.integers(len(talk)))]
                prefix = prefix[:int(rng.uniform(0.8, 1.8) * RATE)].astype(np.float32)
                p_level = float(np.sqrt((prefix ** 2).mean())) or 1.0
                prefix *= level / p_level * float(10 ** (rng.uniform(-6, 3) / 20))
                gap = rng.normal(0, 30, int(rng.uniform(0.0, 0.3) * RATE))
                audio = np.clip(np.concatenate([prefix, gap, phrase]), -32000, 32000)
                target.parent.mkdir(parents=True, exist_ok=True)
                with wave.open(str(target), "wb") as f:
                    f.setnchannels(1)
                    f.setsampwidth(2)
                    f.setframerate(RATE)
                    f.writeframes(audio.astype(np.int16).tobytes())
            out.append((str(target), new_group, 1, split))
    return out


# --- streaming, exactly as live -----------------------------------------------

_MODEL = None
_LEAD = None


def _init(ambient: str) -> None:
    global _MODEL, _LEAD
    from openwakeword.model import Model
    import train_wake as tw
    _MODEL = Model(wakeword_model_paths=[str(config.WAKE_WORD_MODEL)],
                   vad_threshold=config.WAKE_VAD_THRESHOLD)
    room = tw._read_wav(Path(ambient))
    _LEAD = room[:RATE] if len(room) >= RATE else np.zeros(RATE, np.int16)


def stream_clip(path: str):
    """-> (scores per chunk, {chunk index: 16x96 features})"""
    import train_wake as tw
    audio = tw._read_wav(Path(path))
    if len(audio) < CHUNK:
        return np.zeros(0, np.float32), {}
    tw.reset_streams(_MODEL)
    tail = np.zeros(RATE // 4, np.int16)
    stream = np.concatenate([_LEAD, audio, tail]).astype(np.int16)
    first = len(_LEAD) // CHUNK
    scores, feats = [], {}
    for n, i in enumerate(range(0, len(stream) - CHUNK + 1, CHUNK)):
        out = _MODEL.predict(stream[i:i + CHUNK])
        score = float(max(out.values())) if out else 0.0
        if n < first:
            continue
        scores.append(score)
        if score >= KEEP_SCORE:
            f = np.asarray(_MODEL.preprocessor.get_features(16), np.float32)
            feats[len(scores) - 1] = f.reshape(16 * 96)
    return np.array(scores, np.float32), feats


def load_cache() -> dict[str, tuple[np.ndarray, dict[int, np.ndarray]]]:
    if not CACHE.exists():
        return {}
    # ⚠ EVERY ARRAY READ ONCE, INTO A LOCAL. An NpzFile is not a dict: each
    # z["feats"] decompresses the whole 14 MB array again, and z["feats"][row]
    # is a view that keeps that fresh copy alive. Indexed inside the loop that
    # was one full copy PER FEATURE ROW -- ~2300 x 14 MB, about 32 GB -- and on
    # 2026-09-13 it took the Pi from 2 GB available to 130 MB in 45 s and reset
    # it, before a single clip was streamed. A run with no cache yet never
    # reaches this, which is why the first retrain survived and the next did not.
    with np.load(CACHE) as z:
        paths, lengths, scores, idx, feats = (
            z[k] for k in ("paths", "lengths", "scores", "idx", "feats"))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    per: dict[int, dict[int, np.ndarray]] = {}
    for row, (n, k) in enumerate(idx):
        per.setdefault(int(n), {})[int(k)] = feats[row]
    return {str(p): (scores[offsets[n]:offsets[n + 1]], per.get(n, {}))
            for n, p in enumerate(paths)}


def save_cache(cache) -> None:
    paths = list(cache)
    idx, feats = [], []
    for n, p in enumerate(paths):
        for k, v in cache[p][1].items():
            idx.append((n, k))
            feats.append(v)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, paths=np.array(paths),
        lengths=np.array([len(cache[p][0]) for p in paths]),
        scores=np.concatenate([cache[p][0] for p in paths]) if paths else np.zeros(0),
        idx=np.array(idx, np.int32).reshape(-1, 2),
        feats=np.array(feats, np.float32).reshape(-1, 16 * 96))


def build_cache(paths: list[str], workers: int, cache=None):
    """Features for every clip, computing only the ones not already cached."""
    cache = load_cache() if cache is None else cache
    missing = [p for p in dict.fromkeys(paths) if p not in cache]
    if missing:
        started = time.monotonic()
        print(f"streaming {len(missing)} new clips "
              f"({len(cache)} already cached)…", flush=True)
        ambient = str(WAKE_DIR / "ambient.wav")
        with ProcessPoolExecutor(workers, initializer=_init,
                                 initargs=(ambient,)) as ex:
            for n, (p, result) in enumerate(
                    zip(missing, ex.map(stream_clip, missing, chunksize=4)), 1):
                cache[p] = result
                if n % 50 == 0:
                    # SAVED AS IT GOES. The Pi rebooted twice under this job
                    # on 2026-09-12, and a cache written only at the end lost
                    # every clip streamed before each one.
                    save_cache(cache)
                    print(f"  {n}/{len(missing)} clips  "
                          f"{time.monotonic() - started:.0f}s", flush=True)
        save_cache(cache)
        print(f"  done in {time.monotonic() - started:.0f}s", flush=True)
    return cache


# --- fitting ------------------------------------------------------------------

def assemble(rows, cache):
    clips, feats = [], []
    for path, group, label, split in rows:
        scores, per = cache.get(path, (np.zeros(0, np.float32), {}))
        rowmap = {}
        for k, v in per.items():
            rowmap[k] = len(feats)
            feats.append(v)
        clips.append({"path": path, "group": group, "label": label,
                      "split": split, "scores": scores, "feat": rowmap})
    return clips, np.array(feats, np.float32).reshape(-1, 16 * 96)


def training_rows(clips, feats, split: str, min_score: float):
    x, y = [], []
    for c in clips:
        if c["split"] != split and split != "all":
            continue
        if c["group"] == "fire then command":
            continue                                    # unlabelled
        for k, row in c["feat"].items():
            if c["scores"][k] >= min_score:
                x.append(feats[row])
                y.append(c["label"])
    return np.array(x, np.float32), np.array(y, np.int32)


def fit(x, y, c_value: float):
    from sklearn.linear_model import LogisticRegression
    mean = x.mean(axis=0)
    scale = x.std(axis=0) + 1e-3
    lr = LogisticRegression(C=c_value, class_weight="balanced", max_iter=4000)
    lr.fit((x - mean) / scale, y)
    return {"mean": mean.astype(np.float32), "scale": scale.astype(np.float32),
            "W0": lr.coef_.reshape(-1, 1).astype(np.float32),
            "b0": lr.intercept_.reshape(1).astype(np.float32)}


def probability(model, x: np.ndarray) -> np.ndarray:
    z = ((x - model["mean"]) / model["scale"]) @ model["W0"][:, 0] + model["b0"][0]
    return 1.0 / (1.0 + np.exp(-z))


def live_decision(scores, feat_rows, feats, model, threshold: float,
                  patience: int, verify: float | None) -> bool:
    """Would the live detector have woken on this clip? Mirrors wake_word.py."""
    above = 0
    for k, s in enumerate(scores):
        above = above + 1 if s >= threshold else 0
        if above >= max(1, patience):
            above = 0
            if verify is None or model is None:
                return True
            row = feat_rows.get(k)
            if row is None:
                continue
            if probability(model, feats[row][None, :])[0] >= verify:
                return True
    return False


def report(clips, feats, model, threshold: float, patience: int,
           verify_levels: list[float], split: str = "test") -> None:
    groups: dict[str, list] = {}
    for c in clips:
        if c["split"] == split:
            groups.setdefault(c["group"], []).append(c)
    head = f"{'':30s}{'openWakeWord':>12s}" + "".join(
        f"{'v>=' + format(v, '.2f'):>10s}" for v in verify_levels)
    print(f"\n-- held-out clips, threshold {threshold}, patience {patience} "
          f"(how many WAKE her) --")
    print(head)
    for name in sorted(groups, key=lambda g: (groups[g][0]["label"] == 0, g)):
        cs = groups[name]
        base = sum(live_decision(c["scores"], c["feat"], feats, None,
                                 threshold, patience, None) for c in cs)
        cells = [f"{base:>6d}/{len(cs):<4d}  "]
        for v in verify_levels:
            n = sum(live_decision(c["scores"], c["feat"], feats, model,
                                  threshold, patience, v) for c in cs)
            cells.append(f"{n:>6d}/{len(cs):<3d}")
        kind = "SHOULD wake" if cs[0]["label"] else "should NOT"
        print(f"  {name[:18]:18s} {kind:10s}" + "".join(cells))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--write", action="store_true",
                    help="fit on ALL clips and save config.WAKE_VERIFIER_MODEL")
    ap.add_argument("--synth", type=Path, default=None,
                    help="dir with pos/ pos_aug/ neg/ Piper clips (other voices)")
    ap.add_argument("--vosk", type=Path, default=None,
                    help="TSV of what Vosk heard in each fire (clip, path, text)")
    ap.add_argument("--after-speech", type=int, default=2,
                    help="copies of every positive with conversation in front")
    ap.add_argument("--test-since", default="20260911-2350",
                    help="fires at or after this stamp are the unseen conversation")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--C", type=float, nargs="+", default=[0.001, 0.01])
    ap.add_argument("--train-score", type=float, default=0.3)
    ap.add_argument("--levels", type=float, nargs="+",
                    default=[0.2, 0.3, 0.4, 0.5, 0.7])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cache = load_cache()
    rows = clip_table(args.synth, args.test_since, read_vosk(args.vosk),
                      set(cache))
    if args.after_speech:
        rows += after_speech(rows, args.after_speech)
    print(f"{len(rows)} clips ({sum(r[2] for r in rows)} wake, "
          f"{sum(1 - r[2] for r in rows)} not)", flush=True)
    cache = build_cache([r[0] for r in rows], args.workers, cache)
    clips, feats = assemble(rows, cache)

    x, y = training_rows(clips, feats, "train", args.train_score)
    print(f"\ntraining chunks: {int(y.sum())} wake, {int((1 - y).sum())} not "
          f"(from chunks scoring >= {args.train_score})")
    for c_value in args.C:
        model = fit(x, y, c_value)
        print(f"\n==== C = {c_value}")
        for patience in (1, 2):
            report(clips, feats, model, config.WAKE_THRESHOLD, patience,
                   args.levels)

    if args.write:
        c_value = args.C[0]
        xa, ya = training_rows(clips, feats, "all", args.train_score)
        model = fit(xa, ya, c_value)
        out = args.out or Path(config.WAKE_VERIFIER_MODEL)
        np.savez(out, **model, C=np.array(c_value, np.float32),
                 train_score=np.array(args.train_score, np.float32))
        print(f"\nwrote {out}  (C={c_value}, {int(ya.sum())} wake chunks, "
              f"{int((1 - ya).sum())} not, from ALL clips)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
