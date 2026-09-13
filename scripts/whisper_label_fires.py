#!/usr/bin/env python3
"""Label harvested wake fires with Whisper, so the verifier can be retrained.

    ./venv/bin/python scripts/whisper_label_fires.py --recent 160

WHY. scripts/ship_verifier.py learns which fires were really "hey Friday" from
an INDEPENDENT recogniser -- never from the wake model's own opinion, which
would just record its existing mistakes back into its training set. Vosk was
that recogniser (assets/models/wake/vosk_heard.tsv) and it labelled 68 of ~348
fires at 3.6 s a clip. Whisper is already loaded on this Pi, is at least as
good on Harsh's voice, and can catch up with the pile.

⚠ THIS IS A LABELLER, NOT A JUDGE. It writes down what it heard and nothing
else; whether a clip counts as a wake is still decided by ship_verifier's
FRIDAY regex over that text, exactly as it is for Vosk. A clip Whisper mangles
simply goes unlabelled rather than becoming a wrong positive.

Output is assets/models/wake/whisper_heard.tsv, same three columns as the Vosk
file (clip, path, heard) and kept separate so the two labellers never get
confused for one another. Re-running only transcribes clips not already in it.
"""
import argparse
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                                # noqa: E402

OUT = config.ROOT / "assets/models/wake/whisper_heard.tsv"


def clips(recent: int) -> list[Path]:
    """The newest fires from every pile the harvester writes to."""
    found: list[tuple[float, Path]] = []
    for directory in (Path(config.WAKE_FIRE_DIR),
                      Path(config.WAKE_REJECT_DIR),
                      config.ROOT / "assets/models/wake/unsure",
                      config.ROOT / "assets/models/wake/negatives"):
        for p in directory.glob("fire-*.wav"):
            try:
                found.append((p.stat().st_mtime, p))
            except OSError:
                continue        # the detector trims these while we read them
    found.sort()
    return [p for _, p in found[-recent:]]


def already() -> dict[str, str]:
    if not OUT.exists():
        return {}
    rows = {}
    for line in OUT.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 3:
            rows[parts[0]] = parts[2]
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recent", type=int, default=160,
                    help="how many of the newest clips to label")
    ap.add_argument("--threads", type=int, default=3)
    args = ap.parse_args()

    from faster_whisper import WhisperModel
    model = WhisperModel(config.STT_MODEL, device="cpu", local_files_only=True,
                         download_root=str(config.MODEL_DIR / "whisper"),
                         compute_type=config.STT_COMPUTE_TYPE,
                         cpu_threads=args.threads)

    done = already()
    todo = [p for p in clips(args.recent) if p.name not in done]
    print(f"{len(done)} already labelled, {len(todo)} to do", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT.exists()
    with open(OUT, "a") as out:
        if new:
            out.write("clip\tpath\twhisper_heard\n")
        started = time.monotonic()
        for i, p in enumerate(todo, 1):
            try:
                with wave.open(str(p)) as w:
                    audio = (np.frombuffer(w.readframes(w.getnframes()),
                                           dtype=np.int16)
                             .astype(np.float32) / 32768.0)
            except Exception:
                continue
            # English pinned: the clip is a wake word, not a sentence, and
            # auto-detect on 3 s of room noise is what invents Korean.
            segments, _ = model.transcribe(
                audio, language="en", beam_size=1, vad_filter=False,
                condition_on_previous_text=False, max_new_tokens=48,
                no_repeat_ngram_size=3, temperature=[0.0],
                without_timestamps=True)
            heard = " ".join(s.text.strip() for s in segments).strip()
            heard = heard.replace("\t", " ").replace("\n", " ")
            out.write(f"{p.name}\t{p}\t{heard}\n")
            out.flush()
            if i % 10 == 0 or i == len(todo):
                rate = (time.monotonic() - started) / i
                print(f"  {i}/{len(todo)}  {rate:.1f}s/clip  "
                      f"~{rate * (len(todo) - i) / 60:.1f} min left", flush=True)

    rows = already()
    import re
    friday = re.compile(r"\bfri?day\b|\bfrida\b|\bfry day\b", re.I)
    hits = [k for k, v in rows.items() if friday.search(v)]
    print(f"\n{len(rows)} labelled, {len(hits)} contain her name", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
