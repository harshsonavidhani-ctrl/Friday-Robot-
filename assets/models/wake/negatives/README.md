# Hard negatives — real speech that is NOT the wake word

This folder is the fix for the bug where Friday answered a conversation.

## What went wrong

Measured 2026-09-05: **97 wake fires in one evening** while Harsh talked to a
friend — median score 0.88, **max 1.00**. The false fires score as high as the
real ones, so no threshold separates them and tuning `WAKE_THRESHOLD` is wasted
effort.

The cause is the training set, not the code. The model shipped on 2026-08-27
had never heard a real conversation:

* its **negatives** were Piper TTS phrases and 45 s of an *empty* room;
* **60% of its positives** were 1280 augmentations of 40 clips of one voice.

So it learned *"a real human voice in this room"* — which is what a
conversation is made of.

## What goes in here

| file | what it is | in git? |
|---|---|---|
| `conversation-*.wav` | minutes of normal talking | **no** — private, large, and re-recordable in ten minutes |
| `fire-*.wav` | a clip harvested from a **live false fire** | **yes** — each one is a moment the model was actually wrong, in this room, on this mic, and re-recording never reproduces it |

Next door, and never trained on as negatives:

| file | what it is |
|---|---|
| `../unsure/fire-*.wav` | fires that **may hold a real "hey Friday"** — a command or a request to her followed. Kept for Harsh to judge by ear. |
| `../labels.tsv` | every harvested fire, its label, and the evidence for it |
| `../holdout.txt` | the files the hold-out model never sees, so `--evaluate` measures something real |

## How the fires in here were labelled (2026-09-10)

Not by the detector — it cannot know it was wrong — and not by Whisper, which
turned out not to hear the phrase: unhinted, `tiny` found "Friday" in 0 of
Harsh's first 6 recordings, and even hinted with "Hey Friday." it missed all
8 of his quiet takes. The label comes from **what the turn after the fire
heard**, read out of `logs/friday.log`:

| evidence after the fire | label | clips |
|---|---|---|
| nobody spoke for the full 8 s (`NOTHING SAID`, no retry) | negative | 77 |
| conversation followed, nothing addressed to her | negative | 54 |
| her own narration, or talk that matched a command by accident | negative | 9 |
| speech followed but was never transcribed (lock timeouts, empties) | **unsure** | 113 |
| a real command followed | **unsure** | 17 |
| a request to her followed — "What can you do?", "how are you?" | **unsure** | 16 |
| no turn in the log | **unsure** | 1 |

Conservative on purpose. A real "hey Friday" filed here teaches the model to
ignore Harsh, which is a worse bug than the one being fixed. **18 of the 44
clips first filed here failed this test** — 6 were followed by a real command,
4 by a request to her, 8 by speech nobody transcribed — and were moved to
`../unsure/`.

## What the trainer does with them

* **It trains on the END of each clip.** A fire clip is the 3 s *before* the
  fire, and the model was scoring the last 2 s when it fired. `place()` kept
  the *first* 2 s, so the second of audio that actually fooled the model was
  thrown away.
* **Reversed and reshuffled copies of every real recording** — Harsh's 40 takes
  and every harvested fire, labelled or not — go in as negatives. Same voice,
  room and mic, with the order of the sounds destroyed, so the model has to
  learn the phrase rather than the voice. No label needed: a reversed
  "hey Friday" is not "hey Friday".

## How to fill it

```bash
# 1. Ten minutes of normal conversation. Say anything EXCEPT the wake word.
#    Saying "Friday" on its own is GOOD — that is the confusion worth training.
./venv/bin/python scripts/train_wake.py --record-negatives 600

# 2. Every live fire is already being saved to state/wake_fires/.
#    Leave Friday running through a real conversation, then label them.
#    The unsure pile is labelled the same way:
./venv/bin/python scripts/train_wake.py --review-fires
./venv/bin/python scripts/train_wake.py --review-fires --fires assets/models/wake/unsure

# 3. Retrain, with a hold-out model trained beside the real one...
./venv/bin/python scripts/train_wake.py --holdout assets/models/wake/holdout.txt \
    --out assets/models/wake/friday.candidate.onnx

# 4. ...and measure BOTH columns on clips the hold-out model never heard.
./venv/bin/python scripts/train_wake.py --evaluate --holdout assets/models/wake/holdout.txt \
    --compare assets/models/wake/friday.onnx assets/models/wake/friday.candidate.holdout.onnx
```

## Read both columns

`--evaluate` reports catches **and** false fires, replayed the way the live
detector hears them (threshold and `WAKE_PATIENCE` as configured). Catch rate
alone is what shipped the model that fires 97 times an evening: it was 80% and
looked like a success, because nothing in the script had ever asked the other
question.
