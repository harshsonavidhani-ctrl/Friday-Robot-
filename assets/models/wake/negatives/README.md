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

## How to fill it

```bash
# 1. Ten minutes of normal conversation. Say anything EXCEPT the wake word.
#    Saying "Friday" on its own is GOOD — that is the confusion worth training.
./venv/bin/python scripts/train_wake.py --record-negatives 600

# 2. Every live fire is already being saved to state/wake_fires/.
#    Leave Friday running through a real conversation, then label them:
./venv/bin/python scripts/train_wake.py --review-fires

# 3. Retrain, and read BOTH columns of the report this time.
./venv/bin/python scripts/train_wake.py
./venv/bin/python scripts/train_wake.py --score
```

## Read both columns

`--score` now reports catch rate **and** false fires per hour. Catch rate alone
is what shipped the model that fires 97 times an evening: it was 80% and looked
like a success, because nothing in the script had ever asked the other
question.
