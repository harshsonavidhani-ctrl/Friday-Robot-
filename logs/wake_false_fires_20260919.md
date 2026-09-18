# The wake word fires on the television — measured 2026-09-19, ~02:00-03:00

Harsh: *"wake word false firing and also like when someone is talking"*.

## What the day actually looked like

    fires ACCEPTED (became a turn)   92
      ...that contained her name      5
      ...that did not                87   <-- 94%
    fires REJECTED by the verifier  245

~337 wake-word triggers in under two hours, against maybe thirty times he
spoke to her.

## What woke her

    'Thank you very much for watching.'
    'subscribe, can see that they are both also.'
    'You ready?'   'Okay, I did.'   'safe life.'   'You go.'
    (and many with no intelligible speech at all)

A video playing in the room, exactly as he said.

## Why the verifier cannot be tuned out of this

Its scores on the two groups overlap almost completely:

    real  "hey Friday":  min 0.36   median 0.84   max 0.91   (n=5)
    the television    :  min 0.12   median 0.61   max 0.95   (n=75)

The TV scores HIGHER at the top end than his real wakes do. Sweeping the
threshold (currently 0.35):

    0.35: real 5/5   false through 71/87
    0.50: real 4/5   false through 46/87
    0.70: real 4/5   false through 28/87
    0.80: real 3/5   false through 14/87
    0.95: real 0/5   false through  0/87

There is no line to draw. The verifier was trained 2026-09-13, before this
video was in the room, so it has never seen this audio and is confidently
wrong about it.

## Ruled out by measuring, not guessing

* **Whisper on the wake clip as the gate.** It separates the two groups
  PERFECTLY here (5/5 real, 87/87 false) -- and takes **5.2 s median** on this
  Pi with her running (min 0.16, max 7.9). Unusable as a gate. CLAUDE.md also
  warns against a record-transcribe-check loop for the wake word.
* **The auto-gain lifting room noise into the model.** `WAKE_AUTO_GAIN` is
  already `False`.

## What is left

**The retrain is the only lever left that does not cost real wakes.**

1. **Retrain the verifier with this audio as negatives.** The right fix, and
   what the existing tooling is for: `scripts/train_verifier.py`,
   `scripts/ship_verifier.py`, `scripts/verifier_before_after.py`.
   - negatives available: 87 accepted-but-false + 245 rejected = 332 from today
   - positives are the thin part: only 5 today, so pull his from the 09-13
     corpus (605 clips, 84 wake)
   - labels for today's 92 are in
     `assets/models/wake/whisper_heard_20260919.tsv`, same three-column format
     `verifier_before_after.py` already reads
2. ~~`WAKE_PATIENCE` 1 -> 2~~ **-- MEASURED, AND IT IS WORSE.** Replaying
   today's clips through the real decision at each setting:

        thr  pat  vthr | real kept | false through
        0.50   1  0.35 |    5/5    |    70/87      <-- today's setting
        0.50   1  0.60 |    3/5    |    38/87
        0.50   2  0.35 |    1/5    |    26/87
        0.50   2  0.60 |    1/5    |    12/87
        0.50   3  0.35 |    2/5    |     6/87
        0.50   3  0.60 |    1/5    |     5/87

   Patience 2 throws away FOUR OF HIS FIVE real wakes to remove two thirds of
   the false ones. Patience 3 keeps two. The 2026-09-06 numbers that made
   patience look good were measured before the verifier existed, on his voice
   in a quiet room, not against a television. Do not ship this.
   (⚠ five positives is a thin basis -- indicative, not conclusive. The
   retrain is the answer either way, and it needs more of his voice anyway.)

⚠ The training data is only worth collecting while the video keeps playing
near her. If the room goes quiet, these negatives stop being representative.
