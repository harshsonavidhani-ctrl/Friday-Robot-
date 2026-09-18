"""corrections.py — File 09 step 1: when Harsh says that was wrong, keep it.

    corrections.spoke(reply)                    remember what she just said
    corrections.capture(said, note=...)         keep a correction, or not
    corrections.preferences()                   read them back

THE POINT IS THE SEPARATE CATEGORY. These are not ordinary memory notes. A
note is something that is TRUE ("Harsh's sister is called Riya"); a correction
is something about how Friday should BEHAVE ("do not call him Sujoo"), and the
two get used differently -- a later step reads the preferences back as standing
instructions, which would be wrong to do with everything she has ever been
told. The facts table has had a `kind` column since it was written, so this
needed no schema change: kind=config.LEARNED_PREFERENCE_KIND, and
memory.facts_of_kind() reads them back.

WHAT IS STORED, per the spec: what was corrected, what the right behaviour is,
and when. The first two are the sentence; the date is the row's created_at,
which the facts table already keeps and preferences() returns.

TWO WAYS IN, because they fail in different places.

  1. THE REPLY OBJECT. The brain returns `correction` alongside reply/emotion/
     memory_note (file 08's contract, extended -- not replaced). It read the
     whole turn, so it can say what the right behaviour is, and it understands
     "nahi yeh galat hai" without anyone listing that phrase anywhere.
  2. THE PHRASE TRIPWIRE. config.CORRECTION_PHRASES, matched against what he
     actually said. This one still works with no internet, over the spend cap,
     or when the model simply did not notice it was being corrected -- and it
     is the case the spec names. It stores HER LAST REPLY as what was
     corrected, because that is what he was reacting to.

⚠ IT MUST NOT INVENT A PREFERENCE. Everything saved here is read back later as
something Friday believes about how to behave, so a false positive is worse
than a miss: Harsh has already had to delete seven junk facts by hand. Hence no
bare "wrong" in the patterns (see config), nothing saved without something to
attach it to, and source="correction" on every row so where it came from is
visible in the database.
"""

from __future__ import annotations

import logging
import re
import threading
import time

import config

log = logging.getLogger("friday.corrections")

_lock = threading.Lock()
_last_reply = ""
_last_reply_at = 0.0
# ONE CAPTURE PER THING HE SAID. main.py offers every turn twice -- once from
# the brain path, which may carry the reply object's `correction`, and once
# from the end-of-turn hook, which is the only path a turn answered offline
# takes. The brain's version is better, and it arrives first, so the second
# offer of the same sentence is dropped rather than stored again.
_last_said = ""
_last_said_at = 0.0

_PATTERNS = tuple(re.compile(p, re.I) for p in
                  getattr(config, "CORRECTION_PHRASES", ()))


def spoke(reply: str) -> None:
    """Remember what Friday just said, so a correction has a subject."""
    global _last_reply, _last_reply_at
    text = (reply or "").strip()
    if not text:
        return
    with _lock:
        _last_reply, _last_reply_at = text, time.monotonic()


def last_reply() -> str:
    with _lock:
        return _last_reply


def looks_like_correction(said: str) -> str | None:
    """The correction phrase he used, or None. Never raises."""
    text = (said or "").strip()
    if not text:
        return None
    for pattern in _PATTERNS:
        found = pattern.search(text)
        if found:
            return found.group(0)
    return None


def _sentence(said: str, note: str | None, phrase: str | None) -> str | None:
    """The one line that gets stored, or None when there is nothing to store."""
    if note and note.strip():
        # The brain already wrote "what was wrong, what to do instead".
        return note.strip()
    previous = last_reply()
    if not previous:
        # Nothing to attach it to. A bare "that's wrong" with no reply behind
        # it is not a preference, it is a complaint about the room.
        return None
    return (f'Harsh corrected Friday: he said "{said.strip()}" after she said '
            f'"{previous}" — do not answer that way again.')


def capture(said: str, note: str | None = None,
            reply: str | None = None) -> dict | None:
    """Keep a correction if this turn was one. Returns the stored row or None.

    `note` is the reply object's `correction` field; when it is set this is a
    correction on the brain's word and no phrase has to match. Never raises --
    a failure here must never cost Harsh his answer.
    """
    if not getattr(config, "CORRECTION_CAPTURE", True):
        return None
    global _last_said, _last_said_at
    try:
        phrase = looks_like_correction(said)
        if not phrase and not (note and note.strip()):
            return None
        spoken = (said or "").strip()
        with _lock:
            again = (spoken and spoken == _last_said
                     and time.monotonic() - _last_said_at < 120.0)
            _last_said, _last_said_at = spoken, time.monotonic()
        if again:
            log.debug("already kept the correction for %r", spoken[:40])
            return None
        text = _sentence(said, note, phrase)
        if not text:
            log.info("correction heard (%r) but nothing to attach it to",
                     phrase)
            return None
        limit = int(getattr(config, "CORRECTION_MAX_CHARS", 240))
        if len(text) > limit:
            text = text[:limit].rsplit(" ", 1)[0] + "…"

        import memory
        stored = memory.remember(
            text,
            kind=getattr(config, "LEARNED_PREFERENCE_KIND",
                         "learned_preference"),
            source=getattr(config, "CORRECTION_SOURCE", "correction"))
        log.info("learned preference %s (%s): %r",
                 "updated" if stored["updated"] else "kept",
                 "brain" if note else f"phrase {phrase!r}", stored["text"])
        return stored
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not keep the correction (%s: %s)",
                    type(exc).__name__, exc)
        return None


def preferences(limit: int = 50) -> list[dict]:
    """Everything Friday has been corrected on, newest first. Never raises.

    Each row carries what was stored and WHEN -- the date the spec asks for is
    the facts table's created_at, not a second copy inside the sentence.
    """
    try:
        import memory
        kind = getattr(config, "LEARNED_PREFERENCE_KIND", "learned_preference")
        rows = []
        for fact in memory.facts_of_kind(kind, limit):
            row = fact.as_dict()
            row["date"] = time.strftime("%Y-%m-%d",
                                        time.localtime(fact.created_at))
            rows.append(row)
        return rows
    except Exception as exc:                    # noqa: BLE001
        log.warning("could not read the learned preferences: %s", exc)
        return []


def main() -> int:
    """What Friday has been corrected on."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    rows = preferences()
    if not rows:
        print("nothing yet")
        return 0
    for row in rows:
        print(f"{row['date']}  [{row['source']}]  {row['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
