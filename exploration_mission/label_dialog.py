"""Asking what an area is called, and understanding the answer (File 21 step 6).

OFFLINE GRAMMAR, NO MODEL (spec 4.7). Harsh answers in English or Hinglish;
faster-whisper (language "en", so Hinglish comes out in Latin script) gives a
transcript; this turns it into one of

    NAME(x)   ACCEPT   SKIP   STOP   FINISH   UNCLEAR

with regular expressions and word lists only. Every pattern must match the
WHOLE normalised utterance, tried top to bottom, first match wins.

    parse_label("Friday, save it as kitchen.")        -> NAME kitchen
    parse_label("isko guest room bolo")               -> NAME guest room
    parse_label("haan sahi hai", suggestion="kitchen") -> ACCEPT kitchen
    parse_label("wapas chalo")                        -> FINISH

`LabelDialog.run` is the conversation itself (spec 4.7 flow): ask -> listen ->
parse -> confirm, one re-ask, then the area is saved as "area N" and queued --
an area is NEVER thrown away because nobody answered. It speaks and listens
only through the adapters it is handed, and saves only through area_store.

The normaliser and the HALT / FINISH word lists are shared with intents.py,
which imports them from here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from exploration_mission import area_store, speech

log = logging.getLogger("friday.exploration_mission.label_dialog")

# --- normalisation (spec 4.7) ------------------------------------------------------

WAKE = ("hey friday", "ok friday", "okay friday", "hi friday", "friday")
FILLERS = frozenset({"please", "plz", "umm", "um", "uh", "hmm", "ok", "okay",
                     "acha", "achha"})
NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    ("one two three four five six seven eight nine ten eleven twelve thirteen "
     "fourteen fifteen sixteen seventeen eighteen nineteen twenty").split(), 1)}
_KEEP = re.compile(r"[^\w' ]+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    """Cased tokens after: punctuation off (except '), - and _ to spaces,
    wake word off the front, fillers out, number words to digits."""
    s = str(text or "").replace("’", "'").replace("‘", "'")
    s = s.replace("-", " ").replace("_", " ")
    s = _KEEP.sub(" ", s)
    toks = [t for t in (raw.strip("'") for raw in s.split()) if t]
    lower = [t.lower() for t in toks]
    for wake in WAKE:                                   # longest first
        w = wake.split()
        if lower[:len(w)] == w:
            toks, lower = toks[len(w):], lower[len(w):]
            break
    out = []
    for tok, low in zip(toks, lower):
        if low in FILLERS:
            continue
        out.append(NUMBER_WORDS.get(low, tok))
    return out


def normalize(text: str) -> str:
    """The form every pattern is matched against (lowercase)."""
    return " ".join(_tokens(text)).lower()


def normalize_cased(text: str) -> str:
    """The same string with Harsh's capitals kept, for display names."""
    return " ".join(_tokens(text))


# --- word lists --------------------------------------------------------------------

YES = ("yes", "yeah", "yep", "haan", "han", "ha", "haa", "ji")
NO = ("no", "nahi", "nahin", "galat", "wrong")
ACCEPT_STARTS = YES + ("sahi", "correct", "right", "theek hai", "thik hai")
SKIP_STARTS = ("skip", "not now", "later", "leave it", "chhodo", "chodo",
               "rehne do", "baad mein", "baad me", "pata nahi", "i don't know",
               "i dont know")
# BARE rejects any utterance containing one of these (spec 4.7).
COMMAND_WORDS = frozenset({"go", "back", "come", "wapas", "chalo", "bas",
                           "continue", "resume", "explore", "stop", "ruko",
                           "list", "status", "rename", "forget", "delete", "home"})

# Whole-utterance HALT and FINISH (spec 4.8). intents.py uses these too.
HALT_RE = re.compile(r"^(stop|ruko|ruk jao|ruk ja|bas)$")
FINISH_RE = re.compile(
    r"^(stop exploring|stop exploration|finish exploring|finish exploration|"
    r"that's enough|thats enough|that is enough|go back|come back|wapas aao|"
    r"wapas chalo|wapas aa jao|bas karo)$")

_DO = r"(?:karo|kar do|kardo|krdo|kr do|kar)"
# Longer alternatives first inside each group (spec 4.7). The tag says whether
# the sentence ORDERS a name ("save this as X") or merely STATES something
# ("it's X"). A statement is a name only as the answer to her question: on
# Harsh's own turn log (1056 transcripts, 2026-09-30) the statements matched
# 30 ordinary remarks ("That's it.", "Yeah, it's good.") and the orders none.
IMPERATIVE, DECLARATIVE = "imperative", "declarative"
NAME_PATTERNS: tuple[tuple[str, re.Pattern], ...] = tuple(
    (tag, re.compile(rx)) for tag, rx in (
        (IMPERATIVE, r"^(?:save|call|name|label|mark) (?:this area|this room|it|"
                     r"this|that|here) (?:as|to) (?P<x>.+)$"),
        (IMPERATIVE, r"^(?:call|name) (?:this area|this room|it|this|that) (?P<x>.+)$"),
        (DECLARATIVE, r"^(?:it's|its|it is|this is|that's|thats|that is) "
                      r"(?:(?:the|a|my|our) )?(?P<x>.+)$"),
        (IMPERATIVE, r"^(?:is area ko|is room ko|ise|isko|isse) (?P<x>.+?) (?:ke )?"
                     r"(?:naam|name) se (?:save|call) " + _DO + r"$"),
        (IMPERATIVE, r"^(?:ise|isko|isse) (?P<x>.+?) (?:save|call) " + _DO + r"$"),
        (IMPERATIVE, r"^(?:iska|is ka|iss ka) naam (?P<x>.+?) (?:rakho|rakh do|rakhna)$"),
        (DECLARATIVE, r"^(?:iska|is ka|iss ka) naam (?P<x>.+?) hai$"),
        (IMPERATIVE, r"^(?:ise|isko) (?P<x>.+?) (?:bolo|kaho)$"),
        (DECLARATIVE, r"^(?:ye|yeh|yah) (?P<x>.+?) (?:hai|he)$"),
    ))

# A BARE answer (1-3 words, no pattern) is refused when it is one of these.
# Not in the spec's lists; every entry is from his turn log. faster-whisper
# writes "Thank you." / "Thanks for watching!" / "Bye." on silence and noise
# (22 + 3 + 7 times), and the rest are things he says TO her, not names.
NOT_A_NAME = frozenset({
    "thank you", "thanks", "thank you very much", "thanks for watching",
    "thank you for watching", "bye", "bye bye", "goodbye", "see you later",
    "i love you", "love you", "hello", "hi", "hey", "ah", "oh", "off", "on",
    "alrighty", "alright", "nothing", "you", "so",
    # pronouns and acknowledgements: "That's it." / "It's good." name nothing
    "it", "this", "that", "here", "there", "good", "very good", "fine", "nice",
    "great", "done", "haha"})
# ...or contains one of these: the first words of his own orders to Friday
# (offline.py's reflexes) and question words, with or without "'s".
NOT_IN_A_NAME = frozenset({
    "move", "drive", "turn", "forward", "backward", "backwards", "left", "right",
    "open", "play", "search", "tell", "show", "set", "remind", "watch", "look",
    "where", "when", "which", "whose", "is", "are", "can", "do", "does", "did"})


def _starts(norm: str, phrases) -> bool:
    return any(norm == p or norm.startswith(p + " ") for p in phrases)


def yes_no(text: str) -> bool | None:
    """A confirm answer: True (yes), False (no), None (neither)."""
    norm = normalize(text)
    if _starts(norm, ACCEPT_STARTS):
        return True
    if _starts(norm, NO):
        return False
    return None


def name_slot(text: str, declarative: bool = True) -> tuple[str, str] | None:
    """(lowercase X, cased X) if an EN/HI naming pattern matches the whole
    utterance -- with or without a leading yes/no word -- else None. X is NOT
    validated here. `declarative=False` keeps only the sentences that ORDER a
    name: what intents.py uses when no question is on the table."""
    toks = _tokens(text)
    low = [t.lower() for t in toks]
    for strip_lead in (False, True):
        if strip_lead:
            if not low or low[0] not in YES + NO:
                return None
            while low and low[0] in YES + NO:
                toks, low = toks[1:], low[1:]
        norm, cased = " ".join(low), " ".join(toks)
        for tag, rx in NAME_PATTERNS:
            if tag == DECLARATIVE and not declarative:
                continue
            m = rx.match(norm)
            if m:
                a, b = m.span("x")
                x_cased = cased[a:b] if len(cased) == len(norm) else norm[a:b]
                return m.group("x"), x_cased
    return None


# --- the parser ----------------------------------------------------------------------

@dataclass
class Parsed:
    kind: str                       # NAME | ACCEPT | SKIP | STOP | FINISH | UNCLEAR
    name: str | None = None         # normalised
    display: str | None = None      # as spoken
    why: str = ""

    def __str__(self) -> str:
        return f"{self.kind} {self.name}" if self.name else self.kind


def _named(x_cased: str, max_words: int, max_chars: int) -> Parsed:
    try:
        name, display = area_store.validate_name(x_cased, max_words, max_chars)
    except area_store.InvalidName as exc:
        return Parsed("UNCLEAR", why=str(exc))
    if name in NOT_A_NAME:
        return Parsed("UNCLEAR", why=f"{name!r} is not a name")
    return Parsed("NAME", name, display)


def parse_label(text: str | None, suggestion: str | None = None,
                max_words: int = 4, max_chars: int = 30) -> Parsed:
    """What an answer to "what should I call it?" means (spec 4.7).

    Order: HALT, FINISH (they still work while a dialog is open), then the
    naming patterns (EN, HI, then again after a leading yes/no), ACCEPT (only
    when a suggestion was offered), SKIP, and last a BARE 1-3 word name.
    """
    norm = normalize(text or "")
    if not norm:
        return Parsed("UNCLEAR", why="silence")
    if HALT_RE.match(norm):
        return Parsed("STOP")
    if FINISH_RE.match(norm):
        return Parsed("FINISH")
    slot = name_slot(text or "")
    if slot is not None:
        return _named(slot[1], max_words, max_chars)
    if _starts(norm, ACCEPT_STARTS):
        if not suggestion:
            return Parsed("UNCLEAR", why="yes, but nothing was suggested")
        try:
            name, display = area_store.validate_name(suggestion, max_words, max_chars)
        except area_store.InvalidName as exc:
            return Parsed("UNCLEAR", why=f"the suggestion is not a name: {exc}")
        return Parsed("ACCEPT", name, display)
    if _starts(norm, SKIP_STARTS):
        return Parsed("SKIP")
    words = norm.split()
    if 1 <= len(words) <= 3:
        if norm in NOT_A_NAME:
            return Parsed("UNCLEAR", why=f"{norm!r} is not a name")
        # "what's" counts as "what": a question is not a name.
        roots = [w[:-2] if w.endswith("'s") else w for w in words]
        bad = [w for w, r in zip(words, roots)
               if r in COMMAND_WORDS or r in area_store.RESERVED
               or w in NOT_IN_A_NAME or (r != w and r in NOT_IN_A_NAME)]
        if bad:
            return Parsed("UNCLEAR", why=f"{bad[0]!r} is a command or reserved word")
        return _named(normalize_cased(text or ""), max_words, max_chars)
    return Parsed("UNCLEAR", why="not a name")


# --- object hints (spec 4.7 point 9) ---------------------------------------------------

SUGGESTIONS: dict[str, str] = {
    "refrigerator": "kitchen", "oven": "kitchen", "microwave": "kitchen",
    "sink": "kitchen", "bed": "bedroom", "toilet": "bathroom",
    "sofa": "living room", "couch": "living room", "tv": "living room",
    "tvmonitor": "living room", "diningtable": "dining room",
    "dining table": "dining room",
}


def suggest(sightings, min_conf: float = 0.6, min_sightings: int = 2) -> str | None:
    """A room name from what the detector saw INSIDE the area: labels with
    conf >= min_conf seen in >= min_sightings frames. `sightings` is
    [(label, conf), ...], one per frame it was seen in. The room with the most
    evidence wins; a tie is no suggestion at all."""
    counts: dict[str, int] = {}
    for label, conf in sightings or []:
        if float(conf) >= min_conf:
            counts[str(label).lower()] = counts.get(str(label).lower(), 0) + 1
    rooms: dict[str, int] = {}
    for label, n in counts.items():
        room = SUGGESTIONS.get(label)
        if room and n >= min_sightings:
            rooms[room] = rooms.get(room, 0) + n
    if not rooms:
        return None
    best = sorted(rooms.items(), key=lambda kv: -kv[1])
    if len(best) > 1 and best[0][1] == best[1][1]:
        return None
    return best[0][0]


# --- the conversation -------------------------------------------------------------------

@dataclass
class DialogResult:
    outcome: str                    # saved | unnamed | stopped | finished
    area: dict | None = None
    transcript: list[dict] = field(default_factory=list)


class LabelDialog:
    """spec 4.7 flow. Motors are already confirmed stopped by the caller."""

    def __init__(self, adapters, store, listen_s: float = 10.0,
                 reask_once: bool = True, max_name_words: int = 4,
                 max_name_chars: int = 30):
        self.a = adapters
        self.store = store
        self.listen_s = listen_s
        self.reask_once = reask_once
        self.max_words = max_name_words
        self.max_chars = max_name_chars

    # -- small helpers --
    def _say(self, log_: list, key: str, **fields) -> None:
        log_.append({"said": speech.say(self.a, key, **fields)})

    def _hear(self, log_: list, suggestion: str | None = None) -> Parsed:
        try:
            heard = self.a.voice.listen_once(self.listen_s)
        except Exception as exc:                        # noqa: BLE001
            log.warning("could not listen: %s", exc)
            heard = None
        parsed = parse_label(heard, suggestion, self.max_words, self.max_chars)
        log_.append({"heard": heard or "", "parsed": str(parsed), "why": parsed.why})
        return parsed

    def _save(self, seg: dict, name: str | None, source: str) -> dict:
        return self.store.add_area(
            seg["env_id"], seg.get("map_id"), polygon=seg["polygon"],
            center=seg["center"], area_m2=seg["area_m2"],
            radius_m=seg.get("radius_m"), name=name, source=source,
            mission_id=seg.get("mission_id"), complete=seg.get("complete", True),
            hints=seg.get("hints"), frame=seg.get("frame", area_store.FRAME_MAP))

    def _unnamed(self, seg: dict, log_: list, outcome: str = "unnamed",
                 speak: bool = True) -> DialogResult:
        area = self._save(seg, None, "auto")
        if speak:
            self._say(log_, "unnamed", name=area["display_name"])
        return DialogResult(outcome, area, log_)

    def _named(self, seg: dict, parsed: Parsed, log_: list,
               may_loop: bool = True) -> DialogResult:
        """Save under parsed.display, handling a name that is taken (4.7 #8)."""
        check = self.store.check_name(seg["env_id"], parsed.display, seg["polygon"])
        if check.kind in ("ok", "same_place"):
            area = self._save(seg, parsed.display, "voice")
            self._say(log_, "saved", name=area["display_name"])
            return DialogResult("saved", area, log_)
        if check.kind in ("duplicate_area", "duplicate_landmark"):
            # "I already have a kitchen. Should I call this kitchen 2?"
            # yes -> kitchen 2 · a new name -> use it (one loop at most) ·
            # anything else -> unnamed.
            self._say(log_, "duplicate", name=check.name, suggestion=check.suggestion)
            answer = self._hear(log_, suggestion=check.suggestion)
            if answer.kind == "STOP":
                return DialogResult("stopped", None, log_)
            if answer.kind == "FINISH":
                return self._unnamed(seg, log_, "finished", speak=False)
            if answer.kind == "ACCEPT":
                area = self._save(seg, check.suggestion, "voice")
                self._say(log_, "saved", name=area["display_name"])
                return DialogResult("saved", area, log_)
            if answer.kind == "NAME" and may_loop:
                return self._named(seg, answer, log_, may_loop=False)
        return self._unnamed(seg, log_)

    def named_now(self, seg: dict, parsed: Parsed) -> DialogResult:
        """"Save this as X" said out of the blue (LABEL_NOW): no question, the
        name is already known -- but a taken name still gets "kitchen 2?"."""
        return self._named(seg, parsed, [])

    # -- the flow --
    def run(self, seg: dict, suggestion: str | None = None) -> DialogResult:
        """`seg`: env_id, map_id, polygon, center, area_m2 (+ radius_m,
        mission_id, complete, hints, frame). Returns what happened; the area is
        saved before this returns in every case except STOP."""
        log_: list[dict] = []
        if suggestion:
            self._say(log_, "ask_suggestion", suggestion=suggestion)
        else:
            self._say(log_, "ask_label")
        parsed = self._hear(log_, suggestion)
        if parsed.kind == "UNCLEAR" and self.reask_once:
            self._say(log_, "reask")
            parsed = self._hear(log_, suggestion)
        if parsed.kind == "STOP":
            return DialogResult("stopped", None, log_)      # pending stays pending
        if parsed.kind == "FINISH":
            return self._unnamed(seg, log_, "finished", speak=False)
        if parsed.kind in ("NAME", "ACCEPT"):
            return self._named(seg, parsed, log_)
        return self._unnamed(seg, log_)                      # SKIP, UNCLEAR, silence
