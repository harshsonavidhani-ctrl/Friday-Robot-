"""reply_contract.py — the shape of Friday's final spoken turn.

File 08, step 1. Every turn ends with ONE object, after any tool calls:

    {"reply": "what Friday says out loud",
     "emotion": "one of face_control.EMOTIONS",
     "memory_note": "something worth keeping, or null"}

Tools (web search, weather, memory, vision) stay ordinary tool calls. This is
only the last thing she says.

WHY IT IS A PROMPT CONTRACT AND NOT STRUCTURED OUTPUTS. The API can constrain a
reply to a JSON schema (`output_config.format`), but that is a 400 whenever the
response carries citations -- and a web search answer always does. Web search
is the one ability Harsh explicitly asked for (f389b86), so the schema cannot be
enforced by the API. It is asked for in the system prompt instead, and this
module is TOLERANT BY CONSTRUCTION: a reply that ignores the contract and comes
back as plain words is still spoken exactly as before, with no emotion and no
note. A malformed object must never make her go silent.

WHY IT STREAMS. Piper starts on sentence one, and time to the first spoken word
is what Harsh waits for (see brain.py, SPEAKING EARLY). Waiting for the whole
object before speaking would add the full generation time to every answer. So
`ReplyStream` reads the object as it arrives and hands out the characters of
the "reply" string the moment they are decoded -- which is why the prompt asks
for "reply" FIRST. emotion and memory_note are read once the object closes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import config
from face_control import EMOTIONS

# File 08 step 2 -- THE EMOTION TABLE, in two halves.
#
# Owned by the hardware and the pipeline. The voice loop sets listening,
# thinking and speaking as a turn moves; the face's own idle decay sets sleepy
# and sleeping. The model must never output one: a reply object that does is
# read as "no emotion chosen" (see _clean_emotion), and main.py falls back.
PIPELINE_OWNED = ("listening", "speaking", "thinking", "sleepy", "sleeping")

# Content-driven: the model picks exactly one per reply. Harsh's table,
# verbatim in meaning. Order is the order he wrote it.
EMOTION_TRIGGERS: dict[str, str] = {
    "neutral": "default, no strong signal",
    "happy": "compliment, good news, positive exchange",
    "sad": "bad news, user upset, a task failed",
    "angry": "user is rude/insulting, repeated failure frustration",
    "surprised": "unexpected input or sudden fact",
    "confused": "ambiguous command, unclear intent",
    "dizzy": "internal error, a tool call failed",
    "success": "task completed, goal achieved",
    "excited": "fun plan, anticipation",
    "wink": "joke, tease, sarcasm",
    "love": "user expresses affection toward Friday",
    "curious": "new/unknown object seen, new topic raised",
    "proud": "user praises Friday's skill, or Friday completes something notable",
    "bored": "idle small talk, repetitive mundane request",
    "scared": "loud/startling event, critical low battery",
}

# The two halves must be exactly the face's 20, or a face exists that nobody
# may choose (or a name is offered that the face cannot draw).
assert set(EMOTION_TRIGGERS) | set(PIPELINE_OWNED) == set(EMOTIONS), \
    "reply_contract's emotion table has drifted from face_control.EMOTIONS"
assert not set(EMOTION_TRIGGERS) & set(PIPELINE_OWNED)

_TABLE = "\n".join(f"  - `{name}` — {why}"
                   for name, why in EMOTION_TRIGGERS.items())

INSTRUCTIONS = f"""\
## How every turn ends — the reply object

Your FINAL message in every turn, after any tools have finished, is exactly one
JSON object and nothing else — no markdown fence, no words before or after it:

{{"reply": "...", "emotion": "...", "memory_note": null}}

- `reply` — the words you say out loud, and ONLY those. It is spoken by a speech
  synthesiser the moment you write it, so it comes FIRST, and every rule in
  Voice applies to it. `IGNORE` goes here when it applies.
- `emotion` — the face you show once you have finished speaking. Exactly one
  row of this table: the one whose trigger matches what Harsh said or what
  happened this turn.
{_TABLE}
  **The face is where your reaction shows; your words stay professional.** So
  an insult is `angry` even when you only say "Understood.", a fun plan is
  `excited` even when you answer it plainly, a fact he offers that you did not
  expect is `surprised`, and the same request again and again is `bored`.
  `neutral` only when no row matches — a plain question answered plainly.
  Never {", ".join(f"`{n}`" for n in PIPELINE_OWNED)}: the robot sets those
  itself.
  **Judge THIS turn alone.** An emotion does not carry over: after a happy or
  sad moment, the next turn about something unrelated is `neutral` again (or
  whatever that turn itself calls for). The emotions in your earlier replies
  are not a mood to keep.
- `memory_note` — one clear third-person sentence worth keeping after this
  conversation ("Harsh's sister is called Riya"), or `null`. Almost always
  `null`. Set it only when Harsh told you something that will still be true
  later: a name, a preference, a standing instruction, where something lives.
  It is saved for you after you speak — do not also call `remember` for it.
  Never for small talk, for what you just did, or for anything you can sense
  or look up again.

Before a tool call you may say one short plain sentence ("Let me look that
up."). After the tools, write the object straight away: never the answer in
plain words first and then the object — everything plain is spoken too, so he
hears the answer twice.
"""

# Put on the LAST user message of every request, on the wire only (brain.py
# _outgoing). Measured 2026-09-17 on Haiku: the fast path kept the contract every
# time with the rules in the system prompt alone, but the full turn with tools
# broke it three times out of three -- after a tool result the model answers the
# way it always has, and the rules are 13,000 cached tokens back.
#
# ⚠ WORDED SO A LEAD-IN SURVIVES. "No plain text before it" cost the
# "Let me look that up." that covers a web search: first word 1.4 s -> 12.6 s.
REMINDER = ("[When you give the answer, give it only as the reply object "
            '{"reply": ..., "emotion": ..., "memory_note": ...} -- never in plain '
            "words first. One short plain sentence before a tool call is fine.]")

# What a ```json fence looks like PART-WAY through arriving.
_FENCE_SO_FAR = re.compile(r"^(`{1,3}|```[a-zA-Z]*\s*)$")
_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "", "f": "", "n": "\n",
            "r": "", "t": " "}


@dataclass
class Reply:
    """One finished turn, as the contract describes it."""
    reply: str
    emotion: str | None = None
    memory_note: str | None = None
    structured: bool = False          # False = came back as plain words

    def as_dict(self) -> dict[str, Any]:
        return {"reply": self.reply, "emotion": self.emotion,
                "memory_note": self.memory_note}


def _normal(text: str) -> str:
    """Text compared for an echo: case, spacing and backticks ignored."""
    return " ".join(text.replace("`", "").lower().split())


def _clean_emotion(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip().lower()
    # A pipeline-owned name is not the model's to choose (file 08 step 2).
    return name if name in EMOTION_TRIGGERS else None


def _clean_note(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    note = value.strip()
    # A model asked for null sometimes writes the word instead.
    if not note or note.lower() in ("null", "none", "n/a"):
        return None
    return note


class ReplyStream:
    """Reads a response as it streams; hands out only what is to be SPOKEN.

        stream = ReplyStream()
        stream.new_block()                 # at each text content_block_start
        say(stream.feed(delta))            # at each text_delta
        result = stream.finish()           # Reply

    Plain text passes straight through, unchanged. A text block that opens
    with `{` (or a ```json fence) is read as the reply object: only the
    decoded characters of its "reply" string are handed out, and the rest is
    kept for finish(). Both can appear in one response -- a web search turn is
    "Let me look that up." in one text block and the object in a later one.

    The object may be split across several text blocks (web search answers
    arrive cut up by citations), so a new block never interrupts an object
    that is still open.

    ⚠ AN OBJECT CAN ALSO START IN THE MIDDLE OF PLAIN TEXT. Live, after an
    observe(): three plain sentences describing the room, then `{"reply": ...`
    in the SAME text block -- and before this was handled, the JSON itself went
    to Piper. So `{` followed by `"` is an object wherever it appears.

    ⚠ AND THEN THE ANSWER IS SAID TWICE. The plain sentences were already
    spoken by the time the object arrives, and its "reply" is the same answer
    again. Words cannot be unsaid, so two guards, both read only for the
    object's reply -- its emotion and memory_note are always kept:
      * an EXACT echo -- the fast path wrote "I'm sorry to hear that, Harsh."
        and then the object with the same words (2 of 4 live). The reply is
        held back while it still matches the plain text since the last tool,
        released the moment it differs, and dropped if it ends still matching.
      * a PARAPHRASED echo -- the gold price in plain words, then again in the
        object with the numbers spelled out. Nothing matches character for
        character, so an object after BRAIN_REPLY_ECHO_CHARS of plain text is
        not spoken at all. A short lead-in ("Here's what I found.") is under
        it, and the answer in the object is then still spoken.
    """

    def __init__(self) -> None:
        self._mode = "idle"            # idle | fence | brace | plain | json
        self._held = ""                # whitespace/fence held while idle
        self._plain_since_tool = 0     # non-space plain chars since a tool
        self._mute = False             # this object's reply is an echo
        self._since_tool: list[str] = []  # the plain text itself, since a tool
        self._echo_of = ""             # normalised plain text it may repeat
        self._holding: list[str] | None = None  # reply held while it matches
        self._gap = ""                  # a space owed after a tool call
        self._plain: list[str] = []
        self._objects: list[dict[str, Any]] = []
        self._spoken_json: list[str] = []
        self._reset_object()

    def _reset_object(self) -> None:
        self._raw: list[str] = []
        self._depth = 0
        self._in_string = False
        self._escape = ""              # "" | "\\" | "\\uXXXX" in progress
        self._surrogate = ""
        self._expect_key = False
        self._string: list[str] = []
        self._key: str | None = None
        self._string_is_key = False
        self._said: list[str] = []

    # -- input --------------------------------------------------------------

    def tool_boundary(self) -> None:
        """A non-text block began (a tool call or a search result). Plain
        text before it was a lead-in, not the answer."""
        self._plain_since_tool = 0
        self._since_tool = []
        if self._plain and not self._plain[-1][-1:].isspace():
            self._gap = " "            # "up." + "Today's" is one sentence
        self.new_block()

    def new_block(self) -> None:
        """A new text content block begins. Plain text may turn into an object
        here; an object that is still open simply carries on."""
        if self._mode == "plain":
            self._mode = "idle"

    def feed(self, text: str) -> str:
        out: list[str] = []
        for char in text:
            out.append(self._char(char))
        return "".join(out)

    def _char(self, char: str) -> str:
        if self._mode == "json":
            return self._json_char(char)
        if self._mode == "plain":
            if char == "{":
                self._held = char
                self._mode = "brace"
                return ""
            if char == "`":
                self._held = char
                self._mode = "fence"
                return ""
            return self._plain_char(char)
        if self._mode == "brace":
            self._held += char
            if not char.strip():
                return ""
            if char == '"':
                rest, self._held = self._held[1:], ""
                separator = self._begin_object()
                for held in rest:
                    self._json_char(held)
                return separator
            return self._to_plain()
        if self._mode == "fence":
            self._held += char
            if char == "{":
                self._held = ""
                return self._begin_object()
            # A fence that is not followed by an object was just backticks.
            if (len(self._held.strip()) > 12
                    or not _FENCE_SO_FAR.match(self._held.lstrip())):
                return self._to_plain()
            return ""
        # idle: hold whitespace until we know what this block is
        if not char.strip():
            self._held += char
            return ""
        if char == "{":
            self._held = ""
            return self._begin_object()
        if char == "`":
            self._held += char
            self._mode = "fence"
            return ""
        self._held += char
        return self._to_plain()

    def _plain_char(self, text: str) -> str:
        text, self._gap = self._gap + text, ""
        self._plain.append(text)
        self._since_tool.append(text)
        self._plain_since_tool += len("".join(text.split()))
        return text

    def _to_plain(self) -> str:
        self._mode = "plain"
        held, self._held = self._held, ""
        return self._plain_char(held)

    def _begin_object(self) -> str:
        """Start reading an object at its `{`. Returns the separator, if any.

        A space after a plain preamble, or "Let me look that up." and the
        answer arrive as "up.Tokyo" and the sentence splitter cannot see the
        boundary between them.
        """
        self._mode = "json"
        self._reset_object()
        self._mute = (self._plain_since_tool
                      >= getattr(config, "BRAIN_REPLY_ECHO_CHARS", 40))
        self._echo_of = _normal("".join(self._since_tool))
        self._holding = [] if (self._echo_of and not self._mute) else None
        self._json_char("{")
        before = "".join(self._plain)
        return " " if before and not before[-1].isspace() else ""

    def _json_char(self, char: str) -> str:
        self._raw.append(char)
        if self._in_string:
            return self._string_char(char)
        if char == '"':
            self._in_string = True
            self._string = []
            self._string_is_key = self._depth == 1 and self._expect_key
            return ""
        if char in "{[":
            self._depth += 1
            if char == "{" and self._depth == 1:
                self._expect_key = True
        elif char in "}]":
            self._depth -= 1
            if self._depth == 0:
                self._close_object()
        elif char == "," and self._depth == 1:
            self._expect_key = True
        return ""

    def _speaking(self) -> bool:
        return (not self._string_is_key and self._depth == 1
                and self._key == "reply")

    def _string_char(self, char: str) -> str:
        if self._escape:
            self._escape += char
            if self._escape.startswith("\\u"):
                if len(self._escape) < 6:
                    return ""
                try:
                    decoded = chr(int(self._escape[2:], 16))
                except ValueError:
                    decoded = ""
                self._escape = ""
                # Surrogate pairs (emoji) arrive as two \\u escapes.
                if "\ud800" <= decoded <= "\udbff":
                    self._surrogate = decoded
                    return ""
                if self._surrogate and "\udc00" <= decoded <= "\udfff":
                    decoded = (self._surrogate + decoded).encode(
                        "utf-16", "surrogatepass").decode("utf-16")
                self._surrogate = ""
                return self._decoded(decoded)
            if char == "u":
                return ""
            self._escape = ""
            return self._decoded(_ESCAPES.get(char, char))
        if char == "\\":
            self._escape = "\\"
            return ""
        if char == '"':
            if self._speaking() and self._holding is not None:
                # The whole reply matched what was already said: an echo.
                self._holding = None
            self._in_string = False
            value = "".join(self._string)
            if self._string_is_key:
                self._key = value
                self._expect_key = False
            return ""
        return self._decoded(char)

    def _decoded(self, text: str) -> str:
        self._string.append(text)
        if not self._speaking() or self._mute:
            return ""
        if self._holding is not None:
            self._holding.append(text)
            held = "".join(self._holding)
            if self._echo_of.startswith(_normal(held)):
                return ""                      # still the same words
            self._holding = None               # it differs: say it...
            text = held
            if _normal(held).startswith(self._echo_of):
                # ...but only what comes AFTER the words already said:
                # "Tokyo." then {"reply": "Tokyo. It is the capital."}.
                for cut in range(len(held) + 1):
                    if _normal(held[:cut]) == self._echo_of:
                        text = held[cut:]
                        break
        self._said.append(text)
        return text

    def _close_object(self) -> None:
        raw = "".join(self._raw)
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            self._objects.append(parsed)
        else:
            # Unparseable, but the reply was already spoken as it streamed.
            self._objects.append({"reply": "".join(self._said)})
        self._spoken_json.append("".join(self._said))
        self._mode = "idle"
        self._reset_object()

    # -- output -------------------------------------------------------------

    def flush_held(self) -> str:
        """End of the response: anything still held back that is speech.
        A fence or whitespace that never became an object is released as plain
        text; the caller's _spoken_only strips stray backticks."""
        if self._mode in ("idle", "fence") and self._held.strip():
            return self._to_plain()
        return ""

    def finish(self) -> Reply:
        """What the response amounted to.

        An object cut off before it closed (max_tokens) still counts: its reply
        was already spoken, so it is kept, with no emotion and no note.
        """
        if self._mode == "json" and self._said:
            self._objects.append({"reply": "".join(self._said)})
            self._spoken_json.append("".join(self._said))
        plain = "".join(self._plain).replace("`", "").strip()
        if not self._objects:
            return Reply(reply=plain)
        last = self._objects[-1]
        spoken = " ".join(s.strip() for s in [plain, *self._spoken_json]
                          if s.strip())
        return Reply(reply=spoken,
                     emotion=_clean_emotion(last.get("emotion")),
                     memory_note=_clean_note(last.get("memory_note")),
                     structured="emotion" in last or "memory_note" in last)


def parse(text: str) -> Reply:
    """A whole response at once -- the fast path, which is not streamed."""
    stream = ReplyStream()
    stream.new_block()
    stream.feed(text or "")
    stream.flush_held()
    return stream.finish()
