"""offline.py — the handful of things Friday can still do with no internet.

The brain lives behind the Anthropic API, so no network means no thinking. That
must not mean a robot that sits there saying nothing: the commands that matter
most in the room — drive, stop, turn, what time is it — are all things a dozen
lines of matching can handle without a model.

    intent = offline.understand("aage jao")
    if intent:                              # matched a known command
        intent.tool, intent.args            # -> "move", {...}
        intent.say                          # what to say while doing it

WHAT THIS IS NOT. It is not a fallback brain and it must never pretend to be
one. If it does not recognise something it returns None, main.py says so
plainly, and Friday keeps listening. Guessing would be worse than admitting the
network is down.

STOP IS ALWAYS CHECKED FIRST, online or off. If Harsh says stop, no round trip
to a model should stand between him and the motors halting.

HINDI AND ENGLISH BOTH. Harsh mixes them mid-sentence, so both are matched
against the same intent — "aage jao" and "go forward" are the same command.
"""

from __future__ import annotations

import logging
import re
import socket
import threading
import time
from dataclasses import dataclass, field

import config
import offline_extras

log = logging.getLogger("friday.offline")


@dataclass
class Intent:
    """One recognised command.

    `steps` carries a SEQUENCE of tool calls, because a directional move is
    never one call: the MCP server refuses move() unless observe() ran within
    the last few seconds, and moving ages that look -- so driving somewhere is
    observe, move, and for "go left" a pivot in between. That rule is enforced
    in the server, not here, and it is the reason Friday cannot be talked into
    driving blind. `tool`/`args` remain for the single-call commands.
    """
    name: str
    tool: str | None = None
    args: dict = field(default_factory=dict)
    say: str = ""
    steps: list = field(default_factory=list)
    # SOMETHING ONLY THE SPINE CAN DO, named rather than done. "Go to sleep"
    # has to reach into the wake detector, and a parser that imports
    # wake_word to pause it is a parser that cannot be tested without a
    # microphone. So the intent NAMES the action and main.py performs it,
    # which keeps this file what it says it is: text in, intent out.
    local: str = ""
    # WHETHER THE TOOL'S ANSWER IS THE ANSWER. "Go forward" is finished when
    # the wheels turn and `say` covers it; "how far is the wall" is not
    # finished until Friday says 53 centimetres. Without this the informational
    # commands ran the tool, logged the result and told Harsh nothing -- a
    # question answered into the log file.
    speak_result: bool = False

    def plan(self) -> list:
        """Every tool call this intent makes, in order."""
        if self.steps:
            return list(self.steps)
        return [(self.tool, dict(self.args))] if self.tool else []


# Matched in order, so the specific ones come before the general ones. Each
# pattern is searched, not fullmatch'd, because whisper adds punctuation and
# the wake word is often still on the front of the transcript.
_RULES: list[tuple[str, str, str | None, dict, str]] = [
    # name        pattern                                    tool          args                                    say
    ("stop",      r"\b(stop|halt|ruk|ruko|रुक|रुको|थम)\b",   "stop",       {},                                     "Stopping."),
    # "Friday, explore" -> the saved autonomous run. Harsh asked for the wake
    # word to be the switch for this, and it is in the OFFLINE table on purpose:
    # the whole point of firing an autonomous run by voice is that it works
    # without a round trip to a model that may be unreachable. It sits directly
    # under stop so that "stop exploring" can never be read as "explore".
    #
    # `explore` and `exploration` only. NOT "look around", which collides with
    # the "turn around" row two lines below, and not "discover", which whisper
    # produces for half a dozen unrelated phrases.
    ("explore",   r"\b(explore|exploring|exploration)\b",     "explore_and_return", {},                             ""),
    # English "turn around" only. Romanised "peeche mudo" still resolves to
    # backward -- the go/turn verb matching above the table wins over this row
    # -- and reversing when asked to turn around is a mild wrong answer, not a
    # dangerous one, so it is left rather than fixed with a special case.
    ("around",    r"\b(turn around|turn round|about turn|spin around|ghoom|peeche mud\w*|पीछे मुड़|घूम)\b", "turn_degrees", {"degrees": 180},                      "Turning around."),
    ("backward",  r"\b(back|backward|reverse|peeche|पीछे)\b", "move",      {"direction": "backward",
                                                                            "duration_seconds": 1.0},              "Going back."),
    ("forward",   r"\b(forward|ahead|aage|आगे|चल)\b",         "move",      {"direction": "forward",
                                                                            "duration_seconds": 1.0},              "Going forward."),
    ("left",      r"\b(left|baaye|baayen|बाय|बाएं|बायें)\b",  "turn_degrees", {"degrees": -90},                    "Turning left."),
    ("right",     r"\b(right|daaye|daayen|दाय|दाएं|दायें)\b", "turn_degrees", {"degrees": 90},                     "Turning right."),
    # ⚠ THE ASK, NOT THE WORD. This row used to be the bare word `time`
    # anywhere inside five words, so on 2026-09-19 a video saying "See you
    # next time." got the clock read out to it -- twice in ten minutes. A
    # reflex speaks without asking anybody, so it has to be the whole ask:
    # a question about the time, or the word on its own ("Friday, time?").
    ("time",      r"\b(what(?:'?s| is)?\s+(?:the\s+)?time|what time|"
                  r"time (?:is it|now)|tell me the time|the time\b|"
                  r"kitne baje|kya baje|baje h(?:ai|ain)?|kitna time|"
                  r"time kya|samay kya|kya samay|कितने बजे|समय क्या)\b"
                  r"|^\s*(?:the\s+)?(?:time|clock|samay|समय)[\s?.!।]*$",
                                                          None,        {},                                     ""),
]

_COMPILED = [(name, re.compile(pattern, re.IGNORECASE | re.UNICODE),
              tool, args, say)
             for name, pattern, tool, args, say in _RULES]

# The rows above that turn the wheels. See _STRICT_MAX_WORDS.
_MOVING_RULES = frozenset({"around", "backward", "forward", "left", "right"})

# The rows that DO something when a brain is there to be asked instead. Each
# one has to look like an order before it may skip that brain -- see
# _looks_like_an_order. `stop` is deliberately not here: it is reached through
# is_stop() on the always-on path, it is never refused, and a television that
# stops the wheels has done no harm.
_ORDERS_ONLY = _MOVING_RULES | {"explore"}


# --- directions, and how far ------------------------------------------------
# Separate from _RULES because a direction is not a fixed command: it carries
# an amount ("forward two seconds", "back a bit") and, for left and right, a
# choice between pivoting on the spot and actually going that way.
_DIRECTIONS = [
    ("forward",  r"\b(forward|ahead|straight|aage|आगे)\b"),
    ("backward", r"\b(back|backward|backwards|reverse|peeche|पीछे)\b"),
    ("left",     r"\b(left|baaye|baayen|बाय|बाएं|बायें)\b"),
    ("right",    r"\b(right|daaye|daayen|दाय|दाएं|दायें)\b"),
]
_DIRECTION_RE = [(n, re.compile(p, re.IGNORECASE | re.UNICODE))
                 for n, p in _DIRECTIONS]

# "go left" means travel that way; "turn left" means pivot and stop. Friday is
# differential drive, so travelling left is a pivot THEN a drive -- she cannot
# strafe, and pretending otherwise would just make her ignore half the words.
_GO_RE = re.compile(r"\b(go|move|drive|walk|come|chal|chalo|jao|चल|चलो|जाओ)\b",
                    re.IGNORECASE | re.UNICODE)
_TURN_RE = re.compile(r"\b(turn|rotate|spin|face|pivot|mudo|मुड़|मुडो)\b",
                      re.IGNORECASE | re.UNICODE)

# ORDER MATTERS: the specific words are tried first, and the bare articles
# last. With "a" checked first, "half a meter" matched the "a" and became a
# whole metre -- twice the distance asked for, which on a robot is the kind of
# rounding that ends in a wall.
_WORD_NUMBERS = [("half", 0.5), ("quarter", 0.25), ("one", 1), ("two", 2),
                 ("three", 3), ("four", 4), ("five", 5), ("six", 6),
                 ("seven", 7), ("eight", 8), ("nine", 9), ("ten", 10),
                 ("a", 1), ("an", 1)]


def _amount(text: str) -> tuple[float | None, str]:
    """How far, and in what unit. (None, "") when nothing was said."""
    number = None
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match:
        number = float(match.group(1))
    else:
        for word, value in _WORD_NUMBERS:
            if re.search(rf"\b{word}\b", text):
                number = value
                break
    if re.search(r"\b(cm|centimet|inch)\b", text):
        return (number or 30.0), "cm"
    if re.search(r"\b(met(er|re)s?|m)\b", text):
        return (number or 1.0) * 100.0, "cm"
    if re.search(r"\b(degrees?|deg)\b", text) and number:
        return number, "deg"
    if re.search(r"\b(seconds?|sec|s)\b", text) and number:
        return number, "s"
    if re.search(r"\b(a little|a bit|slightly|thoda|थोड़ा)\b", text):
        return 0.6, "s"
    if re.search(r"\b(a lot|far|long way|bahut|बहुत)\b", text):
        return 3.0, "s"
    return (number, "s") if number else (None, "")


def _direction_intent(heard: str) -> Intent | None:
    """A movement command, as the sequence of tool calls that carries it out."""
    found = next((n for n, r in _DIRECTION_RE if r.search(heard)), None)
    if found is None:
        return None
    going = bool(_GO_RE.search(heard))
    turning = bool(_TURN_RE.search(heard))
    size, unit = _amount(heard)

    # Bare "left" with no verb keeps the old meaning: pivot and stop.
    if found in ("left", "right") and not going:
        degrees = size if unit == "deg" and size else 90.0
        degrees = -degrees if found == "left" else degrees
        # observe() first here too: turn_degrees is under the same freshness
        # rule as move, so a bare "turn left" after a pause was refused for a
        # stale look exactly like "go left" was.
        return Intent(name=found, say=f"Turning {found}.",
                      steps=[("observe", {"camera": "up"}),
                             ("turn_degrees", {"degrees": degrees})])

    steps: list = []
    say = ""
    if found in ("left", "right"):
        # Pivot first, then drive: travelling left IS turning and going.
        degrees = -90.0 if found == "left" else 90.0
        # LOOK BEFORE THE PIVOT TOO. The server trusts an observation for a
        # few seconds only, and turn_degrees is under the same rule as move.
        # With the pivot first it was judged against whatever look happened
        # to be lying around -- 17s old in the case that found this -- so
        # "go left" had its turn refused every time and then drove forward.
        steps.append(("observe", {"camera": "up"}))
        steps.append(("turn_degrees", {"degrees": degrees}))
        say = f"Going {found}."
        drive = "forward"
    else:
        drive = found
        say = "Going forward." if found == "forward" else "Going back."

    # observe() before every move, because the server requires it and moving
    # ages it. This is the safety interrupt's precondition, not a formality.
    steps.append(("observe", {"camera": "up"}))
    if unit == "cm":
        distance = size if drive == "forward" else -abs(size)
        steps.append(("move_distance", {"cm": distance}))
    else:
        seconds = size if unit == "s" and size else 1.0
        seconds = max(0.2, min(seconds, 5.0))     # MAX_MOVE_SECONDS in config
        steps.append(("move", {"direction": drive,
                               "duration_seconds": seconds}))
    return Intent(name=f"go_{found}", say=say, steps=steps)


# --- the rest of the robot --------------------------------------------------
# EVERY MCP TOOL WORTH SAYING OUT LOUD, not just the four that move the wheels.
# The 37 tools on the server were reachable only through the brain, and the
# brain needs an API key and a network -- so in the state Friday actually runs
# in, "what do you see" and "set a timer for ten minutes" were not commands at
# all. They are now, and they work with the network unplugged.
#
# These are INFORMATIONAL or one-shot and none of them drives the robot, so
# they are matched more loosely than the movement rules above. A wrong match
# here costs a sentence; a wrong match on a direction costs a wall.
_ASK_RULES: list[tuple[str, str, str | None, dict, str]] = [
    # -- perception --
    ("see",        r"\b(what (do|can) you see|what's in front|whats in front|"
                   r"describe (the room|what you see)|look around|have a look|"
                   r"kya dikh)\b",                        "say_what_you_see", {},          ""),
    ("photo",      r"\b(take a (photo|picture|snap)|photo (lo|khich)|tasveer)\b",
                                                          "capture_image",    {},          "Taking a picture."),
    ("distance",   r"\b(how far|how close|what's the distance|clearance|"
                   r"kitni door|kitna door)\b",           "get_distance",     {},          ""),
    ("scan",       r"\b(scan (the room|around|environment)|what's around you|"
                   r"whats around you)\b",                "scan_environment", {},          "Having a look around."),

    # -- how she is --
    ("battery",    r"\b(battery|charge left|how much (power|charge)|"
                   r"kitni battery)\b",                   "get_battery",      {},          ""),
    ("status",     r"\b(status|report|diagnostics|self test|self-test)\b",
                                                          "get_status",       {},          ""),

    # -- the map --
    ("map_here",   r"\b(map (this |the )?(room|here|place)|make a map|"
                   r"build a map)\b",                     "map_here",         {},          "Mapping the room."),
    ("map_what",   r"\b(describe the map|what does the map|how big is the "
                   r"(map|room)|map summary|show me the map)\b",
                                                          "describe_map",     {},          ""),
    ("lidar",      r"\b(lidar|laser scan|do a sweep)\b",  "lidar_sweep",      {},          ""),
    # 2026-10-20: the rooms she can be sent to (File 14's places, which named
    # areas on the saved map become) -- so he knows what "go to ..." takes.
    ("places",     r"\b((which|what) (rooms|places|areas) (do you know|have you "
                   r"(saved|learned|learnt))|list (the |your |my )?(rooms|places|areas)|"
                   r"where can you go|kaun se (room|kamre))\b",
                                                          "list_places",      {"spoken": True}, ""),

    # -- autonomy --
    ("explore_status", r"\b(exploration status|are you (still )?exploring|"
                       r"how('s| is) the (run|exploration))\b",
                                                          "exploration_status", {},        ""),

    # -- the screen --
    ("alarms",     r"\b(what alarms|list (the )?alarms|any alarms|"
                   r"alarms (set|hain))\b",               "list_alarms",      {},          ""),
    # "all", because the tool wants an id from list_alarms and a keyword
    # parser has no way to know which one he means. Said out loud so he hears
    # what actually happened rather than assuming one alarm went.
    ("cancel",     r"\b(cancel (the |my )?(alarm|timer)|delete (the )?alarm|"
                   r"alarm (cancel|hatao))\b",            "cancel_alarm",     {"which": "all"},
                                                                                            "Cancelling your alarms and timers."),
    ("clock",      r"\b(show (me )?the clock|clock (dikhao|lagao)|"
                   r"put the clock (up|on))\b",           "show_clock",       {},          "Clock's up."),
    ("dismiss",    r"\b(clear the screen|dismiss (the|that)|close the (screen|app)|"
                   r"screen (band|saaf))\b",              "dismiss_screen",   {},          "Cleared."),
    ("screen",     r"\b(what('s| is) on (the )?screen|screen (pe|par) kya)\b",
                                                          "get_screen",       {},          ""),
    # THE ASK, NOT THE WORD -- the same rule the `time` row learned the hard
    # way. A bare `weather` keyword inside a sentence would put a card on the
    # panel every time a video mentioned the sky. weather_now fetches through
    # weather_source (wttr.in, no key, ~3 s) and draws the card, so the spoken
    # answer and the panel are the same reading.
    # FILE 09 STEP 6. Counts rows in a database she already wrote; no model,
    # no money, so it is a reflex rather than a brain turn.
    ("growth",     r"\b(how much have you (learned|grown)|what level are you|"
                   r"your (level|growth|progress|stats)|how many things "
                   r"(do you know|have you learned))\b",
                                                          "how_much_have_you_grown", {},   ""),
    # FILE 09 STEP 4, the read-back only. It reads a file she has already
    # written, so it costs nothing and cannot be made to spend money in a
    # loop; `reflect_now`, which DOES cost a call, is left to the brain and
    # to the dashboard button.
    ("learned",    r"\b(what (have|did) you learn(ed|t)?|what'?ve you learn|"
                   r"have you learn(ed|t)|anything (new )?you'?ve learn|"
                   r"kya seekha|seekha kya|your reflection)\b",
                                                          "what_have_you_learned", {},     ""),
    # ⚠ EVERY WAY HE ACTUALLY ASKS, because the ones it missed went to the
    # brain and the brain answered from the situation block -- the right
    # numbers, said out loud, with NO CARD. 2026-09-20 00:47: "I have a
    # headache, tell me the weather." -> spoken, nothing on the screen.
    # Harsh: "the weather voice command should show the animation when asked".
    ("weather",    r"\b(what('?s| is)? (the )?weather|how('?s| is) the weather|"
                   r"(tell|show) (me )?(the |you?r )?weather|weather "
                   r"(animation|card|screen|today|now|outside|report|like)|"
                   r"is it (going to )?rain(ing)?|how (hot|cold) is it|"
                   r"(what('?s| is)? the )?temperature (outside|today|now)|"
                   r"mausam (kaisa|kya|batao|dikhao)|kaisa mausam|"
                   r"मौसम कैसा|मौसम बताओ)\b|^\s*(the )?weather[\s?.!]*$",
                                                          "weather_now",      {},          ""),
]

_ASK_COMPILED = [(name, re.compile(pattern, re.IGNORECASE | re.UNICODE),
                  tool, args, say)
                 for name, pattern, tool, args, say in _ASK_RULES]


# --- things that carry a value ----------------------------------------------
# Each of these needs a number, a time or a phrase out of the sentence, so it
# cannot be a row in a table.

# The face's own vocabulary, from mcp_server/tools/expression.py. Only the
# FEELINGS -- listening/thinking/speaking belong to the pipeline and setting
# them by hand fights it.
_FEELINGS = ("happy", "love", "proud", "excited", "curious", "surprised",
             "sad", "wink", "angry", "neutral")
_FACE_RE = re.compile(
    r"\b(?:be|look|act|show me|show)\s+(?:a\s+)?(" + "|".join(_FEELINGS) +
    r")\b|\b(smile|wink)\b", re.IGNORECASE)

_SPEED_NAMED = [
    (r"\b(full speed|top speed|as fast as you can|full power)\b", 100),
    (r"\b(half speed|medium speed)\b", 50),
    (r"\b(slow(ly)? down|go slow(ly)?|slower|crawl|dheere)\b", 35),
    (r"\b(speed up|go faster|faster|tez)\b", 85),
]
_SPEED_NAMED_RE = [(re.compile(p, re.IGNORECASE | re.UNICODE), v)
                   for p, v in _SPEED_NAMED]
_SPEED_PERCENT_RE = re.compile(
    r"\b(?:speed|power)\s*(?:to|at|of)?\s*(\d{1,3})\s*(?:%|percent)?\b|"
    r"\b(\d{1,3})\s*(?:%|percent)\s*(?:speed|power)?\b", re.IGNORECASE)

_ALARM_RE = re.compile(
    r"\b(?:set (?:an?|the) alarm|wake me(?: up)?|alarm)\b\s*"
    r"(?:for|at|to)?\s*(.+)", re.IGNORECASE)
_TIMER_RE = re.compile(
    r"\b(?:set (?:a|the) timer|start a timer|timer|remind me in|"
    r"countdown)\b\s*(?:for|of)?\s*(.+)", re.IGNORECASE)
_REMEMBER_RE = re.compile(r"\bremember (?:that |this[:,]? )?(.+)",
                          re.IGNORECASE)
_RECALL_RE = re.compile(
    r"\b(?:what do you (?:know|remember) about|do you remember|recall)\s+"
    r"(.+)", re.IGNORECASE)

# "Not now." A wake word that fires on a conversation makes this necessary,
# but it stays useful after the model is fixed -- sometimes the answer to a
# robot is simply not yet.
_SLEEP_RE = re.compile(
    r"\b(go to sleep|stop listening|don'?t listen|be quiet|leave me alone|"
    r"do not disturb|shut up|so ?jao|chup (raho|ho jao)|sun'?na band)\b",
    re.IGNORECASE | re.UNICODE)
_HELP_RE = re.compile(
    r"\b(what can you do|what commands|list (your )?commands|help me out|"
    r"^help$|kya kar sakti)\b", re.IGNORECASE | re.UNICODE)

# FILE 15: CONTINUOUS NARRATION, ON AND OFF. Only his own words turn it on
# (Harsh: "when voice command given to it then only") -- there is no brain
# tool that can, so a garbled line the brain half-understood never starts a
# robot that talks every few seconds. The OFF phrases are checked first, and
# above the stop rule, so "stop describing" ends the narration and is never
# read as "keep describing" or as a motor stop. A bare "describe what you
# see" is NOT here: that is one look, and goes to the brain.
_NARRATE_OFF_RE = re.compile(
    r"\b((stop|quit|end|cancel|enough|pause)( the| your| with the)? "
    r"(describing|narrating|narration|descriptions?|commentary)|"
    r"(don'?t|do not|no more) (keep )?(describe|describing|narrate|narrating)|"
    r"(narration|narrating|describing|description|commentary) (mode )?off|"
    r"turn off (the )?(narration|describing|description|commentary))\b",
    re.IGNORECASE)
_NARRATE_ON_RE = re.compile(
    r"\b((keep|start|begin|continue|carry on|go on) (describing|narrating)|"
    r"keep (telling|saying) (me )?what you (can )?see|"
    r"(describe|narrate) (everything|what you (can )?see|the (room|scene|view))"
    r" (continuously|constantly|all the time|as it changes|as you go)|"
    r"(narration|narrate|describing|description|commentary) mode( on)?|"
    r"turn on (the )?(narration|describing|description|commentary)|"
    r"continuous (description|narration|describing|commentary))\b",
    re.IGNORECASE)

# "Stop exploring" is not "stop". Both halt the wheels, but only one of them
# ends the autonomous run, and the run restarts the wheels a second later --
# so a bare stop() against a live exploration looks like the robot ignoring
# him. Checked before the stop rule, and it does BOTH.
# ASKING IS NOT COMMANDING. Beside _STOP_EXPLORE_RE because it is the same
# hazard from the other side: these are QUESTIONS about the run, and every one
# of them contains the word the command matcher below fires on.
_EXPLORE_STATUS_RE = re.compile(
    r"\b(exploration status|are you (still )?exploring|"
    r"how('s| is) the (run|exploration))\b", re.IGNORECASE)

_STOP_EXPLORE_RE = re.compile(
    r"\b(stop|end|cancel|abort|band)\b[^.]*\b(explor\w*|run|roaming|"
    r"wandering)\b|\bstop the run\b", re.IGNORECASE | re.UNICODE)


# A leading "Friday," is not part of the command. openWakeWord fires at the
# END of the phrase, so the wake word itself is usually behind the capture --
# but on a borderline score it fires early and whisper puts it on the front.
_ADDRESS_RE = re.compile(
    r"^\s*(?:ok(?:ay)?\s+|hey\s+|hi\s+)?friday[\s,.!]*|^\s*please\s+",
    re.IGNORECASE)


# HINDI PUTS THE VERB LAST. "thoda sa aage chalo" is as much an order as "go
# forward a bit" -- the verb is only at the other end of the sentence. Hindi
# verbs only: an English sentence that ENDS in "go" ("that's where the left
# one should go") usually is not an order.
_HINDI_ORDER_RE = re.compile(r"(chal|chalo|jao|mudo|चल|चलो|जाओ|मुड़|मुडो)",
                             re.IGNORECASE | re.UNICODE)

# "Can you please turn left now" is an order said politely, and the verb sits
# behind the politeness. Stripped from the FRONT only, and only these words:
# "can you believe it's on the left" still starts with "believe" without them.
_POLITE_RE = re.compile(
    r"^\s*(?:(?:can|could|will|would)\s+you\s+)?(?:please\s+)?", re.IGNORECASE)

# "Explore the room" is an order with its verb in front, exactly like "go
# forward"; "we explore the caves later" is a sentence about exploring.
_EXPLORE_VERB_RE = re.compile(r"(explore|exploring|explorer)",
                              re.IGNORECASE | re.UNICODE)


def _is_imperative(heard: str) -> bool:
    """Does this sentence START with an order -- or, in Hindi, end with one?

    THE POSITION OF THE VERB IS THE SIGNAL, not its presence. "I think we
    should go right now" contains a motion verb and a direction and is not a
    command; "go forward two meters" is the same two ingredients in the order
    that makes them one. Anything that merely CONTAINS the words is a sentence
    about going right, and belongs to the brain.
    """
    words = _POLITE_RE.sub("", _ADDRESS_RE.sub("", heard)).split()
    if not words:
        return False
    first = words[0].strip(".,!?")
    last = words[-1].strip(".,!?।")
    return bool(_GO_RE.fullmatch(first) or _TURN_RE.fullmatch(first)
                or _EXPLORE_VERB_RE.fullmatch(first)
                or _HINDI_ORDER_RE.fullmatch(last))


# THE WORDS AN ORDER IS ALLOWED TO BE MADE OF. Everything here is the command
# itself, its amount, or filler that changes nothing: the verb, the direction,
# a number and its unit, "a bit", "now". What is left over after taking them
# out is the sentence the words were sitting in -- and a sentence about going
# right is not an order to go right.
_ORDER_WORDS_RE = re.compile(
    r"\b(go|going|move|moving|drive|driving|walk|come|turn|turning|rotate|"
    r"spin|pivot|face|start|begin|karo|kar|chal|chalo|jao|mudo|"
    r"forward|forwards|ahead|straight|back|backward|backwards|reverse|"
    r"left|right|around|round|explore|exploring|exploration|"
    r"now|just|then|ok|okay|and|to|for|a|an|the|bit|little|lot|slightly|far|"
    r"minute|minutes|min|mins|"
    r"degree|degrees|deg|second|seconds|sec|metre|metres|meter|meters|m|cm|"
    r"centimetre|centimetres|centimeter|centimeters|inch|inches|"
    r"half|quarter|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"aage|peeche|baaye|baayen|daaye|daayen|thoda|sa|bahut|"
    r"आगे|पीछे|बाय|बाएं|बायें|दाय|दाएं|दायें|थोड़ा|बहुत|चल|चलो|जाओ|मुड़|मुडो)\b",
    re.IGNORECASE | re.UNICODE)


def _bare_order(heard: str) -> bool:
    """Is this the order and nothing else -- "forward", "a bit to the left"?"""
    rest = _ORDER_WORDS_RE.sub(
        " ", _POLITE_RE.sub("", _ADDRESS_RE.sub("", heard)))
    return not re.sub(r"[\d\s.,!?;:\-\u0964]+", " ", rest).strip()


def _looks_like_an_order(heard: str) -> bool:
    """Would a person hearing this say it was an order, not a sentence?

    ⚠ THE LENGTH WAS NEVER THE TEST, AND ON 2026-09-19 IT SHOWED. The wheels
    and the autonomous run were reflexes as soon as a direction word appeared
    inside five words, and a video in the room said "F right A, speak Hindi."
    -- five words -- so she pivoted ninety degrees at a television. The same
    night "Now go back to it." drove her backwards.

    In STRICT mode -- the path where a brain is available and this runs first
    -- refusing costs nothing but a round trip to something that understands
    the sentence, while acting costs a pivot nobody asked for. So the order
    has to look like one at any length: the verb in front (_is_imperative), or
    nothing in the sentence but the order itself (_bare_order).
    """
    return _is_imperative(heard) or _bare_order(heard)


def _clean_tail(text: str) -> str:
    """The part of the sentence after a command word, tidied for a tool.

    Whisper's punctuation and a trailing "please" are not part of a time or a
    fact, and app_control.parse_when raises on the ones it cannot read.
    """
    tail = re.sub(r"\b(please|thanks|thank you|for me|na|yaar)\b", " ", text,
                  flags=re.IGNORECASE)
    return tail.strip(" .,!?;:-\u0964").strip()


# --- FILE 14: navigation, with no model at all -------------------------------
#
# Harsh, 2026-09-23: "no ai will be used all behaviour in offline model". So
# "go to the kitchen", "go home", "come back", "try again" and "explore and
# come back" are matched here and handed to the navigation tools directly --
# online or off, the brain is never asked (they are all in _REFLEX_INTENTS).
#
# ⚠ CHECKED FIRST IN _match, AND THAT ORDER IS LOAD BEARING, in both
# directions:
#   * ahead of `_local_intent`, whose explore parser claims any sentence with
#     "explore" in it -- "explore and come back" would be a plain run with no
#     way back;
#   * ahead of the "backward" row, whose `\bback\b` would make "come back"
#     REVERSE the robot;
#   * but BEHIND stop and sleep, which `_nav_intent` checks for itself and
#     yields to: "stop, don't come back" must stop, and "go to sleep" is not
#     a place.
#
# Every rule is anchored to the START of the order (after "Friday," and "can
# you please"), because these start journeys that last minutes. "You should
# come back tomorrow" on the television is not an order.

_NAV_LEAD = r"^(?:(?:ok(?:ay)?|now|so|and|abhi|chalo)[\s,]+)*"

_NAV_TRY_AGAIN_RE = re.compile(
    _NAV_LEAD + r"(?:try (?:it )?again|try once more|one more try|"
    r"resume(?: (?:navigation|driving|the route))?|"
    r"(?:phir|fir|dobara) se (?:try|koshish)(?: karo| kar)?|"
    r"(?:phir|fir|dobara) (?:try|koshish) karo|"
    r"फिर से (?:कोशिश|ट्राई) करो|दोबारा कोशिश करो)[\s.!?]*$",
    re.IGNORECASE | re.UNICODE)

_NAV_HOME_RE = re.compile(
    _NAV_LEAD + r"(?:(?:go|come|head|drive|get|return)(?: back)? home|"
    r"return to home|(?:go|head|drive|return)(?: back)? to (?:your |the )?dock|"
    r"(?:go )?dock yourself|ghar(?: wapas| vapas)? (?:chalo|chal|jao|jaao|aao)|"
    r"घर (?:वापस )?(?:चलो|जाओ|आओ))\b",
    re.IGNORECASE | re.UNICODE)

_NAV_COME_BACK_RE = re.compile(
    _NAV_LEAD + r"(?:come back(?: here| to me| to where you started| to the "
    r"start)?|return to (?:me|where you started|the start|your start)|"
    r"(?:wapas|vapas) (?:aao|aa jao|aa ja)|वापस आओ|वापस आ जाओ)[\s.!?]*$",
    re.IGNORECASE | re.UNICODE)

_NAV_GO_TO_RE = re.compile(
    _NAV_LEAD + r"(?:go|drive|head|navigate|move|walk|take me|get)"
    r"(?: over| back| straight)? to (?P<rest>.+)$",
    re.IGNORECASE | re.UNICODE)

# Hindi puts the verb last: "kitchen chalo", "the desk tak jao".
_NAV_HINDI_GO_RE = re.compile(
    r"^(?P<rest>.+?)\s+(?:(?:mein|me|main|par|pe|tak|ki taraf)\s+)?"
    r"(?:chalo|chal|jao|jaao|chale jao)[\s.!?]*$",
    re.IGNORECASE | re.UNICODE)

_NAV_BACK_TAIL_RE = re.compile(
    r"\b(?:come back|coming back|return|and back|wapas|vapas|वापस)\b",
    re.IGNORECASE | re.UNICODE)

# "Remember this as the kitchen" -- saving WHERE SHE IS under a name. Without
# it the navigation commands above could only go to places a brain had saved.
# It must say "as" (or "is", for "this spot is the kitchen"): "remember this,
# the exam is on Monday" is a memory note and stays with the remember rule.
_NAV_SAVE_PLACE_RE = re.compile(
    _NAV_LEAD + r"(?:(?:remember|save|mark|name|call)\s+"
    r"(?:this|here|this place|this spot|this position|where (?:i|you) "
    r"(?:am|are)(?: standing)?)\s+(?:as|is)\s+|"
    r"this (?:place|spot) is (?:called )?|"
    r"(?:is|yeh|ye|iss|is) jagah(?: ko)?\s+)"
    r"(?P<name>[^,.!?]+?)"
    r"(?:\s+(?:bolo|kaho|yaad rakho|naam do|hai|save karo))?[\s.!?]*$",
    re.IGNORECASE | re.UNICODE)

# Words that are never a place, so "go to the left" and "go to bed" stay
# with the rules that own them.
_NOT_A_PLACE = frozenset({
    "left", "right", "forward", "back", "backward", "front", "sleep", "bed",
    "the left", "the right", "the front", "the back", "it", "him", "her",
    "them", "me", "you", "there", "here", "work", "school",
})


def _known_places() -> list[str]:
    """Saved place names, longest first, from the places table. Never guessed."""
    try:
        from slam import places                 # noqa: PLC0415
        return sorted((row["name"] for row in places.all()), key=len,
                      reverse=True)
    except Exception:                           # noqa: BLE001
        return []


def _split_place(rest: str) -> tuple[str, str, bool]:
    """(place, the rest of the sentence, is it a place she knows)."""
    rest = rest.strip(" .,!?")
    for name in _known_places():
        bare = name[4:] if name.startswith("the ") else name
        m = re.match(r"(?:the\s+)?" + re.escape(bare) + r"\b", rest,
                     re.IGNORECASE)
        if m:
            return name, rest[m.end():], True
    # Not one she knows. Take the words up to the first joint as the name, so
    # the tool can say plainly that she does not know it -- never a guess.
    head = re.split(r"\s*(?:,|\band\b|\bthen\b|\baur\b)\s*", rest, 1)
    place = head[0].strip()
    tail = rest[len(place):]
    return place, tail, False


def _nav_tail(tail: str) -> tuple[bool | None, str]:
    """"and come back" / "and tell me the time" after the place."""
    come_back = True if _NAV_BACK_TAIL_RE.search(tail) else None
    then = _NAV_BACK_TAIL_RE.sub(" ", tail)
    then = re.sub(r"\b(?:and|then|aur|here|to me|after(?:wards)?)\b", " ",
                  then, flags=re.IGNORECASE)
    then = _clean_tail(" ".join(then.split()))
    return come_back, then


def _nav_intent(heard: str, strict: bool = False) -> Intent | None:
    """A navigation order, or None. See the block comment above."""
    text = _POLITE_RE.sub("", _ADDRESS_RE.sub("", heard)).strip(" .,!?")
    if not text:
        return None
    # Stop, sleep and questions about the run always win.
    if (_COMPILED[0][1].search(text) or _STOP_EXPLORE_RE.search(text)
            or _SLEEP_RE.search(text) or _EXPLORE_STATUS_RE.search(text)):
        return None
    words = len(text.split())

    # "Explore and come back" -- a run with the way back armed.
    # Up to 20 words: "go explore for two minutes and come back to the spot
    # where you started" is 15, and at the old limit of 12 it fell through to
    # a plain run with no way back (2026-09-23 23:52).
    if (_EXPLORE_VERB_RE.search(text) and _NAV_BACK_TAIL_RE.search(text)
            and words <= 20):
        if strict and not _looks_like_an_order(heard):
            return None
        try:
            import exploration_mode             # noqa: PLC0415
            minutes = exploration_mode.parse_request(text)
        except Exception:                       # noqa: BLE001
            minutes = None
        args = {}
        if minutes is not None:
            args["minutes"] = minutes
        return Intent(name="explore_return", tool="explore_and_return",
                      args=args)

    if words <= 6 and _NAV_TRY_AGAIN_RE.search(text):
        return Intent(name="try_again", tool="try_again")

    m = _NAV_SAVE_PLACE_RE.search(text)
    if m and words <= 10:
        name = " ".join(m.group("name").lower().split())
        if name and len(name.split()) <= 4 and name not in _NOT_A_PLACE:
            return Intent(name="remember_place", tool="remember_place",
                          args={"name": name})

    m = _NAV_GO_TO_RE.search(text)
    if m:
        place, tail, known = _split_place(m.group("rest"))
        # THE DOCK IS HOME (home_dock.py): "go to your dock" is "go home".
        if re.fullmatch(r"(?:your |the |my )?(?:dock|docking station|home)",
                        (place or "").strip().lower()) \
                and not tail.strip(" .,!?"):
            return Intent(name="nav_home", tool="nav_home")
        if place and place.lower() not in _NOT_A_PLACE and (
                known or len(place.split()) <= 3):
            come_back, then = _nav_tail(tail)
            args = {"place": place}
            if come_back is not None:
                args["come_back"] = True
            if then:
                args["then"] = then
            return Intent(name="nav_go", tool="nav_go", args=args)

    if words <= 6 and _NAV_HOME_RE.search(text):
        return Intent(name="nav_home", tool="nav_home")

    # Hindi, verb last -- only for a place she KNOWS, because "chalo" ends a
    # great many sentences that are not about going anywhere.
    m = _NAV_HINDI_GO_RE.search(text)
    if m and words <= 8:
        place, tail, known = _split_place(m.group("rest"))
        if known and not tail.strip(" .,!?"):
            return Intent(name="nav_go", tool="nav_go", args={"place": place})

    if words <= 6 and _NAV_COME_BACK_RE.search(text):
        return Intent(name="nav_come_back", tool="nav_come_back")
    return None


_BRIEFING_RE = re.compile(
    r"\b(?:daily briefing|my briefing|morning briefing|brief me|"
    r"what(?:'s| is) my day(?: look like)?|how(?:'s| does) my day look)\b",
    re.IGNORECASE)


_REMEMBER_FOREVER_RE = re.compile(
    r"^(?:please\s+)?(?:always remember|never forget|remember forever|"
    r"remember this forever|don'?t ever forget|do not ever forget|"
    r"never ever forget)(?:\s+that|\s+this)?[:,]?\s+(?P<fact>.{3,})$",
    re.IGNORECASE)
_REMEMBER_EXPLICIT_RE = re.compile(
    r"^(?:please\s+)?(?:remember|note|make a note)\s+(?:that\s+|this[:,]?\s+)"
    r"(?P<fact>.{3,})$", re.IGNORECASE)
_FORGET_RE = re.compile(
    r"^(?:please\s+)?(?:forget|delete|erase|remove)\s+"
    r"(?:what i (?:said|told you) about|that|everything (?:i said )?about|"
    r"the (?:memory|fact|note) (?:about|that)|my (?:memory|note) about|about)"
    r"\s+(?P<what>.{2,})$", re.IGNORECASE)
# THE WHOLE SENTENCE IS THE ANSWER. "yes please turn left" while a delete
# question is open is an order, not a yes -- fullmatch, with only filler words
# allowed around the answer.
_PENDING_ANSWER_RE = re.compile(
    r"(?:(?:yes|yeah|yep|yup|sure|ok(?:ay)?|haan|confirm)"
    r"(?:\s+(?:please|delete it|do it|go ahead|forget it))?"
    r"|delete it|do it|go ahead|forget it"
    r"|(?:no|nope|nahi|cancel|don'?t|never ?mind|keep it)"
    r"(?:\s+(?:thanks|thank you|keep it|don'?t delete it))?"
    r"|(?:the\s+)?(?:number\s+)?(?:first|second|third|fourth|fifth|one|two|"
    r"three|four|five|[1-5])(?:\s+one)?)", re.IGNORECASE)


def _third_person(fact: str) -> str:
    """"my sister is Priya" -> "Harsh's sister is Priya". Memories are kept in
    the third person (the remember tool's own rule); saved as heard, "my
    sister" would later read as HER sister."""
    name = getattr(config, "OWNER_NAME", "Harsh")
    for pattern, repl in ((r"\bI am\b|\bI'm\b", f"{name} is"),
                          (r"\bI have\b|\bI've\b", f"{name} has"),
                          (r"\bmy\b|\bmine\b", f"{name}'s"),
                          (r"\bmyself\b|\bme\b|\bI\b", name)):
        fact = re.sub(pattern, repl, fact, flags=re.IGNORECASE)
    return fact


_LAPTOP = r"(?:my|the)\s+(?:laptop|lap top|computer)"
_YT = r"(?:youtube|you tube)"
# (pattern, site, play) -- first match wins; the query is group "q".
_LAPTOP_SEARCH = [
    (rf"^(?:play|put on)\s+(?P<q>.+?)\s+on\s+{_YT}(?:\s+on\s+{_LAPTOP})?$",
     "youtube", True),
    (rf"^(?:open\s+)?{_YT}\s+(?:and\s+)?play\s+(?P<q>.+?)(?:\s+on\s+{_LAPTOP})?$",
     "youtube", True),
    (rf"^(?:search|look up|find)\s+(?:for\s+)?(?P<q>.+?)\s+on\s+{_YT}"
     rf"(?:\s+on\s+{_LAPTOP})?$", "youtube", False),
    (rf"^(?:open\s+)?{_YT}\s+(?:and\s+)?(?:search|look up|find)\s+(?:for\s+)?"
     rf"(?P<q>.+?)(?:\s+on\s+{_LAPTOP})?$", "youtube", False),
    (rf"^(?:search|google|look up)\s+(?:for\s+)?(?P<q>.+?)\s+on\s+{_LAPTOP}$",
     "web", False),
    (rf"^on\s+{_LAPTOP},?\s+(?:search|google|look up)\s+(?:for\s+)?(?P<q>.+)$",
     "web", False),
    (r"^google\s+(?:for\s+)?(?P<q>.+)$", "web", False),
]
_LAPTOP_SEARCH = [(re.compile(p, re.IGNORECASE), site, play)
                  for p, site, play in _LAPTOP_SEARCH]
_NOTES_RE = re.compile(
    r"^(?:please\s+)?(?:(?:(?:make|take|write|prepare|create)\s+(?:me\s+)?"
    r"(?:some\s+)?notes|notes)\s+(?:on|about)"
    r"|(?:properly|deeply|thoroughly)\s+research"
    r"|(?:do|run)\s+(?:a\s+|some\s+)?(?:deep|proper|full|thorough)\s+research"
    r"\s+(?:on|into|about)|deep\s+research\s+(?:on|into|about))"
    rf"\s*[:,]?\s*(?P<q>.{{3,}}?)(?:\s+on\s+{_LAPTOP})?$", re.IGNORECASE)
_NOTES_TAIL_RE = re.compile(
    rf"^research\s+(?P<q>.{{3,}}?)(?:[,\s]+(?:properly|in depth|thoroughly)"
    rf"|\s+on\s+{_LAPTOP})$", re.IGNORECASE)


# "PLAY A SONG" with no name (2026-10-20: "hey Friday play song", ignored 4
# times -- only "play X on YouTube" existed, and whisper wrote "Play Asong").
_PLAY_ANY_RE = re.compile(
    r"^(?:play|put\s+on|start)\s+(?:a\s+|some\s+|any\s+|me\s+a\s+)?"
    r"(?:song|songs|music|gaana|gaane|gana)(?:\s+please)?"
    rf"(?:\s+on\s+{_LAPTOP})?$"
    r"|^(?:gaana|gaane|song|music)\s+(?:bajao|chalao|lagao)$", re.IGNORECASE)
# What whisper makes of "play a song" when he is a few metres away
# (2026-10-20 22:31-22:35, all ignored: "Blair Assong", "Blear song.", "Bleed
# during that song."): a short sentence that starts like "play" and ends in
# "song". Only after a wake, so a real "I like that song" needs "hey Friday"
# in front of it too.
_PLAY_SONG_HEARD_RE = re.compile(
    r"^(?:pl|bl|pr|fl|cl)[a-z]*\b[a-z' ]{0,24}?songs?[.!?]*$", re.IGNORECASE)
_PLAY_SONG_HEARD_MAX_WORDS = 5
# "play <name>" without "on YouTube" -- the song is the point, YouTube is
# where songs are. Not a game, not "play with me", not "play it again".
_PLAY_NAMED_RE = re.compile(
    rf"^(?:play|put\s+on)\s+(?P<q>.+?)(?:\s+on\s+{_LAPTOP})?$", re.IGNORECASE)
_PLAY_NOT_A_SONG = re.compile(
    r"^(?:a\s+game|games?|with\s+me|it\s+again|again|dead|catch|chess|cards?"
    r"|the\s+next\s+\w+|next|pause|it|that)$", re.IGNORECASE)
_PLAY_MAX_WORDS = 8

# ONE KEY on his laptop (helper 4, POST /key) -- (pattern, action).
_MEDIA = r"(?:the\s+|this\s+)?(?:video|song|music|track|youtube|movie|it)"
_LAPTOP_KEYS = [
    (rf"^(?:pause|resume|unpause|continue)(?:\s+{_MEDIA})?(?:\s+on\s+{_LAPTOP})?$", "play_pause"),
    # "stop" only when the media is NAMED: "stop it" / "stop that" stop HER
    # (test_orders.py -- they did, until this table took "it" for a video).
    (rf"^stop\s+(?:the\s+|this\s+)?(?:video|song|music|track|youtube|movie)"
     rf"(?:\s+on\s+{_LAPTOP})?$", "play_pause"),
    (rf"^(?:play\s+)?(?:the\s+)?next\s+(?:video|song|track|one)(?:\s+on\s+{_LAPTOP})?$"
     rf"|^skip(?:\s+{_MEDIA})?$", "next"),
    # "play a better song, change it" was searched for, word for word (22:30).
    (r"^(?:play\s+)?(?:a\s+)?(?:another|different|better|new|some\s+other)\s+"
     r"(?:song|one|video|track)(?:[,\s]+(?:change\s+it|please))?$"
     r"|^change\s+(?:the\s+|this\s+)?(?:song|video|track|it)$", "next"),
    (r"^(?:play\s+)?(?:the\s+)?(?:previous|last)\s+(?:video|song|track)$", "previous"),
    (rf"^(?:turn\s+(?:the\s+)?)?(?:laptop\s+)?volume\s+up(?:\s+on\s+{_LAPTOP})?$"
     rf"|^(?:increase|raise)\s+(?:the\s+)?(?:laptop\s+)?volume(?:\s+on\s+{_LAPTOP})?$"
     rf"|^(?:make\s+it\s+)?louder\s+on\s+{_LAPTOP}$|^turn\s+it\s+up$", "volume_up"),
    (rf"^(?:turn\s+(?:the\s+)?)?(?:laptop\s+)?volume\s+down(?:\s+on\s+{_LAPTOP})?$"
     rf"|^(?:decrease|lower|reduce)\s+(?:the\s+)?(?:laptop\s+)?volume(?:\s+on\s+{_LAPTOP})?$"
     rf"|^(?:make\s+it\s+)?quieter\s+on\s+{_LAPTOP}$|^turn\s+it\s+down$", "volume_down"),
    (rf"^(?:un)?mute(?:\s+(?:the\s+)?(?:laptop|sound|video|music|volume))?(?:\s+on\s+{_LAPTOP})?$",
     "mute"),
    (r"^(?:make\s+it\s+|go\s+|put\s+it\s+(?:in\s+|on\s+)?)?full\s*-?\s*screen$"
     r"|^(?:exit|leave)\s+full\s*-?\s*screen$", "fullscreen"),
    (r"^close\s+(?:the|this|that)\s+tab$", "close_tab"),
    (rf"^lock\s+(?:{_LAPTOP}|(?:my|the)\s+screen)$", "lock"),
    (r"^open\s+(?:my\s+)?(?:friday\s+)?notes(?:\s+folder)?$", "open_notes"),
]
_LAPTOP_KEYS = [(re.compile(p, re.IGNORECASE), a) for p, a in _LAPTOP_KEYS]

# APPS on his laptop (helper 5, 2026-10-20: "open whatsapp, open this game ...
# she knows what's in my laptop"). The NAME goes to the laptop, which matches
# it against its own Start menu; nothing else is sent.
_OPEN_APP_RE = re.compile(
    rf"^(?:open|launch|start up|run)\s+(?P<q>.{{2,40}}?)(?:\s+(?:app|application))?"
    rf"(?:\s+on\s+{_LAPTOP})?$", re.IGNORECASE)
_NOT_AN_APP = re.compile(r"^(?:the\s+)?(?:door|window|your eyes|eyes|map|dashboard|"
                         r"face|camera|mouth)$", re.IGNORECASE)
_LIST_GAMES_RE = re.compile(
    r"^(?:what|which)\s+games?\s+(?:do\s+i\s+have|are\s+(?:there\s+)?on\s+"
    rf"{_LAPTOP}|have\s+i\s+got|can\s+i\s+play)|^(?:list|show)\s+(?:me\s+)?my\s+games$",
    re.IGNORECASE)
_LIST_APPS_RE = re.compile(
    rf"^what(?:'?s| is)\s+(?:on|in)\s+{_LAPTOP}$|^(?:what|which)\s+apps\s+(?:do\s+i\s+have|"
    rf"are\s+on\s+{_LAPTOP})|^what\s+do\s+i\s+have\s+on\s+{_LAPTOP}$", re.IGNORECASE)
# HER playing a slow game, on his word only (laptop_games.py, 2026-10-20).
_YOU_PLAY_RE = re.compile(
    r"^(?:(?:now\s+)?you\s+play(?:\s+(?:the|this|a)\s+game|\s+it|\s+(?P<g>.{2,30}?))?"
    r"|play\s+(?:it|this|the\s+game|(?P<g2>.{2,30}?))\s+(?:yourself|for\s+me)"
    r"|(?:it'?s\s+)?your\s+turn|can\s+you\s+play\s+(?:it|this|the\s+game))$",
    re.IGNORECASE)
# Recording HIS driving, to teach her (2026-10-21). "stop recording" is NOT
# here: it is the stop rule (a stop must stay a stop), and the stop tool ends
# a recording too.
_RECORD_RE = re.compile(
    r"^(?:(?:start\s+)?record(?:ing)?\s+(?:my|me|the|how\s+i)\s+(?:driv(?:e|ing)|game(?:play)?|play(?:ing)?)"
    r"|start\s+recording(?:\s+(?:my|the)\s+(?:driving|game|gameplay))?"
    r"|(?:watch|learn)\s+(?:how\s+)?i\s+drive)$", re.IGNORECASE)
_STOP_PLAYING_RE = re.compile(
    r"^(?:stop\s+playing(?:\s+the\s+game)?|i'?ll\s+play(?:\s+now)?|my\s+turn|"
    r"give\s+(?:it|me\s+the\s+game)\s+back|let\s+me\s+play)$", re.IGNORECASE)
_SUGGEST_RE = re.compile(
    r"^(?:suggest\s+(?:me\s+)?(?:a|some)\s+game|what\s+should\s+i\s+play|"
    r"i'?m\s+(?:so\s+)?bored|i\s+am\s+(?:so\s+)?bored|main\s+bore\s+ho\s+raha\s+(?:hoon|hu)|"
    r"give\s+me\s+a\s+game)$", re.IGNORECASE)


def _laptop_intent(heard: str) -> Intent | None:
    """File 20: his laptop -- a search on its screen, or notes put there."""
    text = _ADDRESS_RE.sub("", heard).strip(" .!?")
    # whisper writes "play a song" as "Play Asong" (2026-10-20)
    text = re.sub(r"\basong\b", "a song", text, flags=re.IGNORECASE)
    for pattern, action in _LAPTOP_KEYS:
        if pattern.match(text):
            return Intent(name="laptop_key", tool="laptop_control",
                          args={"action": action})
    for pattern in (_NOTES_RE, _NOTES_TAIL_RE):
        match = pattern.match(text)
        if match and len(match.group("q").strip(" ,.?")) >= 3:
            return Intent(name="laptop_notes", tool="laptop_notes",
                          args={"topic": match.group("q").strip(" ,.?")})
    for pattern, site, play in _LAPTOP_SEARCH:
        match = pattern.match(text)
        if match and match.group("q").strip(" ,.?"):
            return Intent(name="laptop_search", tool="laptop_search",
                          args={"query": match.group("q").strip(" ,.?"),
                                "site": site, "play": play})
    if _PLAY_ANY_RE.match(text) or (
            _PLAY_SONG_HEARD_RE.match(text)
            and len(text.split()) <= _PLAY_SONG_HEARD_MAX_WORDS):
        return Intent(name="laptop_search", tool="laptop_search",
                      args={"query": config.LAPTOP_DEFAULT_MUSIC,
                            "site": "youtube", "play": True})
    if _RECORD_RE.match(text):
        return Intent(name="laptop_record", tool="laptop_record", args={"action": "start"})
    if _STOP_PLAYING_RE.match(text):
        return Intent(name="laptop_game", tool="laptop_game_play", args={"action": "stop"})
    match = _YOU_PLAY_RE.match(text)
    if match:
        hint = (match.group("g") or match.group("g2") or "").strip(" ,.?")
        return Intent(name="laptop_game", tool="laptop_game_play",
                      args={"action": "start", "game": hint})
    if _LIST_GAMES_RE.match(text):
        return Intent(name="laptop_list", tool="laptop_apps", args={"kind": "games"})
    if _LIST_APPS_RE.match(text):
        return Intent(name="laptop_list", tool="laptop_apps", args={"kind": "all"})
    if _SUGGEST_RE.match(text):
        return Intent(name="laptop_suggest", tool="laptop_suggest", args={})
    match = _OPEN_APP_RE.match(text)
    if match:
        q = match.group("q").strip(" ,.?")
        if q and len(q.split()) <= 5 and not _NOT_AN_APP.match(q):
            return Intent(name="laptop_app", tool="laptop_open_app", args={"name": q})
    match = _PLAY_NAMED_RE.match(text)
    if match:
        q = match.group("q").strip(" ,.?")
        if (q and len(q.split()) <= _PLAY_MAX_WORDS
                and not _PLAY_NOT_A_SONG.match(q)):
            # An installed GAME by that name if he has one, else YouTube.
            return Intent(name="laptop_play", tool="laptop_play", args={"query": q})
    return None


def _memory_intent(heard: str) -> Intent | None:
    """File 17: "always remember ...", "remember that ...", "forget ...", and
    his answer to the forget question. The whole sentence has to BE the ask
    (anchored at the start after "Friday,"), so "do you remember when..." is
    a question for the brain, not something to save."""
    text = _ADDRESS_RE.sub("", heard).strip(" .!?")
    try:
        import memory_governance                # noqa: PLC0415
        waiting = memory_governance.pending() is not None
    except Exception:                           # noqa: BLE001
        waiting = False
    if waiting and _PENDING_ANSWER_RE.fullmatch(text):
        return Intent(name="forget_answer", tool="forget_confirm",
                      args={"answer": text})
    match = _REMEMBER_FOREVER_RE.match(text)
    if match:
        return Intent(name="remember_forever", tool="remember_forever",
                      args={"fact": _third_person(match.group("fact").strip(" ."))})
    match = _REMEMBER_EXPLICIT_RE.match(text)
    if match:
        return Intent(name="remember_explicit", tool="remember",
                      args={"fact": _third_person(match.group("fact").strip(" ."))},
                      say="I'll remember that.")
    match = _FORGET_RE.match(text)
    if match:
        return Intent(name="forget", tool="forget_memory",
                      args={"description": match.group("what").strip(" .")})
    return None


_ENV_NEW_RE = re.compile(
    r"^(?:this is|we are|we're)\s+(?:in\s+|at\s+)?(?:a\s+)?"
    r"(?:new|different)\s+(?:place|environment|house|home|room)"
    r"(?:\s+called\s+(?P<name>.+))?$"
    r"|^new environment(?:\s+called\s+(?P<name2>.+))?$", re.IGNORECASE)
_ENV_SWITCH_RE = re.compile(
    r"^(?:we(?:'re| are)\s+(?:back\s+)?(?:at|in)|switch\s+(?:the\s+)?"
    r"(?:map|environment|place)\s+to|load\s+(?:the\s+)?)\s+(?P<name>.+?)"
    r"(?:\s+map)?$", re.IGNORECASE)
_ENV_NAME_RE = re.compile(
    r"^(?:call|name)\s+(?:it|that|this|this area|that area|the new area)\s+"
    r"(?:as\s+)?(?P<name>.+)$", re.IGNORECASE)


def _env_intent(heard: str) -> Intent | None:
    """File 19. "this is a new place [called X]", "we're at <known place>",
    "call it <name>" (only while an area waits for a name)."""
    text = _ADDRESS_RE.sub("", heard).strip(" .!?")
    try:
        import environments                     # noqa: PLC0415
    except Exception:                           # noqa: BLE001
        return None
    match = _ENV_NEW_RE.match(text)
    if match:
        name = (match.group("name") or match.group("name2") or "").strip()
        return Intent(name="env_new", local="env_new", args={"name": name})
    match = _ENV_SWITCH_RE.match(text)
    if match:
        try:
            known = environments.by_name(match.group("name").strip())
        except Exception:                       # noqa: BLE001
            known = None
        if known is not None:                   # only a place she knows
            return Intent(name="env_switch", local="env_switch",
                          args={"name": known["name"]})
    match = _ENV_NAME_RE.match(text)
    if match:
        try:
            waiting = environments.pending()
        except Exception:                       # noqa: BLE001
            waiting = []
        if waiting:
            return Intent(name="env_name", local="env_name",
                          args={"id": waiting[-1]["id"],
                                "name": match.group("name").strip()})
    return None


def _day_intent(heard: str) -> Intent | None:
    """File 16: reminders, to-dos, routines, the briefing. All tool calls, so
    they run through the MCP server like every other command, and all answer
    out loud (_SPEAK_RESULT)."""
    try:
        import day_calendar                     # noqa: PLC0415
        cal = day_calendar.parse_request(heard)
    except Exception:                           # noqa: BLE001
        cal = None
    if cal is not None:
        if cal["action"] == "add":
            import time as _time                # noqa: PLC0415
            when = (_time.strftime("%Y-%m-%d %H:%M",
                                   _time.localtime(cal["start"]))
                    if cal["start"] else "")
            return Intent(name="calendar_add", tool="calendar_add_event",
                          args={"title": cal["title"], "when": when})
        if cal["action"] == "remove":
            return Intent(name="calendar_remove", tool="calendar_remove",
                          args={"which": cal["which"]})
        if cal["tomorrow"] or cal["days"] > 1:
            return Intent(name="calendar_ask", tool="calendar_upcoming",
                          args={"days": 2 if cal["tomorrow"] else cal["days"]})
        return Intent(name="calendar_ask", tool="calendar_today")
    try:
        import reminders                        # noqa: PLC0415
        asked = reminders.parse_request(heard)
    except Exception:                           # noqa: BLE001
        asked = None
    if asked is not None:
        if asked["action"] == "add" and asked["due_at"] is not None:
            import time as _time                # noqa: PLC0415
            when = _time.strftime("%Y-%m-%d %H:%M",
                                  _time.localtime(asked["due_at"]))
            return Intent(name="reminder_add", tool="add_reminder",
                          args={"text": asked["text"], "when": when})
        if asked["action"] == "add":
            return Intent(name="todo_add", tool="add_todo",
                          args={"text": asked["text"]})
        if asked["action"] == "done":
            return Intent(name="reminder_done", tool="complete_reminder",
                          args={"which": asked["which"]})
        if asked["action"] == "list":
            return Intent(name="reminder_list", tool="list_reminders",
                          args={"which": "todo" if asked["which"] == "todo"
                                else "all"})
    if _BRIEFING_RE.search(heard):
        return Intent(name="briefing", tool="daily_briefing")
    try:
        import routines                         # noqa: PLC0415
        routine = routines.match(heard)
    except Exception:                           # noqa: BLE001
        routine = None
    if routine is not None:
        return Intent(name="routine", tool="run_routine",
                      args={"name": routine})
    return None


def _local_intent(heard: str, strict: bool = False) -> Intent | None:
    """The two things the spine does itself: stop listening, and say what it
    knows how to do. Checked before every tool rule, because both phrasings
    contain words that a tool rule would otherwise claim."""
    # File 15 -- first, because "stop describing" contains "stop" and "keep
    # describing what you see" contains the "see" question.
    if _NARRATE_OFF_RE.search(heard):
        return Intent(name="narration", local="narration", args={"on": False},
                      say="Okay, I'll stop describing.")
    if _NARRATE_ON_RE.search(heard):
        return Intent(name="narration", local="narration", args={"on": True},
                      say="Okay, I'll keep telling you what I see as it "
                          "changes. Say stop describing when you've had "
                          "enough.")

    if _SLEEP_RE.search(heard):
        minutes, unit = _amount(heard)
        if unit == "s" and minutes:
            minutes = float(minutes)
        else:
            minutes = config.SLEEP_DEFAULT_MINUTES
        if re.search(r"\b(hour|ghanta|ghante)\b", heard):
            minutes *= 60.0
        minutes = max(1.0, min(minutes, config.SLEEP_MAX_MINUTES))
        return Intent(name="sleep", local="sleep",
                      args={"minutes": minutes},
                      say=f"Okay, I'll stop listening for "
                          f"{minutes:.0f} minutes.")

    # QUIET MODE BY VOICE (file 16: "movie time" turns it on, so there has to
    # be a way to turn it off). After the sleep rule on purpose -- "be quiet"
    # still means stop listening, as it always has; "quiet mode", "you can
    # talk again" are this.
    try:
        import mode as mode_module              # noqa: PLC0415
        quiet = mode_module.quiet_asked_for(heard)
    except Exception:                           # noqa: BLE001
        quiet = None
    if quiet is not None:
        return Intent(name="quiet", local="quiet", args={"on": quiet},
                      say=("Quiet mode on. I'll still answer if you ask."
                           if quiet else "Okay, I can talk again."))

    if _HELP_RE.search(heard):
        return Intent(name="help", local="help", say=help_text())

    # FILE 09 STEP 8. "Friday, professional mode" / "demo mode off". A file
    # write, so it needs no tool and no model -- and it must work when the
    # brain is unreachable, because the moment somebody says "demo mode on"
    # is the moment a demo is about to start. The words live in mode.py with
    # the mode they set; this only turns them into an intent.
    try:
        import mode as mode_module              # noqa: PLC0415
        wanted = mode_module.asked_for(heard)
    except Exception:                           # noqa: BLE001
        wanted = None
    if wanted:
        return Intent(
            name="mode", local="mode", args={"mode": wanted},
            say=("Professional mode on." if wanted == "professional"
                 else "Back to normal."))

    # "EXPLORE FOR FIVE MINUTES". The duration is pulled out of the sentence
    # by exploration_mode, which owns that parsing -- a second copy here would
    # be a second thing to keep in step with it. Marked `local` because the
    # session runs as its own process rather than as a tool call: it lasts
    # minutes, and a turn that blocks for minutes is a robot that has stopped
    # listening, including for "stop".
    #
    # start_exploration is still what actually drives; exploration_mode calls
    # it. Nothing about the run itself changed.
    # A QUESTION ABOUT THE RUN IS NOT A REQUEST TO START ONE. parse_request()
    # below matches any sentence containing "explore", and _local_intent runs
    # before every rule table, so "are you still exploring?" started a
    # twenty-minute autonomous run and the explore_status rule in _ASK_RULES
    # was unreachable. Falling through hands it to that rule.
    if _EXPLORE_STATUS_RE.search(heard):
        return None

    try:
        import exploration_mode
        minutes = exploration_mode.parse_request(heard)
    except Exception:                           # noqa: BLE001
        minutes = None
    if minutes is not None:
        # ⚠ AN ORDER TO EXPLORE, NOT A SENTENCE CONTAINING THE WORD.
        # parse_request matches any sentence with "explore" in it, and this
        # runs before every rule table, so with a brain available a video
        # saying "we explore the caves later" would send the robot off on a
        # twenty-minute autonomous run. Strict mode asks for the shape of an
        # order first; anything else goes to the brain, which can read it.
        if strict and not _looks_like_an_order(heard):
            return None
        # THE TOOL CALL IS THE INTENT, and the narrator is a follow-up. This
        # used to be local-only, which pushed start_exploration into a spawned
        # process and put 30 s between "Friday explore" and the wheels moving.
        # main.py calls this on the session it already has, then attaches.
        # ALWAYS COMES BACK (Harsh, 2026-09-24: "i say friday explore after
        # exploring should come back where it started"). The explore trip
        # marks the spot, explores on one tracked map and drives back; it
        # speaks for itself, so nothing is said here and no narrator is
        # attached. The duration parsed from the sentence is still used.
        return Intent(name="explore", tool="explore_and_return",
                      args={"minutes": minutes})
    return None


def _valued_intent(heard: str) -> Intent | None:
    """A command that carries a number, a time or a phrase."""
    # -- the face --
    match = _FACE_RE.search(heard)
    if match:
        feeling = match.group(1) or match.group(2) or "happy"
        feeling = "happy" if feeling == "smile" else feeling
        return Intent(name="face", tool="set_face",
                      args={"emotion": feeling}, say="")

    # -- how fast --
    match = _SPEED_PERCENT_RE.search(heard)
    if match:
        percent = int(match.group(1) or match.group(2))
        if 1 <= percent <= 100:
            return Intent(name="speed", tool="set_speed",
                          args={"percent": percent},
                          say=f"Speed {percent} percent.")
    for pattern, percent in _SPEED_NAMED_RE:
        if pattern.search(heard):
            return Intent(name="speed", tool="set_speed",
                          args={"percent": percent},
                          say=f"Speed {percent} percent.")

    # -- timers before alarms --
    # "remind me in ten minutes" is a TIMER; "remind me at seven" is an alarm.
    # The timer pattern is checked first because "alarm" appears in neither and
    # "in" is the word that decides, which only the timer rule looks for.
    match = _TIMER_RE.search(heard)
    if match and not re.search(r"\balarm\b", heard):
        duration = _clean_tail(match.group(1))
        if duration:
            return Intent(name="timer", tool="set_timer",
                          args={"duration": duration}, say="")

    match = _ALARM_RE.search(heard)
    if match:
        when = _clean_tail(match.group(1))
        if when:
            return Intent(name="alarm", tool="set_alarm",
                          args={"time": when}, say="")

    # -- memory --
    match = _REMEMBER_RE.search(heard)
    if match:
        fact = _clean_tail(match.group(1))
        if fact:
            return Intent(name="remember", tool="remember",
                          args={"fact": fact}, say="I'll remember that.")

    match = _RECALL_RE.search(heard)
    if match:
        query = _clean_tail(match.group(1))
        if query:
            return Intent(name="recall", tool="recall",
                          args={"query": query}, say="")
    return None


def _cm(mm) -> str:
    """Millimetres as a person would say them."""
    try:
        mm = float(mm)
    except (TypeError, ValueError):
        return "I can't tell"
    if mm >= 1000:
        return f"about {mm / 1000:.1f} metres"
    return f"about {mm / 10:.0f} centimetres"


def speakable(name: str, result: str) -> str:
    """One sayable sentence from a tool's return value.

    MOST TOOLS ALREADY ANSWER IN PROSE -- get_distance says "53 cm ahead (532
    mm, from the ToF + LiDAR)" and only needs its parenthesis trimmed. Two do
    not: get_battery and get_status return JSON, because the brain reads them
    and the brain is not a speaker. Read out raw, that is a robot saying
    "under_voltage_since_boot false" to a person, so those two get shaped by
    hand and everything else falls through.
    """
    import json

    text = (result or "").strip()
    if not text:
        return ""
    if name == "remember_place" and text.startswith("Saved"):
        # "Saved “home” at x=1.23 y=0.45 on my map..." is for the log. Out
        # loud, the coordinates are noise.
        saved = re.search(r"“([^”]+)”", text)
        return (f"Saved. This spot is {saved.group(1)}." if saved
                else "Saved.")
    if not text.startswith("{"):
        # Sensor jargon in brackets is for the log, not for the room.
        return re.sub(r"\s*\([^)]*\)", "", text).strip() or text

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return text[:200]
    if not isinstance(data, dict):
        return text[:200]

    if name == "battery":
        percent = data.get("percent") or ""
        power = data.get("power") or {}
        # The flag names are Broadcom's, not English. "soft temp limit now"
        # read out loud is a robot reciting a register.
        said = {"under_voltage": "on low voltage",
                "frequency_capped": "clocked down",
                "throttled": "throttling",
                "soft_temp_limit": "warm enough to be holding back"}
        warnings = [said.get(k[:-4], k[:-4].replace("_", " "))
                    for k, v in power.items() if k.endswith("_now") and v]
        if "unknown" in str(percent).lower():
            line = "I don't have a battery gauge, so I can't give you a number."
        else:
            line = f"Battery's at {percent}."
        if warnings:
            line += f" I am {' and '.join(warnings)} right now."
        return line

    if name == "status":
        bits = []
        if data.get("cpu_percent") is not None:
            bits.append(f"CPU {float(data['cpu_percent']):.0f} percent")
        if data.get("cpu_temperature_c") is not None:
            bits.append(f"{float(data['cpu_temperature_c']):.0f} degrees")
        if data.get("memory_percent") is not None:
            bits.append(f"memory {float(data['memory_percent']):.0f} percent")
        if data.get("disk_percent") is not None:
            bits.append(f"disk {float(data['disk_percent']):.0f} percent")
        line = ", ".join(bits) if bits else "I could not read my own status."
        note = data.get("cpu_temperature_note") or ""
        if "hot" in note.lower():
            line += ". I'm running hot"
        return line + "."

    if name == "map_what":
        # describe_map already writes its own one-line summary for exactly
        # this, and it is better than anything assembled from the other keys.
        return data.get("summary") or ""

    if name == "scan":
        safe = data.get("safe_directions") or {}
        blocked = [way for way, ok in safe.items() if ok is False]
        forward = (data.get("clearance_mm") or {}).get("forward")
        lead = f"{_cm(forward)} ahead." if forward is not None else ""
        if blocked:
            return f"{lead} I can't go {' or '.join(blocked)}.".strip()
        return f"{lead} Everything around me looks clear.".strip()

    if name == "lidar":
        nearest = data.get("nearest_mm")
        if nearest is None:
            return ""
        return (f"Nearest thing is {_cm(nearest)} away, furthest "
                f"{_cm(data.get('furthest_mm'))}.")

    if name == "explore_status":
        if data.get("exploring"):
            return "Yes, I'm exploring right now."
        state = ((data.get("last_run") or {}).get("state") or "").lower()
        return (f"No, I'm not exploring. The last run {state}." if state
                else "No, I'm not exploring.")

    # Anything else that came back as JSON: read the short scalar fields and
    # leave the rest, rather than reciting a structure out loud.
    parts = [f"{k.replace('_', ' ')} {v}" for k, v in list(data.items())[:4]
             if isinstance(v, (str, int, float, bool)) and len(str(v)) < 60]
    return ", ".join(parts) + "." if parts else ""


def help_text() -> str:
    """What Friday can be told with no brain and no network.

    A voice interface has no menu, so the list has to be sayable. Grouped and
    short on purpose: read out, a list of thirty commands is not help.
    """
    return (
        "I can drive — forward, back, left, right, turn around, and stop. "
        "I can go to a place I know, go home, come back, and try again if I "
        "get stuck. "
        "I can explore on my own, map the room, and tell you what I see or "
        "how far away things are. I can set alarms and timers, show the "
        "clock, tell you the time, the date and my battery. I can do sums, "
        "tell a joke, flip a coin or roll a dice. On your laptop I can play "
        "a song or search, pause, skip, change the volume, go full screen, "
        "close the tab or lock it. Say go to sleep and I'll stop listening "
        "for a while."
    )


# A direction with no verb around it, in a long sentence, is usually not a
# command -- "I left my keys at home" is six words with "left" in the middle.
# FOR MOVEMENT IT APPLIES WITH OR WITHOUT A BRAIN (2026-09-10). It used to be
# strict mode only, on the grounds that with no brain a miss is silence. But a
# wake word that fires on a conversation hands this parser the conversation,
# and the log shows what that cost: "I'm making a mistake right now" pivoted
# her right, and her own narrator's "a wall right up against me ... on the
# left" pivoted her left. A missed order costs a repeat; a pivot nobody asked
# for costs a wall. Harsh, asked: "yes ignore direction words in long
# sentence". Everything that does not move her still caps in strict mode only.
# Five, measured against real commands rather than picked: "go forward two
# meters" is four words, "turn around" is two, and the longest thing here that
# is genuinely an order is "set a timer for ten minutes" -- which carries a
# value and never reaches this cap. At seven, "I left my keys on the table"
# fit under it exactly and turned the robot ninety degrees.
# The commands whose whole point is the answer. Kept as a set rather than a
# sixth column on every rule, because only a handful of rows differ.
#
# `see` is NOT here: say_what_you_see speaks through Friday's own speaker
# already (speak=True), and repeating its return value says everything twice.
_SPEAK_RESULT = frozenset({
    "distance", "battery", "status", "map_what", "screen", "alarms",
    "explore_status", "scan", "lidar", "recall", "alarm", "timer", "cancel",
    "weather", "learned", "growth",
    # remember_place answers in prose and does not speak for itself.
    # (explore_and_return does, since 2026-09-24 -- it is not here.)
    "remember_place",
    "reminder_add", "todo_add", "reminder_done", "reminder_list", "routine",
    "briefing", "calendar_add", "calendar_ask", "calendar_remove",
    "remember_forever", "forget", "forget_answer",
    "laptop_search", "laptop_notes", "laptop_key", "places",
    "laptop_app", "laptop_play", "laptop_list", "laptop_suggest", "laptop_game", "laptop_record",
})

_STRICT_MAX_WORDS = 5

# WHAT MAY SKIP THE BRAIN. Named explicitly, because "everything this file can
# parse" is the wrong set: "how far is the wall" and "how far is the MOON" are
# the same five words to a regex, and only one of them is a range finder.
#
# A command earns its place here by being an ORDER rather than a question --
# something whose answer does not change with the subject, and which is worse
# for going through a model: slower (a pivot after a 20 s round trip is not a
# pivot he asked for), or costlier, or both. Everything else is left to the
# brain, which is better at it and is already paid for.
_REFLEX_INTENTS = frozenset({
    "stop", "stop_exploring", "explore", "sleep", "help", "time", "mode",
    "narration",
    "left", "right", "around", "forward", "backward",
    "go_left", "go_right", "go_forward", "go_backward",
    "speed", "face",
    "alarm", "timer", "cancel", "alarms", "clock", "dismiss", "screen",
    "battery", "status", "weather", "learned", "growth",
    # File 14 navigation: never a brain turn, online or off (Harsh,
    # 2026-09-23). The tools announce themselves, so none of these is in
    # _SPEAK_RESULT except the one that is an existing brain-facing tool.
    "nav_go", "nav_home", "nav_come_back", "try_again", "explore_return",
    "remember_place",
    # File 16 -- all offline, all through tools that answer in words.
    "reminder_add", "todo_add", "reminder_done", "reminder_list", "routine",
    "briefing", "quiet", "calendar_add", "calendar_ask", "calendar_remove",
    # File 17
    "remember_forever", "remember_explicit", "forget", "forget_answer",
    # File 19
    "env_new", "env_switch", "env_name",
    # File 20
    "laptop_search", "laptop_notes", "laptop_key", "places",
    "laptop_app", "laptop_play", "laptop_list", "laptop_suggest", "laptop_game", "laptop_record",
    # offline_extras.py (2026-10-20); "talk" is never strict, so not here
    "date", "calc", "joke", "coin", "dice", "pick",
})


def understand(text: str, strict: bool = False) -> Intent | None:
    """The command in this text, or None if it is not one we know.

    `strict` is for the path where a brain IS available and this runs first as
    a reflex. It refuses the loose matches rather than acting on them, because
    the cost of being wrong is different on the two paths: with no brain, a
    refusal is Friday saying she cannot help; with a brain, a refusal just
    means the sentence goes to something that understands it properly.
    """
    intent = _match(text, strict)
    if intent is not None:
        intent.speak_result = intent.name in _SPEAK_RESULT
    if intent is None or not strict:
        return intent
    # A question the brain should answer, matched by a rule meant for an
    # order. Hand it back rather than acting on it.
    return intent if intent.name in _REFLEX_INTENTS else None


def _match(text: str, strict: bool) -> Intent | None:
    if not text or not text.strip():
        return None
    heard = text.strip().lower()
    # Counted without a leading "hey Friday," -- the name is not part of the
    # order, and counting it pushed "hey Friday, can you turn left" over.
    words = len(_ADDRESS_RE.sub("", heard).split())

    # NAVIGATION FIRST -- it yields to stop and sleep on its own, and must
    # beat the explore parser and the "back" row. See _nav_intent.
    nav = _nav_intent(heard, strict)
    if nav is not None:
        return nav

    # "STOP LISTENING" IS NOT "STOP". Above the stop rule, and the only thing
    # that is, because the stop rule matches the word `stop` on its own and
    # would take this one -- halting the wheels, staying wide awake, and doing
    # the exact opposite of what was asked.
    local = _local_intent(heard, strict)
    if local is not None:
        return local

    # FILE 16: reminders, the to-do list, routines, the briefing. Before the
    # stop rule ("remind me to stop at the shop at 6" is a reminder, not a
    # halt) and before the timer rule ("remind me in 10 minutes to call mom"
    # is a reminder; a bare "remind me in 10 minutes" still falls through to
    # the timer, and "remind me at seven" to the alarm, exactly as before).
    day = _day_intent(text.strip())     # original case: "lunch with Rahul"
    if day is not None:
        return day

    # FILE 20: his laptop -- "search X on my laptop", "play X on YouTube",
    # "notes on X". Before the stop rule: "play stop motion on youtube".
    asked = _laptop_intent(text.strip())
    if asked is not None:
        return asked

    # FILE 17: remember for ever / forget (asks first) / his yes-or-which.
    kept = _memory_intent(text.strip())
    if kept is not None:
        return kept

    # FILE 19: which environment this is, and naming a new area. Local
    # intents on purpose: switching or creating an environment is HIS word,
    # never a tool the brain could call on its own.
    place = _env_intent(text.strip())
    if place is not None:
        return place

    # STOP THE RUN, NOT JUST THE WHEELS. A bare stop() against a live
    # exploration halts the motors only until the run issues its next command,
    # which from across the room is Friday ignoring him.
    if _STOP_EXPLORE_RE.search(heard):
        return Intent(name="stop_exploring", say="Stopping the run.",
                      steps=[("stop_exploration", {}), ("stop", {})])

    # Stop wins over everything that could move.
    if _COMPILED[0][1].search(heard):
        return Intent(name="stop", tool="stop", args={}, say="Stopping.")

    # The date, a calculator, jokes, a coin, a dice (offline_extras.py). Before
    # the speed rule: "what's 15 percent of 200" used to set her SPEED to 15%.
    extra = offline_extras.understand(_ADDRESS_RE.sub("", heard), strict)
    if extra is not None:
        return Intent(name=extra[0], say=extra[1])

    # THE SPECIFIC PHRASES BEFORE THE SINGLE WORDS, and this order is load
    # bearing. "How much charge is LEFT" contains a direction, and with the
    # movement rules first that question made the robot pivot ninety degrees
    # -- a question answered by turning. Every rule in this table is a whole
    # phrase, so none of them can swallow a bare "left" or "go forward".
    for name, pattern, tool, args, say in _ASK_COMPILED:
        if pattern.search(heard):
            return Intent(name=name, tool=tool, args=dict(args), say=say)

    # Time, which is answered here rather than by a tool. Matched on the
    # address-stripped text, so a bare "Friday, time?" is still the bare ask.
    if _COMPILED[-1][1].search(_ADDRESS_RE.sub("", heard)):
        return Intent(name="time", say=_time_phrase())

    valued = _valued_intent(heard)
    if valued is not None:
        return valued

    moving = _direction_intent(heard)
    if moving is not None:
        # THE VERB IS WHAT MAKES IT AN ORDER. A bare "left" is a command; the
        # same word inside a sentence usually is not -- brain or no brain,
        # see _STRICT_MAX_WORDS.
        if words > _STRICT_MAX_WORDS and not _is_imperative(heard):
            return None
        if strict and not _looks_like_an_order(heard):
            return None
        return moving

    for name, pattern, tool, args, say in _COMPILED:
        if pattern.search(heard):
            if words > _STRICT_MAX_WORDS and (
                    strict or (name in _MOVING_RULES
                               and not _is_imperative(heard))):
                return None
            if (strict and name in _ORDERS_ONLY
                    and not _looks_like_an_order(heard)):
                return None
            return Intent(name=name, tool=tool, args=dict(args), say=say)
    return None


def is_stop(text: str) -> bool:
    """Just the stop check, for the always-on path."""
    intent = understand(text)
    return intent is not None and intent.name == "stop"


def _time_phrase() -> str:
    now = time.localtime()
    hour = now.tm_hour % 12 or 12
    suffix = "in the morning" if now.tm_hour < 12 else (
        "in the afternoon" if now.tm_hour < 18 else "in the evening")
    if now.tm_min == 0:
        return f"It's {hour} o'clock {suffix}."
    return f"It's {hour}:{now.tm_min:02d} {suffix}."


# --- is there internet? --------------------------------------------------

_last_check = 0.0
_last_result = True
_lock = threading.Lock()


def online(force: bool = False) -> bool:
    """Can Friday reach the API? Cached, because this is asked every turn.

    A TCP connect to the API host, not a ping and not an HTTP request: ICMP is
    frequently filtered and an HTTP request costs a round trip and possibly
    money. Failing closed (assuming offline) would be worse than the occasional
    wasted attempt, so anything unexpected counts as online and the real call
    is allowed to produce the real error.
    """
    # ⚠ IT NEVER BLOCKS A TURN ANY MORE (2026-09-13). The connect timeout does
    # not cover the DNS lookup in front of it, and on this Pi that lookup
    # sometimes takes 5 s: 14 spoken turns sat exactly 5.0 s between the
    # transcript and the brain, and online(force=True) measured 5.04 s then
    # 0.03 s back to back. A stale cache now answers with the last result at
    # once and refreshes on a thread; only force=True waits for the answer.
    global _refreshing
    with _lock:
        fresh = time.monotonic() - _last_check < config.ONLINE_CHECK_TTL_S
        if not force:
            if not fresh and not _refreshing:
                _refreshing = True
                threading.Thread(target=_refresh, name="online-check",
                                 daemon=True).start()
            return _last_result
    _refresh()
    return _last_result


_refreshing = False


def _refresh() -> None:
    global _last_check, _last_result, _refreshing
    try:
        with socket.create_connection((config.ONLINE_CHECK_HOST,
                                       config.ONLINE_CHECK_PORT),
                                      timeout=config.ONLINE_CHECK_TIMEOUT_S):
            result = True
    except OSError:
        result = False
    except Exception:                           # noqa: BLE001
        result = True
    with _lock:
        _last_result = result
        _last_check = time.monotonic()
        _refreshing = False


def main() -> int:
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        intent = understand(text)
        print(f"{text!r} -> {intent}")
        return 0
    print(f"online: {online(force=True)}")
    for probe in ("aage jao", "stop", "turn left", "what time is it",
                  "tell me about the moon"):
        print(f"  {probe!r:28} -> {understand(probe)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
