"""More things Friday answers with no model and no API credit (2026-10-20).

Harsh: "build offline voice commands ... these commands will not require api
credits". Everything here is text in, sentence out, on the Pi: the date, a
calculator, jokes, a coin, a dice, a random number, and a little small talk
for when the brain cannot be reached. offline._match calls `understand()`
after the stop check and before the speed and movement rules -- before them
because "what's 15 percent of 200" used to set her SPEED to 15%.

Small talk is offline-only (`strict` returns None for it): with a brain, "how
are you" deserves a real answer in her own words, not a canned line.
"""

from __future__ import annotations

import math
import random
import re
import time

# -- the date ---------------------------------------------------------------------

_DATE_RE = re.compile(
    r"^(?:what(?:'?s| is)\s+(?:the\s+|today'?s\s+)?date(?:\s+today)?"
    r"|what\s+(?:is\s+)?(?:the\s+)?date\s+(?:is\s+it\s+)?today"
    r"|what\s+day\s+is\s+(?:it|today)(?:\s+today)?"
    r"|which\s+day\s+is\s+(?:it|today)"
    r"|today'?s\s+date|the\s+date|date\s+today"
    r"|aaj\s+(?:kya|kaun\s*sa|konsa)\s+(?:date|din|tarikh|day)\s+hai"
    r"|aaj\s+ki\s+(?:date|tarikh)(?:\s+kya\s+hai)?)$", re.IGNORECASE)


def date_phrase(now: time.struct_time | None = None) -> str:
    now = now or time.localtime()
    day = now.tm_mday
    suffix = "th" if 11 <= day % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"It's {time.strftime('%A', now)}, the {day}{suffix} of {time.strftime('%B %Y', now)}."


# -- a calculator -----------------------------------------------------------------

_SMALL = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
          "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
          "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
          "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
          "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000}
_OPS = [  # spoken -> symbol; longest first
    (r"multiplied\s+by", "*"), (r"divided\s+by", "/"), (r"to\s+the\s+power\s+of", "^"),
    (r"times", "*"), (r"into", "*"), (r"x", "*"), (r"over", "/"), (r"plus", "+"),
    (r"minus", "-"), (r"and", "+"), (r"\*", "*"), (r"/", "/"), (r"\+", "+"), (r"-", "-"),
    (r"×", "*"), (r"÷", "/"), (r"\^", "^")]
_CALC_LEAD = re.compile(
    r"^(?:what(?:'?s| is)|how\s+much\s+is|calculate|compute|solve|tell\s+me)\s+(?P<e>.+?)\s*\??$",
    re.IGNORECASE)
_PERCENT_RE = re.compile(r"^(?P<a>[\d.]+)\s*(?:%|percent|per\s*cent)\s+of\s+(?P<b>[\d.]+)$")
_ROOT_RE = re.compile(r"^(?:the\s+)?square\s+root\s+of\s+(?P<a>[\d.]+)$")
_SQUARED_RE = re.compile(r"^(?P<a>[\d.]+)\s+(?P<k>squared|cubed)$")


def _numbers(text: str) -> str:
    """'twenty five' -> '25' (simple spoken numbers only); digits pass through."""
    words = text.replace(",", "").split()
    out, acc, have = [], 0, False
    for w in words:
        lw = w.lower()
        if lw in _SMALL:
            v = _SMALL[lw]
            if v in (100, 1000):
                acc = (acc or 1) * v
            else:
                acc += v
            have = True
            continue
        if have:
            out.append(str(acc))
            acc, have = 0, False
        out.append(w)
    if have:
        out.append(str(acc))
    return " ".join(out)


def _tokens(expr: str) -> list | None:
    s = " " + expr.lower() + " "
    for pattern, sym in _OPS:
        s = re.sub(rf"(?<=[\s\d]){pattern}(?=[\s\d])", f" {sym} ", s)
    toks = s.split()
    out: list = []
    for t in toks:
        if t in "+-*/^":
            out.append(t)
        else:
            try:
                out.append(float(t))
            except ValueError:
                return None
    return out


def _evaluate(toks: list) -> float | None:
    """Numbers and + - * / ^ with the usual precedence. None if malformed."""
    if not toks or len(toks) % 2 == 0:
        return None
    if any(isinstance(t, str) for t in toks[0::2]) or any(
            not isinstance(t, str) for t in toks[1::2]):
        return None
    vals, ops = [toks[0]], []
    prec = {"+": 1, "-": 1, "*": 2, "/": 2, "^": 3}

    def apply():
        b, a, op = vals.pop(), vals.pop(), ops.pop()
        if op == "/" and b == 0:
            raise ZeroDivisionError
        if op == "^" and abs(b) > 64:
            raise OverflowError
        vals.append({"+": a + b, "-": a - b, "*": a * b, "/": a / b if b else 0,
                     "^": a ** b}[op])

    for i in range(1, len(toks), 2):
        op, num = toks[i], toks[i + 1]
        while ops and prec[ops[-1]] >= prec[op] and op != "^":
            apply()
        ops.append(op)
        vals.append(num)
    while ops:
        apply()
    return vals[0]


def _say_number(x: float) -> str:
    if abs(x - round(x)) < 1e-9 and abs(x) < 1e15:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}".rstrip("0").rstrip(".")


def calculate(text: str) -> str | None:
    """The answer as a sentence, or None if this is not arithmetic."""
    m = _CALC_LEAD.match(text.strip())
    if not m:
        return None
    e = _numbers(m.group("e").strip(" ?.!"))
    e = re.sub(r"^(?:the\s+)?(?:answer\s+(?:to|of)\s+)?", "", e, flags=re.IGNORECASE)
    try:
        p = _PERCENT_RE.match(e)
        if p:
            a, b = float(p.group("a")), float(p.group("b"))
            return f"{_say_number(a)} percent of {_say_number(b)} is {_say_number(a * b / 100)}."
        r = _ROOT_RE.match(e)
        if r:
            a = float(r.group("a"))
            return f"The square root of {_say_number(a)} is {_say_number(math.sqrt(a))}."
        q = _SQUARED_RE.match(e)
        if q:
            a = float(q.group("a"))
            n = 2 if q.group("k") == "squared" else 3
            return f"{_say_number(a)} {q.group('k')} is {_say_number(a ** n)}."
        toks = _tokens(e)
        if not toks or len(toks) < 3:
            return None
        value = _evaluate(toks)
    except ZeroDivisionError:
        return "You can't divide by zero."
    except (OverflowError, ValueError):
        return "That number is too big for me."
    if value is None:
        return None
    return f"It's {_say_number(value)}."


# -- jokes, coins, dice -------------------------------------------------------------

JOKES = [
    "Why did the robot go on holiday? It needed to recharge its batteries.",
    "I told my motors a joke. They didn't laugh, but they did get a little wound up.",
    "Why was the computer cold? It left its Windows open.",
    "What do you call a robot that always takes the long way round? R2 Detour.",
    "Why did the LiDAR break up with the camera? It said, I see you for who you really are, in 360 degrees.",
    "I'd tell you a UDP joke, but you might not get it.",
    "Why do programmers prefer dark mode? Because light attracts bugs.",
    "My map and I have something in common. We're both still working out where we are.",
    "What's a robot's favourite music? Heavy metal.",
    "Why did the robot cross the road? Because it was programmed by a chicken.",
    "There are 10 kinds of people: those who understand binary and those who don't.",
    "I asked the Wi-Fi for a joke. It said, sorry, connection lost.",
    "Why don't robots ever panic? They have nerves of steel.",
    "What did the battery say to the robot? You're nothing without me.",
]
_last_joke = [-1]

_JOKE_RE = re.compile(r"^(?:tell\s+(?:me\s+)?(?:a|another|one\s+more)\s+joke|(?:say|tell)\s+something\s+funny"
                      r"|make\s+me\s+laugh|(?:a|another)\s+joke|joke\s+(?:sunao|batao)"
                      r"|koi\s+joke\s+(?:sunao|batao))(?:\s+please)?$", re.IGNORECASE)
_COIN_RE = re.compile(r"^(?:flip|toss)\s+a\s+coin|^heads\s+or\s+tails$|^coin\s+(?:flip|toss)$", re.IGNORECASE)
_DICE_RE = re.compile(r"^roll\s+(?:a|the|one)\s+(?:dice|die)$|^roll\s+(?:the\s+)?dice$", re.IGNORECASE)
_PICK_RE = re.compile(
    r"^(?:pick|choose|give\s+me|say)\s+a\s+(?:random\s+)?number\s+(?:between|from)\s+"
    r"(?P<a>-?\d+)\s+(?:and|to)\s+(?P<b>-?\d+)$", re.IGNORECASE)


def joke(rng: random.Random = random) -> str:
    i = _last_joke[0]
    while i == _last_joke[0]:
        i = rng.randrange(len(JOKES))
    _last_joke[0] = i
    return JOKES[i]


# -- small talk (offline only) ---------------------------------------------------

_TALK = [
    (re.compile(r"^(?:how\s+are\s+you(?:\s+doing)?|how'?s\s+it\s+going|kaise\s+ho|kaisi\s+ho)$", re.I),
     "I'm good. My brain is offline right now, but my offline commands still work."),
    (re.compile(r"^(?:who\s+are\s+you|what'?s\s+your\s+name|what\s+is\s+your\s+name|tum\s+kaun\s+ho)$", re.I),
     "I'm Friday, your robot."),
    (re.compile(r"^(?:thank\s+you|thanks|thank\s+you\s+so\s+much|thanks\s+a\s+lot|shukriya|dhanyavaad)$", re.I),
     "You're welcome."),
    (re.compile(r"^(?:hi|hello|hey|hello\s+there|namaste)$", re.I),
     "Hi Harsh."),
]


def understand(text: str, strict: bool, rng: random.Random = random) -> tuple[str, str] | None:
    """(intent name, the sentence to say) or None. `text` is already stripped of
    "Friday," and trailing punctuation by the caller."""
    t = text.strip(" .!?,")
    if not t:
        return None
    if _DATE_RE.match(t):
        return "date", date_phrase()
    answer = calculate(t)
    if answer is not None:
        return "calc", answer
    if _JOKE_RE.match(t):
        return "joke", joke(rng)
    if _COIN_RE.match(t):
        return "coin", rng.choice(["Heads.", "Tails."])
    if _DICE_RE.match(t):
        return "dice", f"You rolled a {rng.randint(1, 6)}."
    m = _PICK_RE.match(t)
    if m:
        a, b = sorted((int(m.group("a")), int(m.group("b"))))
        if b - a <= 1_000_000:
            return "pick", f"{rng.randint(a, b)}."
    if not strict:
        for pattern, line in _TALK:
            if pattern.match(t):
                return "talk", line
    return None
