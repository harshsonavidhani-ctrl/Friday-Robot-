"""Every sentence the exploration mission says (File 21 step 6, spec 4.13).

FIXED TEMPLATES, NO MODEL. Each line is in LINES below, filled with nothing
but names and numbers the code already holds, and spoken by Piper on this Pi
(through nav_voice -> tts.speak(), CLAUDE.md PART 4). She says the same thing
online, offline and with no API key, because nothing here knows what a key is.

    text, mood = render("saved", name="Kitchen")
    say(adapters, "saved", name="Kitchen")        # face + voice

STYLE is File 14's and the exploration's own: short, declarative, present
tense. MOOD is the spec's intended feeling, mapped onto the closest of the 20
faces that exist (face_control.EMOTIONS). Two have no face of their own:
"worried" shows as scared and "determined" as thinking. No emotion is added.

The only liberty taken with the spec's wording is grammar: "1 new area", not
"1 new areas". Numbers: {m2} whole, {d} one decimal, and "square meters" /
"meters" as words because Piper reads them better than "m2".
"""

from __future__ import annotations

LIST_LIMIT = 8

# key -> (template, mood). Plural-sensitive lines are built in functions below.
LINES: dict[str, tuple[str, str]] = {
    "mission_start":  ("Starting exploration. I've marked this spot as my "
                       "starting point.", "excited"),
    "cant_start":     ("I'm not sure where I am. Please place me at my home "
                       "spot.", "confused"),
    "ask_label":      ("I've finished exploring this area. What should I call "
                       "it?", "curious"),
    "ask_suggestion": ("I've finished exploring this area. It looks like a "
                       "{suggestion}. What should I call it?", "curious"),
    "reask":          ("Sorry, I didn't catch that. What should I call this "
                       "area?", "confused"),
    "saved":          ("Saved as {name}.", "happy"),
    "duplicate":      ("I already have a {name}. Should I call this "
                       "{suggestion}?", "thinking"),
    "unnamed":        ("I'll save it as {name} for now. You can rename it "
                       "later.", "neutral"),
    "paused":         ("Stopping. Say continue when you're ready.", "neutral"),
    "resumed":        ("Continuing exploration.", "happy"),
    "time_cap":       ("That's enough exploring for now.", "neutral"),
    "returning":      ("Going back to where I started.", "neutral"),
    "rung3":          ("Checking where I am.", "thinking"),
    "arrival_adjust": ("I'm not quite where I started. Moving onto the spot.", "determined"),
    "rung5":          ("Retracing my path.", "determined"),
    "rung6":          ("Getting as close as I can.", "determined"),
    "help":           ("I'm stuck about {d} meters from where I started. "
                       "Please help me or clear the way.", "worried"),
    "help_wait":      ("I'll wait here. Say Friday, come back, to try again.",
                       "sad"),
    "battery":        ("My battery is low, so I'm stopping here.", "worried"),
    "arrived":        ("I've reached where I started. Exploration saved.",
                       "happy"),
    # The second half of "arrived", for when the RL run has just said "I am
    # back where I started." itself and saying it again would be an echo.
    "saved_only":     ("Exploration saved.", "happy"),
    "list_none":      ("I haven't saved any areas yet.", "neutral"),
    "renamed":        ("Renamed {old} to {new}.", "happy"),
    "forget_ask":     ("Should I forget {name}?", "neutral"),
    "forgot":         ("Forgot {name}.", "neutral"),
    "unknown_place":  ("I don't know where {name} is yet.", "confused"),
    "map_reset":      ("My map was reset, so I need to re-explore my saved "
                       "areas.", "confused"),
    "continue_or_new": ("Continue the last exploration or start a new one?",
                        "curious"),
    "curiosity_ask":  ("I'm curious about the house. Can I explore for a "
                       "while?", "curious"),
    "curiosity_auto": ("I'm going to explore for a while.", "excited"),
    "cant_now":       ("I can't do that right now.", "neutral"),
    # NOT in the spec's table: main.py refuses a move ordered more than
    # COMMAND_MAX_AGE_S ago (transcription is slow on this Pi); a mission order
    # that would set her driving is refused the same way, in her own words.
    "stale_order":    ("That order is too old now, so I didn't move. Please "
                       "say it again.", "neutral"),
    "failsafe":       ("I've stopped and I need help.", "worried"),
    # "go to X" in the middle of a mission (U2): File 14 announces its own
    # journeys (spec 4.13: "if File 14 already announces something, don't say
    # it twice"), so these are File 14's sentences, word for word (nav_voice.py
    # setting_off / arrived / no_route minus "Stopping here." -- she goes on
    # exploring, which "Continuing exploration." then says).
    "goto_off":       ("Heading to {name}.", "neutral"),
    "goto_arrived":   ("I have arrived at {name}.", "happy"),
    "goto_failed":    ("I cannot find a route to {name}.", "confused"),
    # NOT in the spec's table: SET_HOME must "confirm" (4.8) and the table has
    # no line for it, so it reuses the mission-start wording.
    "home_set":       ("I've marked this spot as my starting point.", "happy"),
}

# The spec's moods -> the 20 faces that exist. Anything else -> neutral.
MOOD_TO_FACE: dict[str, str] = {
    "excited": "excited", "confused": "confused", "curious": "curious",
    "happy": "happy", "thinking": "thinking", "neutral": "neutral",
    "proud": "proud", "sad": "sad",
    "worried": "scared",          # no "worried" face; the nearest that exists
    "determined": "thinking",     # no "determined" face
}


def face_for(mood: str, available=None) -> str:
    face = MOOD_TO_FACE.get(mood, "neutral")
    if available is not None and face not in available:
        return "neutral"
    return face


# --- formatting ------------------------------------------------------------------

def fmt_m2(value: float) -> str:
    return str(int(round(float(value))))


def fmt_d(value: float) -> str:
    return f"{float(value):.1f}"


def join_names(names, limit: int = LIST_LIMIT) -> str:
    """"a", "a and b", "a, b and c"; past `limit`: the first `limit`, then
    "and k more"."""
    names = [str(n) for n in names if str(n).strip()]
    if not names:
        return ""
    if len(names) > limit:
        return ", ".join(names[:limit]) + f" and {len(names) - limit} more"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _areas(n: int) -> str:
    return "area" if n == 1 else "areas"


def render(key: str, **fields) -> tuple[str, str]:
    """(sentence, mood) for a fixed line. {d} and {m2} are formatted here."""
    template, mood = LINES[key]
    if "d" in fields:
        fields["d"] = fmt_d(fields["d"])
    if "m2" in fields:
        fields["m2"] = fmt_m2(fields["m2"])
    return template.format(**fields), mood


def summary(m2: float, new_names, total: int, time_cap: bool = False,
            not_entered: int = 0) -> tuple[str, str]:
    """The end-of-exploration line (spec 4.13 Summary / none new / time-cap
    prefix / not-entered). `new_names` are this mission's new areas, in the
    order to say them."""
    new_names = list(new_names)
    n = len(new_names)
    parts = []
    if time_cap:
        parts.append(LINES["time_cap"][0])
    if n:
        line = (f"Exploration complete. My map now covers about {fmt_m2(m2)} "
                f"square meters, and I saved {n} new {_areas(n)}: "
                f"{join_names(new_names)}.")
        if total > n:
            line += f" In total I know {total} {_areas(total)}."
        parts.append(line)
        mood = "proud"
    else:
        parts.append(f"Exploration complete. My map now covers about "
                     f"{fmt_m2(m2)} square meters. I didn't find any new areas.")
        mood = "neutral"
    if not_entered > 0:
        parts.append(f"I also found {not_entered} {_areas(not_entered)} I didn't "
                     f"enter. You can name them on the dashboard.")
    return " ".join(parts), mood


def list_line(names) -> tuple[str, str]:
    """`names`: new areas first (the caller orders them)."""
    names = list(names)
    if not names:
        return LINES["list_none"]
    return (f"I know {len(names)} {_areas(len(names))}: {join_names(names)}.",
            "neutral")


def status_line(m2: float, n: int) -> tuple[str, str]:
    return (f"My map covers about {fmt_m2(m2)} square meters, and I've saved "
            f"{n} {_areas(n)} so far.", "neutral")


def interrupted_line(n: int) -> tuple[str, str]:
    return (f"My last exploration was interrupted. I saved {n} {_areas(n)}. "
            f"Say continue to resume, or explore to start again.", "neutral")


# --- saying it -------------------------------------------------------------------

def speak(adapters, text: str, mood: str) -> str:
    """Face first (so the look lands with the first word), then the voice.
    Neither may take the mission down: a lost face or voice is logged by the
    adapter and the line is still returned for the mission log."""
    try:
        adapters.face.set(face_for(mood, adapters.face.names()))
    except Exception:                                   # noqa: BLE001
        pass
    try:
        adapters.voice.say(text)
    except Exception:                                   # noqa: BLE001
        pass
    return text


def say(adapters, key: str, **fields) -> str:
    text, mood = render(key, **fields)
    return speak(adapters, text, mood)
