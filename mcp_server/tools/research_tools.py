"""File 20's tools: Friday uses Harsh's laptop -- a search on his screen, or notes.

The laptop ONLY: when it cannot be reached she says so and does nothing else.
laptop_notes returns AT ONCE (research outlasts the 25 s tool limit); the
result is said between turns when it is saved. See laptop_research.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
import config
import laptop_research as lr
from mcp_server.registry import tool

NOT_AVAILABLE = "My laptop isn't available right now."


def _up() -> bool:
    return lr.configured() and lr.available()[0]


@tool
def laptop_search(query: str, site: str = "web", play: bool = False) -> str:
    """Search on Harsh's LAPTOP screen: a Google search, a YouTube search, or
    play the top YouTube video. Use whenever he asks to search, google or look
    something up on his laptop, or to search or play something on YouTube.
    It only opens the search in his browser -- you do not see the results, so
    do not describe them.

    Args:
        query: the words to search for (not a URL).
        site: "web" (Google) or "youtube".
        play: YouTube only -- true to play the top video instead of listing.
    """
    query = (query or "").strip()
    site = (site or "web").strip().lower()
    if not query:
        return "What should I search for?"
    if site not in ("web", "youtube"):
        site = "web"
    import rate_limit
    ok, why = rate_limit.allow("laptop_open")
    if not ok:
        return f"I've opened a lot on your laptop already — {why}."
    if not _up():
        return NOT_AVAILABLE
    rate_limit.used("laptop_open")
    opened, why = lr.open_search(query, site, bool(play) and site == "youtube")
    if not opened:
        return f"I couldn't open that on your laptop ({why})."
    if site == "youtube":
        return (f"Playing {query} on YouTube on your laptop." if play
                else f"Searching YouTube for {query} on your laptop.")
    return f"Searching for {query} on your laptop."


_KEY_DONE = {
    "play_pause": "Done.", "next": "Next one.", "previous": "Going back one.",
    "volume_up": "Louder.", "volume_down": "Quieter.", "mute": "Done.",
    "fullscreen": "Full screen.", "close_tab": "Closed the tab.",
    "lock": "Your laptop is locked.", "open_notes": "Opening your notes folder.",
}


@tool
def laptop_control(action: str) -> str:
    """Press one control key on Harsh's LAPTOP. action is exactly one of:
    play_pause (pause or resume the video/music), next, previous, volume_up,
    volume_down, mute (toggles), fullscreen, close_tab, lock (locks the
    laptop), open_notes (opens his Friday Notes folder). Use for "pause the
    video", "next song", "volume up", "close this tab", "lock my laptop".
    Nothing else can be pressed or typed.
    """
    action = (action or "").strip().lower().replace(" ", "_").replace("-", "_")
    if action not in lr.KEY_ACTIONS:
        return "I can't do that on your laptop."
    import rate_limit
    ok, why = rate_limit.allow("laptop_key")
    if not ok:
        return f"That's a lot of key presses on your laptop — {why}."
    if not _up():
        return NOT_AVAILABLE
    rate_limit.used("laptop_key")
    done, why = lr.press_key(action)
    if done:
        return _KEY_DONE[action]
    if why == "old helper":
        return ("Your laptop helper is the old version, so I can't do that yet. "
                "Copy the new friday_laptop_agent.py to your laptop.")
    if "browser is not the window in front" in why:
        return "Your browser isn't the window in front, so I left it alone."
    return f"I couldn't do that on your laptop ({why})."


_OLD_HELPER = ("Your laptop helper is the old version, so I can't open apps yet. "
               "Copy the new friday_laptop_agent.py to your laptop.")


def _say_list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _launch_named(app: dict) -> str:
    import rate_limit
    ok, why = rate_limit.allow("laptop_launch")
    if not ok:
        return f"I've opened a lot of apps already — {why}."
    rate_limit.used("laptop_launch")
    started, why = lr.launch(app["name"])
    if started:
        return f"Opening {app['name']} on your laptop."
    if why == "old helper":
        return _OLD_HELPER
    return f"I couldn't open {app['name']} ({why})."


@tool
def laptop_open_app(name: str) -> str:
    """Open an app or game that is installed on Harsh's LAPTOP, by name:
    "open WhatsApp", "open Minecraft", "launch Spotify". Only apps on its
    Start menu (system tools are left out on the laptop itself).

    Args:
        name: the app's name as he said it, e.g. "whatsapp", "minecraft".
    """
    name = (name or "").strip()
    if not name:
        return "Which app should I open?"
    if not _up():
        return NOT_AVAILABLE
    listed, why = lr.apps()
    if why == "old helper" and not listed:
        return _OLD_HELPER
    app, others = lr.match_app(name, listed)
    if app is None and others:
        return f"Which one: {_say_list([o['name'] for o in others])}?"
    if app is None:
        return f"I don't see {name} on your laptop."
    return _launch_named(app)


@tool
def laptop_play(query: str) -> str:
    """"Play X": a GAME installed on Harsh's laptop if he has one called that,
    otherwise X plays on YouTube on his laptop.

    Args:
        query: what he wants to play, e.g. "minecraft" or "tum hi ho".
    """
    query = (query or "").strip()
    if not query:
        return "What should I play?"
    if not _up():
        return NOT_AVAILABLE
    listed, _ = lr.apps()
    app, _others = lr.match_app(query, [a for a in listed if lr.is_game(a)])
    if app is not None:
        return _launch_named(app)
    return laptop_search(query, "youtube", True)


@tool
def laptop_apps(kind: str = "all") -> str:
    """What is installed on Harsh's laptop: kind "games" for just the games,
    "all" for everything (said briefly). Uses the last list the laptop gave
    if it is off now."""
    listed, why = lr.apps(refresh=_up())
    if not listed:
        return (_OLD_HELPER if why == "old helper" else
                "I don't know what's on your laptop yet. Turn it on and I'll look.")
    games = [a["name"] for a in listed if lr.is_game(a)]
    if (kind or "all").lower().startswith("game"):
        if not games:
            return "I don't see any games on your laptop."
        shown = games[:8] + ([f"{len(games) - 8} more"] if len(games) > 8 else [])
        return f"You have {len(games)} game{'s' if len(games) != 1 else ''}: {_say_list(shown)}."
    some = [a["name"] for a in listed if not lr.is_game(a)][:6]
    words = f"Your laptop has {len(listed)} apps, like {_say_list(some)}"
    if games:
        words += f", and games: {_say_list(games[:5])}"
    return words + "."


@tool
def laptop_suggest(kind: str = "game") -> str:
    """Suggest something to play on Harsh's laptop ("I'm bored", "what should
    I play?"): one of his installed games, or music if he has none."""
    import random
    listed, _ = lr.apps(refresh=False)
    games = [a["name"] for a in listed if lr.is_game(a)]
    if games:
        pick = random.choice(games)
        return f"How about {pick}? Say: play {pick}."
    return "How about some music? Say: play a song."


RECORDING_FILE = Path(config.ROOT) / "state" / "laptop_recording.json"


def recording_on() -> bool:
    try:
        return bool(json.loads(RECORDING_FILE.read_text()).get("running"))
    except (OSError, ValueError):
        return False


def _mark_recording(running: bool, **extra) -> None:
    try:
        RECORDING_FILE.parent.mkdir(parents=True, exist_ok=True)
        RECORDING_FILE.write_text(json.dumps({"running": running, "at": time.time(), **extra}))
    except OSError:
        pass


# what she calls each vehicle's recording, and the keys it reads
VEHICLE_WORDS = {
    "car": ("driving", "W, A, S, D"), "bike": ("bike riding", "W, A, S, D"),
    "boat": ("boat driving", "W, A, S, D"),
    "heli": ("helicopter flying", "W, A, S, D and numpad 8, 5, 4, 6"),
    "plane": ("flying", "W, A, S, D and numpad 8, 5, 4, 6"),
}


@tool
def laptop_record(action: str = "start", vehicle: str = "car") -> str:
    """Record Harsh DRIVING the game in front on his laptop, to teach Friday to
    drive it -- ONLY when he asks ("record my driving", "record my flying",
    "stop recording"). Saves the game picture, its minimap and whether W, A,
    S, D are held (a heli or plane: also numpad 8, 5, 4, 6), 10 times a
    second, only while that game is in front. Nothing else is read.

    Args:
        action: "start" or "stop".
        vehicle: car, bike, boat, heli or plane -- what he is about to drive.
    """
    import laptop_games
    if (action or "start").strip().lower() == "stop":
        got, why = lr.record("", "stop")
        _mark_recording(False)
        if got is None:
            return f"I couldn't stop the recording ({why})."
        mins = float(got.get("seconds") or 0.0) / 60.0
        return (f"Stopped recording. I have {mins:.0f} minutes of your driving "
                f"({int(got.get('frames') or 0)} pictures).")
    if not _up():
        return NOT_AVAILABLE
    game = laptop_games.game_in_front(lr.idle(), lr.known_apps())
    if game is None:
        return "Open the game and leave it in front, then tell me to record."
    vehicle = (vehicle or "car").strip().lower()
    if vehicle not in VEHICLE_WORDS:
        vehicle = "car"
    got, why = lr.record(game, "start", float(config.LAPTOP_RECORD_MAX_S), vehicle)
    if got is None:
        return ("Your laptop's helper is too old to record. Copy the new one."
                if why == "old helper" else f"I couldn't start recording ({why}).")
    if got.get("vehicle", "car") != vehicle:          # a helper older than 19
        lr.record("", "stop")
        return f"Your laptop's helper is too old to record a {vehicle}. Copy the new one."
    _mark_recording(True, game=game, session=got.get("session"), vehicle=vehicle)
    doing, keys = VEHICLE_WORDS[vehicle]
    return (f"Recording your {doing} in {game}. Play normally, and say stop recording "
            f"when you're done. I only save the game picture and the {keys} keys.")


@tool
def laptop_game_play(action: str = "start", game: str = "") -> str:
    """Friday plays a game on Harsh's laptop herself -- ANY of his games, ONLY
    when he asks: "you play the game", "you play Asphalt". action "start"
    plays the game he named (opening it first if it is not in front) or the
    game in front; "stop" hands it back. Slow games are played by clicking,
    the rest with the keyboard, one look per move.

    Args:
        action: "start" or "stop".
        game: the game he named, if any.
    """
    import laptop_games
    if (action or "start").strip().lower() == "stop":
        return ("Okay, your turn." if laptop_games.stop_playing()
                else "I'm not playing anything.")
    if not _up():
        return NOT_AVAILABLE
    return laptop_games.play_start(game or "")


@tool
def laptop_notes(topic: str) -> str:
    """Research a topic on the web and put notes about it on Harsh's LAPTOP,
    opened on his screen. Use for "make notes on X", "notes about X",
    "properly research X", "research X in depth". Returns at once; the notes
    take a minute and you will be told when they are saved -- just tell him
    you're on it.

    Args:
        topic: the topic, fully stated (he will not be asked again).
    """
    topic = (topic or "").strip()
    if not topic:
        return "What should I make notes on?"
    import rate_limit
    ok, why = rate_limit.allow("laptop_notes")
    if not ok:
        return f"I've made a lot of notes already — {why}."
    if not _up():
        return f"{NOT_AVAILABLE} I can't put notes there."
    rate_limit.used("laptop_notes")
    lr.start_notes(topic)
    return (f"I'm researching {topic} and I'll put the notes on your laptop — "
            f"I'll tell you when they're open.")


@tool
def research_detail(query: str = "") -> str:
    """What the notes Friday made say, when Harsh asks for more ("tell me more
    about that", "what did you find out about X").

    Args:
        query: words from the topic; empty for the most recent notes.
    """
    job = lr.latest(query)
    if job is None:
        return "I haven't made notes on anything like that yet."
    detail = (job.get("detail") or job.get("summary") or "").strip()
    return (f"On {job['query']}: {detail[:1500]} The notes are on your laptop "
            f"at {job['path']}.")
