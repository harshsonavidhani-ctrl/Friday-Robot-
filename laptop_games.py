"""Friday and Harsh's laptop games (2026-10-20).

Harsh: "if she opens app can she control it like if it is a game", then -- asked
-- "both but 2nd one will be command driven, I will say friday you play the
game". So:

  WATCH (automatic): while HE is playing (the laptop touched in the last
  LAPTOP_GAME_WATCH_ACTIVE_S, a game of his in front), at most once per
  LAPTOP_GAME_WATCH_GAP_S she looks at the game window and reacts in a few
  words -- or says nothing (SKIP). Spoken by reminders.ReminderWatch, between
  turns, through initiative.decide() (never while he talks to her).

  PLAY (his command only): "Friday, you play the game" / "you play Asphalt".
  ANY of his games (Harsh 2026-10-21: "any game if i ask her, not a
  particular one"). A game he names that is not in front is opened first.
  SLOW games (SLOW_GAMES: solitaire, chess...) are played by clicking: a
  picture, one vision call, one click. Every other game is played with the
  keyboard (helper 8 /step): she holds a few game keys for a moment, the
  laptop hands back the next picture, she thinks again. The game runs on
  while she thinks, so she is slow at fast games -- GAME_PAUSE_KEYS can name
  a game's pause key so it waits for her between moves. Ends on "stop" (the
  stop tool calls stop_playing), "stop playing", a finished game, the game no
  longer in front, GAME_PLAY_MAX_MOVES / _STEPS, or the money.

THE LAPTOP DECIDES what she may see and touch: helper 6 hands over a picture
of, and takes a click in, ONLY the window in front, and ONLY when its title
matches the game named (laptop/friday_laptop_agent.py /screen, /click). With
WhatsApp or a browser in front she gets nothing.

Paid: every look is one vision call (scene_describe._call: the daily spend
cap, RATE_LIMITS "game_watch" / "game_play", the cost recorded). Out of API
credit: she says so and does neither.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

import config

log = logging.getLogger("friday.laptop_games")

SLOW_GAMES = re.compile(r"solitaire|chess|minesweeper|sudoku|mahjong|jigsaw|crossword|"
                        r"2048|checkers|draughts|freecell|spider|klondike|tic.?tac|"
                        r"connect ?4|wordle|\bword\b|puzzle|cards?\b", re.IGNORECASE)
_GENERIC = {"game", "games", "launcher", "for", "windows", "the", "app", "and",
            "casual", "edition", "microsoft", "collection", "free"}

WATCH_SYSTEM = ("You are Friday, a small home robot watching your friend Harsh play "
                "{game} on his laptop. Look at the screenshot of the game. If something "
                "is worth a quick remark, react in at most 12 words, like an excited "
                "friend sitting beside him (cheer, warn, tease). If nothing is, reply "
                "exactly SKIP. Never describe anything outside the game.")
PLAY_SYSTEM = ("You are Friday, a small robot playing {game} on a laptop by clicking. "
               "You see a screenshot of the game window. Choose the single best next "
               "click. x and y are fractions of the image: x 0 = left edge, 1 = right "
               "edge; y 0 = top, 1 = bottom. Reply with ONLY a JSON object: "
               '{{"x": 0.42, "y": 0.61, "why": "a few words"}} -- or {{"done": true, '
               '"why": "a few words"}} if the game is over, won, or no move helps. '
               "Click only on the board, cards or pieces: never menus, settings, ads, "
               "buttons that leave the game, close or minimise, and never the top "
               "strip of the window (its title bar).")
KEYS_SYSTEM = ("You are Friday, a small robot playing {game} on a laptop with the "
               "keyboard. You see a screenshot of the game window. Choose the next "
               "move: keys held down together for a moment. Reply with ONLY a JSON "
               'object: {{"keys": ["up", "left"], "ms": 600, "why": "a few words"}} -- '
               "at most 3 keys from: {keys}; ms from 100 to {max_ms}. The game keeps "
               "running while you think, so prefer short holds. Racing: up = accelerate, "
               "left/right = steer, down = brake; steer toward the middle of the road. "
               "Side-view driving (Hill Climb Racing): right = gas, left = brake; if the "
               "nose points up or the car is in the air nose-up, brake briefly. "
               "GTA: w = forward/accelerate, s = brake/reverse, a/d = steer, f = get "
               "in or out of a car, space = handbrake, shift = sprint. "
               'Add "keep": true to keep those keys held while you decide your next '
               "move (gas on a hill, driving straight) -- otherwise they are let go "
               "and the car slows while you think. "
               'To press a button in a menu you may click instead: {{"x": 0.5, '
               '"y": 0.8, "why": "..."}} (x, y fractions of the image). Or {{"done": '
               'true, "why": "a few words"}} if the game is over or you are stuck. '
               "Never quit the game, open settings or a shop, buy anything, or click "
               "the top strip of the window (its title bar).")

DRIVE_SYSTEM = ("You are Friday, a small robot playing {game} on a laptop. An AUTOPILOT "
                "on the laptop drives the car for you between your looks, ten times a "
                "second: it follows the road on the minimap, holds the gas and backs out "
                "of crashes. You look at the screenshot and give directions. Reply with "
                'ONLY a JSON object: {{"drive": {{"bias": 0, "speed": 1}}, "why": "a few '
                'words"}} -- bias -1 = keep/turn left, 0 = follow the road, 1 = '
                "keep/turn right; speed 0 = stop, 1 = full. For anything the autopilot "
                'cannot do, a manual move instead (it pauses for it): {{"keys": ["f"], '
                '"ms": 200, "why": "..."}} -- keys from: {keys}; f gets in or out of a '
                'car, s reverses. Or {{"done": true, "why": "..."}}. Never quit the '
                "game, open online modes or a shop, or buy anything.")

# OFFLINE CHEERING (Harsh 2026-10-20: "it can be offline?" -> "yes add the
# offline cheering"). No picture and no model: she knows WHICH game is in front
# and that he is playing it, and cheers in that game's spirit. Used whenever a
# real look is not possible -- no API credit, an older laptop helper, a failed
# look -- and on its own with LAPTOP_GAME_WATCH_ONLINE False.
CHEER_KINDS = [
    ("gta", re.compile(r"grand theft|\bgta\b", re.I), [
        "Watch out for the cops!", "Drive like you stole it! Oh wait.",
        "Five stars incoming, run!", "Nice ride! Don't crash it."]),
    ("racing", re.compile(r"asphalt|need for speed|forza|racing|\bcar\b|bike|hill climb|drift", re.I), [
        "Floor it! Nitro now!", "Don't hit the wall!", "Faster! You've got this!",
        "Ooh, nice drift!", "Overtake them!"]),
    ("horror", re.compile(r"evil nun|granny|horror|scary|crimson snow|haunt|zombie", re.I), [
        "I'm scared! Don't let her catch you!", "Did you hear that? Hide!",
        "Run, run, run!", "Look behind you! Just kidding. Maybe."]),
    ("shooter", re.compile(r"free ?fire|pubg|bgmi|call of duty|valorant|counter|fortnite|apex|"
                           r"battlefield|overwatch", re.I), [
        "Watch your back!", "Booyah! Get that win!", "Find cover!",
        "Head shot! Nice!", "Squad up, you've got this!"]),
    ("cards", re.compile(r"solitaire|freecell|spider|klondike|mahjong|chess|sudoku|minesweeper|"
                         r"puzzle|crossword|2048|checkers", re.I), [
        "Look for the aces first!", "Think it through, no rush.",
        "Ooh, that's a clever move.", "You're getting close!"]),
    ("building", re.compile(r"minecraft|terraria|roblox|stardew|build", re.I), [
        "Build something huge!", "Watch out for creepers!", "Ooh, what are you making?",
        "Don't dig straight down!"]),
    ("flying", re.compile(r"dcs|flight|plane|jet|pilot", re.I), [
        "Pull up! Pull up!", "Smooth landing, captain!", "Watch your altitude!"]),
]
CHEER_ANY = ["You've got this!", "Go Harsh!", "Ooh, this looks fun!", "Nice one!"]
HELLO_LINES = ["Ooh, {game}! Let's go!", "{game}! I'll cheer for you.",
               "Yay, {game} time!"]

_state = {"last_watch": 0.0, "last_game": "", "recent": []}
TITLE_BAND = 0.06            # top of the picture = the window's title bar: never clicked
_play_lock = threading.Lock()
_play: dict = {"thread": None, "stop": None, "game": "", "moves": 0, "why": ""}


def _publish(running: bool, last: str = "") -> None:
    """What she is playing, for the dashboard (it runs in another process):
    config.LAPTOP_GAME_STATE_FILE. Never raises."""
    try:
        path = Path(config.LAPTOP_GAME_STATE_FILE)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps({
            "running": running, "game": _play.get("game", ""),
            "moves": _play.get("moves", 0), "why": _play.get("why", ""),
            "fast": bool(_play.get("fast")), "last": last, "at": time.time()}))
        tmp.replace(path)
    except Exception as exc:                        # noqa: BLE001
        log.debug("game state not written: %s", exc)


def state() -> dict:
    """The last _publish(), or {} -- read by the dashboard."""
    try:
        return json.loads(Path(config.LAPTOP_GAME_STATE_FILE).read_text())
    except (OSError, ValueError):
        return {}


def game_words(name: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", (name or "").lower())
    return {w for w in words if len(w) >= 3 and w not in _GENERIC}


def title_matches(title: str, game: str) -> bool:
    """The helper's own rule (laptop/friday_laptop_agent.title_matches)."""
    words = game_words(game)
    return bool(words) and bool(words & set(re.findall(r"[a-z0-9]+", (title or "").lower())))


def game_in_front(info: dict | None, apps: list[dict]) -> str | None:
    """Which of his games is the window in front, from the laptop's /idle:
    its window title (helper 6), or the program's name (helper 5 has no
    title: "Asphalt8.exe" still says Asphalt)."""
    if not info:
        return None
    import laptop_research as lr
    seen = " ".join([str(info.get("title") or ""),
                     re.sub(r"\.exe$", "", str(info.get("app") or ""), flags=re.I)])
    seen = re.sub(r"([a-z])([0-9])", r"\1 \2", seen.lower())
    words = set(re.findall(r"[a-z0-9]+", seen))
    for w in list(words):                       # "GTA5.exe" -> grand theft auto
        words |= set(lr.NICKNAMES.get(w, "").split())
    hits = [a["name"] for a in apps if lr.is_game(a) and game_words(a["name"]) & words]
    return max(hits, key=lambda n: len(game_words(n) & words)) if hits else None


def cheer_kind(game: str) -> str:
    for kind, pattern, _lines in CHEER_KINDS:
        if pattern.search(game):
            return kind
    return "any"


def cheer_line(game: str, rng=None) -> str:
    """An offline cheer for `game`: hello the first time, then its kind's
    lines, never one of the last few again."""
    import random
    rng = rng or random
    short = re.split(r"\s+[-–:]\s+|\s+for windows", game, maxsplit=1, flags=re.I)[0]
    if game != _state["last_game"]:
        _state["last_game"] = game
        line = rng.choice(HELLO_LINES).format(game=short)
    else:
        kind = cheer_kind(game)
        pool = next((lines for k, _p, lines in CHEER_KINDS if k == kind), CHEER_ANY)
        fresh = [l for l in pool + CHEER_ANY if l not in _state["recent"]] or pool
        line = rng.choice(fresh)
    _state["recent"] = (_state["recent"] + [line])[-5:]
    return line


def _credits_out() -> bool:
    try:
        import brain
        return brain.credits_out()
    except Exception:                               # noqa: BLE001
        return False


# --- WATCH: she reacts while he plays ------------------------------------------

def watch_item(now: float | None = None) -> dict | None:
    """A reaction to say about the game he is playing now, or None. The cheap
    gates first; the one paid look only when they all pass. Never raises."""
    now = time.time() if now is None else now
    try:
        if not getattr(config, "LAPTOP_GAME_WATCH", False) or playing():
            return None
        online = (getattr(config, "LAPTOP_GAME_WATCH_ONLINE", True)
                  and not _credits_out())
        gap = float(config.LAPTOP_GAME_WATCH_GAP_S if online
                    else getattr(config, "LAPTOP_GAME_CHEER_GAP_S", 150.0))
        if now - _state["last_watch"] < gap:
            return None
        import laptop_play
        import laptop_research as lr
        if not laptop_play.laptop_on(now):
            return None
        info = lr.idle()
        idle_s = None if not info else info.get("idle_s")
        if idle_s is None or float(idle_s) > float(config.LAPTOP_GAME_WATCH_ACTIVE_S):
            return None                             # he is not playing right now
        game = game_in_front(info, lr.known_apps())
        if not game:
            return None
        _state["last_watch"] = now                  # one try per gap, paid or not
        if not online:
            return {"kind": "game_watch", "say": cheer_line(game), "game": game,
                    "offline": True}
        png, why = lr.screen(game)
        if not png:
            log.info("game watch: no picture of %s (%s) -- cheering instead", game, why)
            return {"kind": "game_watch", "say": cheer_line(game), "game": game,
                    "offline": True}
        import scene_describe
        said, why = scene_describe._call("game_watch", png, WATCH_SYSTEM.format(game=game),
                                         f"Harsh is playing {game}. React, or SKIP.", 60,
                                         media_type="image/png")
        if not said:
            log.info("game watch: no look (%s) -- cheering instead", why)
            return {"kind": "game_watch", "say": cheer_line(game), "game": game,
                    "offline": True}
        line = said.strip().strip('"')
        if not line or line.upper().startswith("SKIP"):
            return None
        return {"kind": "game_watch", "say": line[:200], "game": game}
    except Exception as exc:                        # noqa: BLE001
        log.debug("game watch skipped: %s: %s", type(exc).__name__, exc)
        return None


# --- PLAY: she plays a slow game, only when he says so -------------------------

def playing() -> bool:
    t = _play.get("thread")
    return bool(t is not None and t.is_alive())


def stop_playing() -> bool:
    """Called by the stop tool and by "stop playing". True if she was playing."""
    with _play_lock:
        ev = _play.get("stop")
        if ev is not None and playing():
            ev.set()
            return True
    return False


def autopilot_for(game: str) -> dict | None:
    """config.GAME_AUTOPILOT's settings for this game (by a word of its name)."""
    words = game_words(game)
    for word, settings in (getattr(config, "GAME_AUTOPILOT", {}) or {}).items():
        if word.lower() in words:
            return dict(settings)
    return None


def parse_move(text: str | None, keys_ok: bool = False) -> dict | None:
    """The model's JSON move -> {"x", "y"}, {"keys", "ms"} (keys_ok: a game
    played with the keyboard) or {"done": True}; None if unusable."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        got = json.loads(m.group(0))
    except ValueError:
        return None
    if not isinstance(got, dict):
        return None
    if got.get("done") is True:
        return {"done": True, "why": str(got.get("why", ""))[:80]}
    if keys_ok and isinstance(got.get("drive"), dict):
        try:
            bias = max(-1.0, min(1.0, float(got["drive"].get("bias", 0.0))))
            speed = max(0.0, min(1.0, float(got["drive"].get("speed", 1.0))))
        except (TypeError, ValueError):
            return None
        return {"drive": True, "bias": bias, "speed": speed,
                "why": str(got.get("why", ""))[:80]}
    if keys_ok and isinstance(got.get("keys"), list):
        import laptop_research as lr
        keys = [str(k).lower() for k in got["keys"]][:3]
        if not keys or any(k not in lr.GAME_KEYS for k in keys):
            return None
        try:
            ms = int(float(got.get("ms", 500)))
        except (TypeError, ValueError):
            return None
        ms = min(max(ms, 100), int(getattr(config, "GAME_STEP_MAX_MS", 1500)))
        return {"keys": keys, "ms": ms, "keep": got.get("keep") is True,
                "why": str(got.get("why", ""))[:80]}
    try:
        x, y = float(got["x"]), float(got["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return None
    if y < TITLE_BAND:
        # The picture includes the window's own title bar -- minimise, maximise,
        # CLOSE (live Solitaire, 2026-10-20). Never clicked, whatever she says.
        return None
    return {"x": x, "y": y, "why": str(got.get("why", ""))[:80]}


def _say(text: str) -> None:
    try:
        import nav_voice
        nav_voice.VOICE.say_text(text)
    except Exception as exc:                        # noqa: BLE001
        log.info("game line not spoken (%s): %s", exc, text)


def pause_key(game: str) -> str | None:
    """The key that pauses this game (config.GAME_PAUSE_KEYS, by a word of its
    name), or None: then the game runs on while she thinks."""
    words = game_words(game)
    for word, key in (getattr(config, "GAME_PAUSE_KEYS", {}) or {}).items():
        if word.lower() in words:
            return key
    return None


def _wait_in_front(game: str, stop: threading.Event, wait_s: float) -> bool:
    """After opening it: is the game the window in front within wait_s?"""
    import laptop_research as lr
    end = time.time() + wait_s
    while time.time() < end:
        if game_in_front(lr.idle(), lr.known_apps()) == game:
            return True
        if stop.wait(1.0):
            return False
    return False


def _loop(game: str, stop: threading.Event, say=_say, fast: bool = False,
          wait_s: float = 0.0) -> str:
    import laptop_research as lr
    import scene_describe
    if wait_s and not _wait_in_front(game, stop, wait_s):
        why = "stopped" if stop.is_set() else "it didn't open in front"
        _play["why"] = why
        _publish(False)
        say("Okay, your turn." if why == "stopped" else f"I couldn't play {game}: {why}.")
        return why
    if fast:
        max_moves = int(getattr(config, "GAME_PLAY_MAX_STEPS", 200))
        pause = 0.0
        system = KEYS_SYSTEM.format(game=game, keys=", ".join(lr.GAME_KEYS),
                                    max_ms=int(getattr(config, "GAME_STEP_MAX_MS", 1500)))
    else:
        max_moves = int(getattr(config, "GAME_PLAY_MAX_MOVES", 60))
        pause = float(getattr(config, "GAME_PLAY_PAUSE_S", 1.5))
        system = PLAY_SYSTEM.format(game=game)
    pkey = pause_key(game) if fast else None
    paused = False
    # The fast half (helper 10): the laptop drives between her looks.
    auto = autopilot_for(game) if fast else None
    if auto and lr.helper_version() < 10:
        auto = None
    if auto:
        system = DRIVE_SYSTEM.format(game=game, keys=", ".join(lr.GAME_KEYS))
        pause = float(getattr(config, "GAME_DRIVE_LOOK_S", 1.0))
        got, err = lr.autopilot(game, **auto, for_s=float(config.GAME_AUTOPILOT_FOR_S))
        if got is None:
            log.info("autopilot did not start (%s): plain steps", err)
            auto = None
            system = KEYS_SYSTEM.format(game=game, keys=", ".join(lr.GAME_KEYS),
                                        max_ms=int(getattr(config, "GAME_STEP_MAX_MS", 1500)))
    recent: list[str] = []
    bad = 0
    png = None
    why = "I've made my moves"
    for n in range(max_moves):
        if stop.is_set():
            why = "stopped"
            break
        if png is None:
            png, err = lr.screen(game)
            if not png:
                why = ("the game isn't in front any more" if "not the window in front" in err
                       else f"I can't see the game ({err})")
                break
        hint = (" My last moves: " + "; ".join(recent[-4:]) + ".") if recent else ""
        text, err = scene_describe._call("game_play", png, system,
                                         f"Your move {n + 1} in {game}.{hint}", 200,
                                         media_type="image/png")
        if text is None:
            why = ("my API credits have run out" if err == "no API credit"
                   else f"I couldn't think ({err})")
            break
        move = parse_move(text, keys_ok=fast)
        if move and move.get("drive") and not auto:
            move = None                                 # no driver for this game
        if move is None:
            bad += 1
            if bad >= 3:
                why = "I got confused"
                break
            png = None
            continue
        if move.get("done"):
            why = move.get("why") or "the game is over"
            break
        if stop.is_set():
            why = "stopped"
            break
        if move.get("drive"):
            # Directions for the laptop's driver; it keeps driving while she
            # looks and thinks again. Renewing it is what keeps it alive.
            got, err = lr.autopilot(game, **(auto or {"mode": "road"}), bias=move["bias"],
                                    speed=move["speed"],
                                    for_s=float(config.GAME_AUTOPILOT_FOR_S))
            png = None
            label = f"drive bias {move['bias']:+.1f} speed {move['speed']:.1f}"
            if got is None:
                why = ("the game isn't in front any more" if "not the window in front" in err
                       else f"the autopilot would not drive ({err})")
                break
        elif "keys" in move:
            # One step: (un-pause,) hold the keys, the next picture, (pause).
            # "keep": they stay down while she decides the next one (helper 9).
            kept = (int(getattr(config, "GAME_KEEP_MS", 4000))
                    if move.get("keep") and not pkey else 0)
            png, err = lr.step(game, move["keys"], move["ms"],
                               before=pkey if paused else None, after=pkey, shot=True,
                               keep=kept)
            paused = bool(pkey)
            label = f"{'+'.join(move['keys'])} {move['ms']} ms"
            if png is None:
                why = ("the game isn't in front any more" if "not the window in front" in err
                       else f"I couldn't press the keys ({err})")
                break
        else:
            if paused:
                lr.step(game, [], 0, before=pkey, shot=False)
                paused = False
            ok, err = lr.click(game, move["x"], move["y"])
            png = None
            label = f"click ({move['x']:.2f}, {move['y']:.2f})"
            if not ok:
                why = ("the game isn't in front any more" if "not the window in front" in err
                       else f"I couldn't click ({err})")
                break
        _play["moves"] = n + 1
        recent.append(f"{label} {move['why']}".strip())
        _publish(True, recent[-1])
        if stop.wait(pause):
            why = "stopped"
            break
    if fast:
        try:
            if auto:
                lr.autopilot(game, "off")
            lr.step(game, [], 0, shot=False)          # let go of any kept keys now
        except Exception:                           # noqa: BLE001
            pass
    _play["why"] = why
    _publish(False, recent[-1] if recent else "")
    log.info("game play of %s ended after %d moves: %s", game, _play["moves"], why)
    if why == "stopped":
        say("Okay, your turn.")
    else:
        say(f"I'm done playing {game}: {why}.")
    return why


def play_start(game_hint: str = "", say=_say) -> str:
    """Start playing (his command): the game he named -- opened first if it is
    not in front -- or else the game in front. Any game: slow ones by
    clicking, the rest with the keyboard (helper 8). Returns at once."""
    import laptop_research as lr
    import rate_limit
    if playing():
        return f"I'm already playing {_play['game']}. Say stop to take over."
    if _credits_out():
        return "My API credits have run out, so I can't see the game to play it."
    info = lr.idle()
    if info is None:
        return ("I can't see your laptop's screen. Turn it on, or copy the new "
                "friday_laptop_agent.py to it.")
    apps = lr.known_apps()
    game = game_in_front(info, apps)
    wait_s = 0.0
    if game_hint:
        app, close = lr.match_app(game_hint, [a for a in apps if lr.is_game(a)])
        if app is None and close:
            return "Which one: " + ", ".join(a["name"] for a in close) + "?"
        if app is None and game is None:
            return f"I can't find {game_hint} on your laptop."
        if app is not None and app["name"] != game:
            fast_named = not SLOW_GAMES.search(app["name"])
            if fast_named and lr.helper_version() < 8:
                return _OLD_HELPER.format(game=app["name"])
            ok, why = rate_limit.allow("laptop_launch")
            if not ok:
                return f"I've opened enough apps for now ({why})."
            rate_limit.used("laptop_launch")
            ok, why = lr.launch(app["name"])
            if not ok:
                return f"I couldn't open {app['name']} ({why})."
            game, wait_s = app["name"], float(getattr(config, "GAME_OPEN_WAIT_S", 45.0))
    if game is None:
        return "Open the game first and leave it in front, then tell me to play."
    fast = not SLOW_GAMES.search(game)
    if fast and not wait_s and lr.helper_version() < 8:
        return _OLD_HELPER.format(game=game)
    stop = threading.Event()
    t = threading.Thread(target=_loop, args=(game, stop, say, fast, wait_s),
                         name="laptop-game", daemon=True)
    with _play_lock:
        _play.update(thread=t, stop=stop, game=game, moves=0, why="", fast=fast)
    _publish(True, "opening it" if wait_s else "")
    t.start()
    opening = f"Opening {game}. " if wait_s else ""
    if fast:
        return (f"{opening}Okay, my turn! I'm playing {game}. I think between moves, "
                f"so I'm slow at fast games, but here goes! Say stop when you want it back.")
    return f"{opening}Okay, my turn! I'm playing {game}. Say stop when you want it back."


_OLD_HELPER = ("To play {game} I need to press its keys, and your laptop's helper is "
               "too old for that. Copy the new friday_laptop_agent.py to the laptop.")
