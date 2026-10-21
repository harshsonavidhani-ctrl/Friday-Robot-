"""laptop_research.py — File 20: Friday uses Harsh's laptop, on his word.

    laptop_research.available()             is the laptop helper up? (one look)
    laptop_research.open_search(q, site, play)  Google / YouTube on his screen
    laptop_research.start_notes(topic)      research + notes; returns at once
    laptop_research.finished()              notes jobs not yet announced
    laptop_research.latest(q)               the newest finished job (for detail)

Harsh, 2026-09-27: "just access my laptop and search the web or any like
youtube and the friday brain will control when voice command received", and
"search the web in laptop only". The laptop side is
laptop/friday_laptop_agent.py, a small helper in HIS Windows login (it starts
at login) that only opens a Google/YouTube search it builds itself and saves
notes files. Nothing here can send it a URL, a file name or a command.

THE LAPTOP ONLY. There is no fallback on the Pi: laptop not reachable -> she
says so and nothing else happens (no search, no notes written anywhere).

NOTES ARE ASYNCHRONOUS. Writing them (Haiku + web search, her key, under the
daily cap) outlasts STAGE_TOOL_TIMEOUT_S, so laptop_notes checks the laptop
is up, starts the job on a thread in the MCP server and returns at once. The
job makes ONE model call chain and ONE save -- never a retry -- and writes
state/research/<id>.json; ReminderWatch (reminders.py) says the result
between turns. Every saved note is remembered as kind "research".

SECRETS: secrets/laptop_agent.json {url, token, mac} -- gitignored, never in
code.

THE LAPTOP'S ADDRESS CHANGES (hostel DHCP) and Windows answers no name
lookups there (mDNS tried, 2026-09-27). So when the saved url stops
answering, available() tries at most LAPTOP_REFIND_MAX other addresses where
the laptop was last seen -- the Pi's ARP table for its network card's `mac`,
then the newest SSH logins to the Pi (he works on the Pi FROM the laptop) --
and adopts one ONLY if the helper there answers /health with our token.

THE LAPTOP SAYS WHERE IT IS (2026-09-30). Both of those only know the new
address AFTER the laptop has talked to the Pi; that evening it moved three
times and had not, so she said "not available" with the helper running. The
helper now sends a small signed UDP hello every 30 s ("I am at ip:port",
never the token) and start_listener() -- a thread in friday-mcp -- hears it.
A hello is only a hint: it must be signed with our token, fresh, and come
from the address it names, and the address is saved ONLY after the helper
there answers /health with our token, the same rule as above. So the saved
url is already right when he asks. state/laptop_seen.json is its last hello.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import config

log = logging.getLogger("friday.laptop_research")

KIND = "research"

NOTES_SYSTEM = (
    "You write study notes for Harsh, researched on the web. Search several "
    "reliable sources first. Then reply in EXACTLY this shape:\n\n"
    "SUMMARY:\n<at most 40 words, plain spoken English -- it is read aloud: "
    "no URLs, no markdown>\n\n"
    "NOTES:\n<Markdown notes: a # title, short sections with ## headings, "
    "bullet points, key numbers and dates, caveats, and a ## Sources list of "
    "the pages you used>")


def _settings() -> dict | None:
    try:
        data = json.loads(Path(config.LAPTOP_AGENT_FILE).read_text())
    except (OSError, ValueError):
        return None
    if not data.get("url") or not data.get("token"):
        return None
    return data


def configured() -> bool:
    return _settings() is not None


def _call(base: str, token: str, path: str, body: dict | None,
          timeout: float) -> dict:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        base.rstrip("/") + path, data=data,
        method="POST" if body is not None else "GET",
        headers={"X-Friday-Token": token,
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return json.loads(reply.read().decode("utf-8", "replace"))


def _request(path: str, body: dict | None, timeout: float) -> dict:
    """One HTTP call to the laptop helper. Raises on any failure."""
    s = _settings()
    if s is None:
        raise ConnectionError("the laptop is not set up "
                              "(secrets/laptop_agent.json)")
    return _call(s["url"], s["token"], path, body, timeout)


def _health(timeout: float, base: str | None = None) -> tuple[bool, str]:
    """Is the helper up at the saved url -- or at `base`, another address it
    may have now, which is asked WITHOUT being saved."""
    try:
        if base is None:
            got = _request("/health", None, timeout)
        else:
            got = _call(base, (_settings() or {}).get("token", ""),
                        "/health", None, timeout)
    except urllib.error.HTTPError as exc:
        return False, f"the laptop refused us ({exc.code})"
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"
    return bool(got.get("ok")), ""


def _sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True,
                              timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def candidates(host_now: str) -> list[str]:
    """Other addresses the laptop may have now, most likely first."""
    s = _settings() or {}
    mac = str(s.get("mac") or "").lower()
    found: list[str] = []
    said = seen()
    if said and time.time() - said["at"] <= float(config.LAPTOP_HELLO_MAX_AGE_S):
        found.append(said["ip"])            # where it just said it is
    if mac:
        for line in _sh(["ip", "neigh", "show"]).splitlines():
            if mac in line.lower() and "FAILED" not in line:
                found.append(line.split()[0])
    logins = _sh(["journalctl", "-u", "ssh", "-n", "300", "--no-pager",
                   "-o", "cat"])
    found += reversed(re.findall(r"Accepted \S+ for \S+ from "
                                 r"(\d+\.\d+\.\d+\.\d+)", logins))
    out: list[str] = []
    for ip in found:
        if ip != host_now and ip not in out:
            out.append(ip)
    return out[:int(config.LAPTOP_REFIND_MAX)]


def _save_url(url: str) -> None:
    path = Path(config.LAPTOP_AGENT_FILE)
    data = json.loads(path.read_text())
    data["url"] = url
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.chmod(0o600)
    tmp.replace(path)


def available() -> tuple[bool, str]:
    """Is the helper up and accepting our token? One quick look at the saved
    address; if that fails, a few quick looks where the laptop was last seen."""
    ok, why = _health(float(config.LAPTOP_HEALTH_TIMEOUT_S))
    s = _settings()
    if ok or s is None or "refused us" in why:
        return ok, why
    parts = urllib.parse.urlsplit(s["url"])
    for ip in candidates(parts.hostname or ""):
        moved = parts._replace(netloc=f"{ip}:{parts.port or 8765}").geturl()
        if _health(float(config.LAPTOP_REFIND_TIMEOUT_S), moved)[0]:
            _save_url(moved)
            log.info("the laptop moved: %s -> %s", parts.hostname, ip)
            return True, ""
    return False, why


# --- the laptop says where it is ------------------------------------------------

def _sign(token: str, ts: int, ip: str, port: int) -> str:
    # The same line as laptop/friday_laptop_agent.py::sign (the helper is
    # copied to the laptop alone, so it cannot be imported from here).
    text = f"friday-laptop|{ts}|{ip}|{port}"
    return hmac.new(token.encode(), text.encode(), "sha256").hexdigest()


def read_hello(data: bytes, sender: str,
               now: float | None = None) -> tuple[dict | None, str]:
    """({ip, port, via}, "") for a hello from OUR helper, else (None, why)."""
    s = _settings()
    if s is None:
        return None, "the laptop is not set up"
    try:
        got = json.loads(data[:1024])
        ts, port = int(got["ts"]), int(got["port"])
        ip = str(ipaddress.IPv4Address(got["ip"]))
        sig = str(got["sig"])
        if got.get("friday") != "laptop" or not 0 < port < 65536:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        return None, "not a hello"
    if not hmac.compare_digest(sig.encode(),
                               _sign(s["token"], ts, ip, port).encode()):
        return None, "not signed with our token"
    age = (time.time() if now is None else now) - ts
    if abs(age) > float(config.LAPTOP_HELLO_MAX_AGE_S):
        return None, f"{age:+.0f} s old (a replay, or a clock is wrong)"
    if ip != sender:
        return None, f"says {ip} but came from {sender}"
    return {"ip": ip, "port": port, "via": str(got.get("via") or "")[:12]}, ""


def seen() -> dict | None:
    """The laptop's last hello: {ip, port, via, at}."""
    try:
        got = json.loads(Path(config.LAPTOP_SEEN_FILE).read_text())
        return {"ip": str(got["ip"]), "port": int(got["port"]),
                "via": str(got.get("via", "")), "at": float(got["at"])}
    except (OSError, ValueError, TypeError, KeyError):
        return None


_tried: dict[str, float] = {}          # url -> when it last did not answer


def note_hello(data: bytes, sender: str) -> tuple[bool, str]:
    """One datagram. A good hello is written down, and when it names an
    address other than the saved one that address is asked for /health and
    saved if our helper answers. (it was a hello from our laptop?, why not)."""
    hello, why = read_hello(data, sender)
    if hello is None:
        return False, why
    last = seen()
    now = time.time()
    if last is None or last["ip"] != hello["ip"] or now - last["at"] >= 20.0:
        path = Path(config.LAPTOP_SEEN_FILE)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps({**hello, "at": now}))
        tmp.replace(path)
    s = _settings() or {}
    url = f"http://{hello['ip']}:{hello['port']}"
    if s.get("url", "").rstrip("/") == url:
        return True, ""
    if now - _tried.get(url, 0.0) < float(config.LAPTOP_HELLO_RETRY_S):
        return True, ""
    if _health(float(config.LAPTOP_REFIND_TIMEOUT_S), url)[0]:
        _tried.pop(url, None)
        _save_url(url)
        log.info("the laptop said hello from a new address: %s -> %s",
                 s.get("url"), url)
    else:
        _tried.clear()                      # bounded: one address at a time
        _tried[url] = now
        log.info("a hello from %s, but no helper answered there", url)
    return True, ""


def start_listener(port: int | None = None) -> int:
    """Hear the helper's hellos on a daemon thread. Returns the UDP port it
    is listening on, or 0 when it is switched off or the port is taken."""
    if not config.LAPTOP_HELLO:
        return 0
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", int(config.LAPTOP_HELLO_PORT
                                  if port is None else port)))
    except OSError as exc:
        sock.close()
        log.warning("not listening for the laptop's hello: %s", exc)
        return 0

    def listen() -> None:
        said = 0.0
        while True:
            try:
                data, (sender, _) = sock.recvfrom(2048)
                ok, why = note_hello(data, sender)
                if not ok and time.time() - said > 60.0:    # never a log flood
                    said = time.time()
                    log.info("a hello from %s was ignored: %s", sender, why)
            except Exception as exc:                        # noqa: BLE001
                log.warning("hello listener: %s: %s", type(exc).__name__, exc)
                time.sleep(1.0)

    threading.Thread(target=listen, name="laptop-hello", daemon=True).start()
    bound = sock.getsockname()[1]
    log.info("listening for the laptop's hello on udp %s", bound)
    return bound


# The helper's POST /key names (helper 4+). The laptop keeps its own copy and
# refuses anything else; this one only stops a typo reaching the network.
KEY_ACTIONS = ("play_pause", "next", "previous", "volume_up", "volume_down",
               "mute", "fullscreen", "close_tab", "lock", "open_notes")


def press_key(action: str) -> tuple[bool, str]:
    """One fixed key on his laptop. (done?, why not). 'old helper' when the
    laptop still runs a helper from before /key existed (it answers 404)."""
    if action not in KEY_ACTIONS:
        return False, "unknown action"
    try:
        got = _request("/key", {"action": action},
                       float(config.LAPTOP_OPEN_TIMEOUT_S))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False, "old helper"
        try:
            why = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except Exception:                           # noqa: BLE001
            why = ""
        return False, why or f"the laptop said {exc.code}"
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"
    return bool(got.get("ok")), got.get("error", "")


# --- apps on his laptop (helper 5+) -----------------------------------------------
# Harsh, 2026-10-20: "open every app ... open whatsapp, open this game ... she
# knows what's in my laptop". The laptop lists its own Start menu; the Pi keeps
# the last list (config.LAPTOP_APPS_FILE) so she still knows it when it is off.

# A game TITLE anywhere in the name is a game ("Minecraft Launcher" is how you
# play Minecraft). The words game/games alone only count when the name is not
# a tool or a store: 2026-10-20 his list had "Game Bar", "Google Play Games"
# and "Rockstar Games Launcher" called games.
GAME_TITLES = re.compile(
    r"minecraft|roblox|valorant|fortnite|gta|grand theft|counter[- ]strike|cs ?2|"
    r"pubg|bgmi|free ?fire|apex|overwatch|league of legends|dota|rocket league|"
    r"fifa|ea (sports )?fc|need for speed|forza|asphalt|subway surfers|"
    r"hill climb|candy crush|solitaire|minesweeper|chess|among us|genshin|"
    r"call of duty|\bcod\b|battlefield|assassin|far cry|witcher|cyberpunk|"
    r"hollow knight|terraria|stardew|cuphead|fall guys|pinball|sudoku|"
    r"evil nun|crimson snow|dcs world|flight simulator|racing|"
    # 2026-10-21, his laptop: Granny 3, Granny Chapter Two, Ice Scream 1 Scary
    # Horror, The Twins were not games to her ("you play Granny" -> not found).
    r"granny|ice scream|the twins|scary|horror|zombie|\bkart\b|drift", re.IGNORECASE)
GAME_WORDS = re.compile(r"\bgames?\b", re.IGNORECASE)
NOT_A_GAME = re.compile(r"launcher|game bar|play games|store|xbox|settings|"
                        r"\bbar\b|helper|services|overlay", re.IGNORECASE)
MAINTENANCE = re.compile(r"^(?:repair|update|uninstall|configure|setup|install)\b|"
                         r"\b(?:manual|documentation|readme|support|website|"
                         r"benchmark|crash reporter)\b", re.IGNORECASE)
LAUNCHERS = re.compile(r"^(steam|epic games launcher|xbox|ea app|ubisoft connect|"
                       r"battle\.net|riot client|gog galaxy)$", re.IGNORECASE)


def _save_apps(apps: list[dict]) -> None:
    path = Path(config.LAPTOP_APPS_FILE)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.tmp")
        tmp.write_text(json.dumps({"at": time.time(), "apps": apps}))
        tmp.replace(path)
    except OSError as exc:
        log.warning("could not keep the laptop's app list: %s", exc)


def known_apps() -> list[dict]:
    """The last app list the laptop gave (may be old; [] if never)."""
    try:
        return list(json.loads(Path(config.LAPTOP_APPS_FILE).read_text()).get("apps") or [])
    except (OSError, ValueError):
        return []


def apps(refresh: bool = True) -> tuple[list[dict], str]:
    """(apps, why not fresh). Asks the laptop when `refresh`; falls back to the
    last list it gave. 'old helper' when the laptop has no /apps yet."""
    if refresh:
        try:
            got = _request("/apps", None, float(config.LAPTOP_OPEN_TIMEOUT_S))
            listed = [{"name": str(a.get("name", "")), "via": str(a.get("via", ""))}
                      for a in got.get("apps") or [] if a.get("name")]
            if listed:
                _save_apps(listed)
                return listed, ""
        except urllib.error.HTTPError as exc:
            return known_apps(), ("old helper" if exc.code == 404 else f"laptop said {exc.code}")
        except Exception as exc:                    # noqa: BLE001
            return known_apps(), f"{type(exc).__name__}"
    return known_apps(), "not asked"


def is_game(app: dict) -> bool:
    name = str(app.get("name", ""))
    if LAUNCHERS.match(name) or MAINTENANCE.search(name):
        return False
    if app.get("via") in ("steam", "epic") or GAME_TITLES.search(name):
        return True
    return bool(GAME_WORDS.search(name)) and not NOT_A_GAME.search(name)


_GENERIC = re.compile(r"\b(?:launcher|app|application|desktop|for windows|beta|preview)\b")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# What people call a game -> words in its real name.
NICKNAMES = {"gta": "grand theft auto", "gta 5": "grand theft auto", "gta v": "grand theft auto",
             "nfs": "need for speed", "cod": "call of duty", "cs": "counter strike",
             "cs go": "counter strike", "ff": "free fire", "free fire max": "free fire",
             "lol": "league of legends", "mc": "minecraft"}

# What Whisper writes for an app's name -> the name (2026-10-20 22:28-22:35:
# "open WhatsApp" came out as "open watch app" and "open words app", and she
# said "I don't see watch on your laptop" three times).
MISHEARD = {w: "whatsapp" for w in (
    "watch app", "watch apps", "words app", "word app", "what s app", "whats app",
    "what app", "wats app", "watts app", "what sapp", "watsapp", "wattsapp",
    "whatsup", "whats up", "what s up", "watch up")}


def match_app(spoken: str, listed: list[dict]) -> tuple[dict | None, list[dict]]:
    """(the app, other close ones). Exact name, then the name starting with what
    he said, then every word he said in the name; ONE hit only -- two equally
    good hits come back as (None, [both]) so she can ask which."""
    want = _norm(re.sub(r"^(?:the|my)\s+", "", spoken.strip(), flags=re.IGNORECASE))
    want = re.sub(r"\s+(?:on my laptop|on the laptop)$", "", want)
    want = MISHEARD.get(want, want)
    want = re.sub(r"\s+(?:app|application|game)$", "", want)
    want = NICKNAMES.get(want, want)
    if not want:
        return None, []
    names = [(_norm(a["name"]), a) for a in listed]
    for rule in (lambda n: n == want,
                 lambda n: n.startswith(want + " ") or n.startswith(want),
                 lambda n: all(w in n.split() for w in want.split()),
                 lambda n: want.replace(" ", "") in n.replace(" ", "")):
        hits = [a for n, a in names if rule(n)]
        if len(hits) == 1:
            return hits[0], []
        if len(hits) > 1:
            # Only a GENERIC extra word decides it ("minecraft" -> "Minecraft
            # Launcher" over "Minecraft Dungeons" would still be a guess, so it
            # asks): the one hit that IS what he said plus launcher/app/...
            plain = [a for a in hits if _GENERIC.sub("", _norm(a["name"])).strip() == want]
            if len(plain) == 1:
                return plain[0], [a for a in hits if a is not plain[0]][:3]
            return None, sorted(hits, key=lambda a: a["name"].lower())[:4]
    # Nothing by that name: was it a mishearing? The voice rule has already
    # dropped "app" ("open watch app" arrives as "watch").
    alias = MISHEARD.get(want + " app")
    if alias and alias != want:
        return match_app(alias, listed)
    return None, []


def launch(name: str) -> tuple[bool, str]:
    """Start one app by its exact listed name. (started?, why not)."""
    try:
        got = _request("/launch", {"name": name}, float(config.LAPTOP_OPEN_TIMEOUT_S))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            try:
                why = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
            except Exception:                       # noqa: BLE001
                why = ""
            return False, ("no such app" if why == "no such app" else "old helper")
        return False, f"the laptop said {exc.code}"
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"
    return bool(got.get("ok")), got.get("error", "")


def screen(game: str) -> tuple[bytes | None, str]:
    """A PNG of the game window -- the laptop refuses (409) unless that game is
    the window in front. (png, "") or (None, why)."""
    s = _settings()
    if s is None:
        return None, "the laptop is not set up"
    url = (s["url"].rstrip("/") + "/screen?" + urllib.parse.urlencode({"game": game}))
    req = urllib.request.Request(url, headers={"X-Friday-Token": s["token"]})
    try:
        with urllib.request.urlopen(req, timeout=float(config.LAPTOP_OPEN_TIMEOUT_S)) as reply:
            if reply.headers.get("Content-Type", "") != "image/png":
                return None, "not a picture"
            return reply.read(), ""
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, "old helper"
        try:
            why = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except Exception:                           # noqa: BLE001
            why = ""
        return None, why or f"the laptop said {exc.code}"
    except Exception as exc:                        # noqa: BLE001
        return None, f"{type(exc).__name__}"


def click(game: str, x: float, y: float) -> tuple[bool, str]:
    """One click inside the game's window, at fractions of it. (done?, why)."""
    try:
        got = _request("/click", {"game": game, "x": float(x), "y": float(y)},
                       float(config.LAPTOP_OPEN_TIMEOUT_S))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False, "old helper"
        try:
            why = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except Exception:                           # noqa: BLE001
            why = ""
        return False, why or f"the laptop said {exc.code}"
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}"
    return bool(got.get("ok")), got.get("error", "")


# The helper's own GAME_KEYS (helper 8): the only key names /step takes.
GAME_KEYS = ("up", "down", "left", "right", "space", "enter", "escape", "shift",
             "w", "a", "s", "d", "n", "v", "x", "z",
             "f", "e", "q", "r", "c", "g", "h", "tab")          # helper 9: the last row


def helper_version() -> int:
    """The laptop helper's version (0 when it cannot be read)."""
    try:
        got = _request("/health", None, float(config.LAPTOP_HEALTH_TIMEOUT_S))
    except Exception:                               # noqa: BLE001
        return 0
    try:
        return int(got.get("version") or 0)
    except (TypeError, ValueError):
        return 0


def step(game: str, keys: list[str], ms: int, before: str | None = None,
         after: str | None = None, shot: bool = True,
         keep: int = 0) -> tuple[bytes | None, str]:
    """One move in a real-time game (helper 8): tap `before`, hold `keys`
    together for `ms`, picture, tap `after`. The laptop refuses unless that
    game is the window in front, and lets go of the keys the moment it is not.
    keep (helper 9): ms the keys stay down AFTER the picture, until the next
    step -- so the car does not stop while she decides.
    (png -- b"" with shot=False -- , "") or (None, why)."""
    s = _settings()
    if s is None:
        return None, "the laptop is not set up"
    sent = {"game": game, "keys": list(keys), "ms": int(ms), "before": before,
            "after": after, "shot": bool(shot)}
    if keep:
        sent["keep"] = int(keep)            # helper 8 ignores it: keys let go as before
    body = json.dumps(sent).encode()
    req = urllib.request.Request(s["url"].rstrip("/") + "/step", data=body, method="POST",
                                 headers={"X-Friday-Token": s["token"],
                                          "Content-Type": "application/json"})
    timeout = float(config.LAPTOP_OPEN_TIMEOUT_S) + int(ms) / 1000.0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as reply:
            if reply.headers.get("Content-Type", "") == "image/png":
                return reply.read(), ""
            got = json.loads(reply.read().decode("utf-8", "replace") or "{}")
            return (b"", "") if got.get("ok") else (None, got.get("error", "no"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, "old helper"
        try:
            why = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except Exception:                           # noqa: BLE001
            why = ""
        return None, why or f"the laptop said {exc.code}"
    except Exception as exc:                        # noqa: BLE001
        return None, f"{type(exc).__name__}"


def autopilot(game: str, mode: str, **settings) -> tuple[dict | None, str]:
    """Start / steer / stop the laptop's own fast driver (helper 10, POST
    /autopilot): mode "road" follows the minimap's road, "cruise" holds
    settings["hold"], "off" stops. settings: box, keys, hold, bias (-1 left ..
    1 right), speed (0..1), for_s. (its report, "") or (None, why)."""
    body = {"game": game, "mode": mode, **settings}
    try:
        got = _request("/autopilot", body, float(config.LAPTOP_OPEN_TIMEOUT_S))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None, "old helper"
        try:
            why = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except Exception:                           # noqa: BLE001
            why = ""
        return None, why or f"the laptop said {exc.code}"
    except Exception as exc:                        # noqa: BLE001
        return None, f"{type(exc).__name__}"
    return got, ""


def look(game: str, dx: int, dy: int = 0) -> tuple[bool, str]:
    """Turn a 3D game's camera: move the laptop's mouse by (dx, dy), no click,
    only while that game is in front (helper 15, POST /look). (done?, why)."""
    try:
        got = _request("/look", {"game": game, "dx": int(dx), "dy": int(dy)},
                       float(config.LAPTOP_OPEN_TIMEOUT_S))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False, "old helper"
        try:
            why = json.loads(exc.read().decode("utf-8", "replace")).get("error", "")
        except Exception:                           # noqa: BLE001
            why = ""
        return False, why or f"the laptop said {exc.code}"
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}"
    return bool(got.get("ok")), got.get("error", "")


def autopilot_status() -> dict | None:
    try:
        return _request("/autopilot", None, float(config.LAPTOP_HEALTH_TIMEOUT_S))
    except Exception:                               # noqa: BLE001
        return None


def idle() -> dict | None:
    """{"idle_s", "app"} from the laptop, or None (down, or an old helper)."""
    try:
        got = _request("/idle", None, float(config.LAPTOP_HEALTH_TIMEOUT_S))
    except Exception:                               # noqa: BLE001
        return None
    return got if got.get("idle_s") is not None else None


def open_search(query: str, site: str = "web",
                play: bool = False) -> tuple[bool, str]:
    """Google or YouTube on his screen. (opened?, why not)."""
    try:
        got = _request("/open", {"site": site, "query": query, "play": play},
                       float(config.LAPTOP_OPEN_TIMEOUT_S))
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"
    return bool(got.get("ok")), got.get("error", "")


# --- notes jobs -------------------------------------------------------------------

def _dir() -> Path:
    path = Path(config.RESEARCH_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(job: dict) -> None:
    path = _dir() / f"{job['id']}.json"
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(job))
    tmp.replace(path)
    # BOUNDED (file 09's rule): keep the newest RESEARCH_KEEP jobs.
    for old in sorted(_dir().glob("*.json"))[:-int(config.RESEARCH_KEEP)]:
        old.unlink(missing_ok=True)


def jobs() -> list[dict]:
    out = []
    for path in sorted(_dir().glob("*.json")):
        try:
            out.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return out


def finished() -> list[dict]:
    """Jobs that ended (done / failed) and were not said yet."""
    return [j for j in jobs() if j.get("status") in ("done", "failed")
            and not j.get("announced_at")]


def mark_announced(job_id: str) -> None:
    for job in jobs():
        if job["id"] == job_id:
            job["announced_at"] = time.time()
            _write(job)


def latest(query: str = "") -> dict | None:
    """The newest saved notes, or the newest whose topic shares words."""
    done = [j for j in jobs() if j.get("status") == "done"]
    if query.strip():
        want = set(re.findall(r"\w{3,}", query.lower()))
        done = [j for j in done
                if want & set(re.findall(r"\w{3,}", j["query"].lower()))]
    return done[-1] if done else None


def _remember(job: dict) -> None:
    try:
        import memory
        text = (f"Notes ({time.strftime('%Y-%m-%d', time.localtime(job['at']))})"
                f" on {job['query']}: {job.get('summary', '').strip()} Saved "
                f"on the laptop at {job['path']}.")
        memory.remember(text, kind=KIND, source="laptop")
    except Exception as exc:                        # noqa: BLE001
        log.warning("notes not remembered: %s", exc)


def start_notes(topic: str) -> dict:
    """Research + save on a thread. The job record is returned at once."""
    job = {"id": f"{int(time.time() * 1000)}", "query": topic.strip(),
           "mode": "notes", "status": "running", "at": time.time()}
    _write(job)
    threading.Thread(target=_run, args=(job,), name="laptop-notes",
                     daemon=True).start()
    return job


def _run(job: dict) -> None:
    result, why = compose_notes(job["query"])
    if result is None:
        job.update(status="failed", stage="research", why=why)
    else:
        try:
            got = _request("/notes", {"title": job["query"],
                                      "markdown": result["notes"]},
                           float(config.LAPTOP_SAVE_TIMEOUT_S))
            if not got.get("ok"):
                raise RuntimeError(got.get("error") or "the laptop said no")
            job.update(status="done", summary=result["summary"],
                       detail=result["notes"], path=got.get("path"))
        except Exception as exc:                    # noqa: BLE001
            job.update(status="failed", stage="save",
                       why=f"{type(exc).__name__}: {str(exc)[:120]}",
                       summary=result["summary"], detail=result["notes"])
    job["ended"] = time.time()
    # Remembered BEFORE the record says it ended, so the moment she can
    # announce it, "what did you find out about X" already finds it too.
    if job["status"] == "done":
        _remember(job)
    _write(job)
    log.info("notes %s: %s (%s) %s", job["id"], job["status"],
             job["query"][:60], job.get("why", ""))


def _split(text: str) -> tuple[str, str]:
    found = re.search(r"SUMMARY:\s*(.*?)\s*NOTES:\s*(.*)$", text, re.S | re.I)
    if found:
        return found.group(1).strip(), found.group(2).strip()
    return " ".join(text.split()[:40]), text.strip()


def compose_notes(topic: str) -> tuple[dict | None, str]:
    """({summary, notes}, "") or (None, why). Haiku 4.5 + the web search tool."""
    try:
        import anthropic

        import brain
        import scene_describe
        key = scene_describe._api_key()
        if not key:
            return None, "no API key"
        if brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
            return None, "over the daily spend cap"
        model = config.LAPTOP_NOTES_MODEL
        client = anthropic.Anthropic(api_key=key,
                                     timeout=float(config.LAPTOP_NOTES_TIMEOUT_S))
        messages = [{"role": "user", "content": f"Make notes on: {topic}"}]
        reply = None
        for _ in range(3):
            reply = client.messages.create(
                model=model, max_tokens=int(config.LAPTOP_NOTES_MAX_TOKENS),
                system=NOTES_SYSTEM,
                tools=[{"type": "web_search_20250305", "name": "web_search",
                        "max_uses": int(config.LAPTOP_NOTES_MAX_SEARCHES)}],
                messages=messages)
            brain._record_spend(model, reply.usage,
                                brain._cost_usd(model, reply.usage))
            if reply.stop_reason != "pause_turn":
                break
            messages = [messages[0],
                        {"role": "assistant", "content": reply.content}]
        if reply.stop_reason == "refusal":
            return None, "the model would not research that"
        text = "".join(b.text for b in reply.content
                       if getattr(b, "type", "") == "text").strip()
        if not text:
            return None, "no notes came back"
        summary, notes = _split(text)
        return {"summary": summary, "notes": notes}, ""
    except Exception as exc:                        # noqa: BLE001
        return None, f"{type(exc).__name__}: {str(exc)[:100]}"
