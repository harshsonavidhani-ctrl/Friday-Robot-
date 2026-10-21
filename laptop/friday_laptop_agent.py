#!/usr/bin/env python3
"""Friday's laptop helper (File 20). Runs on HARSH'S WINDOWS LAPTOP, in his login.

Once, in PowerShell, in the folder you copied it to (Python 3 needed):

    python friday_laptop_agent.py --install

From then on it starts by itself, windowless, every time you log in (a
Startup-folder entry), and --install starts it right away too. Windows asks
once whether Python may use the network: tick Private AND Public, Allow.
`--uninstall` removes the Startup entry. Standard library only.

WHAT IT DOES -- AND ONLY THIS
  GET  /health                                    {"ok": true}
  POST /open   {"site": "web" | "youtube" | "store", "query": "...", "play": false}
        Opens a Google or YouTube search for those words in the default
        browser; "play" on YouTube opens the top video instead. "store"
        (helper 8) opens a Microsoft Store SEARCH for the words -- nothing is
        installed unless someone presses Get. Only those, and only a URL
        built HERE from the words -- never a URL, a file name or a command
        sent by the Pi.
  POST /step   {"game", "keys": [...], "ms", "before", "after", "shot", "keep"} (helper 8+)
        Holds up to 3 GAME_KEYS (arrows, space, enter, escape, shift, w a s d
        n v x z; helper 9 adds f e q r c g h tab) together for at most
        STEP_MAX_MS, ONLY while that game is the window in front -- let go the
        moment it is not -- and hands back the next picture. "keep" (helper 9,
        ms, at most KEEP_MAX_MS) leaves them down after the picture until the
        next /step, so a car keeps going while she decides. How she plays a
        fast game, a short step at a time.
  POST /notes  {"title": "...", "markdown": "..."}
        Writes ONE .md file into NOTES_DIR (default ~/Friday_Notes), never
        overwriting, never outside that folder, and opens it in Notepad.
  POST /key    {"action": "<one of KEY_ACTIONS>"}            (helper 4+)
        Presses ONE fixed key from a short list: play/pause, next, previous,
        volume up/down, mute (Windows media keys -- the browser playing
        YouTube answers them even when it is not in front); full screen and
        close tab ("f", Ctrl+W -- ONLY when a browser is the window in front,
        so it can never close a document); lock the laptop; open the notes
        folder. The Pi sends a NAME from that list and nothing else: no key
        codes, no text to type.
  GET  /apps                                                  (helper 5+)
        The apps on THIS laptop's Start menu, by name (Store apps and games
        included): {"apps": [{"name", "via"}]}. Read here with PowerShell's
        Get-StartApps, refreshed at most every APPS_REFRESH_S. System tools
        (APP_DENY: terminals, uninstallers, registry, reset/disk tools...)
        are left out of the list -- and so can never be launched.
  POST /launch {"name": "WhatsApp"}                           (helper 5+)
        Starts ONE app from that list, matched by its exact name (any case),
        through the Start menu's own entry. The Pi sends a NAME; the program
        that starts is the laptop's own Start-menu entry for it, never a path
        or a command from the Pi.
  GET  /idle                                                  (helper 5+)
        {"idle_s": seconds since the keyboard/mouse were last touched, "app":
        the program in front, "title": its window title (helper 6)}. Friday
        plays on the laptop on her own only when nobody has touched it for a
        while (she never interrupts you).
  GET  /screen?game=<name>                                    (helper 6+)
        A PNG of the window in front -- ONLY when its title matches that game
        (a significant word of the name in the title), else 409 and nothing.
        Never the whole desktop, never another window: with WhatsApp or a
        browser in front she gets nothing.
  POST /click {"game": "<name>", "x": 0..1, "y": 0..1}        (helper 6+)
        One left click INSIDE that game's window (its client area, a margin
        kept from the edges), and only while its title matches. How she plays
        a slow game (solitaire, chess) when Harsh says "you play".

  Every request must carry the shared token in the X-Friday-Token header;
  anything without it gets 401 and nothing else. No other endpoints.

IT ALSO SAYS WHERE IT IS. The hostel Wi-Fi gives this laptop a new address
every time it reconnects, and the Pi could only find it again AFTER the
laptop had talked to the Pi (2026-09-30: "My laptop isn't available right
now" with this helper running fine). So every hello_every_s it sends one
small UDP packet -- "I am at <address>, port <port>", signed with the token,
never the token itself -- to the Pi where it last heard from it (the address
of the last request that carried the token, kept in friday_pi.json next to
this script) and to the whole network. It sends nothing else, and it takes
nothing in on that port.

WHY A PROGRAM IN YOUR LOGIN AND NOT SSH: a program started through Windows'
SSH server runs in a session with no desktop, so a browser it opens never
appears on your screen.

CONFIG: friday_agent.json next to this script (copied from the Pi's
secrets/friday_agent.json): {"token": "...", "port": 8765, "bind": "0.0.0.0",
"notes_dir": "~/Friday_Notes"}. Keep it private: it holds the token.
Optional: "hello_port" (8766), "hello_every_s" (30), "hello_broadcast"
(true), "pi" (the Pi's address, only needed before the Pi has ever called).
Log: friday_laptop_helper.log next to this script.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import logging.handlers
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BODY = 256 * 1024
MAX_QUERY = 300
MAX_TITLE = 200

# POST /key: the name the Pi may send -> what is pressed here. Windows
# virtual-key codes; a volume step is 2%, so one "volume up" is 5 presses.
KEY_ACTIONS = {
    "play_pause": ("vk", 0xB3, 1),
    "next": ("vk", 0xB0, 1),
    "previous": ("vk", 0xB1, 1),
    "volume_up": ("vk", 0xAF, 5),
    "volume_down": ("vk", 0xAE, 5),
    "mute": ("vk", 0xAD, 1),
    "fullscreen": ("browser", (0x46,), 1),           # f (YouTube's own toggle)
    "close_tab": ("browser", (0x11, 0x57), 1),       # Ctrl+W
    "lock": ("lock", None, 1),
    "open_notes": ("notes", None, 1),
}
BROWSERS = ("chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe")

# /apps and /launch (helper 5): never listed, so never launched.
APP_DENY = re.compile(
    r"uninstall|command prompt|\bcmd\b|powershell|terminal|windows tools|regedit|"
    r"registry|recovery|\breset\b|disk|format|administrative|control panel|"
    r"task manager|system configuration|\bservices\b|event viewer|device manager|"
    r"computer management|policy|defragment|bitlocker|credential|firewall|"
    r"\bbios\b|firmware|remote desktop|\brun\b|odbc|iscsi|hyper-v|sandbox",
    re.IGNORECASE)
APPS_REFRESH_S = 600.0
# /screen and /click (helper 6): words that do not identify a game window.
_GENERIC_WORDS = {"game", "games", "launcher", "for", "windows", "the", "app",
                  "and", "casual", "edition", "microsoft", "collection", "free"}
SCREEN_MAX_W = 960                         # the picture is scaled down to this
# /step (helper 8): game keys only, held for a moment, ONLY while that game is
# the window in front. name -> (virtual key, scan code, extended key?)
GAME_KEYS = {
    "up": (0x26, 0x48, True), "down": (0x28, 0x50, True),
    "left": (0x25, 0x4B, True), "right": (0x27, 0x4D, True),
    "space": (0x20, 0x39, False), "enter": (0x0D, 0x1C, False),
    "escape": (0x1B, 0x01, False), "shift": (0x10, 0x2A, False),
    "w": (0x57, 0x11, False), "a": (0x41, 0x1E, False), "s": (0x53, 0x1F, False),
    "d": (0x44, 0x20, False), "n": (0x4E, 0x31, False), "v": (0x56, 0x2F, False),
    "x": (0x58, 0x2D, False), "z": (0x5A, 0x2C, False),
    # helper 9 (GTA V: F gets in a car, E horn/ability, Q cover, R reload,
    # C look back, G throw, H headlights, Tab weapon wheel)
    "f": (0x46, 0x21, False), "e": (0x45, 0x12, False), "q": (0x51, 0x10, False),
    "r": (0x52, 0x13, False), "c": (0x43, 0x2E, False), "g": (0x47, 0x22, False),
    "h": (0x48, 0x23, False), "tab": (0x09, 0x0F, False),
}
STEP_MAX_MS = 2000                         # longest a key is ever held
STEP_MAX_KEYS = 3
# /step "keep" (helper 9): the keys STAY DOWN after the picture, for at most
# KEEP_MAX_MS, so a car does not stop while the next move is being decided
# (2026-10-21: "the time it analyzes the car was stopping"). Let go at once
# by the next /step, by the game leaving the front, or when the time is up.
KEEP_MAX_MS = 6000
STARTUP_NAME = "friday_laptop_helper.cmd"
PI_FILE = "friday_pi.json"                  # where the Pi last called from
NOWHERE = "192.0.2.1"                       # never sent to: only picks a route

log = logging.getLogger("friday_laptop")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if not str(cfg.get("token") or "").strip() or len(cfg["token"]) < 24:
        raise SystemExit("friday_agent.json needs a token of at least 24 "
                         "characters (copy it from the Pi).")
    cfg.setdefault("port", 8765)
    cfg.setdefault("bind", "0.0.0.0")
    cfg.setdefault("notes_dir", "~/Friday_Notes")
    cfg.setdefault("hello_port", 8766)
    cfg.setdefault("hello_every_s", 30)
    cfg.setdefault("hello_broadcast", True)
    return cfg


# --- hello: telling the Pi where this laptop is -------------------------------

def sign(token: str, ts: int, ip: str, port: int) -> str:
    text = f"friday-laptop|{ts}|{ip}|{port}"
    return hmac.new(token.encode(), text.encode(), "sha256").hexdigest()


def hello_packet(token: str, ip: str, port: int, via: str = "direct",
                 now: float | None = None) -> bytes:
    """One "I am at ip:port", signed. The token itself is never in it."""
    ts = int(time.time() if now is None else now)
    return json.dumps({"friday": "laptop", "ts": ts, "ip": ip, "port": port,
                       "via": via, "sig": sign(token, ts, ip, port)}).encode()


def own_address(toward: str) -> str:
    """This laptop's address on the network that leads to `toward`. A UDP
    connect sends nothing; it only asks Windows which way it would go."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect((toward, 9))
        return probe.getsockname()[0]


class Hello:
    def __init__(self, cfg: dict, port: int, store: Path) -> None:
        self.cfg = cfg
        self.port = port                    # where the helper listens (TCP)
        self.store = store
        self.pi = str(cfg.get("pi") or "")
        try:
            self.pi = str(json.loads(store.read_text()).get("pi") or self.pi)
        except (OSError, ValueError, AttributeError):
            pass
        self._wake = threading.Event()
        self._said = ""

    def seen(self, ip: str) -> None:
        """A request with the token came from `ip`: that is the Pi."""
        if ip == self.pi:
            return
        self.pi = ip
        try:
            self.store.write_text(json.dumps({"pi": ip}) + "\n")
        except OSError as exc:
            log.info("hello: the Pi's address was not saved (%s)", exc)
        self._wake.set()                    # tell it where we are, now

    def _note(self, text: str) -> None:    # the log keeps changes, not beats
        if text != self._said:
            self._said = text
            log.info("hello: %s", text)

    def beat(self) -> list[str]:
        """One hello. Returns where it was sent."""
        pi = self.pi
        where = [(pi, "direct")] if pi else []
        if self.cfg.get("hello_broadcast", True):
            where.append(("255.255.255.255", "broadcast"))
        sent: list[str] = []
        try:
            ip = own_address(pi or NOWHERE)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
                out.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
                out.bind((ip, 0))           # the card that reaches the Pi
                for target, via in where:
                    try:
                        out.sendto(hello_packet(self.cfg["token"], ip,
                                                self.port, via),
                                   (target, int(self.cfg["hello_port"])))
                        sent.append(target)
                    except OSError:
                        continue
        except OSError as exc:
            self._note(f"no network ({exc})")
            return []
        self._note(f"I am at {ip}, telling {' and '.join(sent) or 'nobody'}")
        return sent

    def run(self) -> None:
        while True:
            try:
                self.beat()
            except Exception as exc:                    # noqa: BLE001
                self._note(f"failed ({type(exc).__name__}: {exc})")
            self._wake.wait(max(5.0, float(self.cfg["hello_every_s"])))
            self._wake.clear()


# --- /open: a search, built here, on one of two sites --------------------------

_VIDEO_ID = re.compile(r'"videoId"\s*:\s*"([A-Za-z0-9_-]{11})"')


def _fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept-Language": "en"})
    with urllib.request.urlopen(req, timeout=8) as reply:
        return reply.read(3_000_000).decode("utf-8", "replace")


def build_url(site: str, query: str, play: bool = False, fetch=_fetch) -> str:
    words = urllib.parse.quote_plus(query)
    if site == "store":                         # helper 8: a Store SEARCH page only
        return f"ms-windows-store://search/?query={urllib.parse.quote(query)}"
    if site == "web":
        return f"https://www.google.com/search?q={words}"
    results = f"https://www.youtube.com/results?search_query={words}"
    if not play:
        return results
    try:
        found = _VIDEO_ID.search(fetch(results))
    except Exception as exc:                            # noqa: BLE001
        log.info("top video not found (%s): opening the results", exc)
        return results
    return f"https://www.youtube.com/watch?v={found.group(1)}" if found \
        else results


# --- /notes: one file, inside its own folder, never anywhere else --------------

def notes_dir(cfg: dict) -> Path:
    path = Path(cfg["notes_dir"]).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_notes(title: str, markdown: str, cfg: dict) -> Path:
    folder = notes_dir(cfg)
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "notes"
    stem = f"{datetime.now():%Y-%m-%d}-{slug}"
    path = folder / f"{stem}.md"
    n = 2
    while path.exists():                      # never overwrite an older note
        path = folder / f"{stem}-{n}.md"
        n += 1
    if path.resolve().parent != folder:       # belt and braces
        raise ValueError("refusing to write outside the notes folder")
    path.write_text(markdown.rstrip() + "\n", encoding="utf-8")
    return path


def show_file(path: Path) -> None:
    """Open the notes on his screen: Notepad on Windows (reads .md fine)."""
    if os.name == "nt":
        subprocess.Popen(["notepad.exe", str(path)])
    else:
        webbrowser.open(path.as_uri())


# --- keys (helper 4) --------------------------------------------------------------

def foreground_app() -> str:
    """The program in front, e.g. 'chrome.exe' ('' when it cannot be read)."""
    if os.name != "nt":
        return ""
    import ctypes
    from ctypes import wintypes
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(user32.GetForegroundWindow(), ctypes.byref(pid))
    handle = kernel32.OpenProcess(0x1000, False, pid.value)   # QUERY_LIMITED_INFORMATION
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(1024)
        size = wintypes.DWORD(len(buf))
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return Path(buf.value).name.lower()
    finally:
        kernel32.CloseHandle(handle)


def press(action: str, cfg: dict) -> str:
    """Do one KEY_ACTIONS entry on this laptop. Returns '' or why not."""
    kind, keys, times = KEY_ACTIONS[action]
    if kind == "notes":
        folder = notes_dir(cfg)
        if os.name == "nt":
            os.startfile(str(folder))                 # noqa: S606 -- our own folder
        else:
            webbrowser.open(folder.as_uri())
        return ""
    if os.name != "nt":
        return "keys work on Windows only"
    import ctypes
    user32 = ctypes.windll.user32
    if kind == "lock":
        user32.LockWorkStation()
        return ""
    if kind == "browser":
        app = foreground_app()
        if app not in BROWSERS:
            return f"the browser is not the window in front ({app or 'unknown'})"
        codes = keys
        extended = 0
    else:
        codes = (keys,)
        extended = 0x0001                              # KEYEVENTF_EXTENDEDKEY
    for _ in range(times):
        for vk in codes:
            user32.keybd_event(vk, 0, extended, 0)
        for vk in reversed(codes):
            user32.keybd_event(vk, 0, extended | 0x0002, 0)   # KEYEVENTF_KEYUP
        time.sleep(0.02)
    return ""


# --- apps and idle (helper 5) ---------------------------------------------------

def start_apps() -> list[dict]:
    """[{"name", "id", "via"}] from the Start menu (Windows only), with APP_DENY
    left out. `via` is only a hint for the Pi (steam / epic / store / desktop)."""
    if os.name != "nt":
        return []
    got = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command",
         "[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
         "Get-StartApps | ConvertTo-Json -Compress"],
        capture_output=True, encoding="utf-8", errors="replace", timeout=40,
        creationflags=0x08000000)                   # CREATE_NO_WINDOW
    data = json.loads(got.stdout or "[]")
    if isinstance(data, dict):
        data = [data]
    out, seen = [], set()
    for row in data:
        name = str(row.get("Name") or "").strip()
        app_id = str(row.get("AppID") or "").strip()
        if not name or not app_id or len(name) > 80 or APP_DENY.search(name):
            continue
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        low = app_id.lower()
        via = ("steam" if low.startswith("steam:") else
               "epic" if "epicgames" in low else
               "store" if "!" in app_id and "\\" not in app_id else "desktop")
        out.append({"name": name, "id": app_id, "via": via})
    return sorted(out, key=lambda a: a["name"].lower())


def _bring_to_front(name: str, wait_s: float = 40.0) -> None:
    """Find the just-launched app's window by its title and put it in front.
    Windows refuses SetForegroundWindow to a background process, so it is
    preceded by one ALT tap (the usual way: the helper then owns the last
    input). Only a window whose title matches `name` is touched (2026-10-20:
    Solitaire opened BEHIND his terminal and could not be played)."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found: list[int] = []
    proc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def each(hwnd, _lparam):
        if user32.IsWindowVisible(hwnd):
            n = user32.GetWindowTextLengthW(hwnd)
            if n:
                buf = ctypes.create_unicode_buffer(n + 1)
                user32.GetWindowTextW(hwnd, buf, n + 1)
                if title_matches(buf.value, name):
                    found.append(hwnd)
                    return False
        return True

    end = time.time() + wait_s
    while time.time() < end and not found:
        user32.EnumWindows(proc(each), 0)
        if not found:
            time.sleep(0.5)
    if not found:
        log.info("launched %s, but found no window of it to bring forward", name)
        return
    hwnd = found[0]
    user32.keybd_event(0x12, 0, 0, 0)                          # ALT down
    user32.keybd_event(0x12, 0, 0x0002, 0)                     # ALT up
    user32.ShowWindow(hwnd, 9)                                 # SW_RESTORE
    user32.SetForegroundWindow(hwnd)


def open_app(app_id: str, name: str = "") -> None:
    """Start one Start-menu entry by its own id (from start_apps, never the Pi),
    then bring its window to the front. A Google Play Games game's entry is a
    googleplaygames:// link, not an app id: handed to shell:AppsFolder it
    opened File Explorer on Documents (helper 9, 2026-10-21: Hill Climb
    Racing, Asphalt 8) -- so a link is opened as a link."""
    log.info("launch %s (%s)", name, app_id[:120])
    if "://" in app_id and os.name == "nt":
        os.startfile(app_id)                          # noqa: S606 -- the laptop's own entry
    else:
        subprocess.Popen(["explorer.exe", "shell:AppsFolder\\" + app_id])
    if name and os.name == "nt":
        threading.Thread(target=_bring_to_front, args=(name,), daemon=True).start()


def game_words(name: str) -> set[str]:
    """The words of a game's name that a window title must show (helper 6)."""
    words = re.findall(r"[a-z0-9]+", name.lower())
    return {w for w in words if len(w) >= 3 and w not in _GENERIC_WORDS}


def title_matches(title: str, game: str) -> bool:
    words = game_words(game)
    have = set(re.findall(r"[a-z0-9]+", (title or "").lower()))
    return bool(words) and bool(words & have)


def _dpi_aware() -> None:
    """Real pixels everywhere (helper 7). With display scaling (125%/150%) a
    process that does not say this gets SCALED window sizes but REAL screen
    pixels: the 2026-10-20 Solitaire picture had a strip of desktop on the left
    and the right of the board cut off. Per-monitor aware, else system aware."""
    if os.name != "nt":
        return
    import ctypes
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)          # PER_MONITOR
    except Exception:                                           # noqa: BLE001
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:                                       # noqa: BLE001
            pass


def foreground_title() -> str:
    if os.name != "nt":
        return ""
    import ctypes
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value


def _client_rect() -> tuple[int, int, int, int] | None:
    """(left, top, width, height) of the foreground window's client area, in
    screen pixels."""
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    return (origin.x, origin.y, w, h) if w > 50 and h > 50 else None


def _png(width: int, height: int, bgra: bytes) -> bytes:
    """Raw top-down BGRA pixels -> a PNG (standard library only)."""
    import struct
    import zlib
    rgb = bytearray(width * height * 3)
    rgb[0::3] = bgra[2::4]
    rgb[1::3] = bgra[1::4]
    rgb[2::3] = bgra[0::4]
    stride = width * 3
    raw = b"".join(b"\x00" + bytes(rgb[y * stride:(y + 1) * stride]) for y in range(height))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def grab(box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0),
         max_w: int = SCREEN_MAX_W) -> tuple[int, int, bytes] | None:
    """(width, height, top-down BGRA bytes) of `box` (fractions of the window in
    front's client area), scaled down to max_w. Helper 10: a box, so the
    autopilot can read the game's minimap at full resolution, fast."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes
    rect = _client_rect()
    if rect is None:
        return None
    left, top, cw, ch = rect
    x0, y0, x1, y1 = box
    left, top = left + int(x0 * cw), top + int(y0 * ch)
    w, h = max(1, int((x1 - x0) * cw)), max(1, int((y1 - y0) * ch))
    scale = min(1.0, max_w / float(w))
    ow, oh = max(1, int(w * scale)), max(1, int(h * scale))
    user32, gdi32 = ctypes.windll.user32, ctypes.windll.gdi32
    screen_dc = user32.GetDC(0)
    mem_dc = gdi32.CreateCompatibleDC(screen_dc)
    bmp = gdi32.CreateCompatibleBitmap(screen_dc, ow, oh)
    old = gdi32.SelectObject(mem_dc, bmp)
    try:
        gdi32.SetStretchBltMode(mem_dc, 4)                     # HALFTONE
        gdi32.StretchBlt(mem_dc, 0, 0, ow, oh, screen_dc, left, top, w, h, 0x00CC0020)

        class BITMAPINFOHEADER(ctypes.Structure):
            _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                        ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                        ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                        ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                        ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                        ("biClrImportant", wintypes.DWORD)]

        info = BITMAPINFOHEADER()
        info.biSize = ctypes.sizeof(info)
        info.biWidth, info.biHeight = ow, -oh                  # top-down rows
        info.biPlanes, info.biBitCount, info.biCompression = 1, 32, 0
        buf = ctypes.create_string_buffer(ow * oh * 4)
        gdi32.GetDIBits(mem_dc, bmp, 0, oh, buf, ctypes.byref(info), 0)
        return ow, oh, buf.raw
    finally:
        gdi32.SelectObject(mem_dc, old)
        gdi32.DeleteObject(bmp)
        gdi32.DeleteDC(mem_dc)
        user32.ReleaseDC(0, screen_dc)


def screen_png(box: tuple[float, float, float, float] | None = None) -> bytes | None:
    """The foreground window's client area (or `box` of it) as a PNG, scaled
    to SCREEN_MAX_W."""
    got = grab(box or (0.0, 0.0, 1.0, 1.0))
    return _png(*got) if got else None


def click_at(fx: float, fy: float) -> str:
    """One left click at (fx, fy) of the foreground client area. '' or why not."""
    if os.name != "nt":
        return "clicks work on Windows only"
    import ctypes
    rect = _client_rect()
    if rect is None:
        return "no window to click in"
    left, top, w, h = rect
    fx, fy = min(max(float(fx), 0.02), 0.98), min(max(float(fy), 0.02), 0.98)
    x, y = int(left + fx * w), int(top + fy * h)
    user32 = ctypes.windll.user32
    user32.SetCursorPos(x, y)
    time.sleep(0.03)
    user32.mouse_event(0x0002, 0, 0, 0, 0)                     # LEFTDOWN
    time.sleep(0.04)
    user32.mouse_event(0x0004, 0, 0, 0, 0)                     # LEFTUP
    return ""


def _game_key(name: str, up: bool) -> None:
    import ctypes
    vk, scan, ext = GAME_KEYS[name]
    flags = (0x0001 if ext else 0) | (0x0002 if up else 0)  # EXTENDEDKEY, KEYUP
    ctypes.windll.user32.keybd_event(vk, scan, flags, 0)


def hold_keys(keys: list[str], ms: int, still_in_front) -> str:
    """Hold GAME_KEYS down together for `ms`; let go of all of them the moment
    the game stops being the window in front. Always released. '' or why it
    stopped early (helper 8: how a racing game is driven)."""
    if os.name != "nt":
        return "keys work on Windows only"
    down: list[str] = []
    why = ""
    try:
        for key in keys:
            _game_key(key, False)
            down.append(key)
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            if not still_in_front():
                why = "the game is no longer the window in front"
                break
            time.sleep(min(0.025, max(0.0, end - time.monotonic())))
    finally:
        for key in reversed(down):
            _game_key(key, True)
    return why


def hold_then_keep(keys: list[str], ms: int, keep_ms: int, still_in_front,
                   stop: threading.Event) -> tuple[str, threading.Thread | None]:
    """Hold `keys` for `ms` like hold_keys, then return with them STILL DOWN:
    a thread lets go after keep_ms, when `stop` is set, or the moment the game
    leaves the front. ('' or why, that thread)."""
    if os.name != "nt":
        return "keys work on Windows only", None
    down: list[str] = []

    def release() -> None:
        for key in reversed(down):
            _game_key(key, True)

    try:
        for key in keys:
            _game_key(key, False)
            down.append(key)
        end = time.monotonic() + ms / 1000.0
        while time.monotonic() < end:
            if not still_in_front():
                release()
                return "the game is no longer the window in front", None
            time.sleep(min(0.025, max(0.0, end - time.monotonic())))
    except BaseException:
        release()
        raise

    def keeper() -> None:
        until = time.monotonic() + keep_ms / 1000.0
        try:
            while time.monotonic() < until and not stop.is_set():
                if not still_in_front():
                    break
                stop.wait(0.05)
        finally:
            release()

    t = threading.Thread(target=keeper, name="keys-kept", daemon=True)
    t.start()
    return "", t


def idle_info() -> dict:
    """Seconds since the keyboard or mouse was last used, and the app in front."""
    if os.name != "nt":
        return {"idle_s": None, "app": "", "title": ""}
    import ctypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))
    idle_ms = (ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF
    return {"idle_s": round(idle_ms / 1000.0, 1), "app": foreground_app(),
            "title": foreground_title()[:120]}


# --- the autopilot (helper 10) ---------------------------------------------------
# Harsh, 2026-10-21, after GTA V: "do it for fast 3d games, it's slow while
# processing". Looking at a picture and deciding takes Friday ~3 s (Claude
# 10+ s) and a car does not wait. So the FAST half lives here, next to the
# game: ~10 times a second it reads the game's MINIMAP (GTA: light-grey road
# lines, the car's arrow always at the same spot), finds where the road goes
# ahead of the arrow, and steers there with short taps; it holds the gas and
# backs out when the picture stops changing (stuck on a wall). The SLOW half --
# the Pi -- only says go / bias left or right / slower / stop, or takes a
# manual step (which switches this off).
#
# Safety, same rules as /step: GAME_KEYS only, only while that game is the
# window in front (checked every tick), every key let go when it stops, and it
# stops by itself unless the Pi checks in every AUTO_LEASE_S -- so a Pi that
# dies or loses Wi-Fi never leaves a car driving.

AUTO_LEASE_S = 8.0
AUTO_MAX_S = 120.0
AUTO_TICK_S = 0.1
AUTO_MODES = ("road", "cruise")
AUTO_KEYS = {"gas": "w", "brake": "s", "left": "a", "right": "d"}      # GTA's
MAP_BOX = (0.0, 0.787, 0.162, 0.99)       # GTA V's minimap, fractions of the window
MAP_MAX_W = 320
ROAD_LO, ROAD_HI, ROAD_SAT = 92, 190, 28  # minimap road: light grey, not white
ARROW_WHITE = 165                         # both halves of the arrow; roads are below 160
ARROW_AT = (0.52, 0.70)                   # the player arrow in MAP_BOX
MAP_BAR_FROM = 0.86                       # health/armour bar below it
STEER_DEAD_DEG, STEER_FULL_DEG, BIAS_DEG = 6.0, 35.0, 45.0
LOOKAHEAD = 0.09                          # pure-pursuit distance, fraction of the box width
STUCK_AFTER_S, STUCK_DIFF, REVERSE_S = 2.0, 3.0, 1.2


# What the Pi may tune per run (POST /autopilot "tune"), with the range each is
# held to -- numbers only, so tuning never needs a new helper copied over.
TUNE_RANGES = {
    "road_lo": (40, 200), "road_hi": (80, 255), "road_sat": (5, 80), "arrow_white": (150, 255),
    "arrow_x": (0.2, 0.8), "arrow_y": (0.3, 0.9), "lookahead": (0.03, 0.3),
    "steer_dead": (0.0, 30.0), "steer_full": (10.0, 90.0), "straight_pref": (0.0, 1.0),
    "stuck_diff": (0.5, 20.0), "reverse_s": (0.3, 3.0),
    "follow_route": (0.0, 1.0), "route_b": (100, 255), "route_gap": (20, 200),
    "route_r": (0, 255),
}


def tune_defaults() -> dict:
    return {"road_lo": ROAD_LO, "road_hi": ROAD_HI, "road_sat": ROAD_SAT,
            "arrow_white": ARROW_WHITE, "arrow_x": ARROW_AT[0], "arrow_y": ARROW_AT[1],
            "lookahead": LOOKAHEAD, "steer_dead": STEER_DEAD_DEG,
            "steer_full": STEER_FULL_DEG, "straight_pref": 0.18,
            "stuck_diff": STUCK_DIFF, "reverse_s": REVERSE_S,
            # GTA's GPS route (a waypoint): purple (168, 84, 243) over the road
            "follow_route": 1.0, "route_b": 180, "route_gap": 100, "route_r": 100}


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    try:
        x0, y0, x1, y1 = (float(v) for v in str(text).split(","))
    except ValueError:
        return None
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return x0, y0, x1, y1


def _bgr(bgra: bytes, w: int, x: int, y: int) -> tuple[int, int, int]:
    i = (y * w + x) * 4
    return bgra[i + 2], bgra[i + 1], bgra[i]


def _is_road(r: int, g: int, b: int, t: dict | None = None) -> bool:
    mx, mn = max(r, g, b), min(r, g, b)
    if t is None:
        return ROAD_LO <= mx <= ROAD_HI and mx - mn <= ROAD_SAT
    return t["road_lo"] <= mx <= t["road_hi"] and mx - mn <= t["road_sat"]


def _is_route(r: int, g: int, b: int, t: dict) -> bool:
    return b >= t["route_b"] and b - g >= t["route_gap"] and r >= t["route_r"]


def find_arrow(w: int, h: int, bgra: bytes,
               t: dict | None = None) -> tuple[float, float, float] | None:
    """(x, y, heading) of the player arrow near ARROW_AT; heading in degrees,
    0 = up the map, positive = clockwise.

    GTA's arrow is HALF white and half light grey (lit and shaded sides).
    Helper 10-12 took only the white half and called the pixel farthest from
    its centre the tip -- on half an arrow that is often a back corner, and on
    2026-10-21 the car was steered back the way it came. Now: both halves
    (brighter than any road), the arrow's mirror axis (the angle where the
    shape best matches its own reflection), and the tip is the end of that
    axis that reaches farthest from the centre."""
    t = t or tune_defaults()
    ex, ey, rad = t["arrow_x"] * w, t["arrow_y"] * h, 0.12 * w
    pts = []
    for y in range(max(0, int(ey - rad)), min(int(h * MAP_BAR_FROM), int(ey + rad))):
        for x in range(max(0, int(ex - rad)), min(w, int(ex + rad))):
            if min(_bgr(bgra, w, x, y)) >= t["arrow_white"]:
                pts.append((x, y))
    if len(pts) < 4:
        return None
    # the arrow is the biggest bright blob nearest the expected spot
    blob = _blob_near(pts, ex, ey)
    if len(blob) < 4:
        return None
    cx = sum(p[0] for p in blob) / len(blob)
    cy = sum(p[1] for p in blob) / len(blob)
    # The two halves meet along the arrow's axis, so the axis is square to
    # the line from the white half's centre to the grey half's (helper 14:
    # the minimap is drawn in perspective, and a mirror test alone read an
    # arrow pointing straight up as 120 deg on the live drive).
    lit = [(x, y) for x, y in blob if min(_bgr(bgra, w, x, y)) >= 215]
    shade = [(x, y) for x, y in blob if min(_bgr(bgra, w, x, y)) < 215]
    if len(lit) >= 4 and len(shade) >= 4:
        lx, ly = sum(p[0] for p in lit) / len(lit), sum(p[1] for p in lit) / len(lit)
        sx, sy = sum(p[0] for p in shade) / len(shade), sum(p[1] for p in shade) / len(shade)
        if math.hypot(sx - lx, sy - ly) >= 1.5:
            across = math.degrees(math.atan2(sx - lx, -(sy - ly)))
            axis = across - 90.0
            a = math.radians(axis)
            ux, uy = math.sin(a), -math.cos(a)
            alongs = [(x - cx) * ux + (y - cy) * uy for x, y in blob]
            heading = axis if max(alongs) >= -min(alongs) else axis + 180.0
            return cx, cy, (heading + 180.0) % 360.0 - 180.0
    cells = {(x, y) for x, y in blob}
    best_axis, best_score = 0.0, -1.0
    for deg in range(0, 180, 5):
        a = math.radians(deg)
        ux, uy = math.sin(a), -math.cos(a)
        score = 0
        for x, y in blob:
            dx, dy = x - cx, y - cy
            along = dx * ux + dy * uy
            # reflect across the axis: keep `along`, flip the sideways part
            rx, ry = cx + 2 * along * ux - dx, cy + 2 * along * uy - dy
            if (round(rx), round(ry)) in cells:
                score += 1
        if score > best_score:
            best_axis, best_score = float(deg), float(score)
    a = math.radians(best_axis)
    ux, uy = math.sin(a), -math.cos(a)
    alongs = [(x - cx) * ux + (y - cy) * uy for x, y in blob]
    heading = best_axis if max(alongs) >= -min(alongs) else best_axis + 180.0
    return cx, cy, (heading + 180.0) % 360.0 - 180.0


def _blob_near(pts: list[tuple[int, int]], ex: float, ey: float) -> list[tuple[int, int]]:
    """The 8-connected group of `pts` with the most pixels, ties to the
    nearest the expected spot (a white icon nearby is not the arrow)."""
    left = set(pts)
    groups = []
    while left:
        seed = left.pop()
        group, todo = [seed], [seed]
        while todo:
            x, y = todo.pop()
            for nx in (x - 1, x, x + 1):
                for ny in (y - 1, y, y + 1):
                    if (nx, ny) in left:
                        left.discard((nx, ny))
                        group.append((nx, ny))
                        todo.append((nx, ny))
        groups.append(group)

    def key(g):
        gx = sum(p[0] for p in g) / len(g)
        gy = sum(p[1] for p in g) / len(g)
        return (len(g) >= 8, -math.hypot(gx - ex, gy - ey))
    return max(groups, key=key)


def read_map(w: int, h: int, bgra: bytes, span: int = 90, step: int = 6,
             t: dict | None = None) -> dict | None:
    """Where the road goes ahead of the car, from the minimap: a ray every
    `step` degrees within +-span of the arrow's heading, each as long as it
    stays on road. target = the best ray, straight ahead preferred."""
    t = t or tune_defaults()
    arrow = find_arrow(w, h, bgra, t)
    if arrow is None:
        return None
    cx, cy, heading = arrow
    rmax, limit_y = int(0.45 * w), h * MAP_BAR_FROM
    follow = t["follow_route"] >= 0.5
    rays: dict[int, int] = {}
    route: dict[int, int] = {}                  # how far each ray stays on the GPS route
    for rel in range(-span, span + 1, step):
        a = math.radians(heading + rel)
        dx, dy = math.sin(a), -math.cos(a)
        length, on_route, misses, started = 0, 0, 0, False
        for r in range(3, rmax):
            x, y = int(cx + dx * r), int(cy + dy * r)
            if not (0 <= x < w and 0 <= y < limit_y):
                break
            pix = _bgr(bgra, w, x, y)
            purple = follow and _is_route(*pix, t)
            if purple or _is_road(*pix, t):
                started, misses, length = True, 0, r
                if purple:
                    on_route = r
            else:
                misses += 1
                if (started and misses > 2) or (not started and r > 0.05 * w):
                    break
        rays[rel], route[rel] = length, on_route
    pref = t["straight_pref"] * (rmax / 60.0)
    # A waypoint's purple route near the car wins over plain road: that is
    # how it takes the right turn at a junction instead of the straightest.
    use_route = follow and max(route.values()) >= 0.04 * w
    pick = route if use_route else rays
    best = max(pick, key=lambda k: pick[k] - pref * abs(k))
    if use_route:
        ring = _route_ring(w, h, bgra, cx, cy, heading, t["lookahead"] * w, limit_y, t)
        if ring is not None:
            return {"heading": round(heading), "target": round(ring), "ray": best,
                    "ahead": rays.get(0, 0), "rmax": rmax, "best_len": pick[best],
                    "route": True}
    # Aim at the MIDDLE of the road a short way along that ray (pure pursuit):
    # following the ray alone kept the car on the road but scraping its edge
    # (live 2026-10-21: a hedge on the left, then a wall on the right).
    aim = best
    look = min(t["lookahead"] * w, max(pick[best] * 0.8, 0.0))
    if look >= 0.03 * w:
        a = math.radians(heading + best)
        px, py = cx + math.sin(a) * look, cy - math.cos(a) * look
        nx, ny = math.cos(a), math.sin(a)               # to the right of the ray
        edges = []
        for side in (-1, 1):
            far = 0
            for d in range(1, int(0.12 * w)):
                x, y = int(px + side * nx * d), int(py + side * ny * d)
                if not (0 <= x < w and 0 <= y < limit_y):
                    break
                pix = _bgr(bgra, w, x, y)
                if not (_is_route(*pix, t) if use_route else _is_road(*pix, t)):
                    break
                far = d
            edges.append(far)
        shift = (edges[1] - edges[0]) / 2.0             # + = the middle is to the right
        mx, my = px + nx * shift, py + ny * shift
        aim = math.degrees(math.atan2(mx - cx, -(my - cy))) - heading
        aim = (aim + 180.0) % 360.0 - 180.0
    return {"heading": round(heading), "target": round(aim), "ray": best,
            "ahead": rays.get(0, 0), "rmax": rmax, "best_len": pick[best],
            "route": use_route}


def _route_ring(w: int, h: int, bgra: bytes, cx: float, cy: float, heading: float,
                look: float, limit_y: float, t: dict) -> float | None:
    """Where the purple route crosses a circle of radius `look` round the car:
    the angle (to the heading) of the crossing nearest straight ahead. A ray
    only sees straight lines; the circle follows a bend or a turn coming up
    (the junction test: road straight on, route turning left)."""
    look = max(4.0, look)
    angles = []
    for y in range(max(0, int(cy - look - 2)), min(int(limit_y), int(cy + look + 2))):
        for x in range(max(0, int(cx - look - 2)), min(w, int(cx + look + 2))):
            d = math.hypot(x - cx, y - cy)
            if abs(d - look) > 1.5 or not _is_route(*_bgr(bgra, w, x, y), t):
                continue
            rel = math.degrees(math.atan2(x - cx, -(y - cy))) - heading
            rel = (rel + 180.0) % 360.0 - 180.0
            if abs(rel) <= 120.0:
                angles.append(rel)
    if not angles:
        return None
    angles.sort()
    groups, cur = [], [angles[0]]
    for a in angles[1:]:
        if a - cur[-1] > 15.0:
            groups.append(cur)
            cur = [a]
        else:
            cur.append(a)
    groups.append(cur)
    best = min(groups, key=lambda g: abs(sum(g) / len(g)))
    return sum(best) / len(best)


def _thumb(got: tuple[int, int, bytes] | None) -> list[int] | None:
    """Grey levels of a small whole-window picture (stuck test)."""
    if not got:
        return None
    w, h, bgra = got
    return [sum(_bgr(bgra, w, x, y)) // 3 for y in range(0, h, 2) for x in range(0, w, 2)]


class Autopilot:
    """The fast half of playing a fast game. One at a time; see the block above."""

    def __init__(self, titler, grabber=grab, presser=None) -> None:
        self.titler, self.grabber = titler, grabber
        self.presser = presser or _game_key
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._down: set[str] = set()
        self.cfg: dict = {}
        self.lease_until = self.until = 0.0
        self.state: dict = {"running": False, "why": "never started"}

    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def set(self, cfg: dict) -> dict:
        """Start, or change and renew, the autopilot. cfg is validated by the caller."""
        now = time.monotonic()
        with self._lock:
            self.cfg.update(cfg)
            self.lease_until = now + AUTO_LEASE_S
            self.until = now + min(float(cfg.get("for_s", AUTO_MAX_S)), AUTO_MAX_S)
            start = not self.running()
            if start:
                self._stop.clear()
                self.state = {"running": True, "mode": self.cfg["mode"], "ticks": 0,
                              "stucks": 0, "why": ""}
                self._thread = threading.Thread(target=self._run, name="autopilot",
                                                daemon=True)
        if start:
            self._thread.start()
        return self.report()

    def renew(self) -> None:
        with self._lock:
            self.lease_until = time.monotonic() + AUTO_LEASE_S

    def stop(self, why: str = "stopped") -> None:
        if self.running():
            self.state["why"] = why
            self._stop.set()
            self._thread.join(timeout=1.5)
        self._release_all()

    def report(self) -> dict:
        out = {k: v for k, v in self.state.items()}
        out["running"] = self.running()
        return out

    def _keys(self, want: set[str]) -> None:
        for key in sorted(self._down - want):
            self.presser(key, True)
            self._down.discard(key)
        for key in sorted(want - self._down):
            self.presser(key, False)
            self._down.add(key)

    def _release_all(self) -> None:
        for key in sorted(self._down):
            try:
                self.presser(key, True)
            except Exception:                           # noqa: BLE001
                pass
        self._down.clear()

    def _wait(self, seconds: float) -> bool:
        """Sleep; True if told to stop."""
        return self._stop.wait(max(0.0, seconds))

    def _run(self) -> None:
        why = "stopped"
        thumb_prev, still_since, gas_since, next_thumb = None, None, None, 0.0
        last_target = 0
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                cfg = dict(self.cfg)
                if now > self.lease_until:
                    why = "the Pi stopped checking in"
                    break
                if now > self.until:
                    why = "time is up"
                    break
                if not title_matches(self.titler(), cfg["game"]):
                    why = "the game is no longer the window in front"
                    break
                self.state["ticks"] = self.state.get("ticks", 0) + 1
                keys, speed = cfg["keys"], float(cfg.get("speed", 1.0))
                if cfg["mode"] == "cruise":
                    self._keys(set(cfg.get("hold") or []))
                    if self._wait(AUTO_TICK_S):
                        break
                    continue
                t = cfg.get("tune") or tune_defaults()
                got = self.grabber(tuple(cfg.get("box") or MAP_BOX), MAP_MAX_W)
                nav = read_map(*got, t=t) if got else None
                if nav is None:
                    self._keys(set())
                    self.state["last"] = {"map": "no arrow (on foot? map hidden?)"}
                    if self._wait(AUTO_TICK_S * 3):
                        break
                    continue
                error = max(-90.0, min(90.0, nav["target"] + float(cfg.get("bias", 0.0)) * BIAS_DEG))
                last_target = error
                gas = speed > 0 and not (nav["best_len"] == 0 and abs(error) > 60)
                duty = (0.0 if abs(error) < t["steer_dead"]
                        else min(1.0, abs(error) / t["steer_full"]))
                steer = keys["left"] if error < 0 else keys["right"]
                self.state["last"] = {"target": nav["target"], "ray": nav.get("ray"),
                                      "route": nav.get("route"),
                                      "error": round(error),
                                      "ahead": nav["ahead"], "heading": nav["heading"],
                                      "steer": steer if duty else "", "duty": round(duty, 2),
                                      "gas": gas}
                # stuck: the gas has been on and the whole picture stopped changing
                gas_since = (gas_since or now) if gas else None
                if now >= next_thumb:
                    next_thumb = now + 0.5
                    thumb = _thumb(self.grabber((0.0, 0.0, 1.0, 1.0), 48))
                    if thumb and thumb_prev and len(thumb) == len(thumb_prev):
                        diff = sum(abs(a - b) for a, b in zip(thumb, thumb_prev)) / len(thumb)
                        self.state["last"]["change"] = round(diff, 1)
                        still_since = (still_since or now) if diff < t["stuck_diff"] else None
                    thumb_prev = thumb
                if (gas_since and still_since and now - gas_since > STUCK_AFTER_S
                        and now - still_since > 1.0):
                    self.state["stucks"] = self.state.get("stucks", 0) + 1
                    back_steer = keys["right"] if last_target < 0 else keys["left"]
                    self._keys({keys["brake"], back_steer})
                    if self._wait(t["reverse_s"]):
                        break
                    self._keys(set())
                    gas_since = still_since = None
                    continue
                # one tick: steer for duty*tick, gas for speed*tick
                tick = AUTO_TICK_S
                first = {k for k, on in ((keys["gas"], gas and speed > 0), (steer, duty > 0)) if on}
                self._keys(first)
                t_steer, t_gas = duty * tick, (tick if speed >= 0.95 else speed * tick)
                cut = min(t for t in (t_steer, t_gas, tick) if t > 0) if first else tick
                if self._wait(cut):
                    break
                rest = set(first)
                if t_steer <= cut and duty < 1.0:
                    rest.discard(steer)
                if t_gas <= cut and speed < 0.95:
                    rest.discard(keys["gas"])
                self._keys(rest)
                if self._wait(tick - cut):
                    break
        except Exception as exc:                        # noqa: BLE001
            why = f"error: {type(exc).__name__}: {str(exc)[:80]}"
            log.warning("autopilot %s", why)
        finally:
            self._release_all()
            if not self.state.get("why"):
                self.state["why"] = why
            self.state["running"] = False
            log.info("autopilot ended: %s", self.state["why"])


# --- the listener ---------------------------------------------------------------

class Agent:
    def __init__(self, cfg: dict, opener=webbrowser.open, shower=show_file,
                 fetch=_fetch, hello: Hello | None = None, presser=press,
                 lister=start_apps, launcher=open_app, idler=idle_info,
                 titler=foreground_title, shooter=screen_png, clicker=click_at,
                 holder=hold_keys, keeper=hold_then_keep, grabber=grab,
                 autokey=None) -> None:
        self.holder, self.keeper = holder, keeper
        self.autopilot = Autopilot(titler, grabber=grabber, presser=autokey)
        self._step_lock = threading.Lock()
        self._kept: tuple[threading.Event, threading.Thread | None] | None = None
        self.cfg = cfg
        self.opener = opener
        self.shower = shower
        self.fetch = fetch
        self.hello = hello
        self.presser = presser
        self.lister, self.launcher, self.idler = lister, launcher, idler
        self.titler, self.shooter, self.clicker = titler, shooter, clicker
        self._apps: list[dict] = []
        self._apps_at = 0.0
        self._apps_lock = threading.Lock()

    def stop_keeping(self) -> None:
        """Let go of keys a previous /step kept down -- and WAIT until they are
        up, or the old thread would release keys this step just pressed."""
        kept, self._kept = self._kept, None
        if kept is not None:
            kept[0].set()
            if kept[1] is not None:
                kept[1].join(timeout=1.0)

    def apps(self, fresh: bool = False) -> list[dict]:
        """The Start-menu list, cached APPS_REFRESH_S (PowerShell takes a second)."""
        with self._apps_lock:
            if fresh or not self._apps or time.time() - self._apps_at > APPS_REFRESH_S:
                self._apps = self.lister()
                self._apps_at = time.time()
            return list(self._apps)

    def handler(self):
        agent = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "FridayLaptopHelper/14"

            def log_message(self, fmt, *args):         # no tokens in logs
                log.info("%s %s", self.address_string(), fmt % args)

            def _send(self, code: int, body: dict) -> None:
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _authorised(self) -> bool:
                given = self.headers.get("X-Friday-Token", "")
                ok = hmac.compare_digest(given.encode(),
                                         agent.cfg["token"].encode())
                if ok and agent.hello is not None:
                    agent.hello.seen(self.client_address[0])
                if ok:
                    agent.autopilot.renew()             # the Pi is still there
                return ok

            def do_GET(self):                           # noqa: N802
                if not self._authorised():
                    return self._send(401, {"error": "unauthorised"})
                if self.path == "/apps":
                    try:
                        apps = agent.apps()
                    except Exception as exc:            # noqa: BLE001
                        return self._send(500, {"error": f"{type(exc).__name__}"})
                    return self._send(200, {"ok": True, "apps": [
                        {"name": a["name"], "via": a.get("via", "")} for a in apps]})
                if self.path == "/idle":
                    return self._send(200, {"ok": True, **agent.idler()})
                if self.path.startswith("/screen?"):
                    return self._screen()
                if self.path == "/autopilot":
                    return self._send(200, {"ok": True, **agent.autopilot.report()})
                if self.path != "/health":
                    return self._send(404, {"error": "not found"})
                return self._send(200, {"ok": True, "version": 14})

            def do_POST(self):                          # noqa: N802
                if not self._authorised():
                    return self._send(401, {"error": "unauthorised"})
                if self.path not in ("/open", "/notes", "/key", "/launch", "/click", "/step",
                                     "/autopilot"):
                    return self._send(404, {"error": "not found"})
                size = int(self.headers.get("Content-Length") or 0)
                if size <= 0 or size > MAX_BODY:
                    return self._send(413, {"error": "body too large or empty"})
                try:
                    body = json.loads(self.rfile.read(size))
                    if not isinstance(body, dict):
                        raise TypeError
                except (ValueError, TypeError):
                    return self._send(400, {"error": "send a JSON object"})
                try:
                    if self.path == "/open":
                        return self._open(body)
                    if self.path == "/key":
                        return self._key(body)
                    if self.path == "/launch":
                        return self._launch(body)
                    if self.path == "/click":
                        agent.autopilot.stop("a manual move")
                        return self._click(body)
                    if self.path == "/step":
                        agent.autopilot.stop("a manual move")
                        return self._step(body)
                    if self.path == "/autopilot":
                        return self._autopilot(body)
                    return self._notes(body)
                except Exception as exc:                # noqa: BLE001
                    log.warning("%s failed: %s", self.path, exc)
                    return self._send(500, {"error": f"{type(exc).__name__}: "
                                                     f"{str(exc)[:200]}"})

            def _open(self, body: dict):
                site = str(body.get("site") or "web")
                query = str(body.get("query") or "").strip()
                if site not in ("web", "youtube", "store"):
                    return self._send(400, {"error": "site is web, youtube or store"})
                if not query or len(query) > MAX_QUERY:
                    return self._send(400, {"error": "query is empty or too long"})
                url = build_url(site, query, bool(body.get("play")),
                                fetch=agent.fetch)
                agent.opener(url)
                return self._send(200, {"ok": True, "url": url})

            def _game_in_front(self, game: str):
                """None if `game` is the window in front, else (code, why)."""
                if not game or len(game) > 80 or not game_words(game):
                    return 400, "send the game's name"
                title = agent.titler()
                if not title_matches(title, game):
                    return 409, "that game is not the window in front"
                return None

            def _screen(self):
                query = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(query)
                game = params.get("game", [""])[0].strip()
                refused = self._game_in_front(game)
                if refused:
                    return self._send(refused[0], {"ok": False, "error": refused[1]})
                box = None
                if params.get("box"):
                    box = parse_box(params["box"][0])
                    if box is None:
                        return self._send(400, {"error": "box is x0,y0,x1,y1 in 0..1"})
                png = agent.shooter(box) if box else agent.shooter()
                if not png:
                    return self._send(409, {"ok": False, "error": "no picture"})
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(png)))
                self.end_headers()
                self.wfile.write(png)
                return None

            def _click(self, body: dict):
                game = str(body.get("game") or "").strip()
                refused = self._game_in_front(game)
                if refused:
                    return self._send(refused[0], {"ok": False, "error": refused[1]})
                try:
                    fx, fy = float(body.get("x")), float(body.get("y"))
                except (TypeError, ValueError):
                    return self._send(400, {"error": "send x and y between 0 and 1"})
                if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
                    return self._send(400, {"error": "send x and y between 0 and 1"})
                why = agent.clicker(fx, fy)
                if why:
                    return self._send(409, {"ok": False, "error": why})
                return self._send(200, {"ok": True})

            def _step(self, body: dict):
                """{game, keys[], ms, before?, after?, shot?}: tap `before`
                (e.g. escape to un-pause), hold `keys` for `ms`, take the
                picture, tap `after` (pause again) -- so a fast game is played
                one short step at a time. Game keys only, game in front only."""
                game = str(body.get("game") or "").strip()
                refused = self._game_in_front(game)
                if refused:
                    return self._send(refused[0], {"ok": False, "error": refused[1]})
                keys = body.get("keys") or []
                taps = [body.get("before"), body.get("after")]
                try:
                    ms = int(body.get("ms") or 0)
                    keep = int(body.get("keep") or 0)
                except (TypeError, ValueError):
                    ms = keep = -1
                if (not isinstance(keys, list) or len(keys) > STEP_MAX_KEYS
                        or any(k not in GAME_KEYS for k in keys)
                        or any(t is not None and t not in GAME_KEYS for t in taps)
                        or not 0 <= ms <= STEP_MAX_MS or not 0 <= keep <= KEEP_MAX_MS):
                    return self._send(400, {"error": "keys from GAME_KEYS, ms 0..%d, "
                                                     "keep 0..%d" % (STEP_MAX_MS, KEEP_MAX_MS)})
                if not agent._step_lock.acquire(blocking=False):
                    return self._send(409, {"ok": False, "error": "busy"})
                try:
                    agent.stop_keeping()
                    def front() -> bool:
                        return title_matches(agent.titler(), game)
                    why = ""
                    if taps[0]:
                        why = agent.holder([taps[0]], 60, front)
                        time.sleep(0.15)
                    if not why and keys and ms and keep:
                        stop = threading.Event()
                        why, kept = agent.keeper(keys, ms, keep, front, stop)
                        if not why:
                            agent._kept = (stop, kept)
                    elif not why and keys and ms:
                        why = agent.holder(keys, ms, front)
                    png = agent.shooter() if not why and body.get("shot") else None
                    if not why and taps[1]:
                        why = agent.holder([taps[1]], 60, front)
                finally:
                    agent._step_lock.release()
                if why:
                    return self._send(409, {"ok": False, "error": why})
                if png:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/png")
                    self.send_header("Content-Length", str(len(png)))
                    self.end_headers()
                    self.wfile.write(png)
                    return None
                return self._send(200, {"ok": True})

            def _autopilot(self, body: dict):
                """{game, mode: road|cruise|off, box?, keys?, hold?, bias?, speed?, for_s?}"""
                mode = str(body.get("mode") or "")
                if mode == "off":
                    agent.autopilot.stop("turned off")
                    return self._send(200, {"ok": True, **agent.autopilot.report()})
                game = str(body.get("game") or "").strip()
                refused = self._game_in_front(game)
                if refused:
                    return self._send(refused[0], {"ok": False, "error": refused[1]})
                keys = dict(AUTO_KEYS)
                given = body.get("keys") or {}
                hold = body.get("hold") or []
                box = body.get("box")
                if box is not None:
                    box = parse_box(",".join(str(v) for v in box) if isinstance(box, list)
                                    else str(box))
                bad = (mode not in AUTO_MODES or not isinstance(given, dict)
                       or set(given) - set(AUTO_KEYS)
                       or any(v not in GAME_KEYS for v in given.values())
                       or not isinstance(hold, list) or len(hold) > STEP_MAX_KEYS
                       or any(k not in GAME_KEYS for k in hold)
                       or (mode == "cruise" and not hold)
                       or ("box" in body and box is None))
                if bad:
                    return self._send(400, {"error": "mode road|cruise|off; keys gas/brake/"
                                                     "left/right and hold from GAME_KEYS; "
                                                     "box x0,y0,x1,y1"})
                keys.update(given)

                def num(name: str, lo: float, hi: float, default: float) -> float:
                    try:
                        return min(hi, max(lo, float(body.get(name, default))))
                    except (TypeError, ValueError):
                        return default
                tune = tune_defaults()
                given_tune = body.get("tune") or {}
                if not isinstance(given_tune, dict) or set(given_tune) - set(TUNE_RANGES):
                    return self._send(400, {"error": "tune keys: " + ", ".join(sorted(TUNE_RANGES))})
                for name, value in given_tune.items():
                    lo, hi = TUNE_RANGES[name]
                    try:
                        tune[name] = min(hi, max(lo, float(value)))
                    except (TypeError, ValueError):
                        return self._send(400, {"error": f"tune {name} must be a number"})
                cfg = {"game": game, "mode": mode, "keys": keys, "hold": hold, "tune": tune,
                       "box": box or MAP_BOX, "bias": num("bias", -1.0, 1.0, 0.0),
                       "speed": num("speed", 0.0, 1.0, 1.0),
                       "for_s": num("for_s", 1.0, AUTO_MAX_S, 30.0)}
                agent.stop_keeping()                    # kept /step keys go up first
                return self._send(200, {"ok": True, **agent.autopilot.set(cfg)})

            def _launch(self, body: dict):
                wanted = str(body.get("name") or "").strip().lower()
                if not wanted or len(wanted) > 80:
                    return self._send(400, {"error": "send {name}"})
                apps = agent.apps()
                hit = next((a for a in apps if a["name"].lower() == wanted), None)
                if hit is None:
                    apps = agent.apps(fresh=True)        # installed a minute ago?
                    hit = next((a for a in apps if a["name"].lower() == wanted), None)
                if hit is None:
                    return self._send(404, {"ok": False, "error": "no such app"})
                try:
                    agent.launcher(hit["id"], hit["name"])
                except TypeError:                       # a launcher without names (tests)
                    agent.launcher(hit["id"])
                return self._send(200, {"ok": True, "name": hit["name"]})

            def _key(self, body: dict):
                action = str(body.get("action") or "")
                if action not in KEY_ACTIONS:
                    return self._send(400, {"error": "unknown action"})
                why = agent.presser(action, agent.cfg)
                if why:
                    return self._send(409, {"ok": False, "error": why})
                return self._send(200, {"ok": True, "action": action})

            def _notes(self, body: dict):
                title = str(body.get("title") or "").strip()
                markdown = str(body.get("markdown") or "")
                if not title or len(title) > MAX_TITLE or not markdown.strip():
                    return self._send(400, {"error": "send {title, markdown}"})
                path = write_notes(title, markdown, agent.cfg)
                try:
                    agent.shower(path)
                except Exception as exc:                # noqa: BLE001
                    log.info("saved but not opened: %s", exc)
                return self._send(200, {"ok": True, "path": str(path)})

        return Handler


# --- Windows: start at every login ------------------------------------------------

def _startup_file() -> Path:
    return (Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" /
            "Start Menu" / "Programs" / "Startup" / STARTUP_NAME)


def _pythonw() -> str:
    exe = Path(sys.executable)
    windowless = exe.with_name("pythonw.exe")
    return str(windowless if windowless.exists() else exe)


def _store_python() -> bool:
    """Microsoft Store Python: NEW files it creates under AppData go to a
    private copy (the package's LocalCache), so a Startup entry it wrote
    would never run. Command Prompt has to write that one file instead."""
    return "WindowsApps" in sys.executable or "WindowsApps" in sys.prefix


def install() -> int:
    if os.name != "nt":
        raise SystemExit("--install is for the Windows laptop.")
    script = Path(__file__).resolve()
    if _store_python():
        print("This is the Microsoft Store Python, which cannot write to the "
              "Startup folder itself.\nTo start the helper at every login, "
              "paste this ONE line into Command Prompt (not PowerShell):\n\n"
              f'echo @start "" pythonw "{script}" > '
              f'"%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs\\Startup\\'
              f'{STARTUP_NAME}"\n')
    else:
        target = _startup_file()
        target.write_text(f'@start "" "{_pythonw()}" "{script}"\r\n')
        print(f"Starts at every login now ({target}).")
    subprocess.Popen([_pythonw(), str(Path(__file__).resolve())],
                     creationflags=getattr(subprocess, "DETACHED_PROCESS", 0),
                     close_fds=True)
    print("Started. If Windows asks about the network: tick Private AND "
          "Public, then Allow.")
    return 0


def uninstall() -> int:
    if os.name != "nt":
        raise SystemExit("--uninstall is for the Windows laptop.")
    _startup_file().unlink(missing_ok=True)
    print("Removed from Startup. (A copy already running stops at logout, or "
          "end pythonw.exe in Task Manager.)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "friday_agent.json"))
    ap.add_argument("--install", action="store_true",
                    help="start at every Windows login, and start now")
    ap.add_argument("--uninstall", action="store_true",
                    help="stop starting at login")
    args = ap.parse_args()
    if args.install:
        load_config(Path(args.config).expanduser())     # fail early, not at login
        return install()
    if args.uninstall:
        return uninstall()

    handler = logging.handlers.RotatingFileHandler(
        HERE / "friday_laptop_helper.log", maxBytes=500_000, backupCount=1,
        encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)

    cfg = load_config(Path(args.config).expanduser())
    notes_dir(cfg)
    _dpi_aware()
    hello = Hello(cfg, int(cfg["port"]), HERE / PI_FILE)
    server = ThreadingHTTPServer((cfg["bind"], int(cfg["port"])),
                                 Agent(cfg, hello=hello).handler())
    log.info("listening on %s:%s, notes -> %s", cfg["bind"], cfg["port"],
             notes_dir(cfg))
    threading.Thread(target=hello.run, name="hello", daemon=True).start()
    if sys.stdout is not None:
        print(f"Friday's laptop helper: listening on {cfg['bind']}:"
              f"{cfg['port']}, notes -> {notes_dir(cfg)}. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
