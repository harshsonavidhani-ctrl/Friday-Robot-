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
  POST /open   {"site": "web" | "youtube", "query": "...", "play": false}
        Opens a Google or YouTube search for those words in the default
        browser; "play" on YouTube opens the top video instead. Only those
        two sites, and only a URL built HERE from the words -- never a URL, a
        file name or a command sent by the Pi.
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
        the program in front}. Friday plays on the laptop on her own only
        when nobody has touched it for a while (she never interrupts you).

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
         "Get-StartApps | ConvertTo-Json -Compress"],
        capture_output=True, text=True, timeout=40,
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


def open_app(app_id: str) -> None:
    """Start one Start-menu entry by its own id (from start_apps, never the Pi)."""
    subprocess.Popen(["explorer.exe", "shell:AppsFolder\\" + app_id])


def idle_info() -> dict:
    """Seconds since the keyboard or mouse was last used, and the app in front."""
    if os.name != "nt":
        return {"idle_s": None, "app": ""}
    import ctypes

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(info)
    ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info))
    idle_ms = (ctypes.windll.kernel32.GetTickCount() - info.dwTime) & 0xFFFFFFFF
    return {"idle_s": round(idle_ms / 1000.0, 1), "app": foreground_app()}


# --- the listener ---------------------------------------------------------------

class Agent:
    def __init__(self, cfg: dict, opener=webbrowser.open, shower=show_file,
                 fetch=_fetch, hello: Hello | None = None, presser=press,
                 lister=start_apps, launcher=open_app, idler=idle_info) -> None:
        self.cfg = cfg
        self.opener = opener
        self.shower = shower
        self.fetch = fetch
        self.hello = hello
        self.presser = presser
        self.lister, self.launcher, self.idler = lister, launcher, idler
        self._apps: list[dict] = []
        self._apps_at = 0.0
        self._apps_lock = threading.Lock()

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
            server_version = "FridayLaptopHelper/5"

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
                if self.path != "/health":
                    return self._send(404, {"error": "not found"})
                return self._send(200, {"ok": True, "version": 5})

            def do_POST(self):                          # noqa: N802
                if not self._authorised():
                    return self._send(401, {"error": "unauthorised"})
                if self.path not in ("/open", "/notes", "/key", "/launch"):
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
                    return self._notes(body)
                except Exception as exc:                # noqa: BLE001
                    log.warning("%s failed: %s", self.path, exc)
                    return self._send(500, {"error": f"{type(exc).__name__}: "
                                                     f"{str(exc)[:200]}"})

            def _open(self, body: dict):
                site = str(body.get("site") or "web")
                query = str(body.get("query") or "").strip()
                if site not in ("web", "youtube"):
                    return self._send(400, {"error": "site is web or youtube"})
                if not query or len(query) > MAX_QUERY:
                    return self._send(400, {"error": "query is empty or too long"})
                url = build_url(site, query, bool(body.get("play")),
                                fetch=agent.fetch)
                agent.opener(url)
                return self._send(200, {"ok": True, "url": url})

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
