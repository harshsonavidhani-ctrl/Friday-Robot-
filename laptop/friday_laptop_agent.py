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

  Every request must carry the shared token in the X-Friday-Token header;
  anything without it gets 401 and nothing else. No other endpoints.

WHY A PROGRAM IN YOUR LOGIN AND NOT SSH: a program started through Windows'
SSH server runs in a session with no desktop, so a browser it opens never
appears on your screen.

CONFIG: friday_agent.json next to this script (copied from the Pi's
secrets/friday_agent.json): {"token": "...", "port": 8765, "bind": "0.0.0.0",
"notes_dir": "~/Friday_Notes"}. Keep it private: it holds the token.
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
import subprocess
import sys
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
STARTUP_NAME = "friday_laptop_helper.cmd"

log = logging.getLogger("friday_laptop")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if not str(cfg.get("token") or "").strip() or len(cfg["token"]) < 24:
        raise SystemExit("friday_agent.json needs a token of at least 24 "
                         "characters (copy it from the Pi).")
    cfg.setdefault("port", 8765)
    cfg.setdefault("bind", "0.0.0.0")
    cfg.setdefault("notes_dir", "~/Friday_Notes")
    return cfg


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


# --- the listener ---------------------------------------------------------------

class Agent:
    def __init__(self, cfg: dict, opener=webbrowser.open, shower=show_file,
                 fetch=_fetch) -> None:
        self.cfg = cfg
        self.opener = opener
        self.shower = shower
        self.fetch = fetch

    def handler(self):
        agent = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "FridayLaptopHelper/2"

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
                return hmac.compare_digest(given.encode(),
                                           agent.cfg["token"].encode())

            def do_GET(self):                           # noqa: N802
                if not self._authorised():
                    return self._send(401, {"error": "unauthorised"})
                if self.path != "/health":
                    return self._send(404, {"error": "not found"})
                return self._send(200, {"ok": True})

            def do_POST(self):                          # noqa: N802
                if not self._authorised():
                    return self._send(401, {"error": "unauthorised"})
                if self.path not in ("/open", "/notes"):
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
    server = ThreadingHTTPServer((cfg["bind"], int(cfg["port"])),
                                 Agent(cfg).handler())
    log.info("listening on %s:%s, notes -> %s", cfg["bind"], cfg["port"],
             notes_dir(cfg))
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
