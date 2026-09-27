#!/usr/bin/env python3
"""Friday's laptop research agent (File 20). Runs on HARSH'S LAPTOP, by hand.

    pip install anthropic                      # once
    export ANTHROPIC_API_KEY=...               # or `ant auth login`
    python3 friday_laptop_agent.py             # listens until Ctrl-C

It is NOT part of the robot and nothing installs it as a service: when it is
not running, Friday answers from her own web search instead and says so.

WHAT IT DOES -- AND ONLY THIS
  POST /research  {"query": "...", "mode": "answer" | "notes"}
      Deep research with Claude + web search. Returns a short, speakable
      summary and the fuller detail. In "notes" mode it ALSO writes one
      Markdown file into NOTES_DIR (default ~/Friday_Notes) and returns its
      path.
  GET /health     {"ok": true} -- so Friday can tell whether it is up.

  Every request must carry the shared token in the X-Friday-Token header;
  anything without it gets 401 and nothing else. No other endpoints, no
  command execution, no file reads, no writes outside NOTES_DIR, one research
  at a time (a second request while busy gets 429).

CONFIG: friday_agent.json next to this script (copied from the Pi's
secrets/friday_agent.json -- see the Pi's CLAUDE.md PART 11):
    {"token": "...", "port": 8765, "bind": "0.0.0.0",
     "notes_dir": "~/Friday_Notes", "model": "claude-opus-5", "max_searches": 8}
Keep it private: it holds the token.
"""

from __future__ import annotations

import argparse
import hmac
import json
import re
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAX_BODY = 8 * 1024
MAX_QUERY = 1000
MAX_CONTINUATIONS = 5

SYSTEM = (
    "You are the research assistant of Friday, a home robot, working for "
    "Harsh. Research the question properly: search several independent, "
    "reliable sources, cross-check them, and prefer recent information when "
    "it matters. Then reply in EXACTLY this shape:\n\n"
    "SUMMARY:\n<at most 60 words, plain spoken English -- it will be read "
    "aloud: no URLs, no markdown, no lists>\n\n"
    "DETAIL:\n<the full answer in Markdown: headings, bullet points, numbers "
    "and caveats as needed; say where sources disagree>")


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    if not str(cfg.get("token") or "").strip() or len(cfg["token"]) < 24:
        raise SystemExit("friday_agent.json needs a token of at least 24 "
                         "characters (copy it from the Pi).")
    cfg.setdefault("port", 8765)
    cfg.setdefault("bind", "0.0.0.0")
    cfg.setdefault("notes_dir", "~/Friday_Notes")
    cfg.setdefault("model", "claude-opus-5")
    cfg.setdefault("max_searches", 8)
    return cfg


# --- the research -------------------------------------------------------------

def research(query: str, cfg: dict) -> dict:
    """One multi-source research run. -> {"summary", "detail", "sources"}."""
    import anthropic

    client = anthropic.Anthropic()
    tools = [{"type": "web_search_20260209", "name": "web_search",
              "max_uses": int(cfg["max_searches"])}]
    messages = [{"role": "user", "content": query}]
    response = None
    for _ in range(MAX_CONTINUATIONS + 1):
        response = client.beta.messages.create(
            model=cfg["model"],
            max_tokens=16000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            thinking={"type": "adaptive"},
            output_config={"effort": "high"},
            system=SYSTEM,
            tools=tools,
            messages=messages,
        )
        if response.stop_reason != "pause_turn":
            break
        # Resume the server-side search loop: resend with the paused turn.
        messages = [{"role": "user", "content": query},
                    {"role": "assistant", "content": response.content}]
    if response.stop_reason == "refusal":
        return {"summary": "I couldn't research that one.", "detail": "",
                "sources": []}

    text = "".join(b.text for b in response.content
                   if getattr(b, "type", "") == "text")
    sources: dict[str, str] = {}
    for block in response.content:
        for cite in getattr(block, "citations", None) or []:
            url = getattr(cite, "url", None)
            if url:
                sources.setdefault(url, getattr(cite, "title", "") or url)
    summary, detail = _split(text)
    return {"summary": summary, "detail": detail,
            "sources": [{"url": u, "title": t} for u, t in sources.items()]}


def _split(text: str) -> tuple[str, str]:
    found = re.search(r"SUMMARY:\s*(.*?)\s*DETAIL:\s*(.*)$", text, re.S | re.I)
    if found:
        return found.group(1).strip(), found.group(2).strip()
    words = text.split()
    return " ".join(words[:60]), text.strip()


# --- notes: one file, inside its own folder, never anywhere else ---------------

def notes_dir(cfg: dict) -> Path:
    path = Path(cfg["notes_dir"]).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_notes(query: str, result: dict, cfg: dict) -> str:
    folder = notes_dir(cfg)
    slug = re.sub(r"[^a-z0-9]+", "-", query.lower()).strip("-")[:60] or "notes"
    stem = f"{datetime.now():%Y-%m-%d}-{slug}"
    path = folder / f"{stem}.md"
    n = 2
    while path.exists():                      # never overwrite an older note
        path = folder / f"{stem}-{n}.md"
        n += 1
    if path.resolve().parent != folder:       # belt and braces
        raise ValueError("refusing to write outside the notes folder")
    lines = [f"# {query.strip()}", "",
             f"*Researched by Friday on {datetime.now():%Y-%m-%d %H:%M}*", "",
             "## Summary", "", result["summary"], "", result["detail"], ""]
    if result["sources"]:
        lines += ["## Sources", ""] + [f"- [{s['title']}]({s['url']})"
                                       for s in result["sources"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# --- the listener ---------------------------------------------------------------

class Agent:
    def __init__(self, cfg: dict, research_fn=research) -> None:
        self.cfg = cfg
        self.research = research_fn
        self.busy = threading.Lock()

    def handler(self):
        agent = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "FridayLaptopAgent/1"

            def log_message(self, fmt, *args):         # no tokens in logs
                sys.stderr.write(f"{datetime.now():%H:%M:%S} "
                                 f"{self.address_string()} {fmt % args}\n")

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
                return self._send(200, {"ok": True, "busy": agent.busy.locked()})

            def do_POST(self):                          # noqa: N802
                if not self._authorised():
                    return self._send(401, {"error": "unauthorised"})
                if self.path != "/research":
                    return self._send(404, {"error": "not found"})
                size = int(self.headers.get("Content-Length") or 0)
                if size <= 0 or size > MAX_BODY:
                    return self._send(413, {"error": "body too large or empty"})
                try:
                    body = json.loads(self.rfile.read(size))
                    query = str(body["query"]).strip()
                    mode = str(body.get("mode") or "answer")
                except (ValueError, KeyError, TypeError):
                    return self._send(400, {"error": "send {query, mode}"})
                if not query or len(query) > MAX_QUERY:
                    return self._send(400, {"error": "query is empty or too long"})
                if mode not in ("answer", "notes"):
                    return self._send(400, {"error": "mode is answer or notes"})
                if not agent.busy.acquire(blocking=False):
                    return self._send(429, {"error": "busy with another research"})
                try:
                    started = time.monotonic()
                    result = agent.research(query, agent.cfg)
                    out = {"ok": True, "summary": result["summary"],
                           "detail": result["detail"],
                           "sources": result["sources"],
                           "seconds": round(time.monotonic() - started, 1)}
                    if mode == "notes":
                        out["path"] = write_notes(query, result, agent.cfg)
                    return self._send(200, out)
                except Exception as exc:                # noqa: BLE001
                    return self._send(500, {"error": f"{type(exc).__name__}: "
                                                     f"{str(exc)[:200]}"})
                finally:
                    agent.busy.release()

        return Handler


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "friday_agent.json"))
    args = ap.parse_args()
    cfg = load_config(Path(args.config).expanduser())
    notes_dir(cfg)
    server = ThreadingHTTPServer((cfg["bind"], int(cfg["port"])),
                                 Agent(cfg).handler())
    print(f"Friday's laptop agent: listening on {cfg['bind']}:{cfg['port']}, "
          f"notes -> {notes_dir(cfg)}, model {cfg['model']}. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
