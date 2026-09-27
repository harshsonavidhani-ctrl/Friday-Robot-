#!/usr/bin/env python3
"""File 20: Friday uses Harsh's laptop. No laptop, no browser, no money.

    ./venv/bin/python scripts/test_file20.py

The laptop helper (laptop/friday_laptop_agent.py) runs IN THIS PROCESS on
127.0.0.1 with a fake browser and a fake "open in Notepad", so the real HTTP
path, the token check and the notes folder are exercised end to end. Every
file is a temp copy: memory.db, the rate-limit counters, the job records, the
listening / speaking marks and the settings file. The notes model call is
faked (compose_notes), and so is Anthropic where compose_notes itself is
tested.
"""

from __future__ import annotations

import ast
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "laptop"))

import config                                           # noqa: E402

TMP = Path(tempfile.mkdtemp(prefix="friday-file20-"))
config.MEMORY_DB = TMP / "memory.db"
config.RATE_LIMIT_FILE = TMP / "rate_limits.json"
config.RESEARCH_DIR = TMP / "research"
config.LAPTOP_AGENT_FILE = TMP / "secrets" / "laptop_agent.json"
config.LISTENING_FILE = TMP / "listening.json"
config.SPEAKING_FILE = TMP / "speaking.json"
config.MODE_FILE = TMP / "mode.json"
config.BRIEFING = False
config.LAPTOP_HEALTH_TIMEOUT_S = 1.0

import friday_laptop_agent as agent_mod                 # noqa: E402
import laptop_research as lr                            # noqa: E402
import memory                                           # noqa: E402
import offline                                          # noqa: E402
import reminders                                        # noqa: E402
from mcp_server.tools import research_tools as rt       # noqa: E402

_results: list[bool] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append(bool(ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail and not ok:
        print(f"        {detail}")


def wait_for(cond, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def fresh_limits() -> None:
    config.RATE_LIMIT_FILE.unlink(missing_ok=True)


TOKEN = "t" * 40
NOTES = TMP / "Friday_Notes"
CFG = {"token": TOKEN, "port": 0, "bind": "127.0.0.1", "notes_dir": str(NOTES)}
opened: list[str] = []
shown: list[Path] = []
fetched: list[str] = []
YT_PAGE = 'x "videoId":"dQw4w9WgXcQ" y "videoId":"aaaaaaaaaaa"'


def fake_fetch(url: str) -> str:
    fetched.append(url)
    return YT_PAGE


server = ThreadingHTTPServer(
    ("127.0.0.1", 0),
    agent_mod.Agent(CFG, opener=opened.append, shower=shown.append,
                    fetch=fake_fetch).handler())
PORT = server.server_address[1]
threading.Thread(target=server.serve_forever, daemon=True).start()
BASE = f"http://127.0.0.1:{PORT}"


def raw(path: str, body=None, token: str | None = TOKEN,
        data: bytes | None = None) -> tuple[int, dict]:
    if data is None and body is not None:
        data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if token is not None:
        headers["X-Friday-Token"] = token
    req = urllib.request.Request(BASE + path, data=data, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def settings(url: str, token: str = TOKEN) -> None:
    config.LAPTOP_AGENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    config.LAPTOP_AGENT_FILE.write_text(json.dumps({"url": url,
                                                    "token": token}))


# ---------------------------------------------------------------------------
print("the laptop helper: token-gated, three endpoints, nothing else")
check("no token -> 401", raw("/health", token=None)[0] == 401)
check("wrong token -> 401", raw("/health", token="x" * 40)[0] == 401)
code, _ = raw("/open", {"site": "web", "query": "cats"}, token="x" * 40)
check("open with a wrong token -> 401, and nothing opened",
      code == 401 and opened == [])
code, _ = raw("/notes", {"title": "t", "markdown": "m"}, token="x" * 40)
check("notes with a wrong token -> 401, and nothing written",
      code == 401 and not NOTES.exists() or not list(NOTES.glob("*")))
check("health with the token", raw("/health") == (200, {"ok": True}))
check("any other path -> 404", raw("/etc/passwd")[0] == 404
      and raw("/run", {"cmd": "calc.exe"})[0] == 404
      and raw("/research", {"query": "x"})[0] == 404)
check("not JSON -> 400", raw("/open", data=b"not json")[0] == 400)
check("a JSON list -> 400", raw("/open", data=b"[1]")[0] == 400)
check("empty query -> 400", raw("/open", {"query": "  "})[0] == 400)
check("a site other than web/youtube -> 400",
      raw("/open", {"site": "file", "query": "C:/Windows"})[0] == 400)
check("a query over 300 chars -> 400",
      raw("/open", {"query": "a" * 301})[0] == 400)
check("a body over 256 KB -> 413",
      raw("/notes", data=b"{" + b" " * 270_000 + b"}")[0] == 413)
check("none of that opened anything", opened == [])
(TMP / "short.json").write_text(json.dumps({"token": "short"}))
try:
    agent_mod.load_config(TMP / "short.json")
    short_ok = False
except SystemExit:
    short_ok = True
check("the helper refuses to start with a short token", short_ok)

print("\n/open builds the URL itself")
code, got = raw("/open", {"site": "web", "query": "best phones & prices"})
check("web -> a Google search, words encoded",
      code == 200 and opened[-1] ==
      "https://www.google.com/search?q=best+phones+%26+prices", opened[-1:])
code, got = raw("/open", {"site": "youtube", "query": "lofi music"})
check("youtube -> the results page, nothing fetched",
      opened[-1] == "https://www.youtube.com/results?search_query=lofi+music"
      and fetched == [])
code, got = raw("/open", {"site": "youtube", "query": "believer",
                          "play": True})
check("youtube + play -> the TOP video",
      opened[-1] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
      and fetched == ["https://www.youtube.com/results?search_query=believer"],
      opened[-1:])
check("play on the web is just a search",
      raw("/open", {"site": "web", "query": "x", "play": True})[0] == 200
      and opened[-1].startswith("https://www.google.com/search?q="))
code, got = raw("/open", {"site": "web",
                          "query": "https://evil.example/x.exe"})
check("a URL sent as the query is only ever searched for",
      opened[-1].startswith("https://www.google.com/search?q=https%3A%2F%2F"))
check("no top video found -> the results page",
      agent_mod.build_url("youtube", "x", True, fetch=lambda u: "nothing")
      == "https://www.youtube.com/results?search_query=x")


def boom(url):
    raise OSError("offline")


check("YouTube unreachable -> the results page",
      agent_mod.build_url("youtube", "x", True, fetch=boom)
      == "https://www.youtube.com/results?search_query=x")
check("a bad video id is never used",
      agent_mod.build_url("youtube", "x", True,
                          fetch=lambda u: '"videoId":"../../etc"')
      .endswith("results?search_query=x"))

print("\n/notes: one file, in its folder, opened")
code, got = raw("/notes", {"title": "Green tea benefits",
                           "markdown": "# Green tea\n\n- good"})
path = Path(got.get("path", ""))
check("writes ONE .md inside the notes folder",
      code == 200 and path.parent == NOTES.resolve() and path.suffix == ".md"
      and path.read_text() == "# Green tea\n\n- good\n", str(got))
check("...and opens it on his screen", shown == [path])
code, again = raw("/notes", {"title": "Green tea benefits",
                             "markdown": "# again"})
check("the same topic twice never overwrites",
      again["path"].endswith("-2.md") and path.read_text().startswith(
          "# Green tea"), again.get("path"))
code, evil = raw("/notes", {"title": "../../../Windows/System32/x",
                            "markdown": "m"})
evil_path = Path(evil.get("path", "/"))
check("a title of ../../../Windows/System32/x stays in the folder",
      code == 200 and evil_path.parent == NOTES.resolve()
      and ".." not in evil_path.name, evil.get("path"))
check("a title with no letters still gets a safe name",
      Path(raw("/notes", {"title": "???", "markdown": "m"})[1]["path"])
      .name.endswith("-notes.md"))
check("no title / no text -> 400",
      raw("/notes", {"title": "", "markdown": "m"})[0] == 400
      and raw("/notes", {"title": "t", "markdown": "  "})[0] == 400)

source = (ROOT / "laptop" / "friday_laptop_agent.py").read_text()
tree = ast.parse(source)
popen_args = [ast.unparse(n.args[0]) for n in ast.walk(tree)
              if isinstance(n, ast.Call) and ast.unparse(n.func) ==
              "subprocess.Popen"]
check("the only programs it starts: Notepad, and itself (--install)",
      sorted(popen_args) == ["['notepad.exe', str(path)]",
                             "[_pythonw(), str(Path(__file__).resolve())]"],
      str(popen_args))
check("...never a shell, eval, exec or os.system",
      not any(w in source for w in ("shell=True", "eval(", "exec(",
                                    "os.system", "os.popen")))
check("...and it never opens a URL it did not build",
      "agent.opener(url)" in source and source.count("opener(") == 1)

# ---------------------------------------------------------------------------
print("\nthe Pi side: is it up?")
check("no settings file: not configured, not available",
      not lr.configured() and lr.available()[0] is False)
settings(BASE)
check("settings file: available", lr.available() == (True, ""))
settings(BASE, token="w" * 40)
ok, why = lr.available()
check("a wrong token: not available, and says it was refused",
      not ok and "401" in why, why)
closed = socket.socket()
closed.bind(("127.0.0.1", 0))
dead_port = closed.getsockname()[1]
closed.close()
settings(f"http://127.0.0.1:{dead_port}")
started = time.monotonic()
ok, why = lr.available()
check("nothing listening: not available, at once",
      not ok and time.monotonic() - started < 1.0, why)
silent = socket.socket()
silent.bind(("127.0.0.1", 0))
silent.listen(1)                                         # never answers
settings(f"http://127.0.0.1:{silent.getsockname()[1]}")
started = time.monotonic()
ok, why = lr.available()
took = time.monotonic() - started
check("a laptop that never answers: gives up after the health timeout",
      not ok and took < config.LAPTOP_HEALTH_TIMEOUT_S + 0.5, f"{took:.1f}s")

# ---------------------------------------------------------------------------
print("\nthe laptop ONLY: down means she says so, and nothing else")
composed: list[str] = []
real_compose = lr.compose_notes
gate = threading.Event()
gate.set()
compose_result = {"value": ({"summary": "Tea is good for you.",
                             "notes": "# Tea\n\n- antioxidants"}, "")}


def fake_compose(topic):
    composed.append(topic)
    gate.wait(10)
    return compose_result["value"]


lr.compose_notes = fake_compose
settings(f"http://127.0.0.1:{dead_port}")
n_opened = len(opened)
said = rt.laptop_search("cat videos", "youtube")
check("search, laptop down: 'My laptop isn't available right now.'",
      said == "My laptop isn't available right now.", said)
said = rt.laptop_notes("green tea")
check("notes, laptop down: says so plainly",
      said == "My laptop isn't available right now. I can't put notes there.",
      said)
check("...no search anywhere, no model call, no job, nothing opened",
      composed == [] and lr.jobs() == [] and len(opened) == n_opened)
config.LAPTOP_AGENT_FILE.unlink()
check("never set up at all: the same",
      rt.laptop_search("x") == "My laptop isn't available right now.")
check("empty asks what", rt.laptop_search("  ") ==
      "What should I search for?" and rt.laptop_notes("") ==
      "What should I make notes on?")
no_fallback = (ROOT / "laptop_research.py").read_text() + \
    (ROOT / "mcp_server/tools/research_tools.py").read_text()
check("there is no Pi-side answer path any more",
      "fallback_answer" not in no_fallback and "direct answer" not in
      no_fallback)

# ---------------------------------------------------------------------------
print("\nlaptop up: a search opens on his screen")
settings(BASE)
fresh_limits()
said = rt.laptop_search("the best budget phones")
check("web", said == "Searching for the best budget phones on your laptop."
      and opened[-1].endswith("q=the+best+budget+phones"), said)
fresh_limits()
said = rt.laptop_search("believer", "youtube", True)
check("YouTube, play", said == "Playing believer on YouTube on your laptop."
      and opened[-1].endswith("v=dQw4w9WgXcQ"), said)
fresh_limits()
said = rt.laptop_search("lofi", "youtube")
check("YouTube, search", said == "Searching YouTube for lofi on your laptop.")
said = rt.laptop_search("again", "youtube")
check("a second one inside 3 s is held by the rate limit",
      said.startswith("I've opened a lot on your laptop already"), said)
fresh_limits()
rt.laptop_search("x", "ftp")
check("an unknown site becomes a web search",
      opened[-1].startswith("https://www.google.com/"))

print("\nlaptop up: notes -- returns at once, said later")
fresh_limits()
gate.clear()
started = time.monotonic()
said = rt.laptop_notes("the history of tea")
check("returns at once while the notes are written",
      time.monotonic() - started < 2.0 and "put the notes on your laptop"
      in said, said)
check("...the job is recorded as running",
      [j["status"] for j in lr.jobs()] == ["running"])
watch = reminders.ReminderWatch()
check("nothing is said while it runs", watch.step() is None)
gate.set()
check("the job finishes", wait_for(lambda: lr.jobs()[-1]["status"] == "done"))
job = lr.jobs()[-1]
saved = Path(job["path"])
check("the notes were saved on the laptop, and opened",
      saved.parent == NOTES.resolve() and saved.read_text().startswith(
          "# Tea") and shown[-1] == saved)
item = watch.step()
check("ReminderWatch offers it, once",
      item is not None and item["kind"] == "research"
      and watch.step() is None, str(item))
check("...'open on your laptop, saved in Friday Notes as <file>. In short: …'",
      item["say"] == f"Your notes on the history of tea are open on your "
                     f"laptop, saved in Friday Notes as {saved.name}. In "
                     f"short: Tea is good for you.", item["say"])
watch.missed(item)
check("handed back (he started talking): offered again",
      (watch.step() or {}).get("job") == item["job"])
config.LISTENING_FILE.write_text(json.dumps({"listening": True,
                                             "at": time.time()}))
check("held while he is talking", reminders.ReminderWatch().step() is None)
config.LISTENING_FILE.unlink()
watch.delivered(item)
check("said -> never again", lr.finished() == [] and watch.step() is None)
facts = memory.db().execute(
    "SELECT text, source FROM facts WHERE kind='research'").fetchall()
check("remembered as kind 'research', with where the notes are",
      len(facts) == 1 and facts[0]["source"] == "laptop"
      and "the history of tea" in facts[0]["text"]
      and saved.name in facts[0]["text"], str([dict(f) for f in facts]))
check("research_detail reads the notes back",
      "antioxidants" in rt.research_detail("tea")
      and saved.name in rt.research_detail("tea"))
check("research_detail on nothing like it",
      rt.research_detail("volcanoes") ==
      "I haven't made notes on anything like that yet.")

print("\none try, never a retry")
fresh_limits()
compose_result["value"] = (None, "over the daily spend cap")
n_composed = len(composed)
rt.laptop_notes("volcanoes")
check("research failed: status failed, one attempt",
      wait_for(lambda: lr.jobs()[-1]["status"] == "failed")
      and len(composed) - n_composed == 1)
item = watch.step()
check("...she says there are no notes", item and item["say"] ==
      "I couldn't research volcanoes just now, so there are no notes.",
      str(item))
watch.delivered(item)

compose_result["value"] = ({"summary": "s", "notes": "# n"}, "")
fresh_limits()
n_files = len(list(NOTES.glob("*.md")))
real_request = lr._request


def laptop_gone(path, body, timeout):
    if path == "/notes":
        raise ConnectionError("gone")
    return real_request(path, body, timeout)


lr._request = laptop_gone
rt.laptop_notes("rivers")
check("laptop went away before the save: failed, nothing written",
      wait_for(lambda: lr.jobs()[-1]["status"] == "failed")
      and len(list(NOTES.glob("*.md"))) == n_files
      and lr.jobs()[-1]["stage"] == "save")
lr._request = real_request
item = watch.step()
check("...and she says so", item and "my laptop went away before I could "
      "save the notes" in item["say"], str(item))
watch.delivered(item)
check("a failed job is not remembered",
      memory.db().execute("SELECT count(*) FROM facts WHERE kind='research'")
      .fetchone()[0] == 1)

fresh_limits()
rt.laptop_notes("first")
said = rt.laptop_notes("second")
check("a second notes request inside 20 s is refused by the rate limit",
      said.startswith("I've made a lot of notes already"), said)
wait_for(lambda: lr.finished() != [])
old_keep = config.RESEARCH_KEEP
config.RESEARCH_KEEP = 3
lr._write({"id": "9999999999999", "query": "q", "mode": "notes",
           "status": "done", "at": time.time()})
check("job records are bounded", len(lr.jobs()) == 3)
config.RESEARCH_KEEP = old_keep
lr.compose_notes = real_compose

# ---------------------------------------------------------------------------
print("\ncompose_notes (anthropic faked)")
import anthropic                                        # noqa: E402
import brain                                            # noqa: E402
import scene_describe                                   # noqa: E402


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[dict] = []
        self.messages = self

    def create(self, **kw):
        self.sent.append(kw)
        return self.replies.pop(0)


usage = types.SimpleNamespace(input_tokens=10, output_tokens=5,
                              cache_read_input_tokens=0,
                              cache_creation_input_tokens=0,
                              server_tool_use=None)


def reply(stop, text=""):
    content = [types.SimpleNamespace(type="text", text=text)] if text else []
    return types.SimpleNamespace(stop_reason=stop, usage=usage,
                                 content=content)


spend: list = []
real = (scene_describe._api_key, brain.spent_today, brain._record_spend,
        anthropic.Anthropic)
try:
    scene_describe._api_key = lambda: None
    check("no key: no call, says why",
          lr.compose_notes("x") == (None, "no API key"))
    scene_describe._api_key = lambda: "sk-test"
    brain.spent_today = lambda: config.DAILY_SPEND_CAP_USD
    check("over the daily cap: no call",
          lr.compose_notes("x") == (None, "over the daily spend cap"))
    brain.spent_today = lambda: 0.0
    brain._record_spend = lambda *a: spend.append(a)
    fake = FakeClient([reply("pause_turn"), reply(
        "end_turn", "SUMMARY:\nRayleigh scattering.\n\nNOTES:\n# Sky\n\n- blue")])
    anthropic.Anthropic = lambda *a, **k: fake
    got = lr.compose_notes("why is the sky blue")
    check("Haiku + web_search_20250305, pause_turn resumed, spend recorded",
          got == ({"summary": "Rayleigh scattering.", "notes": "# Sky\n\n- blue"},
                  "") and len(fake.sent) == 2
          and fake.sent[0]["model"] == config.LAPTOP_NOTES_MODEL
          and fake.sent[0]["tools"][0]["type"] == "web_search_20250305"
          and fake.sent[1]["messages"][-1]["role"] == "assistant"
          and len(spend) == 2, str(got))
    fake = FakeClient([reply("refusal")])
    anthropic.Anthropic = lambda *a, **k: fake
    check("a refusal is no notes", lr.compose_notes("x")[0] is None)
finally:
    (scene_describe._api_key, brain.spent_today, brain._record_spend,
     anthropic.Anthropic) = real

# ---------------------------------------------------------------------------
print("\nvoice, config, memory types, secrets")
for heard, tool, args in [
        ("Friday, play Believer on YouTube", "laptop_search",
         {"query": "Believer", "site": "youtube", "play": True}),
        ("search for cat videos on youtube", "laptop_search",
         {"query": "cat videos", "site": "youtube", "play": False}),
        ("open youtube and play lofi music", "laptop_search",
         {"query": "lofi music", "site": "youtube", "play": True}),
        ("search the best budget phones on my laptop", "laptop_search",
         {"query": "the best budget phones", "site": "web", "play": False}),
        ("google weather in Pune", "laptop_search",
         {"query": "weather in Pune", "site": "web", "play": False}),
        ("Friday, make notes on the French revolution", "laptop_notes",
         {"topic": "the French revolution"}),
        ("properly research quantum computing", "laptop_notes",
         {"topic": "quantum computing"}),
        ("research black holes on my laptop", "laptop_notes",
         {"topic": "black holes"}),
        ("play stop motion on youtube", "laptop_search",
         {"query": "stop motion", "site": "youtube", "play": True})]:
    i = offline.understand(heard, strict=True)
    check(f"'{heard}' -> {tool}",
          i is not None and i.tool == tool and i.args == args,
          str(i and (i.tool, i.args)))
for heard in ("what is the capital of France", "search for my keys",
              "remember that my exam is on Monday", "stop", "play some music"):
    i = offline.understand(heard, strict=True)
    check(f"'{heard}' is not the laptop",
          i is None or i.tool not in ("laptop_search", "laptop_notes"))
check("laptop phrases are reflexes that speak their result",
      {"laptop_search", "laptop_notes"} <= offline._REFLEX_INTENTS
      and {"laptop_search", "laptop_notes"} <= offline._SPEAK_RESULT)
check("'laptop' / 'youtube' / 'research' skip the no-tools fast path",
      {"laptop", "youtube", "research"} <= set(config.BRAIN_FAST_NEVER))
check("both laptop tools are rate limited",
      {"laptop_open", "laptop_notes"} <= set(config.RATE_LIMITS))
import memory_governance                                # noqa: E402
check("'research' is a documented memory type",
      "research" in memory_governance.TYPES)
ignored = all(subprocess.run(["git", "check-ignore", "-q", name],
                             cwd=ROOT).returncode == 0
              for name in ("secrets/laptop_agent.json",
                           "secrets/friday_agent.json"))
check("the token files are gitignored", ignored)
check("no token in the code",
      not any(TOKEN in p.read_text(errors="ignore")
              for p in (ROOT / "laptop_research.py",
                        ROOT / "mcp_server/tools/research_tools.py")))

server.shutdown()
silent.close()
print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
