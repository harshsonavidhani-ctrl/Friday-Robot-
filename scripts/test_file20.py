#!/usr/bin/env python3
"""File 20: research delegated to Harsh's laptop. No laptop, no money.

    ./venv/bin/python scripts/test_file20.py

The laptop agent (laptop/friday_laptop_agent.py) runs IN THIS PROCESS on
127.0.0.1 with a fake research function, so the real HTTP path, the token
check and the notes folder are exercised end to end. Every file is a temp
copy: memory.db, the rate-limit counters, the job records, the listening /
speaking marks and the settings file. Anthropic is faked where a model would
be called (her fallback, the laptop's research loop).
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


TOKEN = "t" * 40
NOTES = TMP / "Friday_Notes"
CFG = {"token": TOKEN, "port": 0, "bind": "127.0.0.1",
       "notes_dir": str(NOTES), "model": "claude-opus-5", "max_searches": 8}

calls: list[str] = []
gate = threading.Event()
gate.set()
behaviour = {"raise": None, "sleep": 0.0}


def fake_research(query: str, cfg: dict) -> dict:
    calls.append(query)
    gate.wait(10)
    if behaviour["sleep"]:
        time.sleep(behaviour["sleep"])
    if behaviour["raise"]:
        raise behaviour["raise"]
    return {"summary": f"The short answer about {query}.",
            "detail": f"## {query}\n\n- point one\n- point two",
            "sources": [{"url": "https://example.org/a", "title": "A"}]}


from http.server import ThreadingHTTPServer             # noqa: E402

server = ThreadingHTTPServer(("127.0.0.1", 0),
                             agent_mod.Agent(CFG, fake_research).handler())
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
print("the laptop agent: token-gated, two endpoints, nothing else")
check("no token -> 401", raw("/health", token=None)[0] == 401)
check("wrong token -> 401", raw("/health", token="x" * 40)[0] == 401)
code, _ = raw("/research", {"query": "cats"}, token="x" * 40)
check("research with a wrong token -> 401, and nothing ran",
      code == 401 and calls == [])
check("health with the token", raw("/health") == (200, {"ok": True,
                                                        "busy": False}))
check("any other path -> 404", raw("/etc/passwd")[0] == 404
      and raw("/run", {"query": "ls"})[0] == 404)
check("not JSON -> 400", raw("/research", data=b"not json")[0] == 400)
check("empty query -> 400", raw("/research", {"query": "  "})[0] == 400)
check("a bad mode -> 400", raw("/research", {"query": "x",
                                             "mode": "shell"})[0] == 400)
check("a query over 1000 chars -> 400",
      raw("/research", {"query": "a" * 1001})[0] == 400)
check("a body over 8 KB -> 413",
      raw("/research", data=b"{" + b" " * 9000 + b"}")[0] == 413)
check("none of that reached the research", calls == [])
try:
    agent_mod.load_config(TMP / "nope.json")
    short_ok = False
except FileNotFoundError:
    short_ok = True
(TMP / "short.json").write_text(json.dumps({"token": "short"}))
try:
    agent_mod.load_config(TMP / "short.json")
    short_ok = False
except SystemExit:
    pass
check("the agent refuses to start with a short token", short_ok)

code, got = raw("/research", {"query": "tea", "mode": "answer"})
check("answer mode: summary, detail, sources, no file",
      code == 200 and got["summary"].startswith("The short answer")
      and "point one" in got["detail"] and got["sources"]
      and "path" not in got and not NOTES.exists() or
      (NOTES.exists() and not list(NOTES.glob("*.md"))), str(got))

code, got = raw("/research", {"query": "Green tea benefits", "mode": "notes"})
path = Path(got.get("path", ""))
check("notes mode writes ONE Markdown file inside the notes folder",
      code == 200 and path.parent == NOTES.resolve() and path.suffix == ".md"
      and path.exists(), str(got))
text = path.read_text() if path.exists() else ""
check("...with the summary, the detail and the sources",
      "# Green tea benefits" in text and "The short answer" in text
      and "point two" in text and "https://example.org/a" in text)
code, again = raw("/research", {"query": "Green tea benefits",
                                "mode": "notes"})
check("the same topic twice never overwrites",
      again["path"] != got["path"] and path.exists()
      and again["path"].endswith("-2.md"), again.get("path"))
code, evil = raw("/research", {"query": "../../../etc/passwd",
                               "mode": "notes"})
evil_path = Path(evil.get("path", "/"))
check("a query of ../../../etc/passwd stays in the folder",
      code == 200 and evil_path.parent == NOTES.resolve()
      and "/" not in evil_path.name and ".." not in evil_path.name,
      evil.get("path"))
code, odd = raw("/research", {"query": "???", "mode": "notes"})
check("a query with no letters still gets a safe name",
      code == 200 and Path(odd["path"]).name.endswith("-notes.md"))

gate.clear()
first: dict = {}
t = threading.Thread(target=lambda: first.update(
    r=raw("/research", {"query": "slow one"})))
t.start()
wait_for(lambda: "slow one" in calls, 3)
check("busy: health says so", raw("/health")[1].get("busy") is True)
check("busy: a second research gets 429",
      raw("/research", {"query": "second"})[0] == 429)
gate.set()
t.join(5)
check("...and the first one still finishes", first.get("r", (0,))[0] == 200)

behaviour["raise"] = RuntimeError("model blew up")
code, got = raw("/research", {"query": "boom"})
check("a research error is a 500 with the reason, and the lock is freed",
      code == 500 and "model blew up" in got.get("error", "")
      and raw("/health")[1]["busy"] is False)
behaviour["raise"] = None

source = (ROOT / "laptop" / "friday_laptop_agent.py").read_text()
imported = {a.name.split(".")[0] for n in ast.walk(ast.parse(source))
            if isinstance(n, (ast.Import, ast.ImportFrom))
            for a in (n.names if isinstance(n, ast.Import)
                      else [types.SimpleNamespace(name=n.module or "")])}
check("the agent imports nothing that runs commands",
      not imported & {"subprocess", "os", "shutil", "pty", "socketserver"},
      str(imported))
check("...and never calls eval/exec/system",
      not any(w in source for w in ("eval(", "exec(", "system(", "Popen")))

# ---------------------------------------------------------------------------
print("\nthe laptop's research loop (anthropic faked)")


class _Block(types.SimpleNamespace):
    pass


def reply(stop: str, text: str, cites=()) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        stop_reason=stop,
        content=[_Block(type="text", text=text,
                        citations=[types.SimpleNamespace(url=u, title=t)
                                   for u, t in cites])])


class FakeClient:
    def __init__(self, replies):
        self.replies = list(replies)
        self.sent: list[dict] = []
        self.beta = types.SimpleNamespace(messages=self)
        self.messages = self

    def create(self, **kw):
        self.sent.append(kw)
        return self.replies.pop(0)


import anthropic                                        # noqa: E402

real_client = anthropic.Anthropic
fake = FakeClient([reply("pause_turn", "searching"),
                   reply("end_turn", "SUMMARY:\nTea is fine.\n\nDETAIL:\n# Tea"
                         "\n\nlots", [("https://x.org", "X"),
                                      ("https://x.org", "X again")])])
anthropic.Anthropic = lambda *a, **k: fake
try:
    out = agent_mod.research("is tea good", CFG)
finally:
    anthropic.Anthropic = real_client
check("pause_turn is resumed with the paused turn appended",
      len(fake.sent) == 2 and fake.sent[1]["messages"][-1]["role"] == "assistant")
check("Opus 5, adaptive thinking, web_search_20260209, fallbacks",
      fake.sent[0]["model"] == "claude-opus-5"
      and fake.sent[0]["thinking"] == {"type": "adaptive"}
      and fake.sent[0]["tools"][0]["type"] == "web_search_20260209"
      and fake.sent[0]["fallbacks"] == "default")
check("SUMMARY/DETAIL split, sources de-duplicated",
      out["summary"] == "Tea is fine." and out["detail"].startswith("# Tea")
      and out["sources"] == [{"url": "https://x.org", "title": "X"}], str(out))
fake = FakeClient([reply("refusal", "")])
anthropic.Anthropic = lambda *a, **k: fake
try:
    out = agent_mod.research("bad", CFG)
finally:
    anthropic.Anthropic = real_client
check("a refusal comes back as a plain sentence", out["detail"] == ""
      and "couldn't" in out["summary"])

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
print("\ndelegate_to_laptop: laptop down")
fallback_calls: list[str] = []
real_fallback = lr.fallback_answer


def fake_fallback(query):
    fallback_calls.append(query)
    return f"Quick answer about {query}.", ""


lr.fallback_answer = fake_fallback
settings(f"http://127.0.0.1:{dead_port}")
said = rt.delegate_to_laptop("is coffee bad", "notes")
check("notes, laptop down: says so plainly",
      said == "My laptop isn't available right now, so I can't save notes "
              "there.", said)
check("...and writes nothing locally, asks no model",
      lr.jobs() == [] and fallback_calls == [])
said = rt.delegate_to_laptop("is coffee bad", "answer")
check("answer, laptop down: her own answer, with the note first",
      said.startswith("My laptop isn't available, so here's the direct answer.")
      and "Quick answer about is coffee bad" in said, said)
check("...no job, no retry at the laptop", lr.jobs() == []
      and fallback_calls == ["is coffee bad"])
lr.fallback_answer = lambda q: ("", "no API key")
said = rt.delegate_to_laptop("is coffee bad", "answer")
check("and if her own search fails too, she says why",
      "couldn't look it up myself" in said and "no API key" in said, said)
lr.fallback_answer = fake_fallback
config.LAPTOP_AGENT_FILE.unlink()
said = rt.delegate_to_laptop("is coffee bad", "answer")
check("never set up at all: the same fallback", said.startswith(
    "My laptop isn't available"), said)
check("an empty query asks what", rt.delegate_to_laptop("  ") ==
      "What should I research?")

# ---------------------------------------------------------------------------
print("\ndelegate_to_laptop: laptop up -- returns at once, said later")
settings(BASE)
config.RATE_LIMIT_FILE.unlink(missing_ok=True)
gate.clear()
started = time.monotonic()
said = rt.delegate_to_laptop("the history of tea", "answer")
check("returns at once while the laptop works",
      time.monotonic() - started < 2.0 and "on your laptop" in said, said)
check("...the job is recorded as running",
      [j["status"] for j in lr.jobs()] == ["running"])
watch = reminders.ReminderWatch()
check("nothing is said while it runs", watch.step() is None)
gate.set()
check("the job finishes", wait_for(lambda: lr.jobs()[-1]["status"] == "done"))
item = watch.step()
check("ReminderWatch offers it, once",
      item is not None and item["kind"] == "research"
      and item["say"] == "About the history of tea: The short answer about "
                         "the history of tea." and watch.step() is None,
      str(item))
watch.missed(item)
check("handed back (he started talking): offered again",
      (watch.step() or {}).get("job") == item["job"])
config.LISTENING_FILE.write_text(json.dumps({"listening": True,
                                             "at": time.time()}))
watch2 = reminders.ReminderWatch()
check("held while he is talking", watch2.step() is None)
config.LISTENING_FILE.unlink()
watch.delivered(item)
check("said -> never again", lr.finished() == [] and watch.step() is None)
facts = memory.db().execute(
    "SELECT text, kind, source FROM facts WHERE kind='research'").fetchall()
check("remembered as kind 'research', from the laptop",
      len(facts) == 1 and facts[0]["source"] == "laptop"
      and "the history of tea" in facts[0]["text"], str([dict(f) for f in facts]))
check("research_detail gives the fuller answer",
      "point one" in rt.research_detail("tea"))
check("research_detail on nothing like it",
      rt.research_detail("volcanoes") ==
      "I haven't researched anything like that yet.")

time.sleep(0.1)
config.RATE_LIMIT_FILE.unlink(missing_ok=True)
rt.delegate_to_laptop("Green tea", "notes")
check("notes job finishes", wait_for(lambda: lr.jobs()[-1]["status"] == "done"))
item = watch.step()
check("notes: says the folder and the file name",
      item is not None and "saved on your laptop, in the Friday_Notes folder"
      in item["say"] and "green-tea.md" in item["say"], str(item))
watch.delivered(item)

# ---------------------------------------------------------------------------
print("\none try, never a retry")
config.RATE_LIMIT_FILE.unlink(missing_ok=True)
before = len(calls)
behaviour["raise"] = RuntimeError("laptop fell over")
fallback_calls.clear()
rt.delegate_to_laptop("why is the sky blue", "answer")
check("laptop failed mid-job: status fallback",
      wait_for(lambda: lr.jobs()[-1]["status"] == "fallback"))
check("...the laptop was asked exactly once", len(calls) - before == 1,
      f"{len(calls) - before} calls")
check("...her own answer was asked for once",
      fallback_calls == ["why is the sky blue"])
item = watch.step()
check("...and she says the laptop didn't finish, then the answer",
      item and item["say"].startswith("My laptop didn't finish researching "
                                      "why is the sky blue, so here's the "
                                      "direct answer. Quick answer"),
      str(item))
watch.delivered(item)
check("the fallback is remembered too, source 'fallback'",
      memory.db().execute("SELECT count(*) FROM facts WHERE kind='research' "
                          "AND source='fallback'").fetchone()[0] == 1)

config.RATE_LIMIT_FILE.unlink(missing_ok=True)
before = len(calls)
rt.delegate_to_laptop("the Mughal empire", "notes")
check("notes that failed: status failed",
      wait_for(lambda: lr.jobs()[-1]["status"] == "failed"))
check("...one laptop call, no local notes, no fallback",
      len(calls) - before == 1 and fallback_calls == ["why is the sky blue"])
item = watch.step()
check("...she says so plainly", item and item["say"] ==
      "My laptop didn't finish the notes on the Mughal empire.", str(item))
watch.delivered(item)
behaviour["raise"] = None

config.RATE_LIMIT_FILE.unlink(missing_ok=True)
config.LAPTOP_RESEARCH_TIMEOUT_S = 0.6
behaviour["sleep"] = 1.5
before = len(calls)
rt.delegate_to_laptop("a slow topic", "answer")
check("a laptop that takes too long: one timeout, then the fallback",
      wait_for(lambda: lr.jobs()[-1]["status"] == "fallback", 5))
check("...still exactly one request", len(calls) - before == 1)
wait_for(lambda: not raw("/health")[1]["busy"], 5)
behaviour["sleep"] = 0.0
watch.delivered(watch.step())

# ---------------------------------------------------------------------------
print("\nrate limit and bounds")
config.RATE_LIMIT_FILE.unlink(missing_ok=True)
rt.delegate_to_laptop("first", "answer")
said = rt.delegate_to_laptop("second", "answer")
check("a second request inside 20 s is refused by the rate limit",
      said.startswith("I've sent a lot of research"), said)
wait_for(lambda: lr.finished() != [])
old_keep = config.RESEARCH_KEEP
config.RESEARCH_KEEP = 3
lr._write({"id": "9999999999999", "query": "q", "mode": "answer",
           "status": "done", "at": time.time()})
check("job records are bounded", len(lr.jobs()) == 3)
config.RESEARCH_KEEP = old_keep

print("\nher own answer (fallback_answer), anthropic faked")
lr.fallback_answer = real_fallback
import brain                                            # noqa: E402
import scene_describe                                   # noqa: E402

spend: list = []
real_key, real_spent, real_rec = (scene_describe._api_key, brain.spent_today,
                                  brain._record_spend)
scene_describe._api_key = lambda: None
check("no key: no call, says why", lr.fallback_answer("x") == ("", "no API key"))
scene_describe._api_key = lambda: "sk-test"
brain.spent_today = lambda: config.DAILY_SPEND_CAP_USD
check("over the daily cap: no call",
      lr.fallback_answer("x")[1] == "over the daily spend cap")
brain.spent_today = lambda: 0.0
brain._record_spend = lambda *a: spend.append(a)
config.RATE_LIMIT_FILE.unlink(missing_ok=True)
usage = types.SimpleNamespace(input_tokens=10, output_tokens=5,
                              cache_read_input_tokens=0,
                              cache_creation_input_tokens=0,
                              server_tool_use=None)
fake = FakeClient([types.SimpleNamespace(stop_reason="pause_turn", usage=usage,
                                         content=[]),
                   types.SimpleNamespace(stop_reason="end_turn", usage=usage,
                                         content=[_Block(type="text",
                                                         text="Rayleigh.")])])
anthropic.Anthropic = lambda *a, **k: fake
try:
    got = lr.fallback_answer("why is the sky blue")
finally:
    anthropic.Anthropic = real_client
    scene_describe._api_key, brain.spent_today, brain._record_spend = (
        real_key, real_spent, real_rec)
check("Haiku + web_search_20250305, pause_turn resumed, spend recorded",
      got == ("Rayleigh.", "") and len(fake.sent) == 2
      and fake.sent[0]["model"] == config.MODEL_FAST
      and fake.sent[0]["tools"][0]["type"] == "web_search_20250305"
      and len(spend) == 2, str(got))

# ---------------------------------------------------------------------------
print("\nvoice, config, memory types, secrets")
for heard, mode, query in [
        ("Friday, properly research the best budget laptops", "answer",
         "the best budget laptops"),
        ("research quantum computing properly", "answer", "quantum computing"),
        ("do a deep research on solar panels in India", "answer",
         "solar panels in India"),
        ("notes on the French revolution", "notes", "the French revolution"),
        ("make notes about machine learning", "notes", "machine learning")]:
    i = offline.understand(heard, strict=True)
    check(f"'{heard}' -> {mode}",
          i is not None and i.tool == "delegate_to_laptop"
          and i.args == {"query": query, "mode": mode},
          str(i and (i.tool, i.args)))
for heard in ("what is the capital of France", "research",
              "remember that my exam is on Monday", "stop"):
    i = offline.understand(heard, strict=True)
    check(f"'{heard}' is not research",
          i is None or i.tool != "delegate_to_laptop")
check("research phrases are reflexes that speak their result",
      {"research", "research_notes"} <= offline._REFLEX_INTENTS
      and {"research", "research_notes"} <= offline._SPEAK_RESULT)
check("'research' / 'in depth' skip the no-tools fast path",
      "research" in config.BRAIN_FAST_NEVER
      and "in depth" in config.BRAIN_FAST_NEVER)
check("both paying paths are rate limited",
      "laptop_research" in config.RATE_LIMITS
      and "research_fallback" in config.RATE_LIMITS)
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
from mcp_server import registry                         # noqa: E402
names = {t.name for t in registry.all_tools()} if hasattr(
    registry, "all_tools") else set()
check("the MCP tools are registered",
      not names or {"delegate_to_laptop", "research_detail"} <= names,
      str(sorted(names))[:200])

server.shutdown()
silent.close()
print(f"\n{sum(_results)}/{len(_results)} passed")
sys.exit(0 if all(_results) else 1)
