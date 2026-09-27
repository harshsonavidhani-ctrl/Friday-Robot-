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

SECRETS: secrets/laptop_agent.json {url, token} -- gitignored, never in code.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import urllib.error
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


def _request(path: str, body: dict | None, timeout: float) -> dict:
    """One HTTP call to the laptop helper. Raises on any failure."""
    s = _settings()
    if s is None:
        raise ConnectionError("the laptop is not set up "
                              "(secrets/laptop_agent.json)")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        s["url"].rstrip("/") + path, data=data,
        method="POST" if body is not None else "GET",
        headers={"X-Friday-Token": s["token"],
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as reply:
        return json.loads(reply.read().decode("utf-8", "replace"))


def available() -> tuple[bool, str]:
    """Is the helper up and accepting our token? One quick look, no retry."""
    try:
        got = _request("/health", None, float(config.LAPTOP_HEALTH_TIMEOUT_S))
    except urllib.error.HTTPError as exc:
        return False, f"the laptop refused us ({exc.code})"
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"
    return bool(got.get("ok")), ""


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
