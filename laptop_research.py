"""laptop_research.py — File 20: deep research and notes, done on Harsh's laptop.

    laptop_research.available()          is the laptop agent up? (one quick look)
    laptop_research.start(query, mode)   send it off; returns at once
    laptop_research.fallback_answer(q)   her own Haiku + web search answer
    laptop_research.finished()           done jobs not yet announced
    laptop_research.latest(q)            the newest finished job (for detail)

WHY ASYNCHRONOUS. A real multi-source research run takes a minute or more,
and every MCP tool call has STAGE_TOOL_TIMEOUT_S (25 s). So delegate_to_laptop
checks the laptop is up (LAPTOP_HEALTH_TIMEOUT_S), starts the job on a thread
in the MCP server, and returns at once ("researching it on your laptop"). The
job waits ONE LAPTOP_RESEARCH_TIMEOUT_S -- never retries -- and writes
state/research/<id>.json. ReminderWatch (reminders.py) says the result between
turns, through the same may-she-speak gate as reminders.

FALLBACK (spec step 4). Laptop not up: an "answer" is answered at once from her
own Haiku + web search, with a spoken note that the laptop wasn't available.
Laptop failed or timed out mid-job: the same fallback, announced later. A
"notes" request is NEVER written locally -- notes were asked for on the laptop;
she says plainly that it wasn't available.

MEMORY (step 6): every finished job is remembered as kind "research" (query,
summary, date, notes path) so "what did you find out about X" works later.

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
from typing import Any

import config

log = logging.getLogger("friday.laptop_research")

KIND = "research"


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
    """One HTTP call to the laptop agent. Raises on any failure."""
    s = _settings()
    if s is None:
        raise ConnectionError("the laptop agent is not set up "
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
    """Is the agent up and accepting our token? One quick look, no retry."""
    try:
        got = _request("/health", None, float(config.LAPTOP_HEALTH_TIMEOUT_S))
    except urllib.error.HTTPError as exc:
        return False, f"the laptop agent refused us ({exc.code})"
    except Exception as exc:                        # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:80]}"
    if got.get("busy"):
        return False, "the laptop is busy with another research"
    return bool(got.get("ok")), ""


# --- jobs -------------------------------------------------------------------------

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
    jobs = sorted(_dir().glob("*.json"))
    for old in jobs[:-int(config.RESEARCH_KEEP)]:
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
    """Jobs that ended (done / fallback / failed) and were not said yet."""
    return [j for j in jobs() if j.get("status") in ("done", "fallback", "failed")
            and not j.get("announced_at")]


def mark_announced(job_id: str) -> None:
    for job in jobs():
        if job["id"] == job_id:
            job["announced_at"] = time.time()
            _write(job)


def latest(query: str = "") -> dict | None:
    """The newest finished job, or the newest whose query shares words."""
    done = [j for j in jobs() if j.get("status") in ("done", "fallback")]
    if query.strip():
        want = set(re.findall(r"\w{3,}", query.lower()))
        done = [j for j in done
                if want & set(re.findall(r"\w{3,}", j["query"].lower()))]
    return done[-1] if done else None


def _remember(job: dict) -> None:
    try:
        import memory
        text = (f"Research ({time.strftime('%Y-%m-%d', time.localtime(job['at']))})"
                f" on {job['query']}: {job.get('summary', '').strip()}")
        if job.get("path"):
            text += f" Notes saved on the laptop at {job['path']}."
        memory.remember(text, kind=KIND,
                        source="laptop" if job["status"] == "done" else "fallback")
    except Exception as exc:                        # noqa: BLE001
        log.warning("research not remembered: %s", exc)


def start(query: str, mode: str = "answer") -> dict:
    """Send it to the laptop on a thread. The job record is returned at once."""
    job = {"id": f"{int(time.time() * 1000)}", "query": query.strip(),
           "mode": mode, "status": "running", "at": time.time()}
    _write(job)
    threading.Thread(target=_run, args=(job,), name="laptop-research",
                     daemon=True).start()
    return job


def _run(job: dict) -> None:
    try:
        got = _request("/research", {"query": job["query"], "mode": job["mode"]},
                       float(config.LAPTOP_RESEARCH_TIMEOUT_S))
        if not got.get("ok"):
            raise RuntimeError(got.get("error") or "the laptop said no")
        job.update(status="done", summary=got.get("summary", ""),
                   detail=got.get("detail", ""), path=got.get("path"),
                   sources=got.get("sources") or [],
                   seconds=got.get("seconds"))
    except Exception as exc:                        # noqa: BLE001
        why = f"{type(exc).__name__}: {str(exc)[:120]}"
        log.info("laptop research failed (%s) — one try, no retry", why)
        if job["mode"] == "notes":
            job.update(status="failed", why=why)
        else:
            summary, _ = fallback_answer(job["query"])
            job.update(status="fallback", why=why, summary=summary,
                       detail=summary)
    job["ended"] = time.time()
    # Remembered BEFORE the record says it ended, so the moment she can
    # announce it, "what did you find out about X" already finds it too.
    if job["status"] in ("done", "fallback"):
        _remember(job)
    _write(job)
    log.info("research %s: %s (%s)", job["id"], job["status"], job["query"][:60])


# --- her own answer: Haiku + web search -------------------------------------------

def fallback_answer(query: str) -> tuple[str, str]:
    """(answer, "") or ("", why). Haiku 4.5 + the basic web search tool."""
    try:
        import anthropic

        import brain
        import rate_limit
        import scene_describe
        key = scene_describe._api_key()
        if not key:
            return "", "no API key"
        if brain.spent_today() >= config.DAILY_SPEND_CAP_USD:
            return "", "over the daily spend cap"
        ok, why = rate_limit.allow("research_fallback")
        if not ok:
            return "", why
        client = anthropic.Anthropic(api_key=key,
                                     timeout=float(config.RESEARCH_FALLBACK_TIMEOUT_S))
        messages = [{"role": "user", "content": query}]
        reply = None
        for _ in range(3):
            reply = client.messages.create(
                model=config.MODEL_FAST, max_tokens=600,
                system=("Answer for a home robot to say out loud: search the "
                        "web if needed, then give the answer in at most 3 "
                        "short spoken sentences. No URLs, no lists."),
                tools=[{"type": "web_search_20250305", "name": "web_search",
                        "max_uses": 3}],
                messages=messages)
            brain._record_spend(config.MODEL_FAST, reply.usage,
                                brain._cost_usd(config.MODEL_FAST, reply.usage))
            if reply.stop_reason != "pause_turn":
                break
            messages = [{"role": "user", "content": query},
                        {"role": "assistant", "content": reply.content}]
        rate_limit.used("research_fallback")
        text = "".join(b.text for b in reply.content
                       if getattr(b, "type", "") == "text").strip()
        return (text, "") if text else ("", "no answer came back")
    except Exception as exc:                        # noqa: BLE001
        return "", f"{type(exc).__name__}: {str(exc)[:100]}"
