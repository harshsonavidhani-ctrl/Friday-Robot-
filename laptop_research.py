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
