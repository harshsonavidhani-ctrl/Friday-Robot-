"""people.py — File 12 step 3: what she remembers, grouped by who it is about.

    people.people()          every name her memory mentions, most-said first
    people.about("Sujoo")    what she has learned about that person
    people.pages()           both at once, for the dashboard

Harsh's spec: "Group friday_memory.py entries by a person's name mentioned in
conversation (e.g. a name that comes up in what was said), not by visual face
identification — that part is deferred."

⚠ THIS IS A HEURISTIC AND THE PAGE SHOWS ITS WORKING. There is no person
field in the facts table and no face recognition yet, so a "person" here is a
capitalised word that survives a stop list. That will occasionally be wrong —
a place, a product, a model name — which is why every page lists the FACTS
THEMSELVES rather than a summary: a wrong grouping is then visible in one
glance instead of quietly becoming something she believes about somebody.

WHAT IS DELIBERATELY EXCLUDED, and why each one earned its place:
  * the robot's own cast — Friday, Piper, Claude, Whisper, Haiku, Sonnet:
    they are in half her memories and none of them is a person she knows;
  * places and calendar words — Chhindwara, Monday, September;
  * the technical nouns this project is made of — LiDAR, SLAM, GitHub;
  * anything that only ever appears at the START of a sentence, which is
    where an ordinary word gets a capital letter by grammar rather than by
    being a name.

The last rule is the one that does the real work. "Energy cannot be created"
never becomes a person called Energy, because Energy is never seen anywhere
but first.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import config

log = logging.getLogger("friday.people")

# A name, and the multi-word proper nouns among them: consecutive capitalised
# words are ONE candidate, so "Bharatiya Janata Party" is one entry and not
# three people called Bharatiya, Janata and Party.
_WORD = re.compile(r"\b([A-Z][a-z]{2,15}(?:[- ][A-Z][a-z]{1,15})*)\b")
# ⚠ A QUOTE COUNTS AS A SENTENCE START. Without the quote characters here,
# every idiom she has looked up became a person: "All right" gave a page for
# All, "Come on with it" gave one for Come. Measured on her real memory,
# 2026-09-20.
_SENTENCE_START = re.compile(
    r"(?:^|[.!?]\s+|\n|[\"\u201c\u2018']\s*)\s*([A-Z][a-z]{2,15})\b")

# Never a person in this house. Lowercased for comparison.
NOT_PEOPLE = {
    # her own cast
    "friday", "piper", "claude", "whisper", "haiku", "sonnet", "opus",
    "anthropic", "openwakeword", "vosk", "python", "raspberry", "arduino",
    # places and the calendar
    "chhindwara", "pune", "india", "delhi", "tokyo", "japan", "bombay",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "june", "july",
    "august", "september", "october", "november", "december",
    # the project's own nouns
    "lidar", "slam", "github", "wifi", "usb", "pdf", "fsrs", "mcp",
    "kelvin", "celsius", "ohm", "carnot", "kirchhoff",
    # sentence furniture that survives the start-of-sentence rule
    "the", "this", "that", "there", "then", "they", "his", "her", "him",
    "she", "who", "what", "when", "where", "why", "how", "and", "but",
    "for", "not", "was", "were", "has", "have", "had", "can", "will",
    "yes", "okay", "right", "every", "some", "any", "one", "two",
    # organisations read like names and are not people
    "party", "bjp", "hospital", "college", "university", "school",
    "company", "limited", "institute",
}


def _candidates(text: str) -> set[str]:
    """Capitalised words in this text that are not at a sentence start.

    A word seen ONLY where grammar would capitalise it anyway is not
    evidence of a name; the same word seen mid-sentence elsewhere is.
    """
    starts = set(_SENTENCE_START.findall(text or ""))
    out = set()
    for phrase in _WORD.findall(text or ""):
        words = re.split(r"[- ]", phrase)
        # A multi-word phrase survives if ANY of its words is not merely a
        # sentence opener; a single word must not be one.
        if all(w in starts for w in words):
            continue
        if any(w.lower() in NOT_PEOPLE for w in words):
            continue
        out.add(phrase)
    return out


def _rows() -> list[dict]:
    """Every fact, as plain dicts. Never raises."""
    try:
        import memory                                     # noqa: PLC0415
        return [{"id": f.id, "text": f.text, "kind": f.kind,
                 "source": f.source, "created_at": f.created_at}
                for f in memory.all_facts(
                    limit=int(getattr(config, "PEOPLE_MAX_FACTS", 500)))]
    except Exception as exc:                              # noqa: BLE001
        log.warning("could not read the facts: %s", exc)
        return []


_cache: tuple[float, list] = (0.0, [])


def pages(max_age_s: float | None = None) -> list[dict]:
    """One page per named person: the name, and every fact that mentions it.

    Sorted by how much she knows about them. A name mentioned once is still
    a page -- that is how a new person starts -- but PEOPLE_MIN_MENTIONS can
    raise the bar if the list ever gets noisy.
    """
    global _cache
    ttl = float(getattr(config, "PEOPLE_CACHE_S", 20.0)
                if max_age_s is None else max_age_s)
    when, cached = _cache
    if cached and time.time() - when <= ttl:
        return cached
    facts = _rows()
    found: dict[str, dict[str, Any]] = {}
    for fact in facts:
        for name in _candidates(fact["text"]):
            page = found.setdefault(name, {"name": name, "facts": [],
                                           "last": 0.0})
            page["facts"].append(fact)
            page["last"] = max(page["last"], float(fact["created_at"] or 0))
    floor = int(getattr(config, "PEOPLE_MIN_MENTIONS", 1))
    out = [p for p in found.values() if len(p["facts"]) >= floor]
    for page in out:
        page["count"] = len(page["facts"])
        page["last_said"] = (time.strftime("%Y-%m-%d",
                                           time.localtime(page["last"]))
                             if page["last"] else "")
        page["facts"].sort(key=lambda f: f["created_at"] or 0, reverse=True)
    out.sort(key=lambda p: (p["count"], p["last"]), reverse=True)
    out = out[:int(getattr(config, "PEOPLE_MAX_PAGES", 12))]
    _cache = (time.time(), out)
    return out


def people() -> list[dict]:
    """Just the names and their counts — the index of the pages."""
    return [{"name": p["name"], "count": p["count"],
             "last_said": p["last_said"]} for p in pages()]


def about(name: str) -> dict | None:
    """One person's page, by name. Case-insensitive."""
    name = (name or "").strip().lower()
    for page in pages():
        if page["name"].lower() == name:
            return page
    return None


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    found = pages()
    if not found:
        print("nobody is named in her memory yet")
        return 0
    for page in found:
        print(f"\n{page['name']}  ({page['count']} "
              f"mention{'s' if page['count'] != 1 else ''}, "
              f"last {page['last_said']})")
        for fact in page["facts"][:6]:
            print(f"  - {fact['text'][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
