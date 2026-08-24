"""memory.py — what Friday remembers, between turns and between boots.

Three different things, deliberately kept apart:

  * FACTS — things worth keeping ("the kitchen is through the left door",
    "Harsh prefers short answers"). Written only when the brain decides
    something is worth keeping. Never a dump of every transcript.
  * EVENTS — a log of what Friday DID and how it went. This is what makes
    "last time I tried that, it failed" possible.
  * SUMMARIES — old events collapsed into a sentence, so the log can grow
    without the context growing with it.

    memory.remember("the kitchen is through the left door")
    memory.recall("where is the kitchen")      -> ranked facts
    memory.log_event("drove forward", task="move", outcome="failed")
    memory.context_block("go to the kitchen")  -> text for the brain

WHY TF-IDF AND NOT EMBEDDINGS. Real sentence embeddings would recall better —
they match "car" to "vehicle", which this cannot. But this Pi is already
CPU-bound: the face holds a core at ~96%, faster-whisper runs at ~5x realtime
under that load, and openWakeWord wants a fifth of a core continuously. Adding
a transformer to compete for the same cores would make Friday slower at
hearing and answering in order to be marginally better at remembering, which
is the wrong trade on this machine.

scikit-learn is already installed, the corpus is tens to hundreds of facts,
and word+character n-grams handle typos and partial words well at that size.
The retrieval interface is deliberately narrow (_Index) so a real embedding
backend can replace it without touching anything else.

THE DATABASE IS THE SOURCE OF TRUTH. The index is derived and can always be
rebuilt from it, so a corrupt or missing index is never worse than a slow
first query.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Iterable

import config

log = logging.getLogger("friday.memory")

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id          INTEGER PRIMARY KEY,
    text        TEXT NOT NULL,
    kind        TEXT NOT NULL DEFAULT 'fact',
    source      TEXT,
    created_at  REAL NOT NULL,
    last_used   REAL,
    uses        INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY,
    task        TEXT,
    text        TEXT NOT NULL,
    outcome     TEXT,
    meta        TEXT,
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS summaries (
    id           INTEGER PRIMARY KEY,
    text         TEXT NOT NULL,
    covers_to_id INTEGER NOT NULL,
    event_count  INTEGER NOT NULL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_task ON events(task);
CREATE INDEX IF NOT EXISTS events_created ON events(created_at);
"""

OUTCOMES = ("worked", "failed", "partial")


class MemoryError(RuntimeError):
    pass


@dataclass
class Fact:
    id: int
    text: str
    kind: str
    source: str | None
    created_at: float
    score: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "kind": self.kind,
                "source": self.source, "score": round(self.score, 3),
                "age": _ago(self.created_at)}


# --- the database --------------------------------------------------------

_db: sqlite3.Connection | None = None
_lock = threading.RLock()


def db() -> sqlite3.Connection:
    """The connection, opened and migrated once.

    check_same_thread=False because the MCP server runs tools on worker
    threads; every write goes through _lock, so the connection is never used
    concurrently despite that.
    """
    global _db
    if _db is not None:
        return _db
    with _lock:
        if _db is not None:
            return _db
        path = config.MEMORY_DB
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), check_same_thread=False)
        connection.row_factory = sqlite3.Row
        # WAL so a reader never blocks the writer -- the brain reads memory on
        # every turn while tools may be writing to it.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)
        connection.commit()
        _db = connection
        log.info("memory open at %s", path)
        return _db


def close() -> None:
    global _db
    with _lock:
        if _db is not None:
            _db.close()
            _db = None
    _index.clear()


# --- retrieval index -----------------------------------------------------

class _Index:
    """Ranks stored text against a query.

    Narrow on purpose: fit(texts) then search(query). Swapping TF-IDF for real
    embeddings means replacing this class and nothing else.

    ⚠ IT IS DELIBERATELY A HYBRID. Character n-grams alone looked fine on
    relevant queries and were badly wrong on irrelevant ones: "what is the
    capital of Japan" scored 0.405 against "the kitchen is through the left
    door", purely on shared letter sequences. A confident score on an unrelated
    fact is worse than no recall at all, because it puts nonsense in the prompt
    on every turn.

    So word overlap gates the result and characters only refine it. Words carry
    the meaning; characters forgive the typos and the Hindi/English mixing that
    whole-word matching handles badly. With the same example the score drops to
    ~0.10 — under the floor, so nothing is recalled, which is correct.
    """

    def __init__(self) -> None:
        self._word = self._char = None
        self._word_matrix = self._char_matrix = None
        self._ids: list[int] = []
        self._dirty = True

    def clear(self) -> None:
        self._word = self._char = None
        self._word_matrix = self._char_matrix = None
        self._ids = []
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def _fit(self, rows: list[sqlite3.Row]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._ids = [row["id"] for row in rows]
        texts = [row["text"] for row in rows]
        if not texts:
            self._word = self._char = None
            self._word_matrix = self._char_matrix = None
            self._dirty = False
            return
        # STOPWORDS MUST GO. With a corpus this small, IDF barely penalises
        # "the"/"is", so "what is the capital of Japan" matched "the kitchen
        # is through the left door" on nothing but those words. Hindi words
        # are not in the English stop list, so the mixing still works.
        self._word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                     sublinear_tf=True, min_df=1,
                                     lowercase=True, stop_words="english")
        self._word_matrix = self._word.fit_transform(texts)
        self._char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     sublinear_tf=True, min_df=1,
                                     lowercase=True)
        self._char_matrix = self._char.fit_transform(texts)
        self._dirty = False

    def search(self, query: str, rows: list[sqlite3.Row],
               limit: int) -> list[tuple[int, float]]:
        if self._dirty or self._word_matrix is None:
            self._fit(rows)
        if self._word_matrix is None or not query.strip():
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        word = cosine_similarity(self._word.transform([query]),
                                 self._word_matrix)[0]
        char = cosine_similarity(self._char.transform([query]),
                                 self._char_matrix)[0]
        weight = config.MEMORY_WORD_WEIGHT
        scores = weight * word + (1.0 - weight) * char
        ranked = sorted(zip(self._ids, scores), key=lambda p: p[1],
                        reverse=True)
        return [(i, float(s)) for i, s in ranked[:limit]
                if s >= config.MEMORY_MIN_SCORE]


_index = _Index()


# --- facts ---------------------------------------------------------------

def remember(text: str, kind: str = "fact", source: str | None = None) -> dict:
    """Store something worth keeping. Returns what was stored.

    Near-duplicates are UPDATED rather than appended. Without that, being told
    the same thing twice would slowly fill recall with copies of one fact and
    crowd out everything else.
    """
    text = (text or "").strip()
    if not text:
        raise MemoryError("nothing to remember")
    if len(text) > config.MEMORY_MAX_FACT_CHARS:
        text = text[:config.MEMORY_MAX_FACT_CHARS].rsplit(" ", 1)[0] + "…"

    with _lock:
        existing = _near_duplicate(text)
        now = time.time()
        if existing is not None:
            db().execute("UPDATE facts SET text=?, source=?, created_at=? "
                         "WHERE id=?", (text, source, now, existing))
            db().commit()
            _index.mark_dirty()
            log.info("updated fact %d: %r", existing, text[:60])
            return {"id": existing, "text": text, "kind": kind,
                    "updated": True}

        cursor = db().execute(
            "INSERT INTO facts (text, kind, source, created_at) "
            "VALUES (?,?,?,?)", (text, kind, source, now))
        db().commit()
        _index.mark_dirty()
        log.info("remembered %d: %r", cursor.lastrowid, text[:60])
        return {"id": cursor.lastrowid, "text": text, "kind": kind,
                "updated": False}


def _near_duplicate(text: str) -> int | None:
    """The id of a stored fact that says the same thing, if there is one."""
    rows = _fact_rows()
    if not rows:
        return None
    hits = _index.search(text, rows, limit=1)
    if hits and hits[0][1] >= config.MEMORY_DUPLICATE_SCORE:
        return hits[0][0]
    return None


def _fact_rows() -> list[sqlite3.Row]:
    return list(db().execute(
        "SELECT id, text, kind, source, created_at FROM facts "
        "ORDER BY id"))


def recall(query: str, limit: int | None = None) -> list[Fact]:
    """Facts relevant to this query, best first."""
    limit = limit or config.MEMORY_RECALL_LIMIT
    with _lock:
        rows = _fact_rows()
        if not rows:
            return []
        by_id = {row["id"]: row for row in rows}
        hits = _index.search(query, rows, limit)
        found = []
        for fact_id, score in hits:
            row = by_id.get(fact_id)
            if row is None:
                continue
            found.append(Fact(id=row["id"], text=row["text"],
                              kind=row["kind"], source=row["source"],
                              created_at=row["created_at"], score=score))
        if found:
            db().executemany(
                "UPDATE facts SET uses=uses+1, last_used=? WHERE id=?",
                [(time.time(), f.id) for f in found])
            db().commit()
        return found


def forget(fact_id: int) -> bool:
    with _lock:
        cursor = db().execute("DELETE FROM facts WHERE id=?", (fact_id,))
        db().commit()
        _index.mark_dirty()
        return cursor.rowcount > 0


def all_facts(limit: int = 200) -> list[Fact]:
    with _lock:
        rows = db().execute(
            "SELECT id, text, kind, source, created_at FROM facts "
            "ORDER BY id DESC LIMIT ?", (limit,))
        return [Fact(id=r["id"], text=r["text"], kind=r["kind"],
                     source=r["source"], created_at=r["created_at"])
                for r in rows]


# --- events --------------------------------------------------------------

def log_event(text: str, task: str | None = None, outcome: str | None = None,
              meta: dict | None = None) -> dict:
    """Record something Friday did, and how it went.

    outcome should be one of OUTCOMES when it is known. It is what makes
    recall_experience() useful later — an event with no outcome records that
    something happened but teaches nothing.
    """
    text = (text or "").strip()
    if not text:
        raise MemoryError("nothing to log")
    if outcome is not None and outcome not in OUTCOMES:
        raise MemoryError(
            f"outcome must be one of {', '.join(OUTCOMES)} — got {outcome!r}")
    with _lock:
        cursor = db().execute(
            "INSERT INTO events (task, text, outcome, meta, created_at) "
            "VALUES (?,?,?,?,?)",
            (task, text, outcome, json.dumps(meta) if meta else None,
             time.time()))
        db().commit()
        return {"id": cursor.lastrowid, "task": task, "text": text,
                "outcome": outcome}


def recall_experience(task: str, limit: int | None = None) -> list[dict]:
    """What happened last time Friday did something like this.

    Matched on the task label first because that is exact and cheap, then on
    the event text, so "move" finds moves even when the wording differs.
    """
    limit = limit or config.MEMORY_EXPERIENCE_LIMIT
    task = (task or "").strip()
    if not task:
        return []
    with _lock:
        rows = list(db().execute(
            "SELECT id, task, text, outcome, created_at FROM events "
            "WHERE outcome IS NOT NULL AND (task = ? OR text LIKE ?) "
            "ORDER BY created_at DESC LIMIT ?",
            (task, f"%{task}%", limit)))
        return [{"text": r["text"], "task": r["task"],
                 "outcome": r["outcome"], "age": _ago(r["created_at"])}
                for r in rows]


def recent_events(limit: int = 10) -> list[dict]:
    with _lock:
        rows = db().execute(
            "SELECT task, text, outcome, created_at FROM events "
            "ORDER BY id DESC LIMIT ?", (limit,))
        return [{"task": r["task"], "text": r["text"],
                 "outcome": r["outcome"], "age": _ago(r["created_at"])}
                for r in rows]


# --- summarising old events ---------------------------------------------

def summarise_old(keep: int | None = None) -> dict | None:
    """Collapse events older than the most recent `keep` into one line.

    WHY. The event log has to grow forever — it is the record — but the
    brain's context must not. Once there are more events than are useful
    verbatim, the oldest are folded into a counted summary and deleted, so
    what reaches the brain stays a fixed size no matter how long Friday runs.

    Deliberately statistical rather than a language-model summary: it costs
    nothing, it cannot hallucinate, and "9 moves, 7 worked, 2 failed" is more
    useful for deciding what to try next than a paragraph would be.
    """
    keep = keep or config.MEMORY_EVENTS_KEPT
    with _lock:
        total = db().execute("SELECT COUNT(*) c FROM events").fetchone()["c"]
        if total <= keep:
            return None
        cutoff = db().execute(
            "SELECT id FROM events ORDER BY id DESC LIMIT 1 OFFSET ?",
            (keep - 1,)).fetchone()
        if cutoff is None:
            return None
        boundary = cutoff["id"]

        rows = list(db().execute(
            "SELECT task, outcome FROM events WHERE id < ?", (boundary,)))
        if not rows:
            return None

        tallies: dict[str, dict[str, int]] = {}
        for row in rows:
            task = row["task"] or "other"
            bucket = tallies.setdefault(task, {})
            key = row["outcome"] or "unrecorded"
            bucket[key] = bucket.get(key, 0) + 1

        parts = []
        for task, counts in sorted(tallies.items()):
            detail = ", ".join(f"{n} {name}" for name, n in sorted(
                counts.items(), key=lambda p: -p[1]))
            parts.append(f"{task}: {sum(counts.values())} ({detail})")
        text = "earlier — " + "; ".join(parts)

        db().execute(
            "INSERT INTO summaries (text, covers_to_id, event_count, "
            "created_at) VALUES (?,?,?,?)",
            (text, boundary, len(rows), time.time()))
        db().execute("DELETE FROM events WHERE id < ?", (boundary,))
        db().commit()
        log.info("summarised %d old events", len(rows))
        return {"text": text, "collapsed": len(rows)}


def summaries(limit: int = 3) -> list[str]:
    with _lock:
        rows = db().execute(
            "SELECT text FROM summaries ORDER BY id DESC LIMIT ?", (limit,))
        return [r["text"] for r in rows]


# --- what the brain sees -------------------------------------------------

def context_block(query: str, task: str | None = None) -> str:
    """Everything relevant to this turn, as text to prepend to the prompt.

    Returns "" when nothing is relevant — an empty section is worse than no
    section, because it teaches the model that this block is usually noise.
    """
    try:
        facts = recall(query)
        experience = recall_experience(task or query)
        past = summaries(limit=1)
    except Exception as exc:                    # noqa: BLE001
        # Memory is an enhancement. A broken database must never stop Friday
        # from answering.
        log.warning("could not build the memory context: %s", exc)
        return ""

    sections: list[str] = []
    if facts:
        lines = "\n".join(f"- {f.text}" for f in facts)
        sections.append(f"What you remember that may be relevant:\n{lines}")
    if experience:
        lines = "\n".join(
            f"- {e['text']} → {e['outcome']} ({e['age']})" for e in experience)
        sections.append(
            f"How similar things went before (learn from this):\n{lines}")

    # The rolling summary is CONTEXT for the above, not a section in its own
    # right. Attaching it whenever it exists put "earlier — look: 4 worked…"
    # in front of "what is 2 plus 2", which is noise on every unrelated turn
    # and teaches the model to skip the whole block.
    if sections and past:
        sections.append(f"Older activity, summarised: {past[0]}")
    return "\n\n".join(sections)


def stats() -> dict:
    with _lock:
        one = lambda q: db().execute(q).fetchone()[0]   # noqa: E731
        return {
            "facts": one("SELECT COUNT(*) FROM facts"),
            "events": one("SELECT COUNT(*) FROM events"),
            "summaries": one("SELECT COUNT(*) FROM summaries"),
            "database": str(config.MEMORY_DB),
        }


def _ago(when: float | None) -> str:
    if not when:
        return "unknown"
    seconds = max(0, time.time() - when)
    if seconds < 90:
        return "just now"
    if seconds < 5400:
        return f"{int(seconds // 60)} min ago"
    if seconds < 172800:
        return f"{int(seconds // 3600)} h ago"
    return f"{int(seconds // 86400)} days ago"


# --- conversation memory (in RAM, dropped on restart) -------------------

_turns: list[dict] = []


def add_turn(role: str, text: str) -> None:
    _turns.append({"role": role, "text": text, "at": time.time()})
    del _turns[:-config.MEMORY_TURNS_KEPT]


def recent_turns(limit: int = 6) -> list[dict]:
    return _turns[-limit:]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Friday's memory.")
    ap.add_argument("--remember", metavar="TEXT")
    ap.add_argument("--recall", metavar="QUERY")
    ap.add_argument("--event", metavar="TEXT")
    ap.add_argument("--task", metavar="NAME")
    ap.add_argument("--outcome", choices=OUTCOMES)
    ap.add_argument("--context", metavar="QUERY")
    ap.add_argument("--summarise", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")

    if args.remember:
        print(remember(args.remember, source="cli"))
    if args.event:
        print(log_event(args.event, task=args.task, outcome=args.outcome))
    if args.recall:
        found = recall(args.recall)
        if not found:
            print("(nothing relevant)")
        for fact in found:
            print(f"  {fact.score:.3f}  {fact.text}")
    if args.context:
        print(context_block(args.context) or "(no relevant memory)")
    if args.summarise:
        print(summarise_old() or "(nothing old enough to summarise)")
    if args.list:
        for fact in all_facts():
            print(f"  [{fact.id}] {fact.text}")
        for event in recent_events():
            print(f"  ({event['age']}) {event['text']} -> {event['outcome']}")
    if args.stats or not any(vars(args).values()):
        print(json.dumps(stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
