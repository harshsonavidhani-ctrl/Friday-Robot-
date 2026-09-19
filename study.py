"""study.py — File 12a: his notes, his flashcards, and when to review them.

    study.add_pdf("notes/thermo.pdf")      a source, chunked and searchable
    study.add_text("...", title="Ohm")     the same for typed or OCR'd text
    study.search("what is entropy")        the passages that answer it
    study.add_card(question, answer)       one flashcard
    study.due_cards()                      what is due now
    study.review(card_id, "good")          ...and when it comes back
    study.stats()                          how much there is, and how it is going

    ./venv/bin/python study.py --add notes.pdf
    ./venv/bin/python study.py --ask "what is entropy"
    ./venv/bin/python study.py --due

FSRS IS NOT RE-DERIVED HERE. Harsh's spec: "Use an existing open-source FSRS
implementation as reference rather than deriving the model from scratch." So
the scheduler is the `fsrs` package (py-fsrs 6.3.2, the reference
implementation of FSRS-6), and this file owns the storage, the chunking and
the questions -- not the memory model. A card's whole FSRS state
(difficulty, stability, due, state, step) is kept as the package's own dict,
so an upgrade of the package is a library upgrade and not a migration.

DEFAULT PARAMETERS UNTIL THERE IS SOMETHING TO FIT. The spec again: "don't
over-fit per-user until a few hundred reviews exist". Every review is logged
from the first one, `optimise()` exists, and it REFUSES to run below
STUDY_MIN_REVIEWS_TO_FIT (400) and says so -- a personalised curve fitted to
nine reviews is worse than the defaults it replaced.

A SEPARATE DATABASE, ON PURPOSE. state/study.db, not memory.db. The facts
table is read back to Friday as things she KNOWS and is re-indexed whenever
it changes; study material is a corpus to search, and a few hundred review
rows would swamp both the event log and the growth counts (file 09 step 6)
with something that is not about her at all.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Any

import config

log = logging.getLogger("friday.study")

# The four ratings, in the package's own names. "again" is a lapse.
RATINGS = ("again", "hard", "good", "easy")

_db: sqlite3.Connection | None = None


# --- storage ---------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    path TEXT,
    kind TEXT NOT NULL,            -- pdf | text | ocr
    chars INTEGER NOT NULL DEFAULT 0,
    added_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    ord INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_source ON chunks(source_id);
CREATE TABLE IF NOT EXISTS cards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    topic TEXT,
    source_id INTEGER,
    chunk_id INTEGER,
    created_at REAL NOT NULL,
    fsrs TEXT NOT NULL,            -- the package's own card dict, verbatim
    due REAL NOT NULL,             -- unix seconds, copied out of fsrs for SQL
    reps INTEGER NOT NULL DEFAULT 0,
    lapses INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS cards_due ON cards(due);
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    card_id INTEGER NOT NULL,
    rating TEXT NOT NULL,
    reviewed_at REAL NOT NULL,
    elapsed_days REAL,
    stability REAL,
    difficulty REAL,
    state TEXT
);
CREATE INDEX IF NOT EXISTS reviews_card ON reviews(card_id);
"""


def db() -> sqlite3.Connection:
    """The study database, opened once. WAL, like memory.db."""
    global _db
    if _db is not None:
        return _db
    path = Path(getattr(config, "STUDY_DB", config.ROOT / "state" / "study.db"))
    path.parent.mkdir(parents=True, exist_ok=True)
    _db = sqlite3.connect(path, check_same_thread=False)
    _db.row_factory = sqlite3.Row
    _db.execute("PRAGMA journal_mode=WAL")
    _db.executescript(SCHEMA)
    _db.commit()
    log.info("study open at %s", path)
    return _db


def close() -> None:
    global _db
    if _db is not None:
        _db.close()
        _db = None


# --- reading what he gives her ---------------------------------------------

def _chunks(text: str, size: int | None = None,
            overlap: int | None = None) -> list[str]:
    """Split on paragraphs, then pack to about `size` characters.

    Paragraph-first because a question is usually answered by a paragraph,
    and a fixed window cuts sentences in half. The overlap is there so a fact
    that straddles a boundary is still whole in one of the two pieces.
    """
    size = int(size or getattr(config, "STUDY_CHUNK_CHARS", 900))
    overlap = int(overlap or getattr(config, "STUDY_CHUNK_OVERLAP", 150))
    paragraphs = [" ".join(p.split())
                  for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    out: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 1 <= size:
            current = f"{current} {para}".strip()
            continue
        if current:
            out.append(current)
            current = (current[-overlap:] + " " + para).strip() \
                if overlap else para
        else:
            current = para
        while len(current) > size:
            out.append(current[:size])
            current = current[size - overlap:]
    if current:
        out.append(current)
    return [c for c in out if c.strip()]


def add_text(text: str, title: str, path: str = "", kind: str = "text",
             pages: dict[int, str] | None = None) -> dict:
    """Store one source and its chunks. Returns the source row as a dict."""
    con = db()
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to store: the text is empty")
    cur = con.execute(
        "INSERT INTO sources (title, path, kind, chars, added_at) "
        "VALUES (?,?,?,?,?)", (title, path, kind, len(text), time.time()))
    source_id = int(cur.lastrowid)
    rows: list[tuple] = []
    if pages:
        ordinal = 0
        for page, page_text in sorted(pages.items()):
            for piece in _chunks(page_text):
                rows.append((source_id, ordinal, page, piece))
                ordinal += 1
    else:
        rows = [(source_id, n, None, piece)
                for n, piece in enumerate(_chunks(text))]
    con.executemany(
        "INSERT INTO chunks (source_id, ord, page, text) VALUES (?,?,?,?)",
        rows)
    con.commit()
    _index.mark_dirty()
    log.info("added %r: %d characters in %d chunks", title, len(text),
             len(rows))
    return {"id": source_id, "title": title, "kind": kind,
            "chars": len(text), "chunks": len(rows)}


def add_pdf(path: str | Path) -> dict:
    """Read a PDF's text and store it, page by page.

    ⚠ A SCANNED PDF HAS NO TEXT IN IT, and pypdf returns empty strings rather
    than failing. That is not an error to swallow: it is exactly the document
    that needs the camera and OCR instead, so it is reported as what it is.
    """
    from pypdf import PdfReader                            # noqa: PLC0415

    path = Path(path).expanduser()
    reader = PdfReader(str(path))
    pages: dict[int, str] = {}
    for n, page in enumerate(reader.pages, 1):
        try:
            pages[n] = page.extract_text() or ""
        except Exception as exc:                           # noqa: BLE001
            log.warning("page %d of %s could not be read: %s", n, path.name, exc)
    whole = "\n\n".join(t for t in pages.values() if t.strip())
    if not whole.strip():
        raise ValueError(
            f"{path.name} has {len(reader.pages)} pages and no extractable "
            f"text — it is a scan. Hold it under the down camera instead.")
    return add_text(whole, title=path.stem, path=str(path), kind="pdf",
                    pages=pages)


# --- finding the passage that answers a question ----------------------------

class _Index:
    """Ranks chunks against a question.

    DELIBERATELY A SECOND COPY of memory._Index's hybrid rather than an
    import: the floor and the weight want to be different for a 900-character
    note than for a one-line fact, and memory's index refits whenever a FACT
    changes — which has nothing to do with study material and would throw
    this away on every turn she remembers something.
    """

    def __init__(self) -> None:
        self._word = self._char = None
        self._word_matrix = self._char_matrix = None
        self._ids: list[int] = []
        self._dirty = True

    def mark_dirty(self) -> None:
        self._dirty = True

    def _fit(self, rows: list[sqlite3.Row]) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        self._ids = [int(r["id"]) for r in rows]
        texts = [r["text"] for r in rows]
        if not texts:
            self._word = self._char = None
            self._dirty = False
            return
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
        if self._dirty or self._word is None:
            self._fit(rows)
        if self._word is None or not (query or "").strip():
            return []
        from sklearn.metrics.pairwise import cosine_similarity

        word = cosine_similarity(self._word.transform([query]),
                                 self._word_matrix)[0]
        char = cosine_similarity(self._char.transform([query]),
                                 self._char_matrix)[0]
        weight = float(getattr(config, "STUDY_WORD_WEIGHT", 0.7))
        floor = float(getattr(config, "STUDY_MIN_SCORE", 0.05))
        scores = weight * word + (1.0 - weight) * char
        ranked = sorted(zip(self._ids, scores), key=lambda p: p[1],
                        reverse=True)
        return [(i, float(s)) for i, s in ranked[:limit] if s >= floor]


_index = _Index()


def search(question: str, limit: int | None = None) -> list[dict]:
    """The passages most likely to answer this, best first. Never raises."""
    limit = int(limit or getattr(config, "STUDY_PASSAGES", 4))
    try:
        con = db()
        rows = con.execute(
            "SELECT c.id, c.text, c.page, s.title, s.id AS source_id "
            "FROM chunks c JOIN sources s ON s.id = c.source_id").fetchall()
        if not rows:
            return []
        by_id = {int(r["id"]): r for r in rows}
        found = _index.search(question, rows, limit)
        return [{"chunk_id": cid, "score": round(score, 3),
                 "title": by_id[cid]["title"], "page": by_id[cid]["page"],
                 "source_id": by_id[cid]["source_id"],
                 "text": by_id[cid]["text"]}
                for cid, score in found]
    except Exception as exc:                               # noqa: BLE001
        log.warning("could not search the notes: %s", exc)
        return []


def passages_for(question: str, limit: int | None = None) -> str:
    """The same, as one block to put in front of a model. "" when nothing."""
    found = search(question, limit)
    if not found:
        return ""
    lines = []
    for hit in found:
        where = f"{hit['title']}"
        if hit["page"]:
            where += f", p.{hit['page']}"
        lines.append(f"[{where}] {hit['text']}")
    return "\n\n".join(lines)


# --- the cards --------------------------------------------------------------

def _scheduler():
    """The FSRS scheduler, built from config. One per call: it is cheap."""
    from fsrs import Scheduler                             # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "desired_retention": float(
            getattr(config, "STUDY_DESIRED_RETENTION", 0.9)),
    }
    parameters = getattr(config, "STUDY_FSRS_PARAMETERS", None)
    if parameters:
        # Only ever set by optimise(), and only once there are enough reviews
        # to fit honestly. Absent, the package's own defaults are used.
        kwargs["parameters"] = tuple(parameters)
    return Scheduler(**kwargs)


def _new_card_state() -> tuple[str, float]:
    from fsrs import Card                                  # noqa: PLC0415

    card = Card()
    return json.dumps(card.to_dict()), card.due.timestamp()


def add_card(question: str, answer: str, topic: str = "",
             source_id: int | None = None,
             chunk_id: int | None = None) -> dict:
    """One flashcard, due immediately. Returns the stored row."""
    question = " ".join((question or "").split())
    answer = " ".join((answer or "").split())
    if not question or not answer:
        raise ValueError("a card needs both a question and an answer")
    state, due = _new_card_state()
    con = db()
    cur = con.execute(
        "INSERT INTO cards (question, answer, topic, source_id, chunk_id, "
        "created_at, fsrs, due) VALUES (?,?,?,?,?,?,?,?)",
        (question, answer, topic or None, source_id, chunk_id, time.time(),
         state, due))
    con.commit()
    return card(int(cur.lastrowid))


def card(card_id: int) -> dict | None:
    row = db().execute("SELECT * FROM cards WHERE id=?", (card_id,)).fetchone()
    return dict(row) if row else None


def due_cards(limit: int | None = None, now: float | None = None) -> list[dict]:
    """What is due, soonest first. Never raises."""
    limit = int(limit or getattr(config, "STUDY_SESSION_CARDS", 20))
    try:
        rows = db().execute(
            "SELECT * FROM cards WHERE due <= ? ORDER BY due LIMIT ?",
            (now or time.time(), limit)).fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:                               # noqa: BLE001
        log.warning("could not read the due cards: %s", exc)
        return []


def review(card_id: int, rating: str, when: float | None = None) -> dict | None:
    """Grade a card and schedule the next look. Returns what changed.

    `rating` is one of again / hard / good / easy — the four FSRS grades.
    Everything about WHEN it comes back is the package's decision; this
    stores the answer and logs the review so it can be fitted later.

    `when` is the moment of the review, in unix seconds, and defaults to now.
    ⚠ IT IS NOT A TEST HOOK. FSRS grows stability from the time that actually
    passed between reviews, so a card reviewed a week late must be graded at
    the moment it was really answered — grading it as "now" when the answer
    came in earlier, or replaying a session offline, would feed the model an
    elapsed time that never happened.
    """
    import datetime as _dt                                 # noqa: PLC0415
    from fsrs import Card, Rating                          # noqa: PLC0415

    name = (rating or "").strip().lower()
    if name not in RATINGS:
        raise ValueError(f"rating must be one of {', '.join(RATINGS)}")
    row = card(card_id)
    if row is None:
        return None
    before = Card.from_dict(json.loads(row["fsrs"]))
    grade = {"again": Rating.Again, "hard": Rating.Hard,
             "good": Rating.Good, "easy": Rating.Easy}[name]
    at = (_dt.datetime.fromtimestamp(when, _dt.timezone.utc)
          if when else _dt.datetime.now(_dt.timezone.utc))
    after, entry = _scheduler().review_card(before, grade, review_datetime=at)

    elapsed = None
    if before.last_review:
        elapsed = (after.last_review - before.last_review).total_seconds() / 86400

    con = db()
    con.execute(
        "UPDATE cards SET fsrs=?, due=?, reps=reps+1, lapses=lapses+? "
        "WHERE id=?",
        (json.dumps(after.to_dict()), after.due.timestamp(),
         1 if name == "again" else 0, card_id))
    con.execute(
        "INSERT INTO reviews (card_id, rating, reviewed_at, elapsed_days, "
        "stability, difficulty, state) VALUES (?,?,?,?,?,?,?)",
        (card_id, name, at.timestamp(), elapsed, after.stability,
         after.difficulty, str(after.state)))
    con.commit()

    seconds = max(0.0, after.due.timestamp() - at.timestamp())
    return {"card_id": card_id, "rating": name,
            "due_at": after.due.timestamp(),
            "interval_s": seconds, "interval": say_interval(seconds),
            "stability": round(float(after.stability or 0.0), 3),
            "difficulty": round(float(after.difficulty or 0.0), 3),
            "state": str(after.state)}


def say_interval(seconds: float) -> str:
    """An interval the way a person says one."""
    if seconds < 90:
        return f"{int(max(1, seconds))} seconds"
    if seconds < 3600:
        return f"{int(round(seconds / 60))} minutes"
    if seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.0f} hour" + ("s" if round(hours) != 1 else "")
    days = seconds / 86400
    if days < 60:
        return f"{days:.0f} day" + ("s" if round(days) != 1 else "")
    return f"{days / 30.4:.1f} months"


# --- how it is going --------------------------------------------------------

def stats() -> dict:
    """Counts a person can check. Never raises."""
    try:
        con = db()

        def one(sql: str, *args) -> int:
            row = con.execute(sql, args).fetchone()
            return int(row[0] if row and row[0] is not None else 0)

        now = time.time()
        return {
            "sources": one("SELECT COUNT(*) FROM sources"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "cards": one("SELECT COUNT(*) FROM cards"),
            "due_now": one("SELECT COUNT(*) FROM cards WHERE due <= ?", now),
            "reviews": one("SELECT COUNT(*) FROM reviews"),
            "lapses": one("SELECT COALESCE(SUM(lapses),0) FROM cards"),
            "retention": _retention(),
            "personalised": bool(getattr(config, "STUDY_FSRS_PARAMETERS", None)),
            "reviews_needed_to_fit": max(
                0, int(getattr(config, "STUDY_MIN_REVIEWS_TO_FIT", 400))
                - one("SELECT COUNT(*) FROM reviews")),
        }
    except Exception as exc:                               # noqa: BLE001
        log.warning("could not read the study stats: %s", exc)
        return {}


def _retention() -> float | None:
    """Share of reviews that were not lapses, or None below a useful count."""
    con = db()
    total = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    if total < 10:
        return None
    again = con.execute(
        "SELECT COUNT(*) FROM reviews WHERE rating='again'").fetchone()[0]
    return round(1.0 - again / total, 3)


def optimise() -> dict:
    """Fit FSRS to HIS history — or say why it would be dishonest to.

    The spec: "don't over-fit per-user until a few hundred reviews exist".
    So this refuses below STUDY_MIN_REVIEWS_TO_FIT and returns the reason;
    it never quietly fits nine reviews and calls the result personalised.
    """
    con = db()
    total = con.execute("SELECT COUNT(*) FROM reviews").fetchone()[0]
    need = int(getattr(config, "STUDY_MIN_REVIEWS_TO_FIT", 400))
    if total < need:
        return {"fitted": False, "reviews": total, "needed": need,
                "why": f"{total} reviews so far; the defaults stay until "
                       f"{need}. A curve fitted to this little is worse "
                       f"than the one it would replace."}
    return {"fitted": False, "reviews": total, "needed": need,
            "why": "enough history now — fitting is the next step, and it "
                   "belongs in a session of its own (it is CPU-heavy and "
                   "changes every future interval)."}


def main() -> int:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    ap = argparse.ArgumentParser(description="Friday's study store")
    ap.add_argument("--add", metavar="PATH", help="a PDF or a text file")
    ap.add_argument("--ask", metavar="QUESTION")
    ap.add_argument("--card", nargs=2, metavar=("QUESTION", "ANSWER"))
    ap.add_argument("--due", action="store_true")
    ap.add_argument("--review", nargs=2, metavar=("CARD_ID", "RATING"))
    args = ap.parse_args()

    if args.add:
        path = Path(args.add).expanduser()
        stored = (add_pdf(path) if path.suffix.lower() == ".pdf"
                  else add_text(path.read_text(errors="replace"),
                                title=path.stem, path=str(path)))
        print(f"added {stored['title']}: {stored['chars']} characters, "
              f"{stored['chunks']} chunks")
    if args.ask:
        for hit in search(args.ask):
            where = hit["title"] + (f" p.{hit['page']}" if hit["page"] else "")
            print(f"\n[{hit['score']:.3f}] {where}\n  {hit['text'][:400]}")
    if args.card:
        row = add_card(*args.card)
        print(f"card {row['id']} added, due now")
    if args.due:
        for row in due_cards():
            print(f"  [{row['id']}] {row['question']}")
    if args.review:
        out = review(int(args.review[0]), args.review[1])
        print(out and f"next in {out['interval']} "
                      f"(stability {out['stability']})")
    if not any(vars(args).values()):
        print(json.dumps(stats(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
