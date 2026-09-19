#!/usr/bin/env python3
"""test_study.py — File 12a: the notes, the cards, and the intervals.

    ./venv/bin/python scripts/test_study.py

SCRATCH DATABASE, NO NETWORK, NO MODEL. config.STUDY_DB is pointed at a temp
file before study is imported, so this never touches the real corpus — and
FSRS is arithmetic, so the scheduling half needs nothing but the package.

THE TEST HARSH ASKED FOR, in his order: "load a PDF, ask questions about it,
add flashcards, review a few, and confirm review intervals behave sensibly
(short at first, lengthening as recall succeeds)".

⚠ THE REVIEWS ARE GRADED AT THE MOMENT THEY WOULD REALLY HAVE HAPPENED.
FSRS grows stability from the time that ACTUALLY PASSED between reviews, so
five reviews in the same second teach it nothing: the first version of this
test showed 2 days, 2 days, 2 days and looked like a broken scheduler. Each
answer here is given at the card's own due time, which is what a person
does.
"""
from __future__ import annotations

import sys
import tempfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import config                                             # noqa: E402

_TMP = Path(tempfile.mkdtemp(prefix="friday-study-"))
config.STUDY_DB = _TMP / "study.db"

import study                                              # noqa: E402

PASS, FAIL = "\033[32mPASS\033[0m", "\033[31mFAIL\033[0m"
_results: list[bool] = []

NOTES = [
    ("Thermodynamics", [
        "The first law of thermodynamics states that energy cannot be "
        "created or destroyed, only transferred. For a closed system the "
        "change in internal energy equals the heat added minus the work "
        "done by the system.",
        "Entropy measures the number of microscopic arrangements that give "
        "the same macroscopic state. The second law says the entropy of an "
        "isolated system never decreases.",
        "A Carnot engine is the most efficient heat engine possible between "
        "two temperatures. Its efficiency is one minus Tcold over Thot, "
        "with both temperatures in kelvin.",
    ]),
    ("Circuits", [
        "Ohm's law relates voltage, current and resistance: V equals I "
        "times R. Doubling the resistance at a fixed voltage halves the "
        "current through the component.",
        "Kirchhoff's current law says the sum of currents entering a "
        "junction equals the sum leaving it, because charge is conserved.",
        "A capacitor stores energy in an electric field. In an RC circuit "
        "the voltage reaches about sixty three percent of its final value "
        "after one time constant, which is R times C.",
    ]),
]


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append(bool(ok))
    print(f"  {PASS if ok else FAIL}  {name}")
    if detail and not ok:
        print(f"        {detail}")
    return bool(ok)


def write_pdf(path: Path, title: str, paragraphs: list[str]) -> Path:
    """A real PDF, written by hand — no generator library on this Pi.

    Small enough to read: five objects, one uncompressed text stream. It
    exists so the PDF path is tested on an actual PDF rather than on a .txt
    renamed, which would have proved nothing about pypdf.
    """
    lines = [title, ""]
    for para in paragraphs:
        words, row = para.split(), ""
        for word in words:
            if len(row) + len(word) > 78:
                lines.append(row)
                row = word
            else:
                row = f"{row} {word}".strip()
        lines += [row, ""]
    body = "BT /F1 11 Tf 40 780 Td 14 TL\n" + "".join(
        f"({l.replace('(', '[').replace(')', ']')}) Tj T*\n" for l in lines) + "ET"
    stream = body.encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources "
        b"<< /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n"
        + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + obj + b"\nendobj\n"
    start = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n"
            f"{start}\n%%EOF\n").encode()
    path.write_bytes(bytes(out))
    return path


def main() -> int:
    print("1 — LOAD A PDF")
    for title, paragraphs in NOTES:
        pdf = write_pdf(_TMP / f"{title.lower()}.pdf", title, paragraphs)
        stored = study.add_pdf(pdf)
        check(f"{title}: {stored['chars']} characters in "
              f"{stored['chunks']} chunk(s)", stored["chunks"] >= 1)
    check("a PDF with no text in it says so, rather than storing nothing",
          _scan_is_reported())

    print("\n2 — ASK QUESTIONS ABOUT IT")
    for question, want in (
            ("what does the second law say about entropy", "entropy"),
            ("how efficient is a carnot engine", "carnot"),
            ("what is ohm's law", "ohm"),
            ("what happens after one time constant in an RC circuit",
             "time constant")):
        hits = study.search(question, limit=2)
        got = hits[0]["text"].lower() if hits else ""
        check(f"{question!r} -> {hits[0]['title'] if hits else 'nothing'}",
              want in got, f"best passage was {got[:90]!r}")
    check("a question the notes cannot answer returns nothing",
          not study.search("who won the cricket world cup in 1983"),
          "a confident wrong passage is worse than none")

    print("\n3 — ADD FLASHCARDS")
    entropy = study.add_card(
        "What does the second law of thermodynamics say?",
        "The entropy of an isolated system never decreases.",
        topic="thermodynamics")
    ohm = study.add_card("State Ohm's law.", "V = I x R", topic="circuits")
    check("both cards are stored and due now", len(study.due_cards()) == 2)
    check("a card with no answer is refused", _empty_card_refused())

    print("\n4 — REVIEW, AND WATCH THE INTERVAL")
    intervals = []
    when = study.card(entropy["id"])["due"]
    for n in range(5):
        out = study.review(entropy["id"], "good", when=when)
        intervals.append(out["interval_s"])
        print(f"        good #{n + 1}: next in {out['interval']:<12} "
              f"stability {out['stability']}")
        when = out["due_at"]          # answered when it actually comes up
    check("the first interval is minutes, not days", intervals[0] < 3600,
          f"{intervals[0]:.0f}s")
    check("each success lengthens the next interval",
          all(b > a for a, b in zip(intervals, intervals[1:])),
          " -> ".join(study.say_interval(i) for i in intervals))
    check("five good answers reach months, not days",
          intervals[-1] > 60 * 86400, study.say_interval(intervals[-1]))

    print("\n   ...and one he forgets:")
    when = study.card(ohm["id"])["due"]
    good = study.review(ohm["id"], "good", when=when)
    lapse = study.review(ohm["id"], "again", when=good["due_at"])
    print(f"        good:  next in {good['interval']}")
    print(f"        again: next in {lapse['interval']}")
    check("a lapse sends it back to minutes",
          lapse["interval_s"] < good["interval_s"],
          "forgetting has to shorten the interval")
    check("...and is counted as a lapse",
          study.card(ohm["id"])["lapses"] == 1)

    print("\n5 — WHAT IT KNOWS ABOUT ITSELF")
    stats = study.stats()
    print(f"        {stats}")
    check("the reviews are all logged", stats["reviews"] == 7,
          f"{stats['reviews']} logged — the fit later needs every one")
    check("it is NOT claiming personalised parameters",
          stats["personalised"] is False)
    check("...and says how many reviews are still needed",
          stats["reviews_needed_to_fit"] ==
          config.STUDY_MIN_REVIEWS_TO_FIT - 7)
    check("optimise() refuses to fit this little history",
          study.optimise()["fitted"] is False)

    passed = sum(_results)
    print(f"\n{passed} passed, {len(_results) - passed} failed")
    print(f"(scratch database: {config.STUDY_DB})")
    return 0 if passed == len(_results) else 1


def _scan_is_reported() -> bool:
    """A PDF whose pages have no text must raise, not store an empty source."""
    blank = _TMP / "scan.pdf"
    write_pdf(blank, "", [])
    try:
        study.add_pdf(blank)
        return False
    except ValueError as exc:
        return "scan" in str(exc).lower()
    except Exception:                                     # noqa: BLE001
        return False


def _empty_card_refused() -> bool:
    try:
        study.add_card("a question with no answer", "")
        return False
    except ValueError:
        return True


if __name__ == "__main__":
    raise SystemExit(main())
