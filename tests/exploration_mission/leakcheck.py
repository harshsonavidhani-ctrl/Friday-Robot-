"""Did a test leave a file in the REAL state/exploration_mission/?

Since 2026-10-02 the mission runs for real, so that folder is no longer empty
and the live service keeps writing to it while the tests run (its heartbeat,
the conversation clock, its checkpoint in the real environment's folder). A
test therefore cannot ask "is the folder empty"; it asks "did anything NEW
appear that the live service would not have written":

    before = leakcheck.snapshot()
    ... the test ...
    check("no leak", not leakcheck.leaked(before))
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / "state" / "exploration_mission"
# What the live service and friday-brain write by themselves.
LIVE = {"heard.json", "watch_alive.json", "live.json", "hold.json", "asking.json",
        "return.json"}


def snapshot() -> set[str]:
    if not FOLDER.exists():
        return set()
    return {str(p.relative_to(FOLDER)) for p in FOLDER.rglob("*") if p.is_file()}


def leaked(before: set[str]) -> list[str]:
    envs_before = {p.split("/")[1] for p in before if p.startswith("envs/")
                   and p.count("/") >= 2}
    out = []
    for rel in sorted(snapshot() - before):
        name = rel.rsplit("/", 1)[-1]
        if name in LIVE or name.startswith("."):
            continue
        parts = rel.split("/")
        if parts[0] == "envs" and len(parts) > 2 and parts[1] in envs_before:
            continue                    # the real environment's own files
        out.append(rel)
    return out
