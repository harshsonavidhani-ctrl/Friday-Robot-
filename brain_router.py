"""brain_router.py — step 9 §10: which brain answered, and what it cost.

    router = BrainRouter()
    d = await router.decide(DRIVE, "which frontier next? ...")
    d.model      -> "claude-haiku-4-5"
    d.escalated  -> True if Haiku said UNSURE and Sonnet was woken

THE SPLIT, from Harsh's spec: moment-to-moment exploration driving -- frontier
selection, micro-navigation, obstacle checks -- runs on HAIKU. SONNET 5 is woken
only to plan a full run, to make a semantic tagging decision, or when Haiku is
stuck or uncertain. Every decision is logged with the brain that handled it.

WHY THE ESCALATION IS A SENTINEL AND NOT A CONFIDENCE SCORE. A model asked to
rate its own confidence produces a number that reads precise and is not, and
tuning a threshold against it is tuning against noise. Haiku is instead told to
reply with exactly `UNSURE` when it cannot answer from what it was given -- a
decision it either makes or does not. That is testable offline, which the whole
of this file is except for `_call`.

⚠ THE API PATH IS THE ONLY UNTESTED PART, DELIBERATELY. `.env` does not exist on
this Pi, so ANTHROPIC_API_KEY is unset and nothing here has ever reached the
API. Everything else -- routing, escalation, the spend cap, the cost log -- is
pure and covered by scripts/test_brain_router.py. `available()` reports the
difference so a caller can tell "routed but not called" from "answered".

MODEL IDS ARE EXACT AND CARRY NO DATE SUFFIX. config.MODEL_FAST was
"claude-haiku-4-5-20251001" until 2026-09-08, which the API rejects; it had
never been caught because nothing had ever called it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import config

log = logging.getLogger("friday.brain_router")

# The four kinds of decision this robot makes with a model in the loop.
DRIVE = "drive"     # frontier choice, micro-navigation, obstacle checks
PLAN = "plan"       # planning a whole exploration run
TAG = "tag"         # "is that a sofa, and should I remember it?"
STUCK = "stuck"     # Haiku could not answer, or the robot is wedged

# The routing table IS the policy, in one readable place.
ROUTE = {
    DRIVE: config.MODEL_FAST,
    PLAN: config.MODEL_MAIN,
    TAG: config.MODEL_MAIN,
    STUCK: config.MODEL_MAIN,
}

UNSURE = "UNSURE"

_DRIVE_SYSTEM = (
    "You are the fast reflex brain of a small exploring robot. Answer in one "
    "short line. You are given only what the sensors saw. If that is not "
    f"enough to answer confidently, reply with exactly {UNSURE} and nothing "
    "else -- a slower brain will take over, which is cheap and correct. "
    "Guessing is neither."
)


@dataclass
class Decision:
    """One routed decision, whatever happened to it."""
    kind: str
    model: str
    text: str | None = None
    escalated: bool = False
    escalated_from: str | None = None
    skipped: str | None = None          # why no call was made, if none was
    usd: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    @property
    def answered(self) -> bool:
        return self.text is not None and self.skipped is None


def spent_today(path=None) -> float:
    """Dollars already logged today. Never raises; a missing log is 0."""
    path = path or config.COST_LOG
    today = time.strftime("%Y-%m-%d")
    total = 0.0
    try:
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except (ValueError, json.JSONDecodeError):
                    continue        # a torn last line must not hide the rest
                if row.get("date") == today:
                    total += float(row.get("usd") or 0.0)
    except OSError:
        return 0.0
    return total


class BrainRouter:
    """Routes a decision to the cheapest brain that can make it."""

    def __init__(self, cap_usd: float | None = None, cost_log=None) -> None:
        self.cap_usd = (config.DAILY_SPEND_CAP_USD if cap_usd is None
                        else cap_usd)
        self.cost_log = cost_log or config.COST_LOG
        self.decisions: list[Decision] = []

    # -- policy, all of it pure -------------------------------------------

    @staticmethod
    def route(kind: str) -> str:
        """Which model handles this kind of decision."""
        if kind not in ROUTE:
            # An unknown kind goes to the CAPABLE model, not the cheap one.
            # The failure mode of guessing wrong in the other direction is a
            # robot driving on an answer from a model that was never given the
            # right prompt for the job.
            log.warning("unknown decision kind %r -- routing to %s",
                        kind, config.MODEL_MAIN)
            return config.MODEL_MAIN
        return ROUTE[kind]

    @staticmethod
    def is_unsure(text: str | None) -> bool:
        """Did the fast brain decline to answer?

        Exact-ish rather than substring-anywhere: a reply that MENTIONS being
        unsure while still answering is an answer, and escalating it would pay
        Sonnet rates for work already done.
        """
        if text is None:
            return True
        stripped = text.strip().strip(".!").upper()
        return stripped == UNSURE

    def over_cap(self) -> bool:
        return spent_today(self.cost_log) >= self.cap_usd

    def available(self) -> bool:
        """Is there a key to call with at all?"""
        return bool(getattr(config, "ANTHROPIC_API_KEY", None))

    # -- the one impure part ----------------------------------------------

    async def _call(self, model: str, prompt: str, system: str,
                    max_tokens: int) -> tuple[str, Any]:
        """One API call. The ONLY place this module touches the SDK.

        Thinking differs by model and getting it wrong is a 400, not a
        degradation: Sonnet 5 takes {"type": "adaptive"} and REJECTS
        budget_tokens, while Haiku 4.5 has no adaptive mode and no effort
        setting. The fast path asks for no thinking at all, which is the point
        of the fast path.
        """
        import anthropic
        client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if model == config.MODEL_MAIN:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": "low"}
        reply = await client.messages.create(**kwargs)
        text = "".join(b.text for b in reply.content
                       if getattr(b, "type", None) == "text")
        return text, reply.usage

    # -- the thing callers use --------------------------------------------

    async def decide(self, kind: str, prompt: str, system: str | None = None,
                     max_tokens: int = 300) -> Decision:
        """Route, call, escalate if the fast brain declined, and log."""
        model = self.route(kind)
        started = time.monotonic()
        d = Decision(kind=kind, model=model)

        if not self.available():
            d.skipped = ("no ANTHROPIC_API_KEY -- routed but not called. "
                         "Write it to ~/friday/.env and restart the brain.")
            return self._finish(d, started)
        if self.over_cap():
            d.skipped = (f"daily spend cap reached "
                         f"(${spent_today(self.cost_log):.2f} of "
                         f"${self.cap_usd:.2f})")
            return self._finish(d, started)

        use_system = system or (_DRIVE_SYSTEM if kind == DRIVE else "")
        try:
            text, usage = await self._call(model, prompt, use_system,
                                           max_tokens)
        except Exception as exc:                        # noqa: BLE001
            # A brain that cannot be reached must not stop a robot that is
            # driving; the caller decides what to do with an unanswered
            # decision, and every caller here has a non-model fallback.
            d.skipped = f"{type(exc).__name__}: {exc}"
            return self._finish(d, started)

        d.text = text
        self._charge(d, model, usage)

        # ESCALATION. Only the fast path can escalate, and only once: a second
        # hop would be Sonnet asking Sonnet.
        if kind == DRIVE and self.is_unsure(text):
            d.escalated = True
            d.escalated_from = model
            d.model = config.MODEL_MAIN
            if self.over_cap():
                d.skipped = "unsure, and the spend cap blocked the escalation"
                return self._finish(d, started)
            try:
                text2, usage2 = await self._call(
                    config.MODEL_MAIN, prompt, system or "", max_tokens)
                d.text = text2
                self._charge(d, config.MODEL_MAIN, usage2)
            except Exception as exc:                    # noqa: BLE001
                d.skipped = f"escalation failed: {type(exc).__name__}: {exc}"

        return self._finish(d, started)

    # -- bookkeeping -------------------------------------------------------

    def _charge(self, d: Decision, model: str, usage: Any) -> None:
        from dashboard.events import turn_cost_usd
        i = getattr(usage, "input_tokens", 0) or 0
        o = getattr(usage, "output_tokens", 0) or 0
        d.input_tokens += i
        d.output_tokens += o
        d.usd += turn_cost_usd(model, i, o,
                               getattr(usage, "cache_read_input_tokens", 0) or 0,
                               getattr(usage, "cache_creation_input_tokens", 0) or 0)

    def _finish(self, d: Decision, started: float) -> Decision:
        d.seconds = round(time.monotonic() - started, 3)
        self.decisions.append(d)
        self._log(d)
        return d

    def _log(self, d: Decision) -> None:
        """One line per decision, in the same file the brain already uses.

        `brain` and `kind` are what make the split readable afterwards:
            jq -r .brain logs/spend.jsonl | sort | uniq -c
        """
        row = {"date": time.strftime("%Y-%m-%d"), "at": d.at,
               "kind": f"decision:{d.kind}", "brain": d.model,
               "model": d.model, "usd": round(d.usd, 6),
               "input": d.input_tokens, "output": d.output_tokens,
               "escalated": d.escalated, "escalated_from": d.escalated_from,
               "skipped": d.skipped, "seconds": d.seconds}
        try:
            config.LOG_DIR.mkdir(parents=True, exist_ok=True)
            with open(self.cost_log, "a") as handle:
                handle.write(json.dumps(row) + "\n")
        except OSError as exc:
            log.warning("could not write the decision log: %s", exc)

    def split(self) -> dict[str, Any]:
        """How the work divided, for the report and the dashboard."""
        by: dict[str, int] = {}
        usd: dict[str, float] = {}
        for d in self.decisions:
            by[d.model] = by.get(d.model, 0) + 1
            usd[d.model] = round(usd.get(d.model, 0.0) + d.usd, 6)
        return {"decisions": len(self.decisions), "by_brain": by,
                "usd_by_brain": usd,
                "escalations": sum(1 for d in self.decisions if d.escalated),
                "unanswered": sum(1 for d in self.decisions if d.skipped),
                "usd_total": round(sum(d.usd for d in self.decisions), 6)}
