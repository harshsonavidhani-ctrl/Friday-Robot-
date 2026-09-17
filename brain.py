"""brain.py — Claude, the MCP tool loop, and nothing else.

Takes a transcript, returns spoken text. Everything physical happens through
the friday-robot MCP server; this module never imports motor.py, vision.py or
sensors.py. That rule is what keeps the safety interrupt un-bypassable — see
mcp_server/safety.py. The brain cannot reach a motor except by asking.

WHY A LOCAL TOOL LOOP AND NOT THE MCP CONNECTOR. The Anthropic API can connect
to an MCP server for you (`mcp_servers` + `mcp_toolset`), but that works by
Anthropic's servers dialling OUT to a public URL. Friday's MCP server is bound
to 127.0.0.1 — deliberately, because its tools drive motors and it has no
authentication. Nothing outside this Pi can or should reach it. So the brain is
its own MCP client: it lists the tools locally, hands their schemas to Claude,
and executes the calls itself. Everything stays on the robot.

CACHING. Order on the wire is tools -> system -> messages, and a cache is a
prefix match, so the two stable things (the tool list and personality.md) are
cached and the volatile thing (what Harsh just said) goes after them. The tool
list is sorted by name for exactly this reason: MCP returns tools in whatever
order the registry happened to build them, and an unstable order would
invalidate the cache on every restart.

SPEAKING EARLY. The reply is streamed and sentences are handed to the caller
the moment they complete, so Piper can start on sentence one instead of waiting
for the whole answer. Synthesis is only ~1.1x realtime on this Pi, so that
matters.

THE REPLY OBJECT (file 08, step 1). Every turn ends with {"reply", "emotion",
"memory_note"} -- see reply_contract.py. ask() still returns the spoken words,
so every caller is unchanged; the whole object is on `brain.last_response`.
The "reply" string is spoken AS IT STREAMS, character by character out of the
unfinished JSON, so the object costs no time to the first word.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import anthropic

import config
import reply_contract

log = logging.getLogger("friday.brain")

# Sentence boundary for the streaming hand-off. Includes the Devanagari danda:
# Harsh mixes Hindi and English and a reply can end either way.
_SENTENCE_END = re.compile(r"(?<=[.!?।])(\s+|$)")

SentenceCallback = Callable[[str], Any]

# Compiled on first use from config.BRAIN_FAST_NEVER. Module-level so the list
# is turned into a regex once rather than on every question.
_FAST_NEVER_RE: re.Pattern[str] | None = None


class BrainError(RuntimeError):
    pass


class SpendCapReached(BrainError):
    """The daily budget is spent. Deliberately its own type so the caller can
    say something useful out loud rather than reporting a generic failure."""


# --- spend ---------------------------------------------------------------

def _today() -> str:
    return time.strftime("%Y-%m-%d")


def spent_today() -> float:
    """Dollars spent today, from the rolling log. Never raises."""
    total = 0.0
    try:
        with open(config.COST_LOG) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get("date") == _today():
                    total += float(row.get("usd", 0.0))
    except OSError:
        pass
    return total


def _record_spend(model: str, usage: Any, usd: float) -> None:
    """Append one turn's cost. Logging must never break a turn."""
    try:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(config.COST_LOG, "a") as handle:
            handle.write(json.dumps({
                "date": _today(),
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "model": model,
                "usd": round(usd, 6),
                "input": getattr(usage, "input_tokens", 0),
                "output": getattr(usage, "output_tokens", 0),
                "cache_read": getattr(usage, "cache_read_input_tokens", 0),
                "cache_write": getattr(usage, "cache_creation_input_tokens", 0),
            }) + "\n")
    except OSError as exc:
        log.warning("could not write the spend log: %s", exc)


def _cost_usd(model: str, usage: Any) -> float:
    from dashboard.events import turn_cost_usd
    return turn_cost_usd(
        model,
        getattr(usage, "input_tokens", 0) or 0,
        getattr(usage, "output_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
    )


# --- the MCP side --------------------------------------------------------

def _to_anthropic_tools(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """MCP tool definitions -> Anthropic tool definitions.

    Sorted by name so the cached prefix is byte-identical between runs. MCP
    hands them back in registry order, which is import order, which is not
    guaranteed stable — and an unstable tool list silently destroys the cache.
    """
    tools = []
    for tool in sorted(mcp_tools, key=lambda t: t.name):
        # BOTH SPELLINGS, because the MCP SDK renamed this field. 1.x hands
        # back `inputSchema`, 2.x `input_schema`, and with 2.x installed every
        # brain turn died on AttributeError before a single request was sent --
        # "the brain failed: 'Tool' object has no attribute 'inputSchema'".
        # An empty tool list would have been worse than this crash: she would
        # have answered fluently while unable to do anything.
        schema = (getattr(tool, "input_schema", None)
                  or getattr(tool, "inputSchema", None)
                  or {"type": "object", "properties": {}})
        tools.append({
            "name": tool.name,
            "description": tool.description or tool.name,
            "input_schema": schema,
        })
    return tools


@asynccontextmanager
async def mcp_session():
    """A connected session to the friday-robot server."""
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(config.MCP_HTTP_URL) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            yield session


def _result_text(result: Any) -> str | list[dict[str, Any]]:
    """Flatten an MCP tool result into something Claude can read.

    Images are passed through as image blocks rather than described, because
    look() exists precisely so the model can SEE the room.
    """
    blocks: list[dict[str, Any]] = []
    text_parts: list[str] = []
    for item in result.content:
        kind = getattr(item, "type", None)
        if kind == "text":
            text_parts.append(item.text)
        elif kind == "image":
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": getattr(item, "mimeType", "image/jpeg"),
                    "data": item.data,
                },
            })
    if blocks:
        if text_parts:
            blocks.append({"type": "text", "text": "\n".join(text_parts)})
        return blocks
    return "\n".join(text_parts).strip() or "(no output)"


# --- the brain -----------------------------------------------------------

class Brain:
    """One conversation with Friday.

        brain = Brain()
        reply = await brain.ask("what's the capital of Japan?", on_sentence=say)
    """

    def __init__(self, model: str | None = None,
                 system_prompt: str | None = None) -> None:
        if not config.ANTHROPIC_API_KEY:
            raise BrainError(
                "ANTHROPIC_API_KEY is not set, so the brain cannot run. That "
                "is correct while BUILDING Friday (Claude Code bills the "
                "subscription instead). Export it to actually run Friday.")
        self.model = model or getattr(config, "BRAIN_MODEL", config.MODEL_MAIN)
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.system = system_prompt or self._load_system_prompt()
        # THE CONTRACT GOES AT THE END OF THE SYSTEM PROMPT, which is cached:
        # it is the same bytes every turn, so it costs nothing after the first.
        if getattr(config, "BRAIN_STRUCTURED_REPLY", True):
            self.system = f"{self.system}\n\n{reply_contract.INSTRUCTIONS}"
        self.messages: list[dict[str, Any]] = []
        # The last turn's reply object: {"reply", "emotion", "memory_note"}.
        # emotion and memory_note are None when the model did not give a valid
        # one -- including when it answered in plain words instead.
        self.last_response: dict[str, Any] | None = None
        self.last_cost_usd = 0.0
        self.last_tools_used: list[str] = []
        self.last_model = self.model
        self.last_fast = False
        # THE TOOL LIST, LISTED ONCE. It was fetched from the MCP server on
        # every single turn, and it cannot change while the server is up --
        # the registry is built at import. Cheap (85 ms measured) but it sat
        # in front of the first token on every question, and it is free to
        # drop. Cleared on any MCP failure so a restarted server is picked up.
        self._tools: list[dict[str, Any]] | None = None
        # (index, enriched content) for the question this turn -- see
        # _outgoing. Reset per turn by _run; None means send history as stored.
        # (index, content) for the question this turn -- see _outgoing. The
        # content is a plain string normally, or image+text blocks when she
        # looked before answering (BRAIN_EYES_FIRST).
        self._enrich: tuple[int | None, Any] = (None, None)
        # WHEN EACH MESSAGE WAS ADDED, one entry per entry in self.messages and
        # trimmed with it. Only images need it (see _outgoing and
        # BRAIN_IMAGE_TTL_S), but keeping one stamp per message is simpler than
        # tracking which messages happen to carry a picture.
        self._stamps: list[float] = []

    async def warm(self) -> float:
        """Pay the cold-connection cost before Harsh is waiting on it.

        A COLD DNS LOOKUP COSTS 5.04 s ON THIS PI. Without this it is paid by
        the first question of the evening, which is the one being watched.
        models.list() is a GET that bills no tokens and warms DNS, TLS and the
        connection pool together. Never raises: a robot that cannot reach the
        API must still boot, and the turn itself reports that properly.

        ⚠ USE warm_in_background() FROM main.py, NOT THIS. main.py's wake loop
        is a BLOCKING generator inside an async function, so once it starts the
        event loop never gets control back and an awaitable scheduled here is
        starved forever. That is not theory: as a bare create_task this logged
        "warm in 21.59s" on one start and nothing whatsoever on the next two.
        """
        started = time.monotonic()
        try:
            await self.client.models.list(limit=1)
        except Exception as exc:                # noqa: BLE001
            log.info("could not warm the API connection (%s: %s) — the first "
                     "question will pay for it instead", type(exc).__name__, exc)
            return -1.0
        took = time.monotonic() - started
        log.info("API connection warm in %.2fs", took)
        return took

    def warm_in_background(self) -> "threading.Thread":
        """The same warm-up, on a thread, the way tts and stt are preloaded.

        A THREAD rather than a task because the caller's event loop stops
        running the moment the wake loop starts -- see warm(). This is the same
        pattern main.py already uses for Piper and faster-whisper, for the same
        reason, and it is the reason those two have always worked.

        It uses a SEPARATE sync client, so what it warms is DNS and the TLS
        handshake to api.anthropic.com -- NOT this Brain's connection pool.
        That is the right trade: DNS is 5.04 s of the cost cold and 0.02-0.15 s
        warm, and the pool is worth about half a second. Never raises.
        """
        def go() -> None:
            started = time.monotonic()
            try:
                anthropic.Anthropic(
                    api_key=config.ANTHROPIC_API_KEY).models.list(limit=1)
            except Exception as exc:            # noqa: BLE001
                log.info("could not warm the API connection (%s: %s) — the "
                         "first question will pay for it instead",
                         type(exc).__name__, exc)
                return
            log.info("API connection warm in %.2fs",
                     time.monotonic() - started)

        thread = threading.Thread(target=go, name="brain-warm", daemon=True)
        thread.start()
        return thread

    @staticmethod
    def _load_system_prompt() -> str:
        try:
            return config.SYSTEM_PROMPT_FILE.read_text().strip()
        except OSError as exc:
            raise BrainError(
                f"cannot read the system prompt at "
                f"{config.SYSTEM_PROMPT_FILE}: {exc}") from exc

    def _system_blocks(self) -> list[dict[str, Any]]:
        block: dict[str, Any] = {"type": "text", "text": self.system}
        if config.ENABLE_PROMPT_CACHING:
            block["cache_control"] = {"type": "ephemeral"}
        return [block]

    def _server_tools(self) -> list[dict[str, Any]]:
        """Anthropic-hosted tools. Web search, for facts Friday cannot know."""
        if not config.ENABLE_WEB_SEARCH:
            return []
        # THE VERSION DEPENDS ON THE MODEL, and the wrong one is a 400 on every
        # turn that carries tools. web_search_20260209 (dynamic filtering)
        # needs Sonnet 5 / Sonnet 4.6 / Opus 4.6+; Haiku 4.5, which the brain
        # runs on since 2026-09-16 (config.BRAIN_MODEL), only takes the basic
        # web_search_20250305.
        kind = ("web_search_20260209" if self.model == config.MODEL_MAIN
                else "web_search_20250305")
        return [{
            "type": kind,
            "name": "web_search",
            "max_uses": config.WEB_SEARCH_MAX_USES,
        }]

    @staticmethod
    def _with_memory(text: str) -> str:
        """The question, with anything Friday already knows about it attached.

        This is what makes memory feel like memory rather than a filing
        cabinet: Harsh should not have to ask Friday to look something up, any
        more than a person has to. The recall tool still exists for deliberate
        checks, but the common case is handled here.

        Nothing is attached when nothing is relevant — an empty "here is what
        you remember" section every turn would teach the model to ignore the
        section entirely, and would cost tokens for the privilege.

        Never allowed to fail the turn: memory is an enhancement, and a broken
        database must not stop Friday from answering.
        """
        if not config.MEMORY_IN_CONTEXT:
            return text
        try:
            import memory
            block = memory.context_block(text)
        except Exception as exc:                # noqa: BLE001
            log.warning("memory lookup failed, answering without it: %s", exc)
            return text
        if not block:
            return text
        return f"{block}\n\n---\n\n{text}"

    @staticmethod
    def _with_situation(text: str) -> str:
        """The question, with what she can already see about herself attached.

        Same place and the same reason as the memory block: after the cached
        prefix, never inside it. What it buys is a whole round trip -- "are you
        exploring?" used to cost a tool call and a second request to answer
        "no", and the answer was in a JSON file on disk. See situation.py, and
        note that everything in it carries a maximum age, because stale state
        spoken in the present tense is worse than no state at all.
        """
        if not getattr(config, "SITUATION_IN_CONTEXT", True):
            return text
        try:
            import situation
            block = situation.block()
        except Exception as exc:                # noqa: BLE001
            log.warning("situation lookup failed, answering without it: %s", exc)
            return text
        if not block:
            return text
        return f"{block}\n\n---\n\n{text}"

    # -- the fast path ----------------------------------------------------

    @staticmethod
    def _might_need_the_robot(text: str) -> bool:
        """Could answering this possibly need her body, sensors or memory?

        A local word check, run before any request. WHOLE WORDS, not
        substrings: as a substring list this matched "time" inside "what's 17
        TIMES 23?" and sent a mental-arithmetic question down the slow path,
        which took 14.4 s to say 391. See BRAIN_FAST_NEVER in config.
        """
        global _FAST_NEVER_RE
        if _FAST_NEVER_RE is None:
            words = sorted(config.BRAIN_FAST_NEVER, key=len, reverse=True)
            _FAST_NEVER_RE = re.compile(
                r"\b(?:" + "|".join(re.escape(w) for w in words) + r")\b",
                re.IGNORECASE)
        return bool(_FAST_NEVER_RE.search(text))

    async def _try_fast(self, text: str,
                        on_sentence: SentenceCallback | None) -> str | None:
        """Answer with the small model and no tools, or None to escalate.

        Returns None whenever there is ANY doubt -- the words suggest the
        robot, the history contains tool calls (which cannot be replayed to a
        request that defines no tools), the model said UNSURE, or the call
        failed at all. Every one of those falls through to the normal turn.
        """
        if not getattr(config, "BRAIN_FAST_PATH", True):
            return None
        if self._might_need_the_robot(text):
            return None
        # A history holding tool_use blocks cannot go to a request with no
        # tools defined -- the API rejects it. Once she has used a tool this
        # conversation, the fast path is done until the history trims away.
        if any(_mentions_tools(m) for m in self.messages):
            return None

        model = getattr(config, "BRAIN_FAST_MODEL", config.MODEL_FAST)
        unsure = getattr(config, "BRAIN_FAST_UNSURE", "UNSURE")
        system = (
            f"{self.system}\n\n"
            f"You are answering on the FAST PATH: no tools, no sensors, no "
            f"memory lookup and no web. Answer only what "
            f"you know from general knowledge, from this conversation, and "
            f"from the 'right now' notes attached to the question. If the "
            f"answer needs any of the things you have not got, or needs to "
            f"know anything else about this robot, this room or this house, "
            f"reply with exactly {unsure} and nothing else — a slower brain "
            f"with all of it takes over, which is cheap and correct. Guessing "
            f"is not.\n\n"
            f"Say {unsure} for anything whose answer could have CHANGED since "
            f"you were trained, too — news, prices, scores, results, who holds "
            f"a post, what happened recently. The slower brain can search the "
            f"web; you cannot, and a confidently stale answer said out loud is "
            f"the worst outcome here.\n\n"
            f"If the words were plainly NOT said to you — a line from a video "
            f"or the TV, people talking to each other, or a fragment that "
            f"makes no sense as something said to a robot — reply with exactly "
            f"{getattr(config, 'BRAIN_IGNORE', 'IGNORE')} and nothing else. "
            f"You stay silent. Do not ask them to repeat it.\n\n"
            + (f"{unsure} and {getattr(config, 'BRAIN_IGNORE', 'IGNORE')} "
               f"go in the reply object's `reply` field like any other answer. "
               f"⚠ Every character of `reply` is said out loud: no markdown, no "
               f"emotion names, no stage directions in it — the face goes in "
               f"`emotion`, nowhere else."
               if getattr(config, "BRAIN_STRUCTURED_REPLY", True) else
               f"⚠ REPLY WITH THE SPOKEN WORDS AND NOTHING ELSE. Every character "
               f"you return is sent straight to a speech synthesiser and said "
               f"out loud. No markdown, no backticks, no emotion names, no stage "
               f"directions — the face is handled for you on this path, and "
               f"writing `speaking` or `face: neutral` makes the robot read "
               f"those words aloud.")
        )
        blocks: list[dict[str, Any]] = [{"type": "text", "text": system}]
        if config.ENABLE_PROMPT_CACHING:
            blocks[0]["cache_control"] = {"type": "ephemeral"}

        # THE SITUATION BLOCK RIDES ON THE FAST PATH TOO, and it is what makes
        # the fast path worth having for questions about her rather than about
        # the world. "What time is it", "how are you feeling", "are you
        # exploring" are all answerable from it with no tool and no sensor, so
        # they come back in about a second instead of going to the big model
        # and, before situation.py existed, a tool round trip on top.
        asked = self._with_situation(text)
        pending = self.messages + [{"role": "user", "content": asked}]
        try:
            message = await self.client.messages.create(
                model=model,
                max_tokens=getattr(config, "BRAIN_FAST_MAX_TOKENS", 300),
                system=blocks,
                messages=pending,
            )
        except Exception as exc:                # noqa: BLE001
            # NOT an error worth failing a turn over. The full path is right
            # behind it and will report anything genuinely broken.
            log.info("fast path unavailable (%s: %s) — using %s",
                     type(exc).__name__, exc, self.model)
            return None

        raw = "".join(b.text for b in message.content
                      if getattr(b, "type", None) == "text")
        parsed = reply_contract.parse(raw)
        reply = _spoken_only(parsed.reply)
        cost = _cost_usd(model, message.usage)
        self.last_cost_usd += cost
        _record_spend(model, message.usage, cost)

        if not reply or reply.strip().strip(".!").upper() == unsure.upper():
            log.info("fast path said %s — escalating to %s", unsure, self.model)
            return None

        self.last_model = model
        self.last_fast = True
        self.last_response = {**parsed.as_dict(), "reply": reply}
        if reply.strip(" .!\"'`").upper() == getattr(
                config, "BRAIN_IGNORE", "IGNORE").upper():
            # NOT SAID TO HER. Kept out of the history: a video's line and her
            # silence are not a conversation she should be carrying forward.
            log.info("fast path: not addressed to her — staying silent "
                     "(%s, $%.5f): %r", model, cost, text[:60])
            return reply
        log.info("fast path answered on %s ($%.5f)%s", model, cost,
                 _contract_note(parsed))
        # The CLEAN question goes into the history, not the one with the
        # situation block on it. The block is true for one moment; kept in
        # history it becomes ten turns of contradictory clocks, all claiming
        # the present tense. Every request gets a freshly built one anyway.
        self._remember({"role": "user", "content": text})
        # The RAW object goes in the history, not just its words: she reads her
        # own earlier turns, and a history of plain answers teaches plain ones.
        self._remember({"role": "assistant", "content": raw.strip() or reply})
        self._trim_history()
        for sentence in _sentences(reply):
            if on_sentence:
                on_sentence(sentence)
        return reply

    async def ask(self, text: str,
                  on_sentence: SentenceCallback | None = None,
                  on_tool: Callable[[str, dict], Any] | None = None,
                  session: Any = None) -> str:
        """One turn. Returns the final spoken reply.

        on_sentence is called with each complete sentence as it streams, so
        speech can start before the answer is finished.
        """
        budget = spent_today()
        if budget >= config.DAILY_SPEND_CAP_USD:
            raise SpendCapReached(
                f"today's ${config.DAILY_SPEND_CAP_USD:.2f} cap is already "
                f"spent (${budget:.2f}). Raise DAILY_SPEND_CAP_USD in config "
                f"or wait until tomorrow.")

        self.last_cost_usd = 0.0
        self.last_tools_used = []
        self.last_model = self.model
        self.last_fast = False
        self.last_response = None

        # THE FAST PATH FIRST, and only when it is plainly safe. Measured on
        # this Pi: 0.75 s to the first token instead of 2.61 s, because most of
        # what Harsh says needs no tool and the tool list alone costs half a
        # second. It hands back None on any doubt at all, and then nothing
        # below has changed.
        fast = await self._try_fast(text, on_sentence)
        if fast is not None:
            return fast

        if session is not None:
            return await self._run(text, session, on_sentence, on_tool)
        async with mcp_session() as own_session:
            return await self._run(text, own_session, on_sentence, on_tool)

    @staticmethod
    def _is_about_what_she_sees(text: str) -> bool:
        """Is this plainly a question about what is in front of her right now?

        Narrow on purpose -- see BRAIN_EYES_WORDS. Missing one costs nothing
        (she calls observe herself); matching one wrongly spends 2 s on the
        camera.
        """
        if not getattr(config, "BRAIN_EYES_FIRST", False):
            return False
        low = text.lower()
        return any(phrase in low for phrase in config.BRAIN_EYES_WORDS)

    async def _eyes_first(self, session: Any) -> list[dict[str, Any]] | None:
        """The current view, fetched before the question is even asked.

        Returns image/text blocks to put in front of the question, or None if
        the camera could not be had -- in which case nothing changes and she
        calls observe herself, one round later.

        Never raises. A camera that is busy (the dashboard and an exploration
        run both take it) must degrade to the ordinary path, not fail a turn.
        """
        started = time.monotonic()
        try:
            outcome = await asyncio.wait_for(
                session.call_tool("observe",
                                  {"camera": config.BRAIN_EYES_CAMERA}),
                config.STAGE_TOOL_TIMEOUT_S)
        except Exception as exc:                # noqa: BLE001
            log.info("could not look before answering (%s: %s) — she will "
                     "call observe herself", type(exc).__name__, exc)
            return None
        content = _result_text(outcome)
        if not isinstance(content, list):
            return None                         # no picture in it
        log.info("looked before answering (%.1fs, camera=%s)",
                 time.monotonic() - started, config.BRAIN_EYES_CAMERA)
        self.last_tools_used.append("observe")
        return content

    def _cached_view(self) -> list[dict[str, Any]] | None:
        """The scene watcher's latest frame (scene.py), if fresh enough to use.

        Costs nothing: the frame is already on disk. Measured before it existed,
        a vision question paid ~2 s of camera inside observe() plus a whole
        second round trip to the model. Returns None when there is no frame or
        it is older than SCENE_FRESH_IMAGE_S -- then the camera path runs.
        """
        import base64
        from pathlib import Path
        try:
            meta = json.loads(Path(config.SCENE_JSON).read_text())
            age = time.time() - float(meta["taken_at"])
            if age > float(config.SCENE_FRESH_IMAGE_S) or age < -5:
                return None
            data = base64.b64encode(Path(config.SCENE_IMAGE).read_bytes())
        except Exception:                       # noqa: BLE001
            return None
        named = ", ".join(f"{o.get('label')} {o.get('side', '')}".strip()
                          for o in meta.get("objects") or []) or "nothing"
        log.info("answering from the scene watcher's frame (%.0fs old)", age)
        self.last_tools_used.append("scene")
        return [
            {"type": "image", "source": {"type": "base64",
                                         "media_type": "image/jpeg",
                                         "data": data.decode()}},
            {"type": "text", "text": (
                f"Your up camera, {age:.0f} s ago. The object detector named: "
                f"{named} -- labels can be wrong, so trust the picture.")},
        ]

    async def _tool_list(self, session: Any) -> list[dict[str, Any]]:
        """The tool list, fetched from the MCP server ONCE and kept.

        It cannot change while the server is up -- the registry is built at
        import -- and it was being re-listed on every turn, in front of the
        first token. Cleared on any MCP failure, so a server that restarted
        underneath us is picked up on the next turn rather than cached wrong.
        """
        if self._tools is not None:
            return self._tools
        try:
            listed = await session.list_tools()
        except Exception:                       # noqa: BLE001
            self._tools = None
            raise
        tools = _to_anthropic_tools(listed.tools)
        if config.ENABLE_PROMPT_CACHING and tools:
            # Breakpoint on the LAST tool: tools render before system and
            # messages, so this caches the whole tool block in one go.
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        self._tools = tools + self._server_tools()
        log.info("tool list cached: %d tools", len(self._tools))
        return self._tools

    def _remember(self, message: dict[str, Any]) -> None:
        """Add one message to the history, stamped with the time."""
        self.messages.append(message)
        self._stamps.append(time.time())

    def _outgoing(self) -> list[dict[str, Any]]:
        """The history as it goes on the wire.

        Two things differ from what is stored:

        1. This turn's question gets the memory and situation blocks put back
           on it, so the model sees them for the turn they belong to and
           nowhere else.
        2. ⚠ PHOTOGRAPHS OLDER THAN BRAIN_IMAGE_TTL_S ARE TAKEN OUT, and
           replaced by a line saying how old they were. Left in, they are the
           most recent picture she has, so "what can you see now" is answered
           from a frame that may be minutes stale -- measured doing exactly
           that for two minutes straight. See the log excerpt in config.py.
           Removing the evidence is what makes this reliable; a prompt asking
           her to distrust a picture she can plainly see does not.
        """
        outgoing = list(self.messages)

        ttl = float(getattr(config, "BRAIN_IMAGE_TTL_S", 25.0))
        if ttl > 0:
            now = time.time()
            for i, message in enumerate(outgoing):
                age = now - (self._stamps[i] if i < len(self._stamps) else now)
                if age <= ttl:
                    continue
                fresh = _without_images(message, age)
                if fresh is not None:
                    outgoing[i] = fresh

        index, enriched = getattr(self, "_enrich", (None, None))
        if index is not None and 0 <= index < len(outgoing):
            outgoing[index] = {**outgoing[index], "content": enriched}

        # THE REPLY-OBJECT REMINDER, on the last user message and on the wire
        # only -- the question or the tool results she is about to answer.
        # Stored, it would pile up once per round. See reply_contract.REMINDER.
        if (getattr(config, "BRAIN_STRUCTURED_REPLY", True) and outgoing
                and outgoing[-1].get("role") == "user"):
            last = outgoing[-1]
            content = last.get("content")
            reminder = {"type": "text", "text": reply_contract.REMINDER}
            if isinstance(content, str):
                content = [{"type": "text", "text": content}, reminder]
            elif isinstance(content, list):
                content = list(content) + [reminder]
            outgoing[-1] = {**last, "content": content}
        return outgoing

    async def _run(self, text: str, session: Any,
                   on_sentence: SentenceCallback | None,
                   on_tool: Callable[[str, dict], Any] | None) -> str:
        tools = await self._tool_list(session)

        # Relevant memory and her own live state ride along with the question
        # rather than going in the system prompt: the system block is cached as
        # a stable prefix, and changing it every turn would throw that cache
        # away on every turn.
        # THE CLEAN QUESTION GOES IN THE HISTORY; the enriched one goes on the
        # wire. Both blocks are true for one moment, and stored they pile up --
        # ten turns of "It is 11:09 pm" all written in the present tense, the
        # oldest as confident as the newest. _stream_once swaps the enriched
        # copy in at this index for every round of this turn, and the history
        # keeps only what Harsh actually said.
        asked: Any = self._with_situation(self._with_memory(text))

        # LOOK BEFORE ASKING, when the question is plainly about what is in
        # front of her. Saves a whole round trip and means the frame she
        # answers from is the one taken for THIS question. See BRAIN_EYES_FIRST.
        if self._is_about_what_she_sees(text):
            # The scene watcher's frame first -- already on disk, so free. Only
            # a missing or stale one makes her wait on the camera.
            seen = self._cached_view()
            if seen is None:
                seen = await self._eyes_first(session)
                if seen is not None and on_tool:
                    on_tool("observe", {"camera": config.BRAIN_EYES_CAMERA})
            if seen is not None:
                asked = list(seen) + [{
                    "type": "text",
                    "text": (f"{asked}\n\n[The picture above was taken within "
                             f"the last few seconds. It is your current view — "
                             f"answer from it, and describe what is actually "
                             f"in it.]")}]

        self._enrich = (len(self.messages), asked)
        self._remember({"role": "user", "content": text})
        # NOT reset here. ask() already did it, and the fast path may have
        # spent something before escalating to this -- zeroing it there would
        # hide that from the turn's cost and from the dashboard.
        final_text = ""
        parsed = reply_contract.Reply(reply="")

        for _ in range(config.MAX_TOOL_ROUNDS):
            reply, spoken, parsed = await self._stream_once(tools, on_sentence)

            self._remember({"role": "assistant",
                            "content": reply.content})

            # WEB SEARCHES NEVER COME THROUGH THE TOOL LOOP BELOW. The server
            # runs them and returns the results inline, so without this a search
            # left no trace: the 2026-09-16 01:01 turn SAID "I'll search for
            # current information" and nothing could tell whether it had (398
            # input tokens says it had not).
            for block in reply.content:
                kind = getattr(block, "type", None)
                if kind == "server_tool_use":
                    inputs = block.input if isinstance(block.input, dict) else {}
                    self.last_tools_used.append(block.name)
                    log.info("web search %r", inputs.get("query", ""))
                    if on_tool:
                        on_tool(block.name, {"query": inputs.get("query", "")})
                elif kind == "web_search_tool_result":
                    found = getattr(block, "content", None)
                    # A failed search is NOT an exception: HTTP 200, and the
                    # content is one error object instead of a list of results.
                    if isinstance(found, list):
                        log.info("web search -> %d results", len(found))
                    else:
                        log.warning("web search failed: %s",
                                    getattr(found, "error_code", found))

            if reply.stop_reason == "refusal":
                detail = getattr(reply, "stop_details", None)
                return ("I can't help with that one — "
                        f"{getattr(detail, 'category', 'declined')}.")

            # A WEB SEARCH CAN PAUSE THE TURN. The server runs its own loop for
            # hosted tools and returns pause_turn when it wants another request.
            # The history above already ends on that paused assistant message,
            # so asking again is all it takes -- the API resumes from it, and
            # does not want a "continue" message added. This used to fall
            # through to the break below and end the answer half-said.
            if reply.stop_reason == "pause_turn":
                log.info("web search paused the turn — resuming")
                continue

            if reply.stop_reason != "tool_use":
                final_text = spoken
                break

            results = []
            for block in reply.content:
                if block.type != "tool_use":
                    continue
                self.last_tools_used.append(block.name)
                if on_tool:
                    on_tool(block.name, dict(block.input))
                log.info("tool %s %s", block.name, block.input)
                started = time.monotonic()
                try:
                    # A TOOL CALL GETS A CEILING. Without one a wedged serial
                    # port or a camera that never returns takes the whole turn
                    # down silently -- the command is heard, the tool never
                    # comes back, and Friday simply stops. The timeout comes
                    # back to the model as a tool error below, so it can say
                    # so or try another way.
                    outcome = await asyncio.wait_for(
                        session.call_tool(block.name, dict(block.input)),
                        config.STAGE_TOOL_TIMEOUT_S)
                    content = _result_text(outcome)
                    is_error = False
                    log.info("tool %s -> ok (%.1fs)", block.name,
                             time.monotonic() - started)
                except asyncio.TimeoutError:
                    content = (f"{block.name} did not answer within "
                               f"{config.STAGE_TOOL_TIMEOUT_S:g}s and was "
                               f"abandoned. The robot may be busy or the "
                               f"device may be wedged.")
                    is_error = True
                    log.error("tool %s TIMED OUT after %gs", block.name,
                              config.STAGE_TOOL_TIMEOUT_S)
                except Exception as exc:            # noqa: BLE001
                    # The MCP server already turns tool failures into readable
                    # strings, so reaching here means the SERVER is unwell.
                    # Report it as a tool error so the model can react rather
                    # than the whole turn dying.
                    content = (f"the robot's tool server did not answer: "
                               f"{type(exc).__name__}: {exc}")
                    is_error = True
                    log.error("MCP call %s failed: %s", block.name, exc)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    **({"is_error": True} if is_error else {}),
                })
            # ALL results in ONE user message. Splitting them teaches the
            # model to stop making parallel calls.
            self._remember({"role": "user", "content": results})
        else:
            final_text = (final_text
                          or "I got stuck going back and forth on that one.")

        self._trim_history()
        self.last_response = {**parsed.as_dict(), "reply": final_text.strip()}
        log.info("reply object:%s", _contract_note(parsed) or " complete")
        return final_text.strip()

    async def _stream_once(self, tools: list[dict[str, Any]],
                           on_sentence: SentenceCallback | None):
        """One streamed request. Returns (final_message, text_spoken, Reply).

        Text goes through reply_contract.ReplyStream on its way to Piper, so a
        reply object is spoken as its "reply" string arrives and the JSON
        around it never is. Plain text passes through untouched.
        """
        pending = ""
        spoken: list[str] = []
        contract = reply_contract.ReplyStream()

        def hand_over(sentence: str) -> None:
            """One finished sentence, on its way to Piper.

            Filtered, because everything here is HEARD. A stray `neutral` on
            its own line is a face directive the model could not carry out any
            other way, and speaking it aloud is worse than dropping it.
            """
            say = _spoken_only(sentence)
            if not say:
                return
            spoken.append(say)
            if on_sentence:
                on_sentence(say)

        def flush(force: bool = False) -> None:
            nonlocal pending
            while True:
                match = _SENTENCE_END.search(pending)
                if not match:
                    break
                sentence, pending = (pending[:match.end()].strip(),
                                     pending[match.end():])
                if sentence:
                    hand_over(sentence)
            if force and pending.strip():
                hand_over(pending.strip())
                pending = ""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.MAX_REPLY_TOKENS,
            "system": self._system_blocks(),
            "messages": self._outgoing(),
            "tools": tools,
        }
        # Adaptive is the only on-mode on Sonnet 5; budget_tokens is removed
        # there and returns a 400. Haiku 4.5 is the other way round: it has no
        # adaptive mode and no effort setting, and sending either is a 400 --
        # so they go only to MODEL_MAIN, the same rule brain_router._call uses.
        if self.model == config.MODEL_MAIN:
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": config.BRAIN_EFFORT}

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if event.type == "content_block_start":
                    if getattr(event.content_block, "type", "") == "text":
                        contract.new_block()
                    else:
                        contract.tool_boundary()
                elif (event.type == "content_block_delta"
                        and getattr(event.delta, "type", "") == "text_delta"):
                    pending += contract.feed(event.delta.text)
                    flush()
            message = await stream.get_final_message()

        pending += contract.flush_held()
        flush(force=True)

        usage = message.usage
        cost = _cost_usd(self.model, usage)
        self.last_cost_usd += cost
        _record_spend(self.model, usage, cost)
        log.info("turn: %s in / %s out, cache read %s, $%.4f",
                 usage.input_tokens, usage.output_tokens,
                 getattr(usage, "cache_read_input_tokens", 0), cost)
        return message, " ".join(spoken), contract.finish()

    def _trim_history(self) -> None:
        """Keep the conversation short.

        Friday is a companion, not a research assistant — old turns cost money
        on every request and add nothing to "what's the time". Trimming from
        the FRONT keeps the cached system+tools prefix intact.
        """
        limit = config.BRAIN_HISTORY_MESSAGES
        # The stamps are one-per-message and must be cut in exactly the same
        # places, or _outgoing ages the wrong messages. Padded first in case
        # anything ever appended without going through _remember.
        while len(self._stamps) < len(self.messages):
            self._stamps.insert(0, time.time())
        if len(self.messages) <= limit:
            return
        trimmed = self.messages[-limit:]
        stamps = self._stamps[-limit:]
        # Never start the history on a tool_result — it would reference a
        # tool_use that has just been trimmed away, and the API rejects that.
        while trimmed and _starts_with_tool_result(trimmed[0]):
            trimmed = trimmed[1:]
            stamps = stamps[1:]
        self.messages = trimmed
        self._stamps = stamps


# A line that is nothing but a face directive: `speaking`, `face: neutral`,
# *thinking*, [curious]. Anchored and whole-line on purpose -- a backtick in
# the MIDDLE of a sentence is part of something she is really saying, and this
# must never eat real speech.
_FACE_DIRECTIVE = re.compile(
    r"^\s*[`*_\[\(]*\s*(?:face\s*[:=]\s*)?"
    r"(?:neutral|happy|sad|angry|surprised|listening|thinking|speaking|"
    r"confused|dizzy|success|excited|wink|love|curious|proud|bored|sleepy|"
    r"sleeping|scared)\s*[`*_\]\)]*\s*$",
    re.IGNORECASE)


# Citation markup written INTO the words. A web search answer normally carries
# its citations as structured blocks, but inside the reply object's JSON string
# there is nowhere for them to go, and Haiku typed them out instead -- live,
# 2026-09-17: '(cite index="1-1">Today's gold price ... </cite>'. Piper would
# have read "cite index equals one dash one". Both brackets seen, so both kept.
_CITE_TAG = re.compile(r"[<(]\s*/?\s*cite\b[^>)]*[>)]", re.IGNORECASE)


def _spoken_only(text: str) -> str:
    """The part of a reply that should actually be said out loud.

    EVERYTHING SHE RETURNS IS HEARD, NOT READ. personality.md tells her to set
    her face, and on a path with no tools to do it with, the small model writes
    the name down instead -- observed: "`speaking`\\n\\nThree hundred and
    ninety-one.\\n\\n`neutral`", which Piper would have read out as three
    sentences including the word "speaking". The prompt now forbids it; this is
    the guard that means a slip is silent rather than spoken.

    Conservative by construction: it drops whole lines that are ONLY a
    directive and strips code-span backticks, and leaves every other character
    alone.
    """
    kept = [line for line in text.splitlines()
            if line.strip() and not _FACE_DIRECTIVE.match(line)]
    cleaned = _CITE_TAG.sub("", "\n".join(kept)).replace("`", "").strip()
    # If that removed everything, the reply WAS only stage directions. Say
    # nothing rather than say "neutral" out loud.
    return cleaned


def _contract_note(parsed: "reply_contract.Reply") -> str:
    """A log suffix saying what the reply object lacked, or "" if nothing.

    The point is to see in friday.log how often Haiku keeps the contract --
    a plain-words answer still works, it just carries no face and no note.
    """
    if not getattr(config, "BRAIN_STRUCTURED_REPLY", True):
        return ""
    if not parsed.structured:
        return " — answered in PLAIN WORDS, not the reply object"
    if parsed.emotion is None:
        return " — reply object had no valid emotion"
    return ""


def _sentences(text: str) -> list[str]:
    """Split a finished reply the same way the stream splits a growing one.

    The fast path is not streamed -- a 300-token reply from the small model
    arrives in well under a second, so streaming it buys nothing -- but Piper
    still wants it a sentence at a time, and it must break in the same places
    the streaming path breaks, or the two sound different.
    """
    parts: list[str] = []
    rest = text
    while True:
        match = _SENTENCE_END.search(rest)
        if not match:
            break
        head, rest = rest[:match.end()].strip(), rest[match.end():]
        if head:
            parts.append(head)
    if rest.strip():
        parts.append(rest.strip())
    return parts


def _said_seconds(age: float) -> str:
    if age < 90:
        return f"{age:.0f} seconds ago"
    if age < 3600:
        return f"{age / 60:.0f} minutes ago"
    return f"{age / 3600:.0f} hours ago"


def _without_images(message: dict[str, Any], age: float) -> dict[str, Any] | None:
    """The same message with expired photographs replaced by a note.

    Returns None when there was no image in it, so the caller can keep the
    original object rather than copy every message on every request.

    The note matters as much as the removal: without it the tool_result would
    come back empty and read as "looking failed", which is a different and
    equally wrong thing for her to believe. It says what happened and what to
    do about it.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return None
    changed = False
    rebuilt: list[Any] = []
    for block in content:
        if not isinstance(block, dict):
            rebuilt.append(block)
            continue
        if block.get("type") == "image":
            changed = True
            continue
        if block.get("type") == "tool_result":
            inner = block.get("content")
            if isinstance(inner, list):
                kept = [b for b in inner
                        if not (isinstance(b, dict) and b.get("type") == "image")]
                if len(kept) != len(inner):
                    changed = True
                    kept.append({
                        "type": "text",
                        "text": (f"[the photograph from this look was taken "
                                 f"{_said_seconds(age)} and has been removed — "
                                 f"it is no longer what is in front of you. "
                                 f"Look again before saying what you can see.]")})
                    block = {**block, "content": kept}
        rebuilt.append(block)
    if not changed:
        return None
    return {**message, "content": rebuilt}


def _mentions_tools(message: dict[str, Any]) -> bool:
    """Does this history entry carry a tool_use or tool_result block?

    A request that defines no tools cannot be handed a history containing
    them -- the API rejects it -- so this is what closes the fast path for the
    rest of a conversation in which she has actually done something.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        kind = (block.get("type") if isinstance(block, dict)
                else getattr(block, "type", None))
        # A WEB SEARCH COUNTS. Its blocks are server_tool_use and
        # web_search_tool_result, not tool_use, and a history holding them sent
        # to the no-tools fast path is the same rejected request.
        if kind in ("tool_use", "tool_result", "server_tool_use",
                    "web_search_tool_result"):
            return True
    return False


def _starts_with_tool_result(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    kind = first.get("type") if isinstance(first, dict) else getattr(
        first, "type", None)
    return kind == "tool_result"


async def _main() -> int:
    """Ask Friday one thing from the command line."""
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Ask Friday's brain something.")
    ap.add_argument("question", nargs="*", default=["What is the capital of Japan?"])
    ap.add_argument("--speak", action="store_true",
                    help="say the reply out loud as it streams")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-7s %(name)s | %(message)s")

    say = None
    if args.speak:
        import tts
        tts.preload()
        say = lambda s: tts.speak(s)        # noqa: E731

    try:
        brain = Brain()
    except BrainError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    question = " ".join(args.question)
    print(f"> {question}")
    try:
        reply = await brain.ask(question, on_sentence=say,
                                on_tool=lambda n, a: print(f"  [tool] {n} {a}"))
    except SpendCapReached as exc:
        print(f"spend cap: {exc}", file=sys.stderr)
        return 3
    except anthropic.APIStatusError as exc:
        print(f"API error {exc.status_code}: {exc.message}", file=sys.stderr)
        return 1
    except anthropic.APIConnectionError as exc:
        print(f"network error: {exc}", file=sys.stderr)
        return 1

    print(f"\n{reply}")
    print(f"\n[${brain.last_cost_usd:.4f} this turn, "
          f"${spent_today():.4f} today, cap ${config.DAILY_SPEND_CAP_USD:.2f}]")
    if brain.last_tools_used:
        print(f"[tools: {', '.join(brain.last_tools_used)}]")
    return 0


def main() -> int:
    import asyncio
    return asyncio.run(_main())


if __name__ == "__main__":
    raise SystemExit(main())
