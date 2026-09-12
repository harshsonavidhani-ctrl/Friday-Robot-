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
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Any, Awaitable, Callable

import anthropic

import config

log = logging.getLogger("friday.brain")

# Sentence boundary for the streaming hand-off. Includes the Devanagari danda:
# Harsh mixes Hindi and English and a reply can end either way.
_SENTENCE_END = re.compile(r"(?<=[.!?।])(\s+|$)")

SentenceCallback = Callable[[str], Any]


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
        self.model = model or config.MODEL_MAIN
        self.client = anthropic.AsyncAnthropic(api_key=config.ANTHROPIC_API_KEY)
        self.system = system_prompt or self._load_system_prompt()
        self.messages: list[dict[str, Any]] = []
        self.last_cost_usd = 0.0
        self.last_tools_used: list[str] = []

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
        return [{
            "type": "web_search_20260209",
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

        if session is not None:
            return await self._run(text, session, on_sentence, on_tool)
        async with mcp_session() as own_session:
            return await self._run(text, own_session, on_sentence, on_tool)

    async def _run(self, text: str, session: Any,
                   on_sentence: SentenceCallback | None,
                   on_tool: Callable[[str, dict], Any] | None) -> str:
        listed = await session.list_tools()
        tools = _to_anthropic_tools(listed.tools)
        if config.ENABLE_PROMPT_CACHING and tools:
            # Breakpoint on the LAST tool: tools render before system and
            # messages, so this caches the whole tool block in one go.
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        tools = tools + self._server_tools()

        # Relevant memory rides along with the question rather than going in
        # the system prompt: the system block is cached as a stable prefix, and
        # changing it every turn would throw that cache away on every turn.
        self.messages.append({"role": "user",
                              "content": self._with_memory(text)})
        self.last_cost_usd = 0.0
        self.last_tools_used = []
        final_text = ""

        for _ in range(config.MAX_TOOL_ROUNDS):
            reply, spoken = await self._stream_once(tools, on_sentence)

            self.messages.append({"role": "assistant",
                                  "content": reply.content})

            if reply.stop_reason == "refusal":
                detail = getattr(reply, "stop_details", None)
                return ("I can't help with that one — "
                        f"{getattr(detail, 'category', 'declined')}.")

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
            self.messages.append({"role": "user", "content": results})
        else:
            final_text = (final_text
                          or "I got stuck going back and forth on that one.")

        self._trim_history()
        return final_text.strip()

    async def _stream_once(self, tools: list[dict[str, Any]],
                           on_sentence: SentenceCallback | None):
        """One streamed request. Returns (final_message, text_spoken)."""
        pending = ""
        spoken: list[str] = []

        def flush(force: bool = False) -> None:
            nonlocal pending
            while True:
                match = _SENTENCE_END.search(pending)
                if not match:
                    break
                sentence, pending = (pending[:match.end()].strip(),
                                     pending[match.end():])
                if sentence:
                    spoken.append(sentence)
                    if on_sentence:
                        on_sentence(sentence)
            if force and pending.strip():
                spoken.append(pending.strip())
                if on_sentence:
                    on_sentence(pending.strip())
                pending = ""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": config.MAX_REPLY_TOKENS,
            "system": self._system_blocks(),
            "messages": self.messages,
            "tools": tools,
            # Adaptive is the only on-mode on Sonnet 5; budget_tokens is
            # removed there and returns a 400.
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": config.BRAIN_EFFORT},
        }

        async with self.client.messages.stream(**kwargs) as stream:
            async for event in stream:
                if (event.type == "content_block_delta"
                        and getattr(event.delta, "type", "") == "text_delta"):
                    pending += event.delta.text
                    flush()
            message = await stream.get_final_message()

        flush(force=True)

        usage = message.usage
        cost = _cost_usd(self.model, usage)
        self.last_cost_usd += cost
        _record_spend(self.model, usage, cost)
        log.info("turn: %s in / %s out, cache read %s, $%.4f",
                 usage.input_tokens, usage.output_tokens,
                 getattr(usage, "cache_read_input_tokens", 0), cost)
        return message, " ".join(spoken)

    def _trim_history(self) -> None:
        """Keep the conversation short.

        Friday is a companion, not a research assistant — old turns cost money
        on every request and add nothing to "what's the time". Trimming from
        the FRONT keeps the cached system+tools prefix intact.
        """
        limit = config.BRAIN_HISTORY_MESSAGES
        if len(self.messages) <= limit:
            return
        trimmed = self.messages[-limit:]
        # Never start the history on a tool_result — it would reference a
        # tool_use that has just been trimmed away, and the API rejects that.
        while trimmed and _starts_with_tool_result(trimmed[0]):
            trimmed = trimmed[1:]
        self.messages = trimmed


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
