"""File 17's tools: remember for ever, and forget -- only after a yes.

The forget pair is a CONVERSATION, not one call: forget_memory finds and asks,
forget_confirm acts on his answer. Nothing is deleted by forget_memory, and
forget_confirm deletes only what the open request named (see
memory_governance.confirm_delete). The same request is what the dashboard's
bin button makes.
"""

from __future__ import annotations

import memory_governance as mg
from mcp_server.registry import tool


@tool
def remember_forever(fact: str) -> str:
    """Save something Harsh wants kept for good ("always remember", "never
    forget"). It is PROTECTED: self-reflection will never flag it as stale.

    Args:
        fact: one clear sentence, third person, e.g. "Harsh's mother's
            birthday is 12 March".
    """
    try:
        stored = mg.remember(fact, protected=True, source="explicit")
    except ValueError as exc:
        return str(exc)
    return f"I'll never forget that {stored['text'].rstrip('.')}."


@tool
def forget_memory(description: str) -> str:
    """Start forgetting something Harsh asked you to forget. This NEVER deletes:
    it finds the memory and returns a question for him (which one / yes or no).
    Say the question; his answer goes to forget_confirm.

    Args:
        description: what he wants forgotten, in his words ("my sister's
            name", "what I said about the meeting").
    """
    return mg.start_forget(description)


@tool
def forget_confirm(answer: str) -> str:
    """His answer to the forget question: "yes", "no", or which one ("the
    second one", "2"). Deletes only after a yes to one named memory.

    Args:
        answer: his words.
    """
    return mg.answer(answer)
