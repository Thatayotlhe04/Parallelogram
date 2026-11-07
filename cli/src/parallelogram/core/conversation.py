"""Shared views of a normalized record.

Every parser normalizes to ``{"messages": [...]}``. Preference (DPO) records
also carry ``"rejected"``: the rejected continuation only, which follows the
same prompt as the chosen one. Rules that read message content call
:func:`conversations` so they check both sides of a pair without knowing
which on-disk format the record came from.
"""
from __future__ import annotations

from copy import copy
from typing import Any


class ConversationRecord(dict):
    """Normalized chat with an explicit prompt boundary outside user fields.

    A dict attribute avoids reserving (and overwriting) dataset metadata keys.
    Mechanical copies must retain it; it is never serialized to disk.
    """

    def __init__(self, record: dict, prompt_length: int):
        super().__init__(record)
        self.prompt_length = prompt_length


def prompt_end(record: dict) -> int:
    if isinstance(record, ConversationRecord):
        return record.prompt_length
    return completion_start(record["messages"])


def completion_start(messages: list) -> int:
    """Index where the trailing run of assistant messages begins.

    For ``[system, user, assistant]`` this is 2; the prompt is everything
    before it. Returns ``len(messages)`` when the last message isn't from
    the assistant (the roles rule reports that case).
    """
    i = len(messages)
    while i and isinstance(messages[i - 1], dict) and messages[i - 1].get("role") == "assistant":
        i -= 1
    return i


def prompt_messages(record: Any) -> list:
    """The prompt part of a record: messages before the final assistant run."""
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return []
    messages = record["messages"]
    return messages[:prompt_end(record)]


def has_rejected(record: Any) -> bool:
    return isinstance(record, dict) and "rejected" in record


def conversations(record: Any) -> list[tuple[str, list]]:
    """``[(label, messages), ...]`` for every conversation a record trains on.

    The label is ``""`` for the main (chosen) conversation and ``"rejected"``
    for the prompt followed by the rejected continuation. Malformed records
    yield only what can be read; the schema rule reports the rest.
    """
    if not isinstance(record, dict):
        return []
    messages = record.get("messages")
    if not isinstance(messages, list):
        return []
    out: list[tuple[str, list]] = [("", messages)]
    rejected = record.get("rejected")
    if isinstance(rejected, list):
        out.append(("rejected", prompt_messages(record) + rejected))
    return out


def where(label: str) -> str:
    """Prefix for issue messages: '' for the main conversation."""
    return "Rejected response: " if label == "rejected" else ""


def message_slots(record: Any):
    """Yield ``(label, index, message)`` for every message exactly once.

    The prompt is shared by both sides of a preference pair, so it is only
    visited through the main conversation; the rejected continuation's
    messages follow with label ``"rejected"``. Per-message checks (empty
    content, encoding, secrets, PII) use this so nothing is reported twice.
    """
    if not isinstance(record, dict):
        return
    messages = record.get("messages")
    if isinstance(messages, list):
        for i, msg in enumerate(messages):
            yield "", i, msg
    rejected = record.get("rejected")
    if isinstance(rejected, list):
        for i, msg in enumerate(rejected):
            yield "rejected", i, msg


def map_messages(record: Any, fn) -> Any:
    """Return a copy of ``record`` with ``fn(message)`` applied to every
    message dict in ``messages`` and ``rejected``. ``fn`` returns the new
    message, or None to drop it."""
    if not isinstance(record, dict):
        return record
    out = copy(record)
    for key in ("messages", "rejected"):
        msgs = record.get(key)
        if not isinstance(msgs, list):
            continue
        new = []
        prompt_length = 0
        for i, msg in enumerate(msgs):
            if not isinstance(msg, dict):
                new.append(msg)
                if isinstance(record, ConversationRecord) and i < record.prompt_length:
                    prompt_length += 1
                continue
            m = fn(msg)
            if m is not None:
                new.append(m)
                if isinstance(record, ConversationRecord) and i < record.prompt_length:
                    prompt_length += 1
        out[key] = new
        if key == "messages" and isinstance(out, ConversationRecord):
            out.prompt_length = prompt_length
    return out
