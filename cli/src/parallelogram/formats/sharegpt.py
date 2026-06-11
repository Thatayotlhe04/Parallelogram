"""ShareGPT format: {"conversations": [{"from": ..., "value": ...}, ...]}.

ShareGPT is normalized into the internal OpenAI-chat shape at parse time so the
entire rule pipeline runs unchanged. The on-disk turn key is ``conversations``
(``conversation`` is also accepted), each turn carrying ``from``/``value``. An
optional top-level ``system`` string is prepended as a system turn.

Role mapping is deliberately conservative: known ``from`` values map to canonical
roles, and anything unrecognized is passed through verbatim so the existing
``schema`` rule flags it as an invalid role rather than us silently coercing it.
"""
from __future__ import annotations

import json
from typing import Any, Iterator

from . import ParseResult


# ShareGPT `from` → canonical role. Unknown values are passed through untouched.
_FROM_TO_ROLE = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "chatgpt": "assistant",
    "assistant": "assistant",
    "bard": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "function_call": "tool",
}

# Canonical role → ShareGPT `from`, for writing records back out (--fix output).
_ROLE_TO_FROM = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
    "tool": "tool",
}


def _normalize(record: Any) -> Any:
    """Map a ShareGPT record to internal OpenAI-chat shape.

    Non-dict records, or dicts without a conversations key, are returned
    unchanged so the schema rule reports them instead of us raising.
    """
    if not isinstance(record, dict):
        return record

    turns = record.get("conversations", record.get("conversation"))
    if not isinstance(turns, list):
        return record

    messages: list[dict] = []

    system = record.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})

    for turn in turns:
        if not isinstance(turn, dict):
            # Let schema flag the malformed turn; preserve it as-is.
            messages.append(turn)
            continue
        frm = turn.get("from")
        role = _FROM_TO_ROLE.get(frm, frm)
        messages.append({"role": role, "content": turn.get("value")})

    return {"messages": messages}


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    """Stream-parse a ShareGPT JSONL file, normalizing each record."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                yield ParseResult(
                    line_no=line_no, raw=raw, record=_normalize(record)
                )
            except json.JSONDecodeError as e:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=None,
                    parse_error=f"{e.msg} at column {e.colno}",
                )


def to_sharegpt(record: Any) -> Any:
    """Inverse of :func:`_normalize` — internal shape back to ShareGPT.

    Used to write ``--fix`` output in the same format it was read. Non-dict
    records or records without ``messages`` are returned unchanged.
    """
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return record

    conversations = [
        {
            "from": _ROLE_TO_FROM.get(msg.get("role"), msg.get("role")),
            "value": msg.get("content"),
        }
        for msg in record["messages"]
        if isinstance(msg, dict)
    ]
    return {"conversations": conversations}
