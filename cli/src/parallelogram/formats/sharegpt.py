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
from .jsonline import PARSE_ERRORS, describe


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
    "function_call": "assistant",
    "observation": "tool",
}

# Canonical role → ShareGPT `from`, for writing records back out (--fix output).
_ROLE_TO_FROM = {
    "user": "human",
    "assistant": "gpt",
    "system": "system",
    "tool": "tool",
}


def is_sharegpt_record(record: Any) -> bool:
    """Return true for ShareGPT-style conversation records."""
    if not isinstance(record, dict):
        return False
    turns = record.get("conversations", record.get("conversation"))
    return isinstance(turns, list)


# Keys normalize() turns into messages; everything else is carried through.
_OWN_KEYS = {"conversations", "conversation", "system", "messages", "rejected"}

def _function_value(value: Any) -> Any:
    try:
        fn = json.loads(value) if isinstance(value, str) else [value]
    except PARSE_ERRORS:
        return value  # tool-calls reports malformed functions without echoing values
    if isinstance(fn, dict):
        fn = dict(fn)
        if isinstance(fn.get("arguments"), dict):
            fn["arguments"] = json.dumps(fn["arguments"], ensure_ascii=False)
    return fn


def _tools_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        definitions = json.loads(value)
        if isinstance(definitions, list):
            definitions = [{"type": "function", "function": d}
                           if isinstance(d, dict) and "type" not in d and "function" not in d else d
                           for d in definitions]
        return definitions
    except PARSE_ERRORS:
        return value  # keep malformed definitions for tool-calls


def normalize(record: Any) -> Any:
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

    pending: list[str] = []
    for index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            # Let schema flag the malformed turn; preserve it as-is.
            messages.append(turn)
            continue
        frm = turn.get("from")
        role = _FROM_TO_ROLE.get(frm, frm) if isinstance(frm, str) else frm
        # Per-turn fields (weight, name, ...) ride along like record-level ones.
        extra = {k: v for k, v in turn.items() if k not in ("from", "value", "role", "content")}
        if frm == "function_call":
            fn = _function_value(turn.get("value"))
            call_id = f"sharegpt_call_{index}"
            pending.append(call_id)
            messages.append({**extra, "role": "assistant", "content": None,
                             "tool_calls": [{"id": call_id, "type": "function", "function": fn}]})
        elif frm == "observation":
            messages.append({**extra, "role": "tool", "content": turn.get("value"),
                             "tool_call_id": pending.pop(0) if pending else ""})
        else:
            messages.append({"role": role, "content": turn.get("value"), **extra})

    # Keep record-level fields (ids, sources, metadata) so rules can scan them
    # for leaked keys and --fix writes them back instead of dropping them.
    extras = {k: v for k, v in record.items() if k not in _OWN_KEYS}
    if "tools" in extras:
        extras["tools"] = _tools_value(extras["tools"])
    return {**extras, "messages": messages}


_normalize = normalize


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
                    line_no=line_no,
                    raw=raw,
                    record=normalize(record),
                    source_format="sharegpt",
                    source_metadata=source_metadata(record),
                )
            except PARSE_ERRORS as e:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=None,
                    parse_error=describe(e),
                    source_format="sharegpt",
                )


def source_metadata(record: Any) -> dict:
    """Remember representation choices and JSON strings for lossless output.

    Original strings may only be restored when their normalized value equals
    the repaired value, so redactions can never be undone by this metadata.
    """
    if not isinstance(record, dict):
        return {}
    turns = record.get("conversations", record.get("conversation", []))
    if not isinstance(turns, list):
        return {}
    return {"turn_key": "conversations" if "conversations" in record else "conversation",
            "system": isinstance(record.get("system"), str) and bool(record["system"]),
            "from": [t.get("from") if isinstance(t, dict) else None for t in turns],
            "tool_values": [t.get("value") if isinstance(t, dict) and t.get("from") == "function_call" else None
                            for t in turns],
            "original_tools": record.get("tools"),
            "tools_string": isinstance(record.get("tools"), str),
            "bare_tools": _bare_tools(record.get("tools"))}


def _bare_tools(value: Any) -> bool:
    try:
        items = json.loads(value) if isinstance(value, str) else value
        return isinstance(items, list) and all(isinstance(t, dict) and "type" not in t and "function" not in t for t in items)
    except PARSE_ERRORS:
        return False


def to_sharegpt(record: Any, metadata: dict | None = None) -> Any:
    """Inverse of :func:`_normalize` — internal shape back to ShareGPT.

    Used to write ``--fix`` output in the same format it was read. Non-dict
    records or records without ``messages`` are returned unchanged.
    """
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return record

    metadata = metadata or {}
    extras = {k: v for k, v in record.items() if k not in ("messages", "rejected")}
    messages = record["messages"]
    if metadata.get("system") and messages and messages[0].get("role") == "system":
        extras["system"] = messages[0].get("content")
        messages = messages[1:]
    conversations = []
    original_roles = metadata.get("from", [])
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            conversations.append(msg)
            continue
        original = original_roles[i] if len(original_roles) == len(messages) else None
        role = msg.get("role")
        frm = original if isinstance(original, str) and _FROM_TO_ROLE.get(original, original) == role else _ROLE_TO_FROM.get(role, role)
        extra = {k: v for k, v in msg.items() if k not in ("role", "content")}
        value = msg.get("content")
        if frm == "function_call":
            call = extra.pop("tool_calls")[0]
            fn = call["function"]
            values = metadata.get("tool_values", [])
            original_value = values[i] if len(values) == len(messages) else None
            if isinstance(original_value, str) and _function_value(original_value) == fn:
                value = original_value
            elif isinstance(fn, dict):
                fn = dict(fn)
                try:
                    original_fn = json.loads(original_value) if isinstance(original_value, str) else {}
                    string_args = isinstance(original_fn, dict) and isinstance(original_fn.get("arguments"), str)
                    if not string_args and isinstance(fn.get("arguments"), str):
                        fn["arguments"] = json.loads(fn["arguments"])
                except PARSE_ERRORS:
                    pass  # a disabled tool-calls rule may allow malformed arguments
                value = json.dumps(fn, ensure_ascii=False)
            else:
                value = fn
        elif frm == "observation":
            extra.pop("tool_call_id", None)
        conversations.append({"from": frm, "value": value, **extra})
    if metadata.get("tools_string") and "tools" in extras:
        definitions = extras["tools"]
        original_tools = metadata.get("original_tools")
        if _tools_value(original_tools) == definitions:
            extras["tools"] = original_tools
        else:
            if metadata.get("bare_tools") and isinstance(definitions, list):
                definitions = [d.get("function", d) if isinstance(d, dict) else d for d in definitions]
            extras["tools"] = json.dumps(definitions, ensure_ascii=False)
    return {metadata.get("turn_key", "conversations"): conversations, **extras}
