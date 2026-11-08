"""Where the secrets and pii rules look.

Credentials leak from any field: ids, metadata, message names, tool-call
arguments, even dict keys. So the secrets rule scans every string in a
record, keys included. A key-shaped string almost never appears by accident,
so this costs no false alarms.

Personal data is different. Structured fields are full of order numbers,
SKUs, UUIDs and ``git@github.com`` remotes that look like phone numbers,
card numbers and emails, and rewriting them corrupts ids and JSON-encoded
arguments. So the pii rule scans the text a model trains on: message content,
including the text parts of multimodal content.
"""
from __future__ import annotations

from copy import copy
from typing import Any, Callable, Iterator

from .conversation import message_slots, where

# Message keys that are structural, never free text a model learns from.
_STRUCTURAL = {"role"}


def iter_strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(path, text)`` for every string inside ``value``, dict keys
    included, depth-first. Iterative, so deep nesting can't exhaust the stack.
    """
    stack: list[tuple[str, Any]] = [(path, value)]
    while stack:
        p, v = stack.pop()
        if isinstance(v, str):
            yield p, v
        elif isinstance(v, dict):
            for k in reversed(list(v)):
                child = f"{p}.{k}" if p else str(k)
                stack.append((child, v[k]))
                if isinstance(k, str):
                    stack.append((f"{p} key" if p else "key", k))
        elif isinstance(v, list):
            for i in reversed(range(len(v))):
                stack.append((f"{p}[{i}]", v[i]))


def scan_targets(record: Any) -> Iterator[tuple[str, str]]:
    """Every string in a normalized record, as ``(label, text)``.

    Labels read naturally in an issue message: ``Message 2`` for content,
    ``Message 0 field 'name'`` for other message keys, ``Field 'id'`` for
    record-level fields.
    """
    if not isinstance(record, dict):
        return
    for label, i, msg in message_slots(record):
        if not isinstance(msg, dict):
            continue
        for key, value in msg.items():
            if isinstance(key, str) and key not in _STRUCTURAL and key != "content":
                yield f"{where(label)}Message {i} field name", key
            if key in _STRUCTURAL:
                continue
            if key == "content" and isinstance(value, str):
                yield f"{where(label)}Message {i}", value
                continue
            for path, text in iter_strings(value, str(key)):
                yield f"{where(label)}Message {i} field '{path}'", text
    for key, value in record.items():
        if key in ("messages", "rejected"):
            continue
        yield "Field name", str(key)
        for path, text in iter_strings(value, str(key)):
            yield f"Field '{path}'", text


def text_targets(record: Any) -> Iterator[tuple[str, str]]:
    """The text a model trains on: message content, string or text parts."""
    if not isinstance(record, dict):
        return
    for label, i, msg in message_slots(record):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            yield f"{where(label)}Message {i}", content
        elif isinstance(content, list):
            for j, part in enumerate(content):
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    yield f"{where(label)}Message {i} part {j}", part["text"]


def _unique(key: Any, taken: dict) -> Any:
    """Two redacted keys in one object both become [REDACTED_SECRET]; number
    the second so neither entry is silently lost."""
    if key not in taken:
        return key
    n = 2
    while f"{key}_{n}" in taken:
        n += 1
    return f"{key}_{n}"


def _map_dict(value: dict, fn: Callable[[str], str], skip: set[str] = frozenset()) -> dict:
    out: dict = {}
    for k, v in value.items():
        if k in skip:
            out[k] = v
            continue
        new_key = fn(k) if isinstance(k, str) and k != "content" else k
        out[_unique(new_key, out)] = _map_strings(v, fn)
    return out


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return _map_dict(value, fn)
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    return value


def map_record_strings(record: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every string ``scan_targets`` yields, keys included.

    Returns None when the record is nested too deeply to rewrite, so the
    fixer drops it rather than writing it out unredacted.
    """
    if not isinstance(record, dict):
        return record
    try:
        out = copy(record)  # keeps a ConversationRecord's prompt boundary
        out.clear()
        for key, value in record.items():
            if key in ("messages", "rejected") and isinstance(value, list):
                out[key] = [_map_dict(m, fn, skip=_STRUCTURAL) if isinstance(m, dict) else m
                            for m in value]
            else:
                new_key = fn(key) if isinstance(key, str) else key
                out[_unique(new_key, out)] = _map_strings(value, fn)
        return out
    except RecursionError:
        return None


def map_record_text(record: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every string ``text_targets`` yields; nothing else changes."""
    if not isinstance(record, dict):
        return record
    out = copy(record)
    for key in ("messages", "rejected"):
        msgs = record.get(key)
        if not isinstance(msgs, list):
            continue
        new = []
        for m in msgs:
            if isinstance(m, dict):
                content = m.get("content")
                if isinstance(content, str):
                    m = {**m, "content": fn(content)}
                elif isinstance(content, list):
                    m = {**m, "content": [
                        {**p, "text": fn(p["text"])} if isinstance(p, dict) and isinstance(p.get("text"), str) else p
                        for p in content]}
            new.append(m)
        out[key] = new
    return out
