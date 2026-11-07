"""Every string a record carries, for rules that must not miss one.

Message content is where training text lives, but credentials and personal
data also turn up in ids, metadata, message names, tool-call arguments and
multimodal content parts. The secrets and pii rules scan and redact all of
them, so a record can't be called clean while a key sits in its ``id``.
"""
from __future__ import annotations

from copy import copy
from typing import Any, Callable, Iterator

from .conversation import message_slots, where

# Message keys that are structural, never free text a model learns from.
_STRUCTURAL = {"role"}


def iter_strings(value: Any, path: str = "") -> Iterator[tuple[str, str]]:
    """Yield ``(path, text)`` for every string inside ``value``, depth-first.

    Iterative, so a deeply nested record can't exhaust the call stack.
    """
    stack: list[tuple[str, Any]] = [(path, value)]
    while stack:
        p, v = stack.pop()
        if isinstance(v, str):
            yield p, v
        elif isinstance(v, dict):
            for k in reversed(list(v)):
                stack.append((f"{p}.{k}" if p else str(k), v[k]))
        elif isinstance(v, list):
            for i in reversed(range(len(v))):
                stack.append((f"{p}[{i}]", v[i]))


def scan_targets(record: Any) -> Iterator[tuple[str, str]]:
    """Yield ``(label, text)`` for every string in a normalized record.

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
        for path, text in iter_strings(value, str(key)):
            yield f"Field '{path}'", text


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    return value


def map_record_strings(record: Any, fn: Callable[[str], str]) -> Any:
    """Apply ``fn`` to every string ``scan_targets`` would yield.

    Returns None when the record is nested too deeply to rewrite, so the
    fixer drops it rather than writing it out unredacted.
    """
    if not isinstance(record, dict):
        return record
    try:
        out = copy(record)  # keeps a ConversationRecord's prompt boundary
        for key, value in record.items():
            if key in ("messages", "rejected") and isinstance(value, list):
                out[key] = [
                    {k: (v if k in _STRUCTURAL else _map_strings(v, fn)) for k, v in m.items()}
                    if isinstance(m, dict) else m
                    for m in value
                ]
            else:
                out[key] = _map_strings(value, fn)
        return out
    except RecursionError:
        return None
