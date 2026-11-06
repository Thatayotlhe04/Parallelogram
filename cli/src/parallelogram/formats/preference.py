"""Preference (DPO) format: {"prompt"?, "chosen", "rejected"}.

Two shapes, both used by TRL's DPO/ORPO/KTO-style trainers:

1. Explicit prompt: ``prompt`` is a string or a message list; ``chosen`` and
   ``rejected`` are the two completions, each a string or a message list.
2. Implicit prompt: no ``prompt`` key; ``chosen`` and ``rejected`` are full
   conversations whose shared leading messages are the prompt.

Normalized to ``{"messages": prompt + chosen, "rejected": rejected}`` where
``rejected`` holds only the rejected continuation. Rules read both sides
through :mod:`parallelogram.core.conversation`.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from . import ParseResult

_KEYS = ("prompt", "chosen", "rejected")
_OTHER_SHAPES = ("messages", "conversations", "conversation")


def is_preference_record(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and ("chosen" in record or "rejected" in record)
        and not any(key in record for key in _OTHER_SHAPES)
    )


def field_errors(record: dict) -> list[str]:
    errors = []
    for key in ("chosen", "rejected"):
        if key not in record:
            errors.append(f"Missing '{key}' field")
        elif not isinstance(record[key], (str, list)):
            errors.append(f"'{key}' must be a string or list; found {type(record[key]).__name__}")
    if "prompt" in record:
        if not isinstance(record["prompt"], (str, list)):
            errors.append(f"'prompt' must be a string or list; found {type(record['prompt']).__name__}")
    elif any(isinstance(record.get(k), str) for k in ("chosen", "rejected")):
        errors.append("Missing 'prompt' field (string completions need an explicit prompt)")
    return errors


def _as_turns(value: Any, role: str) -> list:
    return [{"role": role, "content": value}] if isinstance(value, str) else list(value)


def _common_prefix(a: list, b: list) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def normalize(record: Any) -> Any:
    if not isinstance(record, dict) or field_errors(record):
        return record
    extras = {k: v for k, v in record.items() if k not in _KEYS}
    if "prompt" in record:
        prompt = _as_turns(record["prompt"], "user")
        chosen = _as_turns(record["chosen"], "assistant")
        rejected = _as_turns(record["rejected"], "assistant")
    else:
        chosen_full, rejected_full = list(record["chosen"]), list(record["rejected"])
        n = _common_prefix(chosen_full, rejected_full)
        # Never let the shared prefix swallow a whole side: if one side is a
        # prefix of the other, keep at least its last turn as the completion.
        n = min(n, len(chosen_full) - 1, len(rejected_full) - 1)
        n = max(n, 0)
        prompt, chosen, rejected = chosen_full[:n], chosen_full[n:], rejected_full[n:]
    return {**extras, "messages": prompt + chosen, "rejected": rejected}


def source_metadata(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    kind = lambda v: "str" if isinstance(v, str) else "list"
    return {
        "implicit": "prompt" not in record,
        "kinds": {k: kind(record.get(k)) for k in _KEYS if k in record},
        "extras": [k for k in record if k not in _KEYS],
    }


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                yield ParseResult(line_no=line_no, raw=raw, record=None,
                                  parse_error=f"{exc.msg} at column {exc.colno}",
                                  source_format="preference")
                continue
            yield ParseResult(line_no=line_no, raw=raw, record=normalize(record),
                              source_format="preference", source_metadata=source_metadata(record))


def _restore(turns: list, kind: str, role: str) -> Any:
    if (kind == "str" and len(turns) == 1 and isinstance(turns[0], dict)
            and turns[0].get("role") == role and isinstance(turns[0].get("content"), str)):
        return turns[0]["content"]
    return turns


def to_preference(record: Any, metadata: dict[str, Any] | None = None) -> Any:
    """Inverse of :func:`normalize` for ``--fix --output``."""
    from ..core.conversation import completion_start

    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return record
    meta = metadata or {}
    messages = record["messages"]
    split = completion_start(messages)
    prompt, chosen = messages[:split], messages[split:]
    rejected = record.get("rejected") or []
    kinds = meta.get("kinds", {})
    if meta.get("implicit"):
        out: dict[str, Any] = {"chosen": prompt + chosen, "rejected": prompt + rejected}
    else:
        out = {
            "prompt": _restore(prompt, kinds.get("prompt", "list"), "user"),
            "chosen": _restore(chosen, kinds.get("chosen", "list"), "assistant"),
            "rejected": _restore(rejected, kinds.get("rejected", "list"), "assistant"),
        }
    for key in meta.get("extras", []):
        if key in record:
            out[key] = record[key]
    return out
