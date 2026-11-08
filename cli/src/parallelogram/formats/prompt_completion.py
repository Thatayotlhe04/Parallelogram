"""Normalize prompt/completion pairs without guessing missing training targets."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from . import ParseResult
from .jsonline import PARSE_ERRORS, describe


def is_prompt_completion_record(record: Any) -> bool:
    """Existing chat shapes take precedence, even when their values are malformed."""
    return (
        isinstance(record, dict)
        and ("prompt" in record or "completion" in record)
        and not any(key in record for key in ("messages", "conversations", "conversation", "chosen", "rejected"))
    )


def field_errors(record: dict) -> list[str]:
    """Name bad halves so schema diagnostics refer to the user's input fields."""
    errors = []
    for key in ("prompt", "completion"):
        if key not in record:
            errors.append(f"Missing '{key}' field")
        elif not isinstance(record[key], (str, list)):
            errors.append(f"'{key}' must be a string or list; found {type(record[key]).__name__}")
    return errors


def normalize(record: Any) -> Any:
    """Convert each string half to one turn; malformed halves reach schema intact."""
    if not isinstance(record, dict) or field_errors(record):
        return record
    messages = []
    for key, role in (("prompt", "user"), ("completion", "assistant")):
        value = record[key]
        messages.extend([{"role": role, "content": value}] if isinstance(value, str) else value)
    # Keep other fields (ids, metadata) for the secrets rule and for --fix output.
    extras = {k: v for k, v in record.items() if k not in ("prompt", "completion", "messages", "rejected")}
    return {**extras, "messages": messages}


def source_metadata(record: Any) -> dict[str, Any]:
    """Keep serialization choices outside the canonical messages and rule pipeline."""
    return {
        "string_fields": [key for key in ("prompt", "completion")
                          if isinstance(record, dict) and isinstance(record.get(key), str)]
    }


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    """Stream explicit prompt-completion input with the same JSON diagnostics as chat."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except PARSE_ERRORS as exc:
                yield ParseResult(
                    line_no=line_no, raw=raw, record=None,
                    parse_error=describe(exc),
                    source_format="prompt-completion",
                )
                continue
            yield ParseResult(
                line_no=line_no, raw=raw, record=normalize(record),
                source_format="prompt-completion", source_metadata=source_metadata(record),
            )


def to_prompt_completion(record: Any, metadata: dict[str, Any] | None = None) -> Any:
    """Split off trailing assistant turns and restore the input's string/list choices.

    Without source metadata, emit conversational lists. Only a single matching
    turn can become a string; never flatten or discard messages to force a shape.
    """
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return record
    messages = record["messages"]
    split = len(messages)
    while split and isinstance(messages[split - 1], dict) and messages[split - 1].get("role") == "assistant":
        split -= 1
    result = {"prompt": messages[:split], "completion": messages[split:]}
    extras = {k: v for k, v in record.items() if k not in ("messages", "rejected", "prompt", "completion")}
    string_fields = (metadata or {}).get("string_fields", [])
    for key, role in (("prompt", "user"), ("completion", "assistant")):
        turns = result[key]
        if (key in string_fields and len(turns) == 1
                and isinstance(turns[0], dict) and turns[0].get("role") == role
                and isinstance(turns[0].get("content"), str)):
            result[key] = turns[0]["content"]
    return {**result, **extras}
