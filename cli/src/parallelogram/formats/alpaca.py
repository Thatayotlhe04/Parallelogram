"""Alpaca format: {"instruction", "input"?, "output", "system"?, "history"?}.

The instruction-tuning shape from Stanford Alpaca, also used by LLaMA-Factory
(which adds ``system`` and ``history``: a list of earlier ``[user, assistant]``
pairs). Normalized to chat messages so every rule runs unchanged:

    [system?] + history turns + user(instruction + "\\n\\n" + input) + assistant(output)

The instruction and input are joined with a blank line, which is how Alpaca
templates present them to the model. On ``--fix`` output the split is
restored from the record's source metadata.
"""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from . import ParseResult

_KEYS = ("instruction", "input", "output", "system", "history")
_OTHER_SHAPES = ("messages", "conversations", "conversation", "prompt",
                 "completion", "chosen", "rejected")


def is_alpaca_record(record: Any) -> bool:
    return (
        isinstance(record, dict)
        and ("instruction" in record or "output" in record)
        and not any(key in record for key in _OTHER_SHAPES)
    )


def field_errors(record: dict) -> list[str]:
    """Name the bad fields so schema diagnostics use the user's own keys."""
    errors = []
    for key in ("instruction", "output"):
        if key not in record:
            errors.append(f"Missing '{key}' field")
        elif not isinstance(record[key], str):
            errors.append(f"'{key}' must be a string; found {type(record[key]).__name__}")
    for key in ("input", "system"):
        if key in record and record[key] is not None and not isinstance(record[key], str):
            errors.append(f"'{key}' must be a string; found {type(record[key]).__name__}")
    history = record.get("history")
    if history is not None:
        if not isinstance(history, list) or not all(
            isinstance(p, (list, tuple)) and len(p) == 2 and all(isinstance(x, str) for x in p)
            for p in history
        ):
            errors.append("'history' must be a list of [user, assistant] string pairs")
    return errors


def _user_text(instruction: str, input_: str | None) -> str:
    return f"{instruction}\n\n{input_}" if input_ else instruction


def normalize(record: Any) -> Any:
    if not isinstance(record, dict) or field_errors(record):
        return record
    messages: list[dict] = []
    if record.get("system"):
        messages.append({"role": "system", "content": record["system"]})
    for user, assistant in record.get("history") or []:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    messages.append({"role": "user", "content": _user_text(record["instruction"], record.get("input"))})
    messages.append({"role": "assistant", "content": record["output"]})
    extras = {k: v for k, v in record.items() if k not in _KEYS}
    return {**extras, "messages": messages}


def source_metadata(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    return {
        "instruction": record.get("instruction"),
        "input": record.get("input"),
        "has_input": "input" in record,
        "has_system": "system" in record,
        "has_history": "history" in record,
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
                                  source_format="alpaca")
                continue
            yield ParseResult(line_no=line_no, raw=raw, record=normalize(record),
                              source_format="alpaca", source_metadata=source_metadata(record))


def _split_user(content: str, metadata: dict[str, Any]) -> tuple[str, str]:
    """Recover (instruction, input) from the joined user turn.

    Fixes may have changed the text (mojibake repaired, the end truncated to
    fit the context window), so try the instruction as read and as the
    encoding fix would rewrite it. When the split can't be recovered, keep
    everything in the instruction rather than guess.
    """
    from ..rules.encoding import fix_text

    instruction = metadata.get("instruction")
    if not metadata.get("input") or not isinstance(instruction, str):
        return content, ""
    for candidate in dict.fromkeys((instruction, fix_text(instruction))):
        if content.startswith(candidate + "\n\n"):
            return candidate, content[len(candidate) + 2:]
    return content, ""


def to_alpaca(record: Any, metadata: dict[str, Any] | None = None) -> Any:
    """Inverse of :func:`normalize` for ``--fix --output``."""
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return record
    meta = metadata or {}
    msgs = [m for m in record["messages"] if isinstance(m, dict)]
    out: dict[str, Any] = {}
    system = None
    if msgs and msgs[0].get("role") == "system":
        system = msgs.pop(0).get("content")
    output = msgs.pop().get("content") if msgs and msgs[-1].get("role") == "assistant" else ""
    user = msgs.pop().get("content") if msgs and msgs[-1].get("role") == "user" else ""
    instruction, input_ = _split_user(user, meta)
    out["instruction"] = instruction
    if meta.get("has_input") or input_:
        out["input"] = input_
    out["output"] = output
    if system is not None or meta.get("has_system"):
        out["system"] = system or ""
    if msgs or meta.get("has_history"):
        out["history"] = [[msgs[i].get("content"), msgs[i + 1].get("content")]
                          for i in range(0, len(msgs) - 1, 2)]
    for key in meta.get("extras", []):
        if key in record:
            out[key] = record[key]
    return out
