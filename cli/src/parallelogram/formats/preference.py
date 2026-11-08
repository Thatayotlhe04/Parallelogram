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
from ..core.conversation import SOURCE_MESSAGES, ConversationRecord, prompt_end
from .jsonline import PARSE_ERRORS, describe

_KEYS = ("prompt", "chosen", "rejected")
_OTHER_SHAPES = ("messages", "conversations", "conversation")
_STASH = SOURCE_MESSAGES


def _holds_turns(value: Any) -> bool:
    return isinstance(value, list) and any(
        isinstance(t, dict) and {"from", "value", "role", "content"} & t.keys() for t in value)


def metadata_conversation(record: Any) -> bool:
    """True for a pair whose ``conversation(s)`` column is metadata, not ShareGPT turns.

    argilla/distilabel-capybara-dpo-7k-binarized keeps the source Capybara
    dialogue as ``conversation: [{"input", "output"}, ...]`` beside full
    chat-format ``chosen``/``rejected`` lists. Read as ShareGPT, every row
    fails schema; the pair is the training data.
    """
    if not (isinstance(record, dict) and isinstance(record.get("chosen"), list)
            and isinstance(record.get("rejected"), list)):
        return False
    other = record.get("conversations", record.get("conversation"))
    return (isinstance(other, list) and bool(other) and all(isinstance(t, dict) for t in other)
            and not _holds_turns(other) and _holds_turns(record["chosen"]))


def is_preference_record(record: Any) -> bool:
    if not isinstance(record, dict):
        return False
    if metadata_conversation(record):
        return True
    # HuggingFaceH4/ultrafeedback_binarized-style rows carry a ``messages``
    # column next to chosen/rejected; with both sides present it's a preference pair.
    if "chosen" in record and "rejected" in record and "messages" in record:
        return not any(key in record for key in ("conversations", "conversation"))
    return ("chosen" in record or "rejected" in record) and not any(key in record for key in _OTHER_SHAPES)


def field_errors(record: dict) -> list[str]:
    errors = []
    for key in ("chosen", "rejected"):
        if key not in record:
            errors.append(f"Missing '{key}' field")
        elif not isinstance(record[key], (str, list)):
            errors.append(f"'{key}' must be a string or list; found {type(record[key]).__name__}")
        elif isinstance(record[key], list) and not record[key]:
            errors.append(f"'{key}' must not be empty")
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


def _implicit_split(chosen: list, rejected: list) -> int:
    n = _common_prefix(chosen, rejected)
    # Never let the shared prefix swallow a whole side: if one side is a
    # prefix of the other, keep at least its last turn as the completion.
    return max(min(n, len(chosen) - 1, len(rejected) - 1), 0)


def _redundant_prompt(record: dict) -> bool:
    """True when ``prompt`` repeats part of the shared head of full ``chosen``/``rejected`` conversations.

    HuggingFaceH4/ultrafeedback_binarized and datasets built like it keep the
    prompt string as metadata while both sides already start with that user
    turn. Prepending it again would double the turn, so these are read like the
    implicit-prompt shape and ``prompt`` rides along untouched.

    A string ``prompt`` next to conversations whose shared head already holds a
    user turn is metadata whatever its exact text: the opening turn
    (mlabonne/orpo-dpo-mix-40k), the last one, or a chat-templated rendering
    (argilla/distilabel-capybara-dpo-7k-binarized). TRL's ``maybe_extract_prompt``
    reads these rows the same way, ignoring the string and splitting the
    shared prefix off, so that is what gets checked.
    """
    chosen, rejected, prompt = record["chosen"], record["rejected"], record["prompt"]
    if not (isinstance(chosen, list) and isinstance(rejected, list)):
        return False
    head = chosen[:_implicit_split(chosen, rejected)]
    if isinstance(prompt, list):
        n = len(prompt)
        return bool(prompt) and n <= len(head) and prompt in (head[:n], head[len(head) - n:])
    return any(isinstance(m, dict) and m.get("role") == "user" for m in head)


def normalize(record: Any) -> Any:
    if not isinstance(record, dict) or field_errors(record):
        return record
    extras = {(_STASH if k == "messages" else k): v for k, v in record.items() if k not in _KEYS}
    if "prompt" in record and _redundant_prompt(record):
        chosen_full, rejected_full = list(record["chosen"]), list(record["rejected"])
        n = _implicit_split(chosen_full, rejected_full)
        prompt, chosen, rejected = chosen_full[:n], chosen_full[n:], rejected_full[n:]
        extras = {**extras, "prompt": record["prompt"]}
    elif "prompt" in record:
        prompt = _as_turns(record["prompt"], "user")
        chosen = _as_turns(record["chosen"], "assistant")
        rejected = _as_turns(record["rejected"], "assistant")
    else:
        chosen_full, rejected_full = list(record["chosen"]), list(record["rejected"])
        n = _implicit_split(chosen_full, rejected_full)
        prompt, chosen, rejected = chosen_full[:n], chosen_full[n:], rejected_full[n:]
    return ConversationRecord({**extras, "messages": prompt + chosen, "rejected": rejected},
                              prompt_length=len(prompt))


def source_metadata(record: Any) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {}
    kind = lambda v: "str" if isinstance(v, str) else "list"
    redundant = "prompt" in record and not field_errors(record) and _redundant_prompt(record)
    return {
        "implicit": "prompt" not in record or redundant,
        "kinds": {k: kind(record.get(k)) for k in _KEYS if k in record},
        "extras": [k for k in record if k not in _KEYS or (redundant and k == "prompt")],
        "order": list(record),
    }


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            try:
                record = json.loads(raw)
            except PARSE_ERRORS as exc:
                yield ParseResult(line_no=line_no, raw=raw, record=None,
                                  parse_error=describe(exc),
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
    if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
        return record
    meta = metadata or {}
    messages = record["messages"]
    split = prompt_end(record)
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
        source = _STASH if key == "messages" else key
        if source in record:
            out[key] = record[source]
    order = meta.get("order")
    if order:
        out = {**{k: out[k] for k in order if k in out}, **out}
    return out
