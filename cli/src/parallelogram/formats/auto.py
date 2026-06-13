"""Auto-detect supported chat dataset shapes line by line.

The default CLI path should accept the two shapes users most commonly hand to
fine-tuning stacks directly:

- OpenAI/Qwen chat JSONL:
  ``{"messages": [{"role": ..., "content": ...}]}``
- ShareGPT-style JSONL:
  ``{"conversations": [{"from": ..., "value": ...}]}``

Records are still normalized at the parse boundary, so rules remain format
agnostic. Malformed or unknown records are passed through as OpenAI-chat-shaped
candidates and left for the schema rule to explain.
"""
from __future__ import annotations

import json
from typing import Iterator

from . import ParseResult
from .sharegpt import is_sharegpt_record, normalize as normalize_sharegpt


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    """Stream-parse JSONL and choose the parser behavior per record."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=None,
                    parse_error=f"{e.msg} at column {e.colno}",
                    source_format="auto",
                )
                continue

            if is_sharegpt_record(record):
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=normalize_sharegpt(record),
                    source_format="sharegpt",
                )
            else:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=record,
                    source_format="openai-chat",
                )
