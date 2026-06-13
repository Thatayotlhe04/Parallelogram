"""OpenAI/Qwen chat format: {"messages": [{"role": ..., "content": ...}, ...]}.

iter_jsonl streams the file and yields a ParseResult per non-empty line.
JSON errors are captured rather than raised so the runner can report them
as schema issues alongside other findings.

``ParseResult`` and ``VALID_ROLES`` live in the package ``__init__`` now that
they're shared across formats; they're re-exported here for back-compat with
existing imports (e.g. the schema rule).
"""
from __future__ import annotations

import json
from typing import Iterator

from . import ParseResult, VALID_ROLES  # noqa: F401 — re-exported for back-compat


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    """Stream-parse a JSONL file. Yields one ParseResult per non-empty line."""
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
                    record=record,
                    source_format="openai-chat",
                )
            except json.JSONDecodeError as e:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=None,
                    parse_error=f"{e.msg} at column {e.colno}",
                    source_format="openai-chat",
                )
