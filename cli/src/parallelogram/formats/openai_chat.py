"""OpenAI chat format: {"messages": [{"role": ..., "content": ...}, ...]}.

iter_jsonl streams the file and yields a ParseResult per non-empty line.
JSON errors are captured rather than raised so the runner can report them
as schema issues alongside other findings.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Optional


VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class ParseResult:
    line_no: int
    raw: str
    record: Optional[dict]
    parse_error: Optional[str] = None


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    """Stream-parse a JSONL file. Yields one ParseResult per non-empty line."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                yield ParseResult(line_no=line_no, raw=raw, record=record)
            except json.JSONDecodeError as e:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=None,
                    parse_error=f"{e.msg} at column {e.colno}",
                )
