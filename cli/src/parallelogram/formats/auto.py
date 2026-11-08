"""Auto-detect supported chat dataset shapes line by line.

The default CLI path should accept the shapes users most commonly hand to
fine-tuning stacks directly:

- OpenAI/Qwen chat JSONL:
  ``{"messages": [{"role": ..., "content": ...}]}``
- ShareGPT-style JSONL:
  ``{"conversations": [{"from": ..., "value": ...}]}``
- Prompt/completion JSONL: string halves or conversational message lists.
- Preference (DPO) JSONL: ``prompt``/``chosen``/``rejected``, explicit or implicit prompt.
- Alpaca JSONL: ``instruction``/``input``/``output`` (optionally ``system``, ``history``).

Records are still normalized at the parse boundary, so rules remain format
agnostic. Malformed or unknown records are passed through as OpenAI-chat-shaped
candidates and left for the schema rule to explain.
"""
from __future__ import annotations

import json
from typing import Iterator

from . import ParseResult
from .sharegpt import is_sharegpt_record, normalize as normalize_sharegpt
from . import alpaca, preference, sharegpt
from .prompt_completion import (
    is_prompt_completion_record, normalize as normalize_prompt_completion, source_metadata,
)
from .jsonline import PARSE_ERRORS, describe


def iter_jsonl(path: str) -> Iterator[ParseResult]:
    """Stream-parse JSONL and choose the parser behavior per record."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except PARSE_ERRORS as e:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=None,
                    parse_error=describe(e),
                    source_format="auto",
                )
                continue

            if is_sharegpt_record(record) and "messages" not in record:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=normalize_sharegpt(record),
                    source_format="sharegpt",
                    source_metadata=sharegpt.source_metadata(record),
                )
            elif preference.is_preference_record(record):
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=preference.normalize(record),
                    source_format="preference",
                    source_metadata=preference.source_metadata(record),
                )
            elif is_prompt_completion_record(record):
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=normalize_prompt_completion(record),
                    source_format="prompt-completion",
                    source_metadata=source_metadata(record),
                )
            elif alpaca.is_alpaca_record(record):
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=alpaca.normalize(record),
                    source_format="alpaca",
                    source_metadata=alpaca.source_metadata(record),
                )
            else:
                yield ParseResult(
                    line_no=line_no,
                    raw=raw,
                    record=record,
                    source_format="openai-chat",
                )
