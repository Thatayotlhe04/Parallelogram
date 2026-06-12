"""Dataset formats and the parser dispatch.

Every format parser streams a JSONL file and yields one ``ParseResult`` per
non-empty line, normalizing the record into the internal OpenAI-chat shape
``{"messages": [{"role", "content"}, ...]}``. Normalizing at the parse boundary
means every rule, the fixer, and the report run unchanged regardless of the
on-disk format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Optional


# The internal canonical role set. Every parser maps into this; the schema rule
# validates against it. Kept here (not in a single format module) because it is
# shared across formats.
VALID_ROLES = {"system", "user", "assistant", "tool"}


@dataclass
class ParseResult:
    line_no: int
    raw: str
    record: Optional[dict]
    parse_error: Optional[str] = None
    source_format: Optional[str] = None


def get_parser(dataset_format: str) -> Callable[[str], Iterator[ParseResult]]:
    """Return the ``iter_jsonl`` for a format id, or raise KeyError."""
    from . import auto, openai_chat, sharegpt

    parsers: dict[str, Callable[[str], Iterator[ParseResult]]] = {
        "auto": auto.iter_jsonl,
        "openai-chat": openai_chat.iter_jsonl,
        "sharegpt": sharegpt.iter_jsonl,
    }
    return parsers[dataset_format]


def supported_formats() -> list[str]:
    """The format ids accepted by ``--format``, sorted for stable messaging."""
    return ["auto", "openai-chat", "sharegpt"]
