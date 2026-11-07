"""Shared handling for one JSONL line that won't parse.

json.loads raises more than JSONDecodeError: an integer with more than
4,300 digits raises ValueError and deep nesting raises RecursionError. Every
parser treats all three as an invalid line, so one hostile line is reported
by the schema rule instead of crashing the run (which once made the GitHub
Action read a crash as "warnings only" and pass).
"""
from __future__ import annotations

import json

PARSE_ERRORS = (ValueError, RecursionError)  # JSONDecodeError is a ValueError


def describe(exc: BaseException) -> str:
    if isinstance(exc, json.JSONDecodeError):
        return f"{exc.msg} at column {exc.colno}"
    if isinstance(exc, RecursionError):
        return "nested too deeply to parse"
    return "contains a value Python's JSON parser refuses (such as an integer over 4,300 digits)"
