"""Validate executable tool data without disclosing argument values."""
from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..core.conversation import message_slots, where
from ..core.report import Issue, Severity
from ..core.rules import Rule, registry


def calls_tools(msg: dict) -> bool:
    """Only assistant calls can replace prose content."""
    return msg.get("role") == "assistant" and bool(msg.get("tool_calls") or msg.get("function_call"))


def _reject_constant(value: str) -> None:
    raise ValueError("Non-JSON numeric constant")


@registry.register
class ToolCallsRule(Rule):
    id = "tool-calls"
    description = "Tool definitions and calls must be well-formed and consistent"
    severity = Severity.ERROR
    fixable = False
    fix_description = "No safe mechanical repair; invalid tool-call records are dropped."

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        if not isinstance(record, dict):
            return

        def issue(message: str, index: int | None = None, label: str = ""):
            return Issue(rule_id=self.id, severity=self.severity, line_no=line_no,
                         message=f"{where(label)}{message}",
                         context={"message_index": index} if index is not None else {})

        names = set()
        if "parallel_tool_calls" in record and not isinstance(record["parallel_tool_calls"], bool):
            yield issue("'parallel_tool_calls' must be a boolean")
        for key in ("tools", "functions"):
            if key not in record:
                continue
            definitions = record[key]
            if not isinstance(definitions, list):
                yield issue(f"'{key}' must be a list of function definitions")
                continue
            for i, definition in enumerate(definitions):
                fn = definition
                if key == "tools":
                    if not isinstance(definition, dict) or definition.get("type") != "function":
                        yield issue(f"{key}[{i}] must have type 'function'")
                    fn = definition.get("function") if isinstance(definition, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"].strip():
                    yield issue(f"{key}[{i}] must contain a nonempty function name")
                    continue
                if key == "tools":
                    names.add(fn["name"])
                if "parameters" in fn and not isinstance(fn["parameters"], dict):
                    yield issue(f"{key}[{i}] parameters must be an object")
                if "description" in fn and not isinstance(fn["description"], str):
                    yield issue(f"{key}[{i}] description must be a string")

        seen: set[str] = set()
        for label, i, msg in message_slots(record):
            if not isinstance(msg, dict):
                continue
            calls = msg.get("tool_calls", [])
            if "tool_calls" in msg:
                if msg.get("role") != "assistant":
                    yield issue(f"Message {i}: only assistant messages may contain tool_calls", i, label)
                if not isinstance(calls, list) or not calls:
                    yield issue(f"Message {i}: tool_calls must be a nonempty list", i, label)
                    calls = []
            if "function_call" in msg:
                if msg.get("role") != "assistant":
                    yield issue(f"Message {i}: only assistant messages may contain function_call", i, label)
                calls = [*calls, {"type": "function", "function": msg["function_call"]}]
            for j, call in enumerate(calls):
                prefix = f"Message {i}, call {j}: "
                if not isinstance(call, dict):
                    yield issue(prefix + "call must be an object", i, label)
                    continue
                if "tool_calls" in msg:
                    call_id = call.get("id")
                    if not isinstance(call_id, str) or not call_id.strip():
                        yield issue(prefix + "id must be a nonempty string", i, label)
                    elif call_id in seen:
                        yield issue(prefix + "duplicate call id within record", i, label)
                    else:
                        seen.add(call_id)
                if call.get("type") != "function":
                    yield issue(prefix + "type must be 'function'", i, label)
                fn = call.get("function")
                if not isinstance(fn, dict):
                    yield issue(prefix + "function must be an object", i, label)
                    continue
                name = fn.get("name")
                if not isinstance(name, str) or not name.strip():
                    yield issue(prefix + "function name must be a nonempty string", i, label)
                elif "tools" in record and name not in names:
                    yield issue(prefix + "function name is not declared in tools", i, label)
                args = fn.get("arguments")
                try:
                    parsed = json.loads(args, parse_constant=_reject_constant) if isinstance(args, str) else None
                except (ValueError, RecursionError):
                    parsed = None
                if not isinstance(parsed, dict):
                    yield issue(prefix + "arguments must be a JSON-encoded object", i, label)
