"""Validate training turns, including complete parallel tool-call exchanges."""
from __future__ import annotations

from typing import Any, Iterable

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity
from ..core.conversation import conversations, where


@registry.register
class RolesRule(Rule):
    id = "roles"
    description = "Role sequence must be valid for training"
    fix_description = 'No mechanical repair; records with invalid role sequences are dropped.'
    severity = Severity.ERROR
    fixable = False

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        for label, messages in conversations(record):
            if messages:
                yield from self._check(messages, line_no, where(label))

    def _check(self, messages: list, line_no: int, prefix: str) -> Iterable[Issue]:
        def issue(i: int, text: str):
            return Issue(rule_id=self.id, severity=self.severity, line_no=line_no,
                         message=(f"{prefix}{text} at Message {i}" if text.startswith("Role alternation")
                                  else f"{prefix}Message {i}: {text}"), context={"message_index": i})

        expected = "user"
        pending: dict[str, int] = {}
        answered: set[str] = set()
        legacy: tuple[str, int] | None = None
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict) or not isinstance(msg.get("role"), str):
                return  # schema reports malformed messages
            role = msg["role"]
            if role == "system":
                if i != 0:
                    yield issue(i, "'system' message must be first")
                continue
            if role == "tool":
                call_id = msg.get("tool_call_id")
                if not isinstance(call_id, str) or call_id not in pending:
                    reason = ("tool call was already answered" if isinstance(call_id, str) and call_id in answered
                              else "tool message has no matching earlier call id")
                    yield issue(i, reason)
                else:
                    del pending[call_id]
                    answered.add(call_id)
                continue
            if role == "function":
                if legacy is None or msg.get("name") != legacy[0]:
                    yield issue(i, "function message has no matching earlier function call")
                else:
                    legacy = None
                continue
            for call_index in pending.values():
                yield issue(call_index, "tool call never answered before the next assistant or user turn")
            pending.clear()
            if legacy is not None:
                yield issue(legacy[1], "function call never answered")
                legacy = None
            if role != expected:
                yield issue(i, f"Role alternation broken: expected {expected!r}, got {role!r}")
            expected = "assistant" if role == "user" else "user"
            calls = msg.get("tool_calls")
            if role == "assistant" and isinstance(calls, list) and calls:
                for call in calls:
                    if isinstance(call, dict) and isinstance(call.get("id"), str):
                        pending[call["id"]] = i
                expected = "assistant"
            fn = msg.get("function_call")
            if role == "assistant" and isinstance(fn, dict):
                if isinstance(fn.get("name"), str):
                    legacy = (fn["name"], i)
                expected = "assistant"
        for call_index in pending.values():
            yield issue(call_index, "tool call never answered")
        if legacy is not None:
            yield issue(legacy[1], "function call never answered")
        last = messages[-1]
        if isinstance(last, dict) and last.get("role") != "assistant":
            yield issue(len(messages) - 1, "Conversation must end on 'assistant'")
