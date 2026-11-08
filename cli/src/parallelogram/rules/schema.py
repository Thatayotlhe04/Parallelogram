"""Schema rule — every record must conform to OpenAI/Qwen chat structure.

This is the foundation rule. Other rules assume the schema rule has already
filtered out structurally invalid records, but they still defend against
malformed input because a single record may produce findings from multiple
rules in one pass.
"""
from __future__ import annotations

from typing import Any, Iterable

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity
from ..formats.openai_chat import VALID_ROLES
from .tool_calls import calls_tools
from ..formats import alpaca, preference, prompt_completion

# Formats whose malformed records are explained in their own field names
# ("Missing 'completion' field") instead of the generic "Missing 'messages'".
# Same precedence as auto-detection.
_FIELD_CHECKS = (
    (preference.is_preference_record, preference.field_errors),
    (prompt_completion.is_prompt_completion_record, prompt_completion.field_errors),
    (alpaca.is_alpaca_record, alpaca.field_errors),
)


@registry.register
class SchemaRule(Rule):
    id = "schema"
    description = "Records must be well-formed in a supported dataset format"
    fix_description = 'No mechanical repair; malformed records and invalid JSON lines are omitted.'
    severity = Severity.ERROR
    fixable = False

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        if not isinstance(record, dict):
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message="Record is not a JSON object",
                detail=f"Found type: {type(record).__name__}",
            )
            return

        if "messages" not in record:
            for detect, field_errors in _FIELD_CHECKS:
                if detect(record):
                    errors = field_errors(record)
                    for message in errors:
                        yield Issue(
                            rule_id=self.id, severity=Severity.ERROR,
                            line_no=line_no, message=message,
                        )
                    if errors:
                        return
                    break

        if "messages" not in record:
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message="Missing 'messages' field",
                detail="Every record must have a 'messages' array",
            )
            return

        messages = record["messages"]
        if not isinstance(messages, list):
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message="'messages' is not a list",
                detail=f"Found type: {type(messages).__name__}",
            )
            return

        if len(messages) == 0:
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message="'messages' is empty",
            )
            return

        yield from self._check_messages(messages, line_no, "")

        if "rejected" in record:
            rejected = record["rejected"]
            if not isinstance(rejected, list) or not rejected:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message="Rejected response is empty or not a list",
                )
            else:
                yield from self._check_messages(rejected, line_no, "Rejected response: ")

    def _check_messages(self, messages: list, line_no: int, prefix: str) -> Iterable[Issue]:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"{prefix}Message {i} is not an object",
                    detail=f"Found type: {type(msg).__name__}",
                )
                continue

            if "role" not in msg:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"{prefix}Message {i} missing 'role' field",
                )
            elif not isinstance(msg["role"], str) or msg["role"] not in VALID_ROLES:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"{prefix}Message {i} has invalid role: {msg['role']!r}",
                    detail=f"Valid roles: {sorted(VALID_ROLES)}",
                )

            if calls_tools(msg) and msg.get("content") is None:
                continue
            if "content" not in msg:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"{prefix}Message {i} missing 'content' field",
                )
            elif not isinstance(msg["content"], str):
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"{prefix}Message {i} 'content' is not a string",
                    detail=f"Found type: {type(msg['content']).__name__}",
                )
