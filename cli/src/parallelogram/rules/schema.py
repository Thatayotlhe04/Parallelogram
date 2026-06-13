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


@registry.register
class SchemaRule(Rule):
    id = "schema"
    description = "Records must conform to OpenAI/Qwen chat schema"
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

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"Message {i} is not an object",
                    detail=f"Found type: {type(msg).__name__}",
                )
                continue

            if "role" not in msg:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"Message {i} missing 'role' field",
                )
            elif msg["role"] not in VALID_ROLES:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"Message {i} has invalid role: {msg['role']!r}",
                    detail=f"Valid roles: {sorted(VALID_ROLES)}",
                )

            if "content" not in msg:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"Message {i} missing 'content' field",
                )
            elif not isinstance(msg["content"], str):
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"Message {i} 'content' is not a string",
                    detail=f"Found type: {type(msg['content']).__name__}",
                )
