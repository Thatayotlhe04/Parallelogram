"""Role sequence rule.

Validates the conversation structure most fine-tuning frameworks expect:
- An optional 'system' message, only at the very start.
- Alternating 'user' and 'assistant' turns starting with 'user'.
- The final turn must be 'assistant' (training targets the assistant turn).

'tool' messages are tolerated within the conversation (they don't break the
alternation expectation in current OpenAI/Anthropic-style schemas) but never
consume an alternation slot in v0.1. This is intentionally conservative —
a separate rule can layer on tool-call validation later.
"""
from __future__ import annotations

from typing import Any, Iterable

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity


@registry.register
class RolesRule(Rule):
    id = "roles"
    description = "Role sequence must be valid for training"
    severity = Severity.ERROR
    fixable = False  # restructuring needs judgment — Phase 2 paid layer

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        if not isinstance(record, dict):
            return
        messages = record.get("messages")
        if not isinstance(messages, list) or not messages:
            return

        roles = []
        for m in messages:
            if isinstance(m, dict) and isinstance(m.get("role"), str):
                roles.append(m["role"])
            else:
                # Schema rule will report; bail early to avoid noise.
                return

        # System messages must only appear at index 0.
        for i, role in enumerate(roles):
            if role == "system" and i != 0:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"'system' message at position {i}, must be first",
                )

        start = 1 if roles[0] == "system" else 0
        body = [r for r in roles[start:] if r != "tool"]

        if not body:
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message="No user/assistant turns after system message",
            )
            return

        if body[0] != "user":
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message=f"First non-system role must be 'user', got {body[0]!r}",
            )
            return

        expected = "user"
        for i, role in enumerate(body):
            if role == "system":
                continue  # already flagged above
            if role != expected:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"Role alternation broken at turn {i}: expected {expected!r}, got {role!r}",
                )
                return
            expected = "assistant" if expected == "user" else "user"

        if body[-1] != "assistant":
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message=f"Conversation must end on 'assistant', ended on {body[-1]!r}",
                detail="Training expects the last turn to be the assistant response",
            )
