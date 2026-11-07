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
from ..core.conversation import conversations, where


@registry.register
class RolesRule(Rule):
    id = "roles"
    description = "Role sequence must be valid for training"
    fix_description = 'No mechanical repair; records with invalid role sequences are dropped.'
    severity = Severity.ERROR
    fixable = False  # restructuring needs judgment — Phase 2 paid layer

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        # A preference pair is two conversations sharing a prompt; both must
        # be trainable, so the rejected side is checked the same way.
        for label, messages in conversations(record):
            if messages:
                yield from self._check(messages, line_no, where(label))

    def _check(self, messages: list, line_no: int, prefix: str) -> Iterable[Issue]:
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
                    message=f"{prefix}'system' message at position {i}, must be first",
                )

        start = 1 if roles[0] == "system" else 0
        body = [r for r in roles[start:] if r != "tool"]

        if not body:
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message=f"{prefix}No user/assistant turns after system message",
            )
            return

        if body[0] != "user":
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message=f"{prefix}First non-system role must be 'user', got {body[0]!r}",
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
                    message=f"{prefix}Role alternation broken at turn {i}: expected {expected!r}, got {role!r}",
                )
                return
            expected = "assistant" if expected == "user" else "user"

        if body[-1] != "assistant":
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message=f"{prefix}Conversation must end on 'assistant', ended on {body[-1]!r}",
                detail="Training expects the last turn to be the assistant response",
            )
