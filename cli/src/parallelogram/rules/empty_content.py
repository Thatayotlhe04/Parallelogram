"""Empty content rule — no whitespace-only message content allowed."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity


@registry.register
class EmptyContentRule(Rule):
    id = "empty-content"
    description = "Message content must not be empty or whitespace-only"
    severity = Severity.ERROR
    fixable = True  # mechanical: drop record or drop empty turn

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        if not isinstance(record, dict):
            return
        messages = record.get("messages")
        if not isinstance(messages, list):
            return

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            if not content.strip():
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"Message {i} (role={msg.get('role', '?')}) has empty content",
                    fixable=True,
                )

    def fix_record(self, record: Any, issue: Issue) -> Optional[Any]:
        """Strip empty/whitespace-only messages.

        If the record still has at least one user/assistant pair after
        the strip — and starts with user, ends with assistant — return
        it. Otherwise the record is structurally unsalvageable, return
        None to drop it.

        Note: re-validation in the runner will catch any structure that
        we didn't manage to fix here, so this method can be conservative
        and drop anything ambiguous.
        """
        if not isinstance(record, dict):
            return None
        messages = record.get("messages")
        if not isinstance(messages, list):
            return None

        kept = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, str) and not content.strip():
                continue  # drop this empty turn
            kept.append(msg)

        if not kept:
            return None  # nothing left

        # Salvage check: must start user (or system+user), end assistant.
        # If not, the roles rule will flag it on re-validation and the
        # record will be dropped then. So we just emit and let the
        # pipeline decide.
        return {**record, "messages": kept}
