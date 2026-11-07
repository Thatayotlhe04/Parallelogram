"""Empty content rule — no whitespace-only message content allowed."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity
from ..core.conversation import map_messages, message_slots, where


@registry.register
class EmptyContentRule(Rule):
    id = "empty-content"
    description = "Message content must not be empty or whitespace-only"
    fix_description = 'Remove empty message turns; drop records that fail revalidation.'
    severity = Severity.ERROR
    fixable = True  # mechanical: drop record or drop empty turn

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        for label, i, msg in message_slots(record):
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
                    message=f"{where(label)}Message {i} (role={msg.get('role', '?')}) has empty content",
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
        if not isinstance(record, dict) or not isinstance(record.get("messages"), list):
            return None

        def keep(msg: dict):
            content = msg.get("content")
            return None if isinstance(content, str) and not content.strip() else msg

        fixed = map_messages(record, keep)
        if not fixed.get("messages") or ("rejected" in fixed and not fixed["rejected"]):
            return None  # nothing trainable left on one side

        # Salvage check: the roles rule re-validates on the next pass, so a
        # record whose structure didn't survive the strip is dropped then.
        return fixed
