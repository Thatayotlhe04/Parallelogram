"""Preference rule: a DPO pair must actually express a preference.

When the chosen and rejected answers are the same (after trimming
whitespace), the pair carries no learning signal: the loss is pushed in both
directions at once. These come from generation pipelines that sample the same
answer twice, or from merges that copy one column into the other. The rule
only applies to preference records.
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ..core.conversation import prompt_end
from ..core.report import Issue, Severity
from ..core.rules import Rule, registry


def _text(msgs: list) -> list:
    return [
        (m.get("role"), m.get("content").strip() if isinstance(m.get("content"), str) else m.get("content"))
        for m in msgs if isinstance(m, dict)
    ]


@registry.register
class PreferenceRule(Rule):
    id = "preference"
    description = "Chosen and rejected answers in a preference pair must differ"
    severity = Severity.ERROR
    fixable = False  # which answer is better is a judgment call

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        if not isinstance(record, dict) or not isinstance(record.get("rejected"), list):
            return
        messages = record.get("messages")
        if not isinstance(messages, list):
            return
        chosen = messages[prompt_end(record):]
        if chosen and _text(chosen) == _text(record["rejected"]):
            yield Issue(
                rule_id=self.id,
                severity=Severity.ERROR,
                line_no=line_no,
                message="Chosen and rejected answers are identical",
                detail="The pair carries no preference signal; DPO pushes the same text up and down",
            )
