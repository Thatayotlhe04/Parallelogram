"""Encoding rule — flags BOM markers and common mojibake patterns.

Mojibake here refers to text that was correctly UTF-8 encoded, decoded as
latin-1 by mistake, then re-encoded as UTF-8. The result is a text file that
opens cleanly in any editor but contains nonsense like 'donâ€™t' instead of
"don't". Models trained on mojibake learn to produce mojibake.

Severity is WARNING rather than ERROR — these are recoverable and are
exactly the kind of fix that should run free locally in the eventual --fix.
"""
from __future__ import annotations

from typing import Any, Iterable, Optional

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity
from ..core.conversation import map_messages, message_slots, where


# Canonical mojibake markers from latin-1 → UTF-8 round-trips,
# paired with their correct Unicode characters. Order matters for replacement
# because some patterns are prefixes of others (e.g. â€“ vs â€).
MOJIBAKE_FIXES = [
    ("â€™",     "\u2019"),  # right single quotation mark (')
    ("â€œ",     "\u201C"),  # left double quotation mark (")
    ("â€\x9d", "\u201D"),  # right double quotation mark (")
    ("â€“",     "\u2013"),  # en dash
    ("â€”",     "\u2014"),  # em dash
    ("â€¦",     "\u2026"),  # ellipsis
    ("Ã©",      "\u00E9"),  # é
    ("Ã¨",      "\u00E8"),  # è
    ("Ã¢",      "\u00E2"),  # â
    ("Ã®",      "\u00EE"),  # î
    ("Ã¯",      "\u00EF"),  # ï
    ("Ã´",      "\u00F4"),  # ô
    ("Ã»",      "\u00FB"),  # û
    ("Â ",      "\u00A0"),  # non-breaking space
]

# What we look for during the check pass — just the markers.
MOJIBAKE_MARKERS = [m for m, _ in MOJIBAKE_FIXES]


def fix_text(text: str) -> str:
    """Strip BOMs and repair known mojibake. Idempotent. Also used by format
    writers that must recognise text after this fix has run."""
    fixed = text.replace("\ufeff", "")
    for bad, good in MOJIBAKE_FIXES:
        if bad in fixed:
            fixed = fixed.replace(bad, good)
    return fixed


@registry.register
class EncodingRule(Rule):
    id = "encoding"
    description = "BOM markers and common mojibake patterns"
    fix_description = 'Strip BOM markers and replace known mojibake patterns.'
    severity = Severity.WARNING
    fixable = True

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        for label, i, msg in message_slots(record):
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue

            if "\ufeff" in content:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.WARNING,
                    line_no=line_no,
                    message=f"{where(label)}Message {i} contains BOM (U+FEFF)",
                    fixable=True,
                )

            for marker in MOJIBAKE_MARKERS:
                if marker in content:
                    yield Issue(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        line_no=line_no,
                        message=f"{where(label)}Message {i} contains likely mojibake: {marker!r}",
                        detail="UTF-8 → latin-1 → UTF-8 round-trip artifact",
                        fixable=True,
                    )
                    break  # one mojibake report per message is enough

    def fix_record(self, record: Any, issue: Issue) -> Optional[Any]:
        """Strip BOMs and replace each known mojibake pattern with its
        correct Unicode equivalent. The fix is idempotent — running it
        twice yields the same result as once.
        """
        def repair(msg: dict) -> dict:
            content = msg.get("content")
            return {**msg, "content": fix_text(content)} if isinstance(content, str) else msg

        return map_messages(record, repair)
