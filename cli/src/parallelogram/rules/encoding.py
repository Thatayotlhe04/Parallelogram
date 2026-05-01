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


@registry.register
class EncodingRule(Rule):
    id = "encoding"
    description = "BOM markers and common mojibake patterns"
    severity = Severity.WARNING
    fixable = True

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

            if "\ufeff" in content:
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.WARNING,
                    line_no=line_no,
                    message=f"Message {i} contains BOM (U+FEFF)",
                    fixable=True,
                )

            for marker in MOJIBAKE_MARKERS:
                if marker in content:
                    yield Issue(
                        rule_id=self.id,
                        severity=Severity.WARNING,
                        line_no=line_no,
                        message=f"Message {i} contains likely mojibake: {marker!r}",
                        detail="UTF-8 → latin-1 → UTF-8 round-trip artifact",
                        fixable=True,
                    )
                    break  # one mojibake report per message is enough

    def fix_record(self, record: Any, issue: Issue) -> Optional[Any]:
        """Strip BOMs and replace each known mojibake pattern with its
        correct Unicode equivalent. The fix is idempotent — running it
        twice yields the same result as once.
        """
        if not isinstance(record, dict):
            return record
        messages = record.get("messages")
        if not isinstance(messages, list):
            return record

        new_messages = []
        for msg in messages:
            if not isinstance(msg, dict):
                new_messages.append(msg)
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                new_messages.append(msg)
                continue

            # Strip BOM anywhere in the string
            fixed = content.replace("\ufeff", "")
            # Apply mojibake fixes in declaration order
            for bad, good in MOJIBAKE_FIXES:
                if bad in fixed:
                    fixed = fixed.replace(bad, good)
            new_messages.append({**msg, "content": fixed})

        return {**record, "messages": new_messages}
