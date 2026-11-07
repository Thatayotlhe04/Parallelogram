"""PII rule: flags personal data in training data.

Email addresses, phone numbers and payment card numbers in a dataset become
things the model can reproduce. Unlike secrets, some datasets include them on
purpose (a support bot that should quote a help address), so this is a
WARNING: visible, exit 1, but not a CI failure under the default gate.
``--fix`` replaces each match with a typed placeholder.

To keep noise down: reserved example domains (example.com, *.test, ...) are
ignored, phone numbers must be international (+...) or US-style
(NNN-NNN-NNNN, with 555 test numbers ignored), and card numbers must pass the
Luhn check.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..core.conversation import map_messages, message_slots, where
from ..core.report import Issue, Severity
from ..core.rules import Rule, registry

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b")
PHONE_INTL = re.compile(r"(?<![\w+])\+\d{1,3}(?:[ .-]?\(?\d{1,4}\)?){1,5}(?!\w)")
PHONE_US = re.compile(r"(?<!\d)\(?(\d{3})\)?[ .-](\d{3})[ .-](\d{4})(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")

_EXAMPLE_DOMAINS = ("example.com", "example.org", "example.net")
_EXAMPLE_SUFFIXES = (".example", ".test", ".invalid", ".localhost", ".local")


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def find_pii(text: str) -> list[tuple[str, re.Match]]:
    found: list[tuple[str, re.Match]] = []
    for m in EMAIL.finditer(text):
        domain = m.group(1).lower()
        if (any(domain == d or domain.endswith("." + d) for d in _EXAMPLE_DOMAINS)
                or domain.endswith(_EXAMPLE_SUFFIXES)):
            continue
        found.append(("email address", m))
    for m in PHONE_INTL.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) == 11 and digits.startswith("1") and digits[4:7] == "555":
            continue
        if 8 <= len(digits) <= 15:
            found.append(("phone number", m))
    for m in PHONE_US.finditer(text):
        if m.group(2) != "555":
            found.append(("phone number", m))
    for m in CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and digits[0] in "3456" and len(set(digits)) > 2 and _luhn(digits):
            found.append(("payment card number", m))
    # Drop matches inside earlier ones (a card number can look like a phone).
    found.sort(key=lambda km: (km[1].start(), -km[1].end()))
    kept: list[tuple[str, re.Match]] = []
    for kind, m in found:
        if kept and m.start() < kept[-1][1].end():
            continue
        kept.append((kind, m))
    return kept


_PLACEHOLDER = {"email address": "[REDACTED_EMAIL]", "phone number": "[REDACTED_PHONE]",
                "payment card number": "[REDACTED_CARD]"}


def mask(kind: str, value: str) -> str:
    if kind == "email address":
        local, _, domain = value.partition("@")
        return f"{local[:1]}***@{domain}"
    digits = re.sub(r"\D", "", value)
    return f"…{digits[-4:]}"


def redact(text: str) -> str:
    out = text
    for kind, m in sorted(find_pii(text), key=lambda km: km[1].start(), reverse=True):
        out = out[:m.start()] + _PLACEHOLDER[kind] + out[m.end():]
    return out


@registry.register
class PiiRule(Rule):
    id = "pii"
    description = "Email addresses, phone numbers and card numbers in training data"
    severity = Severity.WARNING
    fixable = True

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        for label, i, msg in message_slots(record):
            if not isinstance(msg, dict) or not isinstance(msg.get("content"), str):
                continue
            for kind, m in find_pii(msg["content"]):
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.WARNING,
                    line_no=line_no,
                    message=f"{where(label)}Message {i} contains {'an' if kind[0] in 'aeiou' else 'a'} {kind}: {mask(kind, m.group(0))}",
                    detail="The model can learn and repeat it; --fix replaces it with a placeholder",
                    fixable=True,
                    context={"kind": kind},
                )

    def fix_record(self, record: Any, issue: Issue) -> Any | None:
        def scrub(msg: dict) -> dict:
            content = msg.get("content")
            return {**msg, "content": redact(content)} if isinstance(content, str) else msg

        return map_messages(record, scrub)
