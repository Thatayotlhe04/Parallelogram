"""Secrets rule: credentials must not be in training data.

A model fine-tuned on a live API key or a private key can repeat it to
anyone who asks the right question, and a published dataset leaks it to
everyone. Scraped chat logs and synthetic data built from real sessions carry
these more often than people expect.

Severity is ERROR: a credential in a dataset is never intended. ``--fix``
replaces each match with ``[REDACTED_SECRET]``. Matches are never printed in
full, so CI logs don't leak what the rule found. Placeholders such as
``sk-YOUR_KEY`` or AWS's documented example key are ignored.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..core.fields import map_record_strings, scan_targets
from ..core.report import Issue, Severity
from ..core.rules import Rule, registry

REDACTED = "[REDACTED_SECRET]"

# (kind, pattern). Order matters: more specific prefixes first.
# Keys often follow an underscore (OPENAI_API_KEY_sk-...), where \b doesn't
# match, so prefixes are anchored on "not preceded by a letter or digit".
_B = r"(?<![A-Za-z0-9])"

PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private key", re.compile(
        r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY(?: BLOCK)?-----[\s\S]*?"
        r"(?:-----END (?:[A-Z]+ )*PRIVATE KEY(?: BLOCK)?-----|\Z)")),
    ("Anthropic API key", re.compile(_B + r"sk-ant-[A-Za-z0-9_-]{20,}")),
    ("OpenAI API key", re.compile(_B + r"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}|" + _B + r"sk-[A-Za-z0-9]{32,}(?![A-Za-z0-9])")),
    ("AWS access key", re.compile(_B + r"(?:AKIA|ASIA)[0-9A-Z]{16}(?![0-9A-Za-z])")),
    # A secret access key has no prefix of its own; require the name beside it.
    ("AWS secret access key", re.compile(
        r"(?i:aws_?secret(?:_?access)?_?key)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+])")),
    ("GitHub token", re.compile(_B + r"gh[pousr]_[A-Za-z0-9]{36,}(?![A-Za-z0-9])|" + _B + r"github_pat_[A-Za-z0-9_]{22,}")),
    ("Hugging Face token", re.compile(_B + r"hf_[A-Za-z0-9]{30,}(?![A-Za-z0-9])")),
    ("Slack token", re.compile(_B + r"xox[abeprs]-[A-Za-z0-9-]{10,}")),
    ("Slack webhook", re.compile(r"https://hooks\.slack\.com/(?:services|workflows|triggers)/[A-Za-z0-9/_-]{20,}")),
    ("Google API key", re.compile(_B + r"AIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])")),
    ("Google OAuth client secret", re.compile(_B + r"GOCSPX-[A-Za-z0-9_-]{24,}")),
    ("Stripe secret key", re.compile(_B + r"(?:sk|rk)_live_[0-9A-Za-z]{16,}(?![0-9A-Za-z])")),
]

_PLACEHOLDER_WORDS = ("your", "example", "xxxx", "dummy", "fake", "redacted", "placeholder", "sample", "test1234")


def looks_placeholder(value: str) -> bool:
    """Documentation and tutorial data is full of fake keys; skip them."""
    low = value.lower()
    if value == "AKIAIOSFODNN7EXAMPLE" or any(
        part in _PLACEHOLDER_WORDS for part in re.split(r"[-_]", low)
    ):
        return True
    body = re.sub(r"^[A-Za-z]+[-_]", "", value)
    return len(set(body)) < 8


def mask(value: str) -> str:
    """Enough to find the match, never enough to use it."""
    return value[:6] + "…" if len(value) > 6 else "…"


def find_secrets(text: str) -> list[tuple[str, re.Match]]:
    found: list[tuple[str, re.Match]] = []
    taken: list[tuple[int, int]] = []
    for kind, pattern in PATTERNS:
        for m in pattern.finditer(text):
            if kind != "private key" and looks_placeholder(m.group(0)):
                continue
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            taken.append((m.start(), m.end()))
            found.append((kind, m))
    return found


def redact(text: str) -> str:
    out = text
    for _, m in sorted(find_secrets(text), key=lambda km: km[1].start(), reverse=True):
        out = out[:m.start()] + REDACTED + out[m.end():]
    return out


@registry.register
class SecretsRule(Rule):
    id = "secrets"
    description = "API keys, tokens and private keys must not be in training data"
    fix_description = 'Redact detected credentials and private keys with placeholders.'
    severity = Severity.ERROR
    fixable = True  # redaction is mechanical

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        # Every string, not only message content: ids, metadata, names and
        # tool-call arguments carry keys too, and a key there leaks just as well.
        for label, text in scan_targets(record):
            for kind, m in find_secrets(text):
                yield Issue(
                    rule_id=self.id,
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message=f"{label} contains a likely {kind}: {mask(m.group(0))}",
                    detail="A model trained on it can repeat it; --fix redacts it",
                    fixable=True,
                    context={"kind": kind},
                )

    def fix_record(self, record: Any, issue: Issue) -> Any | None:
        return map_record_strings(record, redact)
