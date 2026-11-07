"""PII false positives on reserved addresses and non-contact numeric text."""
from __future__ import annotations

import pytest

from parallelogram.rules.pii import find_pii, redact


@pytest.mark.parametrize('domain', ['docs.example.com', 'mail.example.org', 'sub.example.net'])
def test_reserved_subdomains_are_not_personal_addresses(domain):
    text = 'Email support@' + domain
    assert find_pii(text) == []
    assert redact(text) == text
