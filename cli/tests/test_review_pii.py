"""PII false positives on reserved addresses and non-contact numeric text."""
from __future__ import annotations

import pytest
from parallelogram.rules.pii import find_pii, redact


@pytest.mark.parametrize('domain', ['docs.example.com', 'mail.example.org', 'sub.example.net'])
def test_reserved_subdomains_are_not_personal_addresses(domain):
    text = 'Email support@' + domain
    assert find_pii(text) == []
    assert redact(text) == text


@pytest.mark.parametrize('number', ['+1 212-555-0199', '+1 (212) 555-0199', '+12125550199'])
def test_international_us_example_numbers_are_ignored(number):
    assert find_pii(number) == []
    assert redact(number) == number


@pytest.mark.parametrize('text', [
    'coordinates +23.456789, +45.678912',
    '1 +23456789 = 23456790',
    'Order 123-456-7890',
    'checksum a4539148803436467b',
    'identifier abc212-867-5309def',
    '2026-11-07 12:34:56 version 1.2.3.456',
    'ISBN 978-0-306-40615-7',
    'UUID 123e4567-e89b-12d3-a456-426614174000',
    'order 4539 1488 0343 6468',
])
def test_numeric_non_contacts_are_not_redacted(text):
    assert find_pii(text) == []
    assert redact(text) == text


@pytest.mark.parametrize('text,kind', [
    ('Call +267 77 123 456', 'phone number'),
    ('Call (212) 867-5309', 'phone number'),
    ('Call +1 212-867-5309', 'phone number'),
    ('card 4539 1488 0343 6467', 'payment card number'),
    ('email = "person@parallelogram.dev"', 'email address'),
    ('https://host.test/?contact=person@parallelogram.dev', 'email address'),
])
def test_actual_contact_text_still_gets_redacted(text, kind):
    assert [k for k, _ in find_pii(text)] == [kind]
    assert find_pii(redact(text)) == []
