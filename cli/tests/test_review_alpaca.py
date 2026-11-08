"""Alpaca fixes must preserve field boundaries and optional source values."""
from __future__ import annotations

import json

import pytest
from parallelogram.cli import app
from typer.testing import CliRunner

runner = CliRunner()
KEY = 'sk-proj-Ab3dEf9hIjK2mNoPqR5sTuVwXyZ01234'


@pytest.mark.parametrize('instruction,expected', [
    ('Email person@parallelogram.dev', 'Email [REDACTED_EMAIL]'),
    ('Token ' + KEY, 'Token [REDACTED_SECRET]'),
    ('\ufeffEmail person@parallelogram.dev using ' + KEY, 'Email [REDACTED_EMAIL] using [REDACTED_SECRET]'),
])
def test_redacting_instruction_preserves_input(tmp_path, instruction, expected):
    original = {'instruction': instruction, 'input': 'Keep this separate', 'output': 'OK',
                'history': [['Earlier', 'Reply']], 'system': 'Help.', 'id': 7}
    path, out = tmp_path / 'data.jsonl', tmp_path / 'fixed.jsonl'
    path.write_text(json.dumps(original) + '\n')
    result = runner.invoke(app, ['check', str(path), '--fix', '--output', str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text()) == {**original, 'instruction': expected}
    assert runner.invoke(app, ['check', str(out)]).exit_code == 0


@pytest.mark.parametrize('key', ['input', 'system', 'history'])
def test_null_optional_fields_round_trip_without_coercion(tmp_path, key):
    original = {'instruction': 'Q', 'output': 'A', key: None, 'metadata': {'id': 42}}
    path, out = tmp_path / 'data.jsonl', tmp_path / 'fixed.jsonl'
    path.write_text(json.dumps(original) + '\n')
    result = runner.invoke(app, ['check', str(path), '--fix', '--output', str(out)])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text()) == original
