"""Malformed role values must produce schema errors, never tracebacks."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app


@pytest.mark.parametrize('role', [[], {}, ['assistant']])
@pytest.mark.parametrize('shape', ['chat', 'preference', 'sharegpt'])
@pytest.mark.parametrize('command', ['check', 'report'])
def test_unhashable_roles_are_reported(tmp_path, role, shape, command):
    if shape == 'preference':
        record = {'prompt': 'Q', 'chosen': 'A', 'rejected': [{'role': role, 'content': 'B'}]}
    elif shape == 'sharegpt':
        record = {'conversations': [{'from': 'human', 'value': 'Q'}, {'from': role, 'value': 'A'}]}
    else:
        record = {'messages': [{'role': 'user', 'content': 'Q'}, {'role': role, 'content': 'A'}]}
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps(record) + '\n')
    result = CliRunner().invoke(app, [command, str(path), '--json'])
    assert result.exit_code == 2, repr(result.exception)
    payload = json.loads(result.stdout)
    if command == 'check':
        assert any(i['rule'] == 'schema' for i in payload['issues'])
    else:
        assert payload['issues_by_rule']['schema']['errors'] > 0
