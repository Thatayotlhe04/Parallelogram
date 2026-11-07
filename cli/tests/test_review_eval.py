"""Independent 1.1.0 review: eval isolation and failure handling."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app

runner = CliRunner()


def write(path, record):
    path.write_text(json.dumps(record) + '\n', encoding='utf-8')
    return str(path)


@pytest.mark.parametrize('command', ['check', 'report'])
@pytest.mark.parametrize('format_id,record', [
    ('openai-chat', {'messages': [{'role': 'user', 'content': 'Q'}, {'role': 'assistant', 'content': 'A'}]}),
    ('alpaca', {'instruction': 'Q', 'output': 'A'}),
    ('preference', {'prompt': 'Q', 'chosen': 'A', 'rejected': 'B'}),
])
def test_eval_detects_its_own_format(tmp_path, command, format_id, record):
    train = write(tmp_path / 'train.jsonl', record)
    ev = write(tmp_path / 'eval.jsonl', {'prompt': ' Q ', 'completion': 'other'})
    result = runner.invoke(app, [command, train, '--format', format_id, '--eval', ev, '--json'])
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    assert 'eval-overlap' in (payload.get('issues_by_rule') or {i['rule'] for i in payload['issues']})
