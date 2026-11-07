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


@pytest.mark.parametrize('options', [[], ['--fix'], ['--json']])
@pytest.mark.parametrize('command', ['check', 'report'])
@pytest.mark.parametrize('bad_eval', ['not json\n', 'null\n', '{"instruction": "Q"}\n'])
def test_bad_eval_fails_closed(tmp_path, command, options, bad_eval):
    if command == 'report' and options == ['--fix']:
        return
    train = write(tmp_path / 'train.jsonl', {'prompt': 'Q', 'completion': 'A'})
    ev = tmp_path / 'eval.jsonl'
    ev.write_text(bad_eval)
    result = runner.invoke(app, [command, train, '--eval', str(ev), *options])
    assert result.exit_code == 3, result.output
    assert 'eval' in result.stderr.lower()
    assert 'line 1' in result.stderr.lower()
    assert 'Traceback' not in result.output


def test_bad_eval_is_checked_even_when_train_is_empty(tmp_path):
    train = tmp_path / 'train.jsonl'
    train.write_text('')
    ev = tmp_path / 'eval.jsonl'
    ev.write_text('not json\n')
    result = runner.invoke(app, ['check', str(train), '--eval', str(ev)])
    assert result.exit_code == 3, result.output


@pytest.mark.parametrize('command', ['check', 'report'])
def test_empty_eval_is_a_valid_empty_set(tmp_path, command):
    train = write(tmp_path / 'train.jsonl', {'prompt': 'Q', 'completion': 'A'})
    ev = tmp_path / 'eval.jsonl'
    ev.write_text('\n')
    result = runner.invoke(app, [command, train, '--eval', str(ev)])
    assert result.exit_code == 0, result.output
