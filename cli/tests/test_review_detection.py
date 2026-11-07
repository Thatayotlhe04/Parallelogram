"""Mixed files are detected per line, without hiding malformed chat fields."""
from __future__ import annotations

import json

import pytest
from parallelogram.cli import app
from parallelogram.formats import get_parser
from typer.testing import CliRunner


@pytest.mark.parametrize('messages', [None, 'bad', []])
def test_sharegpt_cannot_hide_malformed_messages(tmp_path, messages):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps({'messages': messages, 'conversations': [
        {'from': 'human', 'value': 'Q'}, {'from': 'gpt', 'value': 'A'}]}) + '\n')
    parsed, = get_parser('auto')(str(path))
    assert parsed.source_format == 'openai-chat'
    for command in ('check', 'report'):
        result = CliRunner().invoke(app, [command, str(path)])
        assert result.exit_code == 2, result.output
