"""Credentials in malformed data must not escape through other diagnostics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app

KEY = 'sk-proj-Ab3dEf9hIjK2mNoPqR5sTuVwXyZ01234'


def secret_role_file(tmp_path):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps({'messages': [
        {'role': 'user', 'content': 'Q'}, {'role': KEY, 'content': ''},
    ]}) + '\n')
    return path


@pytest.mark.parametrize('command,options', [
    ('check', []), ('check', ['--json']), ('check', ['--fix', '--json']),
    ('report', []), ('report', ['--json']), ('report', ['--markdown']),
])
def test_secret_roles_never_reach_output(tmp_path, command, options):
    path = secret_role_file(tmp_path)
    out = tmp_path / 'report.txt'
    result = CliRunner().invoke(app, [command, str(path), *options,
                                    *(['--out', str(out)] if command == 'report' else [])],
                                env={'COLUMNS': '300'})
    assert result.exit_code == 2, result.output
    assert KEY not in result.output
    if out.exists():
        assert KEY not in out.read_text()


def test_secret_role_never_reaches_action_summary(tmp_path):
    root = Path(__file__).resolve().parents[2]
    summary = tmp_path / 'summary.md'
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
           'DATASET_PATH': str(secret_role_file(tmp_path)), 'DATASET_FORMAT': 'auto',
           'MAX_SEQ_LEN': '4096', 'TOKENIZER': '', 'BASELINE': '', 'EVAL_PATH': '',
           'FAIL_ON': 'error', 'GITHUB_STEP_SUMMARY': str(summary)}
    result = subprocess.run(['bash', str(root / '.github/scripts/report.sh')], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert KEY not in result.stdout + result.stderr + summary.read_text()


@pytest.mark.parametrize('word', ['your', 'example', 'xxxx', 'dummy', 'fake', 'redacted', 'placeholder', 'sample', 'test1234'])
def test_placeholder_substrings_do_not_hide_credentials(word):
    from parallelogram.rules.secrets import find_secrets, redact

    key = 'sk-proj-Ab3dEf9hIjK2' + word + 'mNoPqR5sTuVwXyZ01234'
    assert len(find_secrets(key)) == 1
    assert redact(key) == '[REDACTED_SECRET]'
