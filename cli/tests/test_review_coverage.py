"""Additional review checks for paths that already behaved correctly."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app
from parallelogram.formats import get_parser
from parallelogram.rules.eval_overlap import EvalInputError, EvalOverlapRule
from parallelogram.rules.secrets import find_secrets

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('rotation', range(5))
def test_detection_is_per_line_and_new_formats_keep_metadata(tmp_path, rotation):
    records = [
        ('openai-chat', {'messages': [{'role': 'user', 'content': 'chat'}, {'role': 'assistant', 'content': 'A'}]}),
        ('sharegpt', {'conversations': [{'from': 'human', 'value': 'share'}, {'from': 'gpt', 'value': 'A'}]}),
        ('prompt-completion', {'prompt': 'pair', 'completion': 'A'}),
        ('alpaca', {'instruction': 'alpaca', 'input': 'input', 'output': 'A', 'system': 'Help',
                    'history': [['Earlier', 'Answer']], 'metadata': {'nested': [1, 2]}}),
        ('preference', {'prompt': 'pref', 'chosen': 'A', 'rejected': 'B', 'metadata': {'nested': [3, 4]}}),
    ]
    records = records[rotation:] + records[:rotation]
    path, out = tmp_path / 'mixed.jsonl', tmp_path / 'fixed.jsonl'
    path.write_text('\n\n'.join(json.dumps(record) for _, record in records) + '\n')
    assert [p.source_format for p in get_parser('auto')(str(path))] == [fmt for fmt, _ in records]
    result = CliRunner().invoke(app, ['check', str(path), '--fix', '--output', str(out)])
    assert result.exit_code == 0, result.output
    assert [json.loads(line) for line in out.read_text().splitlines()] == [rec for _, rec in records]
    assert CliRunner().invoke(app, ['check', str(out)]).exit_code == 0


@pytest.mark.parametrize('eval_prompt,expected', [(' Q ', 2), ('q', 0), ('Q  X', 0)])
def test_eval_matching_is_trimmed_but_case_sensitive(tmp_path, eval_prompt, expected):
    train, ev = tmp_path / 'train.jsonl', tmp_path / 'eval.jsonl'
    train.write_text(json.dumps({'prompt': 'Q', 'chosen': 'A', 'rejected': 'B'}) + '\n')
    ev.write_text(json.dumps({'instruction': eval_prompt, 'output': 'Other'}) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--eval', str(ev)])
    assert result.exit_code == expected, result.output


def test_failed_eval_load_does_not_cache_a_partial_index(tmp_path):
    path = tmp_path / 'eval.jsonl'
    path.write_text(json.dumps({'prompt': 'Q', 'completion': 'A'}) + '\nnot json\n')
    rule = EvalOverlapRule({'eval_path': str(path)})
    for _ in range(2):
        with pytest.raises(EvalInputError):
            rule._load()
    assert rule._full is None


def test_eval_read_failure_is_controlled(tmp_path):
    rule = EvalOverlapRule({'eval_path': str(tmp_path / 'missing.jsonl')})
    with pytest.raises(EvalInputError, match='Could not read eval file'):
        rule._load()


@pytest.mark.parametrize('text', [
    'pip install sk-learn-compatible-estimator-wrapper',
    '550e8400-e29b-41d4-a716-446655440000',
    'sha256 484acc5926032cbfba5c4f97bded8295ed7392f8281db9d1324fe8af82d53235',
    'def api_key(): return "YOUR_API_KEY_HERE"',
])
def test_ordinary_text_is_not_a_secret(text):
    assert find_secrets(text) == []


@pytest.mark.parametrize('eval_text,expected', [('{"prompt":"Q","completion":"A"}\n', 2), ('not json\n', 3), ('', 0)])
def test_action_eval_path_is_literal_and_exit_preserved(tmp_path, eval_text, expected):
    # Actual CLI, not just argv capture: all metacharacters remain filename text.
    train = tmp_path / 'train file.jsonl'
    train.write_text('{"prompt":"Q","completion":"B"}\n')
    ev = tmp_path / '-eval $(touch INJECTED) `touch INJECTED`; file.jsonl'
    ev.write_text(eval_text)
    summary = tmp_path / 'summary file.md'
    env = {**os.environ, 'PATH': str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'],
           'DATASET_PATH': str(train), 'DATASET_FORMAT': 'prompt-completion',
           'MAX_SEQ_LEN': '4096', 'TOKENIZER': '', 'BASELINE': '', 'EVAL_PATH': str(ev),
           'FAIL_ON': 'error', 'GITHUB_STEP_SUMMARY': str(summary)}
    result = subprocess.run(['bash', str(ROOT / '.github/scripts/report.sh')], env=env,
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == expected, result.stderr
    assert not (tmp_path / 'INJECTED').exists()
    if expected != 3:
        assert '# ' in summary.read_text()
