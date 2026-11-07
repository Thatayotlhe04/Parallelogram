"""Preserve preference prompts separately from multi-turn continuations."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app
from parallelogram.core.conversation import conversations
from parallelogram.formats.preference import normalize
from parallelogram.rules.context_window import ContextWindowRule


def u(text):
    return {'role': 'user', 'content': text}


def a(text):
    return {'role': 'assistant', 'content': text}


@pytest.mark.parametrize('implicit', [False, True])
@pytest.mark.parametrize('repair', [False, True])
def test_multiturn_preference_preserves_each_side(tmp_path, implicit, repair):
    prompt = [u('Q')]
    chosen = [a('Yes'), u('Chosen followup'), a('Final chosen')]
    rejected = [a('No'), u('Rejected followup'), a('Final rejected')]
    record = {'chosen': prompt + chosen, 'rejected': prompt + rejected} if implicit else {
        'prompt': prompt, 'chosen': chosen, 'rejected': rejected}
    # Extra fields must not collide with internal prompt-boundary bookkeeping.
    record.update({'id': 4, 'prompt_length': 'opaque metadata'})
    if repair:
        record['chosen'][-1]['content'] = '\ufeffFinal chosen'
    train, out = tmp_path / 'train.jsonl', tmp_path / 'fixed.jsonl'
    train.write_text(json.dumps(record) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--fix', '--output', str(out)])
    assert result.exit_code == 0, result.output
    expected = json.loads(json.dumps(record).replace('\\ufeff', ''))
    assert json.loads(out.read_text()) == expected
    assert CliRunner().invoke(app, ['check', str(out)]).exit_code == 0


def test_rejected_context_uses_only_the_actual_prompt():
    record = normalize({'prompt': [u('Q')],
                        'chosen': [a('short'), u('x' * 400), a('answer')],
                        'rejected': [a('bad'), u('y' * 400), a('worse')]})
    chosen, rejected = conversations(record)
    assert rejected[1] == [u('Q'), a('bad'), u('y' * 400), a('worse')]
    rule = ContextWindowRule()
    assert rule.record_tokens(record) == max(
        rule.record_tokens({'messages': side}) for _, side in (chosen, rejected))


def test_multiturn_preference_matches_eval_prompt(tmp_path):
    train, ev = tmp_path / 'train.jsonl', tmp_path / 'eval.jsonl'
    train.write_text(json.dumps({'prompt': 'Q', 'chosen': [a('Yes'), u('Again'), a('Final')],
                                'rejected': 'No'}) + '\n')
    ev.write_text(json.dumps({'instruction': 'Q', 'output': 'Other'}) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--eval', str(ev), '--json'])
    assert any(i['rule'] == 'eval-overlap' for i in json.loads(result.stdout)['issues'])


def test_distinct_preference_prompt_boundaries_are_not_duplicates():
    from parallelogram.rules.duplicates import record_hash

    first = normalize({'prompt': [u('Q')], 'chosen': [a('Yes'), u('Again'), a('Final')],
                       'rejected': [a('No'), u('Again'), a('Wrong')]})
    second = normalize({'prompt': [u('Q'), a('Yes'), u('Again')], 'chosen': [a('Final')],
                        'rejected': [a('No'), u('Again'), a('Wrong')]})
    assert record_hash(first) != record_hash(second)


def test_context_fix_can_shorten_rejected_user_turn(tmp_path):
    train, out = tmp_path / 'train.jsonl', tmp_path / 'fixed.jsonl'
    record = {'prompt': 'Q', 'chosen': 'Good',
              'rejected': [a('Bad'), u('long ' * 300), a('Worse')]}
    train.write_text(json.dumps(record) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--fix', '--max-seq-len', '80', '--output', str(out)])
    assert result.exit_code == 0, result.output
    fixed = json.loads(out.read_text())
    assert fixed['prompt'] == 'Q' and fixed['chosen'] == 'Good'
    assert fixed['rejected'][0] == a('Bad') and fixed['rejected'][-1] == a('Worse')
    assert len(fixed['rejected'][1]['content']) < len(record['rejected'][1]['content'])
    assert CliRunner().invoke(app, ['check', str(out), '--max-seq-len', '80']).exit_code == 0


@pytest.mark.parametrize('command', ['check', 'report'])
def test_empty_chosen_completion_is_invalid(tmp_path, command):
    path = tmp_path / 'data.jsonl'
    path.write_text(json.dumps({'prompt': [u('Q'), a('Old answer')], 'chosen': [],
                                'rejected': [u('Again'), a('Bad')]}) + '\n')
    result = CliRunner().invoke(app, [command, str(path), '--json'])
    assert result.exit_code == 2, result.output
    payload = json.loads(result.stdout)
    if command == 'check':
        assert any(i['rule'] == 'schema' and 'chosen' in i['message'] for i in payload['issues'])
