"""Cross-record fixes must preserve attribution and original eval exclusions."""
from __future__ import annotations

import json
from collections import Counter

from parallelogram.cli import app
from typer.testing import CliRunner


def test_dataset_drops_keep_prior_fix_attribution(tmp_path):
    train, ev = tmp_path / 'train.jsonl', tmp_path / 'eval.jsonl'
    train.write_text('\n'.join(json.dumps({'prompt': q, 'completion': 'A'})
                               for q in ['\ufeffQ', '\ufeffQ', 'safe']) + '\n')
    ev.write_text(json.dumps({'prompt': 'Q', 'completion': 'A'}) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--eval', str(ev), '--fix', '--json'])
    assert result.exit_code == 1, result.output
    data = json.loads(result.stdout)
    assert data['summary'] == {'total': 3, 'unchanged': 1, 'fixed': 0, 'dropped': 2,
                               'unparseable': 0, 'emitted': 1}
    assert data['outcomes'][0]['rules_fixed'] == ['encoding', 'eval-overlap']
    assert data['outcomes'][1]['rules_fixed'] == ['encoding', 'duplicates']
    assert Counter(data['fixes_by_rule']) == Counter(
        rule for outcome in data['outcomes'] for rule in outcome['rules_fixed'])


def test_redaction_cannot_rescue_an_original_eval_overlap(tmp_path):
    train, ev, out = (tmp_path / name for name in ['train.jsonl', 'eval.jsonl', 'fixed.jsonl'])
    leak = {'prompt': 'Email person@gmail.com', 'completion': 'A'}
    train.write_text(json.dumps(leak) + '\n' + json.dumps({'prompt': 'safe', 'completion': 'B'}) + '\n')
    ev.write_text(json.dumps(leak) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--eval', str(ev), '--fix', '--json', '--output', str(out)])
    assert result.exit_code == 1, result.output
    data = json.loads(result.stdout)
    assert data['summary']['dropped'] == 1
    assert data['outcomes'][0]['rules_fixed'] == ['pii', 'eval-overlap']
    assert json.loads(out.read_text()) == {'prompt': 'safe', 'completion': 'B'}


def test_fixes_recheck_for_pii_revealed_by_encoding_repair(tmp_path):
    train, out = tmp_path / 'train.jsonl', tmp_path / 'fixed.jsonl'
    train.write_text(json.dumps({'instruction': 'Email person@\ufeffgmail.com', 'output': 'OK'}) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--fix', '--output', str(out), '--json'])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text()) == {'instruction': 'Email [REDACTED_EMAIL]', 'output': 'OK'}
    assert CliRunner().invoke(app, ['check', str(out)]).exit_code == 0



def test_revealed_pii_respects_disabled_rule(tmp_path):
    train, out = tmp_path / 'train.jsonl', tmp_path / 'fixed.jsonl'
    train.write_text(json.dumps({'prompt': 'Email person@\ufeffgmail.com', 'completion': 'OK'}) + '\n')
    result = CliRunner().invoke(app, ['check', str(train), '--fix', '--output', str(out), '--disable', 'pii'])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())['prompt'] == 'Email person@gmail.com'


def test_nonconverging_fixer_fails_without_looping():
    from parallelogram.core.fixer import Fixer
    from parallelogram.core.report import Issue, Severity
    from parallelogram.core.rules import Rule

    class NoProgress(Rule):
        id = 'no-progress'
        fixable = True

        def check_record(self, record, line_no):
            return [Issue(self.id, Severity.WARNING, line_no, 'still bad')]

    rule = NoProgress()
    report = Fixer([rule]).fix([(1, {})], list(rule.check_record({}, 1)), set())
    assert report.dropped == 1
    assert report.clean_records == []
    assert report.fixer_errors == ['Record fixes did not converge on line 1']
