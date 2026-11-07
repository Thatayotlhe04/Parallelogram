"""Cross-record fixes must preserve attribution and original eval exclusions."""
from __future__ import annotations

from collections import Counter
import json

from typer.testing import CliRunner

from parallelogram.cli import app


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
