"""The walkthrough must show real file changes synchronized with CLI output."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
from test_landing_benchmarks import ROOT, parsed_page


def test_terminal_recording_matches_cli():
    subprocess.run([sys.executable, str(ROOT / 'scripts/capture_terminal_demo.py')],
                   check=True, cwd=ROOT)


def test_terminal_transcript_is_available_without_javascript():
    elements = parsed_page()
    transcript = next(e for e in elements if e.get('id') == 'terminal-static')
    assert 'hidden' not in transcript
    assert 'sr-only' not in transcript.get('class', '')
    screen = next(e for e in elements if e.get('id') == 'terminal-screen')
    assert screen is not None
    assert len([e for e in elements if 'data-run' in e]) == 3
    assert not any(e.get('id') in ('demo-clock', 'demo-seek', 'terminal-controls', 'terminal-toggle', 'terminal-replay') for e in elements)
    recording = json.loads((ROOT / 'landing/terminal-recording.json').read_text())
    assert [clip['exit_code'] for clip in recording['commands']] == [2, 1, 0]
    assert len(recording['before']) == 3 and len(recording['after']) == 2
    assert recording['before'][0] == recording['before'][1]
    assert recording['after'][0] == recording['before'][0]
    assert 'maya@acme.com' in recording['before'][2]['messages'][1]['content']
    assert '[REDACTED_EMAIL]' in recording['after'][1]['messages'][1]['content']
    assert 'maya@acme.com' not in json.dumps(recording['after'])
    # The illustrative editor must never be confused with an interactive app.
    desktop = next(e for e in elements if e.get('id') == 'demo-desktop')
    assert desktop['aria-hidden'] == 'true'
    assert not any(e.get('id') == 'terminal-clear' for e in elements)


def test_terminal_replay():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the landing-page interaction tests')
    subprocess.run(
        [node, str(Path(__file__).with_name('landing_demo.cjs'))],
        input=json.dumps(parsed_page()), text=True, check=True, cwd=ROOT,
    )
