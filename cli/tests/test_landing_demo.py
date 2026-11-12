"""Check demo playback behavior without a browser or network dependency."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest
from test_landing_benchmarks import ROOT, parsed_page


def test_demo_transcript_and_progressive_controls():
    elements = parsed_page()
    transcript = next(e for e in elements if e.get('id') == 'demo-transcript')
    assert 'open' in transcript  # Full result available without JavaScript.
    player = next(e for e in elements if e.get('id') == 'cli-demo')
    assert 'hidden' in player  # No dead playback controls when scripts are off.
    scenes = [e for e in elements if 'data-scene' in e]
    assert len(scenes) == 3
    screen = next(e for e in elements if e.get('class') == 'demo-screen')
    assert screen['aria-hidden'] == 'true'  # Transcript provides a stable alternative.
    source = (ROOT / 'landing/index.html').read_text()
    assert 'playback timing is illustrative' in source
    assert '3 records  3 clean  0 errors  0 warnings' in source


def test_demo_playback():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the landing-page interaction tests')
    subprocess.run(
        [node, str(Path(__file__).with_name('landing_demo.cjs'))],
        input=json.dumps(parsed_page()), text=True, check=True, cwd=ROOT,
    )
