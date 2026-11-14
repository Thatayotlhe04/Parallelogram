"""The CLI panel is a recording of a real session whose output matches the CLI today."""
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


def test_recording_is_a_real_session():
    sys.path.insert(0, str(ROOT / 'scripts'))
    from capture_terminal_demo import SIZES, cast_text, commands
    assert commands()[0] == 'parallelogram check data/train.jsonl --eval data/eval.jsonl'
    assert commands()[2].endswith('--fix -o data/train.clean.jsonl')
    for cols, rows in SIZES.values():
        header, text = cast_text(ROOT / f'scripts/demo/session-{cols}.cast')
        assert (header['width'], header['height']) == (cols, rows)
        assert 'sk-proj-' in text and '[REDACTED_SECRET]' in text
        assert '2403 records' in text and '0 errors' in text
        for media, limit in ((f'terminal-{cols}.mp4', 2_000_000), (f'terminal-{cols}.webp', 200_000)):
            path = ROOT / 'landing/demo' / media
            assert path.is_file() and path.stat().st_size < limit
    # The editor shows the dataset's real lines, with the key the CLI reports selected.
    from capture_terminal_demo import EDITOR_FIRST, EDITOR_LINE, editor_lines
    from demo_dataset import build, dumps
    rows, total, column, selected = editor_lines()
    lines = dumps(build()[0]).splitlines()
    assert total == 2412 and lines[EDITOR_LINE - 1][column - 1:].startswith('sk-proj-')
    page = (ROOT / 'landing/index.html').read_text()
    assert rows in page and f'<span class="ln">{EDITOR_FIRST}</span>' in rows
    assert f'Ln {EDITOR_LINE}, Col {column} ({selected} selected)' in page


def test_editor_chrome_is_drawn_not_typed():
    """Symbol glyphs render per the visitor's fallback fonts; the editor's icons must not depend on them."""
    page = (ROOT / 'landing/index.html').read_text()
    editor = page[page.index('<div class="win-editor">'):page.index('<div class="win-term"')]
    assert not set(editor) & set('⌄›⑂◆ⓘ≡')
    assert editor.count('<svg class="fi ') >= 12 and 'class="ed-branch"' in editor


def test_mark_is_the_same_lattice_everywhere():
    """One mark: the parallelogram ruled 3x3, in every header and footer and in the favicon."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from build_shared import BRAND, PAGES
    outline, lattice = 'M8 5 L29 5 L24 27 L3 27 Z', 'M15 5 L10 27 M22 5 L17 27 M6.33 12.33 H27.33 M4.67 19.67 H25.67'
    assert outline in BRAND and lattice in BRAND
    for name in PAGES:
        assert (ROOT / 'landing' / name).read_text().count(BRAND) == 2, name
    favicon = (ROOT / 'landing/favicon.svg').read_text()
    assert outline in favicon and lattice in favicon


def test_terminal_transcript_is_available_without_javascript():
    elements = parsed_page()
    transcript = next(e for e in elements if e.get('id') == 'terminal-static')
    assert 'hidden' not in transcript
    assert 'sr-only' not in transcript.get('class', '')
    video = next(e for e in elements if e.get('id') == 'demo-video')
    assert {'muted', 'playsinline', 'loop'} <= set(video) and 'autoplay' not in video
    assert video['data-wide'] == 'demo/terminal-108.mp4' and video['data-narrow'] == 'demo/terminal-60.mp4'
    # The recording is decorative; the transcript carries the content.
    desktop = next(e for e in elements if e.get('id') == 'demo-desktop')
    assert desktop['aria-hidden'] == 'true'
    backdrop = ROOT / 'landing/scene-backdrop.webp'
    assert backdrop.is_file() and backdrop.stat().st_size < 400_000


def test_terminal_replay():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the landing-page interaction tests')
    subprocess.run(
        [node, str(Path(__file__).with_name('landing_demo.cjs'))],
        input=json.dumps(parsed_page()), text=True, check=True, cwd=ROOT,
    )
