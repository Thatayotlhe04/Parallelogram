"""Keep the research readable offline and its interactive controls isolated."""
import json
from html.parser import HTMLParser
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


class Elements(HTMLParser):
    def __init__(self):
        super().__init__()
        self.elements = []

    def handle_starttag(self, tag, attrs):
        self.elements.append({'tag': tag, **dict(attrs)})


def parsed_page():
    page = Elements()
    page.feed((ROOT / 'landing/index.html').read_text())
    return page.elements


def test_research_is_readable_without_javascript():
    elements = parsed_page()
    ids = [e['id'] for e in elements if 'id' in e]
    assert len(ids) == len(set(ids))
    panel = next(e for e in elements if e.get('id') == 'panel-quality')
    assert 'hidden' not in panel
    # The cost calculator and ownership table were removed, with their brand marks.
    assert not any(e.get('id') in ('panel-cost', 'panel-fit', 'api-tier', 'owned-budget') for e in elements)
    assert not any(e.get('role') == 'tab' and e.get('aria-controls', '').startswith('panel-') for e in elements)
    script = next(e for e in elements if e.get('src') == 'benchmarks.js')
    assert 'defer' in script
    assert not any('/a-parallel-paradigm/' in e.get('href', '') for e in elements)
    assert not any(e.get('src', '').startswith('logos/') for e in elements)
    assert not (ROOT / 'landing/logos').exists()
    evidence = json.loads((ROOT / 'landing/benchmark-evidence.json').read_text())
    assert evidence['source_revision'] == '1fcefb66637638ead2c50ab60d49ffaada57d769'


def test_research_interactions():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for the landing-page interaction tests')
    subprocess.run(
        [node, str(Path(__file__).with_name('landing_benchmarks.cjs'))],
        input=json.dumps(parsed_page()), text=True, check=True, cwd=ROOT,
    )
