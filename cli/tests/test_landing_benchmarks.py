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
    for panel in ('panel-cost', 'panel-fit', 'panel-quality'):
        element = next(e for e in elements if e.get('id') == panel)
        assert 'hidden' not in element
        assert element['aria-labelledby'] in ids
    script = next(e for e in elements if e.get('src') == 'benchmarks.js')
    assert 'defer' in script
    assert not any('/a-parallel-paradigm/' in e.get('href', '') for e in elements)
    logos = {e['src'] for e in elements if e.get('src', '').startswith('logos/')}
    assert logos == {'logos/openai.svg', 'logos/claude.svg', 'logos/gemini.svg', 'logos/meta.svg'}
    assert all((ROOT / 'landing' / logo).is_file() for logo in logos)
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
