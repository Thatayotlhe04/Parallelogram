"""Documentation must describe the installed format and version surface."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from parallelogram import __version__
from parallelogram.formats import supported_formats

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('file', ['README.md', 'cli/README.md'])
def test_status_names_current_package_version(file):
    status = re.search(r'## (?:Current )?Status\n(.*?)(?=\n## |\Z)', (ROOT / file).read_text(), re.DOTALL | re.IGNORECASE)
    assert status and 'v' + __version__ in status.group(1)


@pytest.mark.parametrize('file,pattern', [
    ('cli/docs/json-output.md', r'^\| `format` \|.*$'),
    ('action.yml', r'^    description: Dataset parser.*$'),
])
def test_documented_parser_lists_cover_supported_formats(file, pattern):
    line = re.search(pattern, (ROOT / file).read_text(), re.MULTILINE)
    assert line
    for format_id in supported_formats():
        assert format_id in line.group(0)
