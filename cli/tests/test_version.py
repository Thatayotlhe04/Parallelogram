"""The version flag works before subcommand validation."""
from __future__ import annotations

import pytest
from typer.testing import CliRunner

from parallelogram import __version__
from parallelogram.cli import app


@pytest.mark.parametrize("args", [["--version"], ["--version", "check"]])
def test_version(args):
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0
    assert result.stdout == f"parallelogram {__version__}\n"
