"""Version every JSON mode and continue accepting pre-1.0 report baselines."""
from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from parallelogram.cli import app


runner = CliRunner()


@pytest.mark.parametrize("args", [["check"], ["check", "--fix"], ["report"]])
def test_json_schema_version(tmp_path, args):
    path = tmp_path / "data.jsonl"
    path.write_text('{"prompt": "Q", "completion": "A"}\n')
    result = runner.invoke(app, [*args, str(path), "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["schema_version"] == 1


@pytest.mark.parametrize("regression", [False, True])
def test_legacy_04_baseline_without_schema_version(tmp_path, regression):
    path = tmp_path / "data.jsonl"
    path.write_text('{"prompt": "Q", "completion": "A"}\n')
    baseline = tmp_path / "baseline.json"
    # The 0.4.x report shape had all these rates and no schema_version field.
    baseline.write_text(json.dumps({
        "file": "previous.jsonl", "format": "auto", "disabled_rules": [],
        "summary": {"total_records": 1, "clean_records": 1, "exit_code": 0},
        "rates": {
            "clean_fraction": 1.0, "errors_per_record": 0.0,
            "duplicates_per_record": 0.0, "over_budget_per_record": 0.0,
            "dropped_per_record": 0.0,
        },
    }))
    if regression:
        path.write_text('{"prompt": "Q"}\n')
    result = runner.invoke(app, ["report", str(path), "--baseline", str(baseline), "--json"])
    assert result.exit_code == (3 if regression else 0), result.output
    assert ("regressed" if regression else "no regression") in result.stderr
