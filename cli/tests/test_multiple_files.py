from __future__ import annotations

import json
from pathlib import Path

import pytest
from parallelogram.cli import app
from typer.testing import CliRunner


def datasets(tmp_path):
    clean, warnings, errors = [tmp_path / name for name in ("clean.jsonl", "warnings.jsonl", "errors.jsonl")]
    clean.write_text('{"prompt":"Question","completion":"Answer"}\n')
    warnings.write_text('{"prompt":"\\ufeffQuestion","completion":"Answer"}\n')
    errors.write_text('{"messages":[]}\n')
    return clean, warnings, errors


@pytest.mark.parametrize("json_mode", [False, True])
@pytest.mark.parametrize("indices,expected", [([0, 0], 0), ([0, 1], 1), ([2, 0], 2), ([1, 2, 0], 2)])
def test_all_files_checked(tmp_path, json_mode, indices, expected):
    files = datasets(tmp_path)
    selected = [str(files[i]) for i in indices]
    result = CliRunner().invoke(app, ["check", *selected, *(["--json"] if json_mode else [])])
    assert result.exit_code == expected, result.output
    if json_mode:
        payload = json.loads(result.stdout)
        assert isinstance(payload, list) and [p["file"] for p in payload] == selected
        assert all(p["schema_version"] == 1 for p in payload)
    else:
        assert all(Path(p).name in result.stdout for p in selected)


def test_single_json_remains_object(tmp_path):
    path = datasets(tmp_path)[0]
    result = CliRunner().invoke(app, ["check", str(path), "--json"])
    assert isinstance(json.loads(result.stdout), dict)


def test_output_ambiguity_rejected_before_writes(tmp_path):
    first, second, _ = datasets(tmp_path)
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(app, ["check", str(first), str(second), "--output", str(out)])
    assert result.exit_code == 2 and "exactly one" in result.output
    assert not out.exists()


def test_operational_failure_keeps_checking(tmp_path):
    clean, _, errors = datasets(tmp_path)
    result = CliRunner().invoke(app, ["check", str(tmp_path / "missing.jsonl"), str(errors), str(clean), "--json"])
    assert result.exit_code == 3
    payload = json.loads(result.stdout)
    assert len(payload) == 3 and payload[0]["exit_code"] == 3
    assert payload[1]["summary"]["errors"] > 0
    assert payload[2]["summary"]["errors"] == 0


def test_multi_fix_preview(tmp_path):
    clean, warnings, _ = datasets(tmp_path)
    before = warnings.read_bytes()
    result = CliRunner().invoke(app, ["check", str(clean), str(warnings), "--fix", "--dry-run", "--json"])
    assert result.exit_code == 0
    assert len(json.loads(result.stdout)) == 2
    assert warnings.read_bytes() == before
