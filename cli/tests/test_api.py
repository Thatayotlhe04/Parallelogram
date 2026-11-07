"""Public API/CLI parity, bounded diagnostics and non-destructive file handling."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from parallelogram import ParallelogramError, check, fix, rules
from parallelogram import api
from parallelogram.cli import app
from parallelogram.core.report import Issue, Report, Severity
from parallelogram.core.rules import registry
from parallelogram.output.json_output import report_to_dict

EXAMPLES = Path(__file__).parents[1] / "examples"
runner = CliRunner()


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.jsonl")), ids=lambda p: p.name)
@pytest.mark.parametrize("limit", [None, 0, 2, 500])
def test_check_cli_api_parity(path, limit, capsys):
    result = check(path, limit=limit)
    assert capsys.readouterr() == ("", "")
    cli = runner.invoke(app, ["check", str(path), "--json"]
                        + ([] if limit is None else ["--limit", str(limit)]))
    assert cli.exit_code == result.exit_code
    assert json.loads(cli.stdout) == result.to_dict()


@pytest.mark.parametrize("path", sorted(EXAMPLES.glob("*.jsonl")), ids=lambda p: p.name)
@pytest.mark.parametrize("dry_run", [False, True])
def test_fix_cli_api_parity_and_roundtrip(path, dry_run, tmp_path, capsys):
    before = path.read_bytes()
    api_out, cli_out = tmp_path / "api.jsonl", tmp_path / "cli.jsonl"
    result = fix(path, output=api_out, dry_run=dry_run)
    assert capsys.readouterr() == ("", "")
    cli = runner.invoke(app, ["check", str(path), "--fix", "--json", "--output", str(cli_out)]
                        + (["--dry-run"] if dry_run else []))
    assert cli.exit_code == result.exit_code
    assert json.loads(cli.stdout) == result.to_dict()
    assert path.read_bytes() == before
    if dry_run:
        assert not api_out.exists() and not cli_out.exists()
    else:
        assert api_out.read_bytes() == cli_out.read_bytes()
        assert not check(api_out).report.has_errors


def test_check_options_and_filtering_parity(tmp_path, capsys):
    path, evaluation, output = (tmp_path / name for name in ("train", "eval", "out"))
    path.write_text('{"prompt":"A","completion":"B"}\n{"prompt":"C","completion":"D"}\n')
    evaluation.write_text('{"prompt":"A","completion":"Different"}\n')
    result = check(path, eval_path=evaluation, format="prompt-completion",
                   disable=("pii",), max_seq_len=20, limit=1, output=output)
    assert capsys.readouterr() == ("", "")
    cli = runner.invoke(app, ["check", str(path), "--json", "--eval", str(evaluation),
                             "--format", "prompt-completion", "--disable", "pii",
                             "--max-seq-len", "20", "--limit", "1"])
    assert json.loads(cli.stdout) == result.to_dict()
    assert "WARNING" in cli.stderr
    assert json.loads(output.read_text()) == {"prompt": "C", "completion": "D"}
    fixed = fix(path, eval_path=evaluation, disable=("pii",), output=output)
    cli_fix = runner.invoke(app, ["check", str(path), "--fix", "--json", "--eval", str(evaluation),
                                 "--disable", "pii"])
    assert json.loads(cli_fix.stdout) == fixed.to_dict()
    assert fixed.exit_code == 1


def test_limit_orders_by_severity_then_line_without_changing_counts():
    report = Report(total_records=4, valid_records=1, issues=[
        Issue("a", Severity.INFO, None, "note"),
        Issue("b", Severity.WARNING, 1, "warning"),
        Issue("b", Severity.ERROR, 4, "later error"),
        Issue("a", Severity.ERROR, 2, "first error"),
        Issue("a", Severity.WARNING, 3, "later warning"),
    ])
    full = report_to_dict(report)
    assert [i["message"] for i in full["issues"]] == [
        "first error", "later error", "warning", "later warning", "note"]
    assert full["by_rule"] == {"a": {"errors": 1, "warnings": 1, "info": 1},
                               "b": {"errors": 1, "warnings": 1, "info": 0}}
    for limit in (0, 1, 2, 5, 100):
        limited = report_to_dict(report, limit=limit)
        assert limited["issues"] == full["issues"][:limit]
        assert limited["issues_truncated"] == (limit < 5)
        for key in ("summary", "total_records", "valid_records", "by_rule", "schema_version"):
            assert limited[key] == full[key]
    assert not full["issues_truncated"]
    assert report_to_dict(Report(), limit=0)["by_rule"] == {}
    assert not report_to_dict(Report(), limit=0)["issues_truncated"]


@pytest.mark.parametrize("options", [{"limit": -1}, {"limit": True}, {"limit": 1.5},
                                    {"format": "typo"}, {"disable": ["schema"]},
                                    {"disable": ["typo"]}, {"disable": "pii"}])
def test_bad_options_raise_value_error(options):
    with pytest.raises(ValueError):
        check(EXAMPLES / "clean.jsonl", **options)


@pytest.mark.parametrize("operation", [check, fix])
@pytest.mark.parametrize("alias", ["same", "relative", "symlink", "hardlink", "eval"])
def test_refuse_input_aliases(operation, alias, tmp_path, monkeypatch):
    path = tmp_path / "input.jsonl"
    path.write_text('{"prompt":"Q","completion":"A"}\n')
    before = path.read_bytes()
    options = {}
    if alias == "same":
        output = path
    elif alias == "relative":
        monkeypatch.chdir(tmp_path)
        output = Path("input.jsonl")
    elif alias == "eval":
        output = tmp_path / "eval.jsonl"
        output.write_bytes(before)
        options["eval_path"] = output
    else:
        output = tmp_path / "alias.jsonl"
        if alias == "symlink":
            output.symlink_to(path)
        else:
            output.hardlink_to(path)
    with pytest.raises(ValueError, match="output must be different"):
        operation(path, output=output, **options)
    if operation is fix:
        with pytest.raises(ValueError):
            fix(path, output=output, dry_run=True, **options)
    assert path.read_bytes() == before
    assert output.read_bytes() == before


@pytest.mark.parametrize("operation", [check, fix])
def test_operational_errors_are_single_exception_type(operation, tmp_path, capsys):
    with pytest.raises(ParallelogramError, match="Could not"):
        operation(tmp_path / "missing")
    invalid = tmp_path / "invalid"
    invalid.write_bytes(b'\xff\xfe\n')
    # Invalid encoded/JSON content remains a data finding, not an I/O failure.
    assert operation(invalid).exit_code == 2
    with pytest.raises(ParallelogramError):
        operation(EXAMPLES / "clean.jsonl", output=tmp_path)
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("fix_mode", [False, True])
def test_cli_operational_error_has_exit_three_and_no_success_json(tmp_path, fix_mode):
    result = runner.invoke(app, ["check", str(EXAMPLES / "clean.jsonl"), "--json", "--output", str(tmp_path)]
                           + (["--fix"] if fix_mode else []))
    assert result.exit_code == 3
    assert result.stdout == ""
    assert "Could not" in result.stderr


def test_api_does_not_import_cli_or_print():
    proc = subprocess.run([sys.executable, "-c", "import sys; from parallelogram import check, fix, rules; "
                           "assert callable(rules); assert 'parallelogram.cli' not in sys.modules; "
                           "assert 'typer' not in sys.modules"], capture_output=True, text=True, timeout=10)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == proc.stderr == ""


def test_rules_are_class_metadata(monkeypatch):
    metadata = rules()
    assert {r["id"] for r in metadata} == {rc.id for rc in registry.all()}
    assert all(set(r) == {"id", "severity", "fixable", "fix_scope", "description", "fix_description"}
               and r["description"] and r["fix_description"] for r in metadata)
    monkeypatch.setattr(registry.get("schema"), "fix_description", "Updated on the class")
    assert next(r for r in api.rules() if r["id"] == "schema")["fix_description"] == "Updated on the class"
    cli = runner.invoke(app, ["rules", "--json"])
    assert cli.exit_code == 0 and json.loads(cli.stdout) == rules()
    terminal = runner.invoke(app, ["rules"])
    assert terminal.exit_code == 0 and "--fix: Updated on the class" in terminal.stdout
    assert "fix_scope=dataset" in terminal.stdout


def test_cli_refuses_in_place_fix(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"prompt":"Q","completion":"A"}\n')
    before = path.read_bytes()
    cli = runner.invoke(app, ["check", str(path), "--fix", "--output", str(path)])
    assert cli.exit_code == 2
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", [check, fix])
@pytest.mark.parametrize("disable", [(), ("eval-overlap",)])
def test_missing_eval_is_operational_error_even_for_empty_train(operation, disable, tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    with pytest.raises(ParallelogramError, match="missing-eval"):
        operation(path, eval_path=tmp_path / "missing-eval", disable=disable)


def test_failed_atomic_write_preserves_previous_output(tmp_path, monkeypatch):
    from parallelogram.core import io

    output = tmp_path / "clean.jsonl"
    output.write_text("previous output\n")

    def fail_replace(*args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(io.os, "replace", fail_replace)
    with pytest.raises(ParallelogramError, match="replacement failure"):
        fix(EXAMPLES / "clean.jsonl", output=output)
    assert output.read_text() == "previous output\n"
    assert list(tmp_path.iterdir()) == [output]


def test_dry_run_preserves_existing_output(tmp_path):
    output = tmp_path / "existing.jsonl"
    output.write_text("keep me\n")
    fix(EXAMPLES / "broken.jsonl", output=output, dry_run=True)
    assert output.read_text() == "keep me\n"


@pytest.mark.parametrize("args", [["--limit", "-1"], ["--fix", "--limit", "1"], ["--dry-run"]])
def test_incompatible_or_negative_cli_options(args):
    result = runner.invoke(app, ["check", str(EXAMPLES / "clean.jsonl"), "--json", *args])
    assert result.exit_code == 2


def test_default_api_is_offline(tmp_path, monkeypatch):
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("unexpected network access")

    monkeypatch.setattr(socket, "socket", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    assert check(EXAMPLES / "clean.jsonl").exit_code == 0
    assert fix(EXAMPLES / "clean.jsonl", output=tmp_path / "out").exit_code == 0
    assert rules()
