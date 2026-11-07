"""Exercise the action's actual shell entrypoint, including hostile input strings."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github" / "scripts" / "report.sh"


def environment(tmp_path):
    return {
        **os.environ,
        "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
        "DATASET_PATH": str(ROOT / "cli/examples/clean.jsonl"),
        "DATASET_FORMAT": "auto", "MAX_SEQ_LEN": "4096", "TOKENIZER": "",
        "BASELINE": "", "EVAL_PATH": "", "FAIL_ON": "error",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
    }


@pytest.mark.parametrize("code", [0, 1, 2, 3, 127])
@pytest.mark.parametrize("fail_on", ["error", "warning"])
def test_exit_mapping_and_literal_arguments(tmp_path, code, fail_on):
    stub = tmp_path / "parallelogram"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE'], 'w') as f: json.dump(sys.argv[1:], f)\n"
        "print('# Dataset report')\n"
        "sys.exit(int(os.environ['REPORT_EXIT']))\n"
    )
    stub.chmod(0o755)
    env = environment(tmp_path)
    hostile = "-file with spaces; $(touch INJECTED) `touch INJECTED`\n'\""
    env.update({
        "PATH": str(tmp_path) + os.pathsep + env["PATH"],
        "CAPTURE": str(tmp_path / "args.json"), "REPORT_EXIT": str(code),
        "DATASET_PATH": hostile, "DATASET_FORMAT": hostile,
        "MAX_SEQ_LEN": hostile, "TOKENIZER": hostile, "BASELINE": hostile, "EVAL_PATH": hostile,
        "FAIL_ON": fail_on,
    })
    summary = Path(env["GITHUB_STEP_SUMMARY"])
    summary.write_text("Existing summary\n")
    result = subprocess.run(["bash", str(SCRIPT)], env=env, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == (0 if code == 1 and fail_on == "error" else code), result.stderr
    assert json.loads((tmp_path / "args.json").read_text()) == [
        "report", "--markdown", "--format", hostile, "--max-seq-len", hostile,
        "--tokenizer", hostile, "--baseline", hostile, "--eval", hostile, "--", hostile,
    ]
    assert summary.read_text() == "Existing summary\n# Dataset report\n"
    assert not (tmp_path / "INJECTED").exists()


@pytest.mark.parametrize("kind,fail_on,expected", [
    ("clean", "error", 0), ("broken", "error", 2),
    ("warning", "error", 0), ("warning", "warning", 1),
    ("regression", "error", 3), ("clean", "invalid", 2),
])
def test_action_with_real_cli(tmp_path, kind, fail_on, expected):
    env = environment(tmp_path)
    env["FAIL_ON"] = fail_on
    if kind == "broken":
        env["DATASET_PATH"] = str(ROOT / "cli/examples/broken.jsonl")
    elif kind == "warning":
        path = tmp_path / "warnings.jsonl"
        path.write_text(json.dumps({"prompt": "\ufeffQ", "completion": "A"}) + "\n")
        env["DATASET_PATH"] = str(path)
    elif kind == "regression":
        baseline = tmp_path / "baseline.json"
        baseline.write_text('{"rates": {"errors_per_record": 0.0}}')
        env["BASELINE"] = str(baseline)
        env["DATASET_PATH"] = str(ROOT / "cli/examples/broken.jsonl")
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, check=False)
    assert result.returncode == expected, result.stderr
    if fail_on != "invalid":
        assert "#" in Path(env["GITHUB_STEP_SUMMARY"]).read_text()
