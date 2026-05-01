"""JSON output — machine-readable report for CI pipelines."""
from __future__ import annotations

import json
import sys

from ..core.report import Report


def render_json(
    report: Report,
    file_path: str = "",
    disabled_rules: list[str] | None = None,
) -> None:
    payload = {
        "file": file_path,
        "total_records": report.total_records,
        "valid_records": report.valid_records,
        # disabled_rules is always present (empty list when none disabled)
        # so CI tooling can rely on the field's existence and just check
        # if the list is non-empty to refuse merging.
        "disabled_rules": disabled_rules or [],
        "summary": {
            "errors": len(report.errors),
            "warnings": len(report.warnings),
            "clean": report.is_clean,
        },
        "issues": [
            {
                "rule": i.rule_id,
                "severity": i.severity.value,
                "line": i.line_no,
                "message": i.message,
                "detail": i.detail,
                "fixable": i.fixable,
            }
            for i in report.issues
        ],
    }
    json.dump(payload, sys.stdout, indent=2)
    sys.stdout.write("\n")
