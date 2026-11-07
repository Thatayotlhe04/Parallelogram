"""JSON output — machine-readable report for CI pipelines."""
from __future__ import annotations

import json
import sys

from ..core.report import Report


def report_to_dict(
    report: Report,
    file_path: str = "",
    disabled_rules: list[str] | None = None,
    limit: int | None = None,
) -> dict:
    if limit is not None and (type(limit) is not int or limit < 0):
        raise ValueError("limit must be a nonnegative integer or None")
    by_rule: dict[str, dict[str, int]] = {}
    severity_keys = {"error": "errors", "warning": "warnings", "info": "info"}
    for issue in report.issues:
        counts = by_rule.setdefault(issue.rule_id, {"errors": 0, "warnings": 0, "info": 0})
        counts[severity_keys[issue.severity.value]] += 1
    order = {"error": 0, "warning": 1, "info": 2}
    issues = sorted(report.issues, key=lambda i: (order[i.severity.value],
                    i.line_no if i.line_no is not None else float("inf")))
    return {
        "schema_version": 1,
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
        "issues_truncated": limit is not None and len(issues) > limit,
        "by_rule": by_rule,
        "issues": [
            {
                "rule": i.rule_id,
                "severity": i.severity.value,
                "line": i.line_no,
                "message": i.message,
                "detail": i.detail,
                "fixable": i.fixable,
            }
            for i in issues[:limit]
        ],
    }


def render_json(
    report: Report,
    file_path: str = "",
    disabled_rules: list[str] | None = None,
    limit: int | None = None,
) -> None:
    json.dump(report_to_dict(report, file_path, disabled_rules, limit), sys.stdout, indent=2)
    sys.stdout.write("\n")
