"""Runner — pipes records from a parsed JSONL stream through registered rules."""
from __future__ import annotations

from .report import Report, Issue, Severity
from .rules import Rule
from ..formats import get_parser


class Runner:
    """Coordinates rule execution and clean-record collection.

    Key invariant: a record is included in the clean output only if it raised
    no Severity.ERROR issues across any rule, including post-pass rules like
    duplicates that fire in finalize().
    """

    def __init__(self, rules: list[Rule], dataset_format: str = "auto"):
        self.rules = rules
        self.dataset_format = dataset_format
        self._iter_jsonl = get_parser(dataset_format)
        self.source_formats: dict[int, str] = {}

    def run(self, path: str) -> tuple[Report, list[tuple[int, str]]]:
        """Validate and return (report, clean_raw_lines).

        clean_raw_lines is [(line_no, raw_line)] — used for writing clean
        output without re-serializing the records.
        """
        report, _, clean_raw, _, _ = self._run_full(path)
        return report, clean_raw

    def run_with_records(
        self, path: str
    ) -> tuple[Report, list[tuple[int, dict]], list[tuple[int, str]], set[int]]:
        """Run check and return everything the fixer needs.

        Returns:
            report: the validation report
            parsed_records: [(line_no, record)] for every successfully-parsed line
            clean_raw: [(line_no, raw)] for the lines that passed without errors
            unparseable_lines: set of line numbers that failed JSON parsing
        """
        report, parsed, clean_raw, unparseable, _ = self._run_full(path)
        return report, parsed, clean_raw, unparseable

    def _run_full(self, path: str):
        report = Report()
        parsed_records: list[tuple[int, dict]] = []
        candidate_clean: list[tuple[int, str]] = []
        per_line_errors: dict[int, int] = {}
        unparseable_lines: set[int] = set()
        self.source_formats = {}

        for rule in self.rules:
            rule.reset()

        for parsed in self._iter_jsonl(path):
            report.total_records += 1
            line_no = parsed.line_no
            self.source_formats[line_no] = parsed.source_format or self.dataset_format

            if parsed.parse_error is not None:
                report.add(Issue(
                    rule_id="schema",
                    severity=Severity.ERROR,
                    line_no=line_no,
                    message="Invalid JSON",
                    detail=parsed.parse_error,
                ))
                per_line_errors[line_no] = per_line_errors.get(line_no, 0) + 1
                unparseable_lines.add(line_no)
                continue

            parsed_records.append((line_no, parsed.record))
            errors_before = per_line_errors.get(line_no, 0)
            for rule in self.rules:
                for issue in rule.check_record(parsed.record, line_no):
                    report.add(issue)
                    if issue.severity == Severity.ERROR:
                        per_line_errors[line_no] = per_line_errors.get(line_no, 0) + 1

            if per_line_errors.get(line_no, 0) == errors_before:
                # Tentatively clean — duplicates may flip this in finalize.
                candidate_clean.append((line_no, parsed.raw))

        # Cross-record finalisation pass.
        for rule in self.rules:
            for issue in rule.finalize():
                report.add(issue)
                if issue.severity == Severity.ERROR and issue.line_no is not None:
                    per_line_errors[issue.line_no] = per_line_errors.get(issue.line_no, 0) + 1

        clean = [(ln, raw) for ln, raw in candidate_clean if per_line_errors.get(ln, 0) == 0]
        report.valid_records = len(clean)
        return report, parsed_records, clean, unparseable_lines, per_line_errors
