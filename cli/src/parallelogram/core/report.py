"""Issue and Report types — the structured output shape of a validation run."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any


class Severity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class Issue:
    """A single problem found in the dataset."""
    rule_id: str
    severity: Severity
    line_no: Optional[int]
    message: str
    detail: Optional[str] = None
    fixable: bool = False
    context: dict[str, Any] = field(default_factory=dict)


    def __post_init__(self) -> None:
        # Schema/role diagnostics may quote malformed values containing keys.
        # Scrub centrally, including when the secrets rule itself is disabled.
        from ..rules.secrets import redact

        self.message = redact(self.message)
        if self.detail is not None:
            self.detail = redact(self.detail)


@dataclass
class Report:
    """Aggregated result of a validation run."""
    issues: list[Issue] = field(default_factory=list)
    total_records: int = 0
    valid_records: int = 0

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return not self.issues

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)
