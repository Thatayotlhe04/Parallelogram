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
        # Schema/role diagnostics may quote malformed values (a key or an email
        # address used as a role). Scrub centrally, even with the secrets and
        # pii rules disabled. Their own messages are already masked.
        self.message = scrub(self.message)
        if self.detail is not None:
            self.detail = scrub(self.detail)


def scrub(text: str) -> str:
    """Redact secrets, then personal data, from any text shown to a user."""
    from ..rules.pii import redact as redact_pii
    from ..rules.secrets import redact as redact_secrets

    return redact_pii(redact_secrets(text))


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
