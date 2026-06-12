"""Duplicate detection rule.

Catches exact-content duplicate conversations. Even small fractions of dupes
in fine-tuning datasets shift the model toward memorization on the duplicated
examples, producing a model that overfits to those exact phrasings.

Normalization: messages are reduced to {role, content.strip()} before hashing
so trivial whitespace differences don't mask true duplicates.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable

from ..core.rules import Rule, registry
from ..core.report import Issue, Severity


@registry.register
class DuplicatesRule(Rule):
    id = "duplicates"
    description = "Exact-content duplicates cause memorization instead of generalization"
    severity = Severity.ERROR
    fixable = True  # mechanical: drop all but the first occurrence

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self._seen: dict[str, list[int]] = {}

    def reset(self) -> None:
        self._seen = {}

    def _hash(self, record: Any) -> str:
        if not isinstance(record, dict):
            return ""
        messages = record.get("messages", [])
        if not isinstance(messages, list):
            return ""
        try:
            normalized = [
                {
                    "role": m.get("role"),
                    "content": (m.get("content") or "").strip()
                    if isinstance(m.get("content"), str) else m.get("content"),
                }
                for m in messages
                if isinstance(m, dict)
            ]
            canon = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        except (TypeError, ValueError):
            return ""
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        h = self._hash(record)
        if not h:
            return []
        self._seen.setdefault(h, []).append(line_no)
        return []

    def finalize(self) -> Iterable[Issue]:
        for lines in self._seen.values():
            if len(lines) > 1:
                first = lines[0]
                for dup_line in lines[1:]:
                    yield Issue(
                        rule_id=self.id,
                        severity=Severity.ERROR,
                        line_no=dup_line,
                        message=f"Duplicate of line {first}",
                        detail=f"{len(lines)} copies total at lines: {lines}",
                        fixable=True,
                        context={"duplicate_of": first, "all_lines": lines},
                    )

    @property
    def clusters(self) -> list[list[int]]:
        """Duplicate clusters found in the last run: one list of line
        numbers per repeated record (first occurrence included). Read this
        *before* a fix pass — Fixer re-validation resets rule state."""
        return [lines for lines in self._seen.values() if len(lines) > 1]

    def fix_dataset(self, records: list[tuple[int, Any]]) -> list[tuple[int, Any]]:
        """Keep the first occurrence of each unique record; drop the rest.

        Operates on (line_no, record) pairs. The first occurrence (lowest
        line number for each hash) is retained, everything else dropped.
        """
        kept: list[tuple[int, Any]] = []
        seen_hashes: set[str] = set()
        for line_no, record in records:
            h = self._hash(record)
            if not h:
                # Unhashable / malformed — leave it; another rule will handle.
                kept.append((line_no, record))
                continue
            if h in seen_hashes:
                continue  # drop this duplicate
            seen_hashes.add(h)
            kept.append((line_no, record))
        return kept
