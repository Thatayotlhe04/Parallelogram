"""Eval-overlap rule: training data must not contain the eval set.

When an eval example (or just its prompt) also appears in the training set,
the model is graded on questions it was trained to answer, and the eval
score overstates how well it generalizes. Splits made by hand, deduplicated
separately, or regenerated from the same seed leak this way.

Active only when ``--eval <path>`` is given. A train record is flagged when
its whole content matches an eval record, or when its prompt (everything
before the final assistant turn) matches an eval prompt. Matching ignores
leading and trailing whitespace, like the duplicates rule. ``--fix`` drops
the overlapping train records; the eval file is never modified.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from typing import Any

from ..core.conversation import prompt_messages
from ..core.report import Issue, Severity
from ..core.rules import Rule, registry
from .duplicates import record_hash


def prompt_hash(record: Any) -> str:
    prompt = prompt_messages(record)
    if not prompt:
        return ""
    canon = json.dumps(
        [{"role": m.get("role"), "content": m.get("content").strip()
          if isinstance(m.get("content"), str) else m.get("content")}
         for m in prompt if isinstance(m, dict)],
        sort_keys=True, ensure_ascii=False,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


@registry.register
class EvalOverlapRule(Rule):
    id = "eval-overlap"
    description = "Training records must not repeat eval records or eval prompts"
    severity = Severity.ERROR
    fixable = True
    fix_scope = "dataset"  # drops whole train records

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.eval_path: str | None = self.config.get("eval_path")
        self.dataset_format: str = self.config.get("dataset_format", "auto")
        self._full: dict[str, int] | None = None
        self._prompts: dict[str, int] = {}

    def _load(self) -> None:
        if self._full is not None:
            return
        from ..formats import get_parser

        self._full, self._prompts = {}, {}
        if not self.eval_path:
            return
        for parsed in get_parser(self.dataset_format)(self.eval_path):
            if parsed.record is None:
                continue
            h = record_hash(parsed.record)
            if h:
                self._full.setdefault(h, parsed.line_no)
            p = prompt_hash(parsed.record)
            if p:
                self._prompts.setdefault(p, parsed.line_no)

    def _match(self, record: Any) -> tuple[str, int] | None:
        self._load()
        h = record_hash(record)
        if h and h in self._full:
            return "full", self._full[h]
        p = prompt_hash(record)
        if p and p in self._prompts:
            return "prompt", self._prompts[p]
        return None

    def check_record(self, record: Any, line_no: int) -> Iterable[Issue]:
        if not self.eval_path:
            return
        hit = self._match(record)
        if hit is None:
            return
        kind, eval_line = hit
        name = os.path.basename(self.eval_path)
        yield Issue(
            rule_id=self.id,
            severity=Severity.ERROR,
            line_no=line_no,
            message=(f"Record also appears in the eval set ({name}:{eval_line})" if kind == "full"
                     else f"Prompt also appears in the eval set ({name}:{eval_line})"),
            detail="Training on eval examples inflates the eval score; --fix drops this train record",
            fixable=True,
            context={"eval_line": eval_line, "match": kind},
        )

    def fix_dataset(self, records: list[tuple[int, Any]]) -> list[tuple[int, Any]]:
        if not self.eval_path:
            return records
        return [(ln, rec) for ln, rec in records if self._match(rec) is None]
