"""Workbench bridge: the Parallelogram engine behind the browser workbench.

The workbench at parallelogram.dev/workbench runs this file inside Pyodide, in
a worker, next to an unmodified copy of the ``parallelogram`` package. Nothing
here re-implements a rule or a fix: every verdict comes from ``Runner``, every
mechanical repair from ``api.fix`` and every gate copy from ``api.check``, so
the page and the CLI cannot disagree. The bridge only adds what a CLI has no
use for: a stack of reversible patches over one dataset, and the bookkeeping
that keeps every row tied to its line in the file the user opened.

Row ids are physical line numbers of the opened file, so "line 2131" here is
the line 2131 the CLI prints.

The same module runs under CPython, which is how ``cli/tests`` exercises it.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import parallelogram
from parallelogram import api
from parallelogram.core.fixer import Disposition, Fixer
from parallelogram.core.report import Issue, Severity, scrub
from parallelogram.core.rules import registry
from parallelogram.core.runner import Runner
from parallelogram.formats import get_parser
from parallelogram.rules.context_window import ContextWindowRule
from parallelogram.rules.duplicates import record_hash
from parallelogram.rules.pii import find_pii
from parallelogram.rules.secrets import find_secrets

RULE_IDS = [rc.id for rc in registry.all()]
RULE_BIT = {rid: 1 << i for i, rid in enumerate(RULE_IDS)}
PATCH_KINDS = {"fix_all", "fix", "drop_rule", "drop", "edit", "repair"}

# Row states in a snapshot, one per record of the opened file.
CLEAN, WARN, ERROR, GONE = 0, 1, 2, 3


def _line(text: str) -> bytes:
    """One row as it is written to disk: a single physical line."""
    try:
        text = json.dumps(json.loads(text), ensure_ascii=False)
    except (ValueError, RecursionError):
        # Not JSON: keep what was typed, on one line, for the schema rule to report.
        text = " ".join(text.splitlines())
    return text.encode("utf-8") + b"\n"


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").rstrip("\r\n")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass
class Analysis:
    """One full validation pass over a stage, keyed by row id."""
    exit_code: int
    errors: int
    warnings: int
    total: int
    valid: int
    issues: dict[int, list[Issue]]
    records: dict[int, Any]
    formats: dict[int, str]
    unparseable: set[int]
    tokens: dict[int, int]
    hashes: dict[str, int]
    notes: list[str]
    token_method: str
    token_exact: bool
    record_ids: list[int] = field(default_factory=list)
    # The pass itself, kept so the dry fix reuses it instead of validating twice.
    run: tuple = ()
    slim: bool = False

    def release(self) -> None:
        """Keep the verdict, let go of the parsed records. Only the stage on
        screen needs every record in memory; earlier stages need their counts
        and findings, and are re-read from their lines if they come back."""
        self.records, self.hashes, self.run, self.slim = {}, {}, (), True


@dataclass
class Stage:
    """The dataset after a prefix of the ledger."""
    lines: list[tuple[int, bytes]]
    touched: dict[int, int] = field(default_factory=dict)   # row id -> patch number
    gone: dict[int, int] = field(default_factory=dict)      # row id -> patch number
    last: dict[int, bytes] = field(default_factory=dict)    # removed rows, as last seen
    analysis: Analysis | None = None
    fix: Any = None                                          # dry FixReport, lazy
    effect: dict[str, Any] = field(default_factory=dict)    # what the patch did
    _by_id: dict[int, bytes] | None = None
    _line_of: dict[int, int] | None = None

    @property
    def by_id(self) -> dict[int, bytes]:
        if self._by_id is None:
            self._by_id = dict(self.lines)
        return self._by_id

    def line_of(self, rid: int) -> int | None:
        if self._line_of is None:
            self._line_of = {r: n for n, (r, _) in enumerate(self.lines, start=1)}
        return self._line_of.get(rid)


class Bench:
    def __init__(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="pgwb-"))
        self.name = ""
        self.eval_name: str | None = None
        self.base = b""
        self.eval_bytes: bytes | None = None
        self.max_seq_len = 4096
        self.disable: list[str] = []
        self.patches: list[dict] = []
        self._stages: list[tuple[str, Stage]] = []
        self._judges: list | None = None     # rule instances for validate(), built once per setting

    # ── loading ────────────────────────────────────────────────────────
    def load(self, name: str, data: bytes, eval_name: str | None = None,
             eval_data: bytes | None = None) -> None:
        self.name = os.path.basename(name) or "train.jsonl"
        self.base = bytes(data)
        self.eval_name = os.path.basename(eval_name) if eval_name else None
        self.eval_bytes = bytes(eval_data) if eval_data is not None else None
        self.patches = []
        self._stages = []
        self._judges = None

    def set_eval(self, eval_name: str | None, eval_data: bytes | None) -> None:
        self.eval_name = os.path.basename(eval_name) if eval_name else None
        self.eval_bytes = bytes(eval_data) if eval_data is not None else None
        self._stages = []
        self._judges = None

    def set_options(self, max_seq_len: int | None = None, disable: list[str] | None = None) -> None:
        if max_seq_len is not None:
            if type(max_seq_len) is not int or max_seq_len < 1:
                raise ValueError("max_seq_len must be a positive integer")
            self.max_seq_len = max_seq_len
        if disable is not None:
            api._validate_options("auto", disable, None)
            self.disable = sorted(set(disable))
        self._stages = []
        self._judges = None

    @property
    def _eval_path(self) -> Path:
        # Named as the user named it: eval-overlap messages quote the basename.
        return self.dir / "eval" / (self.eval_name or "eval.jsonl")

    @property
    def _eval(self) -> str | None:
        if self.eval_bytes is None:
            return None
        path = self._eval_path
        path.parent.mkdir(exist_ok=True)
        if not path.exists() or path.read_bytes() != self.eval_bytes:
            for old in path.parent.iterdir():
                old.unlink()
            path.write_bytes(self.eval_bytes)
        return str(path)

    def _train_path(self) -> Path:
        d = self.dir / "train"
        d.mkdir(exist_ok=True)
        return d / self.name

    def _write(self, stage: Stage) -> tuple[Path, list[int]]:
        """Write a stage to disk; return its path and line number -> row id."""
        path = self._train_path()
        path.write_bytes(b"".join(raw if raw.endswith(b"\n") else raw + b"\n" for _, raw in stage.lines))
        return path, [rid for rid, _ in stage.lines]

    def _rules(self, disable: list[str] | None = None):
        return api._build_rules(set(self.disable if disable is None else disable), None,
                                self.max_seq_len, self._eval, "auto")

    # ── validation ─────────────────────────────────────────────────────
    def _analyze(self, stage: Stage) -> Analysis:
        if stage.analysis is not None and not stage.analysis.slim:
            return stage.analysis
        path, ids = self._write(stage)
        rules = self._rules()
        runner = Runner(rules, dataset_format="auto")
        report, parsed, _clean, unparseable = runner.run_with_records(str(path))
        result = api.CheckResult(report, str(path), self.disable, _clean)

        issues: dict[int, list[Issue]] = {}
        notes: list[str] = []
        for issue in report.issues:
            if issue.line_no is None:
                if issue.severity == Severity.INFO:
                    notes.append(issue.message + (f" ({issue.detail})" if issue.detail else ""))
                continue
            if issue.severity == Severity.INFO:
                continue
            issues.setdefault(ids[issue.line_no - 1], []).append(issue)

        counter = next((r for r in rules if r.id == "context-window"), None) or ContextWindowRule(
            {"tokenizer": None, "max_seq_len": self.max_seq_len})
        exact, method = counter.counter_info()
        records, tokens, hashes = {}, {}, {}
        for line_no, record in parsed:
            rid = ids[line_no - 1]
            records[rid] = record
            count = counter.record_tokens(record)
            if count is not None:
                tokens[rid] = count
            h = record_hash(record)
            if h:
                hashes.setdefault(h, rid)

        stage.analysis = Analysis(
            exit_code=result.exit_code, errors=len(report.errors), warnings=len(report.warnings),
            total=report.total_records, valid=report.valid_records, issues=issues, records=records,
            formats={ids[ln - 1]: fmt for ln, fmt in runner.source_formats.items()},
            unparseable={ids[ln - 1] for ln in unparseable}, tokens=tokens, hashes=hashes,
            notes=notes, token_method=method, token_exact=exact,
            record_ids=sorted(set(records) | {ids[ln - 1] for ln in unparseable}),
            run=(str(path), ids, rules, report, parsed, unparseable))
        return stage.analysis

    def _dry_fix(self, stage: Stage):
        """What ``check --fix`` would do to this stage, without writing.

        Built the way ``parallelogram report`` builds its projection: the
        fixer runs over the validation pass already made for this stage.
        """
        if stage.fix is None:
            path, ids, rules, report, parsed, unparseable = self._analyze(stage).run
            fr = Fixer(rules).fix(parsed, report.issues, unparseable)
            stage.fix = (api.FixResult(fr, path, self.disable, True), ids)
        return stage.fix

    # ── patches ────────────────────────────────────────────────────────
    def _engine_fix(self, stage: Stage, disable: list[str]):
        """Run the engine's own fixer into a scratch file; map output to row ids."""
        path, ids = self._write(stage)
        out = self.dir / "fixed.jsonl"
        result = api.fix(str(path), output=str(out), eval_path=self._eval, disable=disable,
                         max_seq_len=self.max_seq_len)
        fr = result.report
        written = out.read_bytes().splitlines(keepends=True)
        if fr.unchanged == fr.total_records and not fr.fixer_errors:
            fixed = dict(zip(ids, written))          # byte copy of the input
        else:
            fixed = {ids[ln - 1]: raw for (ln, _), raw in zip(fr.clean_records, written)}
        outcomes = {ids[o.line_no - 1]: o for o in fr.outcomes}
        return fixed, outcomes, fr.fixer_errors

    def _apply(self, prev: Stage, patch: dict, number: int) -> Stage:
        kind = patch["kind"]
        before = dict(prev.lines)
        after: dict[int, bytes | None] = {}

        if kind == "fix_all":
            # Exactly `check --fix`: same fixer, same writer, same drops.
            fixed, outcomes, errors = self._engine_fix(prev, self.disable)
            records = set(outcomes)
            for rid, raw in prev.lines:
                if rid in records:
                    after[rid] = fixed.get(rid)
                elif not raw.strip():
                    after[rid] = None          # blank lines are not records; --fix omits them
            effect = {"fixer_errors": errors}
        elif kind == "fix":
            rule = patch["rule"]
            cls = registry.get(rule)
            if cls is None or not cls.fixable:
                raise ValueError(f"rule {rule!r} has no mechanical fix")
            # The engine's fixer with only this rule (and schema, which cannot be
            # turned off) enabled. A row schema would drop is left as it is:
            # this patch is about one rule, and the gate still sees the row.
            others = [r for r in RULE_IDS if r not in (rule, "schema")]
            fixed, outcomes, errors = self._engine_fix(prev, others)
            for rid, o in outcomes.items():
                if o.disposition == Disposition.FIXED:
                    after[rid] = fixed[rid]
                elif o.disposition == Disposition.DROPPED and "schema" not in o.rules_unfixable:
                    after[rid] = None
            effect = {"fixer_errors": errors}
        elif kind == "drop_rule":
            rule = patch["rule"]
            if rule not in RULE_BIT:
                raise ValueError(f"unknown rule {rule!r}")
            hit = [rid for rid, issues in prev.analysis.issues.items()
                   if any(i.rule_id == rule and i.severity == Severity.ERROR for i in issues)]
            after = {rid: None for rid in hit}
            effect = {}
        elif kind == "drop":
            after = {int(rid): None for rid in patch["rows"] if int(rid) in before}
            effect = {}
        elif kind in ("edit", "repair"):
            after = {int(rid): _line(text) for rid, text in patch["rows"].items() if int(rid) in before}
            effect = {}
        else:
            raise ValueError(f"unknown patch kind {kind!r}")

        stage = Stage(lines=[], touched=dict(prev.touched), gone=dict(prev.gone), last=dict(prev.last))
        changed = dropped = 0
        for rid, raw in prev.lines:
            if rid not in after:
                stage.lines.append((rid, raw))
                continue
            new = after[rid]
            if new is None:
                stage.gone[rid] = number
                stage.last[rid] = raw
                dropped += bool(raw.strip())
            elif new.rstrip(b"\r\n") != raw.rstrip(b"\r\n"):
                stage.lines.append((rid, new))
                stage.touched[rid] = number
                changed += 1
            else:
                stage.lines.append((rid, raw))
        stage.effect = {**effect, "changed": changed, "dropped": dropped}
        return stage

    def set_patches(self, patches: list[dict]) -> None:
        for p in patches:
            if p.get("kind") not in PATCH_KINDS:
                raise ValueError(f"unknown patch kind {p.get('kind')!r}")
        self.patches = patches

    def _base_stage(self) -> Stage:
        return Stage(lines=[(i, raw) for i, raw in enumerate(self.base.splitlines(keepends=True), start=1)])

    def stages(self) -> list[Stage]:
        """Stage 0 is the opened file; stage k is after the k-th enabled patch.
        Unchanged ledger prefixes are served from cache."""
        active = [p for p in self.patches if p.get("enabled", True)]
        keys = ["base"] + [json.dumps({k: v for k, v in p.items() if k not in ("enabled", "label")},
                                      sort_keys=True) for p in active]
        keep = 0
        while keep < len(keys) and keep < len(self._stages) and self._stages[keep][0] == keys[keep]:
            keep += 1
        self._stages = self._stages[:keep]
        for i in range(keep, len(keys)):
            stage = self._base_stage() if i == 0 else self._apply(self._stages[i - 1][1], active[i - 1], i)
            self._analyze(stage)
            self._stages.append((keys[i], stage))
        for _, stage in self._stages[:-1]:
            if not stage.analysis.slim:
                stage.analysis.release()
                stage.fix = None
        self._analyze(self._stages[-1][1])      # the stage on screen is always whole
        return [s for _, s in self._stages]

    # ── what the page asks for ─────────────────────────────────────────
    def snapshot(self) -> dict:
        stages = self.stages()
        base, cur = stages[0], stages[-1]
        a0, a = base.analysis, cur.analysis
        ids = a0.record_ids
        state, mask, tokens, gone, touched = [], [], [], [], []
        for rid in ids:
            issues = a.issues.get(rid, [])
            bits = 0
            for i in issues:
                bits |= RULE_BIT[i.rule_id]
            if rid in cur.gone:
                state.append(GONE)
            elif any(i.severity == Severity.ERROR for i in issues):
                state.append(ERROR)
            else:
                state.append(WARN if issues else CLEAN)
            mask.append(bits)
            tokens.append(a.tokens.get(rid, a0.tokens.get(rid, 0)))
            gone.append(cur.gone.get(rid, 0))
            touched.append(cur.touched.get(rid, 0))

        classes = []
        for meta in api.rules():
            rid_ = meta["id"]
            rows = [r for r, issues in a.issues.items() if any(i.rule_id == rid_ for i in issues)]
            if not rows:
                continue
            errs = sum(1 for r in rows for i in a.issues[r] if i.rule_id == rid_ and i.severity == Severity.ERROR)
            warns = sum(1 for r in rows for i in a.issues[r] if i.rule_id == rid_ and i.severity == Severity.WARNING)
            classes.append({**meta, "bit": RULE_BIT[rid_], "errors": errs, "warnings": warns, "rows": len(rows)})

        dup, leak = {}, {}
        for rid, issues in a.issues.items():
            for i in issues:
                if i.rule_id == "duplicates" and "duplicate_of" in i.context:
                    dup[rid] = ids_at(cur, i.context["duplicate_of"])
                elif i.rule_id == "eval-overlap" and "eval_line" in i.context:
                    leak[rid] = i.context["eval_line"]

        active = [p for p in self.patches if p.get("enabled", True)]
        ledger, k = [], 0
        for p in self.patches:
            entry = {"id": p.get("id"), "kind": p["kind"], "rule": p.get("rule"), "enabled": p.get("enabled", True),
                     "label": p.get("label"), "model": p.get("model"), "rows": len(p["rows"]) if "rows" in p else None}
            if entry["enabled"]:
                k += 1
                s = stages[k]
                entry.update(number=k, exit=s.analysis.exit_code, errors=s.analysis.errors,
                             warnings=s.analysis.warnings, valid=s.analysis.valid, total=s.analysis.total, **s.effect)
            ledger.append(entry)

        return {
            "engine": parallelogram.__version__,
            "file": self.name, "eval": self.eval_name, "bytes": len(self.base),
            "eval_records": sum(1 for ln in (self.eval_bytes or b"").splitlines() if ln.strip()),
            "options": {"max_seq_len": self.max_seq_len, "disable": self.disable},
            "rules": [{**m, "bit": RULE_BIT[m["id"]]} for m in api.rules()],
            "ids": ids, "state": state, "mask": mask, "tokens": tokens, "gone": gone, "touched": touched,
            "base": _counts(a0), "stage": _counts(a),
            "classes": classes,
            "links": {"duplicates": dup, "eval": leak},
            "ledger": ledger, "active_patches": len(active),
            "notes": a.notes, "tokens_info": {"exact": a.token_exact, "method": a.token_method},
            "cli": self.cli_equivalent(),
        }

    def plan(self) -> dict:
        """What ``check --fix`` would do to the current stage, rule by rule.

        Separate from ``snapshot`` because it costs a second pass (the fixer
        re-validates everything it emits); the page paints first and asks after.
        """
        cur = self.stages()[-1]
        a = cur.analysis
        fix_result, fix_ids = self._dry_fix(cur)
        fr = fix_result.report
        outcomes = {fix_ids[o.line_no - 1]: o for o in fr.outcomes}
        by_rule = {}
        for rule in RULE_IDS:
            rows = [r for r, issues in a.issues.items() if any(i.rule_id == rule for i in issues)]
            if rows:
                by_rule[rule] = {
                    "would_repair": sum(1 for r in rows if outcomes[r].disposition == Disposition.FIXED),
                    "would_drop": sum(1 for r in rows if outcomes[r].disposition in
                                      (Disposition.DROPPED, Disposition.UNPARSEABLE))}
        # Rows `--fix` drops for want of a mechanical repair: the data a careful
        # repair can still save. Rows dropped as duplicates or eval leaks are not
        # in this list; dropping them is the repair.
        unrepairable = sorted(r for r, o in outcomes.items()
                              if o.disposition == Disposition.UNPARSEABLE
                              or (o.disposition == Disposition.DROPPED and o.rules_unfixable))
        return {"projection": {**fix_result.to_dict()["summary"], "exit": fix_result.exit_code,
                               "fixes_by_rule": fr.fixes_by_rule, "fixer_errors": fr.fixer_errors},
                "by_rule": by_rule, "unrepairable": unrepairable,
                "dropped": {r: sorted(set(o.rules_unfixable or o.rules_fixed)) for r, o in outcomes.items()
                            if o.disposition in (Disposition.DROPPED, Disposition.UNPARSEABLE)}}

    def peek(self, rid: int) -> dict:
        """A record in two lines, for the card that follows the pointer."""
        cur = self.stages()[-1]
        a = cur.analysis
        raw = cur.by_id.get(rid) or cur.last.get(rid)
        record = a.records.get(rid)
        first = ""
        if isinstance(record, dict) and isinstance(record.get("messages"), list):
            for msg in record["messages"]:
                if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                    first = msg["content"]
                    break
        if not first and raw is not None:
            first = _text(raw)
        issues = a.issues.get(rid, [])
        return {"id": rid, "first": scrub(first[:220]),
                "why": [f"{i.rule_id}: {i.message}" for i in issues[:2]], "more": max(0, len(issues) - 2)}

    def row(self, rid: int) -> dict:
        stages = self.stages()
        base, cur = stages[0], stages[-1]
        a = cur.analysis
        now = cur.by_id.get(rid)
        original = base.by_id.get(rid)
        if original is None:
            raise KeyError(f"no row {rid}")
        gone = cur.gone.get(rid)
        shown = now if now is not None else cur.last.get(rid, original)
        record = a.records.get(rid)
        if record is None and gone:
            record = self._parse(_text(shown))[0]
        line = cur.line_of(rid)
        return {
            "id": rid, "line": line, "raw": _text(shown), "base": _text(original),
            "changed": cur.touched.get(rid, 0), "gone": gone or 0,
            "format": a.formats.get(rid), "record": record,
            "issues": [_issue(i) for i in a.issues.get(rid, [])] if not gone else [],
            "spans": _spans(record), "tokens": a.tokens.get(rid),
        }

    def _parse(self, text: str):
        path = self.dir / "one.jsonl"
        path.write_bytes(_line(text))
        parsed = next(iter(get_parser("auto")(str(path))), None)
        if parsed is None:
            return None, "Empty line", None
        return parsed.record, parsed.parse_error, parsed.source_format

    def validate(self, rid: int, text: str) -> dict:
        """Judge one candidate row with every enabled rule, without applying it.

        Duplicates are decided on the whole dataset, so the candidate is
        compared against the content hashes of the current stage instead.
        """
        cur = self.stages()[-1]
        record, error, fmt = self._parse(text)
        issues: list[Issue] = []
        if error is not None:
            issues.append(Issue("schema", Severity.ERROR, None, "Invalid JSON", error))
        else:
            if self._judges is None:
                self._judges = self._rules()
            for rule in self._judges:
                rule.reset()
                issues += [i for i in rule.check_record(record, 1) if i.severity != Severity.INFO]
            if "duplicates" not in self.disable:
                first = cur.analysis.hashes.get(record_hash(record))
                if first is not None and first != rid:
                    issues.append(Issue("duplicates", Severity.ERROR, None, f"Duplicate of line {first}",
                                        fixable=True))
        errors = sum(i.severity == Severity.ERROR for i in issues)
        return {"id": rid, "record": record, "format": fmt, "spans": _spans(record),
                "issues": [_issue(i) for i in issues], "errors": errors,
                "warnings": len(issues) - errors, "line": _text(_line(text))}

    def search(self, needle: str) -> list[int]:
        """Row ids whose current text contains ``needle`` (case-insensitive)."""
        low = needle.lower().encode("utf-8")
        return [rid for rid, raw in self.stages()[-1].lines if low in raw.lower()]

    # ── operator: model-proposed repairs, judged by the engine ─────────
    def outbound(self, ids: list[int]) -> list[dict]:
        """What may leave the tab for a model: rows with every credential and
        personal detail the engine can find already replaced by placeholders,
        and the engine's (already masked) diagnostics for each row."""
        cur = self.stages()[-1]
        lines = cur.by_id
        out = []
        for rid in ids:
            if rid not in lines:
                continue
            out.append({"id": rid, "row": redact_row(_text(lines[rid])),
                        "findings": [_outbound_finding(i) for i in cur.analysis.issues.get(rid, [])]})
        return out

    def verify(self, proposals: dict) -> list[dict]:
        """Check proposed replacement rows. A proposal passes only if the engine
        finds no error in it; how much of the original text survived is
        reported next to the verdict so invention is visible."""
        cur = self.stages()[-1]
        lines = cur.by_id
        results = []
        for rid, text in proposals.items():
            rid = int(rid)
            verdict = self.validate(rid, text)
            before = _words(redact_row(_text(lines[rid]))) if rid in lines else []
            after = _words(verdict["line"])
            kept, added = _overlap(before, after)
            results.append({**verdict, "ok": verdict["errors"] == 0, "kept": kept, "added": added,
                            "sent": redact_row(_text(lines[rid])) if rid in lines else ""})
        return results

    # ── export ─────────────────────────────────────────────────────────
    def cli_equivalent(self) -> dict:
        """The command that reproduces the current ledger, when one exists."""
        active = [p for p in self.patches if p.get("enabled", True)]
        flags = (f" --eval {self.eval_name}" if self.eval_name else "") + \
                (f" --max-seq-len {self.max_seq_len}" if self.max_seq_len != 4096 else "") + \
                (f" --disable {','.join(self.disable)}" if self.disable else "")
        stem = self.name.removesuffix(".jsonl")
        check = f"parallelogram check {self.name}{flags}"
        if not active:
            return {"exact": True, "command": check}
        if [p["kind"] for p in active] == ["fix_all"]:
            return {"exact": True, "command": f"{check} --fix -o {stem}.clean.jsonl"}
        return {"exact": False, "command": f"parallelogram check {stem}.workbench.jsonl{flags}"}

    def export(self, what: str) -> bytes:
        stages = self.stages()
        cur = stages[-1]
        active = [p for p in self.patches if p.get("enabled", True)]
        if what == "working":
            if not active:
                return self.base
            return b"".join(raw if raw.endswith(b"\n") else raw + b"\n" for _, raw in cur.lines)
        if what == "gate":
            # Only the rows the gate lets through: `check --output`, verbatim.
            path, _ = self._write(cur)
            out = self.dir / "gate.jsonl"
            api.check(str(path), eval_path=self._eval, disable=self.disable,
                      max_seq_len=self.max_seq_len, output=str(out))
            return out.read_bytes()
        if what == "recipe":
            base = stages[0]
            before = dict(base.lines)
            entries, k = [], 0
            for p in self.patches:
                if not p.get("enabled", True):
                    continue
                k += 1
                s = stages[k]
                entry = {"step": k, "kind": p["kind"], "changed": s.effect.get("changed", 0),
                         "dropped": s.effect.get("dropped", 0), "exit_after": s.analysis.exit_code}
                for key in ("rule", "model", "endpoint"):
                    if p.get(key):
                        entry[key] = p[key]
                if p["kind"] == "drop":
                    entry["lines"] = sorted(int(r) for r in p["rows"])
                if p["kind"] in ("edit", "repair"):
                    entry["rows"] = [{"line": int(r), "sha256_before": _sha(before[int(r)].rstrip(b"\r\n")),
                                      "text": _text(_line(t))} for r, t in sorted(
                                          p["rows"].items(), key=lambda kv: int(kv[0])) if int(r) in before]
                entries.append(entry)
            recipe = {
                "recipe_version": 1,
                "parallelogram": parallelogram.__version__,
                "source": {"file": self.name, "sha256": _sha(self.base), "bytes": len(self.base)},
                "eval": None if self.eval_bytes is None else {"file": self.eval_name, "sha256": _sha(self.eval_bytes)},
                "options": {"max_seq_len": self.max_seq_len, "disable": self.disable, "tokenizer": None},
                "steps": entries,
                "result": {"sha256": _sha(self.export("working")), **_counts(cur.analysis)},
                "cli": self.cli_equivalent(),
            }
            return (json.dumps(recipe, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        raise ValueError(f"unknown export {what!r}")


def ids_at(stage: Stage, line_no: int) -> int:
    return stage.lines[line_no - 1][0]


def _counts(a: Analysis) -> dict:
    return {"exit": a.exit_code, "errors": a.errors, "warnings": a.warnings, "total": a.total, "valid": a.valid}


def _issue(i: Issue) -> dict:
    return {"rule": i.rule_id, "severity": i.severity.value, "message": i.message, "detail": i.detail,
            "fixable": i.fixable, "context": i.context}


def _spans(record: Any) -> dict:
    """Where the secrets and pii rules match inside each message's text, so the
    page can underline the characters the finding is about."""
    out: dict[str, list] = {}
    if not isinstance(record, dict):
        return out
    for key in ("messages", "rejected"):
        messages = record.get(key)
        if not isinstance(messages, list):
            continue
        for index, msg in enumerate(messages):
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, str):
                continue
            found = [[m.start(), m.end(), "secrets", kind] for kind, m in find_secrets(content)]
            found += [[m.start(), m.end(), "pii", kind] for kind, m in find_pii(content)]
            if found:
                out[f"{key}:{index}"] = sorted(found)
    return out


def _outbound_finding(issue: Issue) -> str:
    """A finding as it may be sent to a model.

    The CLI's secrets and pii messages end in a partial mask of the value
    (the last four digits of a number, the first letter of an address, the
    first characters of a key) so a person can find it. A model needs none of
    that: the row it gets already carries a placeholder. The mask is cut, and
    the rest is scrubbed in case a message quotes record text.
    """
    message = issue.message
    if issue.rule_id in ("secrets", "pii"):
        message = message.rsplit(": ", 1)[0]
    return f"{issue.rule_id}: {scrub(message)}"


def _scrub_all(value: Any) -> Any:
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {(scrub(k) if isinstance(k, str) else k): _scrub_all(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_all(v) for v in value]
    return value


def redact_row(text: str) -> str:
    """A row with the engine's secret and pii redaction applied to every string."""
    try:
        return json.dumps(_scrub_all(json.loads(text)), ensure_ascii=False)
    except (ValueError, RecursionError):
        return scrub(text)


def _words(text: str) -> list[str]:
    try:
        strings: list[str] = []
        stack = [json.loads(text)]
        while stack:
            v = stack.pop()
            if isinstance(v, str):
                strings.append(v)
            elif isinstance(v, dict):
                stack.extend(v.values())
            elif isinstance(v, list):
                stack.extend(v)
        return " ".join(strings).split()
    except (ValueError, RecursionError):
        return text.split()


def _overlap(before: list[str], after: list[str]) -> tuple[float, float]:
    """(share of the original words still present, share of the proposal that is new),
    as multisets: order changes are not counted as loss."""
    from collections import Counter
    b, a = Counter(before), Counter(after)
    common = sum((b & a).values())
    return (round(common / len(before), 4) if before else 1.0,
            round(1 - common / len(after), 4) if after else 0.0)


# ── module-level entry points the worker calls (JSON in, JSON out) ─────
BENCH = Bench()


def call(name: str, payload: str = "{}") -> str:
    args = json.loads(payload)
    b = BENCH
    if name == "open":
        train = Path(args["train_path"]).read_bytes()
        ev = Path(args["eval_path"]).read_bytes() if args.get("eval_path") else None
        b.load(args["train_name"], train, args.get("eval_name"), ev)
        return json.dumps(b.snapshot())
    if name == "demo":
        import demo_dataset
        train, ev = demo_dataset.build()
        b.load("train.jsonl", demo_dataset.dumps(train).encode("utf-8"),
               "eval.jsonl", demo_dataset.dumps(ev).encode("utf-8"))
        return json.dumps(b.snapshot())
    if name == "eval":
        b.set_eval(args.get("eval_name"), Path(args["eval_path"]).read_bytes() if args.get("eval_path") else None)
        return json.dumps(b.snapshot())
    if name == "options":
        b.set_options(args.get("max_seq_len"), args.get("disable"))
        return json.dumps(b.snapshot())
    if name == "patches":
        previous = b.patches
        b.set_patches(args["patches"])
        try:
            return json.dumps(b.snapshot())
        except Exception:
            b.set_patches(previous)
            raise
    if name == "plan":
        return json.dumps(b.plan())
    if name == "peek":
        return json.dumps(b.peek(int(args["id"])))
    if name == "row":
        return json.dumps(b.row(int(args["id"])))
    if name == "validate":
        return json.dumps(b.validate(int(args["id"]), args["text"]))
    if name == "search":
        return json.dumps(b.search(args["text"]))
    if name == "outbound":
        return json.dumps(b.outbound([int(i) for i in args["ids"]]))
    if name == "verify":
        return json.dumps(b.verify(args["proposals"]))
    if name == "export":
        Path(args["out_path"]).write_bytes(b.export(args["what"]))
        return json.dumps({"path": args["out_path"]})
    raise ValueError(f"unknown call {name!r}")
