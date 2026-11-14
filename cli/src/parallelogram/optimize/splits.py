"""Split loading, immutable record identity, leakage detection, test isolation.

Three splits, three jobs: candidates are what may be trained on, development
examples steer selection and compare policies, and the test split is read
exactly once, after the candidate and recipe are frozen. The code is arranged
so that the last rule is structural rather than a convention:

* the leakage guard sees the test split only as `Fingerprints` (hashes and
  shingle hashes, no text), and hands selection nothing but a set of
  candidate ids to leave out;
* test *examples* come out of `TestVault.open`, which refuses without a valid
  freeze record.

Standard library only, so the guard also runs for `optimize plan` on a
lightweight install.
"""
from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from ..api import _build_rules
from ..core.runner import Runner
from ..rules.duplicates import record_hash
from ..rules.eval_overlap import prompt_hash
from .config import DataSpec, LeakagePolicy, SplitSpec, digest
from .errors import ConfigError, LeakageError, OptimizeError, SplitIsolationError, UnsupportedError

_UNSET = object()


@dataclass(frozen=True)
class Record:
    """One example. `id` never changes for a given source line, so manifests,
    caches and exports can refer to records without carrying their content."""
    id: str
    line_no: int
    raw_sha256: str
    messages: tuple[dict, ...]
    capability: str | None = None
    group: str | None = None
    attrs: dict[str, str] = field(default_factory=dict, compare=False)


@dataclass
class Split:
    name: str
    path: Path
    sha256: str
    records: list[Record]
    warnings_by_rule: dict[str, int] = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        return digest({"sha256": self.sha256, "ids": [r.id for r in self.records]})


def content_fingerprint(records: Iterable[Record]) -> str:
    """Identity of a set of records by id *and* content. An id alone is not
    enough: with `id_field`, a record can be edited and keep its id."""
    return digest([(r.id, r.raw_sha256) for r in records])


def lookup(obj: Any, dotted: str) -> Any:
    """Read `a.b.c` from the raw JSON object; `_UNSET` when any step is missing."""
    for part in dotted.split("."):
        if not isinstance(obj, dict) or part not in obj:
            return _UNSET
        obj = obj[part]
    return obj


def _label(value: Any) -> str | None:
    if value is _UNSET or value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    return str(value)


def _unsupported(record: dict) -> str | None:
    """Why a normalized record cannot be trained as plain chat SFT, if it can't."""
    if "rejected" in record:
        return "preference (chosen/rejected) pairs"
    if "tools" in record or "functions" in record:
        return "tool definitions"
    messages = record["messages"]
    for msg in messages:
        if msg.get("role") in ("tool", "function") or "tool_calls" in msg or "function_call" in msg:
            return "tool calls"
        if not isinstance(msg.get("content"), str):
            return "non-text message content"
    if messages[-1].get("role") != "assistant":
        return "records that do not end with an assistant turn"
    return None


def load_split(spec: SplitSpec, name: str, *, id_field: str | None = None, group_field: str | None = None,
               capability_field: str | None = None, attr_fields: Iterable[str] = ()) -> Split:
    """Parse, validate and identify every record of one split.

    The split must pass the same rules as `parallelogram check`: optimizing
    over data that would break or corrupt training would only select the
    breakage faster. Findings name rules and line numbers, never content.
    """
    path = Path(spec.path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise OptimizeError(f"Could not read the {name} split {path}: {exc}") from exc
    raw_lines = data.splitlines(keepends=True)
    # context-window is skipped here: the backend counts real tokens against
    # model.max_seq_len with the real chat template, which is stricter.
    rules = _build_rules({"context-window"}, None, 4096, None, spec.format)
    runner = Runner(rules, dataset_format=spec.format)
    try:
        report, parsed, _, _ = runner.run_with_records(str(path))
    except (OSError, UnicodeError) as exc:
        raise OptimizeError(f"Could not read the {name} split {path}: {exc}") from exc
    if report.has_errors:
        by_rule: dict[str, int] = {}
        for issue in report.errors:
            by_rule[issue.rule_id] = by_rule.get(issue.rule_id, 0) + 1
        summary = ", ".join(f"{rule}: {count}" for rule, count in sorted(by_rule.items()))
        raise ConfigError(
            f"The {name} split has validation errors ({summary}). Repair it first: "
            f"parallelogram check {path} --fix --output <clean.jsonl>")
    if not parsed:
        raise ConfigError(f"The {name} split {path} has no records")
    warnings: dict[str, int] = {}
    for issue in report.warnings:
        warnings[issue.rule_id] = warnings.get(issue.rule_id, 0) + 1

    records: list[Record] = []
    seen: dict[str, int] = {}
    attr_fields = tuple(attr_fields)
    for line_no, record in parsed:
        reason = _unsupported(record)
        if reason is not None:
            raise UnsupportedError(
                f"The {name} split contains {reason} (first at line {line_no}); this release trains "
                "plain system/user/assistant chat only")
        raw = raw_lines[line_no - 1]
        raw_sha = hashlib.sha256(raw.rstrip(b"\r\n")).hexdigest()
        try:
            source = json.loads(raw)
        except (ValueError, RecursionError) as exc:
            # The parser decodes leniently so the linter can report bad bytes;
            # here the exact source line is needed, so it must decode cleanly.
            raise ConfigError(f"The {name} split has a line that is not valid UTF-8 JSON (line {line_no}); "
                              f"repair it first: parallelogram check {path} --fix --output <clean.jsonl>") from exc
        if id_field is not None:
            value = lookup(source, id_field)
            if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
                raise ConfigError(f"The {name} split has no usable {id_field!r} id at line {line_no}")
            rid = str(value)
        else:
            rid = "sha256:" + raw_sha[:32]
        if rid in seen:
            raise ConfigError(f"The {name} split repeats record id at lines {seen[rid]} and {line_no}; "
                              "ids must be unique")
        seen[rid] = line_no
        records.append(Record(
            id=rid, line_no=line_no, raw_sha256=raw_sha,
            messages=tuple({"role": m["role"], "content": m["content"]} for m in record["messages"]),
            capability=_label(lookup(source, capability_field)) if capability_field else None,
            group=_label(lookup(source, group_field)) if group_field else None,
            attrs={f: label for f in attr_fields if (label := _label(lookup(source, f))) is not None},
        ))
    return Split(name=name, path=path, sha256=hashlib.sha256(data).hexdigest(), records=records,
                 warnings_by_rule=warnings)


# ── fingerprints: what one split may reveal to the guard ────────────────────

_WS = re.compile(r"\s+")
_BINS = 64
_SALTS = (0x9E3779B1, 0x85EBCA77, 0xC2B2AE3D, 0x27D4EB2F)
_MERSENNE = (1 << 61) - 1


def _shingles(record: Record) -> frozenset[int]:
    """Hashed lexical shingles of the whole conversation (roles ignored).

    Word 3-grams for ordinary text; character 5-grams when there are too few
    words for word shingles to say anything.
    """
    text = _WS.sub(" ", " ".join(m["content"] for m in record.messages).lower()).strip()
    words = text.split(" ")
    if len(words) >= 10:
        grams = (" ".join(words[i:i + 3]) for i in range(len(words) - 2))
    elif len(text) >= 5:
        grams = (text[i:i + 5] for i in range(len(text) - 4))
    else:
        grams = iter((text,))
    return frozenset(zlib.crc32(g.encode("utf-8")) for g in grams)


def _band_keys(shingles: frozenset[int]) -> list[tuple]:
    """Locality-sensitive keys: two sets sharing any key are compared exactly.

    One-permutation bins give many cheap bands for long texts; four salted
    whole-set minima keep recall for texts too short to fill the bins.
    """
    bins: list[int | None] = [None] * _BINS
    minima = [_MERSENNE] * len(_SALTS)
    for h in shingles:
        b = h % _BINS
        if bins[b] is None or h < bins[b]:
            bins[b] = h
        for i, salt in enumerate(_SALTS):
            v = (h * salt + i) % _MERSENNE
            if v < minima[i]:
                minima[i] = v
    keys: list[tuple] = [("m", i, v) for i, v in enumerate(minima)]
    for i in range(0, _BINS, 2):
        if bins[i] is not None and bins[i + 1] is not None:
            keys.append(("b", i, bins[i], bins[i + 1]))
    return keys


@dataclass
class Fingerprints:
    """Hashes of one split. Carries no message text, so it is safe to hand to
    code that must not see the split's examples."""
    name: str
    ids: list[str]
    lines: list[int]
    full: dict[str, int]
    prompt: dict[str, int]
    groups: dict[str, int]
    shingles: list[frozenset[int]]
    bands: dict[tuple, list[int]]


def fingerprints(split: Split) -> Fingerprints:
    fp = Fingerprints(split.name, [], [], {}, {}, {}, [], {})
    for i, record in enumerate(split.records):
        normalized = {"messages": list(record.messages)}
        fp.ids.append(record.id)
        fp.lines.append(record.line_no)
        fp.full.setdefault(record_hash(normalized), i)
        fp.prompt.setdefault(prompt_hash(normalized), i)
        if record.group is not None:
            fp.groups.setdefault(record.group, i)
        sh = _shingles(record)
        fp.shingles.append(sh)
        for key in _band_keys(sh):
            fp.bands.setdefault(key, []).append(i)
    return fp


# ── leakage ─────────────────────────────────────────────────────────────────

EXACT, UNCERTAIN = "exact", "uncertain"


@dataclass(frozen=True)
class Finding:
    kind: str            # exact-record | exact-prompt | related-document | near-duplicate
    certainty: str       # exact: a fact about the files. uncertain: a similarity estimate.
    split: str
    id: str
    line: int
    other_split: str
    other_id: str
    other_line: int
    similarity: float | None
    action: str          # excluded | reported

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LeakageReport:
    findings: list[Finding] = field(default_factory=list)
    excluded: set[str] = field(default_factory=set)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            key = f"{f.split}~{f.other_split}:{f.kind}"
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    def to_dict(self, limit: int = 200) -> dict:
        ordered = sorted(self.findings, key=lambda f: (f.certainty != EXACT, f.split, f.line))
        return {
            "counts": self.counts(),
            "excluded_candidates": len(self.excluded),
            "exact_findings": sum(f.certainty == EXACT for f in self.findings),
            "uncertain_findings": sum(f.certainty == UNCERTAIN for f in self.findings),
            "findings": [f.to_dict() for f in ordered[:limit]],
            "findings_truncated": len(ordered) > limit,
            "note": ("Exact findings are identical normalized records, identical prompts, or a shared "
                     "related-document key. Uncertain findings are lexical near-duplicates (shingle "
                     "Jaccard): they can be coincidental, and paraphrased leakage is not detected."),
        }


def _compare(split: Split, other: Fingerprints, threshold: float) -> Iterable[tuple[Record, str, str, int, float | None]]:
    """Yield `(record, kind, certainty, other_index, similarity)` for one direction.

    At most one finding per record, the strongest: a record that is an exact
    copy is not also listed as a near-duplicate of the same example.
    """
    for record in split.records:
        normalized = {"messages": list(record.messages)}
        hit = other.full.get(record_hash(normalized))
        if hit is not None:
            yield record, "exact-record", EXACT, hit, 1.0
            continue
        hit = other.prompt.get(prompt_hash(normalized))
        if hit is not None:
            yield record, "exact-prompt", EXACT, hit, None
            continue
        if record.group is not None and record.group in other.groups:
            yield record, "related-document", EXACT, other.groups[record.group], None
            continue
        sh = _shingles(record)
        seen: set[int] = set()
        best, best_sim = None, 0.0
        for key in _band_keys(sh):
            for j in other.bands.get(key, ()):
                if j in seen:
                    continue
                seen.add(j)
                o = other.shingles[j]
                union = len(sh | o)
                sim = len(sh & o) / union if union else 0.0
                if sim > best_sim:
                    best, best_sim = j, sim
        if best is not None and best_sim >= threshold:
            yield record, "near-duplicate", UNCERTAIN, best, round(best_sim, 4)


def detect_leakage(candidates: Split, dev: Split, test: Fingerprints, policy: LeakagePolicy) -> LeakageReport:
    """Find overlap between splits and decide which candidates to leave out.

    Development and test must not share exact content under any policy: a
    final number measured on examples that steered selection is not a
    held-out number.
    """
    report = LeakageReport()
    dev_fp = fingerprints(dev)
    for record, kind, certainty, j, sim in _compare(dev, test, policy.near_threshold):
        if certainty == EXACT:
            raise LeakageError(
                f"Development and test splits overlap ({kind}: dev line {record.line_no}, test line "
                f"{test.lines[j]}). The test split must be disjoint from everything used for selection.")
        report.findings.append(Finding(kind, certainty, "dev", record.id, record.line_no, "test",
                                       test.ids[j], test.lines[j], sim, "reported"))
    for other in (dev_fp, test):
        for record, kind, certainty, j, sim in _compare(candidates, other, policy.near_threshold):
            mode = (policy.on_near if kind == "near-duplicate"
                    else policy.on_related if kind == "related-document" else policy.on_exact)
            if mode == "error":
                raise LeakageError(
                    f"Candidate line {record.line_no} overlaps the {other.name} split ({kind}, "
                    f"{other.name} line {other.lines[j]}); data.leakage is set to stop on this.")
            action = "excluded" if mode == "exclude" else "reported"
            if action == "excluded":
                report.excluded.add(record.id)
            report.findings.append(Finding(kind, certainty, "candidates", record.id, record.line_no,
                                           other.name, other.ids[j], other.lines[j], sim, action))
    return report


# ── development split halves ────────────────────────────────────────────────

def split_dev(dev: Split, fraction: float, seed: int) -> tuple[list[Record], list[Record]]:
    """Deterministically divide development examples into `(selection, comparison)`.

    Gradient scoring optimizes alignment with the selection half. Comparing
    policies on those same examples would favor the gradient selector for the
    wrong reason, so policies are compared on the other half. The division is
    stratified by capability and depends only on record ids and the seed.
    With `fraction` 1.0 both halves are the whole split, and reports say so.
    """
    if fraction >= 1.0:
        return list(dev.records), list(dev.records)
    by_cap: dict[str | None, list[Record]] = {}
    for record in dev.records:
        by_cap.setdefault(record.capability, []).append(record)
    selection: list[Record] = []
    comparison: list[Record] = []
    for cap in sorted(by_cap, key=lambda c: (c is None, c or "")):
        group = sorted(by_cap[cap], key=lambda r: hashlib.sha256(f"{seed}:{r.id}".encode()).hexdigest())
        if len(group) < 2:
            raise ConfigError(
                f"Development capability {cap!r} has {len(group)} example(s); at least 2 are needed to "
                "keep selection and comparison examples apart (or set data.dev_selection_fraction: 1.0)")
        k = min(len(group) - 1, max(1, round(len(group) * fraction)))
        selection.extend(group[:k])
        comparison.extend(group[k:])
    order = {r.id: i for i, r in enumerate(dev.records)}
    return (sorted(selection, key=lambda r: order[r.id]), sorted(comparison, key=lambda r: order[r.id]))


# ── the test vault ──────────────────────────────────────────────────────────

def freeze_seal(body: dict) -> str:
    """Hash binding a freeze record to the candidate and recipe it froze."""
    return digest({k: body.get(k) for k in ("trial", "selection_sha256", "recipe_fingerprint",
                                            "contract_fingerprint")})


class TestVault:
    """Holds the final test split and releases examples only after a freeze.

    Before the freeze the only thing that leaves is `fingerprints()`, used by
    the leakage guard. `open()` verifies the freeze record's seal so that a
    hand-written or stale record does not unlock the split by accident.
    """
    __test__ = False  # not a pytest class, despite the name

    def __init__(self, data: DataSpec, attr_fields: Iterable[str] = ()):
        self._data = data
        self._attr_fields = tuple(attr_fields)

    def _load(self) -> Split:
        return load_split(self._data.test, "test", id_field=self._data.id_field,
                          group_field=self._data.group_field,
                          capability_field=self._data.capability_field, attr_fields=self._attr_fields)

    def fingerprints(self) -> Fingerprints:
        return fingerprints(self._load())

    def summary(self, count_tokens: Any = None) -> dict:
        """Size and identity of the split, for planning and early length
        checks. Only numbers leave: `count_tokens` runs inside the vault."""
        split = self._load()
        capabilities: dict[str, int] = {}
        untagged = []
        for record in split.records:
            if record.capability is not None:
                capabilities[record.capability] = capabilities.get(record.capability, 0) + 1
            else:
                untagged.append(record.line_no)
        out = {"records": len(split.records), "sha256": split.sha256, "fingerprint": split.fingerprint,
               "capabilities": dict(sorted(capabilities.items())), "untagged_lines": untagged[:5]}
        if count_tokens is not None:
            counts = [count_tokens(r) for r in split.records]
            out.update(tokens=sum(counts), max_tokens=max(counts))
        return out

    def open(self, freeze: Any) -> Split:
        if (not isinstance(freeze, dict) or freeze.get("kind") != "optimize-freeze"
                or freeze.get("seal") != freeze_seal(freeze)):
            raise SplitIsolationError(
                "Final test examples were requested without a valid freeze record. The candidate and "
                "recipe must be frozen before the test split is read.")
        return self._load()
