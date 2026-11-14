"""The experiment contract: one typed, versioned description of an experiment.

Everything that can change a result lives here, so that a run directory can be
traced back to exactly one contract and a cache entry can be keyed by the parts
of it that matter. Parsing is strict on purpose: an unknown or misspelled key is
an error, because a silently ignored `seeds:` or `loss_masking:` would produce a
run that looks like the one the user asked for and is not.

The module is standard-library only (YAML needs the optional PyYAML; JSON does
not), so `plan` and config validation work on the default lightweight install.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

from ..formats import supported_formats
from .errors import ConfigError, MissingDependencyError, UnsupportedError

SCHEMA_VERSION = 1
OBJECTIVE = "sft-lora-causal-lm"
SCALES = ("pilot", "benchmark")
POLICIES = ("random", "diversity", "less")
METRICS = ("loss", "exact_match")
LOSS_MASKS = ("assistant_only", "final_assistant_only")
# Formats whose records are plain chat once normalized. `preference` trains a
# different objective (DPO) and is rejected rather than flattened to SFT.
TRAINABLE_FORMATS = tuple(f for f in supported_formats() if f != "preference")


# ── contract types ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SplitSpec:
    path: Path
    format: str = "auto"


@dataclass(frozen=True)
class LeakagePolicy:
    """What to do when splits overlap.

    Exact and related-document matches are facts about the files; near
    duplicates are a lexical similarity estimate, so by default they are only
    reported and a person decides.
    """
    on_exact: str = "exclude"        # exclude | error
    on_related: str = "exclude"      # exclude | error | report
    on_near: str = "report"          # report | exclude
    near_threshold: float = 0.8


@dataclass(frozen=True)
class DataSpec:
    candidates: SplitSpec
    dev: SplitSpec
    test: SplitSpec
    id_field: str | None = None
    group_field: str | None = None
    capability_field: str | None = None
    dev_selection_fraction: float = 0.5
    dev_split_seed: int = 0
    leakage: LeakagePolicy = LeakagePolicy()


@dataclass(frozen=True)
class ModelSpec:
    path: str
    revision: str | None = None
    tokenizer: str | None = None
    tokenizer_revision: str | None = None
    chat_template: str = "tokenizer"   # "tokenizer" or a path to a Jinja file
    max_seq_len: int = 1024
    on_overlength: str = "error"       # error | exclude
    device: str = "cpu"
    dtype: str = "float32"


@dataclass(frozen=True)
class LoraSpec:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecipeSpec:
    lora: LoraSpec
    seeds: tuple[int, ...]
    optimizer: str = "adamw"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    epochs: int = 3
    batch_size: int = 8
    schedule: str = "linear"           # linear | constant
    warmup_ratio: float = 0.0
    max_grad_norm: float | None = 1.0
    loss_masking: str = "assistant_only"


@dataclass(frozen=True)
class EvalSpec:
    metrics: tuple[str, ...] = ("loss",)
    primary_metric: str = "loss"
    max_new_tokens: int = 32
    batch_size: int = 16
    target_capabilities: tuple[str, ...] = ()
    retain_capabilities: tuple[str, ...] = ()
    retention_tolerance: float = 0.02
    bootstrap_samples: int = 1000


@dataclass(frozen=True)
class CoverageConstraint:
    field: str
    value: str
    min_records: int = 0
    min_tokens: int = 0


@dataclass(frozen=True)
class LessSpec:
    """Settings of the LESS-inspired gradient scoring backend."""
    warmup_fraction: float = 0.05
    warmup_epochs: int = 4
    warmup_max_steps: int | None = None
    warmup_seed: int = 0
    projection: str = "countsketch"     # countsketch | identity
    projection_dim: int = 8192
    projection_seed: int = 0
    optimizer_aware: bool = True
    normalization: str = "cosine"       # cosine | none
    checkpoint_weighting: str = "lr"    # lr | uniform
    target_aggregation: str = "max"     # max | mean
    diversity_lambda: float = 0.0
    pool_factor: float = 4.0
    shard_size: int = 512
    exclude_negative: bool = False
    retention_floor: float | None = None


@dataclass(frozen=True)
class SelectionSpec:
    token_budget: int
    policies: tuple[str, ...] = POLICIES
    seed: int = 0
    coverage: tuple[CoverageConstraint, ...] = ()
    report_fields: tuple[str, ...] = ()
    less: LessSpec = LessSpec()


@dataclass(frozen=True)
class SearchSpec:
    """Extra trials compared on development data, bounded by `max_trials`."""
    token_budgets: tuple[int, ...] = ()
    less_variants: tuple[tuple[tuple[str, Any], ...], ...] = ()
    max_trials: int = 0


@dataclass(frozen=True)
class BaselineSpec:
    base_model: bool = True
    full_data: bool = True
    random_matched_compute: bool = True


@dataclass(frozen=True)
class BudgetSpec:
    """Total budget in forward-token-equivalents (see budget.py)."""
    total_compute: float
    backward_multiplier: float = 2.0


@dataclass(frozen=True)
class OutputSpec:
    run_dir: Path
    cache_dir: Path


@dataclass(frozen=True)
class Experiment:
    schema_version: int
    name: str
    scale: str
    objective: str
    data: DataSpec
    model: ModelSpec
    recipe: RecipeSpec
    evaluation: EvalSpec
    selection: SelectionSpec
    budget: BudgetSpec
    output: OutputSpec
    search: SearchSpec = SearchSpec()
    baselines: BaselineSpec = BaselineSpec()
    source: Path | None = field(default=None, compare=False)

    def to_dict(self) -> dict:
        """JSON-safe copy of the contract (paths as strings, no `source`)."""
        out = _plain(self)
        out.pop("source", None)
        return out

    def fingerprint(self, *sections: str) -> str:
        """Stable hash of the named sections (all result-bearing ones by default).

        `name` and `output` are excluded: renaming a run or moving its
        directory does not change what was measured.
        """
        data = self.to_dict()
        keys = sections or tuple(k for k in data if k not in ("name", "output"))
        return digest({k: data[k] for k in keys})


def digest(value: Any) -> str:
    """sha256 of the canonical JSON form; the basis of every fingerprint."""
    canon = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


# ── parsing ─────────────────────────────────────────────────────────────────

_SCI = re.compile(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+")

class _Section:
    """Typed, exhaustive reader over one mapping of the contract.

    Every key must be consumed; `done()` rejects whatever is left, which is how
    misspelled or unsupported options become errors instead of no-ops.
    """

    def __init__(self, data: Any, where: str):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ConfigError(f"{where}: expected a mapping, found {type(data).__name__}")
        self.data = dict(data)
        self.where = where

    def _path(self, key: str) -> str:
        return f"{self.where}.{key}" if self.where else key

    def take(self, key: str, kind: type | tuple[type, ...], default: Any = ..., *,
             choices: tuple | None = None, nullable: bool = False) -> Any:
        if key not in self.data:
            if default is ...:
                raise ConfigError(f"{self._path(key)}: required")
            return default
        value = self.data.pop(key)
        if value is None and nullable:
            return None
        kinds = kind if isinstance(kind, tuple) else (kind,)
        # bool is an int in Python; a `true` where a number belongs is a typo.
        if isinstance(value, bool) and bool not in kinds:
            raise ConfigError(f"{self._path(key)}: expected {_names(kinds)}, found a boolean")
        if float in kinds and isinstance(value, int):
            value = float(value)
        # YAML 1.1 (PyYAML) reads `2e-4` as a string; only that exact shape of
        # scientific notation is accepted as the number it plainly is.
        if float in kinds and isinstance(value, str) and _SCI.fullmatch(value):
            value = float(value)
        if not isinstance(value, kinds):
            raise ConfigError(f"{self._path(key)}: expected {_names(kinds)}, found {type(value).__name__}")
        if choices is not None and value not in choices:
            raise ConfigError(f"{self._path(key)}: {value!r} is not supported. Valid: {list(choices)}")
        return value

    def section(self, key: str, required: bool = False) -> "_Section":
        if key not in self.data and required:
            raise ConfigError(f"{self._path(key)}: required")
        return _Section(self.data.pop(key, None), self._path(key))

    def strings(self, key: str, default: tuple = (), choices: tuple | None = None) -> tuple[str, ...]:
        value = self.take(key, list, list(default))
        for item in value:
            if not isinstance(item, str):
                raise ConfigError(f"{self._path(key)}: every entry must be a string")
            if choices is not None and item not in choices:
                raise ConfigError(f"{self._path(key)}: {item!r} is not supported. Valid: {list(choices)}")
        if len(set(value)) != len(value):
            raise ConfigError(f"{self._path(key)}: entries must be unique")
        return tuple(value)

    def ints(self, key: str, default: tuple = ()) -> tuple[int, ...]:
        value = self.take(key, list, list(default))
        if any(type(item) is not int for item in value):
            raise ConfigError(f"{self._path(key)}: every entry must be an integer")
        if len(set(value)) != len(value):
            raise ConfigError(f"{self._path(key)}: entries must be unique")
        return tuple(value)

    def done(self) -> None:
        if self.data:
            raise ConfigError(f"{self.where or 'config'}: unknown key(s) {sorted(self.data)}")


def _names(kinds: tuple[type, ...]) -> str:
    return " or ".join({"str": "a string", "int": "an integer", "float": "a number",
                        "bool": "a boolean", "list": "a list", "dict": "a mapping"}[k.__name__] for k in kinds)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _split(section: _Section, base: Path) -> SplitSpec:
    path = section.take("path", str)
    fmt = section.take("format", str, "auto")
    if fmt == "preference":
        raise UnsupportedError(
            f"{section.where}.format: 'preference' (DPO) data is not supported by objective {OBJECTIVE!r}")
    if fmt not in TRAINABLE_FORMATS:
        raise ConfigError(f"{section.where}.format: {fmt!r} is not supported. Valid: {list(TRAINABLE_FORMATS)}")
    section.done()
    return SplitSpec((base / path).resolve() if not Path(path).is_absolute() else Path(path), fmt)


_LESS_FIELDS = {f.name for f in fields(LessSpec)}
# Scoring settings a search variant may change without recomputing gradients
# under a different contract than the plan accounted for.
_VARIANT_KEYS = ("diversity_lambda", "target_aggregation", "normalization", "checkpoint_weighting",
                 "exclude_negative", "retention_floor")


def _less(section: _Section) -> LessSpec:
    warm = section.section("warmup")
    proj = section.section("projection")
    spec = LessSpec(
        warmup_fraction=warm.take("fraction", float, 0.05),
        warmup_epochs=warm.take("epochs", int, 4),
        warmup_max_steps=warm.take("max_steps", int, None, nullable=True),
        warmup_seed=warm.take("seed", int, 0),
        projection=proj.take("kind", str, "countsketch", choices=("countsketch", "identity")),
        projection_dim=proj.take("dim", int, 8192),
        projection_seed=proj.take("seed", int, 0),
        optimizer_aware=section.take("optimizer_aware", bool, True),
        normalization=section.take("normalization", str, "cosine", choices=("cosine", "none")),
        checkpoint_weighting=section.take("checkpoint_weighting", str, "lr", choices=("lr", "uniform")),
        target_aggregation=section.take("target_aggregation", str, "max", choices=("max", "mean")),
        diversity_lambda=section.take("diversity_lambda", float, 0.0),
        pool_factor=section.take("pool_factor", float, 4.0),
        shard_size=section.take("shard_size", int, 512),
        exclude_negative=section.take("exclude_negative", bool, False),
        retention_floor=section.take("retention_floor", float, None, nullable=True),
    )
    warm.done(), proj.done(), section.done()
    _require(0 < spec.warmup_fraction <= 1, "selection.less.warmup.fraction: must be in (0, 1]")
    _require(spec.warmup_epochs >= 1, "selection.less.warmup.epochs: must be at least 1")
    _require(spec.warmup_max_steps is None or spec.warmup_max_steps >= 1,
             "selection.less.warmup.max_steps: must be at least 1")
    _require(spec.projection_dim >= 1, "selection.less.projection.dim: must be at least 1")
    _require(spec.diversity_lambda >= 0, "selection.less.diversity_lambda: must not be negative")
    _require(spec.pool_factor >= 1, "selection.less.pool_factor: must be at least 1")
    _require(spec.shard_size >= 1, "selection.less.shard_size: must be at least 1")
    return spec


def parse_experiment(data: Any, *, base: Path | None = None, source: Path | None = None) -> Experiment:
    """Validate a decoded contract. Relative paths resolve against `base`."""
    base = Path(base or Path.cwd())
    root = _Section(data, "")
    version = root.take("schema_version", int)
    if version != SCHEMA_VERSION:
        raise ConfigError(f"schema_version: {version} is not supported by this release (expected {SCHEMA_VERSION})")
    name = root.take("name", str)
    _require(bool(name.strip()), "name: must not be empty")
    scale = root.take("scale", str, choices=SCALES)
    objective = root.take("objective", str)
    if objective != OBJECTIVE:
        raise UnsupportedError(
            f"objective: {objective!r} is not supported. This release trains {OBJECTIVE!r} only: "
            "supervised instruction tuning of a local Hugging Face causal language model with LoRA.")

    d = root.section("data", required=True)
    leak = d.section("leakage")
    data_spec = DataSpec(
        candidates=_split(d.section("candidates", required=True), base),
        dev=_split(d.section("dev", required=True), base),
        test=_split(d.section("test", required=True), base),
        id_field=d.take("id_field", str, None, nullable=True),
        group_field=d.take("group_field", str, None, nullable=True),
        capability_field=d.take("capability_field", str, None, nullable=True),
        dev_selection_fraction=d.take("dev_selection_fraction", float, 0.5),
        dev_split_seed=d.take("dev_split_seed", int, 0),
        leakage=LeakagePolicy(
            on_exact=leak.take("on_exact", str, "exclude", choices=("exclude", "error")),
            on_related=leak.take("on_related", str, "exclude", choices=("exclude", "error", "report")),
            on_near=leak.take("on_near", str, "report", choices=("report", "exclude")),
            near_threshold=leak.take("near_threshold", float, 0.8),
        ),
    )
    leak.done(), d.done()
    paths = [data_spec.candidates.path, data_spec.dev.path, data_spec.test.path]
    _require(len(set(paths)) == 3, "data: candidates, dev and test must be three different files")
    _require(0 < data_spec.dev_selection_fraction <= 1, "data.dev_selection_fraction: must be in (0, 1]")
    _require(0 < data_spec.leakage.near_threshold <= 1, "data.leakage.near_threshold: must be in (0, 1]")

    m = root.section("model", required=True)
    model_path = m.take("path", str)
    template = m.take("chat_template", str, "tokenizer")
    if template != "tokenizer":
        template = str((base / template).resolve() if not Path(template).is_absolute() else Path(template))
    model = ModelSpec(
        path=_model_ref(model_path, base),
        revision=m.take("revision", str, None, nullable=True),
        tokenizer=_model_ref(m.take("tokenizer", str, None, nullable=True), base),
        tokenizer_revision=m.take("tokenizer_revision", str, None, nullable=True),
        chat_template=template,
        max_seq_len=m.take("max_seq_len", int, 1024),
        on_overlength=m.take("on_overlength", str, "error", choices=("error", "exclude")),
        device=m.take("device", str, "cpu"),
        dtype=m.take("dtype", str, "float32", choices=("float32", "float16", "bfloat16")),
    )
    m.done()
    _require(model.max_seq_len >= 8, "model.max_seq_len: must be at least 8")
    _require(model.device == "cpu" or model.device == "mps" or model.device.startswith("cuda"),
             "model.device: name a device explicitly: cpu, mps, cuda or cuda:N")

    r = root.section("recipe", required=True)
    lora = r.section("lora", required=True)
    lora_spec = LoraSpec(
        r=lora.take("r", int, 8), alpha=lora.take("alpha", float, 16.0),
        dropout=lora.take("dropout", float, 0.0),
        target_modules=lora.strings("target_modules"),
    )
    lora.done()
    optimizer = r.take("optimizer", str, "adamw")
    if optimizer != "adamw":
        raise UnsupportedError(f"recipe.optimizer: {optimizer!r} is not supported; only 'adamw' is implemented")
    recipe = RecipeSpec(
        lora=lora_spec, seeds=r.ints("seeds"), optimizer=optimizer,
        learning_rate=r.take("learning_rate", float, 2e-4),
        weight_decay=r.take("weight_decay", float, 0.0),
        beta1=r.take("beta1", float, 0.9), beta2=r.take("beta2", float, 0.999),
        eps=r.take("eps", float, 1e-8),
        epochs=r.take("epochs", int, 3), batch_size=r.take("batch_size", int, 8),
        schedule=r.take("schedule", str, "linear", choices=("linear", "constant")),
        warmup_ratio=r.take("warmup_ratio", float, 0.0),
        max_grad_norm=r.take("max_grad_norm", float, 1.0, nullable=True),
        loss_masking=_mask(r.take("loss_masking", str, "assistant_only")),
    )
    r.done()
    _require(lora_spec.r >= 1, "recipe.lora.r: must be at least 1")
    _require(bool(lora_spec.target_modules), "recipe.lora.target_modules: name at least one module")
    _require(0 <= lora_spec.dropout < 1, "recipe.lora.dropout: must be in [0, 1)")
    _require(bool(recipe.seeds), "recipe.seeds: give at least one seed")
    _require(recipe.learning_rate > 0, "recipe.learning_rate: must be positive")
    _require(recipe.epochs >= 1 and recipe.batch_size >= 1, "recipe: epochs and batch_size must be at least 1")
    _require(0 <= recipe.warmup_ratio < 1, "recipe.warmup_ratio: must be in [0, 1)")
    _require(0 <= recipe.beta1 < 1 and 0 <= recipe.beta2 < 1 and recipe.eps > 0,
             "recipe: beta1 and beta2 must be in [0, 1) and eps positive")

    e = root.section("evaluation")
    evaluation = EvalSpec(
        metrics=e.strings("metrics", ("loss",), METRICS),
        primary_metric=e.take("primary_metric", str, "loss", choices=METRICS),
        max_new_tokens=e.take("max_new_tokens", int, 32),
        batch_size=e.take("batch_size", int, 16),
        target_capabilities=e.strings("target_capabilities"),
        retain_capabilities=e.strings("retain_capabilities"),
        retention_tolerance=e.take("retention_tolerance", float, 0.02),
        bootstrap_samples=e.take("bootstrap_samples", int, 1000),
    )
    e.done()
    _require("loss" in evaluation.metrics, "evaluation.metrics: 'loss' is always measured and must be listed")
    _require(evaluation.primary_metric in evaluation.metrics,
             "evaluation.primary_metric: must be one of evaluation.metrics")
    _require(evaluation.max_new_tokens >= 1 and evaluation.batch_size >= 1,
             "evaluation: max_new_tokens and batch_size must be at least 1")
    _require(evaluation.retention_tolerance >= 0, "evaluation.retention_tolerance: must not be negative")
    _require(evaluation.bootstrap_samples >= 100, "evaluation.bootstrap_samples: must be at least 100")
    both = set(evaluation.target_capabilities) & set(evaluation.retain_capabilities)
    _require(not both, f"evaluation: {sorted(both)} cannot be both a target and a retained capability")
    if evaluation.target_capabilities or evaluation.retain_capabilities:
        _require(data_spec.capability_field is not None,
                 "data.capability_field: required when evaluation names capabilities")

    s = root.section("selection", required=True)
    coverage = []
    for i, item in enumerate(s.take("coverage", list, [])):
        c = _Section(item, f"selection.coverage[{i}]")
        value = c.take("value", (str, int))
        constraint = CoverageConstraint(field=c.take("field", str), value=str(value),
                                        min_records=c.take("min_records", int, 0),
                                        min_tokens=c.take("min_tokens", int, 0))
        c.done()
        _require(constraint.min_records > 0 or constraint.min_tokens > 0,
                 f"selection.coverage[{i}]: set min_records or min_tokens above zero")
        coverage.append(constraint)
    selection = SelectionSpec(
        token_budget=s.take("token_budget", int),
        policies=s.strings("policies", POLICIES, POLICIES),
        seed=s.take("seed", int, 0),
        coverage=tuple(coverage),
        report_fields=s.strings("report_fields"),
        less=_less(s.section("less")),
    )
    s.done()
    _require(selection.token_budget >= 1, "selection.token_budget: must be at least 1")
    _require(bool(selection.policies), "selection.policies: name at least one policy")

    q = root.section("search")
    variants = []
    for i, item in enumerate(q.take("less_variants", list, [])):
        if not isinstance(item, dict) or not item:
            raise ConfigError(f"search.less_variants[{i}]: expected a non-empty mapping")
        unknown = set(item) - set(_VARIANT_KEYS)
        _require(not unknown, f"search.less_variants[{i}]: only {list(_VARIANT_KEYS)} may vary; found {sorted(unknown)}")
        # Validate by building the full spec the variant would produce.
        merged = {**_plain(selection.less), **item}
        try:
            LessSpec(**merged)
        except TypeError as exc:  # pragma: no cover - guarded by the key check
            raise ConfigError(f"search.less_variants[{i}]: {exc}") from exc
        variants.append(tuple(sorted(item.items())))
    search = SearchSpec(token_budgets=q.ints("token_budgets"), less_variants=tuple(variants),
                        max_trials=q.take("max_trials", int, 0))
    q.done()
    _require(all(1 <= b < selection.token_budget for b in search.token_budgets),
             "search.token_budgets: must be positive and below selection.token_budget, which is the ceiling "
             "for a recommended subset; to explore larger subsets, raise selection.token_budget")
    _require(search.max_trials >= 0, "search.max_trials: must not be negative")
    if (search.token_budgets or search.less_variants) and search.max_trials == 0:
        raise ConfigError("search.max_trials: set a bound above zero to run the listed search trials")
    if search.less_variants:
        _require("less" in selection.policies, "search.less_variants: requires the 'less' policy")

    b = root.section("baselines")
    baselines = BaselineSpec(base_model=b.take("base_model", bool, True),
                             full_data=b.take("full_data", bool, True),
                             random_matched_compute=b.take("random_matched_compute", bool, True))
    b.done()

    g = root.section("budget", required=True)
    budget = BudgetSpec(total_compute=g.take("total_compute", float),
                        backward_multiplier=g.take("backward_multiplier", float, 2.0))
    g.done()
    _require(budget.total_compute > 0, "budget.total_compute: must be positive")
    _require(budget.backward_multiplier >= 0, "budget.backward_multiplier: must not be negative")

    o = root.section("output", required=True)
    run_dir, cache_dir = o.take("run_dir", str), o.take("cache_dir", str, ".parallelogram-cache")
    output = OutputSpec(run_dir=(base / run_dir).resolve(), cache_dir=(base / cache_dir).resolve())
    o.done()
    root.done()

    return Experiment(schema_version=version, name=name, scale=scale, objective=objective, data=data_spec,
                      model=model, recipe=recipe, evaluation=evaluation, selection=selection, budget=budget,
                      output=output, search=search, baselines=baselines, source=source)


def _mask(value: str) -> str:
    if value not in LOSS_MASKS:
        raise UnsupportedError(f"recipe.loss_masking: {value!r} is not supported. Valid: {list(LOSS_MASKS)}")
    return value


def _model_ref(value: str | None, base: Path) -> str | None:
    """A path that exists relative to the contract is local; anything else is
    a Hub id, which is only ever fetched with explicit opt-in."""
    if value is None:
        return None
    local = Path(value) if Path(value).is_absolute() else base / value
    return str(local.resolve()) if local.exists() else value


def check_model_refs(exp: "Experiment") -> None:
    """A model or tokenizer that is not a local directory is a Hub id, and a
    Hub id without a pinned revision can change under the experiment."""
    model = exp.model
    refs = [("model.path", model.path, model.revision, "model.revision")]
    if model.tokenizer:
        refs.append(("model.tokenizer", model.tokenizer, model.tokenizer_revision, "model.tokenizer_revision"))
    for field_name, ref, revision, revision_field in refs:
        if not Path(ref).is_dir() and revision is None:
            raise ConfigError(
                f"{field_name}: {ref!r} is not a local directory, so it is read as a Hub id; set "
                f"{revision_field} to a commit hash so the experiment stays reproducible")


def variant_spec(spec: LessSpec, overrides: tuple[tuple[str, Any], ...]) -> LessSpec:
    """The scoring settings of one search variant."""
    return LessSpec(**{**asdict(spec), **dict(overrides)})


def load_experiment(path: str | Path) -> Experiment:
    """Read and validate a `.yaml`/`.yml` or `.json` experiment contract."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        from .errors import OptimizeError
        raise OptimizeError(f"Could not read experiment {path}: {exc}") from exc
    if path.suffix.lower() == ".json":
        try:
            data = json.loads(text)
        except (ValueError, RecursionError) as exc:
            raise ConfigError(f"{path}: invalid JSON: {exc}") from exc
    elif path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise MissingDependencyError(
                "Reading a YAML experiment needs PyYAML. Install the extras with "
                "pip install 'parallelogram[optimize]', or write the contract as .json.") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    else:
        raise ConfigError(f"{path}: experiment contracts are .yaml, .yml or .json files")
    return parse_experiment(data, base=path.resolve().parent, source=path.resolve())
