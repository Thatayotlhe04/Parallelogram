"""Turn a contract into an experiment matrix that fits the budget.

Planning reads the data and counts tokens but trains nothing. Its output says
which arms will run with which seeds, what each is expected to cost, what was
dropped because the budget does not allow it, and which files a run will
leave behind. `run` executes the same plan, so the plan is not a brochure: an
arm the plan dropped is an arm the run does not perform, and the report lists
it under experiments not performed.

The matrix is built in priority order. Essential units (base-model
evaluation, the scoring overhead, one seed of each requested policy, and the
final evaluation of whatever gets frozen) must fit or the plan is refused.
Optional units are added while they fit: more seeds first, then the
matched-compute baseline, then search trials, then full-data training.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

from .config import Experiment, digest
from .errors import ConfigError
from .splits import LeakageReport, Record, Split, TestVault, detect_leakage, load_split, split_dev

ALL = "all"


def trial_slug(key: str) -> str:
    """Directory name of a trial: readable, filesystem-safe, collision-free."""
    return re.sub(r"[^A-Za-z0-9.+@-]+", "_", key) + "-" + digest(key)[:8]


@dataclass(frozen=True)
class Trial:
    """One arm of the experiment: a policy at a token budget."""
    key: str
    policy: str                 # random | diversity | less | full
    token_budget: int | None    # None for full data
    comparison: str             # matched_tokens | matched_compute | full_data
    role: str                   # primary | baseline | search
    overrides: tuple = ()

    def to_dict(self) -> dict:
        return {"key": self.key, "policy": self.policy, "token_budget": self.token_budget,
                "comparison": self.comparison, "role": self.role, "less_overrides": dict(self.overrides)}


COMPARISON_NOTES = {
    "matched_tokens": "same number of training tokens as the other arms at this budget: isolates which "
                      "records were chosen, and ignores what choosing them cost",
    "matched_compute": "random selection given the optimizer's scoring overhead as extra training tokens: "
                       "asks whether scoring was a better use of compute than simply training on more data",
    "full_data": "training on every candidate: the reference for how much quality a subset gives up, at "
                 "a different (larger) training cost",
}


@dataclass
class Prepared:
    """Everything known about the data before any training."""
    exp: Experiment
    candidates: Split
    dev: Split
    leakage: LeakageReport
    pool: list[Record]
    pool_tokens: list[int]
    dev_selection: list[Record]
    dev_compare: list[Record]
    dev_tokens: dict[str, int]
    test: dict
    exact_tokens: bool
    excluded: dict[str, int]
    target_capabilities: tuple[str, ...]
    retain_capabilities: tuple[str, ...]
    vault: TestVault = field(repr=False, default=None)

    def capability(self, record: Record) -> str:
        return record.capability if self.exp.data.capability_field else ALL


def approximate_tokens(record: Record) -> int:
    """Length-based estimate used when the tokenizer is not installed: about
    four characters per token plus a few template tokens per turn."""
    return sum(-(-len(m["content"]) // 4) + 4 for m in record.messages)


def prepare(exp: Experiment, count_tokens: Callable[[Record], int] | None = None) -> Prepared:
    """Load and guard the splits, and size everything in tokens.

    With `count_tokens` (the backend's tokenizer) counts are exact and length
    limits are enforced; without it they are estimates and `plan` says so.
    """
    data = exp.data
    attr_fields = sorted({c.field for c in exp.selection.coverage} | set(exp.selection.report_fields))
    common = dict(id_field=data.id_field, group_field=data.group_field,
                  capability_field=data.capability_field)
    candidates = load_split(data.candidates, "candidates", attr_fields=attr_fields, **common)
    dev = load_split(data.dev, "dev", **common)
    vault = TestVault(data)
    exact = count_tokens is not None
    count = count_tokens or approximate_tokens
    test = vault.summary(count)
    leakage = detect_leakage(candidates, dev, vault.fingerprints(), data.leakage)

    limit = exp.model.max_seq_len
    excluded = {"leakage": 0, "overlength": 0}
    pool, pool_tokens = [], []
    too_long = []
    for record in candidates.records:
        if record.id in leakage.excluded:
            excluded["leakage"] += 1
            continue
        tokens = count(record)
        if exact and tokens > limit:
            too_long.append(record.line_no)
            excluded["overlength"] += 1
            continue
        pool.append(record)
        pool_tokens.append(tokens)
    if too_long and exp.model.on_overlength == "error":
        raise ConfigError(
            f"{len(too_long)} candidate record(s) exceed model.max_seq_len={limit} tokens (first at line "
            f"{too_long[0]}). Raise the limit, shorten them with `parallelogram check --fix --tokenizer ... "
            "--max-seq-len ...`, or set model.on_overlength: exclude to leave them out explicitly.")
    if not pool:
        raise ConfigError("No candidate records are left after leakage and length exclusions")

    dev_selection, dev_compare = split_dev(dev, data.dev_selection_fraction, data.dev_split_seed)
    dev_tokens = {"selection": sum(count(r) for r in dev_selection),
                  "comparison": sum(count(r) for r in dev_compare)}
    if exact:
        longest = max(count(r) for r in dev.records)
        if longest > limit or test.get("max_tokens", 0) > limit:
            raise ConfigError(f"A development or test example exceeds model.max_seq_len={limit} tokens; "
                              "evaluation examples are never truncated or dropped")

    evaluation = exp.evaluation
    if data.capability_field:
        untagged = [r.line_no for r in dev.records if r.capability is None]
        if untagged:
            raise ConfigError(f"Development example at line {untagged[0]} has no "
                              f"{data.capability_field!r} value; every development example needs one")
        present = {r.capability for r in dev_selection} & {r.capability for r in dev_compare}
        retain = tuple(evaluation.retain_capabilities)
        target = tuple(evaluation.target_capabilities) or tuple(sorted(present - set(retain)))
        missing = [c for c in (*target, *retain) if c not in present]
        if missing or not target:
            raise ConfigError(f"Capabilities {missing or '(no target capability)'} have no development "
                              "examples on both sides of the selection/comparison division")
        if test.get("untagged_lines"):
            raise ConfigError(f"Test example at line {test['untagged_lines'][0]} has no "
                              f"{data.capability_field!r} value; every test example needs one")
        absent = [c for c in (*target, *retain) if c not in test.get("capabilities", {})]
        if absent:
            raise ConfigError(f"Capabilities {absent} have no test examples; the final evaluation could "
                              "not report on them")
    else:
        target, retain = (ALL,), ()
    return Prepared(exp=exp, candidates=candidates, dev=dev, leakage=leakage, pool=pool,
                    pool_tokens=pool_tokens, dev_selection=dev_selection, dev_compare=dev_compare,
                    dev_tokens=dev_tokens, test=test, exact_tokens=exact, excluded=excluded,
                    target_capabilities=target, retain_capabilities=retain, vault=vault)


# ── cost model ──────────────────────────────────────────────────────────────

class CostModel:
    """Estimated cost of each kind of unit, in forward-token-equivalents."""

    def __init__(self, prep: Prepared):
        exp = prep.exp
        self.exp = exp
        self.step = 1.0 + exp.budget.backward_multiplier     # forward + backward per token
        self.pool_tokens = sum(prep.pool_tokens)
        self.pool_records = len(prep.pool)
        self.mean_tokens = self.pool_tokens / self.pool_records
        self.prep = prep

    def evaluation(self, tokens: int, records: int) -> float:
        cost = float(tokens)
        if "exact_match" in self.exp.evaluation.metrics:
            # Prompt once more plus an upper bound on generated tokens.
            cost += tokens + records * self.exp.evaluation.max_new_tokens
        return cost

    def dev_evaluation(self) -> float:
        return self.evaluation(self.prep.dev_tokens["comparison"], len(self.prep.dev_compare))

    def test_evaluation(self) -> float:
        return self.evaluation(self.prep.test.get("tokens", 0), self.prep.test["records"])

    def training(self, tokens: float) -> float:
        return self.step * tokens * self.exp.recipe.epochs

    def checkpoints(self) -> int:
        less = self.exp.selection.less
        if less.warmup_max_steps is None:
            return less.warmup_epochs
        records = max(math.ceil(less.warmup_fraction * self.pool_records), 1)
        per_epoch = max(1, math.ceil(records / self.exp.recipe.batch_size))
        return max(1, min(less.warmup_epochs, math.ceil(less.warmup_max_steps / per_epoch)))

    def warmup(self) -> float:
        less = self.exp.selection.less
        tokens = less.warmup_fraction * self.pool_tokens * less.warmup_epochs
        if less.warmup_max_steps is not None:
            tokens = min(tokens, less.warmup_max_steps * self.exp.recipe.batch_size * self.mean_tokens)
        return self.step * tokens

    def scoring(self) -> float:
        return self.step * self.checkpoints() * (self.pool_tokens + self.prep.dev_tokens["selection"])

    def overhead(self) -> float:
        return self.warmup() + self.scoring()

    def matched_compute_budget(self, overhead: float) -> int | None:
        """Token budget of a random arm that spends the scoring overhead on
        more training data instead, or None when no such arm exists.

        If the overhead is worth more training tokens than the pool has left,
        a random arm cannot spend it: capped at the pool it would be full-data
        training, cost less than the model-aware arm, and still be labeled
        "matched". There is then no matched-compute baseline, and the report
        says so instead of pretending.
        """
        extra = overhead / (self.step * self.exp.recipe.epochs)
        budget = int(self.exp.selection.token_budget + extra)
        return budget if budget < self.pool_tokens else None


def _variant_key(overrides: tuple) -> str:
    return ",".join(f"{k}={v}" for k, v in overrides)


def trial_matrix(exp: Experiment, cost: CostModel,
                 overhead: float | None = None) -> tuple[list[Trial], list[Trial], list[Trial], list[Trial]]:
    """`(primary, matched_compute, search, full)` trials named by the contract.

    `search` is everything the contract lists; the plan keeps the first
    `search.max_trials` of them.
    """
    budget = exp.selection.token_budget
    primary = [Trial(f"{p}@{budget}", p, budget, "matched_tokens", "primary" if p == "less" else "baseline")
               for p in exp.selection.policies]
    matched = []
    if exp.baselines.random_matched_compute and "less" in exp.selection.policies:
        extra_budget = cost.matched_compute_budget(cost.overhead() if overhead is None else overhead)
        if extra_budget is not None:
            matched.append(Trial("random+overhead", "random", extra_budget, "matched_compute", "baseline"))
    search = []
    for b in exp.search.token_budgets:
        if b == budget:
            continue
        search.extend(Trial(f"{p}@{b}", p, b, "matched_tokens", "search") for p in exp.selection.policies)
    for overrides in exp.search.less_variants:
        search.append(Trial(f"less@{budget}~{_variant_key(overrides)}", "less", budget, "matched_tokens",
                            "search", overrides))
    full = [Trial("full", "full", None, "full_data", "baseline")] if exp.baselines.full_data else []
    return primary, matched, search, full


@dataclass
class Plan:
    trials: list[Trial]
    seeds: dict[str, list[int]]
    units: list[dict]
    skipped: list[dict]
    reserve_final: float
    estimate: float
    budget: float
    fits: bool
    problems: list[str]
    matched_compute_feasible: bool = True

    def trial(self, key: str) -> Trial:
        return next(t for t in self.trials if t.key == key)


def build_plan(prep: Prepared, overhead: float | None = None) -> Plan:
    """Fit the experiment matrix into the total budget. See module docstring.

    `overhead` is the measured scoring overhead once it is known (during a
    run); before that the estimate is used to size the matched-compute arm.
    """
    exp = prep.exp
    cost = CostModel(prep)
    primary, matched, declared_search, full = trial_matrix(exp, cost, overhead)
    search = declared_search[:exp.search.max_trials]
    seeds = list(exp.recipe.seeds)
    dev_eval, test_eval = cost.dev_evaluation(), cost.test_evaluation()
    uses_less = "less" in exp.selection.policies

    def trial_cost(trial: Trial) -> float:
        tokens = cost.pool_tokens if trial.policy == "full" else min(trial.token_budget, cost.pool_tokens)
        return cost.training(tokens) + dev_eval

    included: dict[str, list[int]] = {}
    units: list[dict] = []
    skipped: list[dict] = []
    spent = 0.0

    def reserve(with_extra: dict[str, list[int]] | None = None) -> float:
        """Final evaluation: the base model and every arm that could be
        reported on the test split, each with all of its trained seeds."""
        merged = {**included, **(with_extra or {})}
        runs = sum(len(v) for k, v in merged.items())
        return test_eval * (runs + (1 if exp.baselines.base_model else 0))

    def add(unit: dict) -> None:
        nonlocal spent
        units.append(unit)
        spent += unit["estimate_fte"]

    if exp.baselines.base_model:
        add({"unit": "base-model evaluation", "stage": "evaluation", "estimate_fte": dev_eval, "essential": True})
    if uses_less:
        add({"unit": "warmup", "stage": "warmup", "estimate_fte": cost.warmup(), "essential": True})
        add({"unit": "gradient scoring", "stage": "scoring", "estimate_fte": cost.scoring(), "essential": True})
    for trial in primary:
        included[trial.key] = [seeds[0]]
        add({"unit": f"train+evaluate {trial.key} seed {seeds[0]}", "stage": "pilot_training", "trial": trial.key,
             "seed": seeds[0], "estimate_fte": trial_cost(trial), "essential": True})
    problems = []
    if spent + reserve() > exp.budget.total_compute:
        problems.append(
            f"The essential units need about {spent + reserve():,.0f} FTE but budget.total_compute is "
            f"{exp.budget.total_compute:,.0f}. Raise the budget, lower selection.token_budget or epochs, "
            "or drop a policy.")

    def try_group(label: str, group: list[tuple[Trial, int]]) -> bool:
        extra: dict[str, list[int]] = {}
        for trial, seed in group:
            extra[trial.key] = sorted(set(included.get(trial.key, []) + extra.get(trial.key, []) + [seed]))
        group_cost = sum(trial_cost(t) for t, _ in group)
        if problems or spent + group_cost + reserve(extra) > exp.budget.total_compute:
            skipped.append({"what": label, "reason": "does not fit in budget.total_compute",
                            "estimate_fte": group_cost})
            return False
        included.update(extra)
        for trial, seed in group:
            add({"unit": f"train+evaluate {trial.key} seed {seed}", "stage": "pilot_training",
                 "trial": trial.key, "seed": seed, "estimate_fte": trial_cost(trial), "essential": False})
        return True

    used = [seeds[0]]
    for seed in seeds[1:]:
        if not try_group(f"seed {seed} for the primary arms", [(t, seed) for t in primary]):
            break
        used.append(seed)
    wants_matched = exp.baselines.random_matched_compute and uses_less
    if wants_matched and not matched:
        skipped.append({
            "what": "matched-compute baseline",
            "reason": "the scoring overhead is worth more training tokens than the candidate pool holds, so "
                      "no random arm can spend the same compute; training on the whole pool costs less than "
                      "the model-aware arm and is the only reference", "estimate_fte": None})
    for trial in matched:
        try_group(f"matched-compute baseline ({trial.key})", [(trial, s) for s in used])
    for trial in search:
        try_group(f"search trial {trial.key}", [(trial, s) for s in used])
    for trial in full:
        if try_group("full-data training", [(trial, s) for s in used]) or len(used) == 1:
            continue
        if try_group("full-data training (one seed)", [(trial, used[0])]):
            # The all-seeds attempt just recorded is superseded by the fallback.
            skipped[-1] = {"what": f"full-data training for seeds {used[1:]}",
                           "reason": "does not fit in budget.total_compute", "estimate_fte": None}
        else:
            skipped.pop()  # keep one entry for full-data training, not two
    if len(declared_search) > len(search):
        skipped.append({"what": f"{len(declared_search) - len(search)} search trial(s) beyond search.max_trials",
                        "reason": "outside the bounded search budget", "estimate_fte": None})

    trials = [t for t in (*primary, *matched, *search, *full) if t.key in included]
    total = spent + reserve()
    return Plan(trials=trials, seeds=included, units=units, skipped=skipped, reserve_final=reserve(),
                estimate=total, budget=exp.budget.total_compute, fits=not problems, problems=problems,
                matched_compute_feasible=not (wants_matched and not matched))


def plan_to_dict(prep: Prepared, plan: Plan, *, backend_info: dict | None = None) -> dict:
    """The JSON form of `optimize plan` (also saved as plan.json by `run`)."""
    exp = prep.exp
    cost = CostModel(prep)
    by_stage: dict[str, float] = {}
    for unit in plan.units:
        by_stage[unit["stage"]] = by_stage.get(unit["stage"], 0.0) + unit["estimate_fte"]
    by_stage["final_evaluation"] = plan.reserve_final
    run_dir = exp.output.run_dir
    return {
        "schema_version": 1,
        "kind": "optimize-plan",
        "name": exp.name,
        "scale": exp.scale,
        "objective": exp.objective,
        "contract_fingerprint": exp.fingerprint(),
        "fits_budget": plan.fits,
        "problems": plan.problems,
        "data": {
            "candidates": {"path": str(prep.candidates.path), "records": len(prep.candidates.records),
                           "sha256": prep.candidates.sha256, "warnings_by_rule": prep.candidates.warnings_by_rule},
            "pool": {"records": len(prep.pool), "tokens": sum(prep.pool_tokens),
                     "excluded_for_leakage": prep.excluded["leakage"],
                     "excluded_for_length": prep.excluded["overlength"]},
            "dev": {"path": str(prep.dev.path), "records": len(prep.dev.records),
                    "selection_records": len(prep.dev_selection), "comparison_records": len(prep.dev_compare),
                    "shared_halves": exp.data.dev_selection_fraction >= 1.0,
                    "selection_tokens": prep.dev_tokens["selection"],
                    "comparison_tokens": prep.dev_tokens["comparison"]},
            "test": {"path": str(exp.data.test.path), "records": prep.test["records"],
                     "tokens": prep.test.get("tokens"),
                     "note": "read for leakage fingerprints and token counts only; examples stay sealed "
                             "until the candidate is frozen"},
            "token_counts": "exact (model tokenizer and chat template)" if prep.exact_tokens else
                            "estimated (~4 characters per token); install parallelogram[optimize] and make "
                            "the tokenizer available locally for exact counts",
            "target_capabilities": list(prep.target_capabilities),
            "retain_capabilities": list(prep.retain_capabilities),
        },
        "leakage": prep.leakage.to_dict(limit=20),
        "matrix": [{**t.to_dict(), "seeds": plan.seeds[t.key],
                    "question": COMPARISON_NOTES[t.comparison]} for t in plan.trials],
        "units": plan.units,
        "not_planned": plan.skipped,
        "matched_compute_feasible": plan.matched_compute_feasible,
        "budget": {
            "unit": "forward-token-equivalents (FTE): 1 per forward token, "
                    f"+{exp.budget.backward_multiplier:g} per backward token",
            "total_compute": plan.budget, "estimated_total": plan.estimate,
            "estimated_by_stage": by_stage,
            "selection_token_budget": exp.selection.token_budget,
            "search_max_trials": exp.search.max_trials,
        },
        "resource_assumptions": {
            "device": exp.model.device, "dtype": exp.model.dtype, "max_seq_len": exp.model.max_seq_len,
            "model": exp.model.path, "model_downloads": "only with --allow-download",
            "epochs": exp.recipe.epochs, "batch_size": exp.recipe.batch_size,
            "warmup_checkpoints": cost.checkpoints() if "less" in exp.selection.policies else 0,
            "projection": {"kind": exp.selection.less.projection, "dim": exp.selection.less.projection_dim},
            "feature_storage_bytes": (cost.checkpoints() * len(prep.pool) * exp.selection.less.projection_dim * 4
                                      if "less" in exp.selection.policies
                                      and exp.selection.less.projection == "countsketch" else None),
            "selection_memory": "one value per candidate plus features for a shortlist; no pairwise matrix",
            **(backend_info or {}),
        },
        "expected_artifacts": [
            f"{run_dir}/contract.json", f"{run_dir}/plan.json", f"{run_dir}/pool.json",
            f"{run_dir}/leakage.json", f"{run_dir}/ledger.json", f"{run_dir}/current.json",
            f"{run_dir}/scores/*.npz",
            f"{run_dir}/trials/<trial>/seed-<n>/{{selection.json,result.json,adapter/}}",
            f"{run_dir}/frozen.json", f"{run_dir}/final.json", f"{run_dir}/report.json",
            f"{run_dir}/report.md", f"{exp.output.cache_dir}/{{warmup,features,devgrads}}/<key>/",
        ],
    }
