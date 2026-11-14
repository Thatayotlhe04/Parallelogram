"""Execute an experiment: score, select, train, compare, freeze, test.

The order of operations is the argument for trusting the result:

1. Guard the splits and size the plan. Nothing trains if the essential
   units do not fit the budget.
2. Evaluate the base model, run the scoring overhead (warmup, gradients),
   and charge both to the ledger.
3. For every arm and seed: select, train from the same base weights,
   evaluate on the comparison half of the development split.
4. Choose a candidate from development results and freeze it.
5. Only then open the test vault and evaluate once.

After step 4 the run directory is sealed: no further trials are added to it,
so test results cannot steer another round of selection in the same run.

Every artifact is written atomically and every unit of work is keyed, so a
run that is interrupted (Ctrl-C, a crash, an out-of-memory stop) continues
from the last finished unit when the same command is run again.
"""
from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Any, Callable

from .. import __version__
from .backend import PILOT_PARAMETER_LIMIT, Backend, Encoded, TrainStats
from .budget import Cost, Ledger
from .cache import Cache, atomic_write, read_json, write_json
from .config import Experiment, digest, variant_spec
from .errors import ConfigError, SelectionInfeasible
from .less import (alignment, checkpoint_weights, combine, compute_features, dev_gradients, feature_rows,
                   run_warmup)
from .planner import CostModel, Prepared, Trial, build_plan, plan_to_dict, prepare, trial_slug
from .reporting import build_report, decide, load_results, render_markdown
from .selection import (Item, KCenterOrder, Selection, UtilityOrder, distribution, random_order, select,
                        shortlist, text_features, utility_order)
from .splits import Record, content_fingerprint, freeze_seal

CORE_SECTIONS = ("schema_version", "objective", "scale", "data", "model", "recipe", "evaluation")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, exp: Experiment, *, allow_download: bool = False, final: bool = True,
                 log: Callable[[str], None] | None = None):
        self.exp = exp
        self.final = final
        self.log = log or (lambda message: None)
        self.dir = exp.output.run_dir
        self.allow_download = allow_download
        self.not_performed: list[dict] = []
        self._selections: dict[str, Selection] = {}
        self._scores: dict[tuple, Any] = {}
        self._text_features = None
        self._overhead = 0.0
        self._coverage: dict[str, dict] = {}
        self._selection_seconds = 0.0
        self._current: dict[str, str] = {}
        self._matched_compute_feasible = True

    # ── setup ──────────────────────────────────────────────────────────────

    def _check_paths(self) -> None:
        exp = self.exp
        inputs = {exp.data.candidates.path.resolve(), exp.data.dev.path.resolve(), exp.data.test.path.resolve()}
        if any(self.dir.resolve() == p or self.dir.resolve() in p.parents for p in inputs):
            raise ConfigError("output.run_dir must not contain the input datasets")

    def _claim_directory(self) -> None:
        """Bind the run directory to this experiment, or refuse a mismatch.

        The binding covers the contract's data, model, recipe and evaluation
        sections *and* what they point at: the bytes of the three dataset
        files and the model, tokenizer and template revisions. Results
        measured on different data or a different model are not comparable,
        and an edited file under an unchanged path must not be mistaken for
        the same experiment. Selection, search, baselines and budget may
        change until the freeze, which is how later experiments try other
        subset sizes or parameters.
        """
        exp, prep = self.exp, self.prep
        self.dir.mkdir(parents=True, exist_ok=True)
        core = digest({"contract": exp.fingerprint(*CORE_SECTIONS), "model": self.backend.identity,
                       "data": {"candidates": prep.candidates.sha256, "dev": prep.dev.sha256,
                                "test": prep.test["sha256"]}})
        path = self.dir / "contract.json"
        if path.is_file():
            previous = read_json(path)
            if previous["core_fingerprint"] != core:
                raise ConfigError(
                    f"{self.dir} holds a run of a different experiment (the data, model, recipe or "
                    "evaluation changed, or a dataset file was edited). Use a new output.run_dir; caches "
                    "are keyed by content and will be reused where they still apply.")
        frozen = self.dir / "frozen.json"
        if frozen.is_file() and read_json(frozen)["contract_fingerprint"] != exp.fingerprint():
            raise ConfigError(
                f"{self.dir} is sealed: its candidate was frozen under a different contract. Changing "
                "selection or search after the freeze would let test results steer selection. Start a "
                "new output.run_dir.")
        write_json(path, {"schema_version": 1, "core_fingerprint": core,
                          "contract_fingerprint": exp.fingerprint(), "contract": exp.to_dict(),
                          "parallelogram": __version__})

    def _setup(self) -> None:
        exp = self.exp
        self._check_paths()
        self.backend = Backend(exp, allow_download=self.allow_download, log=self.log)
        self.np = self.backend.ml.np
        self.prep: Prepared = prepare(exp, self.backend.count_tokens)
        self._claim_directory()
        self.cache = Cache(exp.output.cache_dir)
        self.ledger = Ledger(self.dir / "ledger.json", exp.budget.total_compute, exp.budget.backward_multiplier)
        self.cost = CostModel(self.prep)
        prep = self.prep
        self.pool: list[Encoded] = [self.backend.encode(r) for r in prep.pool]
        self.items = [Item(r.id, e.tokens, r.attrs, r.raw_sha256) for r, e in zip(prep.pool, self.pool)]
        self.dev_selection = [self.backend.encode(r) for r in prep.dev_selection]
        self.dev_compare = [self.backend.encode(r) for r in prep.dev_compare]
        self.pilot = exp.scale == "pilot" or self.backend.parameters < PILOT_PARAMETER_LIMIT
        self.identity = {**self.backend.identity, "versions": self.backend.versions,
                         "recipe": digest({k: v for k, v in exp.to_dict()["recipe"].items() if k != "seeds"})}
        # Content, not just ids: an edited record that keeps its id must miss every cache.
        self.pool_fingerprint = digest([(r.id, r.raw_sha256, e.tokens) for r, e in zip(prep.pool, self.pool)])
        self.dev_fingerprint = digest({"records": content_fingerprint(prep.dev_compare),
                                       "metrics": list(exp.evaluation.metrics), "identity": self.identity})
        write_json(self.dir / "leakage.json", prep.leakage.to_dict())
        write_json(self.dir / "pool.json", {
            "source": {"path": str(prep.candidates.path), "sha256": prep.candidates.sha256,
                       "records": len(prep.candidates.records)},
            "excluded": prep.excluded,
            "records": [{"id": r.id, "line": r.line_no, "raw_sha256": r.raw_sha256, "tokens": e.tokens,
                         "attrs": r.attrs} for r, e in zip(prep.pool, self.pool)],
        })
        write_json(self.dir / "setup.json", {
            "model": {**self.backend.identity, "parameters": self.backend.parameters,
                      "device": exp.model.device, "versions": self.backend.versions},
            "pilot": self.pilot,
            "pilot_reason": ("scale: pilot" if exp.scale == "pilot" else
                             f"model has fewer than {PILOT_PARAMETER_LIMIT:,} parameters") if self.pilot else None,
            "target_capabilities": list(prep.target_capabilities),
            "retain_capabilities": list(prep.retain_capabilities),
        })

    # ── evaluation ─────────────────────────────────────────────────────────

    def _evaluate(self, model: Any, records: list[Record], encoded: list[Encoded]) -> tuple[dict, Cost]:
        """Per-example metrics, keyed by record id. No text is kept."""
        losses, cost = self.backend.losses(model, encoded)
        out = {"ids": [r.id for r in records], "capability": [self.prep.capability(r) for r in records],
               "loss": losses}
        if "exact_match" in self.exp.evaluation.metrics:
            out["exact_match"], generation = self.backend.exact_match(model, records)
            cost = cost + generation
        return out, cost

    def _base_eval(self) -> None:
        path = self.dir / "base.json"
        if not self.exp.baselines.base_model:
            return
        if path.is_file() and read_json(path).get("dev_fingerprint") == self.dev_fingerprint:
            return
        self.log("evaluating the base model on development data")
        result, cost = self._evaluate(self.backend.model, self.prep.dev_compare, self.dev_compare)
        self.ledger.charge(f"eval:base:dev:{self.dev_fingerprint[:12]}", "evaluation", cost)
        write_json(path, {"dev": result, "dev_fingerprint": self.dev_fingerprint})

    # ── scoring overhead ───────────────────────────────────────────────────

    def _score(self) -> None:
        """Warmup, candidate features and development gradients, all cached
        by content and all charged, whether computed now or reused."""
        if "less" not in self.exp.selection.policies:
            return
        spec = self.exp.selection.less
        warm_inputs = {"identity": self.identity, "pool": self.pool_fingerprint,
                       "warmup": {"fraction": spec.warmup_fraction, "epochs": spec.warmup_epochs,
                                  "max_steps": spec.warmup_max_steps, "seed": spec.warmup_seed}}
        projection = {"kind": spec.projection, "dim": spec.projection_dim, "seed": spec.projection_seed}
        feature_inputs = {"warmup": Cache.key(warm_inputs), "identity": self.identity,
                          "pool": self.pool_fingerprint, "projection": projection,
                          "optimizer_aware": spec.optimizer_aware, "shard_size": spec.shard_size}
        groups: dict[str, list[Encoded]] = {}
        contents: dict[str, list] = {}
        for record, encoded in zip(self.prep.dev_selection, self.dev_selection):
            groups.setdefault(self.prep.capability(record), []).append(encoded)
            contents.setdefault(self.prep.capability(record), []).append((record.id, record.raw_sha256))
        wanted = {*self.prep.target_capabilities, *self.prep.retain_capabilities}
        groups = {name: rows for name, rows in groups.items() if name in wanted}
        dev_inputs = {"warmup": Cache.key(warm_inputs), "identity": self.identity, "projection": projection,
                      "dev": digest({name: contents[name] for name in sorted(groups)})}
        keys = {"warmup": "less:warmup:" + Cache.key(warm_inputs),
                "features": "less:features:" + Cache.key(feature_inputs),
                "devgrads": "less:devgrads:" + Cache.key(dev_inputs)}
        # Scoring under changed settings in an existing run is new spending.
        # Refuse it here if it does not fit, rather than overdraw the budget.
        pending = ((self.cost.warmup() if keys["warmup"] not in self.ledger.entries else 0.0)
                   + (self.cost.scoring() if keys["features"] not in self.ledger.entries
                      or keys["devgrads"] not in self.ledger.entries else 0.0))
        if pending and not self.ledger.affords(pending):
            raise ConfigError(
                f"budget.total_compute does not cover the scoring overhead: about {pending:,.0f} FTE is "
                f"needed and {self.ledger.remaining:,.0f} is left in this run. Raise the budget or use a "
                "new output.run_dir.")
        self.checkpoints, cost, cached = run_warmup(self.backend, self.pool, spec, self.cache, warm_inputs, self.log)
        self.ledger.charge(keys["warmup"], "warmup", cost, cached=cached)
        self.features, cost, cached = compute_features(self.backend, self.pool, self.checkpoints, spec,
                                                       self.cache, feature_inputs, self.log)
        self.ledger.charge(keys["features"], "scoring", cost, cached=cached)
        self.devgrads, cost, cached = dev_gradients(self.backend, groups, self.checkpoints, spec, self.cache,
                                                    dev_inputs)
        self.ledger.charge(keys["devgrads"], "scoring", cost, cached=cached)
        # The overhead of *this* scoring configuration. Entries left by earlier
        # settings stay on the ledger and in the total, but they are not what
        # this candidate's selection cost.
        self._overhead = sum(self.ledger.entries[key]["fte"] for key in keys.values())

    def _scores_for(self, spec) -> Any:
        key = (spec.normalization, spec.checkpoint_weighting, spec.target_aggregation)
        if key not in self._scores:
            np = self.np
            per_capability = alignment(np, self.features, self.devgrads,
                                       checkpoint_weights(self.checkpoints, spec.checkpoint_weighting),
                                       len(self.pool), spec.normalization)
            self._scores[key] = combine(np, per_capability, self.prep.target_capabilities,
                                        self.prep.retain_capabilities, spec.target_aggregation)
        return self._scores[key]

    def _write_score_summary(self) -> None:
        if "less" not in self.exp.selection.policies:
            return
        np, spec = self.np, self.exp.selection.less
        scores = self._scores_for(spec)
        (self.dir / "scores").mkdir(exist_ok=True)
        columns = {"target": scores.target,
                   **({"retention": scores.retention} if scores.retention is not None else {}),
                   **{f"capability:{name}": column for name, column in scores.per_capability.items()}}
        atomic_write(self.dir / "scores" / "primary.npz", lambda f: np.savez(f, **columns))

        def stats(column: Any) -> dict:
            return {"min": float(column.min()), "median": float(np.median(column)), "max": float(column.max()),
                    "mean": float(column.mean()), "negative": int((column < 0).sum())}

        write_json(self.dir / "scores" / "summary.json", {
            "method": "LESS-inspired gradient alignment "
                      f"({'Adam-aware' if spec.optimizer_aware else 'SGD'} update direction, "
                      f"{spec.projection} projection to {self._feature_dim()} dims, "
                      f"{len(self.checkpoints)} checkpoint(s), {spec.normalization} normalization, "
                      f"{spec.checkpoint_weighting} checkpoint weights)",
            "note": "Estimated signals, not measurements. A score estimates how one update from a record "
                    "aligns with the gradient of the development loss. It is not a probability, not a task "
                    "metric, and a negative value is an estimated conflict signal rather than proof that "
                    "a record is wrong or that deleting it will improve training.",
            "records": len(self.pool),
            "ids": "pool.json records, same order as scores/primary.npz",
            "target": stats(scores.target),
            "target_capabilities": list(scores.target_capabilities),
            "retention": stats(scores.retention) if scores.retention is not None else None,
            "retain_capabilities": list(scores.retain_capabilities),
            "by_capability": {name: stats(column) for name, column in sorted(scores.per_capability.items())},
        })

    def _feature_dim(self) -> int:
        first = next((self.features / "checkpoint-000").glob("shard-*.npy"))
        return int(self.np.load(first, mmap_mode="r").shape[1])

    # ── selection ──────────────────────────────────────────────────────────

    def _select(self, trial: Trial, seed: int) -> Selection:
        """The subset for one arm. Selection sees candidates, their token
        counts and attributes, and scores derived from the selection half of
        the development split. It is never given test data or test results."""
        np, exp = self.np, self.exp
        cache_key = f"{trial.key}:{seed}" if trial.policy == "random" else trial.key
        if cache_key in self._selections:
            return self._selections[cache_key]
        n = len(self.items)
        started = time.monotonic()
        if trial.policy == "full":
            selection = Selection("full", list(range(n)), {i: "policy:full" for i in range(n)},
                                  sum(i.tokens for i in self.items), sum(i.tokens for i in self.items), [],
                                  [i.id for i in self.items])
        else:
            available = None
            if trial.policy == "random":
                # A fresh draw per training seed: the spread across seeds then
                # reflects both which random subset was drawn and training noise.
                order = random_order(n, f"{exp.selection.seed}:{trial.key}:{seed}")
            elif trial.policy == "diversity":
                if self._text_features is None:
                    self._text_features = text_features(
                        np, (" ".join(m["content"] for m in r.messages) for r in self.prep.pool))
                order = KCenterOrder(np, self._text_features)
            else:
                spec = variant_spec(exp.selection.less, trial.overrides)
                scores = self._scores_for(spec)
                available = np.ones(n, dtype=bool)
                if spec.exclude_negative:
                    available &= scores.target > 0
                if spec.retention_floor is not None and scores.retention is not None:
                    available &= scores.retention >= spec.retention_floor
                if not available.any():
                    raise SelectionInfeasible(
                        "selection.less: exclude_negative / retention_floor leave no candidate to select from")
                if spec.diversity_lambda > 0:
                    short = shortlist(np, scores.target, [i.tokens for i in self.items], trial.token_budget,
                                      spec.pool_factor, available)
                    rows = feature_rows(np, self.features, len(self.checkpoints) - 1, short, spec.shard_size)
                    order = UtilityOrder(np, scores.target, short, rows, spec.diversity_lambda)
                else:
                    order = utility_order(np, scores.target)
            selection = select(np, self.items, trial.token_budget, exp.selection.coverage, order, available)
            selection.policy = trial.policy
        self._selections[cache_key] = selection
        self._selection_seconds += time.monotonic() - started
        fields = sorted({c.field for c in exp.selection.coverage} | set(exp.selection.report_fields))
        self._coverage.setdefault(trial.key, {
            "records": len(selection.indices), "tokens": selection.tokens, "constraints": selection.coverage,
            "distribution": distribution(self.items, selection.indices, fields)})
        return selection

    def _selection_record(self, trial: Trial, selection: Selection) -> dict:
        rows = []
        scores = None
        if trial.policy == "less":
            scores = self._scores_for(variant_spec(self.exp.selection.less, trial.overrides))
        for rank, index in enumerate(selection.indices):
            row = {"id": self.items[index].id, "rank": rank, "tokens": self.items[index].tokens,
                   "reason": selection.reasons[index]}
            if scores is not None:
                row["scores"] = scores.row(index)
            rows.append(row)
        return {"policy": trial.policy, "token_budget": trial.token_budget, "tokens": selection.tokens,
                "records": len(rows), "sha256": selection.sha256, "coverage": selection.coverage,
                "selected": rows}

    # ── trials ─────────────────────────────────────────────────────────────

    def _trial_dir(self, trial: Trial, seed: int) -> Path:
        return self.dir / "trials" / trial_slug(trial.key) / f"seed-{seed}"

    def _trial_fingerprint(self, trial: Trial, seed: int, selection: Selection) -> str:
        # Order matters as well as membership: batches are drawn from the
        # selection's order, so the same set in another order trains differently.
        return digest({"seed": seed, "selection": selection.sha256, "order": digest(selection.ids),
                       "trial": trial.to_dict(), "dev": self.dev_fingerprint})

    def _run_trial(self, trial: Trial, seed: int) -> None:
        directory = self._trial_dir(trial, seed)
        try:
            selection = self._select(trial, seed)
        except SelectionInfeasible as exc:
            self.not_performed.append({"what": f"{trial.key} seed {seed}", "reason": str(exc)})
            return
        fingerprint = self._trial_fingerprint(trial, seed, selection)
        result_path = directory / "result.json"
        index_key = f"{directory.parent.name}/{directory.name}"
        if result_path.is_file() and read_json(result_path).get("fingerprint") == fingerprint:
            self._mark_current(index_key, fingerprint)
            return
        estimate = self.cost.training(selection.tokens) + self.cost.dev_evaluation()
        reserve = self.cost.test_evaluation() * (len(self._current) + 2)
        if not self.ledger.affords(estimate + reserve):
            self.not_performed.append({
                "what": f"{trial.key} seed {seed}",
                "reason": "the remaining budget does not cover it plus the final evaluation "
                          f"({self.ledger.remaining:,.0f} FTE left)"})
            return
        self.log(f"training {trial.key} seed {seed}: {len(selection.indices)} records, {selection.tokens} tokens")
        examples = [self.pool[i] for i in selection.indices]
        # One ledger entry per execution, not per trial: a training pass that
        # is interrupted and redone was run twice and is charged twice.
        prefix = f"train:{trial_slug(trial.key)}:{seed}:{fingerprint[:12]}"
        attempt = sum(key.startswith(prefix) for key in self.ledger.entries)
        stats = TrainStats()
        with self.backend.adapter(seed) as model:
            try:
                self.backend.train(model, examples, seed=seed, stats=stats)
            finally:
                if stats.tokens:
                    self.ledger.charge(f"{prefix}:{attempt}", "pilot_training", stats.cost, trial=trial.key)
            self.backend.save_adapter(model, directory / "adapter")
            evaluation, eval_cost = self._evaluate(model, self.prep.dev_compare, self.dev_compare)
        self.ledger.charge(f"eval:{trial_slug(trial.key)}:{seed}:{fingerprint[:12]}:dev:{attempt}", "evaluation",
                           eval_cost, trial=trial.key)
        write_json(directory / "selection.json", self._selection_record(trial, selection))
        bm = self.exp.budget.backward_multiplier
        write_json(result_path, {
            "schema_version": 1, "trial": trial.to_dict(), "seed": seed, "fingerprint": fingerprint,
            "selection": {"sha256": selection.sha256, "records": len(selection.indices), "tokens": selection.tokens},
            "train": stats.to_dict(), "dev": evaluation,
            "cost": {"training_fte": stats.cost.fte(bm), "evaluation_fte": eval_cost.fte(bm),
                     "overhead_fte": self._overhead if trial.policy == "less" else 0.0},
        })
        self._mark_current(index_key, fingerprint)

    def _mark_current_reset(self) -> None:
        self._current = {}
        write_json(self.dir / "current.json", {"results": {}})

    def _mark_current(self, index_key: str, fingerprint: str) -> None:
        """Record that a finished result belongs to the current contract.

        A run directory can hold results from earlier settings (a different
        coverage minimum, a dropped search trial). They stay on disk, but the
        report, the freeze and the final evaluation use only what is listed
        here, so a stale subset can never be frozen or exported as current.
        """
        self._current[index_key] = fingerprint
        write_json(self.dir / "current.json", {"results": self._current})

    # ── freeze and final ───────────────────────────────────────────────────

    def _freeze(self) -> dict | None:
        path = self.dir / "frozen.json"
        if path.is_file():
            return read_json(path)
        base = read_json(self.dir / "base.json")["dev"] if (self.dir / "base.json").is_file() else None
        results = load_results(self.dir)
        decision = decide(self.exp.to_dict(), base, results, self.prep.target_capabilities,
                          self.prep.retain_capabilities,
                          matched_compute_feasible=self._matched_compute_feasible)
        if decision["candidate"] is None:
            self.not_performed.append({"what": "freeze and final held-out evaluation",
                                       "reason": decision["conclusion"]["statement"]})
            return None
        entry = results[decision["candidate"]]
        seed = sorted(entry["seeds"])[0]
        body = {
            "kind": "optimize-freeze", "schema_version": 1, "trial": decision["candidate"],
            "policy": entry["trial"]["policy"], "token_budget": entry["trial"]["token_budget"],
            "less_overrides": entry["trial"]["less_overrides"],
            "selection_seed": seed, "selection_sha256": entry["seeds"][seed]["selection"]["sha256"],
            "recipe_fingerprint": self.identity["recipe"], "contract_fingerprint": self.exp.fingerprint(),
            "basis": {"split": "dev (comparison half)", "primary_metric": decision["primary_metric"],
                      "arm_means": decision["arm_means"], "conclusion": decision["conclusion"]},
            "evaluated_arms": sorted(results), "frozen_at": _now(),
        }
        body["seal"] = freeze_seal(body)
        write_json(path, body)
        self.log(f"frozen: {body['trial']}")
        return body

    def _final(self, frozen: dict) -> None:
        """Open the vault and evaluate every trained arm once on test data.

        All arms are evaluated so the frozen candidate can be compared with
        its baselines on held-out data; none of it feeds back, because the
        run directory accepts no new trials after the freeze.
        """
        final_path = self.dir / "final.json"
        if final_path.is_file():
            return
        test = self.prep.vault.open(frozen)
        encoded = [self.backend.encode(r) for r in test.records]
        base_path = self.dir / "base.json"
        if self.exp.baselines.base_model:
            base = read_json(base_path)
            if "test" not in base:
                self.log("final evaluation: base model")
                result, cost = self._evaluate(self.backend.model, test.records, encoded)
                self.ledger.charge("eval:base:test", "final_evaluation", cost)
                write_json(base_path, {**base, "test": result})
        current = read_json(self.dir / "current.json")["results"]
        for index_key, fingerprint in sorted(current.items()):
            result_path = self.dir / "trials" / index_key / "result.json"
            result = read_json(result_path)
            if "test" in result or result["fingerprint"] != fingerprint:
                continue
            self.log(f"final evaluation: {result['trial']['key']} seed {result['seed']}")
            with self.backend.saved_adapter(result_path.parent / "adapter") as model:
                evaluation, cost = self._evaluate(model, test.records, encoded)
            self.ledger.charge(f"eval:{result_path.parent.parent.name}:{result['seed']}:"
                               f"{result['fingerprint'][:12]}:test", "final_evaluation", cost,
                               trial=result["trial"]["key"])
            write_json(result_path, {**result, "test": evaluation})
        exposure_path = self.exp.output.cache_dir / "test-exposure" / f"{test.fingerprint[:32]}.json"
        exposure = read_json(exposure_path) if exposure_path.is_file() else {"runs": []}
        if str(self.dir) not in [r["run_dir"] for r in exposure["runs"]]:
            exposure["runs"].append({"run_dir": str(self.dir), "contract_fingerprint": self.exp.fingerprint(),
                                     "at": _now()})
            write_json(exposure_path, exposure)
        write_json(final_path, {"schema_version": 1, "frozen_trial": frozen["trial"], "completed_at": _now(),
                                "test": {"records": len(test.records), "sha256": test.sha256},
                                "test_exposure": {"final_evaluations": len(exposure["runs"]),
                                                  "note": "final evaluations recorded for this exact test "
                                                          "split in this cache directory"}})

    # ── the whole thing ────────────────────────────────────────────────────

    def execute(self) -> dict:
        self._setup()
        plan = build_plan(self.prep)
        if not plan.fits:
            raise ConfigError(" ".join(plan.problems))
        sealed = (self.dir / "frozen.json").is_file()
        if not sealed:
            self._mark_current_reset()
            self._base_eval()
            self._score()
            self._write_score_summary()
            plan = build_plan(self.prep, self._overhead if "less" in self.exp.selection.policies else None)
            self._matched_compute_feasible = plan.matched_compute_feasible
            write_json(self.dir / "plan.json", plan_to_dict(self.prep, plan, backend_info={
                "model_parameters": self.backend.parameters, "library_versions": self.backend.versions}))
            started = time.monotonic()
            for trial in plan.trials:
                for seed in plan.seeds[trial.key]:
                    self._run_trial(trial, seed)
            # Selection runs on the CPU and passes no tokens through the model,
            # so it costs wall-clock time and zero FTE; it is still on the ledger.
            self.ledger.charge(f"selection:{digest(sorted(self._coverage))[:12]}", "selection",
                               Cost(wall_seconds=self._selection_seconds))
            write_json(self.dir / "coverage.json", {"trials": self._coverage})
            self.log(f"development trials done in {time.monotonic() - started:.1f}s")
        elif (self.dir / "plan.json").is_file():
            self._matched_compute_feasible = read_json(self.dir / "plan.json").get("matched_compute_feasible", True)
        if self.final:
            frozen = self._freeze()
            if frozen is not None:
                self._final(frozen)
        else:
            self.not_performed.append({"what": "freeze and final held-out evaluation",
                                       "reason": "run was asked to stop before the final evaluation"})
        # A sealed run does not redo development trials, so what they left
        # undone is carried over rather than forgotten.
        carried = (read_json(self.dir / "not_performed.json")["items"]
                   if sealed and (self.dir / "not_performed.json").is_file() else [])
        known = {item["what"] for item in self.not_performed}
        write_json(self.dir / "not_performed.json", {
            "items": [item for item in carried if item["what"] not in known
                      and not (self.final and item["what"].startswith("freeze and final"))] + self.not_performed})
        report = build_report(self.dir)
        write_json(self.dir / "report.json", report)
        atomic_write(self.dir / "report.md", lambda f: f.write(render_markdown(report)), mode="w")
        return report


def run(exp: Experiment, *, allow_download: bool = False, final: bool = True,
        log: Callable[[str], None] | None = None) -> dict:
    """Run (or resume) an experiment and return its report."""
    return Run(exp, allow_download=allow_download, final=final, log=log).execute()
