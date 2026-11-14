"""Read a run directory and say what was estimated, measured and not done.

A report is rebuilt from the artifacts on disk every time, with the standard
library only, so it can be produced on a machine without the ML extras and it
cannot drift from what the run actually wrote.

The sections are kept apart on purpose, because they are different kinds of
statement:

* estimated signals: gradient-alignment scores. Heuristics about a loss.
* development results: measured, but they also steered the choice.
* final results: measured once on held-out data after the freeze.
* compute: everything charged, including the optimizer's own overhead.
* coverage: what selection did to the composition of the data.
* limitations and experiments not performed.

The decision logic is here too (`decide`), because the freeze and the report
must apply the same rule.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from .budget import Ledger
from .cache import read_json
from .errors import OptimizeError
from .stats import (CONTRADICTED, INSUFFICIENT, MIN_SEEDS, SUPPORTED, across_seeds, bootstrap_ci, compare,
                    mean)

HIGHER_IS_BETTER = {"loss": False, "exact_match": True}
PILOT_NOTE = ("This is a pilot on a small model. It shows that the pipeline runs end to end and what it "
              "measured here; it does not establish an improvement for larger models or other data.")


# ── reading results ─────────────────────────────────────────────────────────

def load_results(run_dir: Path, kind: str = "result.json") -> dict[str, dict]:
    """`{trial_key: {"trial": {...}, "seeds": {seed: result}}}` for finished runs."""
    out: dict[str, dict] = {}
    index = Path(run_dir) / "current.json"
    # Only results that belong to the current contract count. A directory can
    # also hold results from earlier settings; they are ignored, not mixed in.
    current = read_json(index)["results"] if index.is_file() else None
    for path in sorted(Path(run_dir).glob(f"trials/*/seed-*/{kind}")):
        try:
            result = read_json(path)
        except (OSError, ValueError):
            continue
        if current is not None and current.get(f"{path.parent.parent.name}/{path.parent.name}") != result.get("fingerprint"):
            continue
        entry = out.setdefault(result["trial"]["key"], {"trial": result["trial"], "seeds": {}})
        entry["seeds"][result["seed"]] = result
    return out


def values(evaluation: dict, metric: str, capabilities: Sequence[str] | None = None) -> list[float]:
    """Per-example values of one metric, optionally for some capabilities."""
    column = evaluation[metric]
    if capabilities is None:
        return list(column)
    wanted = set(capabilities)
    return [v for v, c in zip(column, evaluation["capability"]) if c in wanted]


def run_mean(evaluation: dict, metric: str, capabilities: Sequence[str] | None = None) -> float | None:
    v = values(evaluation, metric, capabilities)
    return mean(v) if v else None


def _seed_average(evaluations: Sequence[dict], metric: str, capabilities: Sequence[str]) -> list[float]:
    """Each example's metric averaged over seeds (examples in a fixed order)."""
    columns = [values(e, metric, capabilities) for e in evaluations]
    return [mean(row) for row in zip(*columns)]


def summarize(evaluations: dict[int, dict], metrics: Sequence[str], target: Sequence[str],
              retain: Sequence[str], samples: int) -> dict:
    """One arm's results: per metric, the spread across seeds and the
    evaluation interval, on target capabilities and per capability."""
    seeds = sorted(evaluations)
    out: dict[str, Any] = {"seeds": seeds, "metrics": {}, "by_capability": {}}
    capabilities = sorted({c for e in evaluations.values() for c in e["capability"] if c is not None})
    for metric in metrics:
        per_seed = [run_mean(evaluations[s], metric, target) for s in seeds]
        pooled = _seed_average([evaluations[s] for s in seeds], metric, target)
        out["metrics"][metric] = {
            "scope": "target capabilities",
            "across_seeds": across_seeds(per_seed),
            "evaluation_ci95": bootstrap_ci(pooled, samples),
            "examples": len(pooled),
        }
        for capability in capabilities:
            per_cap = [run_mean(evaluations[s], metric, [capability]) for s in seeds]
            role = "target" if capability in target else "retain" if capability in retain else "other"
            out["by_capability"].setdefault(capability, {"role": role})[metric] = across_seeds(per_cap)
    return out


# ── decision ────────────────────────────────────────────────────────────────

def _oriented(value: float, metric: str) -> float:
    return value if HIGHER_IS_BETTER[metric] else -value


def _evals(entry: dict, split: str) -> dict[int, dict]:
    return {seed: result[split] for seed, result in entry["seeds"].items() if split in result}


def retention(entry_evals: dict[int, dict], base: dict, metric: str, retain: Sequence[str],
              tolerance: float) -> dict:
    """Did the arm keep each retained capability within tolerance of the base model?"""
    rows = []
    for capability in retain:
        trained = mean([run_mean(e, metric, [capability]) for e in entry_evals.values()])
        reference = run_mean(base, metric, [capability])
        change = _oriented(trained, metric) - _oriented(reference, metric)
        rows.append({"capability": capability, "base": reference, "trained": trained,
                     "change_oriented": change, "within_tolerance": change >= -tolerance})
    return {"ok": all(r["within_tolerance"] for r in rows), "tolerance": tolerance, "capabilities": rows}


def _comparison(question: str, kind: str, candidate: tuple[str, dict[int, dict]],
                baseline: tuple[str, dict[int, dict]] | None, base_eval: dict | None, metric: str,
                target: Sequence[str], samples: int) -> dict:
    key, cand = candidate
    cand_seeds = [run_mean(cand[s], metric, target) for s in sorted(cand)]
    cand_examples = _seed_average([cand[s] for s in sorted(cand)], metric, target)
    if baseline is None:  # against the deterministic base model
        base_value = run_mean(base_eval, metric, target)
        result = compare(cand_seeds, [base_value] * max(len(cand_seeds), 1),
                         higher_is_better=HIGHER_IS_BETTER[metric], candidate_examples=cand_examples,
                         baseline_examples=values(base_eval, metric, target), samples=samples)
        return {"question": question, "kind": kind, "candidate": key, "baseline": "base model", **result}
    bkey, base = baseline
    result = compare(cand_seeds, [run_mean(base[s], metric, target) for s in sorted(base)],
                     higher_is_better=HIGHER_IS_BETTER[metric], candidate_examples=cand_examples,
                     baseline_examples=_seed_average([base[s] for s in sorted(base)], metric, target),
                     samples=samples)
    return {"question": question, "kind": kind, "candidate": key, "baseline": bkey, **result}


def decide(contract: dict, base: dict | None, results: dict[str, dict], target: Sequence[str],
           retain: Sequence[str], split: str = "dev", candidate: str | None = None,
           matched_compute_feasible: bool = True) -> dict:
    """Pick (or, on the test split, take as given) the candidate and weigh the evidence.

    On development data the candidate is the arm within the selection token
    budget with the best mean primary metric that keeps retained
    capabilities within tolerance and beats the base model's mean. On the test split the
    candidate is whatever was frozen: nothing is chosen there.

    The headline, `conclusion`, is about the stated goal: a better
    quality-for-total-compute tradeoff than strong baselines. It is
    `supported` only if the candidate is a model-aware (`less`) arm that
    beats both the strongest matched-token baseline and the matched-compute
    baseline beyond noise. Anything short of that is reported as what it is.
    """
    evaluation = contract["evaluation"]
    metric, samples = evaluation["primary_metric"], evaluation["bootstrap_samples"]
    tolerance = evaluation["retention_tolerance"]
    arms = {key: (entry["trial"], _evals(entry, split)) for key, entry in results.items()}
    arms = {key: value for key, value in arms.items() if value[1]}
    scored = {key: mean([run_mean(e, metric, target) for e in evals.values()]) for key, (_, evals) in arms.items()}
    keeps = {key: (retention(evals, base, metric, retain, tolerance) if retain and base else None)
             for key, (_, evals) in arms.items()}
    notes: list[str] = []

    if candidate is None:
        # Only arms inside the contract's selection token budget can be
        # recommended. The matched-compute and full-data arms train on more
        # data by construction; picking the best score among arms of
        # different sizes would just pick the largest.
        ceiling = contract["selection"]["token_budget"]
        eligible = [key for key, (trial, _) in arms.items() if trial["comparison"] == "matched_tokens"
                    and trial["token_budget"] <= ceiling and (keeps[key] is None or keeps[key]["ok"])]
        if base is not None:
            floor = _oriented(run_mean(base, metric, target), metric)
            beaten = [key for key in eligible if _oriented(scored[key], metric) > floor]
            if eligible and not beaten:
                notes.append("No trained subset arm scored better than the base model on the primary "
                             "development metric, so nothing is recommended.")
            eligible = beaten
        if eligible:
            candidate = max(eligible, key=lambda k: (_oriented(scored[k], metric),
                                                     -(arms[k][0]["token_budget"] or 0), k))
        elif not notes:
            notes.append("No trained subset arm kept the retained capabilities within tolerance.")

    comparisons: list[dict] = []
    conclusion = {"status": INSUFFICIENT, "basis": split}
    if candidate is not None and candidate in arms:
        trial, evals = arms[candidate]
        cand = (candidate, evals)
        budget = trial["token_budget"]
        peers = [key for key, (other, _) in arms.items() if key != candidate and other["policy"] != "full"
                 and other["comparison"] == "matched_tokens" and other["token_budget"] == budget
                 and other["policy"] != trial["policy"]]
        strongest = max(peers, key=lambda k: _oriented(scored[k], metric)) if peers else None
        matched_tokens = matched_compute = None
        if base is not None:
            comparisons.append(_comparison("Does training on the candidate subset beat not training at all?",
                                           "vs_base_model", cand, None, base, metric, target, samples))
        if strongest is not None:
            matched_tokens = _comparison(
                "At the same number of training tokens, does the candidate beat the strongest other policy?",
                "matched_tokens", cand, (strongest, arms[strongest][1]), None, metric, target, samples)
            comparisons.append(matched_tokens)
        compute_arm = next((key for key, (other, _) in arms.items() if other["comparison"] == "matched_compute"), None)
        if compute_arm is not None and compute_arm != candidate and trial["policy"] == "less":
            matched_compute = _comparison(
                "Counting the scoring overhead, does the candidate beat random selection that spent the "
                "same total compute on more training data?",
                "matched_compute", cand, (compute_arm, arms[compute_arm][1]), None, metric, target, samples)
            comparisons.append(matched_compute)
        if "full" in arms:
            comparisons.append(_comparison(
                "Does the candidate subset match or beat training on all candidates?", "full_data", cand,
                ("full", arms["full"][1]), None, metric, target, samples))

        if trial["policy"] != "less":
            chosen = ("The best arm within the selection budget on development data was"
                      if split == "dev" else "The candidate frozen from development data was")
            conclusion.update(status=INSUFFICIENT, statement=(
                f"{chosen} {candidate!r}, which is not model-aware selection. "
                "There is no evidence here that gradient scoring improved the quality/compute tradeoff."))
            aware = [k for k, (other, _) in arms.items() if other["policy"] == "less"
                     and other["token_budget"] == budget]
            if aware:
                best_aware = max(aware, key=lambda k: _oriented(scored[k], metric))
                versus = _comparison(
                    "Is the candidate better than the best model-aware arm at the same training tokens?",
                    "vs_model_aware", cand, (best_aware, arms[best_aware][1]), None, metric, target, samples)
                if not any(c["baseline"] == best_aware for c in comparisons):
                    comparisons.append(versus)
                if versus["verdict"] == SUPPORTED:
                    conclusion.update(status="not_supported", statement=(
                        f"On {split} data, {candidate!r} beat the best model-aware arm beyond noise: "
                        "gradient scoring did not improve the quality/compute tradeoff here."))
        elif matched_compute is None and not matched_compute_feasible:
            # Scoring cost at least as much as training on everything, so the
            # candidate spent more in total than full-data training did.
            full = next((c for c in comparisons if c["kind"] == "full_data"), None)
            if full is not None and full["verdict"] == CONTRADICTED:
                conclusion.update(status="not_supported", statement=(
                    f"On {split} data, training on the whole candidate pool cost less in total than scoring "
                    f"plus {candidate!r} and scored better beyond noise: gradient scoring did not improve "
                    "the quality/compute tradeoff here."))
            else:
                conclusion.update(status=INSUFFICIENT, statement=(
                    "The scoring overhead was worth more training tokens than the candidate pool holds, so "
                    "no matched-compute baseline exists, and the candidate spent more total compute than "
                    "training on the whole pool. An improved quality/compute tradeoff cannot be concluded "
                    "from this run, whatever the candidate scored."))
        elif matched_tokens is None or matched_compute is None:
            missing = "a matched-token baseline" if matched_tokens is None else "the matched-compute baseline"
            conclusion.update(status=INSUFFICIENT, statement=(
                f"{missing.capitalize()} was not run, so the quality/compute tradeoff cannot be judged."))
        else:
            verdicts = {matched_tokens["verdict"], matched_compute["verdict"]}
            if verdicts == {SUPPORTED}:
                conclusion.update(status=SUPPORTED, statement=(
                    f"On {split} data, {candidate!r} beat both the strongest matched-token baseline and the "
                    "matched-compute baseline beyond run-to-run variation and evaluation uncertainty."))
            elif CONTRADICTED in verdicts:
                conclusion.update(status="not_supported", statement=(
                    f"On {split} data, a baseline beat {candidate!r} beyond noise: gradient scoring did not "
                    "improve the quality/compute tradeoff here."))
            else:
                conclusion.update(status=INSUFFICIENT, statement=(
                    f"On {split} data, the differences between {candidate!r} and the baselines are within "
                    "noise, or there were fewer than three seeds. No conclusion is justified."))
    else:
        conclusion["statement"] = notes[0] if notes else "No trained arm is available to judge."
    return {
        "split": split, "primary_metric": metric, "higher_is_better": HIGHER_IS_BETTER[metric],
        "candidate": candidate, "arm_means": scored, "retention": keeps, "comparisons": comparisons,
        "conclusion": conclusion, "notes": notes, "minimum_seeds_for_a_conclusion": MIN_SEEDS,
    }


# ── the report ──────────────────────────────────────────────────────────────

def _optional(path: Path) -> Any:
    return read_json(path) if path.is_file() else None


def build_report(run_dir: str | Path) -> dict:
    """Assemble the report for a run directory (complete or partial)."""
    run_dir = Path(run_dir)
    contract_file = _optional(run_dir / "contract.json")
    if contract_file is None:
        raise OptimizeError(f"{run_dir} is not an optimize run directory (no contract.json)")
    contract = contract_file["contract"]
    setup = _optional(run_dir / "setup.json") or {}
    plan = _optional(run_dir / "plan.json") or {}
    ledger = None
    if (run_dir / "ledger.json").is_file():
        ledger = Ledger(run_dir / "ledger.json", contract["budget"]["total_compute"],
                        contract["budget"]["backward_multiplier"]).summary()
    base = _optional(run_dir / "base.json")
    frozen = _optional(run_dir / "frozen.json")
    final = _optional(run_dir / "final.json")
    scores = _optional(run_dir / "scores" / "summary.json")
    not_performed = (_optional(run_dir / "not_performed.json") or {"items": []})["items"]
    results = load_results(run_dir)
    target, retain = setup.get("target_capabilities", []), setup.get("retain_capabilities", [])
    evaluation = contract["evaluation"]
    metrics, samples = evaluation["metrics"], evaluation["bootstrap_samples"]
    pilot = bool(setup.get("pilot", True))
    feasible = plan.get("matched_compute_feasible", True)

    def arms(split: str) -> list[dict]:
        rows = []
        for key, entry in results.items():
            evals = _evals(entry, split)
            if not evals:
                continue
            first = entry["seeds"][sorted(entry["seeds"])[0]]
            rows.append({
                **entry["trial"],
                "selected_records": first["selection"]["records"],
                "training_tokens_per_epoch": first["selection"]["tokens"],
                "compute_fte": {"training": mean([entry["seeds"][s]["cost"]["training_fte"] for s in evals]),
                                "attributed_overhead": first["cost"].get("overhead_fte", 0.0),
                                "note": "per seed; overhead (warmup + scoring) is paid once and attributed "
                                        "to model-aware arms"},
                **summarize(evals, metrics, target, retain, samples),
            })
        return rows

    development = None
    if results:
        development = {
            "note": "Measured on the comparison half of the development split. These results chose the "
                    "candidate, so they are optimistic for it; the held-out test results are the check.",
            "base_model": (summarize({0: base["dev"]}, metrics, target, retain, samples)
                           if base and "dev" in base else None),
            "arms": arms("dev"),
            "decision": decide(contract, base["dev"] if base and "dev" in base else None, results, target, retain,
                               matched_compute_feasible=feasible),
        }
    final_section = None
    if final is not None and frozen is not None:
        final_section = {
            "note": "Measured once on the test split after the candidate and recipe were frozen. Nothing "
                    "was selected or tuned using these numbers.",
            "frozen": {k: frozen[k] for k in ("trial", "policy", "token_budget", "selection_sha256", "frozen_at")},
            "base_model": (summarize({0: base["test"]}, metrics, target, retain, samples)
                           if base and "test" in base else None),
            "arms": arms("test"),
            "decision": decide(contract, base.get("test") if base else None, results, target, retain,
                               split="test", candidate=frozen["trial"], matched_compute_feasible=feasible),
            "test_exposure": final.get("test_exposure"),
        }

    limitations = [
        "Gradient-alignment scores are first-order estimates about a differentiable loss. They are not "
        "calibrated probabilities and are not task metrics; a negative score is an estimated conflict "
        "signal, not proof that a record is wrong or that removing it helps.",
        "One training path is implemented: LoRA instruction tuning of a causal LM on plain chat records.",
        "Near-duplicate leakage detection is lexical; paraphrased overlap between splits is not detected.",
        "Compute is an accounting model (forward-token-equivalents), not measured FLOPs or energy.",
    ]
    if pilot:
        limitations.insert(0, PILOT_NOTE)
    if contract["data"]["dev_selection_fraction"] >= 1.0:
        limitations.append("The same development examples were used to compute gradient targets and to "
                           "compare policies, which favors the gradient selector on development data.")
    if development and len(contract["recipe"]["seeds"]) < MIN_SEEDS:
        limitations.append(f"Fewer than {MIN_SEEDS} seeds were configured, so no comparison can be called "
                           "supported.")
    if final is None and not any(item["what"].startswith("freeze and final") for item in not_performed):
        not_performed = [*not_performed, {"what": "final held-out evaluation",
                                          "reason": "the candidate has not been frozen and evaluated yet"}]

    conclusion = (final_section or development or {}).get("decision", {}).get("conclusion") or {
        "status": INSUFFICIENT, "basis": None, "statement": "No trained arm has finished."}
    return {
        "schema_version": 1,
        "kind": "optimize-report",
        "name": contract["name"],
        "run_directory": str(run_dir),
        "status": "final" if final is not None else "frozen" if frozen is not None
                  else "development" if results else "started",
        "label": "pilot" if pilot else "benchmark",
        "conclusion": {**conclusion, "pilot": pilot},
        "model": setup.get("model"),
        "estimated_signals": scores,
        "development_results": development,
        "final_results": final_section,
        "compute": ledger,
        "coverage": _optional(run_dir / "coverage.json"),
        "leakage": _optional(run_dir / "leakage.json"),
        "limitations": limitations,
        "not_performed": [*plan.get("not_planned", []), *not_performed],
    }


# ── rendering ───────────────────────────────────────────────────────────────

def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _interval(pair: Sequence[float] | None) -> str:
    return "n/a" if not pair else f"[{pair[0]:.4g}, {pair[1]:.4g}]"


def _arm_table(section: dict, metric: str) -> list[str]:
    lines = [f"| arm | comparison | seeds | {metric} mean | across-seed std | min..max | eval 95% CI |",
             "|---|---|---|---|---|---|---|"]
    rows = []
    if section.get("base_model"):
        rows.append(("base model (no training)", "reference", section["base_model"]))
    rows.extend((arm["key"], arm["comparison"], arm) for arm in section["arms"])
    for name, comparison, arm in rows:
        m = arm["metrics"][metric]
        a = m["across_seeds"]
        lines.append(f"| {name} | {comparison} | {a['n']} | {_fmt(a['mean'])} | {_fmt(a['std'])} | "
                     f"{_fmt(a['min'])}..{_fmt(a['max'])} | {_interval(m['evaluation_ci95'])} |")
    return lines


def _decision_lines(decision: dict) -> list[str]:
    lines = [f"Candidate: `{decision['candidate']}`" if decision["candidate"] else "Candidate: none", ""]
    for c in decision["comparisons"]:
        lines.append(f"- {c['question']} (`{c['candidate']}` vs `{c['baseline']}`): **{c['verdict']}**. "
                     f"Difference {_fmt(c['difference'])} (positive favors the candidate); across-seed 95% interval "
                     f"{_interval(c['across_seed_ci95'])}; evaluation 95% interval "
                     f"{_interval(c['evaluation_ci95'])}. {c['reason'].capitalize()}.")
    for key, keep in (decision.get("retention") or {}).items():
        if keep and not keep["ok"]:
            lines.append(f"- `{key}` lost a retained capability beyond the tolerance of {keep['tolerance']}.")
    lines.extend(f"- {note}" for note in decision.get("notes", []))
    return lines


def render_markdown(report: dict) -> str:
    """The report as Markdown. Plain text, no emojis, sections in a fixed order."""
    conclusion = report["conclusion"]
    out = [f"# Optimize report: {report['name']}", "",
           f"Status: {report['status']}. Label: {report['label']}.", "",
           f"**Conclusion ({conclusion.get('basis') or 'no'} data): {conclusion['status'].replace('_', ' ')}.** "
           f"{conclusion.get('statement', '')}"]
    if conclusion.get("pilot"):
        out += ["", PILOT_NOTE]

    out += ["", "## 1. Estimated training-value signals", ""]
    scores = report["estimated_signals"]
    if scores:
        out += [scores["note"], "",
                f"Method: {scores['method']}. Records scored: {scores['records']}. Target alignment: min "
                f"{_fmt(scores['target']['min'])}, median {_fmt(scores['target']['median'])}, max "
                f"{_fmt(scores['target']['max'])}; {scores['target']['negative']} negative."]
        if scores.get("retention"):
            out.append(f"Retention alignment (kept separate): min {_fmt(scores['retention']['min'])}, median "
                       f"{_fmt(scores['retention']['median'])}; {scores['retention']['negative']} negative.")
    else:
        out.append("No gradient scores were computed in this run.")

    for title, key in (("## 2. Measured development results", "development_results"),
                       ("## 3. Final held-out results", "final_results")):
        out += ["", title, ""]
        section = report[key]
        if not section:
            out.append("Not available: " + ("no arm has finished." if key == "development_results"
                                             else "the final evaluation has not been run."))
            continue
        out += [section["note"], ""]
        for metric in section["arms"][0]["metrics"] if section["arms"] else []:
            out += _arm_table(section, metric) + [""]
        out += _decision_lines(section["decision"])
        exposure = section.get("test_exposure")
        if exposure and exposure["final_evaluations"] > 1:
            out += ["", f"This test split has now been used for {exposure['final_evaluations']} final "
                        "evaluations. Each reuse weakens its claim to being held out."]

    out += ["", "## 4. Compute consumed", ""]
    compute = report["compute"]
    if compute:
        out += [f"Unit: {compute['unit']}.", "",
                f"Total {compute['total']:,.0f} of {compute['budget']:,.0f} budgeted "
                f"({compute['wall_seconds']:,.1f} s wall clock). Optimization overhead (warmup, scoring, "
                f"selection): {compute['optimization_overhead']:,.0f} "
                f"({compute['optimization_overhead_share']:.1%} of the total).", "",
                "| stage | FTE | of which from cache | wall seconds |", "|---|---|---|---|"]
        out += [f"| {stage} | {row['fte']:,.0f} | {row['from_cache_fte']:,.0f} | {row['wall_seconds']:,.1f} |"
                for stage, row in compute["by_stage"].items()]
    else:
        out.append("Nothing has been charged yet.")

    out += ["", "## 5. Coverage changes", ""]
    coverage = report["coverage"]
    if coverage and coverage.get("trials"):
        for key, body in coverage["trials"].items():
            out.append(f"`{key}`: {body['records']} records, {body['tokens']} tokens.")
            for c in body["constraints"]:
                out.append(f"- minimum for {c['field']}={c['value']}: selected {c['selected_records']} records / "
                           f"{c['selected_tokens']} tokens (pool {c['pool_records']} / {c['pool_tokens']}).")
            for name, table in body["distribution"].items():
                shares = ", ".join(f"{value} {row['pool_token_share']:.1%} -> {row['selected_token_share']:.1%}"
                                   for value, row in table.items())
                out.append(f"- token share by {name} (pool -> selected): {shares}")
    else:
        out.append("No selection has been made yet.")
    leakage = report["leakage"]
    if leakage:
        out += ["", f"Split overlap: {leakage['exact_findings']} exact and {leakage['uncertain_findings']} "
                    f"uncertain finding(s); {leakage['excluded_candidates']} candidate(s) excluded."]

    out += ["", "## 6. Limitations and experiments not performed", ""]
    out += [f"- {text}" for text in report["limitations"]]
    if report["not_performed"]:
        out += ["", "Not performed:"]
        out += [f"- {item['what']}: {item['reason'].rstrip('.')}." for item in report["not_performed"]]
    return "\n".join(out) + "\n"
