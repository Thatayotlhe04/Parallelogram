"""Optimize: the experiment contract, split isolation, leakage, budgets and
export. Everything here runs on the default install (no torch, no numpy)."""
from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from optimize_helpers import contract, make_data, row, write_contract, write_jsonl
from parallelogram.cli import app
from parallelogram.optimize import (ConfigError, LeakageError, OptimizeError, SplitIsolationError,
                                    UnsupportedError, export, load_experiment, parse_experiment, plan, report)
from parallelogram.optimize import selection as selection_module
from parallelogram.optimize.budget import Cost, Ledger
from parallelogram.optimize.cache import Cache, write_json
from parallelogram.optimize.planner import build_plan, prepare, trial_slug
from parallelogram.optimize.reporting import decide
from parallelogram.optimize.splits import (EXACT, UNCERTAIN, TestVault, detect_leakage, freeze_seal, load_split, split_dev)
from parallelogram.optimize.stats import INSUFFICIENT, SUPPORTED, CONTRADICTED, compare

runner = CliRunner()


@pytest.fixture
def root(tmp_path):
    make_data(tmp_path)
    return tmp_path


def parsed(root: Path, **overrides):
    return parse_experiment(contract(root, **overrides), base=root)


# ── the contract ────────────────────────────────────────────────────────────

def test_contract_parses_and_resolves_paths_against_the_contract_file(root):
    exp = load_experiment(write_contract(root))
    assert exp.schema_version == 1 and exp.objective == "sft-lora-causal-lm"
    assert exp.data.candidates.path == (root / "candidates.jsonl").resolve()
    assert exp.output.run_dir == (root / "run").resolve()
    assert exp.recipe.seeds == (0, 1) and exp.recipe.loss_masking == "assistant_only"
    assert json.loads(json.dumps(exp.to_dict()))["selection"]["token_budget"] == 150


@pytest.mark.parametrize("key,value,message", [
    ("recipe.sedes", [0], "unknown key"),
    ("selection.less.warmup.epoch", 2, "unknown key"),
    ("schema_version", 2, "not supported by this release"),
    ("recipe.seeds", [], "at least one seed"),
    ("recipe.seeds", [0, 0], "unique"),
    ("recipe.epochs", True, "found a boolean"),
    ("selection.token_budget", 0, "at least 1"),
    ("model.device", "auto", "name a device explicitly"),
    ("evaluation.primary_metric", "exact_match", "must be one of evaluation.metrics"),
    ("evaluation.retain_capabilities", ["reverse"], "cannot be both"),
    ("search", {"token_budgets": [100]}, "max_trials"),
    ("search", {"token_budgets": [150], "max_trials": 1}, "below selection.token_budget"),
    ("search", {"less_variants": [{"warmup_epochs": 9}], "max_trials": 1}, "may vary"),
    ("data.test", {"path": "dev.jsonl"}, "three different files"),
    ("budget.total_compute", -1, "must be positive"),
])
def test_invalid_contracts_are_rejected_not_ignored(root, key, value, message):
    with pytest.raises(ConfigError, match=message):
        parsed(root, **{key: value})


@pytest.mark.parametrize("key,value", [
    ("objective", "dpo"),
    ("objective", "full-finetune"),
    ("data.candidates", {"path": "candidates.jsonl", "format": "preference"}),
    ("recipe.loss_masking", "all_tokens"),
    ("recipe.optimizer", "sgd"),
])
def test_unsupported_objectives_and_formats_are_named_as_unsupported(root, key, value):
    with pytest.raises(UnsupportedError, match="not supported"):
        parsed(root, **{key: value})


def test_fingerprint_tracks_results_not_labels(root):
    base = parsed(root)
    assert parsed(root, name="renamed").fingerprint() == base.fingerprint()
    assert parsed(root, **{"output.run_dir": "elsewhere"}).fingerprint() == base.fingerprint()
    for key, value in [("recipe.seeds", [0, 2]), ("recipe.loss_masking", "final_assistant_only"),
                       ("selection.less.projection.seed", 7), ("model.max_seq_len", 40)]:
        assert parsed(root, **{key: value}).fingerprint() != base.fingerprint()


def test_yaml_contract_and_scientific_notation(root):
    yaml = pytest.importorskip("yaml")
    data = contract(root)
    text = yaml.safe_dump(data).replace("learning_rate: 0.01", "learning_rate: 1e-2")
    (root / "experiment.yaml").write_text(text)
    assert load_experiment(root / "experiment.yaml").recipe.learning_rate == 0.01


def test_a_hub_model_needs_a_pinned_revision(root):
    exp = parsed(root, **{"model.path": "some-org/some-model"})
    with pytest.raises(ConfigError, match="model.revision to a commit hash"):
        plan(exp)
    assert plan(parsed(root, **{"model.path": "some-org/some-model", "model.revision": "0" * 40}))["fits_budget"]


@pytest.mark.parametrize("name", ["tiny", "benchmark"])
def test_shipped_example_contracts_are_valid(name):
    pytest.importorskip("yaml")
    exp = load_experiment(Path(__file__).parents[1] / "examples" / "optimize" / name / "experiment.yaml")
    assert exp.scale == ("pilot" if name == "tiny" else "benchmark")
    assert len(exp.recipe.seeds) >= 3 and exp.baselines.random_matched_compute and exp.baselines.full_data
    if name == "benchmark":
        assert len(exp.model.revision) == 40            # a Hub model is pinned to a commit


def test_contract_file_errors(root):
    with pytest.raises(OptimizeError):
        load_experiment(root / "missing.json")
    (root / "bad.json").write_text("{")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_experiment(root / "bad.json")
    (root / "experiment.toml").write_text("")
    with pytest.raises(ConfigError, match=".yaml, .yml or .json"):
        load_experiment(root / "experiment.toml")


# ── record identity and supported shapes ────────────────────────────────────

def test_record_ids_are_immutable_and_content_derived(root):
    exp = parsed(root, **{"data.id_field": None})
    first = load_split(exp.data.candidates, "candidates")
    again = load_split(exp.data.candidates, "candidates")
    assert [r.id for r in first.records] == [r.id for r in again.records]
    assert all(r.id.startswith("sha256:") for r in first.records)
    # Reordering the file moves line numbers, not identities.
    lines = (root / "candidates.jsonl").read_text().splitlines(keepends=True)
    (root / "shuffled.jsonl").write_text("".join(reversed(lines)))
    shuffled = load_split(type(exp.data.candidates)(root / "shuffled.jsonl"), "candidates")
    assert {r.id for r in shuffled.records} == {r.id for r in first.records}
    assert shuffled.fingerprint != first.fingerprint


def test_id_field_must_be_present_and_unique(root):
    exp = parsed(root)
    rows = [row(0, "copy", "abc", "c"), row(0, "copy", "abd", "c")]
    write_jsonl(root / "dup.jsonl", rows)
    with pytest.raises(ConfigError, match="repeats record id"):
        load_split(type(exp.data.candidates)(root / "dup.jsonl"), "candidates", id_field="id")
    write_jsonl(root / "noid.jsonl", [{k: v for k, v in rows[0].items() if k != "id"}])
    with pytest.raises(ConfigError, match="no usable 'id'"):
        load_split(type(exp.data.candidates)(root / "noid.jsonl"), "candidates", id_field="id")


def test_splits_must_pass_validation_first(root):
    exp = parsed(root)
    lines = (root / "candidates.jsonl").read_text().splitlines(keepends=True)
    (root / "candidates.jsonl").write_text("".join(lines + lines[:1]))  # an exact duplicate
    with pytest.raises(ConfigError, match="validation errors.*duplicates"):
        load_split(exp.data.candidates, "candidates")


@pytest.mark.parametrize("record,what", [
    ({"prompt": "q", "chosen": "a", "rejected": "b"}, "preference"),
    ({"messages": [{"role": "user", "content": "q"},
                   {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "1", "type": "function",
                                    "function": {"name": "f", "arguments": "{}"}}]},
                   {"role": "tool", "tool_call_id": "1", "content": "r"},
                   {"role": "assistant", "content": "a"}],
      "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]}, "tool"),
])
def test_unsupported_record_shapes_are_rejected_explicitly(root, record, what):
    exp = parsed(root)
    write_jsonl(root / "odd.jsonl", [record])
    with pytest.raises(UnsupportedError, match=what):
        load_split(type(exp.data.candidates)(root / "odd.jsonl"), "candidates")


# ── leakage ─────────────────────────────────────────────────────────────────

LONG = ("the quick brown fox jumps over the lazy dog while the cat sleeps in the warm afternoon sun and "
        "nobody in the quiet village notices that the baker has already left for the early market today")


def leak_setup(root: Path, extra_candidates: list[dict], **policy):
    data = make_data(root)
    write_jsonl(root / "candidates.jsonl", data["candidates"] + extra_candidates)
    exp = parsed(root, **{"data.group_field": "doc", "data.leakage": policy} if policy
                 else {"data.group_field": "doc"})
    candidates = load_split(exp.data.candidates, "candidates", id_field="id", group_field="doc")
    dev = load_split(exp.data.dev, "dev", id_field="id", group_field="doc")
    vault = TestVault(exp.data)
    return data, exp, candidates, dev, vault


def test_exact_related_and_near_overlap_are_told_apart(tmp_path):
    data = make_data(tmp_path)
    dev0, test0 = data["dev"][0], data["test"][0]
    near_dev = {**row(900, "copy", "zzz", "cand"), "messages": [
        {"role": "user", "content": LONG}, {"role": "assistant", "content": "ok then"}]}
    write_jsonl(tmp_path / "dev.jsonl", data["dev"] + [{**row(901, "copy", "yyy", "dev"), "messages": [
        {"role": "user", "content": LONG + " indeed"}, {"role": "assistant", "content": "ok then"}]}])
    extra = [
        {**dev0, "id": "leak-record"},                                              # same record as dev
        {**test0, "id": "leak-prompt", "messages": [test0["messages"][0],
                                                    {"role": "assistant", "content": "different"}]},
        {**row(902, "upper", "qqq", "cand"), "id": "leak-doc", "doc": data["test"][1]["doc"]},
        {**near_dev, "id": "leak-near"},
    ]
    write_jsonl(tmp_path / "candidates.jsonl", data["candidates"] + extra)
    exp = parsed(tmp_path, **{"data.group_field": "doc"})
    kw = dict(id_field="id", group_field="doc")
    report_ = detect_leakage(load_split(exp.data.candidates, "candidates", **kw),
                             load_split(exp.data.dev, "dev", **kw),
                             TestVault(exp.data).fingerprints(), exp.data.leakage)
    found = {f.id: f for f in report_.findings}
    assert (found["leak-record"].kind, found["leak-record"].certainty, found["leak-record"].other_split) == \
        ("exact-record", EXACT, "dev")
    assert (found["leak-prompt"].kind, found["leak-prompt"].certainty, found["leak-prompt"].other_split) == \
        ("exact-prompt", EXACT, "test")
    assert (found["leak-doc"].kind, found["leak-doc"].certainty) == ("related-document", EXACT)
    assert (found["leak-near"].kind, found["leak-near"].certainty) == ("near-duplicate", UNCERTAIN)
    assert 0.8 <= found["leak-near"].similarity < 1.0
    # Exact and related overlap is removed from the pool; an uncertain finding is only reported.
    assert report_.excluded == {"leak-record", "leak-prompt", "leak-doc"}
    assert found["leak-near"].action == "reported"
    summary = report_.to_dict()
    assert summary["exact_findings"] == 3 and summary["uncertain_findings"] == 1
    assert LONG not in json.dumps(summary) and "different" not in json.dumps(summary)
    assert set(found) == {"leak-record", "leak-prompt", "leak-doc", "leak-near"}  # no false positives


def test_leakage_policies(tmp_path):
    data = make_data(tmp_path)
    extra = [{**data["dev"][0], "id": "leak"}]
    write_jsonl(tmp_path / "candidates.jsonl", data["candidates"] + extra)
    kw = dict(id_field="id")
    for policy, expected in [({"on_exact": "error"}, LeakageError), ({"on_exact": "exclude"}, None)]:
        exp = parsed(tmp_path, **{"data.leakage": policy})
        args = (load_split(exp.data.candidates, "candidates", **kw), load_split(exp.data.dev, "dev", **kw),
                TestVault(exp.data).fingerprints(), exp.data.leakage)
        if expected:
            with pytest.raises(expected):
                detect_leakage(*args)
        else:
            assert detect_leakage(*args).excluded == {"leak"}


def test_development_and_test_must_be_disjoint(tmp_path):
    data = make_data(tmp_path)
    write_jsonl(tmp_path / "test.jsonl", data["test"] + [{**data["dev"][0], "id": "shared"}])
    exp = parsed(tmp_path)
    kw = dict(id_field="id")
    with pytest.raises(LeakageError, match="Development and test splits overlap"):
        detect_leakage(load_split(exp.data.candidates, "candidates", **kw), load_split(exp.data.dev, "dev", **kw),
                       TestVault(exp.data).fingerprints(), exp.data.leakage)


# ── split isolation ─────────────────────────────────────────────────────────

def _strings(value, depth=0):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from _strings(k, depth + 1)
            yield from _strings(v, depth + 1)
    elif isinstance(value, (list, tuple, set, frozenset)):
        for v in value:
            yield from _strings(v, depth + 1)


def test_the_vault_gives_out_hashes_and_numbers_but_no_examples_before_a_freeze(root):
    exp = parsed(root)
    vault = TestVault(exp.data)
    contents = {m["content"] for r in json.loads("[" + ",".join((root / "test.jsonl").read_text().splitlines()) + "]")
                for m in r["messages"]}
    fp = vault.fingerprints()
    leaked = set(_strings(vars(fp))) & contents
    assert not leaked
    assert set(vault.summary(lambda record: 7)) == {"records", "sha256", "fingerprint", "capabilities",
                                                    "untagged_lines", "tokens", "max_tokens"}
    for bad in (None, {}, {"kind": "optimize-freeze"}, {"kind": "optimize-freeze", "seal": "0" * 64}):
        with pytest.raises(SplitIsolationError):
            vault.open(bad)
    body = {"kind": "optimize-freeze", "trial": "less@150", "selection_sha256": "a", "recipe_fingerprint": "b",
            "contract_fingerprint": "c"}
    opened = vault.open({**body, "seal": freeze_seal(body)})
    assert len(opened.records) == 12
    # Altering what was frozen invalidates the seal.
    with pytest.raises(SplitIsolationError):
        vault.open({**body, "trial": "random@150", "seal": freeze_seal(body)})


def test_selection_code_has_no_parameter_through_which_test_data_could_arrive():
    for name in ("select", "random_order", "utility_order", "shortlist", "text_features", "distribution"):
        params = inspect.signature(getattr(selection_module, name)).parameters
        assert not any("test" in p or "vault" in p for p in params), name
    source = inspect.getsource(selection_module)
    assert "TestVault" not in source and "splits" not in source


def test_dev_examples_are_divided_deterministically_and_by_capability(root):
    exp = parsed(root)
    dev = load_split(exp.data.dev, "dev", id_field="id", capability_field="skill")
    selection, comparison = split_dev(dev, 0.5, seed=0)
    assert not {r.id for r in selection} & {r.id for r in comparison}
    assert len(selection) + len(comparison) == len(dev.records)
    for half in (selection, comparison):
        assert {r.capability for r in half} == {"reverse", "copy"}
    assert [r.id for r in split_dev(dev, 0.5, seed=0)[0]] == [r.id for r in selection]
    assert [r.id for r in split_dev(dev, 0.5, seed=1)[0]] != [r.id for r in selection]
    both = split_dev(dev, 1.0, seed=0)
    assert both[0] == both[1] == dev.records


# ── budget, cache, statistics ───────────────────────────────────────────────

def test_ledger_counts_everything_once_and_names_overhead(tmp_path):
    ledger = Ledger(tmp_path / "ledger.json", total_budget=1000, backward_multiplier=2.0)
    ledger.charge("warmup:a", "warmup", Cost(10, 10, 0.5))
    ledger.charge("warmup:a", "warmup", Cost(10, 10, 0.5))          # a resumed run charges nothing new
    ledger.charge("features:a", "scoring", Cost(100, 100, 1.0), cached=True)
    ledger.charge("train:x", "pilot_training", Cost(50, 50, 1.0), trial="x")
    ledger.charge("eval:x", "evaluation", Cost(40, 0, 0.1), trial="x")
    assert ledger.total == 30 + 300 + 150 + 40
    summary = Ledger(tmp_path / "ledger.json", 1000, 2.0).summary()   # survives a restart
    assert summary["total"] == 520 and summary["remaining"] == 480
    assert summary["optimization_overhead"] == 330
    assert summary["by_stage"]["scoring"]["from_cache_fte"] == 300   # cached work is still counted
    assert ledger.affords(480) and not ledger.affords(481)
    with pytest.raises(ValueError):
        ledger.charge("x", "coffee", Cost())


def test_cache_keys_are_content_addressed(tmp_path):
    cache = Cache(tmp_path)
    inputs = {"dataset": "abc", "model": "m1", "template": "t", "masking": "assistant_only",
              "optimizer": {"lr": 1e-3}, "projection": {"dim": 64, "seed": 0}, "seed": 0}
    assert cache.complete("features", inputs) is None
    cache.dir("features", inputs)
    assert cache.complete("features", inputs) is None            # a directory alone is a partial result
    cache.finish("features", inputs, {"rows": 3})
    assert cache.complete("features", inputs)["rows"] == 3
    for key, value in [("dataset", "abd"), ("model", "m2"), ("template", "u"), ("masking", "final_assistant_only"),
                       ("optimizer", {"lr": 2e-3}), ("projection", {"dim": 64, "seed": 1}), ("seed", 1)]:
        changed = {**inputs, key: value}
        assert Cache.key(changed) != Cache.key(inputs)
        assert cache.complete("features", changed) is None
    # An entry whose recorded inputs do not hash to its key is not served.
    marker = tmp_path / "features" / Cache.key(inputs) / "entry.json"
    entry = json.loads(marker.read_text())
    marker.write_text(json.dumps({**entry, "inputs_sha256": "0" * 64}))
    assert cache.complete("features", inputs) is None


def test_a_conclusion_needs_three_seeds_and_a_difference_beyond_noise():
    clear = compare([0.80, 0.82, 0.81], [0.60, 0.62, 0.61], higher_is_better=True)
    assert clear["verdict"] == SUPPORTED and clear["across_seed_ci95"][0] > 0
    assert compare([0.80, 0.82], [0.60, 0.62], higher_is_better=True)["verdict"] == INSUFFICIENT
    assert compare([0.60, 0.70, 0.50], [0.58, 0.69, 0.52], higher_is_better=True)["verdict"] == INSUFFICIENT
    assert compare([0.5, 0.5, 0.5], [0.4, 0.4, 0.4], higher_is_better=True)["verdict"] == INSUFFICIENT
    worse = compare([1.20, 1.22, 1.21], [1.00, 1.02, 1.01], higher_is_better=False)
    assert worse["verdict"] == CONTRADICTED and worse["difference"] < 0
    # Seeds agree, but the per-example difference straddles zero: evaluation uncertainty blocks it.
    noisy = compare([0.80, 0.82, 0.81], [0.60, 0.62, 0.61], higher_is_better=True,
                    candidate_examples=[1, 0, 1, 0, 1, 0], baseline_examples=[0, 1, 0, 1, 0, 1], samples=200)
    assert noisy["verdict"] == INSUFFICIENT and noisy["evaluation_ci95"][0] < 0 < noisy["evaluation_ci95"][1]


# ── planning ────────────────────────────────────────────────────────────────

def plan_for(root: Path, **overrides):
    prep = prepare(parsed(root, **overrides))
    return prep, build_plan(prep)


FEASIBLE = {"recipe.epochs": 10}     # enough training per token that the overhead fits inside the pool


def test_plan_matrix_labels_what_each_comparison_answers(root):
    prep, p = plan_for(root, **FEASIBLE)
    by_key = {t.key: t for t in p.trials}
    assert {k: t.comparison for k, t in by_key.items()} == {
        "random@150": "matched_tokens", "diversity@150": "matched_tokens", "less@150": "matched_tokens",
        "random+overhead": "matched_compute", "full": "full_data"}
    assert by_key["random+overhead"].token_budget > 150          # the overhead, spent on more data instead
    assert all(p.seeds[k] == [0, 1] for k in by_key)
    stages = {u["stage"] for u in p.units}
    assert {"warmup", "scoring", "pilot_training", "evaluation"} <= stages and p.reserve_final > 0
    assert p.fits and p.estimate <= p.budget and not prep.exact_tokens and p.matched_compute_feasible


def test_no_matched_compute_arm_when_the_overhead_exceeds_the_pool(root):
    """With one training epoch and two scoring checkpoints, scoring costs more
    than training on everything. A random arm capped at the pool would be
    full-data training under a "matched" label, so there is no such arm."""
    _, p = plan_for(root)
    assert "random+overhead" not in {t.key for t in p.trials} and not p.matched_compute_feasible
    assert any(s["what"] == "matched-compute baseline" and "whole pool" in s["reason"] for s in p.skipped)
    cfg = parsed(root).to_dict()
    base = _eval({"reverse": [5.0] * 4, "copy": [0.5] * 4})
    arms = {"less@150": ("less", 150, "matched_tokens", [1.00, 1.02, 0.98]),
            "random@150": ("random", 150, "matched_tokens", [2.01, 1.99, 2.03]),
            "full": ("full", None, "full_data", [1.5, 1.51, 1.49])}
    won = decide(cfg, base, _results(arms), ["reverse"], ["copy"], matched_compute_feasible=False)["conclusion"]
    assert won["status"] == INSUFFICIENT and "more total compute than" in won["statement"]   # even a clear win
    arms["full"] = ("full", None, "full_data", [0.5, 0.51, 0.49])
    lost = decide(cfg, base, _results(arms), ["reverse"], ["copy"], matched_compute_feasible=False)["conclusion"]
    assert lost["status"] == "not_supported" and "cost less in total" in lost["statement"]


def test_plan_drops_optional_arms_to_fit_and_refuses_when_essentials_do_not(root):
    _, generous = plan_for(root, **FEASIBLE)
    essential = sum(u["estimate_fte"] for u in generous.units if u["essential"])
    _, tight = plan_for(root, **FEASIBLE, **{"budget.total_compute": essential * 1.6})
    assert tight.fits and "full" not in {t.key for t in tight.trials}
    assert any("full-data" in s["what"] for s in tight.skipped)
    assert tight.estimate <= tight.budget
    _, starved = plan_for(root, **FEASIBLE, **{"budget.total_compute": essential * 0.5})
    assert not starved.fits and "budget.total_compute" in starved.problems[0]


def test_search_is_bounded_by_max_trials(root):
    search = {"token_budgets": [60, 100], "less_variants": [{"diversity_lambda": 0.5}], "max_trials": 2}
    _, p = plan_for(root, search=search)
    assert [t.key for t in p.trials if t.role == "search"] == ["random@60", "diversity@60"]
    assert any("beyond search.max_trials" in s["what"] for s in p.skipped)


def test_plan_api_trains_nothing_and_writes_nothing(root):
    before = sorted(p.name for p in root.iterdir())
    result = plan(write_contract(root))
    assert result["kind"] == "optimize-plan" and result["fits_budget"]
    assert "estimated" in result["data"]["token_counts"] or "exact" in result["data"]["token_counts"]
    assert result["data"]["test"]["note"].startswith("read for leakage fingerprints")
    assert sorted(p.name for p in root.iterdir()) == sorted(before + ["experiment.json"])
    assert any("frozen.json" in a for a in result["expected_artifacts"])


# ── the decision rule ───────────────────────────────────────────────────────

def _eval(losses: dict[str, list[float]]) -> dict:
    ids, caps, values = [], [], []
    for cap, column in losses.items():
        for i, v in enumerate(column):
            ids.append(f"{cap}-{i}"), caps.append(cap), values.append(v)
    return {"ids": ids, "capability": caps, "loss": values}


def _results(arms: dict[str, tuple[str, int | None, str, list[float]]], copy: float = 0.5) -> dict:
    out = {}
    for key, (policy, budget, comparison, per_seed) in arms.items():
        out[key] = {"trial": {"key": key, "policy": policy, "token_budget": budget, "comparison": comparison},
                    "seeds": {s: {"dev": _eval({"reverse": [v - 0.1, v + 0.1, v, v], "copy": [copy] * 4})}
                              for s, v in enumerate(per_seed)}}
    return out


def test_the_candidate_must_fit_the_selection_budget_and_a_tie_is_not_a_win(root):
    cfg = parsed(root).to_dict()
    base = _eval({"reverse": [5.0] * 4, "copy": [0.5] * 4})
    results = _results({
        "less@150": ("less", 150, "matched_tokens", [2.00, 2.02, 1.98]),
        "random@150": ("random", 150, "matched_tokens", [2.01, 1.99, 2.03]),
        "random+overhead": ("random", 900, "matched_compute", [1.0, 1.01, 0.99]),
        "full": ("full", None, "full_data", [0.9, 0.91, 0.89]),
    })
    decision = decide(cfg, base, results, ["reverse"], ["copy"])
    assert decision["candidate"] == "less@150"          # not the bigger arms, though they score better
    kinds = {c["kind"]: c["verdict"] for c in decision["comparisons"]}
    assert kinds["vs_base_model"] == SUPPORTED
    assert kinds["matched_tokens"] == INSUFFICIENT
    assert kinds["matched_compute"] == CONTRADICTED and kinds["full_data"] == CONTRADICTED
    assert decision["conclusion"]["status"] == "not_supported"


def test_supported_requires_beating_matched_tokens_and_matched_compute(root):
    cfg = parsed(root).to_dict()
    base = _eval({"reverse": [5.0] * 4, "copy": [0.5] * 4})
    arms = {"less@150": ("less", 150, "matched_tokens", [1.00, 1.02, 0.98]),
            "random@150": ("random", 150, "matched_tokens", [2.01, 1.99, 2.03]),
            "random+overhead": ("random", 900, "matched_compute", [1.5, 1.51, 1.49])}
    assert decide(cfg, base, _results(arms), ["reverse"], ["copy"])["conclusion"]["status"] == SUPPORTED
    two_seeds = {k: (*v[:3], v[3][:2]) for k, v in arms.items()}
    assert decide(cfg, base, _results(two_seeds), ["reverse"], ["copy"])["conclusion"]["status"] == INSUFFICIENT
    without_compute = {k: v for k, v in arms.items() if k != "random+overhead"}
    verdict = decide(cfg, base, _results(without_compute), ["reverse"], ["copy"])["conclusion"]
    assert verdict["status"] == INSUFFICIENT and "matched-compute" in verdict["statement"]


def test_a_frozen_candidate_is_not_described_as_the_best_on_test(root):
    cfg = parsed(root).to_dict()
    base = _eval({"reverse": [5.0] * 4, "copy": [0.5] * 4})
    arms = {"diversity@150": ("diversity", 150, "matched_tokens", [2.0, 2.02, 1.98]),
            "random@150": ("random", 150, "matched_tokens", [1.9, 1.92, 1.88])}
    results = _results(arms)
    for entry in results.values():
        for seed in entry["seeds"].values():
            seed["test"] = seed["dev"]
    on_test = decide(cfg, base, results, ["reverse"], ["copy"], split="test", candidate="diversity@150")
    assert "frozen from development data" in on_test["conclusion"]["statement"]
    assert "best arm" not in on_test["conclusion"]["statement"]
    on_dev = decide(cfg, base, results, ["reverse"], ["copy"])
    assert on_dev["candidate"] == "random@150" and "best arm within the selection budget" in on_dev["conclusion"]["statement"]


def test_stats_use_a_conservative_t_beyond_the_table():
    from parallelogram.optimize.stats import t95
    values = [t95(df) for df in (1, 2, 30, 31, 39, 40, 60, 120, 10_000)]
    assert values == sorted(values, reverse=True) and values[3] == 2.042 and values[-1] == 1.980


def test_retention_and_the_base_model_gate_the_recommendation(root):
    cfg = parsed(root, **{"evaluation.retention_tolerance": 0.1}).to_dict()
    base = _eval({"reverse": [5.0] * 4, "copy": [0.5] * 4})
    arms = {"less@150": ("less", 150, "matched_tokens", [1.0, 1.02, 0.98]),
            "random@150": ("random", 150, "matched_tokens", [2.0, 2.02, 1.98])}
    forgetful = decide(cfg, base, _results(arms, copy=0.9), ["reverse"], ["copy"])
    assert forgetful["candidate"] is None and "retained capabilities" in forgetful["conclusion"]["statement"]
    worse_than_base = decide(cfg, _eval({"reverse": [0.5] * 4, "copy": [0.5] * 4}), _results(arms),
                             ["reverse"], ["copy"])
    assert worse_than_base["candidate"] is None and worse_than_base["conclusion"]["status"] == INSUFFICIENT


# ── export ──────────────────────────────────────────────────────────────────

AWKWARD = [
    '{"id":"a","messages":[{"role":"user","content":"caf\\u00e9  \\u2603"},{"role":"assistant","content":"x"}],"z":1,"a":{"b":[1,2]}}',
    '{ "id" : "b" ,  "extra": null, "messages": [ {"role": "user", "content": "naïve ☃"}, {"role": "assistant", "content": "y"} ] }',
    '{"id":"c","messages":[{"role":"user","content":"1e3"},{"role":"assistant","content":"z"}],"n":1.0}',
    '{"id":"d","messages":[{"role":"user","content":"tab\\there"},{"role":"assistant","content":"w"}]}',
]


def fake_run(root: Path, selected: list[str], *, frozen: bool = True) -> Path:
    """A run directory with just enough artifacts for report and export."""
    source = root / "candidates.jsonl"
    source.write_bytes(("\r\n".join(AWKWARD[:2]) + "\r\n" + "\n".join(AWKWARD[2:])).encode("utf-8"))  # no final newline
    write_jsonl(root / "dev.jsonl", [row(0, "copy", "abc", "dev"), row(1, "reverse", "abd", "dev")])
    write_jsonl(root / "test.jsonl", [row(0, "copy", "abe", "test"), row(1, "reverse", "abf", "test")])
    exp = parse_experiment(contract(root), base=root)
    split = load_split(exp.data.candidates, "candidates", id_field="id")
    run = root / "run"
    write_json(run / "contract.json", {"schema_version": 1, "core_fingerprint": "core",
                                       "contract_fingerprint": exp.fingerprint(), "contract": exp.to_dict()})
    write_json(run / "pool.json", {"source": {"path": str(source), "sha256": split.sha256, "records": 4},
                                   "excluded": {"leakage": 0, "overlength": 0},
                                   "records": [{"id": r.id, "line": r.line_no, "raw_sha256": r.raw_sha256,
                                                "tokens": 5, "attrs": {}} for r in split.records]})
    write_json(run / "setup.json", {"model": {"model": {"ref": "m", "revision": "r"},
                                              "tokenizer": {"ref": "m", "revision": "r"},
                                              "chat_template_sha256": "t", "versions": {}},
                                    "pilot": True, "target_capabilities": ["reverse"], "retain_capabilities": ["copy"]})
    write_json(run / "trials" / trial_slug("less@150") / "seed-0" / "selection.json", {
        "policy": "less", "token_budget": 150, "tokens": 5 * len(selected), "records": len(selected),
        "sha256": "s", "coverage": [],
        "selected": [{"id": i, "rank": n, "tokens": 5, "reason": "policy:less",
                      "scores": {"target_alignment": 0.5 - n}} for n, i in enumerate(selected)]})
    if frozen:
        write_json(run / "frozen.json", {"kind": "optimize-freeze", "trial": "less@150", "selection_seed": 0})
    return run


def test_export_copies_source_lines_byte_for_byte(tmp_path):
    run = fake_run(tmp_path, ["d", "a", "b"])
    out = tmp_path / "out" / "selected.jsonl"
    before = (tmp_path / "candidates.jsonl").read_bytes()
    manifest = export(run, out)
    lines = out.read_bytes().splitlines(keepends=True)
    assert lines == [AWKWARD[0].encode() + b"\r\n", AWKWARD[1].encode() + b"\r\n", AWKWARD[3].encode() + b"\n"]
    assert (tmp_path / "candidates.jsonl").read_bytes() == before        # inputs untouched
    assert json.loads((tmp_path / "out" / "selected.jsonl.manifest.json").read_text()) == manifest
    assert [r["id"] for r in manifest["records"]] == ["a", "b", "d"]     # source order
    assert [r["source_line"] for r in manifest["records"]] == [1, 2, 4]
    assert manifest["records"][2]["selection_rank"] == 0 and manifest["records"][2]["reason"] == "policy:less"
    assert manifest["records"][0]["scores"] == {"target_alignment": -0.5}
    assert manifest["selection"]["frozen_candidate"] and manifest["source"]["sha256"]
    assert manifest["fingerprints"]["contract"] and manifest["provenance"]["parallelogram"]
    assert "not calibrated" in manifest["evidence"]["note"]
    import hashlib
    assert manifest["output"]["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()


def test_export_never_overwrites_inputs_or_existing_files(tmp_path):
    run = fake_run(tmp_path, ["a"])
    for name in ("candidates.jsonl", "dev.jsonl", "test.jsonl"):
        with pytest.raises(ConfigError, match="never overwrites an input"):
            export(run, tmp_path / name, overwrite=True)
    (tmp_path / "link.jsonl").symlink_to(tmp_path / "candidates.jsonl")
    with pytest.raises(ConfigError, match="never overwrites an input"):
        export(run, tmp_path / "link.jsonl", overwrite=True)
    out = tmp_path / "selected.jsonl"
    export(run, out)
    with pytest.raises(ConfigError, match="already exists"):
        export(run, out)
    export(run, out, overwrite=True)


def test_export_refuses_a_changed_source_and_unknown_arms(tmp_path):
    run = fake_run(tmp_path, ["a"], frozen=False)
    with pytest.raises(ConfigError, match="Nothing is frozen"):
        export(run, tmp_path / "x.jsonl")
    with pytest.raises(ConfigError, match="No finished arm named"):
        export(run, tmp_path / "x.jsonl", trial="random@150")
    manifest = export(run, tmp_path / "x.jsonl", trial="less@150")
    assert manifest["selection"]["frozen_candidate"] is False
    with open(tmp_path / "candidates.jsonl", "ab") as f:
        f.write(b"\n")
    with pytest.raises(OptimizeError, match="has changed since this run"):
        export(run, tmp_path / "y.jsonl", trial="less@150")
    assert not (tmp_path / "y.jsonl").exists()


def test_report_on_a_partial_run_says_what_is_missing(tmp_path):
    run = fake_run(tmp_path, ["a"], frozen=False)
    result = report(run)
    assert result["status"] == "started" and result["label"] == "pilot"
    assert result["conclusion"]["status"] == INSUFFICIENT and result["conclusion"]["pilot"] is True
    assert result["final_results"] is None and result["development_results"] is None
    assert any("final held-out evaluation" in item["what"] for item in result["not_performed"])
    assert any("not calibrated probabilities" in text for text in result["limitations"])


# ── CLI, MCP and the lightweight install ────────────────────────────────────

def test_optimize_cli_exit_codes(root, tmp_path):
    path = write_contract(root)
    ok = runner.invoke(app, ["optimize", "plan", str(path), "--json"])
    assert ok.exit_code == 0 and json.loads(ok.stdout)["kind"] == "optimize-plan"
    human = runner.invoke(app, ["optimize", "plan", str(path)])
    assert human.exit_code == 0 and "experiment matrix" in human.stdout and "Nothing was trained" in human.stdout
    bad = write_contract(root, objective="dpo")
    assert runner.invoke(app, ["optimize", "plan", str(bad)]).exit_code == 2
    starved = write_contract(root, **{"budget.total_compute": 10})
    assert runner.invoke(app, ["optimize", "plan", str(starved)]).exit_code == 2
    assert runner.invoke(app, ["optimize", "plan", str(root / "nope.json")]).exit_code == 3
    assert runner.invoke(app, ["optimize", "report", str(tmp_path)]).exit_code == 3
    assert runner.invoke(app, ["optimize", "export", str(tmp_path), "--output", str(tmp_path / "o.jsonl")]).exit_code == 3


def test_optimize_cli_report_and_export(tmp_path):
    run = fake_run(tmp_path, ["b", "c"])
    shown = runner.invoke(app, ["optimize", "report", str(run)])
    assert shown.exit_code == 0 and "## 6. Limitations and experiments not performed" in shown.stdout
    as_json = runner.invoke(app, ["optimize", "report", str(run), "--json"])
    assert json.loads(as_json.stdout)["kind"] == "optimize-report"
    out = tmp_path / "selected.jsonl"
    done = runner.invoke(app, ["optimize", "export", str(run), "--output", str(out)])
    assert done.exit_code == 0 and "wrote 2 records" in done.stdout and out.is_file()
    assert runner.invoke(app, ["optimize", "export", str(run), "--output", str(out)]).exit_code == 2


def test_existing_commands_and_default_mcp_tools_are_unchanged(tmp_path):
    from parallelogram.mcp_server import OPTIMIZE_TOOLS, TOOLS, handle

    listed = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})["result"]["tools"]
    assert [t["name"] for t in listed] == ["check_dataset", "fix_dataset", "list_rules"]
    call = {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
            "params": {"name": "optimize_report", "arguments": {"run_dir": str(tmp_path)}}}
    assert handle(call)["error"]["code"] == -32602                      # not served unless asked for
    opted = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, optimize=True)["result"]["tools"]
    assert [t["name"] for t in opted] == [t["name"] for t in TOOLS + OPTIMIZE_TOOLS]
    assert "optimize_run" not in {t["name"] for t in opted}
    examples = Path(__file__).parents[1] / "examples"
    assert runner.invoke(app, ["check", str(examples / "clean.jsonl")]).exit_code == 0
    assert runner.invoke(app, ["check", str(examples / "broken.jsonl")]).exit_code == 2
    assert runner.invoke(app, ["report", str(examples / "clean.jsonl"), "--json"]).exit_code == 0
    assert runner.invoke(app, ["rules", "--json"]).exit_code == 0


def test_optimize_mcp_tools(tmp_path, monkeypatch):
    from parallelogram.mcp_server import handle

    make_data(tmp_path)
    path = write_contract(tmp_path)
    monkeypatch.chdir(tmp_path)

    def call(name, arguments):
        return handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": name, "arguments": arguments}}, tmp_path, optimize=True)["result"]

    planned = call("optimize_plan", {"experiment": str(path)})
    assert planned["structuredContent"]["exit_code"] == 0 and planned["structuredContent"]["fits_budget"]
    assert call("optimize_plan", {"experiment": str(tmp_path / "nope.json")})["isError"]
    assert call("optimize_report", {"run_dir": str(tmp_path / "nope")})["isError"]


def test_optimize_mcp_export_is_confined_to_the_root(tmp_path, monkeypatch):
    from parallelogram.mcp_server import handle

    root = tmp_path / "root"
    root.mkdir()
    run = fake_run(root, ["a", "c"])
    monkeypatch.chdir(root)

    def call(arguments):
        return handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": {"name": "optimize_export", "arguments": arguments}}, root, optimize=True)["result"]

    assert "must be inside" in call({"run_dir": str(run), "output": str(tmp_path / "escape.jsonl")})["content"][0]["text"]
    assert not (tmp_path / "escape.jsonl").exists()
    done = call({"run_dir": str(run), "output": "picked.jsonl"})["structuredContent"]
    assert done["output"]["records"] == 2 and done["records_listed"] is False and "records" not in done
    assert (root / "picked.jsonl").is_file() and (root / "picked.jsonl.manifest.json").is_file()
    assert call({"run_dir": str(run), "output": "picked.jsonl"})["isError"]            # exists
    assert call({"run_dir": str(run), "output": "candidates.jsonl", "overwrite": True})["isError"]  # an input
    report_ = handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                      "params": {"name": "optimize_report", "arguments": {"run_dir": str(run)}}}, root,
                     optimize=True)["result"]["structuredContent"]
    assert report_["kind"] == "optimize-report" and report_["exit_code"] == 0


def test_the_default_install_stays_lightweight():
    """Importing the CLI, the API and the optimize package must not pull in
    the ML stack, and the runtime dependencies stay typer and rich."""
    code = ("import sys, parallelogram, parallelogram.cli, parallelogram.optimize, parallelogram.mcp_server\n"
            "import parallelogram.optimize.runner, parallelogram.optimize.less, parallelogram.optimize.backend\n"
            "heavy = [m for m in ('torch', 'transformers', 'peft', 'numpy', 'yaml', 'tokenizers') if m in sys.modules]\n"
            "print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]"
    pyproject = (Path(__file__).parents[1] / "pyproject.toml").read_text()
    dependencies = pyproject.split("dependencies = [")[1].split("]")[0]
    assert sorted(line.strip().strip('",').split(">=")[0] for line in dependencies.strip().splitlines()) == ["rich", "typer"]
    assert "optimize = [" in pyproject


def test_missing_extras_are_an_operational_failure_with_instructions(root, monkeypatch):
    import builtins

    from parallelogram.optimize import MissingDependencyError, run

    real = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name.split(".")[0] in ("torch", "peft", "transformers"):
            raise ImportError(f"No module named {name!r}", name=name)
        return real(name, *args, **kwargs)

    for module in [m for m in sys.modules if m.split(".")[0] in ("torch", "peft", "transformers")]:
        monkeypatch.delitem(sys.modules, module)
    monkeypatch.setattr(builtins, "__import__", without_torch)
    path = write_contract(root)
    with pytest.raises(MissingDependencyError, match=r"pip install 'parallelogram\[optimize\]'"):
        run(path)
    assert runner.invoke(app, ["optimize", "run", str(path), "--quiet"]).exit_code == 3
    # Planning still works, with token counts labeled as estimates.
    planned = plan(path)
    assert "estimated" in planned["data"]["token_counts"] and "tokenizer_unavailable" in planned["resource_assumptions"]
