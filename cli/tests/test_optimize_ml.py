"""Optimize: loss masking, gradient scoring, training, resume and the full
loop, on a tiny model that is really trained. Skipped without the ML extras.

Nothing here asserts that the model-aware selector beats random selection.
The tests check that the numbers are computed correctly and reported
honestly, whichever way they come out.
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
np = pytest.importorskip("numpy")

from typer.testing import CliRunner  # noqa: E402

from optimize_helpers import contract, make_data, row, write_jsonl  # noqa: E402
from parallelogram.cli import app  # noqa: E402
from parallelogram.optimize import (ConfigError, OptimizeError, OutOfMemory, UnsupportedError, export,  # noqa: E402
                                    parse_experiment, plan, report, run)
from parallelogram.optimize import runner as runner_module  # noqa: E402
from parallelogram.optimize.backend import Backend  # noqa: E402
from parallelogram.optimize.cache import read_json  # noqa: E402
from parallelogram.optimize.less import Projector, adam_direction, checkpoint_weights  # noqa: E402
from parallelogram.optimize.planner import trial_slug  # noqa: E402
from parallelogram.optimize.runner import Run  # noqa: E402
from parallelogram.optimize.selection import (Item, KCenterOrder, UtilityOrder, random_order, select,  # noqa: E402
                                              text_features, utility_order)
from parallelogram.optimize.config import CoverageConstraint  # noqa: E402
from parallelogram.optimize.errors import SelectionInfeasible  # noqa: E402
from parallelogram.optimize.splits import Record, TestVault  # noqa: E402
from parallelogram.optimize.tiny import build_tiny_model  # noqa: E402

cli = CliRunner()


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory) -> Path:
    return build_tiny_model(tmp_path_factory.mktemp("model") / "tiny", seed=0, n_embd=32, n_layer=1,
                            n_head=2, n_positions=64)


@pytest.fixture
def root(tmp_path, model_dir) -> Path:
    make_data(tmp_path)
    return tmp_path


def experiment(root: Path, model_dir: Path, **overrides):
    return parse_experiment(contract(root, **{"model.path": str(model_dir), **overrides}), base=root)


def conversation(*turns: tuple[str, str], rid: str = "r") -> Record:
    return Record(id=rid, line_no=1, raw_sha256="", messages=tuple({"role": r, "content": c} for r, c in turns))


def flat(parameters) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in parameters])


# ── loss masking ────────────────────────────────────────────────────────────

def test_only_assistant_tokens_are_supervised(root, model_dir):
    backend = Backend(experiment(root, model_dir))
    record = conversation(("system", "be brief"), ("user", "copy: ab"), ("assistant", "ab"),
                          ("user", "again"), ("assistant", "ab!"))
    encoded = backend.encode(record)
    tokens = backend.tokenizer.convert_ids_to_tokens(list(encoded.ids))
    assert [t for t, m in zip(tokens, encoded.mask) if m] == ["a", "b", "<|end|>", "a", "b", "!", "<|end|>"]
    # The generation prompt, user and system text are never targets.
    assert not any(m for t, m in zip(tokens, encoded.mask) if t in ("<|assistant|>", "<|user|>", "<|system|>"))
    assert encoded.tokens == backend.count_tokens(record) and encoded.supervised == 7

    last_only = Backend(experiment(root, model_dir, **{"recipe.loss_masking": "final_assistant_only"}))
    encoded = last_only.encode(record)
    assert [t for t, m in zip(tokens, encoded.mask) if m] == ["a", "b", "!", "<|end|>"]


def test_loss_matches_a_direct_calculation_and_ignores_padding(root, model_dir):
    backend = Backend(experiment(root, model_dir))
    records = [conversation(("user", "copy: abc"), ("assistant", "abc")),
               conversation(("user", "reverse: abcdefgh"), ("assistant", "hgfedcba"), ("user", "ok"), ("assistant", "k"))]
    encoded = [backend.encode(r) for r in records]
    with torch.no_grad():
        together = backend.example_losses(backend.model, encoded)
        for e, value in zip(encoded, together):
            ids = torch.tensor([e.ids])
            labels = torch.tensor([[t if m else -100 for t, m in zip(e.ids, e.mask)]])
            logits = backend.model(input_ids=ids).logits
            direct = torch.nn.functional.cross_entropy(logits[0, :-1], labels[0, 1:], ignore_index=-100)
            assert math.isclose(float(value), float(direct), rel_tol=1e-5)
            assert math.isclose(float(backend.example_losses(backend.model, [e])[0]), float(value), rel_tol=1e-5)


def test_changing_unsupervised_text_changes_no_target(root, model_dir):
    """Targets are the same tokens whatever the prompt says; only assistant
    positions enter the loss, so the count of supervised tokens is fixed."""
    backend = Backend(experiment(root, model_dir))
    a = backend.encode(conversation(("user", "copy: abc"), ("assistant", "xyz")))
    b = backend.encode(conversation(("user", "something else entirely"), ("assistant", "xyz")))
    assert [t for t, m in zip(a.ids, a.mask) if m] == [t for t, m in zip(b.ids, b.mask) if m]


def test_a_template_that_is_not_incremental_is_refused(root, model_dir):
    (root / "bad.jinja").write_text(
        "{{ messages|length }}{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    backend = Backend(experiment(root, model_dir, **{"model.chat_template": "bad.jinja"}))
    with pytest.raises(UnsupportedError, match="does not render conversations incrementally"):
        backend.encode(conversation(("user", "hi"), ("assistant", "yo")))


def test_template_and_masking_are_part_of_the_model_identity(root, model_dir):
    (root / "other.jinja").write_text(
        "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|> {% endif %}")
    base = Backend(experiment(root, model_dir)).identity
    assert Backend(experiment(root, model_dir, **{"model.chat_template": "other.jinja"})).identity != base
    assert Backend(experiment(root, model_dir, **{"recipe.loss_masking": "final_assistant_only"})).identity != base
    assert Backend(experiment(root, model_dir)).identity == base
    changed = root / "model2"
    shutil.copytree(model_dir, changed)
    config = json.loads((changed / "config.json").read_text())
    (changed / "config.json").write_text(json.dumps({**config, "attn_pdrop": 0.01}))
    assert Backend(experiment(root, changed)).identity["model"]["revision"] != base["model"]["revision"]


# ── gradients and the update direction ──────────────────────────────────────

def warmed(backend: Backend, examples, steps: int = 4, lr: float = 0.05):
    """An adapter a few Adam steps from initialization, with its optimizer."""
    context = backend.adapter(0)
    model = context.__enter__()
    params = [p for _, p in backend.trainable(model)]
    recipe = backend.exp.recipe
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(recipe.beta1, recipe.beta2), eps=recipe.eps, weight_decay=0.0)
    for step in range(steps):
        optimizer.zero_grad()
        backend.example_losses(model, examples[step:step + 3]).mean().backward()
        optimizer.step()
    return context, model, params, optimizer


def pool(backend: Backend, root: Path):
    rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
    records = [Record(id=r["id"], line_no=i + 1, raw_sha256="", messages=tuple(r["messages"]), capability=r["skill"])
               for i, r in enumerate(rows)]
    return records, [backend.encode(r) for r in records]


def test_per_example_gradient_matches_finite_differences(root, model_dir):
    backend = Backend(experiment(root, model_dir))
    _, encoded = pool(backend, root)
    context, model, params, _ = warmed(backend, encoded)
    try:
        gradient = backend.example_gradient(model, encoded[7])
        assert gradient.shape == (sum(p.numel() for p in params),) and float(gradient.norm()) > 0
        generator = torch.Generator().manual_seed(0)
        for _ in range(3):
            direction = torch.randn(gradient.shape, generator=generator)
            direction /= direction.norm()
            eps, losses = 1e-2, []
            for sign in (1.0, -1.0):
                offset = 0
                with torch.no_grad():
                    for p in params:
                        p.add_(sign * eps * direction[offset:offset + p.numel()].view_as(p))
                        offset += p.numel()
                    losses.append(float(backend.example_losses(model, [encoded[7]])[0]))
                    offset = 0
                    for p in params:
                        p.sub_(sign * eps * direction[offset:offset + p.numel()].view_as(p))
                        offset += p.numel()
            numeric = (losses[0] - losses[1]) / (2 * eps)
            assert math.isclose(numeric, float(gradient @ direction), rel_tol=0.03, abs_tol=2e-4)
    finally:
        context.__exit__(None, None, None)


def test_adam_direction_is_exactly_the_step_the_optimizer_takes(root, model_dir):
    backend = Backend(experiment(root, model_dir))
    recipe = backend.exp.recipe
    _, encoded = pool(backend, root)
    lr = 0.05
    context, model, params, optimizer = warmed(backend, encoded, lr=lr)
    try:
        exp_avg = flat([optimizer.state[p]["exp_avg"] for p in params])
        exp_avg_sq = flat([optimizer.state[p]["exp_avg_sq"] for p in params])
        step = int(optimizer.state[params[0]]["step"])
        gradient = backend.example_gradient(model, encoded[9])
        predicted = adam_direction(gradient, exp_avg, exp_avg_sq, step, recipe.beta1, recipe.beta2, recipe.eps)
        before = flat(params).clone()
        optimizer.zero_grad()
        offset = 0
        for p in params:
            p.grad = gradient[offset:offset + p.numel()].view_as(p).clone()
            offset += p.numel()
        optimizer.step()
        actual = (before - flat(params)) / lr
        assert torch.allclose(predicted, actual, rtol=1e-3, atol=1e-5)
        # And it is not just the raw gradient rescaled: Adam reweights coordinates.
        cosine = torch.nn.functional.cosine_similarity(predicted, gradient, dim=0)
        assert float(cosine) < 0.999
    finally:
        context.__exit__(None, None, None)


def test_alignment_predicts_the_loss_change_of_a_real_sgd_step(root, model_dir):
    """What the score claims, checked directly: stepping along one example's
    gradient changes the development loss by about -lr * <g_example, g_dev>."""
    backend = Backend(experiment(root, model_dir))
    _, encoded = pool(backend, root)
    dev, train = encoded[:6], encoded[6:14]
    context, model, params, _ = warmed(backend, encoded[14:])
    try:
        g_dev = torch.stack([backend.example_gradient(model, e) for e in dev]).mean(0)
        with torch.no_grad():
            base_loss = float(backend.example_losses(model, dev).mean())
        start = flat(params).clone()
        agreements = 0
        for example in train:
            g = backend.example_gradient(model, example)
            lr = 0.02 / float(g.norm())
            predicted = -lr * float(g @ g_dev)
            offset = 0
            with torch.no_grad():
                for p in params:
                    p.sub_(lr * g[offset:offset + p.numel()].view_as(p))
                    offset += p.numel()
                actual = float(backend.example_losses(model, dev).mean()) - base_loss
                offset = 0
                for p in params:
                    p.copy_(start[offset:offset + p.numel()].view_as(p))
                    offset += p.numel()
            assert math.isclose(actual, predicted, rel_tol=0.25, abs_tol=2e-5)
            agreements += (actual < 0) == (predicted < 0)
        assert agreements == len(train)
    finally:
        context.__exit__(None, None, None)


def test_projection_is_deterministic_linear_and_preserves_inner_products():
    sizes = [("a.weight", 3000), ("b.weight", 2000)]
    one = Projector(torch, sizes, "countsketch", 256, seed=0)
    generator = torch.Generator().manual_seed(1)
    a, b = torch.randn(5000, generator=generator), torch.randn(5000, generator=generator)
    b = b + 0.5 * a
    assert torch.equal(one(a), Projector(torch, sizes, "countsketch", 256, seed=0)(a))
    assert not torch.equal(one(a), Projector(torch, sizes, "countsketch", 256, seed=1)(a))
    # Independent of the order parameter tensors are listed in.
    swapped = Projector(torch, sizes[::-1], "countsketch", 256, seed=0)
    assert torch.allclose(swapped(torch.cat([a[3000:], a[:3000]])), one(a), atol=1e-5)
    assert torch.allclose(one(2 * a + b), 2 * one(a) + one(b), atol=1e-3)
    assert one(a).shape == (256,)
    estimates = [float(Projector(torch, sizes, "countsketch", 256, seed=s)(a) @
                       Projector(torch, sizes, "countsketch", 256, seed=s)(b)) for s in range(60)]
    truth = float(a @ b)
    stderr = float(np.std(estimates, ddof=1) / math.sqrt(len(estimates)))
    assert abs(np.mean(estimates) - truth) < 4 * stderr          # unbiased
    assert stderr < 0.05 * abs(truth)                            # and tight enough to rank with
    identity = Projector(torch, sizes, "identity", 256, seed=0)
    assert torch.equal(identity(a), a) and identity.dim == 5000


IDENTITY = {"selection.less.projection": {"kind": "identity"}, "selection.policies": ["less"],
            "baselines.random_matched_compute": False, "baselines.full_data": False}


def scored(root: Path, model_dir: Path, **overrides) -> Run:
    run_ = Run(experiment(root, model_dir, **overrides))
    run_._setup()
    run_._score()
    return run_


def test_the_scoring_pipeline_matches_direct_calculations(root, model_dir):
    """Warmup checkpoints, sharded features and aggregation, against the same
    quantities computed by hand from the checkpoints."""
    run_ = scored(root, model_dir, **IDENTITY)
    backend, spec, recipe = run_.backend, run_.exp.selection.less, run_.exp.recipe
    scores = run_._scores_for(spec)
    assert set(scores.per_capability) == {"reverse", "copy"} and len(run_.checkpoints) == 2
    weights = checkpoint_weights(run_.checkpoints, "lr")
    assert math.isclose(sum(weights), 1.0) and weights[0] > weights[1]     # linear decay: earlier epoch, higher lr
    groups: dict[str, list] = {}
    for record, encoded in zip(run_.prep.dev_selection, run_.dev_selection):
        groups.setdefault(record.capability, []).append(encoded)
    picks = [0, 5, 17, len(run_.pool) - 1]
    direct = {name: np.zeros(len(picks)) for name in groups}
    with backend.adapter(spec.warmup_seed) as model:
        names = [n for n, _ in backend.trainable(model)]
        for weight, checkpoint in zip(weights, run_.checkpoints):
            state = torch.load(checkpoint.path, weights_only=True)
            with torch.no_grad():
                for name, p in backend.trainable(model):
                    p.copy_(state["params"][name])
            exp_avg = torch.cat([state["exp_avg"][n].reshape(-1) for n in names])
            exp_avg_sq = torch.cat([state["exp_avg_sq"][n].reshape(-1) for n in names])
            for name, examples in groups.items():
                dev = torch.stack([backend.example_gradient(model, e) for e in examples]).mean(0)
                for column, index in enumerate(picks):
                    update = adam_direction(backend.example_gradient(model, run_.pool[index]), exp_avg, exp_avg_sq,
                                            state["step"], recipe.beta1, recipe.beta2, recipe.eps)
                    direct[name][column] += weight * float(torch.nn.functional.cosine_similarity(update, dev, dim=0))
    for name in groups:
        assert np.allclose(scores.per_capability[name][picks], direct[name], atol=1e-4)
    # Target and retention scores are separate columns, never blended.
    assert np.array_equal(scores.target, scores.per_capability["reverse"])
    assert np.array_equal(scores.retention, scores.per_capability["copy"])
    assert scores.target.min() >= -1.0001 and scores.target.max() <= 1.0001
    row_ = scores.row(0)
    assert set(row_) == {"target_alignment", "retention_alignment", "by_capability"}


def test_scoring_settings_change_what_is_computed(root, model_dir):
    run_ = scored(root, model_dir, **IDENTITY)
    spec = run_.exp.selection.less
    from dataclasses import replace
    cosine = run_._scores_for(spec).target
    raw = run_._scores_for(replace(spec, normalization="none")).target
    uniform = run_._scores_for(replace(spec, checkpoint_weighting="uniform")).target
    assert not np.allclose(cosine, raw) and not np.allclose(cosine, uniform)
    assert np.abs(raw).max() < np.abs(cosine).max()        # raw dot products of small gradients
    sgd = scored(root, model_dir, **IDENTITY, **{"selection.less.optimizer_aware": False,
                                                 "output.run_dir": "run-sgd"})
    assert sgd.features != run_.features                   # a different cache entry, not a stale reuse
    assert not np.allclose(sgd._scores_for(spec).target, cosine)


def test_base_weights_are_untouched_by_training(root, model_dir):
    backend = Backend(experiment(root, model_dir))
    _, encoded = pool(backend, root)
    before = {k: v.clone() for k, v in backend.model.state_dict().items()}
    with backend.adapter(0) as model:
        first = backend.train(model, encoded, seed=0)
        trained = flat([p for _, p in backend.trainable(model)])
    after = backend.model.state_dict()
    assert set(before) == set(after) and all(torch.equal(before[k], after[k]) for k in before)
    assert first.steps == math.ceil(len(encoded) / 4) and first.last_loss < first.first_loss * 1.5
    with backend.adapter(0) as model:
        backend.train(model, encoded, seed=0)
        assert torch.equal(trained, flat([p for _, p in backend.trainable(model)]))     # same seed, same result
    with backend.adapter(1) as model:
        backend.train(model, encoded, seed=1)
        assert not torch.equal(trained, flat([p for _, p in backend.trainable(model)]))


# ── out of memory ───────────────────────────────────────────────────────────

def test_training_backs_off_under_memory_pressure_without_changing_the_update(root, model_dir):
    backend = Backend(experiment(root, model_dir))
    _, encoded = pool(backend, root)
    with backend.adapter(0) as model:
        backend.train(model, encoded, seed=0)
        reference = flat([p for _, p in backend.trainable(model)])
    real = backend.example_losses
    messages = []
    backend.log = messages.append

    def limited(model, batch):
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")
        return real(model, batch)

    backend.example_losses = limited
    with backend.adapter(0) as model:
        stats = backend.train(model, encoded, seed=0)
        squeezed = flat([p for _, p in backend.trainable(model)])
    assert stats.oom_backoffs == 1 and stats.micro_batch == 2 and "out of memory" in messages[0]
    assert torch.allclose(squeezed, reference, atol=1e-5)       # accumulation is exact
    losses, _ = backend.losses(backend.model, encoded[:9])
    backend.example_losses = real
    assert np.allclose(losses, backend.losses(backend.model, encoded[:9])[0], atol=1e-6)

    def never(model, batch):
        raise MemoryError()

    backend.example_losses = never
    with backend.adapter(0) as model:
        with pytest.raises(OutOfMemory, match="single example"):
            backend.train(model, encoded, seed=0)
    with pytest.raises(OutOfMemory):
        backend.losses(backend.model, encoded[:3])


def test_an_unrelated_runtime_error_is_not_mistaken_for_oom(root, model_dir):
    backend = Backend(experiment(root, model_dir))
    _, encoded = pool(backend, root)

    def broken(model, batch):
        raise RuntimeError("shape mismatch")

    backend.example_losses = broken
    with backend.adapter(0) as model:
        with pytest.raises(RuntimeError, match="shape mismatch"):
            backend.train(model, encoded, seed=0)


# ── selection under constraints ─────────────────────────────────────────────

def items_for(n: int = 60) -> list[Item]:
    return [Item(f"r{i}", 10 + (i * 7) % 23, {"lang": ("en", "sw", "fr")[i % 3], "skill": "a" if i < 40 else "b"})
            for i in range(n)]


def orders(items):
    n = len(items)
    utility = np.linspace(1.0, -1.0, n)
    features = text_features(np, (f"text number {i % 7} about topic {i % 5}" for i in range(n)))
    short = list(range(30))
    return {
        "random": lambda: random_order(n, 0),
        "diversity": lambda: KCenterOrder(np, features),
        "less": lambda: utility_order(np, utility),
        "less+mmr": lambda: UtilityOrder(np, utility, short, features[short], 0.5),
    }


@pytest.mark.parametrize("policy", ["random", "diversity", "less", "less+mmr"])
@pytest.mark.parametrize("budget", [37, 150, 400, 10_000])
def test_every_policy_respects_the_token_budget_and_coverage(policy, budget):
    items = items_for()
    constraints = [CoverageConstraint("lang", "sw", min_records=2), CoverageConstraint("skill", "b", min_tokens=30)]
    try:
        chosen = select(np, items, budget, constraints, orders(items)[policy]())
    except SelectionInfeasible:
        assert budget == 37
        return
    assert chosen.tokens == sum(items[i].tokens for i in chosen.indices) <= budget
    assert len(set(chosen.indices)) == len(chosen.indices)
    assert sum(items[i].attrs["lang"] == "sw" for i in chosen.indices) >= 2
    assert sum(items[i].tokens for i in chosen.indices if items[i].attrs["skill"] == "b") >= 30
    assert {c["field"] for c in chosen.coverage} == {"lang", "skill"}
    assert sum(r.startswith("coverage:") for r in chosen.reasons.values()) >= 2
    # The budget is used: nothing that still fits was left behind.
    left = budget - chosen.tokens
    assert all(items[i].tokens > left for i in range(len(items)) if i not in chosen.indices)


def test_selection_orderings_do_what_they_say():
    items = items_for()
    utility = np.linspace(1.0, -1.0, len(items))
    top = select(np, items, 100, [], utility_order(np, utility))
    assert top.indices == sorted(top.indices) and top.indices[0] == 0      # highest estimated utility first
    a = select(np, items, 200, [], random_order(len(items), 3))
    assert a.indices == select(np, items, 200, [], random_order(len(items), 3)).indices
    assert a.indices != select(np, items, 200, [], random_order(len(items), 4)).indices
    assert a.sha256 == select(np, items, 200, [], random_order(len(items), 3)).sha256
    # A retention floor or a negative-score filter removes candidates entirely, coverage included.
    blocked = np.ones(len(items), dtype=bool)
    blocked[:10] = False
    kept = select(np, items, 300, [CoverageConstraint("skill", "a", min_records=3)], utility_order(np, utility), blocked)
    assert not set(kept.indices) & set(range(10))
    with pytest.raises(SelectionInfeasible, match="no candidate has"):
        select(np, items, 300, [CoverageConstraint("lang", "de", min_records=1)], random_order(len(items), 0))


def test_redundancy_penalty_and_k_center_spread_the_selection():
    # Two tight clusters; utility slightly prefers cluster 0.
    texts = [f"alpha beta gamma delta {i % 3}" for i in range(20)] + [f"zeta eta theta iota {i % 3}" for i in range(20)]
    items = [Item(f"r{i}", 10) for i in range(40)]
    features = text_features(np, texts)
    utility = np.concatenate([np.linspace(1.0, 0.9, 20), np.linspace(0.8, 0.7, 20)])
    greedy = select(np, items, 60, [], utility_order(np, utility))
    assert all(i < 20 for i in greedy.indices)
    spread = select(np, items, 60, [], UtilityOrder(np, utility, list(range(40)), features, 0.5))
    assert any(i >= 20 for i in spread.indices) and spread.indices[0] == 0
    centers = select(np, items, 20, [], KCenterOrder(np, features))
    assert {i >= 20 for i in centers.indices} == {True, False}        # one pick from each cluster


# ── cache, shards and resume ────────────────────────────────────────────────

def count_gradients(monkeypatch) -> list:
    calls = []
    real = Backend.example_gradient

    def counting(self, model, example):
        calls.append(example.id)
        return real(self, model, example)

    monkeypatch.setattr(Backend, "example_gradient", counting)
    return calls


def test_features_are_cached_by_content_and_reused(root, model_dir, monkeypatch):
    calls = count_gradients(monkeypatch)
    first = scored(root, model_dir)
    computed = len(calls)
    assert computed == 2 * (len(first.pool) + len(first.dev_selection))     # two checkpoints, everything once
    again = scored(root, model_dir, **{"output.run_dir": "run2"})           # another run, same cache
    assert len(calls) == computed and again.features == first.features
    entries = {e["key"].split(":")[1]: e for e in again.ledger.entries.values() if e["key"].startswith("less:")}
    assert all(e["cached"] for e in entries.values())                        # reused, and still charged
    assert again.ledger.total == pytest.approx(first.ledger.sum(keys=("less:",)))
    assert first.ledger.sum(stage="warmup") > 0 and first.ledger.sum(stage="scoring") > 0


@pytest.mark.parametrize("change", [
    {"selection.less.projection": {"kind": "countsketch", "dim": 64, "seed": 9}},
    {"selection.less.projection": {"kind": "countsketch", "dim": 32}},
    {"selection.less.optimizer_aware": False},
    {"selection.less.warmup": {"fraction": 0.25, "epochs": 2, "seed": 5}},
    {"recipe.learning_rate": 0.02},
    {"recipe.loss_masking": "final_assistant_only"},
    {"recipe.lora": {"r": 4, "alpha": 4, "target_modules": ["c_attn"]}},
    {"model.max_seq_len": 40},
])
def test_any_change_that_affects_features_invalidates_them(root, model_dir, monkeypatch, change):
    first = scored(root, model_dir)
    calls = count_gradients(monkeypatch)
    changed = scored(root, model_dir, **change, **{"output.run_dir": "run2"})
    assert changed.features != first.features and len(calls) > 0


def test_a_changed_dataset_or_template_invalidates_features(root, model_dir, monkeypatch):
    first = scored(root, model_dir)
    calls = count_gradients(monkeypatch)
    lines = (root / "candidates.jsonl").read_text().splitlines(keepends=True)
    (root / "candidates.jsonl").write_text("".join(lines[:-1]))
    assert scored(root, model_dir, **{"output.run_dir": "run2"}).features != first.features
    (root / "t.jinja").write_text(
        "{% for m in messages %}<|{{ m['role'] }}|> {{ m['content'] }}<|end|>{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")
    assert scored(root, model_dir, **{"output.run_dir": "run3", "model.chat_template": "t.jinja"}).features \
        != first.features
    assert len(calls) > 0
    # The seeds a recipe is trained with do not enter scoring, so they do not invalidate it.
    before = len(calls)
    scored(root, model_dir, **{"output.run_dir": "run4", "model.chat_template": "t.jinja", "recipe.seeds": [5]})
    assert len(calls) == before


def test_interrupted_scoring_resumes_at_the_first_unfinished_shard(root, model_dir, monkeypatch):
    reference = scored(root, model_dir, **{"output.cache_dir": "cache-reference", "output.run_dir": "run-reference"})
    shards = sorted(p.relative_to(reference.features) for p in reference.features.rglob("shard-*.npy"))
    assert len(shards) == 2 * math.ceil(len(reference.pool) / 8)

    real = Backend.example_gradient
    calls = []

    def interrupting(self, model, example):
        calls.append(example.id)
        if len(calls) == 20:
            raise KeyboardInterrupt
        return real(self, model, example)

    monkeypatch.setattr(Backend, "example_gradient", interrupting)
    with pytest.raises(KeyboardInterrupt):
        scored(root, model_dir)
    partial = next((root / "cache" / "features").iterdir())
    assert not (partial / "entry.json").exists()
    finished = sorted(p.relative_to(partial) for p in partial.rglob("shard-*.json"))
    assert len(finished) == 2                                   # 16 gradients done, the third shard cut short
    assert not list(partial.rglob("*.tmp"))                     # atomic writes leave no debris

    resumed_calls = count_gradients(monkeypatch)                # replaces the interrupting patch
    resumed = scored(root, model_dir)
    pool_calls = [c for c in resumed_calls if c.startswith("cand-")]
    assert len(pool_calls) == 2 * len(resumed.pool) - 16        # finished shards were not recomputed
    for shard in shards:
        assert np.array_equal(np.load(resumed.features / shard), np.load(reference.features / shard))
    # Interrupted work is paid for: the three gradients of the shard that was
    # cut short were computed and thrown away, and they are on the ledger.
    wasted = 3 * sum(e.tokens for e in resumed.pool[16:19])
    assert resumed.ledger.sum(stage="scoring") == pytest.approx(reference.ledger.sum(stage="scoring") + wasted)

    # A shard whose marker is gone is recomputed; nothing else is.
    (resumed.features / "entry.json").unlink()
    (resumed.features / shards[1]).with_suffix(".json").unlink()
    again = count_gradients(monkeypatch)
    scored(root, model_dir, **{"output.run_dir": "run2"})
    assert len([c for c in again if c.startswith("cand-")]) == 8


# ── the whole loop ──────────────────────────────────────────────────────────

# Four training epochs so that the scoring overhead fits inside the pool and
# a matched-compute arm exists (see the plan tests).
SMALL = {"recipe.seeds": [0, 1, 2], "recipe.epochs": 4, "evaluation.metrics": ["loss", "exact_match"], "evaluation.max_new_tokens": 6,
         "selection.coverage": [{"field": "lang", "value": "sw", "min_records": 2}],
         "selection.report_fields": ["skill", "lang"],
         "search": {"token_budgets": [80], "less_variants": [{"diversity_lambda": 0.5}], "max_trials": 4}}


@pytest.fixture(scope="module")
def finished(tmp_path_factory, model_dir):
    """One complete run shared by the tests below (read-only use)."""
    root = tmp_path_factory.mktemp("finished")
    make_data(root)
    exp = experiment(root, model_dir, **SMALL)
    lines: list[str] = []
    events: list[str] = []
    real_open, real_select = TestVault.open, runner_module.select

    def watched_open(self, freeze):
        events.append("open:" + ("frozen" if (exp.output.run_dir / "frozen.json").is_file() else "NOT-FROZEN"))
        return real_open(self, freeze)

    def watched_select(*args, **kwargs):
        events.append("select")
        return real_select(*args, **kwargs)

    TestVault.open, runner_module.select = watched_open, watched_select
    try:
        result = run(exp, log=lines.append)
    finally:
        TestVault.open, runner_module.select = real_open, real_select
    return {"root": root, "exp": exp, "report": result, "log": lines, "events": events}


def test_the_test_split_is_opened_once_after_the_freeze_and_after_all_selection(finished):
    events = finished["events"]
    assert events.count("open:frozen") == 1 and "open:NOT-FROZEN" not in events
    assert "select" in events and events.index("open:frozen") > max(i for i, e in enumerate(events) if e == "select")
    run_dir = finished["exp"].output.run_dir
    frozen = read_json(run_dir / "frozen.json")
    assert frozen["kind"] == "optimize-freeze" and frozen["basis"]["split"].startswith("dev")
    assert frozen["contract_fingerprint"] == finished["exp"].fingerprint()
    assert read_json(run_dir / "final.json")["frozen_trial"] == frozen["trial"]


def test_report_separates_estimates_from_measurements(finished):
    result = finished["report"]
    assert result["kind"] == "optimize-report" and result["schema_version"] == 1 and result["status"] == "final"
    assert set(result) >= {"estimated_signals", "development_results", "final_results", "compute", "coverage",
                           "limitations", "not_performed", "conclusion"}
    signals = result["estimated_signals"]
    assert "not a probability" in signals["note"] and "estimated conflict signal" in signals["note"]
    assert signals["records"] == 36 and signals["retention"] is not None
    assert signals["target_capabilities"] == ["reverse"] and signals["retain_capabilities"] == ["copy"]
    dev, final = result["development_results"], result["final_results"]
    assert "optimistic" in dev["note"] and "after the candidate and recipe were frozen" in final["note"]
    arms = {a["key"]: a for a in dev["arms"]}
    assert set(arms) == {"random@150", "diversity@150", "less@150", "random+overhead", "full",
                         "random@80", "diversity@80", "less@80", "less@150~diversity_lambda=0.5"}
    assert {a["key"] for a in final["arms"]} == set(arms)
    for arm in arms.values():
        loss = arm["metrics"]["loss"]
        assert loss["across_seeds"]["n"] == 3 and loss["across_seeds"]["std"] is not None     # variation across runs
        assert len(loss["evaluation_ci95"]) == 2                                             # evaluation uncertainty
        assert set(arm["by_capability"]) == {"reverse", "copy"}
        assert arm["by_capability"]["copy"]["role"] == "retain"
    assert arms["random+overhead"]["comparison"] == "matched_compute"
    assert arms["full"]["comparison"] == "full_data" and arms["less@150"]["comparison"] == "matched_tokens"
    assert arms["less@150"]["compute_fte"]["attributed_overhead"] > 0
    assert arms["random@150"]["compute_fte"]["attributed_overhead"] == 0
    assert dev["base_model"]["metrics"]["loss"]["across_seeds"]["n"] == 1
    text = (finished["exp"].output.run_dir / "report.md").read_text()
    order = [text.index(h) for h in ("## 1. Estimated training-value signals", "## 2. Measured development results",
                                     "## 3. Final held-out results", "## 4. Compute consumed",
                                     "## 5. Coverage changes", "## 6. Limitations and experiments not performed")]
    assert order == sorted(order)


def test_the_conclusion_is_honest_whichever_way_it_came_out(finished):
    """No assertion here depends on which arm won."""
    result = finished["report"]
    conclusion = result["conclusion"]
    assert conclusion["status"] in ("supported", "not_supported", "insufficient_evidence")
    assert conclusion["basis"] == "test" and conclusion["pilot"] is True and result["label"] == "pilot"
    assert any("pilot" in text and "does not establish" in text for text in result["limitations"])
    final = result["final_results"]["decision"]
    assert final["candidate"] == result["final_results"]["frozen"]["trial"]
    assert final["candidate"] == result["development_results"]["decision"]["candidate"]   # chosen on dev, not on test
    by_kind = {c["kind"]: c for c in final["comparisons"]}
    for comparison in by_kind.values():
        assert comparison["verdict"] in ("supported", "contradicted", "insufficient_evidence")
        assert comparison["seeds"] == [3, 3] and comparison["evaluation_ci95"] is not None
    if conclusion["status"] == "supported":
        assert by_kind["matched_tokens"]["verdict"] == by_kind["matched_compute"]["verdict"] == "supported"
    exposure = result["final_results"]["test_exposure"]
    assert exposure["final_evaluations"] == 1


def test_a_small_model_is_labeled_a_pilot_even_if_the_contract_says_benchmark(root, model_dir):
    result = run(experiment(root, model_dir, scale="benchmark", **{"selection.policies": ["random"],
                                                                 "baselines.full_data": False}))
    assert result["label"] == "pilot" and result["conclusion"]["pilot"] is True
    setup = read_json(root / "run" / "setup.json")
    assert "fewer than 100,000,000 parameters" in setup["pilot_reason"]
    # Without a model-aware arm there is nothing to conclude about model-aware selection.
    assert result["conclusion"]["status"] == "insufficient_evidence"
    assert result["estimated_signals"] is None


def test_fewer_than_three_seeds_is_insufficient_evidence(root, model_dir):
    result = run(experiment(root, model_dir, **{"recipe.seeds": [0, 1]}))
    assert result["conclusion"]["status"] == "insufficient_evidence"
    for section in ("development_results", "final_results"):
        verdicts = {c["verdict"] for c in result[section]["decision"]["comparisons"]}
        assert verdicts == {"insufficient_evidence"}
    assert any("Fewer than 3 seeds" in text for text in result["limitations"])


def test_every_stage_is_charged_and_overhead_is_visible(finished):
    compute = finished["report"]["compute"]
    stages = compute["by_stage"]
    for stage in ("warmup", "scoring", "pilot_training", "evaluation", "final_evaluation"):
        assert stages[stage]["fte"] > 0, stage
    assert "selection" in stages and stages["selection"]["wall_seconds"] >= 0
    assert compute["total"] == pytest.approx(sum(s["fte"] for s in stages.values()))
    assert compute["optimization_overhead"] == pytest.approx(stages["warmup"]["fte"] + stages["scoring"]["fte"])
    assert 0 < compute["optimization_overhead_share"] < 1 and compute["total"] <= compute["budget"]
    ledger = read_json(finished["exp"].output.run_dir / "ledger.json")
    backward = {e["stage"] for e in ledger["entries"] if e["backward_tokens"]}
    assert backward == {"warmup", "scoring", "pilot_training"}       # evaluation never backpropagates
    # The matched-compute arm really was given more training tokens than the matched-token arms.
    arms = {a["key"]: a for a in finished["report"]["development_results"]["arms"]}
    assert arms["random+overhead"]["training_tokens_per_epoch"] > arms["random@150"]["training_tokens_per_epoch"]
    assert arms["less@150"]["training_tokens_per_epoch"] <= 150 >= arms["random@150"]["training_tokens_per_epoch"]


def test_coverage_and_selection_artifacts(finished):
    result, run_dir = finished["report"], finished["exp"].output.run_dir
    coverage = result["coverage"]["trials"]
    for key in ("random@150", "diversity@150", "less@150", "less@80"):
        constraint = coverage[key]["constraints"][0]
        assert constraint["field"] == "lang" and constraint["selected_records"] >= 2
        assert set(coverage[key]["distribution"]) == {"lang", "skill"}
    assert coverage["full"]["distribution"]["skill"]["reverse"]["pool_token_share"] == \
        coverage["full"]["distribution"]["skill"]["reverse"]["selected_token_share"]
    selection = read_json(run_dir / "trials" / trial_slug("less@150") / "seed-0" / "selection.json")
    assert selection["tokens"] <= 150 and selection["records"] == len(selection["selected"])
    reasons = {row_["reason"].split(":")[0] for row_ in selection["selected"]}
    assert reasons == {"coverage", "policy"}
    assert all({"target_alignment", "retention_alignment"} <= set(row_["scores"]) for row_ in selection["selected"])
    randoms = [read_json(run_dir / "trials" / trial_slug("random@150") / f"seed-{s}" / "selection.json")["sha256"]
               for s in (0, 1, 2)]
    assert len(set(randoms)) == 3                                    # a fresh random draw per seed
    less = [read_json(run_dir / "trials" / trial_slug("less@150") / f"seed-{s}" / "selection.json")["sha256"]
            for s in (0, 1, 2)]
    assert len(set(less)) == 1                                       # a deterministic selection
    scores = np.load(run_dir / "scores" / "primary.npz")
    assert scores["target"].shape == (36,) and "retention" in scores.files


def test_no_training_content_reaches_logs_or_artifacts(finished):
    root, run_dir = finished["root"], finished["exp"].output.run_dir
    contents = set()
    for name in ("candidates", "dev", "test"):
        for line in (root / f"{name}.jsonl").read_text().splitlines():
            contents.update(m["content"] for m in json.loads(line)["messages"])
    haystack = "\n".join(finished["log"])
    for path in [*run_dir.rglob("*.json"), *run_dir.rglob("*.md"), *(root / "cache").rglob("*.json")]:
        if path.parent.name == "adapter":
            continue
        haystack += path.read_text()
    leaked = [c for c in contents if len(c) >= 4 and (f'"{c}"' in haystack or f": {c}" in haystack)]
    assert not leaked
    assert finished["log"] and all("cand-" not in line for line in finished["log"])


def test_export_of_a_real_run_is_faithful(finished, tmp_path):
    root, run_dir = finished["root"], finished["exp"].output.run_dir
    source = (root / "candidates.jsonl").read_bytes()
    out = tmp_path / "selected.jsonl"
    manifest = export(run_dir, out)
    assert (root / "candidates.jsonl").read_bytes() == source
    lines = source.splitlines(keepends=True)
    exported = out.read_bytes().splitlines(keepends=True)
    assert exported == [lines[r["source_line"] - 1] for r in manifest["records"]]
    frozen = read_json(run_dir / "frozen.json")
    assert manifest["selection"]["trial"] == frozen["trial"] and manifest["selection"]["frozen_candidate"]
    assert manifest["selection"]["sha256"] == frozen["selection_sha256"]
    assert manifest["evidence"]["conclusion"] == finished["report"]["conclusion"]
    assert manifest["fingerprints"]["contract"] == finished["exp"].fingerprint()
    baseline = export(run_dir, tmp_path / "random.jsonl", trial="random@150", seed=1)
    assert baseline["selection"]["frozen_candidate"] is (frozen["trial"] == "random@150" and frozen["selection_seed"] == 1)
    with pytest.raises(ConfigError, match="never overwrites an input"):
        export(run_dir, root / "candidates.jsonl", overwrite=True)
    with pytest.raises(ConfigError, match="output.run_dir must not contain"):
        run(experiment(root, finished["exp"].model.path, **{"output.run_dir": "."}))


def test_a_sealed_run_accepts_no_new_trials_and_no_other_experiment(finished, model_dir):
    root = finished["root"]
    before = sorted(p.name for p in (root / "run" / "trials").iterdir())
    more_search = {**SMALL, "search": {"token_budgets": [80, 100], "max_trials": 9}}
    with pytest.raises(ConfigError, match="is sealed"):
        run(experiment(root, model_dir, **more_search))
    with pytest.raises(ConfigError, match="different experiment"):
        run(experiment(root, model_dir, **{**SMALL, "recipe.epochs": 2}))
    with pytest.raises(ConfigError, match="different experiment"):
        run(experiment(root, model_dir, **{**SMALL, "evaluation.metrics": ["loss"]}))
    assert sorted(p.name for p in (root / "run" / "trials").iterdir()) == before
    # Rerunning the same contract is a no-op that returns the same report.
    again = run(finished["exp"])
    assert again["conclusion"] == finished["report"]["conclusion"]
    assert again["compute"]["total"] == finished["report"]["compute"]["total"]


def test_selection_does_not_depend_on_the_test_split(tmp_path, model_dir):
    """Two runs that differ only in their test examples make identical
    selections and freeze the same candidate."""
    outcomes = []
    for name, seed in (("a", 1), ("b", 2)):
        root = tmp_path / name
        make_data(root, seed=0)
        other = make_data(tmp_path / f"other-{name}", seed=10 + seed)
        write_jsonl(root / "test.jsonl", other["test"])
        run(experiment(root, model_dir))
        selections = {p.parent.parent.name + "/" + p.parent.name: read_json(p)["sha256"]
                      for p in (root / "run").glob("trials/*/seed-*/selection.json")}
        dev = {p.parent.parent.name + "/" + p.parent.name: read_json(p)["dev"]["loss"]
               for p in (root / "run").glob("trials/*/seed-*/result.json")}
        outcomes.append((selections, dev, read_json(root / "run" / "frozen.json")["trial"]))
    assert outcomes[0][0] == outcomes[1][0] and len(outcomes[0][0]) == 8
    assert outcomes[0][1] == outcomes[1][1]
    assert outcomes[0][2] == outcomes[1][2]


# ── resume and budget in the loop ───────────────────────────────────────────

def count_training(monkeypatch, stop_at: int | None = None) -> list:
    calls = []
    real = Backend.train

    def counting(self, model, examples, **kwargs):
        calls.append(len(examples))
        if stop_at is not None and len(calls) == stop_at:
            raise KeyboardInterrupt
        return real(self, model, examples, **kwargs)

    monkeypatch.setattr(Backend, "train", counting)
    return calls


def test_an_interrupted_run_resumes_without_repeating_or_double_charging(root, model_dir, monkeypatch):
    exp = experiment(root, model_dir)
    reference_root = root / "reference"
    make_data(reference_root)
    reference = run(experiment(reference_root, model_dir))

    count_training(monkeypatch, stop_at=5)            # warmup is call 1; four trial seeds finish
    with pytest.raises(KeyboardInterrupt):
        run(exp)
    done = sorted((root / "run").glob("trials/*/seed-*/result.json"))
    assert len(done) == 3 and not (root / "run" / "frozen.json").exists()
    assert not list((root / "run").rglob("*.tmp")) and not list((root / "cache").rglob("*.tmp"))
    partial = report(root / "run")
    assert partial["status"] == "development" and partial["final_results"] is None

    calls = count_training(monkeypatch)
    resumed = run(exp)
    assert len(calls) == 8 - 3                        # the three finished trial seeds were not retrained
    assert resumed["status"] == "final"
    # Same decisions and same measurements as a run that was never interrupted.
    assert resumed["conclusion"] == reference["conclusion"]
    for a, b in zip(resumed["final_results"]["arms"], reference["final_results"]["arms"]):
        assert a["key"] == b["key"]
        assert a["metrics"]["loss"]["across_seeds"]["values"] == pytest.approx(b["metrics"]["loss"]["across_seeds"]["values"])
    # Charged once: the warmup cut-off cost nothing extra, finished work was not re-billed.
    assert resumed["compute"]["total"] == pytest.approx(reference["compute"]["total"])

    again = count_training(monkeypatch)
    run(exp)
    assert again == []


def test_no_final_keeps_the_run_open_for_more_search(root, model_dir, monkeypatch):
    opened = []
    monkeypatch.setattr(TestVault, "open", lambda self, freeze: opened.append(1))
    first = run(experiment(root, model_dir), final=False)
    assert first["status"] == "development" and first["final_results"] is None and not opened
    assert not (root / "run" / "frozen.json").exists()
    assert any("stop before the final evaluation" in item["reason"] for item in first["not_performed"])
    monkeypatch.undo()

    calls = count_training(monkeypatch)
    search = {"token_budgets": [90], "less_variants": [{"diversity_lambda": 0.4}], "max_trials": 4}
    second = run(experiment(root, model_dir, search=search), final=False)
    arms = {a["key"] for a in second["development_results"]["arms"]}
    assert {"less@90", "random@90", "diversity@90", "less@150~diversity_lambda=0.4"} <= arms
    assert len(calls) == 4 * 2                        # only the new arms trained; scoring came from the cache
    final = run(experiment(root, model_dir, search=search))
    assert final["status"] == "final" and (root / "run" / "frozen.json").exists()
    assert not any("stop before" in item["reason"] for item in final["not_performed"])


def test_the_run_refuses_before_training_when_the_budget_cannot_cover_the_essentials(root, model_dir, monkeypatch):
    calls = count_training(monkeypatch)
    with pytest.raises(ConfigError, match="budget.total_compute"):
        run(experiment(root, model_dir, **{"budget.total_compute": 2000}))
    assert calls == [] and not (root / "run" / "trials").exists()


def test_a_tight_budget_drops_optional_arms_and_reports_them(root, model_dir):
    planned = plan(experiment(root, model_dir))
    essential = sum(u["estimate_fte"] for u in planned["units"] if u["essential"])
    assert "exact" in planned["data"]["token_counts"]
    budget = essential + planned["budget"]["estimated_by_stage"]["final_evaluation"]
    result = run(experiment(root, model_dir, **{"budget.total_compute": budget}))
    arms = {a["key"]: a for a in result["development_results"]["arms"]}
    assert "full" not in arms
    assert all(a["metrics"]["loss"]["across_seeds"]["n"] == 1 for a in arms.values())
    skipped = " | ".join(item["what"] for item in result["not_performed"])
    assert "full-data training" in skipped and "seed 1" in skipped
    assert result["compute"]["total"] <= budget
    assert result["conclusion"]["status"] == "insufficient_evidence"


def test_overlength_and_leaky_candidates_are_handled_explicitly(root, model_dir):
    rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
    dev0 = json.loads((root / "dev.jsonl").read_text().splitlines()[0])
    long_row = row(700, "copy", "abcdefghijklmnopqrstuvwxyz", "cand")
    write_jsonl(root / "candidates.jsonl", rows + [long_row, {**dev0, "id": "leak"}])
    with pytest.raises(ConfigError, match="exceed model.max_seq_len=48"):
        plan(experiment(root, model_dir))
    planned = plan(experiment(root, model_dir, **{"model.on_overlength": "exclude"}))
    assert planned["data"]["pool"] == {**planned["data"]["pool"], "records": 36, "excluded_for_leakage": 1,
                                       "excluded_for_length": 1}
    assert planned["leakage"]["exact_findings"] == 1
    with pytest.raises(ConfigError, match="match no module"):
        run(experiment(root, model_dir, **{"model.on_overlength": "exclude",
                                           "recipe.lora": {"r": 2, "alpha": 4, "target_modules": ["q_proj"]}}))


def test_models_are_never_downloaded_without_opt_in(root, monkeypatch):
    exp = parse_experiment(contract(root, **{"model.path": "parallelogram-tests/does-not-exist",
                                             "model.revision": "0" * 40}), base=root)
    with pytest.raises(OptimizeError, match="--allow-download"):
        run(exp)
    import os
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    assert not (root / "run" / "trials").exists()


def test_cli_end_to_end(root, model_dir):
    path = root / "experiment.json"
    path.write_text(json.dumps(contract(root, **{"model.path": str(model_dir)})))
    planned = cli.invoke(app, ["optimize", "plan", str(path)])
    assert planned.exit_code == 0 and "exact (model tokenizer" in planned.stdout
    done = cli.invoke(app, ["optimize", "run", str(path), "--json", "--quiet"])
    assert done.exit_code == 0, done.output
    payload = json.loads(done.stdout)
    assert payload["kind"] == "optimize-report" and payload["status"] == "final"
    shown = cli.invoke(app, ["optimize", "report", str(root / "run"), "--out", str(root / "r.md")])
    assert shown.exit_code == 0 and (root / "r.md").read_text().startswith("# Optimize report: test")
    out = root / "selected.jsonl"
    exported = cli.invoke(app, ["optimize", "export", str(root / "run"), "--output", str(out), "--json"])
    assert exported.exit_code == 0 and json.loads(exported.stdout)["output"]["records"] == len(out.read_text().splitlines())
    # The selected file is a valid dataset for the rest of the toolchain.
    assert cli.invoke(app, ["check", str(out)]).exit_code == 0


# ── defects found in review, kept fixed ─────────────────────────────────────

def test_dropout_is_off_during_scoring(root, model_dir):
    """LoRA dropout must not make an example's gradient depend on when it is
    visited: a resumed scoring pass would disagree with an uninterrupted one."""
    lora = {"r": 2, "alpha": 4, "dropout": 0.5, "target_modules": ["c_attn"]}
    backend = Backend(experiment(root, model_dir, **{"recipe.lora": lora}))
    _, encoded = pool(backend, root)
    with backend.adapter(0) as model:
        assert not model.training
        backend.train(model, encoded[:8], seed=0)
        assert not model.training
        first, second = backend.example_gradient(model, encoded[9]), backend.example_gradient(model, encoded[9])
        assert float(first.norm()) > 0 and torch.equal(first, second)
        a, _ = backend.losses(model, encoded[:5])
        b, _ = backend.losses(model, encoded[:5])
        assert a == b


def test_edited_records_with_unchanged_ids_miss_every_cache(root, model_dir, monkeypatch):
    first = scored(root, model_dir)
    rows = [json.loads(line) for line in (root / "candidates.jsonl").read_text().splitlines()]
    for r in rows:      # same ids, same lengths, different answers
        r["messages"][1]["content"] = r["messages"][1]["content"][::-1] or "x"
    write_jsonl(root / "candidates.jsonl", rows)
    with pytest.raises(ConfigError, match="different experiment"):
        scored(root, model_dir)                              # the run directory is bound to the old bytes
    calls = count_gradients(monkeypatch)
    edited = scored(root, model_dir, **{"output.run_dir": "run2"})
    assert edited.features != first.features and len(calls) > 0
    assert edited.pool_fingerprint != first.pool_fingerprint
    assert [e.tokens for e in edited.pool] == [e.tokens for e in first.pool]      # nothing else gave it away


def test_results_from_an_earlier_contract_are_never_frozen(root, model_dir):
    """Change the selection settings and cut the budget so the arms cannot be
    retrained: the old results must not be reported or frozen as current."""
    first = run(experiment(root, model_dir), final=False)
    assert {a["key"] for a in first["development_results"]["arms"]} >= {"less@150", "random@150"}
    spent = first["compute"]["total"]
    coverage = [{"field": "skill", "value": "upper", "min_records": 6}]
    second = run(experiment(root, model_dir, **{"selection.coverage": coverage,
                                                "budget.total_compute": spent + 600}))
    # Full-data training does not depend on coverage, so its result still
    # belongs to the new contract. Every subset arm is stale and is left out.
    assert [a["key"] for a in second["development_results"]["arms"]] == ["full"]
    assert second["development_results"]["decision"]["candidate"] is None
    assert not (root / "run" / "frozen.json").exists() and second["final_results"] is None
    assert any("remaining budget" in item["reason"] for item in second["not_performed"])
    assert all(key.startswith("full-") for key in read_json(root / "run" / "current.json")["results"])
    with pytest.raises(ConfigError, match="Nothing is frozen"):
        export(root / "run", root / "out.jsonl")
    # With budget to retrain, the new contract's arms replace the stale ones.
    third = run(experiment(root, model_dir, **{"selection.coverage": coverage}))
    selection = read_json(root / "run" / "trials" / trial_slug("less@150") / "seed-0" / "selection.json")
    assert selection["coverage"][0]["selected_records"] >= 6 and third["status"] == "final"


def test_rescoring_cannot_overdraw_the_budget(root, model_dir):
    first = run(experiment(root, model_dir), final=False)
    spent = first["compute"]["total"]
    changed = {"selection.less.projection": {"kind": "countsketch", "dim": 32}, "budget.total_compute": spent + 500}
    with pytest.raises(ConfigError, match="does not cover the scoring overhead"):
        run(experiment(root, model_dir, **changed))
    ledger = read_json(root / "run" / "ledger.json")
    assert sum(e["fte"] for e in ledger["entries"]) == pytest.approx(spent)


def test_interrupted_training_is_charged(root, model_dir, monkeypatch):
    reference_root = root / "reference"
    make_data(reference_root)
    reference = run(experiment(reference_root, model_dir, **{"recipe.epochs": 3}), final=False)
    real = Backend.example_losses
    seen = []

    def interrupting(self, model, batch):
        if model.training:
            seen.append(len(batch))
            if len(seen) == 45:          # well into the trial trainings, past the warmup
                raise KeyboardInterrupt
        return real(self, model, batch)

    monkeypatch.setattr(Backend, "example_losses", interrupting)
    with pytest.raises(KeyboardInterrupt):
        run(experiment(root, model_dir, **{"recipe.epochs": 3}), final=False)
    monkeypatch.setattr(Backend, "example_losses", real)
    ledger = read_json(root / "run" / "ledger.json")["entries"]
    training = [e for e in ledger if e["stage"] == "pilot_training"]
    finished = len(list((root / "run").glob("trials/*/seed-*/result.json")))
    assert len(training) == finished + 1                  # the cut-short pass has its own entry
    wasted = training[-1]["fte"]
    assert wasted > 0
    resumed = run(experiment(root, model_dir, **{"recipe.epochs": 3}), final=False)
    assert resumed["compute"]["total"] == pytest.approx(reference["compute"]["total"] + wasted)
    assert [a["key"] for a in resumed["development_results"]["arms"]] == \
        [a["key"] for a in reference["development_results"]["arms"]]


def test_an_arm_with_nothing_to_select_is_skipped_not_fatal(root, model_dir):
    for extra in ({}, {"selection.less.diversity_lambda": 0.5}):
        target = root / ("mmr" if extra else "plain")
        make_data(target)
        result = run(experiment(target, model_dir, **{"selection.less.retention_floor": 2.0}, **extra), final=False)
        arms = {a["key"] for a in result["development_results"]["arms"]}
        assert "less@150" not in arms and {"random@150", "diversity@150", "full"} <= arms
        assert any("leave no candidate" in item["reason"] for item in result["not_performed"])


def test_a_test_example_without_a_capability_is_refused_before_anything_runs(root, model_dir):
    rows = [json.loads(line) for line in (root / "test.jsonl").read_text().splitlines()]
    del rows[3]["skill"]
    write_jsonl(root / "test.jsonl", rows)
    with pytest.raises(ConfigError, match="Test example at line 4 has no 'skill' value"):
        plan(experiment(root, model_dir))
    with pytest.raises(ConfigError, match="Test example at line 4"):
        run(experiment(root, model_dir))
    assert not (root / "run").exists()


def test_without_a_matched_compute_arm_no_tradeoff_win_is_claimed(root, model_dir):
    result = run(experiment(root, model_dir, **{"recipe.seeds": [0, 1, 2]}))
    arms = {a["key"] for a in result["development_results"]["arms"]}
    assert "random+overhead" not in arms and "full" in arms
    assert any(item["what"] == "matched-compute baseline" for item in result["not_performed"])
    assert result["conclusion"]["status"] != "supported"
