"""LESS-inspired gradient scoring.

The idea, from Xia et al., "LESS: Selecting Influential Data for Targeted
Instruction Tuning" (2024): a training example is useful for a target task to
the extent that the parameter update it causes points the same way as the
update that would reduce the target's loss. Concretely, for checkpoints
`i = 1..C` saved during a short LoRA warmup,

    score(z) = sum_i  w_i * cos( P * Gamma(z; theta_i),  P * mean_dev grad L(dev; theta_i) )

where `Gamma` is the direction Adam would move the adapter for example `z`
alone, `P` is a fixed random projection, and `w_i` weights checkpoints by
their learning rate.

What a score is and is not: it is a first-order estimate of how one update
from `z` changes a differentiable development *loss*. It is not a task metric
(answer accuracy is measured separately, by generation), not a probability,
and a negative score is an estimated conflict with the development gradient,
not evidence that an example is wrong or that removing it will help.

Deviations from the reference implementation are listed in
docs/optimize.md; the ones that change numbers are: CountSketch instead of a
dense Rademacher projection, Adam bias correction included in `Gamma`,
checkpoint weights normalized to sum to one, and example-mean loss.
"""
from __future__ import annotations

import hashlib
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

from .backend import Backend, Encoded, TrainStats, is_oom
from .budget import Cost
from .cache import Cache, atomic_write, read_json, write_json
from .config import LessSpec, digest
from .errors import OptimizeError, OutOfMemory


# ── update direction ────────────────────────────────────────────────────────

def adam_direction(g: Any, exp_avg: Any, exp_avg_sq: Any, step: int, beta1: float, beta2: float,
                   eps: float) -> Any:
    """The step Adam would take for gradient `g`, divided by `-lr`.

    Given the optimizer's moment estimates after `step` updates, the next
    update is `-lr * m_hat / (sqrt(v_hat) + eps)` with the moments advanced by
    `g`. Training uses Adam, so this, not the raw gradient, is the direction
    an example actually pushes the weights; the raw gradient would overstate
    examples whose gradient lies along coordinates Adam scales down.
    (Decoupled weight decay adds `-lr * wd * theta`, the same for every
    example, and is left out.)
    """
    t = step + 1
    m_hat = (beta1 * exp_avg + (1 - beta1) * g) / (1 - beta1 ** t)
    v_hat = (beta2 * exp_avg_sq + (1 - beta2) * g * g) / (1 - beta2 ** t)
    return m_hat / (v_hat.sqrt() + eps)


# ── projection ──────────────────────────────────────────────────────────────

class Projector:
    """Deterministic random projection of flat adapter gradients.

    `countsketch` sends coordinate `j` to bucket `h(j)` with sign `s(j)`. It
    is a Johnson-Lindenstrauss transform: inner products are preserved in
    expectation with variance on the order of `|a|^2 |b|^2 / dim`, it costs
    one pass over the gradient, and it needs only the index and sign vectors
    in memory rather than a `parameters x dim` matrix. `identity` keeps the
    gradient as is, for models small enough and for checking the math.

    `h` and `s` come from a generator seeded per parameter tensor by
    (seed, name, size), so the same contract always yields the same
    projection regardless of the order tensors are visited in.
    """

    def __init__(self, torch: Any, sizes: Sequence[tuple[str, int]], kind: str, dim: int, seed: int,
                 device: Any = "cpu"):
        self.torch = torch
        self.kind = kind
        self.parameters = sum(n for _, n in sizes)
        if kind == "identity":
            self.dim = self.parameters
            return
        self.dim = dim
        index, sign = [], []
        for name, n in sizes:
            material = hashlib.sha256(f"{seed}:{name}:{n}".encode()).hexdigest()
            generator = torch.Generator().manual_seed(int(material[:15], 16))
            index.append(torch.randint(0, dim, (n,), generator=generator))
            sign.append(torch.randint(0, 2, (n,), generator=generator) * 2 - 1)
        self.index = torch.cat(index).to(device)
        self.sign = torch.cat(sign).to(device=device, dtype=torch.float32)

    def __call__(self, flat: Any) -> Any:
        flat = flat.float()
        if self.kind == "identity":
            return flat
        out = self.torch.zeros(self.dim, dtype=self.torch.float32, device=flat.device)
        return out.index_add_(0, self.index, flat * self.sign)


# ── warmup ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Checkpoint:
    path: Path
    avg_lr: float
    step: int


def warmup_subset(n: int, spec: LessSpec, batch_size: int) -> list[int]:
    """Indices of the seeded random sample the warmup trains on."""
    k = min(n, max(math.ceil(spec.warmup_fraction * n), min(n, batch_size)))
    return sorted(random.Random(f"warmup:{spec.warmup_seed}").sample(range(n), k))


def run_warmup(backend: Backend, pool: Sequence[Encoded], spec: LessSpec, cache: Cache, inputs: dict,
               log: Callable[[str], None]) -> tuple[list[Checkpoint], Cost, bool]:
    """Train a short LoRA warmup and keep one checkpoint per epoch.

    At initialization LoRA's `B` matrices are zero, so gradients with respect
    to `A` vanish and say nothing; a brief warmup moves the adapter somewhere
    gradients are informative and gives Adam its moment estimates. The warmup
    is bounded by a data fraction, an epoch count and an optional step cap,
    and its checkpoints are fixed in advance (end of each epoch), never chosen
    by looking at results.

    Returns `(checkpoints, cost, from_cache)`. The cost is the original cost
    even on a cache hit, so the ledger can charge it either way.
    """
    torch = backend.torch
    directory = cache.dir("warmup", inputs)
    meta = cache.complete("warmup", inputs)
    if meta is not None:
        checkpoints = [Checkpoint(directory / c["file"], c["avg_lr"], c["step"]) for c in meta["checkpoints"]]
        if all(c.path.is_file() for c in checkpoints):
            return checkpoints, Cost(**meta["cost"]), True
    subset = [pool[i] for i in warmup_subset(len(pool), spec, backend.exp.recipe.batch_size)]
    saved: list[dict] = []
    # Work thrown away by an earlier interrupted attempt is still compute
    # that was spent on this experiment; it is carried here and charged.
    wasted_path = directory / "interrupted.json"
    wasted = Cost(**read_json(wasted_path)["cost"]) if wasted_path.is_file() else Cost()

    def on_epoch(epoch: int, optimizer: Any, avg_lr: float, step: int) -> None:
        state = {"params": {}, "exp_avg": {}, "exp_avg_sq": {}, "step": 0}
        for name, p in backend.trainable(model):
            adam = optimizer.state[p]
            state["params"][name] = p.detach().cpu().clone()
            state["exp_avg"][name] = adam["exp_avg"].detach().cpu().clone()
            state["exp_avg_sq"][name] = adam["exp_avg_sq"].detach().cpu().clone()
            state["step"] = int(adam["step"])
        file = f"checkpoint-{epoch:03d}.pt"
        atomic_write(directory / file, lambda f: torch.save(state, f))
        saved.append({"file": file, "avg_lr": avg_lr, "step": step})

    log(f"warmup: {len(subset)} of {len(pool)} candidates, {spec.warmup_epochs} epoch(s)")
    stats = TrainStats()
    try:
        with backend.adapter(spec.warmup_seed) as model:
            backend.train(model, subset, seed=spec.warmup_seed, epochs=spec.warmup_epochs,
                          max_steps=spec.warmup_max_steps, on_epoch=on_epoch, stats=stats)
    except BaseException:
        write_json(wasted_path, {"cost": asdict(wasted + stats.cost)})
        raise
    if not saved:
        raise OptimizeError("The warmup produced no checkpoint; raise selection.less.warmup.max_steps")
    cost = stats.cost + wasted
    cache.finish("warmup", inputs, {"checkpoints": saved, "cost": asdict(cost), "train": stats.to_dict(),
                                    "warmup_records": len(subset)})
    return [Checkpoint(directory / c["file"], c["avg_lr"], c["step"]) for c in saved], cost, False


def _load_checkpoint(backend: Backend, model: Any, checkpoint: Checkpoint) -> dict:
    """Put a warmup checkpoint's adapter weights into `model`; return its
    flattened Adam state in the same parameter order as the gradients."""
    torch = backend.torch
    state = torch.load(checkpoint.path, map_location="cpu", weights_only=True)
    names = [n for n, _ in backend.trainable(model)]
    if set(names) != set(state["params"]):
        raise OptimizeError("A cached warmup checkpoint does not match this adapter's parameters")
    with torch.no_grad():
        for name, p in backend.trainable(model):
            p.copy_(state["params"][name].to(device=p.device, dtype=p.dtype))
    flat = lambda key: torch.cat([state[key][n].reshape(-1).float() for n in names]).to(backend.device)  # noqa: E731
    return {"exp_avg": flat("exp_avg"), "exp_avg_sq": flat("exp_avg_sq"), "step": state["step"]}


# ── features ────────────────────────────────────────────────────────────────

def _shard_paths(directory: Path, shard: int) -> tuple[Path, Path]:
    return directory / f"shard-{shard:05d}.npy", directory / f"shard-{shard:05d}.json"


def _shard_done(directory: Path, shard: int, ids_sha: str, np: Any) -> bool:
    data, marker = _shard_paths(directory, shard)
    if not (data.is_file() and marker.is_file()):
        return False
    try:
        return read_json(marker).get("ids_sha256") == ids_sha
    except (OSError, ValueError):
        return False


def compute_features(backend: Backend, pool: Sequence[Encoded], checkpoints: Sequence[Checkpoint],
                     spec: LessSpec, cache: Cache, inputs: dict,
                     log: Callable[[str], None]) -> tuple[Path, Cost, bool]:
    """Projected update direction of every candidate at every checkpoint.

    Stored as `checkpoint-NNN/shard-NNNNN.npy` (float32, `shard_size` rows) in
    the candidate pool's order. A shard's JSON marker is written after its
    data, so an interrupted run resumes at the first shard without a marker
    and nothing already computed is redone. Per-example gradients are taken
    one example at a time: simple, exact, and bounded in memory by one
    sequence.
    """
    np, torch = backend.ml.np, backend.torch
    directory = cache.dir("features", inputs)
    meta = cache.complete("features", inputs)
    if meta is not None:
        return directory, Cost(**meta["cost"]), True
    recipe = backend.exp.recipe
    progress_path = directory / "progress.json"
    spent = Cost(**read_json(progress_path)["cost"]) if progress_path.is_file() else Cost()
    shards = [pool[i:i + spec.shard_size] for i in range(0, len(pool), spec.shard_size)]
    with backend.adapter(spec.warmup_seed) as model:
        sizes = [(n, p.numel()) for n, p in backend.trainable(model)]
        projector = Projector(torch, sizes, spec.projection, spec.projection_dim, spec.projection_seed,
                              backend.device)
        for ci, checkpoint in enumerate(checkpoints):
            adam = _load_checkpoint(backend, model, checkpoint)
            target = directory / f"checkpoint-{ci:03d}"
            target.mkdir(parents=True, exist_ok=True)
            for si, shard in enumerate(shards):
                ids_sha = digest([e.id for e in shard])
                if _shard_done(target, si, ids_sha, np):
                    continue
                started = time.monotonic()
                rows = np.zeros((len(shard), projector.dim), dtype=np.float32)
                done_tokens = 0
                try:
                    for row, example in enumerate(shard):
                        try:
                            g = backend.example_gradient(model, example)
                        except Exception as exc:  # noqa: BLE001 - filtered by is_oom
                            if not is_oom(exc):
                                raise
                            backend.free()
                            raise OutOfMemory(
                                f"Out of memory computing the gradient of one {example.tokens}-token example. "
                                "Lower model.max_seq_len or use a device with more memory; finished shards "
                                "are cached and will not be recomputed.") from exc
                        if spec.optimizer_aware:
                            g = adam_direction(g, adam["exp_avg"], adam["exp_avg_sq"], adam["step"],
                                               recipe.beta1, recipe.beta2, recipe.eps)
                        rows[row] = projector(g).cpu().numpy()
                        done_tokens += example.tokens
                finally:
                    # Charge what was computed, finished shard or not. Progress is
                    # written before the shard's marker: if the process dies in
                    # between, the shard is redone and counted twice, which errs
                    # toward over-charging rather than hiding work.
                    spent = spent + Cost(done_tokens, done_tokens, time.monotonic() - started)
                    write_json(progress_path, {"cost": asdict(spent)})
                data_path, marker = _shard_paths(target, si)
                atomic_write(data_path, lambda f, rows=rows: np.save(f, rows))
                write_json(marker, {"ids_sha256": ids_sha, "rows": len(shard)})
                log(f"scoring: checkpoint {ci + 1}/{len(checkpoints)}, shard {si + 1}/{len(shards)}")
    cache.finish("features", inputs, {"cost": asdict(spent), "checkpoints": len(checkpoints),
                                      "shards": len(shards), "rows": len(pool), "dim": projector.dim})
    return directory, spent, False


def dev_gradients(backend: Backend, groups: dict[str, Sequence[Encoded]], checkpoints: Sequence[Checkpoint],
                  spec: LessSpec, cache: Cache, inputs: dict) -> tuple[Path, Cost, bool]:
    """Projected mean development gradient per capability and checkpoint.

    These are plain (SGD) gradients of the development loss: the question is
    which training updates reduce that loss, and its descent direction does
    not depend on the training optimizer.
    """
    np, torch = backend.ml.np, backend.torch
    directory = cache.dir("devgrads", inputs)
    meta = cache.complete("devgrads", inputs)
    if meta is not None:
        return directory, Cost(**meta["cost"]), True
    started = time.monotonic()
    tokens = 0
    with backend.adapter(spec.warmup_seed) as model:
        sizes = [(n, p.numel()) for n, p in backend.trainable(model)]
        projector = Projector(torch, sizes, spec.projection, spec.projection_dim, spec.projection_seed,
                              backend.device)
        for ci, checkpoint in enumerate(checkpoints):
            _load_checkpoint(backend, model, checkpoint)
            vectors = {}
            for name in sorted(groups):
                total = torch.zeros(projector.dim, device=backend.device)
                for example in groups[name]:
                    try:
                        total += projector(backend.example_gradient(model, example))
                    except Exception as exc:  # noqa: BLE001 - filtered by is_oom
                        if not is_oom(exc):
                            raise
                        backend.free()
                        raise OutOfMemory(
                            f"Out of memory computing the gradient of one {example.tokens}-token development "
                            "example. Lower model.max_seq_len or use a device with more memory.") from exc
                    tokens += example.tokens
                vectors[name] = (total / len(groups[name])).cpu().numpy().astype(np.float32)
            atomic_write(directory / f"checkpoint-{ci:03d}.npz", lambda f, v=vectors: np.savez(f, **v))
    cost = Cost(tokens, tokens, time.monotonic() - started)
    cache.finish("devgrads", inputs, {"cost": asdict(cost), "groups": sorted(groups)})
    return directory, cost, False


# ── alignment ───────────────────────────────────────────────────────────────

def checkpoint_weights(checkpoints: Sequence[Checkpoint], weighting: str) -> list[float]:
    """How much each checkpoint counts. `lr` follows LESS (a first-order
    expansion of training weights each step by its learning rate); weights
    are normalized so scores stay in [-1, 1] under cosine normalization."""
    raw = [c.avg_lr if weighting == "lr" else 1.0 for c in checkpoints]
    total = sum(raw)
    if total <= 0:
        return [1.0 / len(raw)] * len(raw)
    return [r / total for r in raw]


def alignment(np: Any, features: Path, devgrads: Path, weights: Sequence[float], rows: int,
              normalization: str) -> dict[str, Any]:
    """Per-capability alignment of every candidate, shard by shard.

    Memory is one shard at a time. With `cosine`, both vectors are unit
    length: gradient norms shrink with sequence length (a mean over more
    tokens), so an unnormalized dot product would mostly rank examples by
    how short they are. `none` returns raw dot products for inspection.
    """
    out: dict[str, Any] = {}
    for ci, weight in enumerate(weights):
        with np.load(devgrads / f"checkpoint-{ci:03d}.npz") as stored:
            dev = {name: stored[name].astype(np.float64) for name in stored.files}
        if normalization == "cosine":
            dev = {name: v / (np.linalg.norm(v) or 1.0) for name, v in dev.items()}
        offset = 0
        for shard in sorted((features / f"checkpoint-{ci:03d}").glob("shard-*.npy")):
            block = np.load(shard).astype(np.float64)
            if normalization == "cosine":
                norms = np.linalg.norm(block, axis=1, keepdims=True)
                block = block / np.where(norms == 0, 1.0, norms)
            for name, v in dev.items():
                column = out.setdefault(name, np.zeros(rows, dtype=np.float64))
                column[offset:offset + len(block)] += weight * (block @ v)
            offset += len(block)
        if offset != rows:
            raise OptimizeError("Cached gradient features do not cover the candidate pool; "
                                "remove the cache entry and rerun")
    return out


def feature_rows(np: Any, features: Path, checkpoint: int, indices: Sequence[int], shard_size: int) -> Any:
    """Unit-normalized features of the given pool rows at one checkpoint.

    Used for the redundancy penalty, which only ever looks at a bounded
    shortlist, so this gathers `len(indices)` rows and never the whole matrix.
    """
    wanted: dict[int, list[tuple[int, int]]] = {}
    for position, index in enumerate(indices):
        wanted.setdefault(index // shard_size, []).append((position, index % shard_size))
    out = None
    for shard, pairs in wanted.items():
        block = np.load(features / f"checkpoint-{checkpoint:03d}" / f"shard-{shard:05d}.npy", mmap_mode="r")
        if out is None:
            out = np.zeros((len(indices), block.shape[1]), dtype=np.float32)
        for position, row in pairs:
            out[position] = block[row]
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1.0, norms)


@dataclass
class Scores:
    """Estimated alignment of each pool record, target and retention apart."""
    per_capability: dict[str, Any]
    target: Any
    retention: Any | None
    target_capabilities: tuple[str, ...]
    retain_capabilities: tuple[str, ...]

    def row(self, index: int) -> dict:
        out = {"target_alignment": round(float(self.target[index]), 6)}
        if self.retention is not None:
            out["retention_alignment"] = round(float(self.retention[index]), 6)
        if len(self.per_capability) > 1:
            out["by_capability"] = {name: round(float(v[index]), 6)
                                    for name, v in sorted(self.per_capability.items())}
        return out


def combine(np: Any, per_capability: dict[str, Any], target: Sequence[str], retain: Sequence[str],
            aggregation: str) -> Scores:
    """Reduce per-capability alignment to a target score and a retention score.

    They are kept apart on purpose. The target score ranks candidates. The
    retention score is the *worst* alignment with any capability that must be
    kept, so one conflict is not averaged away; it is reported, and only
    filters candidates when the contract sets `retention_floor`.
    """
    stack = np.stack([per_capability[name] for name in target])
    target_score = stack.max(axis=0) if aggregation == "max" else stack.mean(axis=0)
    retention = np.stack([per_capability[name] for name in retain]).min(axis=0) if retain else None
    return Scores(per_capability, target_score, retention, tuple(target), tuple(retain))
