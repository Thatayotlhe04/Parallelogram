"""The one supported training path: LoRA on a local Hugging Face causal LM.

Everything that touches torch, transformers or peft lives here and is imported
lazily, so the rest of `parallelogram` (and `optimize plan` on a default
install) never pays for it.

Two choices are worth stating because scoring depends on them:

* The loss of one example is the mean negative log-likelihood over its
  supervised (assistant) tokens, and a batch loss is the mean over examples.
  Scoring uses the same per-example loss, so the gradient that is scored is
  the gradient that training applies. A token-level batch mean (the common
  default) would weight long examples more in training than in scoring.
* The supervised mask is derived from the chat template itself, not from a
  search for marker strings, and encoding refuses templates for which the
  assistant span cannot be located exactly.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import tempfile
import time
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Sequence

from .budget import Cost
from .config import Experiment, check_model_refs, digest
from .errors import ConfigError, MissingDependencyError, OptimizeError, OutOfMemory, UnsupportedError
from .splits import Record

PILOT_PARAMETER_LIMIT = 100_000_000
_MODEL_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "special_tokens_map.json", "chat_template.jinja", "vocab.json", "merges.txt",
                "tokenizer.model", "added_tokens.json")


def require_ml(allow_download: bool = False) -> SimpleNamespace:
    """Import the optional ML stack, or explain how to install it.

    Unless downloads were explicitly allowed, the Hub client is put in offline
    mode before it is imported, so no code path can reach the network.
    """
    if not allow_download:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        import numpy
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise MissingDependencyError(
            f"`parallelogram optimize` needs the optional ML extras ({exc.name} is missing). "
            "Install them with: pip install 'parallelogram[optimize]'") from exc
    transformers.logging.set_verbosity_error()
    try:
        transformers.logging.disable_progress_bar()
    except AttributeError:  # pragma: no cover - older transformers
        pass
    return SimpleNamespace(np=numpy, peft=peft, torch=torch, transformers=transformers)


def is_oom(exc: BaseException) -> bool:
    """Out-of-memory in any of the shapes torch and Python raise it."""
    if isinstance(exc, MemoryError):
        return True
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@dataclass(frozen=True)
class Encoded:
    """Token ids of one rendered conversation and which of them are supervised."""
    id: str
    ids: tuple[int, ...]
    mask: tuple[bool, ...]

    @property
    def tokens(self) -> int:
        return len(self.ids)

    @property
    def supervised(self) -> int:
        return sum(self.mask)


@dataclass
class TrainStats:
    steps: int = 0
    epochs: int = 0
    tokens: int = 0
    examples: int = 0
    wall_seconds: float = 0.0
    first_loss: float | None = None
    last_loss: float | None = None
    epoch_avg_lr: list[float] = field(default_factory=list)
    oom_backoffs: int = 0
    micro_batch: int = 0

    @property
    def cost(self) -> Cost:
        return Cost(forward_tokens=self.tokens, backward_tokens=self.tokens, wall_seconds=self.wall_seconds)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("steps", "epochs", "tokens", "examples", "wall_seconds",
                                              "first_loss", "last_loss", "oom_backoffs", "micro_batch")}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def directory_revision(path: Path) -> str:
    """Content hash of a local model or tokenizer directory.

    A local directory has no commit id, so its revision is what it contains:
    retrain or swap the weights and every cache keyed on it stops matching.
    """
    files = sorted(p for p in Path(path).iterdir()
                   if p.is_file() and (p.name in _MODEL_FILES or p.suffix in (".safetensors", ".bin")))
    if not files:
        raise OptimizeError(f"{path} does not look like a model directory (no config or weights found)")
    return digest({p.name: _sha256_file(p) for p in files})


class Backend:
    """Tokenizer, frozen base model and the operations experiments run on them."""

    def __init__(self, exp: Experiment, *, allow_download: bool = False,
                 log: Callable[[str], None] | None = None, tokenizer_only: bool = False):
        """`tokenizer_only` skips the weights: planning needs exact token
        counts, not a model in memory."""
        self.exp = exp
        self.log = log or (lambda message: None)
        self.ml = require_ml(allow_download)
        self.torch = self.ml.torch
        self.allow_download = allow_download
        self.device = self.torch.device(exp.model.device)
        self._check_device()
        self.tokenizer, tokenizer_rev = self._load_tokenizer()
        self.template_sha256 = hashlib.sha256(self.tokenizer.chat_template.encode("utf-8")).hexdigest()
        self.model, model_rev, self.parameters, self.max_positions = None, None, None, None
        if not tokenizer_only:
            self.model, model_rev = self._load_model()
            self.parameters = sum(p.numel() for p in self.model.parameters())
        self.identity = {
            "model": {"ref": exp.model.path, "revision": model_rev},
            "tokenizer": {"ref": exp.model.tokenizer or exp.model.path, "revision": tokenizer_rev},
            "chat_template_sha256": self.template_sha256,
            "loss_masking": exp.recipe.loss_masking,
            "max_seq_len": exp.model.max_seq_len,
            "dtype": exp.model.dtype,
        }
        self.versions = {"torch": self.torch.__version__, "transformers": self.ml.transformers.__version__,
                         "peft": self.ml.peft.__version__, "numpy": self.ml.np.__version__}
        self._pad = self.tokenizer.pad_token_id
        if self._pad is None:
            self._pad = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 0

    # ── loading ────────────────────────────────────────────────────────────

    def _check_device(self) -> None:
        torch = self.torch
        kind = self.device.type
        if kind == "cuda" and not torch.cuda.is_available():
            raise OptimizeError(f"model.device is {self.exp.model.device!r} but CUDA is not available here")
        if kind == "mps" and not torch.backends.mps.is_available():
            raise OptimizeError("model.device is 'mps' but MPS is not available here")

    def _source(self, ref: str, revision: str | None, what: str) -> tuple[dict, str | None]:
        """Loader arguments and the pinned revision of a local dir or Hub id."""
        if Path(ref).is_dir():
            return {"local_files_only": True}, directory_revision(Path(ref))
        check_model_refs(self.exp)
        return {"revision": revision, "local_files_only": not self.allow_download}, revision

    def _load_tokenizer(self):
        spec = self.exp.model
        ref = spec.tokenizer or spec.path
        revision = spec.tokenizer_revision if spec.tokenizer else spec.revision
        kwargs, rev = self._source(ref, revision, "Tokenizer")
        try:
            tokenizer = self.ml.transformers.AutoTokenizer.from_pretrained(ref, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the Hub client raises many types
            raise OptimizeError(self._unavailable("tokenizer", ref, exc)) from exc
        if not getattr(tokenizer, "is_fast", False):
            raise UnsupportedError("The tokenizer has no fast (Rust) implementation; character offsets are "
                                   "needed to locate assistant tokens exactly")
        if spec.chat_template != "tokenizer":
            try:
                tokenizer.chat_template = Path(spec.chat_template).read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                raise OptimizeError(f"Could not read model.chat_template {spec.chat_template}: {exc}") from exc
        if not tokenizer.chat_template:
            raise ConfigError("The tokenizer has no chat template. Point model.chat_template at a Jinja file; "
                              "guessing a prompt format would train the wrong thing.")
        return tokenizer, rev

    def _load_model(self):
        spec = self.exp.model
        torch = self.torch
        kwargs, rev = self._source(spec.path, spec.revision, "Model")
        try:
            # The keyword was renamed in transformers 5.
            major = int(self.ml.transformers.__version__.split(".")[0])
            dtype = {"dtype" if major >= 5 else "torch_dtype": getattr(torch, spec.dtype)}
            model = self.ml.transformers.AutoModelForCausalLM.from_pretrained(spec.path, **dtype, **kwargs)
        except Exception as exc:  # noqa: BLE001
            raise OptimizeError(self._unavailable("model", spec.path, exc)) from exc
        limit = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
        if limit is not None and spec.max_seq_len > limit:
            raise ConfigError(f"model.max_seq_len is {spec.max_seq_len} but the model accepts at most {limit} positions")
        self.max_positions = limit
        model.to(self.device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        if rev == spec.revision and getattr(model.config, "_commit_hash", None):
            rev = model.config._commit_hash
        return model, rev

    def _unavailable(self, what: str, ref: str, exc: Exception) -> str:
        hint = ("" if self.allow_download else
                " Nothing is downloaded unless you opt in with --allow-download.")
        return f"Could not load the {what} {ref!r} ({type(exc).__name__}).{hint}"

    # ── encoding and the loss mask ─────────────────────────────────────────

    def _render(self, messages: Sequence[dict], generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(list(messages), tokenize=False,
                                                  add_generation_prompt=generation_prompt)

    def assistant_spans(self, messages: Sequence[dict]) -> tuple[str, list[tuple[int, int]]]:
        """Rendered text and the character span of each supervised assistant turn.

        The span of turn `i` runs from the end of "everything before it plus
        the generation prompt" to the end of "everything up to and including
        it", so it covers the answer and the end-of-turn marker the model must
        learn to emit. Both renders must be prefixes of the full conversation;
        a template for which they are not (one that rewrites earlier turns) is
        refused, because any mask derived from it would be a guess.
        """
        full = self._render(messages, False)
        turns = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
        if self.exp.recipe.loss_masking == "final_assistant_only":
            turns = turns[-1:]
        spans = []
        for i in turns:
            before = self._render(messages[:i], True)
            upto = self._render(messages[:i + 1], False)
            if not (upto.startswith(before) and full.startswith(upto) and len(upto) > len(before)):
                raise UnsupportedError(
                    "The chat template does not render conversations incrementally, so assistant tokens "
                    "cannot be located exactly for loss masking. Use a template whose output for a "
                    "conversation starts with its output for the conversation's first turns.")
            spans.append((len(before), len(upto)))
        return full, spans

    def encode(self, record: Record) -> Encoded:
        text, spans = self.assistant_spans(record.messages)
        enc = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc["input_ids"]
        mask = [any(start < hi and end > lo for lo, hi in spans) for start, end in enc["offset_mapping"]]
        return Encoded(id=record.id, ids=tuple(ids), mask=tuple(mask))

    def count_tokens(self, record: Record) -> int:
        return len(self.tokenizer(self._render(record.messages, False), add_special_tokens=False)["input_ids"])

    # ── adapters ───────────────────────────────────────────────────────────

    def _lora_config(self):
        lora = self.exp.recipe.lora
        targets = list(lora.target_modules)
        matched = [m for name, m in self.model.named_modules() if any(name.split(".")[-1] == t for t in targets)]
        if not matched:
            raise ConfigError(f"recipe.lora.target_modules {targets} match no module of the model")
        conv1d = any(type(m).__name__ == "Conv1D" for m in matched)
        return self.ml.peft.LoraConfig(r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout,
                                       target_modules=targets, bias="none", task_type="CAUSAL_LM",
                                       fan_in_fan_out=conv1d)

    @contextmanager
    def adapter(self, seed: int) -> Iterator[Any]:
        """A fresh LoRA adapter on the base model, removed again on exit.

        Every trial starts here, so every trial starts from the same frozen
        base weights and differs only in its seed and its data.
        """
        self.torch.manual_seed(seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # third-party deprecation chatter
            model = self.ml.peft.get_peft_model(self.model, self._lora_config())
        # New LoRA modules start in training mode. Everything except `train`
        # (gradient scoring, evaluation) must run with dropout off, or the
        # same example would score differently each time it is visited.
        model.eval()
        try:
            yield model
        finally:
            self.model = model.unload()
            self.model.eval()
            self.free()

    @contextmanager
    def saved_adapter(self, path: Path) -> Iterator[Any]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = self.ml.peft.PeftModel.from_pretrained(self.model, str(path))
        model.eval()
        try:
            yield model
        finally:
            self.model = model.unload()
            self.model.eval()
            self.free()

    def save_adapter(self, model: Any, path: Path) -> None:
        """Write the adapter to a temporary sibling, then move it into place."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=f".{path.name}.", dir=str(path.parent)))
        try:
            model.save_pretrained(str(tmp))
            if path.exists():
                shutil.rmtree(path)
            os.replace(tmp, path)
        except BaseException:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    def trainable(self, model: Any) -> list[tuple[str, Any]]:
        """`(name, parameter)` of the adapter weights, in a stable order."""
        return [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    def free(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.empty_cache()

    # ── loss ───────────────────────────────────────────────────────────────

    def example_losses(self, model: Any, batch: Sequence[Encoded]) -> Any:
        """Mean supervised-token NLL of each example, as a tensor of shape [B]."""
        torch = self.torch
        width = max(e.tokens for e in batch)
        ids = torch.full((len(batch), width), self._pad, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        supervised = torch.zeros((len(batch), width), dtype=torch.bool)
        for row, e in enumerate(batch):
            ids[row, :e.tokens] = torch.tensor(e.ids, dtype=torch.long)
            attention[row, :e.tokens] = 1
            supervised[row, :e.tokens] = torch.tensor(e.mask, dtype=torch.bool)
        ids, attention, supervised = ids.to(self.device), attention.to(self.device), supervised.to(self.device)
        logits = model(input_ids=ids, attention_mask=attention).logits
        # Position t predicts token t+1; the first token is never a target.
        logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
        targets = ids[:, 1:]
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        keep = supervised[:, 1:].to(nll.dtype)
        return (nll * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)

    def _accumulate(self, model: Any, batch: Sequence[Encoded], micro: int) -> float:
        """Backpropagate the batch-mean loss in chunks of `micro` examples.

        Summing chunk losses divided by the full batch size makes the
        accumulated gradient identical to the one-pass gradient, so backing
        off under memory pressure changes speed, not the update.
        """
        total = 0.0
        for start in range(0, len(batch), micro):
            per_example = self.example_losses(model, batch[start:start + micro])
            (per_example.sum() / len(batch)).backward()
            total += float(per_example.sum().detach())
        return total / len(batch)

    def example_gradient(self, model: Any, example: Encoded) -> Any:
        """Flat gradient of one example's loss with respect to the adapter."""
        loss = self.example_losses(model, [example])[0]
        grads = self.torch.autograd.grad(loss, [p for _, p in self.trainable(model)])
        return self.torch.cat([g.reshape(-1) for g in grads]).detach()

    # ── training ───────────────────────────────────────────────────────────

    def lr_at(self, step: int, total_steps: int) -> float:
        """Learning-rate multiplier at optimizer step `step` (0-based)."""
        recipe = self.exp.recipe
        warm = int(recipe.warmup_ratio * total_steps)
        if step < warm:
            return (step + 1) / warm
        if recipe.schedule == "constant":
            return 1.0
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))

    def train(self, model: Any, examples: Sequence[Encoded], *, seed: int, epochs: int | None = None,
              max_steps: int | None = None,
              on_epoch: Callable[[int, Any, float, int], None] | None = None,
              stats: TrainStats | None = None) -> TrainStats:
        """Run the recipe on `examples`. Deterministic given the seed.

        `on_epoch(epoch_index, optimizer, average_lr, step)` fires after each
        epoch; the warmup uses it to save fixed checkpoints. A caller that
        passes its own `stats` can still read what was spent if training is
        interrupted, so that interrupted work is charged rather than lost.
        """
        torch = self.torch
        recipe = self.exp.recipe
        if not examples:
            raise OptimizeError("Nothing to train on: the selected subset is empty")
        epochs = recipe.epochs if epochs is None else epochs
        per_epoch = math.ceil(len(examples) / recipe.batch_size)
        total_steps = epochs * per_epoch
        if max_steps is not None:
            total_steps = min(total_steps, max_steps)
        params = [p for _, p in self.trainable(model)]
        optimizer = torch.optim.AdamW(params, lr=recipe.learning_rate, betas=(recipe.beta1, recipe.beta2),
                                      eps=recipe.eps, weight_decay=recipe.weight_decay)
        stats = stats if stats is not None else TrainStats()
        stats.micro_batch = recipe.batch_size
        torch.manual_seed(seed)
        model.train()
        started = time.monotonic()
        step = 0
        try:
            for epoch in range(epochs):
                if step >= total_steps:
                    break
                order = list(range(len(examples)))
                random.Random(f"{seed}:{epoch}").shuffle(order)
                lrs = []
                for start in range(0, len(order), recipe.batch_size):
                    if step >= total_steps:
                        break
                    batch = [examples[i] for i in order[start:start + recipe.batch_size]]
                    lr = recipe.learning_rate * self.lr_at(step, total_steps)
                    for group in optimizer.param_groups:
                        group["lr"] = lr
                    while True:
                        try:
                            optimizer.zero_grad(set_to_none=True)
                            loss = self._accumulate(model, batch, min(stats.micro_batch, len(batch)))
                            break
                        except Exception as exc:  # noqa: BLE001 - filtered by is_oom
                            if not is_oom(exc):
                                raise
                            optimizer.zero_grad(set_to_none=True)
                            self.free()
                            if stats.micro_batch <= 1:
                                raise OutOfMemory(
                                    "Out of memory training a single example. Lower model.max_seq_len, "
                                    "use a smaller model or dtype, or pick a device with more memory.") from exc
                            stats.micro_batch = max(1, stats.micro_batch // 2)
                            stats.oom_backoffs += 1
                            self.log(f"out of memory: retrying with {stats.micro_batch} example(s) per forward pass")
                    if recipe.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(params, recipe.max_grad_norm)
                    optimizer.step()
                    step += 1
                    lrs.append(lr)
                    stats.tokens += sum(e.tokens for e in batch)
                    stats.examples += len(batch)
                    if stats.first_loss is None:
                        stats.first_loss = loss
                    stats.last_loss = loss
                stats.epochs += 1
                stats.epoch_avg_lr.append(sum(lrs) / len(lrs))
                if on_epoch is not None:
                    on_epoch(epoch, optimizer, stats.epoch_avg_lr[-1], step)
        finally:
            model.eval()
            stats.steps = step
            stats.wall_seconds = time.monotonic() - started
        return stats

    # ── evaluation ─────────────────────────────────────────────────────────

    def losses(self, model: Any, examples: Sequence[Encoded]) -> tuple[list[float], Cost]:
        """Per-example loss without gradients, halving the batch on OOM."""
        torch = self.torch
        out: list[float] = []
        size = self.exp.evaluation.batch_size
        started = time.monotonic()
        model.eval()
        start = 0
        with torch.no_grad():
            while start < len(examples):
                batch = examples[start:start + size]
                try:
                    out.extend(float(v) for v in self.example_losses(model, batch))
                except Exception as exc:  # noqa: BLE001 - filtered by is_oom
                    if not is_oom(exc):
                        raise
                    self.free()
                    if size <= 1:
                        raise OutOfMemory("Out of memory evaluating a single example. Lower "
                                          "model.max_seq_len or use a device with more memory.") from exc
                    size = max(1, size // 2)
                    self.log(f"out of memory: evaluating {size} example(s) at a time")
                    continue
                start += len(batch)
        return out, Cost(forward_tokens=sum(e.tokens for e in examples), wall_seconds=time.monotonic() - started)

    def exact_match(self, model: Any, records: Sequence[Record]) -> tuple[list[float], Cost]:
        """Greedy-decode each final assistant turn and compare it to the reference.

        This is a task metric, measured by generation, and deliberately has no
        connection to the gradient scores: a subset can align well with the
        development loss and still not move this number.
        """
        torch = self.torch
        scores: list[float] = []
        forward = 0
        started = time.monotonic()
        stops = {self.tokenizer.eos_token_id} - {None}
        model.eval()
        for record in records:
            messages = list(record.messages)
            reference = messages[-1]["content"]
            prompt = self._render(messages[:-1], True)
            full = self._render(messages, False)
            tail = full[len(prompt):]
            # What the template writes after an answer (the end-of-turn marker).
            suffix = tail[len(reference):] if tail.startswith(reference) else ""
            stop_ids = set(stops)
            if suffix:
                first = self.tokenizer(suffix, add_special_tokens=False)["input_ids"][:1]
                stop_ids.update(first)
            ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            room = self.exp.evaluation.max_new_tokens
            if self.max_positions is not None:
                room = min(room, self.max_positions - len(ids))
            if room <= 0:
                scores.append(0.0)
                continue
            inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
            try:
                with torch.no_grad():
                    out = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         max_new_tokens=room, do_sample=False, num_beams=1,
                                         pad_token_id=self._pad, eos_token_id=sorted(stop_ids) or None)
            except Exception as exc:  # noqa: BLE001 - filtered by is_oom
                if not is_oom(exc):
                    raise
                self.free()
                raise OutOfMemory("Out of memory generating an answer for one evaluation example. Lower "
                                  "evaluation.max_new_tokens or model.max_seq_len, or use a device with "
                                  "more memory.") from exc
            new = out[0, len(ids):].tolist()
            forward += len(ids) + len(new)
            text = self.tokenizer.decode(new, skip_special_tokens=False)
            if suffix and suffix in text:
                text = text[:text.index(suffix)]
            else:
                text = self.tokenizer.decode(new, skip_special_tokens=True)
            scores.append(1.0 if " ".join(text.split()) == " ".join(reference.split()) else 0.0)
        return scores, Cost(forward_tokens=forward, wall_seconds=time.monotonic() - started)
