# Parallelogram Optimize

`parallelogram check` answers "will this data break training?".
`parallelogram optimize` answers a different question: given more candidate
data than you want to train on, which subset should you train on, and did
choosing it that way actually help once the cost of choosing is counted?

It recommends a subset using gradient-based scores, then trains the
recommendation and several baselines for real and reports what was measured.
The default conclusion is "insufficient evidence". The tool is built so that a
conclusion in favor of its own selector is the hardest one to reach.

- [Scope](#scope)
- [Quick start](#quick-start)
- [The experiment contract](#the-experiment-contract)
- [Splits and leakage](#splits-and-leakage)
- [Arms and what each comparison answers](#arms-and-what-each-comparison-answers)
- [Compute accounting](#compute-accounting)
- [Scoring: LESS-inspired gradient alignment](#scoring-less-inspired-gradient-alignment)
- [Selection](#selection)
- [The loop and the verdict](#the-loop-and-the-verdict)
- [Reports and export](#reports-and-export)
- [Caching, resume and memory](#caching-resume-and-memory)
- [Python API and MCP](#python-api-and-mcp)
- [Design decisions](#design-decisions)
- [What was actually tested](#what-was-actually-tested)
- [Remaining empirical experiments](#remaining-empirical-experiments)

## Scope

One training path is implemented end to end:

- supervised instruction tuning (`objective: sft-lora-causal-lm`)
- of a local Hugging Face causal language model
- with LoRA, AdamW, and a loss on assistant tokens only
- on plain `system` / `user` / `assistant` chat records, read through the
  existing parsers (OpenAI chat, ShareGPT, prompt/completion, Alpaca).

Rejected explicitly, with an error that names the reason: preference (DPO)
data, tool-calling records, non-text content, records that do not end with
an assistant turn, other objectives, other optimizers, other loss masks,
slow (non-Rust) tokenizers, and chat templates that do not render
conversations incrementally.

The ML dependencies are optional:

```bash
pip install parallelogram               # linter only: typer + rich
pip install 'parallelogram[optimize]'   # adds torch, transformers, peft, numpy, pyyaml
```

`check`, `report`, `rules` and `mcp` never import them. Datasets are processed
locally. A model is downloaded only with `--allow-download`; without it the Hub
client is put in offline mode before it is imported.

## Quick start

A complete experiment that trains for real on a laptop CPU in about two
minutes. It builds its own 113k-parameter model; nothing is downloaded.

```bash
cd cli
pip install -e '.[optimize]'
python examples/optimize/make_tiny_experiment.py      # data + tiny model, ~40 s

parallelogram optimize plan   examples/optimize/tiny/experiment.yaml
parallelogram optimize run    examples/optimize/tiny/experiment.yaml
parallelogram optimize report examples/optimize/tiny/run
parallelogram optimize export examples/optimize/tiny/run --output selected.jsonl
```

| command | does | trains? |
|---|---|---|
| `optimize plan EXPERIMENT` | Validates the contract and the splits, counts tokens, shows the experiment matrix, estimated compute per stage against the budget, what it would drop to fit, and the files a run will write. | no |
| `optimize run EXPERIMENT` | Scores, selects, trains every arm, compares on development data, freezes a candidate, evaluates once on test data. Resumes when rerun. `--no-final` stops before the freeze. | yes |
| `optimize report RUN_DIR` | Rebuilds the report from the run's artifacts. Needs no ML extras. `--json`, `--out FILE`. | no |
| `optimize export RUN_DIR --output FILE` | Copies the selected records byte for byte and writes `FILE.manifest.json`. `--trial` exports another arm. | no |

Exit codes follow the rest of the CLI: `0` the command did what was asked,
`2` invalid contract or arguments (including a plan that does not fit its
budget, and splits that fail validation), `3` operational failure (missing
extras, model not available locally, out of memory, unreadable files). A
finished run exits `0` whatever it concluded; the conclusion is in the report.

## The experiment contract

A `.yaml`, `.yml` or `.json` file, `schema_version: 1`. YAML needs PyYAML,
which comes with the extras; a JSON contract works on the default install,
where `plan` (with estimated token counts), `report` and `export` still run.
Unknown keys are errors. Paths are relative to the contract file. The annotated example is
[`examples/optimize/tiny/experiment.yaml`](../examples/optimize/tiny/experiment.yaml).

| section | what it fixes |
|---|---|
| `data` | `candidates`, `dev`, `test` files (three different files). `id_field` for immutable record ids (default: hash of the source line). `group_field`, a related-document key. `capability_field`, the tag that says which capability an evaluation example tests. `dev_selection_fraction`. `leakage` policy. |
| `model` | `path` (a local directory, or a Hub id with a pinned `revision`), optional separate `tokenizer`, `chat_template` (`tokenizer` or a Jinja file), `max_seq_len`, `on_overlength` (`error` or `exclude`), explicit `device`, `dtype`. |
| `recipe` | LoRA settings, AdamW hyperparameters, `epochs`, `batch_size`, schedule, `loss_masking` (`assistant_only` or `final_assistant_only`), `seeds`. |
| `evaluation` | `metrics` (`loss`, `exact_match`), `primary_metric`, `target_capabilities`, `retain_capabilities`, `retention_tolerance`, bootstrap size. |
| `selection` | `token_budget` for the training subset, `policies`, `coverage` minimums, `less` scoring settings. |
| `search` | Extra subset sizes and scoring variants to compare on development data, capped by `max_trials`. |
| `baselines` | `base_model`, `full_data`, `random_matched_compute`. |
| `budget` | `total_compute` for the whole experiment, overhead and evaluation included. |
| `output` | `run_dir`, `cache_dir`. |

Record ids, dataset hashes, model and tokenizer revisions (for a local
directory, a hash of its files), the chat template's hash and the contract's
fingerprint are written into every run and into the export manifest.

## Splits and leakage

Each split must pass the same rules as `parallelogram check` before anything
else happens. Then:

| finding | certainty | default action |
|---|---|---|
| a candidate is the same normalized record as a dev or test example | exact | candidate excluded |
| a candidate has the same prompt as a dev or test example | exact | candidate excluded |
| a candidate shares a `group_field` value with a dev or test example | exact (related document) | candidate excluded |
| a candidate is lexically close to a dev or test example (shingle Jaccard at or above `near_threshold`) | **uncertain** | reported only |
| dev and test share an exact record, prompt or document | exact | the run is refused |

Exact findings are facts about the files. Near-duplicate findings are a
similarity estimate: they can be coincidental, and paraphrased leakage is not
detected at all. That is why they are reported for a person to judge and do
not remove data unless `on_near: exclude` is set.

Test isolation is structural:

- The leakage guard sees the test split only as hashes and shingle hashes. It
  hands selection a set of candidate ids to leave out, nothing else.
- Test examples come out of a vault that refuses to open without a freeze
  record whose seal matches the frozen candidate and recipe.
- Once a candidate is frozen, the run directory accepts no new trials, so
  test results cannot steer another round in the same run. Each final
  evaluation of a given test file is counted in the cache directory, and the
  report says when a test split has been used more than once.

Development examples are divided in two by `dev_selection_fraction`: one half
provides the gradients that scoring aligns candidates with, the other half
compares policies. Comparing on the examples the selector was optimized for
would favor it for the wrong reason.

## Arms and what each comparison answers

| arm | label | question |
|---|---|---|
| base model, no training | reference | Does fine-tuning help at all? |
| `random@B` | matched tokens | Uniform random selection of `B` tokens. A fresh draw per seed. |
| `diversity@B` | matched tokens | Greedy k-center on model-free lexical features. |
| `less@B` | matched tokens | Gradient-alignment selection. |
| `random+overhead` | matched compute | Random selection given the scoring overhead as extra training tokens. Exists only when the pool can absorb them (see below). |
| `full` | full data | Every candidate. Runs when the budget permits. |

Matched tokens isolates *which* records were chosen and ignores what choosing
cost. Matched compute asks whether scoring was a better use of compute than
simply training on more data. Full data shows how much quality a subset gives
up. These are different questions and the report never merges them.

Every arm trains from the same frozen base weights with the same recipe.

## Compute accounting

One ledger, one unit: the forward-token-equivalent (FTE). A token through a
forward pass costs 1; a backward pass over it costs `backward_multiplier`
more (default 2). It is an accounting model, not measured FLOPs. Wall-clock
seconds are recorded beside it.

Charged stages: `warmup`, `scoring` (candidate and development gradients),
`selection`, `pilot_training`, `evaluation`, `final_evaluation`. The first
three are reported together as optimization overhead. A stage served from the
cache is charged at its original cost and marked as cached.

The plan fits the matrix into `budget.total_compute` in priority order:
essentials first (base evaluation, scoring, one seed per policy, the final
evaluation), then more seeds, the matched-compute arm, search trials, and
full-data training. What does not fit is listed, not silently skipped.

A rule of thumb that falls out of the accounting: scoring costs about
`checkpoints / epochs` of one full-data training run, before any subset is
trained. Selection can only win the matched-compute comparison if the quality
it buys is worth more than that.

When the overhead is worth more training tokens than the pool holds (roughly
when `checkpoints / epochs` is at least `1 - token_budget / pool_tokens`, as
with four warmup epochs and a three-epoch recipe), no random arm can spend
the same compute. Capped at the pool it
would be full-data training under a "matched" label while costing less than
the model-aware arm. In that case the arm is not created, the plan lists it
under what is not planned, and the conclusion cannot be `supported`: the
candidate spent more in total than training on everything.

## Scoring: LESS-inspired gradient alignment

Based on Xia et al., *LESS: Selecting Influential Data for Targeted
Instruction Tuning* (ICML 2024). For checkpoints `i = 1..C` from a short LoRA
warmup:

```
score(z) = sum_i  w_i * cos( P * Gamma(z; theta_i),  P * mean_dev grad L(dev; theta_i) )
```

- `Gamma(z; theta_i)` is the step Adam would take for example `z` alone,
  given the optimizer's moment estimates at checkpoint `i`.
- `P` is a seeded random projection.
- `w_i` is the checkpoint's share of the learning rate.
- The development gradient is a plain gradient, averaged per capability.

How the brief's four concerns are handled:

| concern | handling |
|---|---|
| optimizer behavior | `optimizer_aware: true` scores the Adam update direction, with bias correction, from saved moment estimates. Verified to equal the step `torch.optim.AdamW` takes. |
| sequence length | The per-example loss is a mean over supervised tokens, and `normalization: cosine` removes gradient magnitude, which otherwise shrinks with length and would rank short records first. Training uses the same per-example loss. |
| checkpoint aggregation | Fixed checkpoints at the end of each warmup epoch; `checkpoint_weighting: lr` (default) or `uniform`. |
| normalization | `cosine` (default) or `none` for raw inner products. |

Target and retention are separate columns. The target score is the best (or
mean) alignment across target capabilities and ranks candidates. The
retention score is the worst alignment with any capability that must be
kept. It is reported per record and filters candidates only if
`retention_floor` is set.

What a score is not: it is a first-order estimate about a differentiable
loss. It is not answer accuracy or any other task metric (those are measured
by generation, separately). It is not a probability. A negative score is an
estimated conflict signal, not proof that a record is wrong or that deleting
it will improve training.

### Deviations from the reference method

| reference (LESS) | here | why |
|---|---|---|
| Dense Rademacher projection via TRAK's CUDA kernel | CountSketch: each gradient coordinate goes to one seeded bucket with a seeded sign | No GPU kernel dependency, one pass, memory linear in parameters. Also a Johnson-Lindenstrauss transform; inner products are preserved in expectation. `identity` is available for small models. |
| `Gamma` without Adam bias correction, epsilon inside the square root | Bias correction included, epsilon outside | Matches the optimizer's actual step, which the tests check exactly. |
| Checkpoint weights are raw average learning rates | Normalized to sum to 1 | Same ranking; scores stay in [-1, 1]. |
| Training loss is the framework's token-level mean over a batch | Training loss is the mean over examples of each example's token-mean, the same per-example loss that is scored | The scored gradient is the gradient training applies; a token-level batch mean weights long examples more in training than in scoring. |
| Warmup on 5% for 4 epochs, top 5% selected | Fraction, epochs and step cap are contract settings; selection is by token budget with coverage minimums | The contract, not the method, owns the budget. |
| Gradient clipping not modeled | Not modeled either | A per-step scalar on the gradient; noted as an approximation under Adam. |
| Decoupled weight decay not modeled | Not modeled either | Identical for every example. |
| Selection by score alone | Optional redundancy penalty (`diversity_lambda`) and coverage minimums | Requested by the brief; off by default so the default is closest to the reference. |

## Selection

Every policy runs the same two phases, so policies differ only in ordering:

1. **Coverage.** For each `coverage` minimum (records or tokens of a
   language, domain or skill), take that group's best candidates in the
   policy's own order until the minimum is met. If it cannot be met inside
   the budget, the arm is reported as infeasible.
2. **Fill.** Spend the remaining token budget in the policy's order, skipping
   records that no longer fit.

No pairwise matrix is stored. K-center keeps one distance per candidate and
refreshes it with one batched product per pick. The redundancy penalty keeps
one similarity per shortlisted candidate; the shortlist is the top
`pool_factor x expected picks` by utility. Gradient features live in sharded
`.npy` files and are read a shard at a time.

## The loop and the verdict

1. Guard the splits, size the plan, refuse if the essentials do not fit.
2. Evaluate the base model. Run warmup and scoring. Charge both.
3. For each arm and seed: select, train, evaluate on the comparison half of
   the development split.
4. Pick the candidate: the arm within `selection.token_budget` with the best
   mean primary metric that stays within `retention_tolerance` on retained
   capabilities and beats the base model. Freeze it. Only results that
   belong to the current contract count (`current.json`); results left in
   the directory by earlier settings are ignored.
5. Open the test vault. Evaluate every trained arm once.

Two kinds of noise are reported separately: variation across runs (spread of
per-seed means) and evaluation uncertainty (bootstrap over examples). A
comparison is `supported` only if the difference clears both at the 95%
level with at least three seeds per arm; `contradicted` if the baseline wins
by the same standard; otherwise `insufficient_evidence`.

The headline conclusion is about the stated goal, quality for total compute:

| conclusion | condition |
|---|---|
| `supported` | The candidate is a `less` arm and beats both the strongest matched-token baseline and the matched-compute baseline beyond noise. |
| `not_supported` | A baseline beats the model-aware arm beyond noise. |
| `insufficient_evidence` | Everything else: differences within noise, fewer than three seeds, a missing or impossible baseline, no arm within retention tolerance. |

Runs on models under 100 million parameters, and any run with
`scale: pilot`, are labeled pilots. A pilot's conclusion says what was
measured on that model. It does not establish an improvement for larger
models.

## Reports and export

`optimize report` keeps six sections apart:

1. **Estimated training-value signals.** Score distribution. Heuristics.
2. **Measured development results.** Per arm: mean, across-seed spread,
   evaluation interval, per capability. These chose the candidate, so they
   are optimistic for it.
3. **Final held-out results.** The same table on the test split, measured
   once after the freeze.
4. **Compute consumed.** By stage, with overhead and cached work visible.
5. **Coverage changes.** Token share by field value, pool versus selected.
6. **Limitations and experiments not performed.**

`optimize export` copies the selected source lines unchanged, in source
order: key order, whitespace, escapes, extra fields and line endings survive.
It refuses to write over an input dataset or an existing file (without
`--overwrite`), and refuses if the candidates file changed since the run. The
manifest records, per record, the source id and line, its content hash, the
selection reason (`coverage:<field>=<value>` or `policy:<name>`), rank and
scores, plus contract and model fingerprints, the run's conclusion and
library versions.

Logs and JSON artifacts carry ids, counts, hashes and metrics. They do not
carry message text.

## Caching, resume and memory

- **Content-addressed caches.** Warmup checkpoints, gradient features and
  development gradients are stored under `cache_dir` in directories named by
  the hash of their inputs: every record's id *and* content hash, model and
  tokenizer revisions, chat template, loss masking, `max_seq_len`, dtype,
  LoRA and optimizer settings, warmup settings, projection settings, seeds,
  and library versions. Change any of them and the old entry is simply not
  found. Training seeds do not enter scoring and do not invalidate it.
- **Run directories are bound to bytes.** A run directory records the hashes
  of the three dataset files and the model identity. Edit a dataset under
  the same path and the directory refuses to continue; use a new one.
- **Resume.** Every artifact is written to a temporary sibling and renamed.
  Feature shards carry a completion marker; an interrupted scoring pass
  resumes at the first shard without one. A trial seed is the unit of resume
  for training: a finished one is never retrained, an interrupted one starts
  over. Finished work is charged once. Work that was interrupted and redone
  was done twice and is charged twice: cut-short training passes, warmups
  and gradient shards stay on the ledger. The exception is an evaluation
  pass interrupted midway, which is not charged.
- **Budget.** Before each trial, and before re-scoring under changed
  settings, the run checks that the cost fits what is left and skips or
  refuses otherwise. It does not overdraw.
- **Out of memory.** Training halves the number of examples per forward pass
  and accumulates gradients, which leaves the update unchanged. Evaluation
  halves its batch. If a single example does not fit, the run stops with an
  explanation and everything finished so far stays cached.
- **Device.** Named explicitly in the contract (`cpu`, `cuda`, `cuda:N`,
  `mps`). There is no automatic choice.

## Python API and MCP

```python
from parallelogram.optimize import load_experiment, plan, run, report, export

experiment = load_experiment("experiment.yaml")
plan(experiment)                         # dict, same as `optimize plan --json`
result = run(experiment, final=True)     # dict, same as `optimize run --json`
report("runs/my-run")
manifest = export("runs/my-run", "selected.jsonl")
```

Invalid contracts raise `ConfigError` (a `ValueError`; `UnsupportedError`,
`LeakageError` and `SelectionInfeasible` are subclasses). Operational failures
raise `OptimizeError` (a `ParallelogramError`; `MissingDependencyError` and
`OutOfMemory` are subclasses). Whether downloads are allowed is decided once
per process.

MCP: `parallelogram mcp --optimize` adds `optimize_plan`, `optimize_report`
and `optimize_export` to the server. Without the flag the tool list is
exactly what it was. There is no `optimize_run` tool: a run takes minutes to
hours and the server answers one request at a time. `optimize_export` is
confined to the server root like `fix_dataset`.

## Design decisions

- **A sub-package, not new rules.** `optimize` reuses the parsers, the
  validation runner, `record_hash` / `prompt_hash` and atomic writes. Rules,
  the fixer and the linter's JSON contract are untouched.
- **Exit codes reused, not extended.** See the table above.
- **MCP tools are opt-in** so the default `tools/list` response stays
  byte-identical for existing clients.
- **No backend abstraction.** There is one backend, so there is one class.
- **Own training loop** instead of `Trainer`: about a hundred lines, the loss
  is the one that is scored, and memory back-off is exact.
- **The candidate must fit the selection budget.** Otherwise "best score"
  just picks the arm with the most data.
- **All trained arms are evaluated on test.** The candidate is fixed before
  the vault opens; evaluating the baselines too is what makes a held-out
  comparison possible.
- **Version and changelog are not touched**, per `AGENTS.md`.

## What was actually tested

Environment: Linux, 2 CPU cores, no GPU, no access to the Hugging Face Hub.
Python 3.13, torch 2.15, transformers 5.23, peft 0.21, numpy 2.5. The lower
bounds in the `optimize` extra are declared, not tested.

**Test suite.** 716 tests pass with the extras installed (585 existing, 131
new). On a clean default install without them, 644 pass and 4 are skipped
(the ML test module and three tests that need PyYAML). New tests cover:

- split isolation: the vault refuses without a valid freeze; the test split
  is opened once, after the freeze and after all selection; two runs that
  differ only in their test examples make identical selections and freeze the
  same candidate
- leakage: exact, related-document and near-duplicate findings told apart;
  policies; dev/test overlap refused; no text in findings
- loss masking against a direct cross-entropy calculation; padding
  invariance; a non-incremental template refused
- gradient scoring against direct calculations on the tiny model: per-example
  gradients versus finite differences; the Adam direction versus the real
  optimizer step; alignment versus the loss change of a real SGD step; the
  cached, sharded pipeline versus hand computation from the checkpoints;
  projection determinism, linearity and unbiasedness
- token budget and coverage for every policy; infeasible minimums
- cache invalidation for each keyed input, including edited records that keep
  their ids; cache reuse; interrupted scoring and interrupted runs resuming
  to the same results, with interrupted work charged
- export fidelity, refusals, manifest contents
- existing CLI behavior, default MCP tool list, lightweight import

A mutation pass (breaking the mask, the Adam formula, checkpoint weighting,
gradient accumulation, freeze ordering, budget checks, export, shard resume,
seed threshold, and making selection depend on the test file) was caught by
the suite in every case that changes behavior.

**Independent review.** A separate reviewer that had not seen the code being
written was asked to find defects. It confirmed nine, none of which the
suite caught at the time. All are fixed and each has a regression test:
results from an earlier contract could be frozen; caches and results were
keyed by record id and token count rather than content; the matched-compute
arm was mislabeled when capped at the pool size; an untagged test example
crashed the report after the test split had been opened; LoRA dropout was
active during gradient scoring; re-scoring could overdraw the budget;
interrupted training was not charged; an empty candidate set crashed instead
of skipping the arm; and a test-split statement called the frozen candidate
"the best arm on test data". The review also examined, and found no defect
in, the loss mask, the scoring math, the selection loop (400 fuzzed cases),
the statistics, plan fitting, test isolation, export and MCP path handling.

**Tiny CPU experiment.** `examples/optimize/tiny`, run from an empty cache:
600 synthetic candidates (12,120 tokens), a 113,344-parameter model, 9 arms
times 3 seeds, 82 to 107 s of compute across two runs with identical results,
2.48M of 3.0M FTE. Test-split loss on the
target capability (lower is better):

| arm | comparison | mean | across-seed std |
|---|---|---|---|
| base model | reference | 9.628 | n/a |
| `random@3000` | matched tokens | 3.138 | 0.062 |
| `diversity@3000` | matched tokens | 3.988 | 0.239 |
| **`less@3000`** (frozen candidate) | matched tokens | 3.143 | 0.072 |
| `random+overhead` (10,085 tokens) | matched compute | 2.809 | 0.028 |
| `full` (12,120 tokens) | full data | 2.743 | 0.007 |

Reported conclusion: **not supported**. At matched tokens the gradient
selector and random selection are indistinguishable (difference 0.005,
within noise). Once the scoring overhead is counted, random selection with
the same total compute is better beyond noise. Exact match stayed at 0 for
every arm: the tiny model does not learn reversal well enough to decode it.

Two things from this run are worth knowing, and neither is a claim about real
models:

- The selector concentrated on what looks like the target: 35% of its tokens
  went to reversal records, against 22% in the pool. It also spent 18% on
  reversal prompts with deliberately wrong answers, against 11% in the pool.
  Gradient alignment does not distinguish a correct target example from a
  mislabeled one that resembles it.
- Scoring cost 2.4 times what training the selected subset for one seed
  cost. On a pool this small relative to the recipe, selection cannot pay
  for itself.

One contract value was changed after a first run: `retention_tolerance` was
0.05 nats, no arm (including full data) stayed within it, nothing was frozen
and the final evaluation did not run. It was raised to 0.25 so the freeze and
final-evaluation path would be exercised. The base model was also pretrained
for longer (3,000 steps instead of 600) so that the retained capability
meant something. Neither change was made to produce a win, and none occurred.

## Remaining empirical experiments

Not run, and nothing here claims otherwise:

1. **The benchmark contract** in
   [`examples/optimize/benchmark`](../examples/optimize/benchmark/README.md):
   Qwen2.5-0.5B-Instruct on databricks-dolly-15k, one GPU. The exact
   commands are in its README, with the list of things to verify on first
   contact with the real model and data.
2. **GPU and reduced precision.** `device: cuda`, `mps`, `dtype: bfloat16`
   and `float16` have never executed. Out-of-memory handling was tested by
   injecting the exceptions torch raises, not by exhausting a real device.
3. **A production chat template.** Loss masking was verified on the test
   template only. Templates with default system prompts or leading-space
   tokenization need the same check; encoding fails loudly if the template
   is not incremental.
4. **Older library versions** down to the declared lower bounds.
5. **An out-of-distribution target**, the setting the reference paper
   studies, on a model large enough to matter.
6. **Scale.** Nothing above a few hundred candidates has been run. Feature
   storage is `checkpoints x candidates x projection_dim x 4` bytes; k-center
   time grows with candidates times picks.
