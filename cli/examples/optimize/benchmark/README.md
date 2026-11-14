# Benchmark: Dolly summarization on Qwen2.5-0.5B

**Status: not run.** This directory holds a contract and a data-preparation
script for a realistic experiment. It was written on a CPU-only machine with
no access to the Hugging Face Hub. No result is claimed for it anywhere in
this repository. `prepare_dolly.py` was exercised on stand-in data with the
same schema, and the contract is parsed by the test suite; the training
itself has never been executed.

## What it asks

Tune `Qwen/Qwen2.5-0.5B-Instruct` with LoRA for summarization while keeping
open-ended QA, using about a tenth of `databricks-dolly-15k` as training
data. Compare, over three seeds each:

| arm | comparison | question |
|---|---|---|
| `random@220000`, `diversity@220000` | matched tokens | Is gradient selection better than cheaper ways of picking the same amount of data? |
| `random+overhead` | matched compute | Would the scoring compute have been better spent training on more random data? |
| `full` | full data | How much quality does a tenth of the data give up? |

The stated goal is a better quality/total-compute tradeoff, so the second
row is the one that decides the headline.

## Commands

Requires one GPU with 16 GB or more, and the extras:
`pip install 'parallelogram[optimize]'`.

```bash
cd cli/examples/optimize/benchmark

# 1. Data (CC BY-SA 3.0). Download the single JSONL file yourself, then split it.
#    https://huggingface.co/datasets/databricks/databricks-dolly-15k  ->  databricks-dolly-15k.jsonl
python prepare_dolly.py --input databricks-dolly-15k.jsonl --out data

# 2. Each split must pass the linter (optimize refuses otherwise).
for f in data/candidates.jsonl data/dev.jsonl data/test.jsonl; do parallelogram check "$f"; done

# 3. Confirm the pinned model revision, then fetch the tokenizer and plan.
git ls-remote https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct main
parallelogram optimize plan experiment.yaml --allow-download

# 4. Run. Safe to interrupt; the same command resumes.
parallelogram optimize run experiment.yaml --allow-download

# 5. Read the result and take the subset.
parallelogram optimize report run
parallelogram optimize export run --output selected.jsonl
```

If a split fails step 2 with fixable errors, repair it with
`parallelogram check <file> --fix --output <clean>` and point the contract at
the repaired file.

## Things to check when you run it

These are the points that could not be verified without the model and data:

- `optimize plan` prints the real pool size in tokens. `selection.token_budget`
  (220000) and `budget.total_compute` (2.5e8) were sized from the dataset's
  published record count and a guessed average length. Adjust both to the
  plan's numbers before running; if the budget is too small the plan says
  which arms it drops.
- The plan reports how many candidates exceed `max_seq_len: 512` and are
  excluded.
- The plan must list a `random+overhead` arm. If instead it lists
  "matched-compute baseline" under what is not planned, the scoring overhead
  exceeds what the pool can absorb and the run cannot reach a `supported`
  conclusion; use fewer warmup epochs or a longer recipe.
- Loss masking needs the tokenizer's chat template to render conversations
  incrementally. Encoding stops with an explicit error if Qwen's template
  does not; it has not been tried here.
- GPU execution (`device: cuda`) and reduced precision (`dtype: bfloat16`)
  have not been exercised. The contract uses `float32` to keep the first run
  on the path closest to what was tested.
- Throughput: per-example gradients are computed one example at a time.

## What a result would and would not show

Development and test examples come from the same dataset as the candidates,
with prompts kept disjoint across splits. That is an easier setting for any
selector than a target task from a different distribution. A positive result
here would support "gradient selection helps pick in-distribution data for
one category on one small model". It would not show the same for larger
models, other data, or out-of-distribution targets.
