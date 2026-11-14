# Benchmark: Dolly summarization on Qwen2.5-0.5B

**Status: run once, scaled down, on a Kaggle T4 (2026-11-14).** Result:
**not supported**. On held-out data, random selection beat the gradient
selector at the same number of training tokens, beyond noise, and random
selection given the selector's scoring compute did better still. The report,
the exact launcher and the deviations from this contract are in
[`results/kaggle-t4-2026-11-14`](results/kaggle-t4-2026-11-14/README.md); the
numbers and what they mean are in
[`docs/optimize.md`](../../../docs/optimize.md#benchmark-qwen25-05b-on-a-kaggle-t4).
The contract as written here (full pool, 220,000-token budget, full-data arm)
has not been run.

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
#    Evaluation examples are never truncated, so dev and test are drawn only from
#    prompts that fit max_seq_len, counted with the model's own tokenizer.
python prepare_dolly.py --input databricks-dolly-15k.jsonl --out data \
    --max-eval-tokens 512 --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
    --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775

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

These are the points that could not be verified without the model and data.
The first run settled some of them, noted inline.

- `optimize plan` prints the real pool size in tokens. `selection.token_budget`
  (220000) and `budget.total_compute` (2.5e8) were sized from the dataset's
  published record count and a guessed average length. Adjust both to the
  plan's numbers before running; if the budget is too small the plan says
  which arms it drops. *First run: the capped 5,999-record pool was 863,188
  tokens after exclusions, so 110,000 was 12.7% of it.*
- The plan reports how many candidates exceed `max_seq_len: 512` and are
  excluded. *First run: 470 of 5,999.* Development and test examples must fit
  too; prepare the data with `--max-eval-tokens 512` as shown above.
- The plan must list a `random+overhead` arm. If instead it lists
  "matched-compute baseline" under what is not planned, the scoring overhead
  exceeds what the pool can absorb and the run cannot reach a `supported`
  conclusion; use fewer warmup epochs or a longer recipe. *First run: planned,
  at 727,260 tokens.*
- Loss masking needs the tokenizer's chat template to render conversations
  incrementally. Encoding stops with an explicit error if Qwen's template
  does not. *First run: Qwen2.5's template passed and trained.*
- GPU execution (`device: cuda`) and reduced precision (`dtype: bfloat16`)
  have not been exercised. The contract uses `float32` to keep the first run
  on the path closest to what was tested. *First run: `cuda` with `float32`
  worked on a T4; the base-model evaluation ran out of memory at 16 examples
  per batch and continued at 8 by itself. `bfloat16` is still untried.*
- Throughput: per-example gradients are computed one example at a time.
  *First run, T4, `float32`: scoring 5,529 records at two checkpoints took
  1,728 s; training ran at about 2,300 forward-token-equivalents per second.*

## What a result would and would not show

Development and test examples come from the same dataset as the candidates,
with prompts kept disjoint across splits. That is an easier setting for any
selector than a target task from a different distribution. A positive result
here would support "gradient selection helps pick in-distribution data for
one category on one small model". It would not show the same for larger
models, other data, or out-of-distribution targets.
