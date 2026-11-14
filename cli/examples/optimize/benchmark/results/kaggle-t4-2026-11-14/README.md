# First run: Kaggle T4, 2026-11-14

The benchmark contract in `../../experiment.yaml`, scaled to fit one Kaggle
session, run once from an empty cache on one Tesla T4 (16 GB). The run
finished at 20:52 UTC after 19,834 s (5.5 h) of measured compute.

- `report.md`: the report `parallelogram optimize report` wrote, verbatim.
- `kaggle-run.sh`: the exact script the notebook ran. It installs the CLI from
  branch `claude/optimize`, writes `prepare_dolly.py` as of commit 638ad42,
  prepares the data, runs a smoke pass on a 600-record pool, then plans and
  runs the real experiment.

## Deviations from the contract

Decided before any training result existed:

| setting | contract | this run | why |
|---|---|---|---|
| candidate pool | all of Dolly outside dev/test | 5,999 records (`--max-candidates 6000`), 5,529 after 470 over-length exclusions | one 12 h Kaggle session |
| `selection.token_budget` | 220,000 | 110,000 | a tenth of the capped pool |
| `search.token_budgets` | [110,000] | [55,000] | half the budget, as written |
| `baselines.full_data` | true | false | one session |
| dev/test prompts | any length | only prompts within 512 tokens (`--max-eval-tokens 512`); 324 longer summarization/open_qa prompts stayed candidates | the optimizer never truncates an evaluation example, and the first plan stopped on one |

Everything else is as written: Qwen2.5-0.5B-Instruct at the pinned
revision, `float32`, LoRA r16 on q/v, three epochs, seeds 0 to 2,
`retention_tolerance` 0.05.

Environment: torch 2.11.0+cu130, transformers 5.23.0, peft 0.21.3, numpy
2.1.3, Python 3.13. Kaggle's image ships torchao 0.10, which peft 0.21 refuses
to load alongside; the script uninstalls it.

## Reproduce

Create a Kaggle notebook, set the accelerator to a GPU and turn internet on,
paste the script into one cell under a first line of `%%bash`, and use Save
Version, then Save & Run All (Commit). It clones the public repository and
downloads the model and data itself, so it needs no attached input. The run's
own output goes to `exp/run.log` and appears in the notebook log only when the
run ends.
