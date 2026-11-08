# Public datasets, checked

Parallelogram 1.6.1 run over 21 popular fine-tuning datasets on the Hugging
Face Hub: 675,442 rows, up to the first 50,000 of each split. Every download is
pinned to a revision and its SHA-256 was checked before the scan.

The point was to find false positives before users do. The first pass found
four kinds, all now fixed: Hugging Face `datasets` writing `null` for unused
tool columns, TRL-style string prompts beside full `chosen`/`rejected`
conversations, a `conversation` metadata column beside a DPO pair, and Hermes
`<tool_call>` calls written in the text. What remains below is what the data
actually contains.

## Errors

An error makes `check` exit 2. These are the rows a training run would use
as given.

| Dataset | Split @ revision | Rows | Rows with errors | What |
|---|---|---:|---:|---|
| HuggingFaceH4/ultrafeedback_binarized | train_prefs @ 3949bf5 | 50,000 | 409 | 350 pairs where chosen and rejected are the same text; 61 with an empty response; 4 key-shaped strings |
| trl-lib/ultrafeedback_binarized | train @ 47124cb | 50,000 | 425 | 358 identical pairs; 69 empty responses; 6 key-shaped strings |
| NousResearch/hermes-function-calling-v1 | func_calling @ dae3e1d | 1,893 | 173 | 163 end on tool replies with no final answer (156 of them have more replies than calls); 10 have a tool reply with no call before it |
| minpeter/hermes-function-calling-v1-parsed | func-calling @ 05fef09 | 1,893 | 234 | the same 173, plus 61 calls to a function while `tools` is `[]` (all in the Json Schema subcategory) |
| trl-lib/Capybara | train @ e235e84 | 15,806 | 69 | 68 duplicate conversations; 1 empty turn |
| mlabonne/FineTome-100k | train @ c2343c1 | 50,000 | 73 | each repeats the row before it (56 byte-identical, 17 differ only in surrounding whitespace) |
| mlabonne/orpo-dpo-mix-40k | train @ 0f72511 | 44,245 | 42 | 38 identical pairs; 2 empty turns; 2 key-shaped strings |
| tatsu-lab/alpaca | train @ dce01c9 | 50,000 | 27 | empty `output` |
| HuggingFaceTB/smoltalk | all/train @ 5feaf2f | 50,000 | 25 | 16 conversations end on a user turn; 5 duplicates; 3 rows share one GitHub-token-shaped default in a tool schema |
| HuggingFaceH4/no_robots | train @ e6f9a4a | 9,500 | 15 | two assistant turns in a row |
| HuggingFaceH4/ultrachat_200k | train_sft @ 8049631 | 50,000 | 10 | 9 empty turns; 1 Slack-webhook-shaped URL |
| BelleGroup/train_0.5M_CN | train @ 29c35e9 | 50,000 | 8 | 6 duplicates; 2 whitespace-only answers |
| Intel/orca_dpo_pairs | train @ 624952e | 12,859 | 2 | identical pairs |
| argilla/dpo-mix-7k | train @ f8869fc | 6,750 | 2 | 1 identical pair; 1 key-shaped string |
| argilla/distilabel-capybara-dpo-7k-binarized | train @ 801d130 | 7,563 | 2 | 1 identical pair; 1 key-shaped string |
| yahma/alpaca-cleaned | train @ 12567ca | 50,000 | 2 | duplicates |
| allenai/tulu-3-sft-mixture | train @ b14afda | 50,000 | 1 | duplicate |
| trl-internal-testing/toolcall | train @ 2a528c6 | 7 | 1 | calls `get_air_quality`; the tools list declares `get_air_quality_index` |
| Open-Orca/SlimOrca | train @ ce9ed00 | 50,000 | 0 | |
| garage-bAInd/Open-Platypus | train @ 37141ed | 24,926 | 0 | |
| vicgalle/alpaca-gpt4 | train @ f7e3ded | 50,000 | 0 | |

1,520 of 675,442 rows (0.23%) have at least one error. That is low, and it
should be: these are curated datasets. The rows that remain are the kind a
trainer accepts without complaint and trains on anyway. An identical DPO pair
pushes the same text up and down; an empty response teaches the model to stop;
a conversation that ends on a tool reply teaches calls that never get an
answer.

"Key-shaped" means the string matches a credential pattern (AWS access key,
GitHub or Slack token, Google API key, private key header, Slack webhook). Most are probably made up by
the model that wrote the sample. They were not tested against any service.

Row lists for the reports sent upstream are in [public-datasets/](public-datasets/),
one file per dataset, 0-based row indices as the dataset viewer numbers them.

## Warnings

Warnings make `check` exit 1 and never block on their own.

- **PII**: email addresses and phone numbers, from 20 rows (Open-Platypus) to
  704 (ultrachat_200k). Nearly all are invented contact details inside task
  prompts ("extract the email from this signature"). Reserved example domains
  and 555 numbers are already ignored. Treat the count as "rows to look at",
  not leaks.
- **Context window**: rows over the default 4,096-token estimate, up to 379
  (smoltalk). Pass `--max-seq-len` for your model, and `--tokenizer` for exact
  counts.
- **Encoding**: mojibake-like sequences, up to 130 rows (SlimOrca).

## Reproduce

Each dataset was streamed with `datasets.load_dataset(name, config, split=split,
revision=revision, streaming=True)` and the first 50,000 rows written as JSONL.
Intel/orca_dpo_pairs has no prompt column, so a `prompt` list (its `system`
and `question`) was added; every other file is the rows as published. Then:

```bash
pip install parallelogram
parallelogram check ultrafeedback_binarized.jsonl --json
```

Found something it gets wrong on a public dataset? Open an issue with the
dataset name and a line number.
