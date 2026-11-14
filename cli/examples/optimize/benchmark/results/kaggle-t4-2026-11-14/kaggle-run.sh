#!/bin/bash
# Parallelogram optimize: Qwen2.5-0.5B on Dolly-15k, one GPU, Kaggle.
# The benchmark contract on branch claude/optimize, scaled to fit one Kaggle
# session (decided before any result existed):
#   - candidate pool capped at 6,000 records (prepare_dolly.py --max-candidates)
#   - token budget 110,000 (a tenth of the capped pool), search budget 55,000
#   - full-data arm off
#   - dev/test drawn only from prompts within max_seq_len (prepare_dolly --max-eval-tokens 512)
# Everything else is as written, including retention_tolerance 0.05.
set -euo pipefail
cd /kaggle/working
echo "== gpu"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "== install"
git clone -q --branch claude/optimize https://github.com/Thatayotlhe04/Parallelogram.git
git -C Parallelogram log --oneline -1
pip install -q -e "Parallelogram/cli[optimize]" "transformers==5.23.0" "peft==0.21.3" 2>&1 | tail -2 || true
# The image ships torchao 0.10; peft 0.21 refuses to build LoRA layers while an old torchao is importable.
pip uninstall -y -q torchao 2>&1 | tail -1 || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0), 'transformers', transformers.__version__, 'peft', peft.__version__)"
B=Parallelogram/cli/examples/optimize/benchmark
# prepare_dolly.py as of commit 638ad42 on claude/optimize (the pushed branch is older):
# split fix, pool cap, and dev/test drawn only from prompts that fit max_seq_len.
cat > "$B/prepare_dolly.py" <<'DOLLY_EOF'
"""Turn databricks-dolly-15k into the three splits of the benchmark contract.

    python prepare_dolly.py --input databricks-dolly-15k.jsonl --out data \
        --max-eval-tokens 512 --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
        --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775

Input is the dataset's single JSONL file (fields: instruction, context,
response, category), downloaded by you. Without `--tokenizer` the script
reads a local file and touches no network. Output is `data/{candidates,dev,test}.jsonl` in chat
format with an `id` and the original `category`.

Split rules, all deterministic:

* A record's split is decided by a hash of its *prompt*, so two records that
  ask the same thing can never end up on both sides of a split.
* Development and test examples are drawn only from the target and retained
  categories; everything else, including the rest of those categories, is a
  candidate.
* With `--max-eval-tokens`, a prompt is eligible for development or test
  only if its records fit in that many tokens, counted the way
  `parallelogram optimize` counts them (the tokenizer's chat template, whole
  conversation). The optimizer never truncates or drops an evaluation
  example, so an evaluation split must fit `model.max_seq_len` as written.
  Longer records stay candidates, where `on_overlength` decides.
* Records with an empty instruction or response, and exact repeats, are
  dropped here because `parallelogram optimize` refuses splits that fail
  `parallelogram check`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--target", default="summarization")
    parser.add_argument("--retain", default="open_qa")
    parser.add_argument("--dev-per-category", type=int, default=100)
    parser.add_argument("--test-per-category", type=int, default=300)
    parser.add_argument("--max-candidates", type=int, default=0,
                        help="keep at most this many candidates, sampled by prompt hash within each category "
                             "in proportion to its size (0 keeps all)")
    parser.add_argument("--max-eval-tokens", type=int, default=0,
                        help="only prompts whose records fit in this many tokens may become dev or test "
                             "(requires --tokenizer; 0 applies no limit)")
    parser.add_argument("--tokenizer", help="tokenizer used for --max-eval-tokens: a local directory or a "
                                            "Hugging Face id (downloaded if not cached)")
    parser.add_argument("--tokenizer-revision", help="pinned revision for a Hugging Face tokenizer id")
    args = parser.parse_args()
    if args.max_eval_tokens and not args.tokenizer:
        parser.error("--max-eval-tokens needs --tokenizer")

    rows, seen_prompts, seen_records = [], set(), set()
    with open(args.input, encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            source = json.loads(line)
            instruction = (source.get("instruction") or "").strip()
            context = (source.get("context") or "").strip()
            response = (source.get("response") or "").strip()
            if not instruction or not response:
                continue
            prompt = instruction if not context else f"{instruction}\n\n{context}"
            key = hashlib.sha256(f"{prompt}\x00{response}".encode("utf-8")).hexdigest()
            if key in seen_records:
                continue
            seen_records.add(key)
            rows.append({
                "id": f"dolly-{index:05d}", "category": source.get("category", "unknown"),
                "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}],
                "_prompt": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            })

    # Prompts too long to evaluate at --max-eval-tokens can't be dev or test.
    too_long: set[str] = set()
    if args.max_eval_tokens:
        count = token_counter(args.tokenizer, args.tokenizer_revision)
        for row in rows:
            if row["category"] in (args.target, args.retain) and count(row["messages"]) > args.max_eval_tokens:
                too_long.add(row["_prompt"])
        print(f"evaluation length limit: {len(too_long)} {args.target}/{args.retain} prompt(s) exceed "
              f"{args.max_eval_tokens} tokens and stay candidates")

    # Order evaluation-category prompts by hash; the first ones become test,
    # the next ones dev. Every record sharing a chosen prompt goes with it.
    assignment: dict[str, str] = {}
    for category in (args.target, args.retain):
        prompts = sorted({r["_prompt"] for r in rows if r["category"] == category} - set(assignment) - too_long)
        for i, prompt in enumerate(prompts):
            if i < args.test_per_category:
                assignment[prompt] = "test"
            elif i < args.test_per_category + args.dev_per_category:
                assignment[prompt] = "dev"
    splits: dict[str, list[dict]] = {"candidates": [], "dev": [], "test": []}
    for row in rows:
        split = assignment.get(row["_prompt"], "candidates")
        if split != "candidates":
            # A record of another category that shares an evaluation prompt is
            # dropped: it can't be evaluated as that capability, and keeping it
            # as a candidate would leak the prompt.
            if row["category"] not in (args.target, args.retain):
                continue
            # One record per evaluation prompt: a prompt with two different
            # reference answers would be a duplicate-prompt eval example.
            if row["_prompt"] in seen_prompts:
                continue
            seen_prompts.add(row["_prompt"])
        splits[split].append({k: v for k, v in row.items() if k != "_prompt"})

    if args.max_candidates and len(splits["candidates"]) > args.max_candidates:
        # Deterministic, proportional per category: lowest prompt hashes win.
        by_cat: dict[str, list[dict]] = {}
        for record in splits["candidates"]:
            by_cat.setdefault(record["category"], []).append(record)
        total = len(splits["candidates"])
        key = {r["id"]: r for r in rows}
        kept = []
        for category, records in sorted(by_cat.items()):
            share = round(args.max_candidates * len(records) / total)
            records.sort(key=lambda r: key[r["id"]]["_prompt"])
            kept.extend(records[:share])
        order = {r["id"]: i for i, r in enumerate(splits["candidates"])}
        splits["candidates"] = sorted(kept, key=lambda r: order[r["id"]])

    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        by_category: dict[str, int] = {}
        for record in records:
            by_category[record["category"]] = by_category.get(record["category"], 0) + 1
        print(f"{name}: {len(records)} records {dict(sorted(by_category.items()))}")


def token_counter(ref: str, revision: str | None):
    """Count tokens exactly as `parallelogram optimize` does: the tokenizer's
    chat template over the whole conversation, no added special tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(ref, revision=revision)
    if not tokenizer.chat_template:
        raise SystemExit(f"The tokenizer {ref!r} has no chat template")

    def count(messages: list[dict]) -> int:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    return count


if __name__ == "__main__":
    main()
DOLLY_EOF
echo "== data"
curl -sSL -o dolly.jsonl https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/main/databricks-dolly-15k.jsonl
sha256sum dolly.jsonl
mkdir -p exp
python "$B/prepare_dolly.py" --input dolly.jsonl --out exp/data --max-candidates 6000 \
  --max-eval-tokens 512 --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775
for f in exp/data/*.jsonl; do parallelogram check "$f" --no-color | tail -3 || true; done
python - "$B/experiment.yaml" <<'PY'
import sys
s = open(sys.argv[1]).read()
for old, new in (("  full_data: true", "  full_data: false"),
                 ("  token_budget: 220000", "  token_budget: 110000"),
                 ("  token_budgets: [110000]", "  token_budgets: [55000]")):
    assert old in s, old
    s = s.replace(old, new, 1)
open("exp/experiment.yaml", "w").write(s)
PY
diff "$B/experiment.yaml" exp/experiment.yaml || true
# Smoke pass: every stage once on a tiny pool, so a GPU-only failure costs minutes, not hours.
echo "== smoke"
mkdir -p smoke
python "$B/prepare_dolly.py" --input dolly.jsonl --out smoke/data --max-candidates 600 \
  --dev-per-category 20 --test-per-category 20 \
  --max-eval-tokens 512 --tokenizer Qwen/Qwen2.5-0.5B-Instruct \
  --tokenizer-revision 7ae557604adf67be50417f59c2c2f167def9a775
python - exp/experiment.yaml <<'PY'
import sys
s = open(sys.argv[1]).read()
for old, new in (("  seeds: [0, 1, 2]", "  seeds: [0]"),
                 ("  epochs: 3\n", "  epochs: 1\n"),
                 ("  token_budget: 110000", "  token_budget: 6000"),
                 ("  token_budgets: [55000]", "  token_budgets: [3000]"),
                 ("min_records: 100", "min_records: 5"),
                 ("  bootstrap_samples: 2000", "  bootstrap_samples: 200")):
    assert old in s, old
    s = s.replace(old, new, 1)
open("smoke/experiment.yaml", "w").write(s)
PY
( cd smoke && parallelogram optimize plan experiment.yaml --allow-download | tail -25 \
  && parallelogram optimize run experiment.yaml --allow-download > run.log 2>&1 ) \
  || { echo "smoke failed"; tail -60 smoke/run.log 2>/dev/null; tar czf /kaggle/working/smoke.tgz smoke/run smoke/run.log 2>/dev/null || true; exit 1; }
grep -v "^\[optimize\] training" smoke/run.log | tail -25
echo "== smoke passed ($(date -u +%H:%M) UTC)"
cd exp
echo "== plan"
parallelogram optimize plan experiment.yaml --allow-download
echo "== run (started $(date -u +%H:%M) UTC)"
if ! parallelogram optimize run experiment.yaml --allow-download > run.log 2>&1; then
  echo "run failed"; tail -60 run.log
fi
grep -v "^\[optimize\] training" run.log | tail -40
echo "== report (finished $(date -u +%H:%M) UTC)"
parallelogram optimize report run > /dev/null 2>&1 || true
[ -f run/report.md ] && cp run/report.md /kaggle/working/report.md && cat run/report.md
tar czf /kaggle/working/optimize-run.tgz run/*.json run/report.md run.log experiment.yaml 2>/dev/null || true
