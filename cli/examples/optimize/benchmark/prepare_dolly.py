"""Turn databricks-dolly-15k into the three splits of the benchmark contract.

    python prepare_dolly.py --input databricks-dolly-15k.jsonl --out data

Input is the dataset's single JSONL file (fields: instruction, context,
response, category), downloaded by you; this script reads a local file and
touches no network. Output is `data/{candidates,dev,test}.jsonl` in chat
format with an `id` and the original `category`.

Split rules, all deterministic:

* A record's split is decided by a hash of its *prompt*, so two records that
  ask the same thing can never end up on both sides of a split.
* Development and test examples are drawn only from the target and retained
  categories; everything else, including the rest of those categories, is a
  candidate.
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
    args = parser.parse_args()

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

    # Order evaluation-category prompts by hash; the first ones become test,
    # the next ones dev. Every record sharing a chosen prompt goes with it.
    assignment: dict[str, str] = {}
    for category in (args.target, args.retain):
        prompts = sorted({r["_prompt"] for r in rows if r["category"] == category} - set(assignment))
        for i, prompt in enumerate(prompts):
            if i < args.test_per_category:
                assignment[prompt] = "test"
            elif i < args.test_per_category + args.dev_per_category:
                assignment[prompt] = "dev"
    splits: dict[str, list[dict]] = {"candidates": [], "dev": [], "test": []}
    for row in rows:
        split = assignment.get(row["_prompt"], "candidates")
        if split != "candidates":
            # One record per evaluation prompt: a prompt with two different
            # reference answers would be a duplicate-prompt eval example.
            if row["_prompt"] in seen_prompts:
                continue
            seen_prompts.add(row["_prompt"])
        splits[split].append({k: v for k, v in row.items() if k != "_prompt"})

    args.out.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with open(args.out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        by_category: dict[str, int] = {}
        for record in records:
            by_category[record["category"]] = by_category.get(record["category"], 0) + 1
        print(f"{name}: {len(records)} records {dict(sorted(by_category.items()))}")


if __name__ == "__main__":
    main()
