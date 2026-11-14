"""Generate the data and model for the tiny CPU experiment.

    python examples/optimize/make_tiny_experiment.py            # writes into examples/optimize/tiny/
    parallelogram optimize plan examples/optimize/tiny/experiment.yaml
    parallelogram optimize run  examples/optimize/tiny/experiment.yaml

Everything is synthetic and seeded. The model is a character-level GPT-2 with
about a hundred thousand parameters, pretrained here for a few thousand steps
on two string tasks so that LoRA has something to adapt. Nothing is
downloaded.

The task: candidates mix five kinds of record (reverse a string, copy it,
uppercase it, shift each letter by one, and reversal examples with wrong
answers). The target capability is `reverse`; `copy` must be retained. Only a
fifth of the pool teaches the target, which is the setting targeted selection
is designed for. That makes this a fair smoke test of the machinery and a
poor guide to real data: read its results as a pilot.
"""
from __future__ import annotations

import argparse
import json
import random
import string
from pathlib import Path

from parallelogram.optimize.tiny import build_tiny_model

SKILLS = {
    "reverse": lambda s, rng: s[::-1],
    "copy": lambda s, rng: s,
    "upper": lambda s, rng: s.upper(),
    "next": lambda s, rng: "".join(chr((ord(c) - 97 + 1) % 26 + 97) for c in s),
}


def record(index: int, skill: str, text: str, rng: random.Random, source: str) -> dict:
    if skill == "noisy":
        prompt, answer = f"reverse: {text}", "".join(rng.choice(string.ascii_lowercase) for _ in text)
    else:
        prompt, answer = f"{skill}: {text}", SKILLS[skill](text, rng)
    return {"id": f"{source}-{index:04d}", "skill": skill, "doc": f"{source}-doc-{index // 4:03d}",
            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "tiny")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=600)
    parser.add_argument("--pretrain-steps", type=int, default=3000)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    used: set[str] = set()

    def fresh() -> str:
        while True:
            text = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 6)))
            if text not in used:
                used.add(text)
                return text

    def split(name: str, counts: dict[str, int]) -> list[dict]:
        skills = [skill for skill, n in counts.items() for _ in range(n)]
        rng.shuffle(skills)
        return [record(i, skill, fresh(), rng, name) for i, skill in enumerate(skills)]

    n = args.candidates
    data = {
        "candidates": split("cand", {"reverse": n // 5, "copy": n // 5, "upper": n // 5,
                                     "next": 3 * n // 10, "noisy": n // 10}),
        "dev": split("dev", {"reverse": 40, "copy": 40}),
        "test": split("test", {"reverse": 100, "copy": 100}),
    }
    (args.out / "data").mkdir(parents=True, exist_ok=True)
    for name, rows in data.items():
        with open(args.out / "data" / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # The base model learns the chat format, copying and uppercasing. It never
    # sees reversal, and never sees any string used in the three splits.
    pretrain = [record(i, rng.choice(["copy", "upper"]), fresh(), rng, "pre")["messages"] for i in range(2000)]
    build_tiny_model(args.out / "model", seed=args.seed, pretrain=pretrain, pretrain_steps=args.pretrain_steps)
    print(f"wrote {args.out}/data/{{candidates,dev,test}}.jsonl and {args.out}/model")


if __name__ == "__main__":
    main()
