"""Shared builders for the optimize tests: tiny datasets and contracts."""
from __future__ import annotations

import json
import random
import string
from pathlib import Path

SKILLS = {"reverse": lambda s: s[::-1], "copy": lambda s: s, "upper": lambda s: s.upper()}


def row(index: int, skill: str, text: str, source: str, **extra) -> dict:
    return {"id": f"{source}-{index:03d}", "skill": skill, "lang": "en" if index % 3 else "sw",
            "doc": f"{source}-doc-{index}", **extra,
            "messages": [{"role": "user", "content": f"{skill}: {text}"},
                         {"role": "assistant", "content": SKILLS[skill](text)}]}


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


def make_data(root: Path, *, candidates: int = 36, dev: int = 12, test: int = 12, seed: int = 0) -> dict[str, list[dict]]:
    """Three disjoint splits of short string tasks, written under `root`."""
    rng = random.Random(seed)
    used: set[str] = set()

    def fresh() -> str:
        while True:
            text = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 5)))
            if text not in used:
                used.add(text)
                return text

    data = {
        "candidates": [row(i, ("reverse", "copy", "upper")[i % 3], fresh(), "cand") for i in range(candidates)],
        "dev": [row(i, ("reverse", "copy")[i % 2], fresh(), "dev") for i in range(dev)],
        "test": [row(i, ("reverse", "copy")[i % 2], fresh(), "test") for i in range(test)],
    }
    for name, rows in data.items():
        write_jsonl(root / f"{name}.jsonl", rows)
    (root / "model").mkdir(exist_ok=True)  # a local (here empty) model directory, not a Hub id
    return data


def contract(root: Path, **overrides) -> dict:
    """A small valid contract; `overrides` replace top-level sections or, with
    dotted keys such as `recipe.epochs`, single values."""
    base = {
        "schema_version": 1, "name": "test", "scale": "pilot", "objective": "sft-lora-causal-lm",
        "data": {"candidates": {"path": "candidates.jsonl"}, "dev": {"path": "dev.jsonl"},
                 "test": {"path": "test.jsonl"}, "id_field": "id", "capability_field": "skill"},
        "model": {"path": "model", "max_seq_len": 48, "device": "cpu"},
        "recipe": {"lora": {"r": 2, "alpha": 4, "target_modules": ["c_attn"]}, "seeds": [0, 1],
                   "learning_rate": 0.01, "epochs": 1, "batch_size": 4},
        "evaluation": {"metrics": ["loss"], "primary_metric": "loss", "target_capabilities": ["reverse"],
                       "retain_capabilities": ["copy"], "retention_tolerance": 100.0,
                       "bootstrap_samples": 100},
        "selection": {"token_budget": 150, "policies": ["random", "diversity", "less"],
                      "less": {"warmup": {"fraction": 0.25, "epochs": 2},
                               "projection": {"kind": "countsketch", "dim": 64}, "shard_size": 8}},
        "baselines": {"base_model": True, "full_data": True, "random_matched_compute": True},
        "budget": {"total_compute": 1e7},
        "output": {"run_dir": "run", "cache_dir": "cache"},
    }
    for key, value in overrides.items():
        target = base
        *parents, leaf = key.split(".")
        for part in parents:
            target = target.setdefault(part, {})
        target[leaf] = value
    return base


def write_contract(root: Path, **overrides) -> Path:
    path = root / "experiment.json"
    path.write_text(json.dumps(contract(root, **overrides), indent=2), encoding="utf-8")
    return path
