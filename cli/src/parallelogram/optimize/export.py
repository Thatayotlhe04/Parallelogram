"""Write the selected records to a new file, byte for byte, with a manifest.

Export copies source lines; it never parses and re-serializes them, so key
order, whitespace, escapes and any extra fields survive exactly. The manifest
next to the output says where every line came from and why it was chosen.

Three refusals keep it safe: it will not write over an input dataset, it
will not replace an existing file unless told to, and it will not export
from a candidates file that has changed since the run measured it.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

from .. import __version__
from .cache import atomic_write, read_json, write_json
from .errors import ConfigError, OptimizeError
from .planner import trial_slug
from .reporting import PILOT_NOTE, build_report


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))
    except OSError:
        return False


def export(run_dir: str | Path, output: str | Path, *, trial: str | None = None, seed: int | None = None,
           overwrite: bool = False) -> dict:
    """Export a run's selection to `output`; return the manifest.

    By default the frozen candidate is exported. `trial` names another arm
    (for example a baseline subset, for inspection); the manifest then says
    it is not the frozen candidate.
    """
    run_dir, output = Path(run_dir), Path(output)
    if not (run_dir / "contract.json").is_file():
        raise OptimizeError(f"{run_dir} is not an optimize run directory (no contract.json)")
    contract = read_json(run_dir / "contract.json")
    pool = read_json(run_dir / "pool.json")
    frozen = read_json(run_dir / "frozen.json") if (run_dir / "frozen.json").is_file() else None
    if trial is None:
        if frozen is None:
            raise ConfigError("Nothing is frozen in this run yet. Finish `optimize run`, or name an arm "
                              "with --trial to export its subset for inspection.")
        trial, seed = frozen["trial"], frozen["selection_seed"] if seed is None else seed
    directory = run_dir / "trials" / trial_slug(trial)
    seeds = sorted(int(p.name.split("-", 1)[1]) for p in directory.glob("seed-*") if (p / "selection.json").is_file())
    if not seeds:
        known = sorted(read_json(p)["trial"]["key"] for p in run_dir.glob("trials/*/seed-*/result.json"))
        raise ConfigError(f"No finished arm named {trial!r} in this run. Arms: {sorted(set(known))}")
    seed = seeds[0] if seed is None else seed
    if seed not in seeds:
        raise ConfigError(f"Arm {trial!r} has no seed {seed}; available: {seeds}")
    selection = read_json(directory / f"seed-{seed}" / "selection.json")

    data = contract["contract"]["data"]
    source = Path(pool["source"]["path"])
    for name in ("candidates", "dev", "test"):
        if _same_file(output, Path(data[name]["path"])):
            raise ConfigError(f"--output is the {name} dataset; export never overwrites an input")
    manifest_path = output.with_name(output.name + ".manifest.json")
    for path in (output, manifest_path):
        if path.exists() and not overwrite:
            raise ConfigError(f"{path} already exists; pass --overwrite to replace it")
        if path.is_symlink():
            raise ConfigError(f"{path} is a symlink; refusing to write through it")
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise OptimizeError(f"Could not read the candidates file {source}: {exc}") from exc
    if hashlib.sha256(raw).hexdigest() != pool["source"]["sha256"]:
        raise OptimizeError(f"{source} has changed since this run read it. Exporting would attribute this "
                            "run's selection to different data; rerun the experiment instead.")
    lines = raw.splitlines(keepends=True)
    by_id = {record["id"]: record for record in pool["records"]}
    chosen = sorted((by_id[row["id"]]["line"], row) for row in selection["selected"])
    out = bytearray()
    records = []
    for line_no, row in chosen:
        line = lines[line_no - 1]
        meta = by_id[row["id"]]
        if hashlib.sha256(line.rstrip(b"\r\n")).hexdigest() != meta["raw_sha256"]:
            raise OptimizeError(f"Line {line_no} of {source} no longer matches the run's record")
        out += line if line.endswith(b"\n") else line + b"\n"
        records.append({"id": row["id"], "source_line": line_no, "raw_sha256": meta["raw_sha256"],
                        "tokens": row["tokens"], "selection_rank": row["rank"], "reason": row["reason"],
                        **({"scores": row["scores"]} if "scores" in row else {}),
                        **({"attrs": meta["attrs"]} if meta.get("attrs") else {})})
    report = build_report(run_dir)
    setup = read_json(run_dir / "setup.json")
    is_frozen = frozen is not None and frozen["trial"] == trial and frozen["selection_seed"] == seed
    manifest = {
        "schema_version": 1,
        "kind": "optimize-export-manifest",
        "output": {"path": str(output), "sha256": hashlib.sha256(bytes(out)).hexdigest(), "records": len(records),
                   "tokens": selection["tokens"], "order": "source line order"},
        "source": {**pool["source"], "note": "lines are copied byte for byte; nothing is rewritten"},
        "selection": {"trial": trial, "seed": seed, "policy": selection["policy"],
                      "token_budget": selection["token_budget"], "sha256": selection["sha256"],
                      "coverage": selection["coverage"], "frozen_candidate": is_frozen},
        "evidence": {
            "conclusion": report["conclusion"], "status": report["status"], "label": report["label"],
            "note": ("Scores in this manifest are estimated gradient-alignment signals, not calibrated "
                     "probabilities; `reason` records which rule picked a record, not that the record is "
                     "good. " + (PILOT_NOTE if setup["pilot"] else "")).strip(),
        },
        "fingerprints": {"contract": contract["contract_fingerprint"], "core": contract["core_fingerprint"],
                         "model": setup["model"]["model"], "tokenizer": setup["model"]["tokenizer"],
                         "chat_template_sha256": setup["model"]["chat_template_sha256"]},
        "provenance": {"parallelogram": __version__, "run_directory": str(run_dir),
                       "exported_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                       "library_versions": setup["model"]["versions"]},
        "records": records,
    }
    atomic_write(output, lambda f: f.write(bytes(out)))
    write_json(manifest_path, manifest)
    return manifest
