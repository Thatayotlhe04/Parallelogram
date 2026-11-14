"""Content-addressed cache and atomic artifact writes.

A cache entry's directory name is the hash of everything that determines its
content (dataset fingerprints, model and tokenizer revisions, chat template,
loss masking, optimizer, projection settings, seeds, library versions). There
is no invalidation step to forget: change any input and the key changes, so a
stale entry is simply never looked up.

An entry is complete only once `entry.json` exists; it is written last and
atomically. Anything found without it is a partial result from an interrupted
run, which resumable stages finish instead of trusting.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

from .config import digest

DONE = "entry.json"


def atomic_write(path: Path, write: Callable[[Any], None], mode: str = "wb") -> None:
    """Write through a temporary sibling and rename, so a reader (or a resumed
    run) sees either the whole artifact or none of it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        kwargs = {} if "b" in mode else {"encoding": "utf-8"}
        with os.fdopen(fd, mode, **kwargs) as stream:
            write(stream)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, lambda f: f.write(json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n"),
                 mode="w")


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


class Cache:
    """`<root>/<stage>/<key>/`, where `key` is the hash of the stage's inputs."""

    def __init__(self, root: Path):
        self.root = Path(root)

    @staticmethod
    def key(inputs: dict) -> str:
        return digest(inputs)[:32]

    def dir(self, stage: str, inputs: dict) -> Path:
        path = self.root / stage / self.key(inputs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def complete(self, stage: str, inputs: dict) -> dict | None:
        """The entry's metadata if the stage finished, else None."""
        marker = self.root / stage / self.key(inputs) / DONE
        if not marker.is_file():
            return None
        try:
            meta = read_json(marker)
        except (OSError, ValueError):
            return None
        # A key collision or a hand-edited entry must not be served as a hit.
        return meta if meta.get("inputs_sha256") == digest(inputs) else None

    def finish(self, stage: str, inputs: dict, meta: dict) -> dict:
        entry = {**meta, "inputs_sha256": digest(inputs), "inputs": inputs}
        write_json(self.root / stage / self.key(inputs) / DONE, entry)
        return entry
