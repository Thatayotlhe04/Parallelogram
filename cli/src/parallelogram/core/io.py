"""I/O utilities — atomic writes for JSONL output.

Kept separate from cli.py so it has no dependency on typer/rich and can
be imported by tests, the fixer, or any future module without dragging
the CLI surface into the import graph.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_jsonl(path: Path, lines: list[str]) -> None:
    """Write JSONL to path atomically — write to a temp sibling and rename.

    Guarantees that path either contains the complete output or remains
    untouched. Critical for --fix since users will overwrite their input.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for line in lines:
                if not line.endswith("\n"):
                    line += "\n"
                f.write(line)
        os.replace(tmp_path, path)
    except Exception:
        # Best-effort cleanup; never leave .tmp files behind on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
