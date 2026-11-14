#!/usr/bin/env python3
"""Package the engine the browser workbench runs, and keep it honest.

parallelogram.dev/workbench validates datasets inside the visitor's tab, with
the same Python that ships to PyPI running under Pyodide. This script copies
that Python into ``landing/workbench/engine/engine.zip``:

    parallelogram/      the package modules `check` and `--fix` use, unmodified
    bridge.py           landing/workbench/engine/bridge.py
    demo_dataset.py     scripts/demo_dataset.py (the sample dataset)

The CLI, the MCP server, `optimize` and the terminal renderers are left out:
they need typer, rich or ML packages, and the page uses none of them.

    python scripts/build_workbench.py           # rebuild engine.zip and the operator page
    python scripts/build_workbench.py --check   # fail if either is out of date

The archive is stored uncompressed with fixed timestamps, so the same sources
always give the same bytes. `--check` compares every entry with its source
file, which is what makes "the page runs the released engine" a tested claim
rather than a sentence.
"""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "cli/src/parallelogram"
BENCH = ROOT / "landing/workbench"
ARCHIVE = BENCH / "engine/engine.zip"
PAGE = BENCH / "index.html"
OPERATOR_PAGE = BENCH / "operator/index.html"

# Package paths that go to the browser. Directories are taken whole.
ENGINE = ["__init__.py", "api.py", "core", "formats", "rules", "output/__init__.py", "output/json_output.py"]


def sources() -> dict[str, Path]:
    files: dict[str, Path] = {}
    for entry in ENGINE:
        path = PACKAGE / entry
        if path.is_dir():
            for f in sorted(path.rglob("*.py")):
                files[f"parallelogram/{f.relative_to(PACKAGE).as_posix()}"] = f
        else:
            files[f"parallelogram/{entry}"] = path
    files["bridge.py"] = BENCH / "engine/bridge.py"
    files["demo_dataset.py"] = ROOT / "scripts/demo_dataset.py"
    return dict(sorted(files.items()))


def build() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for name, path in sources().items():
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            z.writestr(info, path.read_bytes())
    return buf.getvalue()


def operator_page() -> str:
    """The operator route is the same document with one attribute changed.

    It is a separate file because the two routes are served with different
    Content-Security-Policy headers (see landing/vercel.json): /workbench/
    cannot open a connection to anything but this site, /workbench/operator/
    can reach the model endpoint the visitor configures.
    """
    page = PAGE.read_text(encoding="utf-8")
    marker = '<html lang="en" data-mode="bench">'
    if page.count(marker) != 1:
        sys.exit(f"FAIL: {PAGE.relative_to(ROOT)} must contain exactly one {marker!r}")
    page = page.replace(marker, '<html lang="en" data-mode="operator">')
    return page.replace('<link rel="canonical" href="https://parallelogram.dev/workbench/">',
                        '<link rel="canonical" href="https://parallelogram.dev/workbench/operator/">')


def check() -> int:
    problems: list[str] = []
    if not ARCHIVE.exists():
        problems.append(f"{ARCHIVE.relative_to(ROOT)} is missing")
    else:
        want = sources()
        with zipfile.ZipFile(ARCHIVE) as z:
            have = set(z.namelist())
            for name in sorted(set(want) - have):
                problems.append(f"engine.zip is missing {name}")
            for name in sorted(have - set(want)):
                problems.append(f"engine.zip has {name}, which is not an engine source")
            for name in sorted(set(want) & have):
                if z.read(name) != want[name].read_bytes():
                    problems.append(f"engine.zip:{name} differs from {want[name].relative_to(ROOT)}")
    if not OPERATOR_PAGE.exists() or OPERATOR_PAGE.read_text(encoding="utf-8") != operator_page():
        problems.append(f"{OPERATOR_PAGE.relative_to(ROOT)} is not the current {PAGE.relative_to(ROOT)}")
    if problems:
        print("Workbench check FAILED. Run `python scripts/build_workbench.py`:\n")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print(f"Workbench check passed: engine.zip matches {len(sources())} source files; operator page is current")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify instead of writing")
    args = parser.parse_args()
    if args.check:
        return check()
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.write_bytes(build())
    OPERATOR_PAGE.parent.mkdir(parents=True, exist_ok=True)
    OPERATOR_PAGE.write_text(operator_page(), encoding="utf-8")
    print(f"wrote {ARCHIVE.relative_to(ROOT)} ({ARCHIVE.stat().st_size:,} bytes, {len(sources())} files)")
    print(f"wrote {OPERATOR_PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
