"""Copy landing/index.md into the Agent view of landing/index.html.

The landing page's Agent view shows the machine-readable page as text. The
site's CSP has no connect-src, so the page cannot fetch index.md at runtime;
its text is embedded between two markers instead. index.md is the source of
truth: edit it, then run this script.

    python scripts/build_agent_view.py          # rewrite index.html
    python scripts/build_agent_view.py --check  # exit 1 if they differ (CI)
"""
from __future__ import annotations

import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "landing"
START = "<!-- agent-md:start -->"
END = "<!-- agent-md:end -->"


def render(markdown: str) -> str:
    return START + html.escape(markdown, quote=False) + END


def main() -> int:
    page_path = ROOT / "index.html"
    page = page_path.read_text(encoding="utf-8")
    markdown = (ROOT / "index.md").read_text(encoding="utf-8")
    try:
        start = page.index(START)
        end = page.index(END, start) + len(END)
    except ValueError:
        print(f"index.html is missing the {START} / {END} markers", file=sys.stderr)
        return 1
    updated = page[:start] + render(markdown) + page[end:]
    if "--check" in sys.argv[1:]:
        if updated != page:
            print("landing/index.html is out of date with landing/index.md; "
                  "run python scripts/build_agent_view.py", file=sys.stderr)
            return 1
        print("Agent view matches landing/index.md")
        return 0
    page_path.write_text(updated, encoding="utf-8")
    print("Wrote the Agent view from landing/index.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
