"""Write the shared header and footer into every page of landing/.

Each page used to carry its own copy of the nav and footer, and the copies
drifted (different icons, stale links, a hand-edited version number). This
script is the one source: edit the templates below, then run it.

    python scripts/build_shared.py          # rewrite the pages
    python scripts/build_shared.py --check  # exit 1 if any page differs (CI)

The GitHub star count is kept from each page (the stars workflow updates it
in place) and the footer version comes from the package's __version__.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANDING = ROOT / "landing"
PAGES = ["index.html", "integrations.html", "privacy.html", "terms.html"]
REPO = "https://github.com/Thatayotlhe04/Parallelogram"

CHEVRON = ('<svg class="chev" viewBox="0 0 12 12" aria-hidden="true"><path d="M3 4.5 6 7.5 9 4.5" '
           'fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" '
           'stroke-linejoin="round"/></svg>')
BRAND = ('<svg class="brand-mark" viewBox="0 0 32 32" aria-hidden="true"><path d="M8 5 L29 5 L24 27 L3 27 Z" '
         'fill="none" stroke="currentColor" stroke-width="2.6" stroke-linejoin="round"/></svg>')
STAR = ('<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.3" '
        'stroke-linejoin="round" d="M8 1.8l1.9 3.9 4.3.6-3.1 3 .7 4.3L8 11.6l-3.8 2 .7-4.3-3.1-3 4.3-.6z"/></svg>')

# (menu label, [(title, description, href)]); a None list is a plain link.
NAV = [
    ("Product", [
        ("Checks", "What each rule catches, and why", "/#checks"),
        ("How it works", "Check, fix, then gate the run", "/#how"),
        ("Changelog", "What shipped, release by release", "/#changelog"),
        ("FAQ", "Formats, tokenizers and limits", "/#faq"),
    ]),
    ("Docs", [
        ("Quickstart", "Install and run your first check", f"{REPO}#quickstart"),
        ("Agents and MCP", "Python API, MCP tools, agent setup", f"{REPO}/blob/main/cli/docs/agents.md"),
        ("Integrations guide", "Agents, CI, pre-commit, trainers", f"{REPO}/blob/main/cli/docs/integrations.md"),
        ("JSON output", "The versioned --json contract", f"{REPO}/blob/main/cli/docs/json-output.md"),
        ("GitHub Action", "Fail a pull request on bad data", f"{REPO}#github-action"),
    ]),
    ("Integrations", "/integrations.html"),
    ("More", [
        ("Full changelog", "Every release, in detail", f"{REPO}/blob/main/cli/CHANGELOG.md"),
        ("PyPI", "The parallelogram package", "https://pypi.org/project/parallelogram/"),
        ("Report a vulnerability", "Privately, through GitHub", f"{REPO}/security/advisories/new"),
        ("llms.txt", "This site, for language models", "/llms.txt"),
        ("Privacy", "What the site and the CLI collect: nothing", "/privacy.html"),
        ("Terms", "Apache-2.0 and the site terms", "/terms.html"),
    ]),
]


def _menu(label: str, items: list[tuple[str, str, str]]) -> str:
    slug = label.lower()
    links = "\n".join(
        f'            <a href="{href}"><strong>{title}</strong><span>{desc}</span></a>'
        for title, desc, href in items)
    return (f'        <div class="nav-item has-menu">\n'
            f'          <button class="nav-trigger" type="button" aria-expanded="false" '
            f'aria-controls="menu-{slug}">{label}{CHEVRON}</button>\n'
            f'          <div class="menu" id="menu-{slug}">\n{links}\n          </div>\n'
            f'        </div>')


def header(stars: str, page: str) -> str:
    here = "/" + page
    desktop = "\n".join(
        _menu(label, target) if isinstance(target, list)
        else f'        <a class="nav-link" href="{target}"' + (' aria-current="page"' if target == here else '') + f'>{label}</a>'
        for label, target in NAV)
    mobile_groups = []
    for label, target in NAV:
        if isinstance(target, list):
            links = "".join(f'<a href="{href}">{title}</a>' for title, _, href in target)
            mobile_groups.append(f'<div class="mnav-group"><p>{label}</p>{links}</div>')
        else:
            mobile_groups.append(f'<div class="mnav-group"><a href="{target}">{label}</a></div>')
    return f'''<header class="nav">
    <div class="nav-inner">
      <a class="brand" href="/" aria-label="Parallelogram home">
        {BRAND}
        <span>parallelogram</span>
      </a>
      <nav class="nav-links" aria-label="Primary">
{desktop}
      </nav>
      <div class="nav-actions">
        <a class="nav-github" href="{REPO}" aria-label="Parallelogram on GitHub">
          {STAR}
          <span>GitHub</span>
          <span class="stars" data-stars="{stars}" title="GitHub stars">{stars}</span>
        </a>
        <a class="btn btn-primary btn-sm" href="/#install">Install</a>
      </div>
      <details class="mnav">
        <summary aria-label="Menu"><span></span><span></span></summary>
        <nav class="mnav-panel" aria-label="Menu">{"".join(mobile_groups)}</nav>
      </details>
    </div>
  </header>'''


def footer(version: str) -> str:
    return f'''<footer class="footer">
    <div class="footer-inner">
      <div class="footer-brand">
        <a class="brand" href="/">
          {BRAND}
          <span>parallelogram</span>
        </a>
        <p>The linter for fine-tuning data.</p>
        <p class="mono dim">v{version} · Apache-2.0 · runs offline</p>
      </div>
      <div class="footer-col">
        <h3>Product</h3>
        <a href="/#checks">Checks</a>
        <a href="/#how">How it works</a>
        <a href="/integrations.html">Integrations</a>
        <a href="/#changelog">Changelog</a>
        <a href="/#faq">FAQ</a>
      </div>
      <div class="footer-col">
        <h3>Resources</h3>
        <a href="{REPO}#readme">Docs</a>
        <a href="{REPO}">GitHub</a>
        <a href="https://pypi.org/project/parallelogram/">PyPI</a>
        <a href="{REPO}#github-action">GitHub Action</a>
        <a href="/llms.txt">llms.txt</a>
      </div>
      <div class="footer-col">
        <h3>Legal</h3>
        <a href="/privacy.html">Privacy</a>
        <a href="/terms.html">Terms</a>
        <a href="/.well-known/security.txt">Security</a>
      </div>
    </div>
    <div class="footer-bar"><span>© 2026 parallelogram.dev</span><span class="mono dim">If it exits 0, your run won't fail because of data.</span></div>
  </footer>'''


def main() -> int:
    version = re.search(r'__version__ = "([^"]+)"',
                        (ROOT / "cli/src/parallelogram/__init__.py").read_text()).group(1)
    check = "--check" in sys.argv[1:]
    stale = []
    for name in PAGES:
        path = LANDING / name
        page = path.read_text(encoding="utf-8")
        found = re.search(r'data-stars="(\d+)"', page)
        stars = found.group(1) if found else "0"
        updated = re.sub(r"<header class=\"nav\">.*?</header>", lambda _: header(stars, name), page, count=1, flags=re.S)
        updated = re.sub(r"<footer class=\"footer\">.*?</footer>", lambda _: footer(version), updated, count=1, flags=re.S)
        if updated != page:
            stale.append(name)
            if not check:
                path.write_text(updated, encoding="utf-8")
    if check:
        if stale:
            print("Shared header/footer out of date in: " + ", ".join(stale)
                  + "; run python scripts/build_shared.py", file=sys.stderr)
            return 1
        print("Shared header and footer match on every page")
        return 0
    print("Updated: " + (", ".join(stale) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
