#!/usr/bin/env python3
"""Fail if anything in landing/ would be blocked by the CSP in landing/vercel.json.

The landing site ships a strict Content-Security-Policy. That policy is only
safe to keep strict if nothing on the page needs an origin it doesn't allow —
this check keeps the two in sync. When it fails, it names the offending file,
line, and the directive to extend in landing/vercel.json.

Policies are per route: vercel.json gives the site pages one policy and each
workbench route its own. Every path must match exactly one of them.

No dependencies; regex-based scanning tuned to this static site. Not a general
CSP evaluator.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

LANDING = Path(__file__).resolve().parent.parent / "landing"

# script types that the browser never executes (CSP script-src ignores them)
NON_EXECUTABLE_SCRIPT_TYPES = {"application/ld+json", "application/json", "speculationrules"}

CONNECT_APIS = re.compile(
    r"\b(fetch\s*\(|XMLHttpRequest\b|WebSocket\b|EventSource\b|sendBeacon\b)"
)


def policies() -> list[tuple[re.Pattern, dict[str, list[str]]]]:
    """Every Content-Security-Policy in vercel.json, with the paths it is served on.

    The site has more than one: the pages share the strict policy, and the two
    workbench routes each have their own (the engine there is WebAssembly, and
    only /workbench/operator/ may open a connection to another origin).
    """
    cfg = json.loads((LANDING / "vercel.json").read_text())
    found = []
    for rule in cfg.get("headers", []):
        for h in rule["headers"]:
            if h["key"].lower() == "content-security-policy":
                directives: dict[str, list[str]] = {}
                for part in h["value"].split(";"):
                    tokens = part.split()
                    if tokens:
                        directives[tokens[0]] = tokens[1:]
                found.append((re.compile(rule["source"]), directives))
    if not found:
        sys.exit("FAIL: no Content-Security-Policy in landing/vercel.json")
    return found


def policy_for(url_path: str, rules=None) -> dict[str, list[str]]:
    """The one policy a path is served with. Zero or several is a failure:
    a page with no policy is unprotected, and two policies on one response
    are intersected by the browser, which is never what was meant."""
    matching = [d for pattern, d in (rules or policies()) if pattern.fullmatch(url_path)]
    if len(matching) != 1:
        sys.exit(f"FAIL: {url_path} is matched by {len(matching)} Content-Security-Policy rules "
                 f"in landing/vercel.json; every path must match exactly one")
    return matching[0]


def governing_page(path: Path) -> str:
    """URL path of the page whose policy governs a script or stylesheet.

    A script runs under the policy of the document that loads it, not the
    policy on its own response. operator.js is only ever imported by the
    operator page; everything else under workbench/ must work on the sealed
    page; the rest of the site shares one policy.
    """
    rel = path.relative_to(LANDING).as_posix()
    if rel == "workbench/operator.js":
        return "/workbench/operator/"
    if rel.startswith("workbench/"):
        return "/workbench/"
    return "/"


def effective(directives: dict[str, list[str]], name: str) -> list[str]:
    return directives.get(name, directives.get("default-src", []))


def origin_allowed(url: str, sources: list[str]) -> bool:
    url = url.strip().strip("'\"")
    if url.startswith("data:"):
        return "data:" in sources
    if url.startswith("blob:"):
        return "blob:" in sources
    m = re.match(r"^(https?:)?//([^/]+)", url)
    if not m:  # relative URL → same origin
        return "'self'" in sources
    host = m.group(2)
    for src in sources:
        if src in ("'self'", "'none'") or src.startswith("'"):
            continue
        pattern = re.escape(src.removeprefix("https://").removeprefix("http://"))
        if re.fullmatch(pattern.replace(r"\*", r"[^.]+"), host):
            return True
    return False


def main() -> int:
    rules = policies()
    problems: list[str] = []
    directives: dict[str, list[str]] = {}

    def check_url(url: str, directive: str, where: str) -> None:
        if not origin_allowed(url, effective(directives, directive)):
            problems.append(
                f"{where}: '{url}' is blocked by {directive} "
                f"{effective(directives, directive) or ['<missing>']} — "
                f"extend {directive} in landing/vercel.json or self-host it"
            )

    for path in sorted(LANDING.rglob("*.html")):
        text = path.read_text()
        rel = path.relative_to(LANDING).as_posix()
        directives = policy_for("/" + rel, rules)
        if path.name == "index.html":
            # A directory is also reachable without the file name, with and without the slash.
            folder = "/" + rel[: -len("index.html")]
            for alias in {folder, folder.rstrip("/") or "/"}:
                if policy_for(alias, rules) != directives:
                    problems.append(f"{rel}: {alias} is served with a different policy than /{rel}")

        for m in re.finditer(r"<script\b([^>]*)>", text):
            attrs = m.group(1)
            line = text.count("\n", 0, m.start()) + 1
            src = re.search(r'src\s*=\s*["\']([^"\']+)', attrs)
            stype = re.search(r'type\s*=\s*["\']([^"\']+)', attrs)
            if src:
                check_url(src.group(1), "script-src", f"{rel}:{line}")
            elif not (stype and stype.group(1).lower() in NON_EXECUTABLE_SCRIPT_TYPES):
                if "'unsafe-inline'" not in effective(directives, "script-src"):
                    problems.append(
                        f"{rel}:{line}: inline executable <script> is blocked by "
                        f"script-src — move it to a .js file"
                    )

        for m in re.finditer(r"<link\b[^>]*>", text):
            tag, line = m.group(0), text.count("\n", 0, m.start()) + 1
            rel_attr = re.search(r'rel\s*=\s*["\']([^"\']+)', tag)
            href = re.search(r'href\s*=\s*["\']([^"\']+)', tag)
            if not (rel_attr and href):
                continue
            kind = rel_attr.group(1).lower()
            if kind == "stylesheet":
                check_url(href.group(1), "style-src", f"{rel}:{line}")
            elif kind in ("icon", "apple-touch-icon"):
                check_url(href.group(1), "img-src", f"{rel}:{line}")
            elif kind == "preload":
                as_attr = re.search(r'as\s*=\s*["\']([^"\']+)', tag)
                directive = {
                    "font": "font-src", "style": "style-src",
                    "script": "script-src", "image": "img-src",
                }.get(as_attr.group(1).lower() if as_attr else "", "default-src")
                check_url(href.group(1), directive, f"{rel}:{line}")

        for m in re.finditer(r"<(img|source)\b[^>]*\bsrc\s*=\s*[\"']([^\"']+)", text):
            line = text.count("\n", 0, m.start()) + 1
            check_url(m.group(2), "img-src", f"{rel}:{line}")

        for m in re.finditer(r"<(iframe|frame|embed|object|video|audio)\b", text):
            line = text.count("\n", 0, m.start()) + 1
            tag = m.group(1)
            directive = "media-src" if tag in ("video", "audio") else "frame-src"
            if effective(directives, directive) in ([], ["'none'"]):
                problems.append(
                    f"{rel}:{line}: <{tag}> is blocked by {directive} "
                    f"(falls back to default-src 'none') — add {directive} to the CSP"
                )

        if "style=" in text and "'unsafe-inline'" not in effective(directives, "style-src"):
            problems.append(f"{rel}: inline style= attributes need 'unsafe-inline' in style-src")

    # vendor/ holds third-party runtimes exactly as published (Pyodide). They are
    # not scanned line by line: the policy of the page that loads them is what
    # confines them, and that policy is checked here like any other.
    scripts = [p for p in sorted(LANDING.rglob("*.js")) if "vendor" not in p.relative_to(LANDING).parts]
    for path in scripts:
        text = path.read_text()
        directives = policy_for(governing_page(path), rules)
        name = path.relative_to(LANDING).as_posix()
        if re.search(r"\bnew\s+Worker\s*\(", text):
            worker_src = directives.get("worker-src", directives.get("child-src", effective(directives, "script-src")))
            if "'self'" not in worker_src:
                problems.append(f"{name}: starts a Worker, which needs worker-src 'self' in the CSP")
        for m in CONNECT_APIS.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            if "'self'" not in effective(directives, "connect-src"):
                problems.append(
                    f"{name}:{line}: {m.group(1)} needs connect-src in the CSP "
                    f"(currently falls back to {effective(directives, 'connect-src') or 'default-src'})"
                )
        for m in re.finditer(r"\b(?:fetch|WebSocket|EventSource)\s*\(\s*[\"'](https?:)?//", text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(
                f"{name}:{line}: cross-origin network call — "
                f"add the origin to connect-src in landing/vercel.json"
            )

    # WebAssembly compiles only where script-src says so, and it must be the
    # narrow keyword: 'unsafe-eval' would also allow eval() of strings.
    for wasm in sorted(LANDING.rglob("*.wasm")):
        page = "/" + wasm.relative_to(LANDING).parts[0] + "/"
        for route in {page, governing_page(wasm)}:
            script_src = effective(policy_for(route, rules), "script-src")
            if "'wasm-unsafe-eval'" not in script_src:
                problems.append(f"{wasm.relative_to(LANDING)}: {route} needs 'wasm-unsafe-eval' in script-src")
    for pattern, d in rules:
        if "'unsafe-eval'" in effective(d, "script-src"):
            problems.append(f"{pattern.pattern}: script-src allows 'unsafe-eval'; use 'wasm-unsafe-eval' for WebAssembly")

    for path in sorted(LANDING.rglob("*.css")):
        text = path.read_text()
        directives = policy_for(governing_page(path), rules)
        for m in re.finditer(r"url\(\s*[\"']?((?:https?:)?//[^\"')]+|data:[^\"')]+)", text):
            line = text.count("\n", 0, m.start()) + 1
            url = m.group(1)
            directive = "font-src" if re.search(r"\.(woff2?|ttf|otf)\b", url) else "img-src"
            check_url(url, directive, f"{path.relative_to(LANDING).as_posix()}:{line}")

    if problems:
        print(f"CSP check FAILED — {len(problems)} resource(s) the deployed CSP would block:\n")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print("CSP check passed: everything landing/ loads is allowed by the policy in vercel.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
