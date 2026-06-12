#!/usr/bin/env python3
"""Fail if anything in landing/ would be blocked by the CSP in landing/vercel.json.

The landing site ships a strict Content-Security-Policy. That policy is only
safe to keep strict if nothing on the page needs an origin it doesn't allow —
this check keeps the two in sync. When it fails, it names the offending file,
line, and the directive to extend in landing/vercel.json.

No dependencies; regex-based scanning tuned to this static site (three HTML
pages, three JS files, one CSS file). Not a general CSP evaluator.
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


def load_csp() -> dict[str, list[str]]:
    cfg = json.loads((LANDING / "vercel.json").read_text())
    for rule in cfg.get("headers", []):
        if rule["source"] != "/(.*)":
            continue
        for h in rule["headers"]:
            if h["key"].lower() == "content-security-policy":
                directives: dict[str, list[str]] = {}
                for part in h["value"].split(";"):
                    tokens = part.split()
                    if tokens:
                        directives[tokens[0]] = tokens[1:]
                return directives
    sys.exit("FAIL: no Content-Security-Policy for source '/(.*)' in landing/vercel.json")


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
    directives = load_csp()
    problems: list[str] = []

    def check_url(url: str, directive: str, where: str) -> None:
        if not origin_allowed(url, effective(directives, directive)):
            problems.append(
                f"{where}: '{url}' is blocked by {directive} "
                f"{effective(directives, directive) or ['<missing>']} — "
                f"extend {directive} in landing/vercel.json or self-host it"
            )

    for path in sorted(LANDING.glob("*.html")):
        text = path.read_text()
        rel = path.name

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

    for path in sorted(LANDING.glob("*.js")):
        text = path.read_text()
        for m in CONNECT_APIS.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            if "'self'" not in effective(directives, "connect-src"):
                problems.append(
                    f"{path.name}:{line}: {m.group(1)} needs connect-src in the CSP "
                    f"(currently falls back to {effective(directives, 'connect-src') or 'default-src'})"
                )
        for m in re.finditer(r"\b(?:fetch|WebSocket|EventSource)\s*\(\s*[\"'](https?:)?//", text):
            line = text.count("\n", 0, m.start()) + 1
            problems.append(
                f"{path.name}:{line}: cross-origin network call — "
                f"add the origin to connect-src in landing/vercel.json"
            )

    for path in sorted(LANDING.glob("*.css")):
        text = path.read_text()
        for m in re.finditer(r"url\(\s*[\"']?((?:https?:)?//[^\"')]+|data:[^\"')]+)", text):
            line = text.count("\n", 0, m.start()) + 1
            url = m.group(1)
            directive = "font-src" if re.search(r"\.(woff2?|ttf|otf)\b", url) else "img-src"
            check_url(url, directive, f"{path.name}:{line}")

    if problems:
        print(f"CSP check FAILED — {len(problems)} resource(s) the deployed CSP would block:\n")
        print("\n".join(f"  - {p}" for p in problems))
        return 1
    print("CSP check passed: everything landing/ loads is allowed by the policy in vercel.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
