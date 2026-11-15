#!/usr/bin/env python3
"""Serve landing/ locally with the response headers from landing/vercel.json.

`python -m http.server` shows the pages but not the policy they ship under,
and the workbench depends on its policy: the sealed route must work with
`connect-src 'self'`, and Pyodide must start with `'wasm-unsafe-eval'` and no
`'unsafe-eval'`. This applies every header rule whose `source` matches the
request path, the way the deployment does.

    python scripts/serve_landing.py [port]        # default 8000
"""
from __future__ import annotations

import json
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LANDING = Path(__file__).resolve().parent.parent / "landing"


def header_rules() -> list[tuple[re.Pattern, list[dict]]]:
    config = json.loads((LANDING / "vercel.json").read_text())
    return [(re.compile(rule["source"]), rule["headers"]) for rule in config.get("headers", [])]


def headers_for(path: str, rules=None) -> dict[str, str]:
    """Response headers for a request path; a later rule overrides an earlier one."""
    out: dict[str, str] = {}
    for pattern, headers in rules or header_rules():
        if pattern.fullmatch(path):
            for h in headers:
                out[h["key"]] = h["value"]
    return out


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                      ".wasm": "application/wasm", ".js": "text/javascript"}

    def end_headers(self) -> None:
        for key, value in headers_for(self.path.split("?", 1)[0]).items():
            if key != "Strict-Transport-Security":      # plain http on localhost
                self.send_header(key, value)
        super().end_headers()

    def log_message(self, *args) -> None:
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(LANDING)))
    print(f"http://127.0.0.1:{port}/  (headers from landing/vercel.json)")
    server.serve_forever()
