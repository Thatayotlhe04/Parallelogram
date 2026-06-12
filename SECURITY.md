# Security Policy

Parallelogram is a local-first dataset linter: the CLI runs entirely on your
machine and the in-browser demo at [parallelogram.dev](https://parallelogram.dev)
processes pasted data in your browser only — nothing is uploaded, stored, or
sent to a server. Because that promise is the core of the product, we take
reports that undermine it seriously.

## Reporting a vulnerability

Please report vulnerabilities privately — do not open a public issue.

- **Preferred:** [GitHub private vulnerability reporting](https://github.com/Thatayotlhe04/Parallelogram/security/advisories/new)
- **Email:** tsenangthatayotlhe04@gmail.com

You should receive an acknowledgement within 72 hours. Please include steps to
reproduce, the affected component (CLI on PyPI, the website, or the in-browser
demo), and the version or commit if known.

## Scope

In scope:

- The `parallelogram` package on PyPI (parsing untrusted JSONL is the main
  attack surface — anything that makes `check`/`fix`/`report` execute code,
  write outside the requested output path, or misreport results)
- The website and in-browser demo at parallelogram.dev (anything that causes
  pasted data to leave the browser, or XSS via crafted dataset content)
- This repository's release and CI pipeline

Out of scope:

- Vulnerabilities in third-party dependencies without a demonstrated impact on
  Parallelogram (report those upstream, but a heads-up is welcome)
- Issues requiring a compromised machine or browser

## Supported versions

Only the latest release on PyPI receives security fixes. The machine-readable
contact info lives at
[parallelogram.dev/.well-known/security.txt](https://parallelogram.dev/.well-known/security.txt).
