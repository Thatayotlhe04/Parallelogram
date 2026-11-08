"""Tool exchanges must validate, survive safe repairs, and stay private."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from parallelogram import check, fix, rules
from parallelogram.cli import app
from parallelogram.formats.sharegpt import normalize, source_metadata, to_sharegpt
from parallelogram.rules.context_window import ContextWindowRule
from parallelogram.rules.duplicates import record_hash
from test_mcp import session, tool, unpack
from typer.testing import CliRunner


def conversation():
    return {"tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
            "parallel_tool_calls": True,
            "messages": [{"role": "user", "content": "Compare the two places."},
                         {"role": "assistant", "content": None, "tool_calls": [
                             {"id": "a", "type": "function", "function": {"name": "lookup", "arguments": '{"place":"A"}'}},
                             {"id": "b", "type": "function", "function": {"name": "lookup", "arguments": '{"place":"B"}'}}]},
                         {"role": "tool", "tool_call_id": "b", "content": "B is warm."},
                         {"role": "tool", "tool_call_id": "a", "content": "A is cold."},
                         {"role": "assistant", "content": "B is warmer than A."}]}


def sharegpt():
    return {"system": "Use the tools.", "tools": json.dumps([{"name": "lookup", "parameters": {"type": "object"}}]),
            "conversations": [{"from": "human", "value": "Find the weather."},
                              {"from": "function_call", "value": json.dumps({"name": "lookup", "arguments": {"place": "A"}})},
                              {"from": "observation", "value": "It is cold."},
                              {"from": "gpt", "value": "It is cold there."}]}


def write(tmp_path, record):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps(record) + "\n")
    return path


def findings(tmp_path, record, rule):
    return [i for i in check(write(tmp_path, record)).to_dict()["issues"] if i["rule"] == rule]


@pytest.mark.parametrize("factory", [conversation, sharegpt])
@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b""])
def test_clean_roundtrip(tmp_path, factory, ending):
    path = tmp_path / "train.jsonl"
    path.write_bytes(b"\r\n  " + json.dumps(factory(), ensure_ascii=False).encode() + b"  " + ending)
    assert check(path).exit_code == 0
    out = tmp_path / "out.jsonl"
    result = CliRunner().invoke(app, ["check", str(path), "--fix", "--output", str(out)])
    assert result.exit_code == 0, result.output
    assert out.read_bytes() == path.read_bytes()
    assert check(out).exit_code == 0


@pytest.mark.parametrize("change,fragment", [
    (lambda r: r["messages"][2].update(tool_call_id="unknown"), "no matching"),
    (lambda r: r["messages"].pop(2), "never answered"),
    (lambda r: r["messages"].insert(4, deepcopy(r["messages"][2])), "already answered"),
    (lambda r: r["messages"].pop(), "must end"),
    (lambda r: r["messages"].__setitem__(slice(2, None), [{"role": "user", "content": "And?"},
                                                         {"role": "assistant", "content": "Done."}]),
     "never answered before the next"),
])
def test_roles_errors(tmp_path, change, fragment):
    record = conversation()
    change(record)
    issues = findings(tmp_path, record, "roles")
    assert any(fragment in i["message"] for i in issues)
    assert all(i["line"] == 1 and "Message " in i["message"] for i in issues)
    assert fix(write(tmp_path, record)).report.dropped == 1


@pytest.mark.parametrize("arguments", ['{"secret":"do-not-print"', '[]', 'null', '"do-not-print"', 5, [], None])
def test_arguments_error_private(tmp_path, arguments):
    record = conversation()
    record["messages"][1]["tool_calls"][0]["function"]["arguments"] = arguments
    issues = findings(tmp_path, record, "tool-calls")
    assert issues and "JSON-encoded object" in issues[0]["message"]
    assert "do-not-print" not in json.dumps(issues)
    assert "Message 1" in issues[0]["message"]


@pytest.mark.parametrize("change", [
    lambda r: r["messages"][1]["tool_calls"][0]["function"].update(name="absent"),
    lambda r: r["messages"][1]["tool_calls"][1].update(id="a"),
    lambda r: r["messages"][1]["tool_calls"][0].update(id=""),
    lambda r: r["messages"][1]["tool_calls"][0].update(type="invalid"),
    lambda r: r["messages"][1]["tool_calls"][0].update(function=[]),
    lambda r: r["messages"][1].update(tool_calls={}),
    lambda r: r["messages"][1].update(tool_calls=[None]),
    lambda r: r["messages"][0].update(tool_calls=[]),
    lambda r: r.update(tools="bad"),
    lambda r: r.update(tools=[None]),
    lambda r: r["tools"][0]["function"].pop("name"),
    lambda r: r["tools"][0]["function"].update(parameters=[]),
    lambda r: r["tools"][0]["function"].update(description=1),
    lambda r: r.update(parallel_tool_calls="yes"),
])
def test_tool_shape_errors(tmp_path, change):
    record = conversation()
    change(record)
    assert findings(tmp_path, record, "tool-calls")


def test_definitions_optional_and_content_optional(tmp_path):
    record = conversation()
    record.pop("tools")
    record["messages"][1].pop("content")
    assert check(write(tmp_path, record)).exit_code == 0
    record["messages"][1]["content"] = ""
    assert check(write(tmp_path, record)).exit_code == 0
    record["messages"][2]["content"] = " "
    assert findings(tmp_path, record, "empty-content")
    result = fix(write(tmp_path, record))
    assert result.report.dropped == 1 and not result.report.clean_records


def test_legacy_calls(tmp_path):
    record = {"functions": [{"name": "lookup", "parameters": {}}], "messages": [
        {"role": "user", "content": "Look it up."},
        {"role": "assistant", "content": None, "function_call": {"name": "lookup", "arguments": "{}"}},
        {"role": "function", "name": "lookup", "content": "Found it."},
        {"role": "assistant", "content": "Here it is."}]}
    assert check(write(tmp_path, record)).exit_code == 0
    record["messages"][2]["name"] = "other"
    assert findings(tmp_path, record, "roles")


@pytest.mark.parametrize("change,rule", [
    (lambda r: r["conversations"].pop(2), "roles"),
    (lambda r: r["conversations"].insert(3, {"from": "observation", "value": "Extra"}), "roles"),
    (lambda r: r["conversations"].pop(), "roles"),
    (lambda r: r["conversations"][1].update(value='{"name":"lookup","arguments":[]}'), "tool-calls"),
    (lambda r: r["conversations"][1].update(value='{"name":"missing","arguments":{}}'), "tool-calls"),
    (lambda r: r["conversations"][1].update(value='broken-private-value'), "tool-calls"),
    (lambda r: r.update(tools='[{"parameters":{}}]'), "tool-calls"),
    (lambda r: r.update(tools='invalid'), "tool-calls"),
    (lambda r: r["conversations"][2].update(value=""), "empty-content"),
])
def test_sharegpt_errors(tmp_path, change, rule):
    record = sharegpt()
    change(record)
    assert findings(tmp_path, record, rule)


def test_sharegpt_repair_preserves_shape(tmp_path):
    record = sharegpt()
    record["conversations"][3]["value"] = "\ufeffIt is cold there."
    path = write(tmp_path, record)
    out = tmp_path / "fixed.jsonl"
    assert fix(path, output=out).exit_code == 0
    result = json.loads(out.read_text())
    assert [t["from"] for t in result["conversations"]] == [t["from"] for t in record["conversations"]]
    assert isinstance(result["tools"], str) and result["system"] == record["system"]
    assert check(out).exit_code == 0
    assert normalize(to_sharegpt(normalize(record), source_metadata(record))) == normalize(record)


def test_counts_and_hashes_include_calls_and_definitions():
    first = conversation()
    second = deepcopy(first)
    second["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"text": "x" * 20000})
    counter = ContextWindowRule()
    assert counter.record_tokens(second) > counter.record_tokens(first) + 4000
    assert record_hash(first) != record_hash(second)
    third = deepcopy(first)
    third["tools"][0]["function"]["description"] = "y" * 20000
    assert counter.record_tokens(third) > counter.record_tokens(first) + 4000
    assert record_hash(first) != record_hash(third)


def test_json_api_mcp(tmp_path):
    record = conversation()
    record["messages"][1]["tool_calls"][0]["function"]["arguments"] = "private-invalid-json"
    path = write(tmp_path, record)
    api_result = check(path).to_dict()
    result = CliRunner().invoke(app, ["check", str(path), "--json"])
    assert result.exit_code == 2
    assert json.loads(result.stdout) == api_result
    result = unpack(session(tmp_path, [tool(1, "check_dataset", {"path": str(path)})])[0])
    assert result.pop("exit_code") == 2
    assert result == api_result
    assert "private-invalid-json" not in json.dumps(result)
    metadata = next(r for r in rules() if r["id"] == "tool-calls")
    assert metadata["severity"] == "error" and not metadata["fixable"]


@pytest.mark.parametrize("name", ["tools_clean.jsonl", "sharegpt_tools_clean.jsonl"])
def test_examples(name):
    assert check(Path(__file__).parents[1] / "examples" / name).exit_code == 0


def test_followup_user_and_multiple_tool_rounds(tmp_path):
    record = conversation()
    record["messages"].insert(4, {"role": "user", "content": "Now summarize."})
    assert check(write(tmp_path, record)).exit_code == 0
    record = conversation()
    call = deepcopy(record["messages"][1])
    call["tool_calls"] = [call["tool_calls"][0]]
    call["tool_calls"][0]["id"] = "c"
    record["messages"][4:4] = [call, {"role": "tool", "tool_call_id": "c", "content": "Confirmed."}]
    assert check(write(tmp_path, record)).exit_code == 0


@pytest.mark.parametrize("factory", [conversation, sharegpt])
def test_secrets_in_arguments_are_redacted_without_losing_tools(tmp_path, factory):
    record = factory()
    secret = "sk-" + "AbCdEf0123456789" * 3
    if "messages" in record:
        record["messages"][1]["tool_calls"][0]["function"]["arguments"] = json.dumps({"token": secret})
    else:
        record["conversations"][1]["value"] = json.dumps({"name": "lookup", "arguments": {"token": secret}})
    path = write(tmp_path, record)
    assert findings(tmp_path, record, "secrets")
    out = tmp_path / "out.jsonl"
    result = fix(path, output=out)
    assert result.exit_code == 0
    assert secret not in out.read_text()
    assert check(out).exit_code == 0
    assert len(json.loads(out.read_text()).get("messages", json.loads(out.read_text()).get("conversations"))) == len(record.get("messages", record.get("conversations")))


def test_strict_json_arguments_and_sharegpt_value_type(tmp_path):
    record = conversation()
    record["messages"][1]["tool_calls"][0]["function"]["arguments"] = '{"number":NaN}'
    assert findings(tmp_path, record, "tool-calls")
    record = sharegpt()
    record["conversations"][1]["value"] = {"name": "lookup", "arguments": {}}
    assert findings(tmp_path, record, "tool-calls")


def test_real_tokenizer_still_discloses_tool_template_estimate():
    from parallelogram.core.tokenization import TokenCounter
    rule = ContextWindowRule()
    rule._counter = TokenCounter(encode_len=len, method="test", exact=True)
    issues = list(rule.check_record(conversation(), 1))
    assert any("tool serialization is approximate" in issue.message for issue in issues)
    assert all(issue.severity.value == "info" for issue in issues)


def test_sharegpt_unchanged_tool_strings_survive_other_repairs(tmp_path):
    record = sharegpt()
    record["tools"] = ' [ { "name" : "lookup", "parameters" : {} } ] '
    record["conversations"][1]["value"] = ' { "name" : "lookup", "arguments" : { "x" : 1 } } '
    record["conversations"][3]["value"] = "\ufeffAnswer"
    path = write(tmp_path, record)
    out = tmp_path / "out.jsonl"
    assert fix(path, output=out).exit_code == 0
    fixed = json.loads(out.read_text())
    assert fixed["tools"] == record["tools"]
    assert fixed["conversations"][1] == record["conversations"][1]
    assert fixed["conversations"][3]["value"] == "Answer"


def test_sharegpt_metadata_does_not_restore_secret_definitions(tmp_path):
    record = sharegpt()
    secret = "sk-" + "AbCdEf0123456789" * 3
    record["tools"] = json.dumps([{"name": "lookup", "description": secret, "parameters": {}}])
    out = tmp_path / "out.jsonl"
    assert fix(write(tmp_path, record), output=out).exit_code == 0
    assert secret not in out.read_text()
    assert check(out).exit_code == 0
def ends_on_call(kind):
    record = conversation()
    if kind == "tool_calls":
        del record["messages"][2:]
    elif kind == "legacy":
        record = {"functions": [{"name": "lookup", "parameters": {"type": "object"}}],
                  "messages": [{"role": "user", "content": "Find A."},
                               {"role": "assistant", "function_call": {"name": "lookup", "arguments": '{"place":"A"}'}}]}
    else:
        record = sharegpt()
        del record["conversations"][2:]
    return record


@pytest.mark.parametrize("kind", ["tool_calls", "legacy", "sharegpt"])
def test_conversation_may_end_on_the_call(tmp_path, kind):
    """OpenAI's own function-calling examples end on the assistant's call: that call is the target."""
    path = write(tmp_path, ends_on_call(kind))
    result = check(path)
    assert result.exit_code == 0, result.to_dict()["issues"]
    out = tmp_path / "out.jsonl"
    cli = CliRunner().invoke(app, ["check", str(path), "--fix", "--output", str(out)])
    assert cli.exit_code == 0, cli.output
    assert json.loads(out.read_text()) == json.loads(path.read_text())


def test_unanswered_call_before_the_final_turn_is_still_an_error(tmp_path):
    record = conversation()
    record["messages"][2:] = [{"role": "tool", "tool_call_id": "a", "content": "A is cold."},
                              {"role": "assistant", "content": "Only A answered."}]
    assert any("never answered" in i["message"] for i in findings(tmp_path, record, "roles"))


def trl_record():
    """The tool-calling example from TRL's dataset-format docs, as written there."""
    schema = {"type": "function", "function": {
        "name": "control_light", "description": "Controls the lights in a room.",
        "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "state": {"type": "string"}},
                       "required": ["room", "state"]},
        "return": {"type": "string", "description": "str: A message indicating the new state of the lights."}}}
    return {"messages": [
        {"role": "user", "content": "Turn on the living room lights."},
        {"role": "assistant", "tool_calls": [{"type": "function", "function": {
            "name": "control_light", "arguments": {"room": "living room", "state": "on"}}}]},
        {"role": "tool", "name": "control_light", "content": "The lights in the living room are now on."},
        {"role": "assistant", "content": "Done!"}], "tools": [schema]}


@pytest.mark.parametrize("tools_as_string", [False, True])
def test_hugging_face_tool_shape_is_clean_and_round_trips(tmp_path, tools_as_string):
    """TRL, Axolotl and Unsloth render through chat templates that want object arguments
    and id-less calls; older datasets store the tools column as a JSON string."""
    record = trl_record()
    if tools_as_string:
        record["tools"] = json.dumps(record["tools"])
    path = write(tmp_path, record)
    result = check(path)
    assert result.exit_code == 0, result.to_dict()["issues"]
    out = tmp_path / "out.jsonl"
    cli = CliRunner().invoke(app, ["check", str(path), "--fix", "--output", str(out)])
    assert cli.exit_code == 0, cli.output
    assert out.read_bytes() == path.read_bytes()


def test_id_less_parallel_calls_are_answered_in_order(tmp_path):
    record = trl_record()
    call = record["messages"][1]["tool_calls"][0]
    record["messages"][1]["tool_calls"] = [call, deepcopy(call)]
    record["messages"][2:2] = [{"role": "tool", "content": "Kitchen is on."}]
    assert check(write(tmp_path, record)).exit_code == 0


@pytest.mark.parametrize("change,fragment", [
    (lambda r: r["messages"][2].update(name="unknown_tool"), "no matching earlier call"),
    (lambda r: r["messages"].insert(3, {"role": "tool", "name": "control_light", "content": "again"}),
     "no matching earlier call"),
    (lambda r: r["messages"][1]["tool_calls"][0].update(id="x"), "must reference its call with tool_call_id"),
    (lambda r: r["messages"][2].update(tool_call_id="x"), "no matching earlier call id"),
])
def test_hugging_face_tool_shape_errors(tmp_path, change, fragment):
    record = trl_record()
    change(record)
    issues = findings(tmp_path, record, "roles")
    assert any(fragment in i["message"] for i in issues), issues


def test_secrets_in_object_arguments_are_found_and_redacted(tmp_path):
    record = trl_record()
    secret = "sk-" + "AbCdEf0123456789" * 3
    record["messages"][1]["tool_calls"][0]["function"]["arguments"]["token"] = secret
    path = write(tmp_path, record)
    assert findings(tmp_path, record, "secrets")
    out = tmp_path / "out.jsonl"
    assert fix(path, output=out).exit_code == 0
    assert secret not in out.read_text()
    assert check(out).exit_code == 0
