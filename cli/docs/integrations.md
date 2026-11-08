# Integrations (1.6)

Run validation locally before training, after dataset edits, and in CI. Install
`parallelogram` in the environment the client launches; no API key or MCP extra
is needed. If the executable is not on the client's PATH, use its absolute path
in the configuration. Use absolute dataset paths when the client's working
directory is uncertain. The MCP server's write root defaults to its working
directory; see [agent tools and boundaries](agents.md#mcp-tools).

```bash
parallelogram integrate --list
parallelogram integrate cursor --agents-md --dry-run
parallelogram integrate cursor --agents-md
```

Run setup from the project root. JSON edits preserve other servers and keys;
invalid JSON or an incompatible object shape exits 2 without overwriting it.
Repeated setup leaves already configured files unchanged. `--dry-run` prints
proposed files without writing. No user-level client configuration is modified.
Symlinks escaping the project are refused, as is using the home directory itself
as the project root.

`--agents-md` adds the [agent instruction block](agents.md#paste-into-agentsmd-or-claudemd)
to `AGENTS.md` (`CLAUDE.md` for Claude Code), replacing the text between
`<!-- parallelogram:start -->` and `<!-- parallelogram:end -->` on subsequent runs.
Text outside the markers stays intact. This explicitly requested project file
is also written for clients whose MCP setup is otherwise manual. Malformed or
duplicate markers are refused before any configuration is written.

## Claude Code

Setup: `parallelogram integrate claude-code --agents-md`.
Writes project `.mcp.json` and, with that flag, `CLAUDE.md`.
Manual alternative: merge this into `.mcp.json`, or run
`claude mcp add --scope project parallelogram -- parallelogram mcp`:

```json
{"mcpServers":{"parallelogram":{"command":"parallelogram","args":["mcp"]}}}
```

Project scope and stdio fields are documented in
[Claude Code MCP](https://code.claude.com/docs/en/mcp).

## Cursor

Setup: `parallelogram integrate cursor --agents-md`.
Writes `.cursor/mcp.json`. Manual alternative: merge this into the same file:

```json
{"mcpServers":{"parallelogram":{"command":"parallelogram","args":["mcp"]}}}
```

See [Cursor's project MCP configuration](https://prod.cursor.com/docs/mcp).

## VS Code / Copilot agent mode

Setup: `parallelogram integrate vscode --agents-md`.
Writes `.vscode/mcp.json`. Manual alternative: merge this into the same file:

```json
{"servers":{"parallelogram":{"type":"stdio","command":"parallelogram","args":["mcp"]}}}
```

This uses VS Code's `servers` object and explicit stdio transport. See the
[VS Code configuration reference](https://code.visualstudio.com/docs/agents/reference/mcp-configuration).

## Gemini CLI

Setup: `parallelogram integrate gemini --agents-md`.
Writes project `.gemini/settings.json`. Manual alternative: merge this there:

```json
{"mcpServers":{"parallelogram":{"command":"parallelogram","args":["mcp"]}}}
```

See [Gemini MCP servers](https://geminicli.com/docs/tools/mcp-server/) and
[project settings scope](https://geminicli.com/docs/reference/configuration/).
The optional instruction file is `AGENTS.md`; if your Gemini setup reads only
`GEMINI.md`, import or copy that block into it.

## Codex

Setup: `parallelogram integrate codex` prints instructions and writes nothing.
Manual alternative: run `codex mcp add parallelogram -- parallelogram mcp`, or
merge into `~/.codex/config.toml`:

```toml
[mcp_servers.parallelogram]
command = "parallelogram"
args = ["mcp"]
```

Current [Codex MCP docs](https://developers.openai.com/codex/mcp) also support
project `.codex/config.toml`. Parallelogram 1.6 deliberately keeps Codex setup
manual; `--agents-md` writes only the requested project instructions.

## Windsurf

Setup: `parallelogram integrate windsurf` prints instructions and writes nothing.
Manual alternative for Windsurf installations: open Settings > Manage MCPs >
View raw config, usually `~/.codeium/windsurf/mcp_config.json`, and merge:

```json
{"mcpServers":{"parallelogram":{"command":"parallelogram","args":["mcp"]}}}
```

The Windsurf documentation URL now redirects to
[Devin Desktop customizations](https://docs.devin.ai/desktop/devin-local/customizations),
which uses different configuration locations. Use the file opened by your
installed client's UI. The Windsurf stdio shape is confirmed by
[Microsoft's Windsurf setup](https://learn.microsoft.com/en-us/azure/developer/azure-mcp-server/get-started/tools/windsurf);
the legacy file path and shape are also documented by
[Grafana's Windsurf integration](https://grafana.com/docs/grafana/latest/ai-tools/mcp/clients/windsurf/).
This command targets Windsurf, and does not migrate settings to Devin Desktop.

## Cline

Setup: `parallelogram integrate cline` prints instructions and writes nothing.
Manual alternative: Cline > MCP Servers > Configure > Configure MCP Servers
opens the extension's `cline_mcp_settings.json`. Merge:

```json
{"mcpServers":{"parallelogram":{"command":"parallelogram","args":["mcp"]}}}
```

Cline CLI instead uses `~/.cline/data/settings/cline_mcp_settings.json`
(or `CLINE_MCP_SETTINGS_PATH`). The UI avoids guessing an OS-specific extension
storage directory. See [Cline's MCP configuration](https://docs.cline.bot/mcp/mcp-overview).

## Zed

Setup: `parallelogram integrate zed` prints instructions and writes nothing.
Manual alternative: use Zed: Open Settings File, normally
`~/.config/zed/settings.json` on macOS/Linux, and merge:

```json
{"context_servers":{"parallelogram":{"command":"parallelogram","args":["mcp"]}}}
```

Zed uses `context_servers` with `command` and `args` directly on the server.
See [Zed MCP configuration](https://zed.dev/docs/ai/mcp).

## Tool-calling datasets

`--format auto` recognizes OpenAI chat and LLaMA-Factory ShareGPT. OpenAI records
may contain `tools`, `parallel_tool_calls`, assistant `tool_calls` with unique
ids, and `tool` replies linked by `tool_call_id`. Arguments must be JSON-encoded
objects. Tool-only assistant content may be absent, null or empty; tool reply
content must be a nonempty string. Replies to parallel calls may arrive in any
order. Every call needs one reply before the next assistant/user turn, and the
conversation must finish on an assistant turn. Legacy `functions`,
`function_call`, and `function` replies are also supported.

ShareGPT `function_call` values contain a JSON object with `name` and `arguments`;
`observation` supplies its reply and `tools` is a JSON string. Parsing generates
deterministic internal call ids and converts bare function definitions into
OpenAI tool wrappers. Fix output restores the ShareGPT representation. A clean
file passed through `--fix --output` is preserved byte-for-byte, including line
endings and whitespace. Broken exchanges are dropped as whole records; the
fixer never deletes or reorders tool messages to repair alternation.

The `tool-calls` rule checks definition and call structure, argument JSON,
function names against declared tools, and record-wide id uniqueness. Its
messages identify the physical line and zero-based message/call index without
printing argument values. `secrets` still scans arguments and tool definitions.
Token estimates include definitions and calls, and duplicate hashing includes
tool data. Tokenization plus template overhead remains an estimate of the
trainer's actual serialized tool template; use the same trainer/template to
confirm exact context limits.

```bash
parallelogram check tools.jsonl sharegpt_tools.jsonl --json
parallelogram check tools.jsonl --fix --output clean.jsonl
```

## pre-commit

The repository provides the Python hook `parallelogram`. Add to your project's
`.pre-commit-config.yaml`, pinning the integration commit until the 1.6 release
is tagged (then use `cli-v1.6.0`):

```yaml
repos:
  - repo: https://github.com/Thatayotlhe04/Parallelogram
    rev: feat/integrations # replace with the reviewed commit SHA or release tag
    hooks:
      - id: parallelogram
```

```bash
pip install pre-commit
pre-commit install
pre-commit run parallelogram --all-files
```

The hook runs `parallelogram check` on matching `\.jsonl$` files, forwarding all
filenames; it never fixes data automatically. Validation is offline; initial
hook installation may download Python dependencies. Every file is checked
independently (duplicates are checked within each file), and the highest exit
code wins: 0 clean, 1 warnings, 2 errors, 3 operational failure. pre-commit blocks
on any nonzero exit code. `--json` emits an array only for multiple inputs;
single-file output remains the schema-version-1 object. Multi-file operational
failures occupy an array entry with `schema_version`, `file`, `exit_code`, and
`error`. `--output` requires one input so files cannot overwrite each other.

For hook development from a checkout:

```bash
pre-commit try-repo . parallelogram --files cli/examples/tools_clean.jsonl cli/examples/sharegpt_tools_clean.jsonl
```

## GitHub Action

The existing repository Action installs the CLI, checks a dataset, and appends
a health report to the job summary. Use these steps in your own workflow;
pin the Action to the reviewed integration commit until `cli-v1.6.0` exists:

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: Thatayotlhe04/Parallelogram@feat/integrations
    with:
      path: data/train.jsonl
      format: auto
      max-seq-len: '4096'
      fail-on: error
```

Use `fail-on: warning` for a stricter gate. Optional `eval` and `baseline`
inputs check train/eval leakage and quality regressions. The integration command
only sets up MCP; it does not create or modify workflows.

## Training frameworks

Validate a local JSONL export before the framework applies its chat template.
Parallelogram reads records, not a remote dataset name, Parquet, or a whole JSON
array. Custom field mappings must be exported to a supported shape first. This
is data preflight, not a model/template compatibility check.

| Framework | Dataset shapes Parallelogram reads | One-line preflight |
|---|---|---|
| [TRL](https://huggingface.co/docs/trl/dataset_formats) | Conversational `messages`; string/conversational `prompt` + `completion`; explicit or implicit prompt `chosen`/`rejected` preference pairs. | `parallelogram check train.jsonl --format auto --max-seq-len 4096` |
| [Axolotl](https://docs.axolotl.ai/docs/dataset-formats/) | OpenAI chat, ShareGPT, Alpaca `instruction`/`input`/`output`, and supported preference exports. | `parallelogram check train.jsonl --format auto --max-seq-len 4096` |
| [Unsloth](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/datasets-guide) | ShareGPT, OpenAI chat, and Alpaca, before rendering to a `text` column. | `parallelogram check train.jsonl --format auto` |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) | Alpaca with `system`/`history`; ShareGPT with `function_call`/`observation` and JSON-string `tools`. | `parallelogram check train.jsonl --format auto` |
| [OpenAI fine-tuning](https://developers.openai.com/api/docs/guides/supervised-fine-tuning) | Text chat `messages`, including the tool-calling format described above. | `parallelogram check train.jsonl --format openai-chat` |

TRL/Hugging Face tool examples may use object-valued arguments and omit call
ids. Those are distinct from the OpenAI wire shape supported here: export
JSON-string arguments and explicit call/reply ids before checking that shape.
Plain `text`-only language-modeling rows, multimodal training schemas, and
framework-specific ranking layouts outside the documented preference format
are not supported. The OpenAI preflight does not check account/model eligibility
or upload-level minimum example counts.
