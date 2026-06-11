/* parallelogram.dev — interactive demo
   ────────────────────────────────────────────────────────────────────────
   An in-browser re-implementation of the parallelogram linter, faithful to
   the real Python rules (cli/src/parallelogram/rules/*) and the terminal
   output (cli/src/parallelogram/output/terminal.py).

   It runs in the CLI's *approximate* token-counting mode — exactly what
   `parallelogram check <file>` does when no --tokenizer is passed, because a
   browser has neither tiktoken nor HuggingFace tokenizers. That means the
   context-window check is advisory (warnings, not errors) and a one-time
   "counts are approximate" note is always emitted, just like the real tool.

   Honesty is the whole point: if this demo flags something, the CLI would
   too — and vice-versa. The bundled samples are the project's own fixtures
   (cli/examples/broken.jsonl and clean.jsonl) so the two can't drift.

   v0.4 additions, each mirroring its Python counterpart by file path:
   - ShareGPT normalization     → cli/src/parallelogram/formats/sharegpt.py
   - --fix pipeline (fixDataset) → cli/src/parallelogram/core/fixer.py, with
     per-rule fixes from rules/encoding.py, rules/empty_content.py,
     rules/context_window.py and duplicates.py's keep-first fix_dataset.
   The fix pane (#demo-fix-body) shows the second command's result for the
   same input, in lockstep with every check run.

   Behaviour: auto-types and lints a sample on a loop until the visitor
   interacts (clicks a sample chip, edits the editor, or hits Run). From then
   on they drive; a "auto-demo" control resumes the loop.

   NOTE: mojibake sequences are written as \u escapes (not literal glyphs) so
   matching is exact regardless of editor/encoding handling. Each escape is
   the Windows-1252 reading of the character's UTF-8 bytes — i.e. the actual
   mojibake — mirroring MOJIBAKE_FIXES in rules/encoding.py.
*/
(() => {
  'use strict';

  // ── constants mirrored from the Python source ─────────────────────────
  const VALID_ROLES = ['system', 'user', 'assistant', 'tool'];
  const MAX_SEQ_LEN = 4096;        // ContextWindowRule default (rules/context_window.py)
  const CHARS_PER_TOKEN = 4;       // core/tokenization.py _CHARS_PER_TOKEN
  const PER_MESSAGE_OVERHEAD = 4;  // rules/context_window.py _PER_MESSAGE_OVERHEAD
  const BOM = '﻿';

  // Mojibake markers, same order as rules/encoding.py (order matters: the
  // first hit wins per message, and some markers are prefixes of others).
  const MOJIBAKE_MARKERS = [
    'â€™',  // → '  right single quote
    'â€œ',  // → "  left double quote
    'â€',  // → "  right double quote (trailing C1 control)
    'â€“',  // → –  en dash
    'â€”',  // → —  em dash
    'â€¦',  // → …  ellipsis
    'Ã©',        // → é
    'Ã¨',        // → è
    'Ã¢',        // → â
    'Ã®',        // → î
    'Ã¯',        // → ï
    'Ã´',        // → ô
    'Ã»',        // → û
    'Â ',        // → non-breaking space
  ];

  const FILE = 'data.jsonl';

  // Replacements index-aligned with MOJIBAKE_MARKERS — the right-hand side
  // of MOJIBAKE_FIXES in rules/encoding.py.
  const MOJIBAKE_REPLACEMENTS = [
    '\u2019', '\u201C', '\u201D', '\u2013', '\u2014', '\u2026',
    '\u00E9', '\u00E8', '\u00E2', '\u00EE', '\u00EF', '\u00F4',
    '\u00FB', '\u00A0',
  ];

  // ── ShareGPT normalization (formats/sharegpt.py) ───────────────────────
  // Known `from` values map to canonical roles; anything unrecognized is
  // passed through verbatim so the schema rule flags it as an invalid role
  // rather than us silently coercing it.
  const FROM_TO_ROLE = {
    human: 'user', user: 'user',
    gpt: 'assistant', chatgpt: 'assistant', assistant: 'assistant', bard: 'assistant',
    system: 'system',
    tool: 'tool', function: 'tool', function_call: 'tool',
  };

  function normalizeShareGPT(record) {
    if (!isObj(record)) return record;
    const turns = record.conversations !== undefined
      ? record.conversations : record.conversation;
    if (!Array.isArray(turns)) return record;

    const messages = [];
    if (typeof record.system === 'string' && record.system) {
      messages.push({ role: 'system', content: record.system });
    }
    for (const turn of turns) {
      if (!isObj(turn)) {
        // Let schema flag the malformed turn; preserve it as-is.
        messages.push(turn);
        continue;
      }
      const frm = turn.from;
      const role = Object.prototype.hasOwnProperty.call(FROM_TO_ROLE, frm)
        ? FROM_TO_ROLE[frm]
        : (frm === undefined ? null : frm);
      messages.push({ role, content: turn.value });
    }
    return { messages };
  }

  // ── the approximate token counter (core/tokenization.py) ──────────────
  const APPROX_NOTE =
    'no tokenizer specified — pass --tokenizer <model> ' +
    '(e.g. gpt-4o, Qwen/Qwen2.5-7B) for an exact count';
  const ceilDiv = (n, d) => Math.floor((n + d - 1) / d);
  const countMsgTokens = (role, content) =>
    ceilDiv(`<${role}>\n${content}`.length, CHARS_PER_TOKEN) + PER_MESSAGE_OVERHEAD;

  // ── per-rule checks (each mirrors one rules/*.py file) ─────────────────
  // Every check returns an array of issue objects:
  //   { rule, severity: 'error'|'warning', line, message, detail? }

  function checkSchema(record, line) {
    const out = [];
    if (!isObj(record)) {
      out.push({ rule: 'schema', severity: 'error', line,
        message: 'Record is not a JSON object',
        detail: `Found type: ${jsonType(record)}` });
      return out;
    }
    if (!('messages' in record)) {
      out.push({ rule: 'schema', severity: 'error', line,
        message: "Missing 'messages' field",
        detail: "Every record must have a 'messages' array" });
      return out;
    }
    const messages = record.messages;
    if (!Array.isArray(messages)) {
      out.push({ rule: 'schema', severity: 'error', line,
        message: "'messages' is not a list",
        detail: `Found type: ${jsonType(messages)}` });
      return out;
    }
    if (messages.length === 0) {
      out.push({ rule: 'schema', severity: 'error', line,
        message: "'messages' is empty" });
      return out;
    }
    messages.forEach((msg, i) => {
      if (!isObj(msg)) {
        out.push({ rule: 'schema', severity: 'error', line,
          message: `Message ${i} is not an object`,
          detail: `Found type: ${jsonType(msg)}` });
        return;
      }
      if (!('role' in msg)) {
        out.push({ rule: 'schema', severity: 'error', line,
          message: `Message ${i} missing 'role' field` });
      } else if (!VALID_ROLES.includes(msg.role)) {
        out.push({ rule: 'schema', severity: 'error', line,
          message: `Message ${i} has invalid role: ${pyRepr(msg.role)}`,
          detail: `Valid roles: ['assistant', 'system', 'tool', 'user']` });
      }
      if (!('content' in msg)) {
        out.push({ rule: 'schema', severity: 'error', line,
          message: `Message ${i} missing 'content' field` });
      } else if (typeof msg.content !== 'string') {
        out.push({ rule: 'schema', severity: 'error', line,
          message: `Message ${i} 'content' is not a string`,
          detail: `Found type: ${jsonType(msg.content)}` });
      }
    });
    return out;
  }

  function checkRoles(record, line) {
    const out = [];
    if (!isObj(record)) return out;
    const messages = record.messages;
    if (!Array.isArray(messages) || messages.length === 0) return out;

    const roles = [];
    for (const m of messages) {
      if (isObj(m) && typeof m.role === 'string') roles.push(m.role);
      else return out; // schema rule will report; bail to avoid noise
    }

    roles.forEach((role, i) => {
      if (role === 'system' && i !== 0) {
        out.push({ rule: 'roles', severity: 'error', line,
          message: `'system' message at position ${i}, must be first` });
      }
    });

    const start = roles[0] === 'system' ? 1 : 0;
    const body = roles.slice(start).filter(r => r !== 'tool');

    if (body.length === 0) {
      out.push({ rule: 'roles', severity: 'error', line,
        message: 'No user/assistant turns after system message' });
      return out;
    }
    if (body[0] !== 'user') {
      out.push({ rule: 'roles', severity: 'error', line,
        message: `First non-system role must be 'user', got ${pyRepr(body[0])}` });
      return out;
    }

    let expected = 'user';
    for (let i = 0; i < body.length; i++) {
      const role = body[i];
      if (role === 'system') continue;
      if (role !== expected) {
        out.push({ rule: 'roles', severity: 'error', line,
          message: `Role alternation broken at turn ${i}: expected ${pyRepr(expected)}, got ${pyRepr(role)}` });
        return out;
      }
      expected = expected === 'user' ? 'assistant' : 'user';
    }

    if (body[body.length - 1] !== 'assistant') {
      out.push({ rule: 'roles', severity: 'error', line,
        message: `Conversation must end on 'assistant', ended on ${pyRepr(body[body.length - 1])}`,
        detail: 'Training expects the last turn to be the assistant response' });
    }
    return out;
  }

  function checkEmptyContent(record, line) {
    const out = [];
    if (!isObj(record)) return out;
    const messages = record.messages;
    if (!Array.isArray(messages)) return out;
    messages.forEach((msg, i) => {
      if (!isObj(msg)) return;
      const content = msg.content;
      if (typeof content !== 'string') return;
      if (content.trim() === '') {
        out.push({ rule: 'empty-content', severity: 'error', line,
          message: `Message ${i} (role=${msg.role ?? '?'}) has empty content` });
      }
    });
    return out;
  }

  function checkEncoding(record, line) {
    const out = [];
    if (!isObj(record)) return out;
    const messages = record.messages;
    if (!Array.isArray(messages)) return out;
    messages.forEach((msg, i) => {
      if (!isObj(msg)) return;
      const content = msg.content;
      if (typeof content !== 'string') return;
      if (content.includes(BOM)) {
        out.push({ rule: 'encoding', severity: 'warning', line,
          message: `Message ${i} contains BOM (U+FEFF)` });
      }
      for (const marker of MOJIBAKE_MARKERS) {
        if (content.includes(marker)) {
          out.push({ rule: 'encoding', severity: 'warning', line,
            message: `Message ${i} contains likely mojibake: ${pyRepr(marker)}`,
            detail: 'UTF-8 → latin-1 → UTF-8 round-trip artifact' });
          break; // one mojibake report per message is enough
        }
      }
    });
    return out;
  }

  // context-window: advisory in approximate mode. Returns issues for one
  // record; the global "approximate" note is emitted once by lintDataset.
  function checkContextWindow(record, line) {
    const out = [];
    if (!isObj(record)) return out;
    const messages = record.messages;
    if (!Array.isArray(messages)) return out;
    let total = 0;
    for (const msg of messages) {
      if (!isObj(msg)) continue;
      const content = msg.content;
      const role = msg.role ?? '';
      if (typeof content !== 'string') continue;
      total += countMsgTokens(role, content);
    }
    if (total <= MAX_SEQ_LEN) return out;
    out.push({ rule: 'context-window', severity: 'warning', line,
      message: `Record may exceed max_seq_len: ~${total} > ${MAX_SEQ_LEN} tokens (estimated)`,
      detail: 'Estimated with a length heuristic; over-long records are truncated ' +
              'silently. Pass --tokenizer <model> to confirm.' });
    return out;
  }

  // ── dataset-level orchestration (mirrors core/runner.py) ──────────────
  // Returns { issues, totalRecords, validRecords }. Issues are ordered the
  // way terminal.py prints them: global (no line) first, then by line.
  function lintDataset(text, format = 'openai-chat') {
    const issues = [];
    const dupes = new Map();       // hash → [lineNos]
    const perLineErrors = new Map();
    let total = 0;
    let noteEmitted = false;

    const addError = (line) =>
      perLineErrors.set(line, (perLineErrors.get(line) || 0) + 1);

    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const stripped = lines[i].trim();
      if (stripped === '') continue;          // blank lines aren't records
      const lineNo = i + 1;
      total++;

      let record;
      try {
        record = JSON.parse(stripped);
      } catch (e) {
        issues.push({ rule: 'schema', severity: 'error', line: lineNo,
          message: 'Invalid JSON', detail: parseErrorDetail(stripped) });
        addError(lineNo);
        continue;
      }

      // Formats normalize at the parse boundary (the architecture section's
      // whole point) — every rule below sees the internal message shape.
      if (format === 'sharegpt') record = normalizeShareGPT(record);

      // The approximate-counting note fires once, the first time the
      // context-window rule runs — exactly as the Python rule emits it.
      if (!noteEmitted) {
        noteEmitted = true;
        issues.push({ rule: 'context-window', severity: 'warning', line: null,
          message: 'Context-window counts are approximate', detail: APPROX_NOTE });
      }

      const found = [
        ...checkSchema(record, lineNo),
        ...checkRoles(record, lineNo),
        ...checkEmptyContent(record, lineNo),
        ...checkEncoding(record, lineNo),
        ...checkContextWindow(record, lineNo),
      ];
      for (const issue of found) {
        issues.push(issue);
        if (issue.severity === 'error') addError(lineNo);
      }

      const h = dupHash(record);
      if (h) {
        if (!dupes.has(h)) dupes.set(h, []);
        dupes.get(h).push(lineNo);
      }
    }

    // Cross-record finalisation: duplicates (mirrors DuplicatesRule.finalize).
    for (const lns of dupes.values()) {
      if (lns.length > 1) {
        const first = lns[0];
        for (const dup of lns.slice(1)) {
          issues.push({ rule: 'duplicates', severity: 'error', line: dup,
            message: `Duplicate of line ${first}`,
            detail: `${lns.length} copies total at lines: [${lns.join(', ')}]` });
          addError(dup);
        }
      }
    }

    // valid = parsed records that raised no error (warnings are fine).
    let valid = 0;
    for (let i = 0; i < lines.length; i++) {
      if (lines[i].trim() === '') continue;
      const lineNo = i + 1;
      if (!(perLineErrors.get(lineNo) > 0)) valid++;
    }

    // Stable ordering: globals first, then ascending line number.
    issues.sort((a, b) => {
      const al = a.line === null ? -1 : a.line;
      const bl = b.line === null ? -1 : b.line;
      return al - bl;
    });

    return { issues, totalRecords: total, validRecords: valid };
  }

  // Normalised-whitespace content hash (mirrors DuplicatesRule._hash).
  function dupHash(record) {
    if (!isObj(record)) return '';
    const messages = record.messages;
    if (!Array.isArray(messages)) return '';
    const normalized = messages.filter(isObj).map(m => ({
      role: m.role ?? null,
      content: typeof m.content === 'string' ? m.content.trim() : (m.content ?? null),
    }));
    return JSON.stringify(normalized);
  }

  // ── --fix pipeline (mirrors core/fixer.py, mechanical tier) ────────────
  // Stage 1: per-record fixes — records with any unfixable-rule issue are
  // dropped outright (partial fixes aren't worth it). Stage 2: cross-record
  // dedupe, keep-first. Stage 3: re-validate — anything still erroring is
  // unfixable at the mechanical tier and is dropped, never papered over.
  const UNFIXABLE_RULES = new Set(['schema', 'roles']);

  // rules/encoding.py fix_record: strip BOMs, replace known mojibake.
  function fixEncoding(record) {
    const messages = record.messages.map((msg) => {
      if (!isObj(msg) || typeof msg.content !== 'string') return msg;
      let fixed = msg.content.split(BOM).join('');
      MOJIBAKE_MARKERS.forEach((bad, i) => {
        if (fixed.includes(bad)) fixed = fixed.split(bad).join(MOJIBAKE_REPLACEMENTS[i]);
      });
      return fixed === msg.content ? msg : { ...msg, content: fixed };
    });
    return { ...record, messages };
  }

  // rules/empty_content.py fix_record: drop whitespace-only turns; null
  // (drop the record) when nothing is left. Re-validation catches whatever
  // structure the strip couldn't salvage.
  function fixEmptyContent(record) {
    const kept = record.messages.filter((msg) =>
      !(isObj(msg) && typeof msg.content === 'string' && msg.content.trim() === ''));
    if (!kept.length) return null;
    return { ...record, messages: kept };
  }

  // rules/context_window.py fix_record: shrink the longest user message in
  // 20% steps; never touch system or assistant turns; null when it can't fit.
  function fixContextWindow(record) {
    const msgs = record.messages.map(m => (isObj(m) ? { ...m } : m));
    for (let iter = 0; iter < 20; iter++) {
      let total = 0;
      for (const m of msgs) {
        if (isObj(m) && typeof m.content === 'string') {
          total += countMsgTokens(m.role ?? '', m.content);
        }
      }
      if (total <= MAX_SEQ_LEN) return { ...record, messages: msgs };
      const userIdxs = msgs
        .map((m, i) => i)
        .filter(i => isObj(msgs[i]) && msgs[i].role === 'user'
          && typeof msgs[i].content === 'string');
      if (!userIdxs.length) return null;
      const longest = userIdxs.reduce((a, b) =>
        (msgs[b].content.length > msgs[a].content.length ? b : a));
      const current = msgs[longest].content;
      if (current.length <= 1) return null;
      msgs[longest] = { ...msgs[longest],
        content: current.slice(0, Math.max(1, Math.floor(current.length * 0.8))) };
    }
    return null; // couldn't fit after 20 iterations — give up (fixer.py cap)
  }

  const RULE_FIXES = {
    'encoding': fixEncoding,
    'empty-content': fixEmptyContent,
    'context-window': fixContextWindow,
  };

  // Returns the fix report the CLI summarizes: totals per disposition plus
  // per-rule fix counts and the dropped lines with their reasons.
  function fixDataset(text, format = 'openai-chat') {
    const lines = text.split('\n');
    const records = [];
    let unparseable = 0;
    for (let i = 0; i < lines.length; i++) {
      const stripped = lines[i].trim();
      if (stripped === '') continue;
      const lineNo = i + 1;
      let record;
      try { record = JSON.parse(stripped); } catch { unparseable++; continue; }
      if (format === 'sharegpt') record = normalizeShareGPT(record);
      records.push([lineNo, record]);
    }

    const issuesFor = (record, lineNo) => [
      ...checkSchema(record, lineNo),
      ...checkRoles(record, lineNo),
      ...checkEmptyContent(record, lineNo),
      ...checkEncoding(record, lineNo),
      ...checkContextWindow(record, lineNo),
    ];

    const outcomes = new Map(); // lineNo → {disposition, reason?}
    const fixesByRule = {};
    const stage1 = [];

    for (const [lineNo, record] of records) {
      const issues = issuesFor(record, lineNo);
      if (!issues.length) {
        stage1.push([lineNo, record]);
        outcomes.set(lineNo, { disposition: 'unchanged' });
        continue;
      }
      const unfixable = [...new Set(
        issues.filter(i => UNFIXABLE_RULES.has(i.rule)).map(i => i.rule))];
      if (unfixable.length) {
        outcomes.set(lineNo, { disposition: 'dropped',
          reason: `${unfixable.join(', ')} (unfixable)` });
        continue;
      }
      let current = record;
      const applied = [];
      let failed = false;
      for (const issue of issues) {
        const fix = RULE_FIXES[issue.rule];
        if (!fix) continue;
        const next = fix(current);
        if (next === null) { failed = true; break; }
        if (next !== current && !applied.includes(issue.rule)) applied.push(issue.rule);
        current = next;
      }
      if (failed) {
        outcomes.set(lineNo, { disposition: 'dropped',
          reason: `${issues[0].rule} (unfixable)` });
        continue;
      }
      stage1.push([lineNo, current]);
      for (const rid of applied) fixesByRule[rid] = (fixesByRule[rid] || 0) + 1;
      outcomes.set(lineNo,
        { disposition: applied.length ? 'fixed' : 'unchanged' });
    }

    // Stage 2: duplicates — keep first occurrence (duplicates.py fix_dataset)
    const stage2 = [];
    const seen = new Set();
    for (const [lineNo, record] of stage1) {
      const h = dupHash(record);
      if (h && seen.has(h)) {
        fixesByRule.duplicates = (fixesByRule.duplicates || 0) + 1;
        outcomes.set(lineNo, { disposition: 'dropped',
          reason: 'duplicates (dropped as fix)' });
        continue;
      }
      if (h) seen.add(h);
      stage2.push([lineNo, record]);
    }

    // Stage 3: re-validate. In the browser's estimate mode context-window
    // re-checks are warnings, so — exactly like the CLI without --tokenizer —
    // they never drop a record here.
    const emitted = [];
    for (const [lineNo, record] of stage2) {
      const stillBad = [...new Set([
        ...checkSchema(record, lineNo),
        ...checkRoles(record, lineNo),
        ...checkEmptyContent(record, lineNo),
      ].filter(i => i.severity === 'error').map(i => i.rule))];
      if (stillBad.length) {
        outcomes.set(lineNo, { disposition: 'dropped',
          reason: `${stillBad[0]} (unfixable)` });
        continue;
      }
      emitted.push(lineNo);
    }

    const report = {
      total: records.length + unparseable,
      unchanged: 0, fixed: 0, dropped: 0,
      unparseable,
      emitted: emitted.length,
      fixesByRule,
      droppedLines: [],
    };
    for (const [lineNo, o] of [...outcomes.entries()].sort((a, b) => a[0] - b[0])) {
      if (o.disposition === 'unchanged') report.unchanged++;
      else if (o.disposition === 'fixed') report.fixed++;
      else if (o.disposition === 'dropped') {
        report.dropped++;
        report.droppedLines.push([lineNo, o.reason]);
      }
    }
    return report;
  }

  // ── helpers ───────────────────────────────────────────────────────────
  function isObj(v) { return typeof v === 'object' && v !== null && !Array.isArray(v); }
  function jsonType(v) {
    if (v === null) return 'NoneType';
    if (Array.isArray(v)) return 'list';
    switch (typeof v) {
      case 'object':  return 'dict';
      case 'string':  return 'str';
      case 'boolean': return 'bool';
      case 'number':  return Number.isInteger(v) ? 'int' : 'float';
      default:        return typeof v;
    }
  }
  // Python-style repr for the short strings we surface in messages.
  function pyRepr(s) {
    if (typeof s !== 'string') return String(s);
    return "'" + s.replace(/\\/g, '\\\\').replace(/'/g, "\\'") + "'";
  }
  function parseErrorDetail(stripped) {
    // Approximate CPython's json error wording for the common case the demo
    // shows. Exhaustive cross-runtime parity isn't possible; the column of
    // the first non-whitespace char matches the usual "Expecting value".
    const col = (stripped.length - stripped.trimStart().length) + 1;
    return `Expecting value at column ${col}`;
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  }

  // Expose the pure linter for browser debugging and parity tests. Harmless
  // in production; lets `window.__pgLint(text)` reproduce a run programmatically.
  if (typeof window !== 'undefined') {
    window.__pgLint = lintDataset;
    window.__pgFix = fixDataset;
  }

  // ── terminal rendering (mirrors output/terminal.py) ───────────────────
  const term = document.getElementById('terminal-body');
  if (!term) return;

  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  const fixTerm = document.getElementById('demo-fix-body');

  function appendTo(targetEl, cls, html) {
    const el = document.createElement('span');
    el.className = `t-line ${cls}`;
    el.innerHTML = html;
    targetEl.appendChild(el);
    return el;
  }

  function appendLine(cls, html) { return appendTo(term, cls, html); }

  function renderIssue(issue) {
    const isErr = issue.severity === 'error';
    const cls = isErr ? 'err' : 'warn';
    const mark = isErr ? '✗' : '!';       // ✗ / !
    const loc = issue.line === null ? '(global)' : `${FILE}:${issue.line}`;
    appendLine(cls,
      `  <span class="${cls}">${mark}</span> ` +
      `<span class="lno">${escapeHtml(loc)}</span> ` +
      `<span class="rid">[${escapeHtml(issue.rule)}]</span> ` +
      `${escapeHtml(issue.message)}`);
    if (issue.detail) {
      appendLine('detail', `      <span class="dim">→ ${escapeHtml(issue.detail)}</span>`);
    }
  }

  function renderSummary(result) {
    const nErr = result.issues.filter(i => i.severity === 'error').length;
    const nWarn = result.issues.filter(i => i.severity === 'warning').length;

    if (result.issues.length === 0) {
      appendLine('panel ok-panel', panelHtml('green',
        `<span class="ok">✓</span> Clean: ${result.totalRecords} records validated`));
      return;
    }
    appendLine('blank', '&nbsp;');
    const border = nErr ? 'red' : 'amber';
    const body =
      `<span class="sum-strong">${result.totalRecords} records</span>  ` +
      `<span class="ok">${result.validRecords} clean</span>  ` +
      `<span class="${nErr ? 'err' : 'dim'}">${nErr} errors</span>  ` +
      `<span class="${nWarn ? 'warn' : 'dim'}">${nWarn} warnings</span>`;
    appendLine('panel', panelHtml(border, body));
  }

  function panelHtml(border, inner) {
    return `<span class="t-panel border-${border}"><span class="t-panel-title">parallelogram</span>${inner}</span>`;
  }

  function clearTerm() {
    term.innerHTML = '';
    if (fixTerm) fixTerm.innerHTML = '';
  }

  // The fix pane: what `--fix --output clean.jsonl` would do to the same
  // input, painted instantly after each check run (it's a second command's
  // result, not a replay). Format matches cli.py's fix summary.
  function renderFixPane(text) {
    if (!fixTerm) return;
    fixTerm.innerHTML = '';
    const fr = fixDataset(text, format);
    appendTo(fixTerm, 'cmd', `<span class="pr">$</span> ${escapeHtml(cmdFor('fix'))}`);

    const fixedRules = Object.keys(fr.fixesByRule);
    if (fixedRules.length) appendTo(fixTerm, 'blank', '&nbsp;');
    for (const rid of fixedRules) {
      const n = fr.fixesByRule[rid];
      appendTo(fixTerm, '',
        `  <span class="ok">✓</span> <span class="rid">${escapeHtml(rid)}</span> · ${n} fix${n === 1 ? '' : 'es'}`);
    }

    if (fr.droppedLines.length) {
      appendTo(fixTerm, 'blank', '&nbsp;');
      appendTo(fixTerm, '', `  <span class="err">✗</span> dropped:`);
      for (const [lineNo, reason] of fr.droppedLines) {
        appendTo(fixTerm, 'detail',
          `      <span class="dim"><span class="lno">${FILE}:${lineNo}</span> → ${escapeHtml(reason)}</span>`);
      }
    }

    appendTo(fixTerm, 'blank', '&nbsp;');
    const border = (fr.dropped || fr.unparseable) ? 'amber' : 'green';
    appendTo(fixTerm, 'panel', panelHtml(border,
      `<span class="sum-strong">${fr.total} records</span>  ` +
      `<span class="dim">${fr.unchanged} unchanged</span>  ` +
      `<span class="${fr.fixed ? 'ok' : 'dim'}">${fr.fixed} fixed</span>  ` +
      `<span class="${fr.dropped ? 'err' : 'dim'}">${fr.dropped} dropped</span>  ` +
      `<span class="${fr.unparseable ? 'err' : 'dim'}">${fr.unparseable} unparseable</span>`));
    appendTo(fixTerm, '',
      `  <span class="ok">→ clean.jsonl written · ${fr.emitted} record${fr.emitted === 1 ? '' : 's'}</span>`);
    fixTerm.scrollTop = 0;
  }

  async function typeCmd(text) {
    const el = appendLine('cmd', '<span class="pr">$</span> ');
    const cur = document.createElement('span');
    cur.className = 'cur';
    el.appendChild(cur);
    if (reduceMotion) {
      cur.insertAdjacentText('beforebegin', text);
    } else {
      for (const ch of text) {
        cur.insertAdjacentText('beforebegin', ch);
        await sleep(16 + Math.random() * 24);
      }
    }
    await sleep(reduceMotion ? 0 : 160);
    cur.remove();
  }

  // Render a full run for `text` under command `cmd`. When `animate`, the
  // command is typed and issues stream in; otherwise it paints instantly.
  async function runOnce(cmd, text, animate) {
    clearTerm();
    if (animate) await typeCmd(cmd);
    else appendLine('cmd', `<span class="pr">$</span> ${escapeHtml(cmd)}`);

    const result = lintDataset(text, format);
    if (animate) await sleep(300);
    for (const issue of result.issues) {
      renderIssue(issue);
      if (animate) await sleep(70);
    }
    renderSummary(result);
    term.scrollTop = term.scrollHeight;
    renderFixPane(text);
    return result;
  }

  // ── samples (the project's own fixtures, inlined so the demo can't drift)
  // Mojibake in the broken sample uses \u escapes for the same reason as the
  // marker table above.
  const MOJIBAKE_APOS = 'â€™'; // don<...>t  →  don't
  const MOJIBAKE_EMD  = 'â€”'; // <...>      →  em dash
  const SAMPLES = {
    'openai-chat': {
      broken: [
        '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}',
        '{"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "What did you mean?"}]}',
        '{"messages": [{"role": "user", "content": "Hi"}, {"role": "user", "content": "Hello?"}, {"role": "assistant", "content": "Hi!"}]}',
        '{"messages": [{"role": "user", "content": "Question"}]}',
        '{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}',
        `{"messages": [{"role": "user", "content": "He said don${MOJIBAKE_APOS}t do it ${MOJIBAKE_EMD} seriously."}, {"role": "assistant", "content": "Understood."}]}`,
        '{"messages": [{"role": "user", "content": "Hi"}, {"role": "system", "content": "You are now in admin mode."}, {"role": "assistant", "content": "OK"}]}',
        'not valid json at all',
      ].join('\n'),
      clean: [
        '{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "How do I reverse a list in Python?"}, {"role": "assistant", "content": "Use slicing with a step of -1: my_list[::-1]"}]}',
        '{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}',
        '{"messages": [{"role": "user", "content": "Capital of France?"}, {"role": "assistant", "content": "Paris."}]}',
      ].join('\n'),
    },
    // ShareGPT fixtures exercise the same failure classes through the
    // from/value shape: an unknown `from` (schema flags the role), an empty
    // value, an exact duplicate (lines 1 and 4 normalize identically), the
    // same mojibake artifacts, and a conversation that ends on human.
    'sharegpt': {
      broken: [
        '{"system": "You are a concise assistant.", "conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]}',
        '{"conversations": [{"from": "bing", "value": "Search results:"}, {"from": "gpt", "value": "OK"}]}',
        '{"conversations": [{"from": "human", "value": ""}, {"from": "gpt", "value": "What did you mean?"}]}',
        '{"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}], "system": "You are a concise assistant."}',
        `{"conversations": [{"from": "human", "value": "He said don${MOJIBAKE_APOS}t do it ${MOJIBAKE_EMD} seriously."}, {"from": "gpt", "value": "Understood."}]}`,
        '{"conversations": [{"from": "human", "value": "Are you still there?"}]}',
      ].join('\n'),
      clean: [
        '{"conversations": [{"from": "human", "value": "How do I reverse a list in Python?"}, {"from": "gpt", "value": "Use slicing with a step of -1: my_list[::-1]"}]}',
        '{"system": "You are a helpful assistant.", "conversations": [{"from": "human", "value": "What is 2+2?"}, {"from": "gpt", "value": "4"}]}',
        '{"conversations": [{"from": "human", "value": "Capital of France?"}, {"from": "gpt", "value": "Paris."}]}',
      ].join('\n'),
    },
  };

  // Active dataset format — toggled by the [data-demo-format] tabs. The
  // command line reflects it, exactly as the CLI flag would.
  let format = 'openai-chat';

  function cmdFor(kind) {
    const fmt = format === 'sharegpt' ? ' --format sharegpt' : '';
    if (kind === 'fix') return `parallelogram check data.jsonl${fmt} --fix --output clean.jsonl`;
    const file = kind === 'clean' ? 'clean.jsonl' : 'data.jsonl';
    return `parallelogram check ${file}${fmt}`;
  }

  // ── controller: auto-loop until the visitor takes over ────────────────
  const editor     = document.getElementById('demo-editor');
  const editorWrap  = document.getElementById('demo-editor-wrap');
  const runBtn     = document.getElementById('demo-run');
  const resumeBtn  = document.getElementById('demo-resume');
  const editToggle = document.getElementById('demo-edit-toggle');
  const chips      = Array.from(document.querySelectorAll('[data-demo-sample]'));

  let mode = 'auto';     // 'auto' | 'manual'
  let loopToken = 0;     // bumps to cancel an in-flight auto-loop

  function setActiveChip(name) {
    chips.forEach(c => c.classList.toggle('is-active', c.dataset.demoSample === name));
  }

  async function autoLoop() {
    const myToken = ++loopToken;
    mode = 'auto';
    resumeBtn?.setAttribute('hidden', '');
    const order = ['broken', 'clean'];
    let i = 0;
    while (myToken === loopToken) {
      const name = order[i % order.length];
      setActiveChip(name);
      if (editor) editor.value = SAMPLES[format][name];
      await runOnce(cmdFor(name), SAMPLES[format][name], !reduceMotion);
      if (myToken !== loopToken) break;
      await sleep(2600);
      i++;
      if (reduceMotion && i >= order.length) break; // one pass is enough w/o motion
    }
  }

  function takeOver() {
    loopToken++;          // cancel any running auto-loop
    mode = 'manual';
    resumeBtn?.removeAttribute('hidden');
  }

  async function runManual(text, cmd, chipName) {
    takeOver();
    setActiveChip(chipName || null);
    await runOnce(cmd || cmdFor('custom'), text, !reduceMotion);
  }

  // sample chips → load + run that fixture, hand control to the user
  chips.forEach(chip => {
    chip.addEventListener('click', () => {
      const name = chip.dataset.demoSample;
      const sample = SAMPLES[format][name];
      const text = sample || (editor ? editor.value : '');
      if (editor && sample) editor.value = sample;
      runManual(text, sample ? cmdFor(name) : cmdFor('custom'), name);
    });
  });

  // format tabs → switch fixtures and the --format flag; an auto-running
  // demo restarts in the new format, a manual one reruns the broken fixture
  const fmtTabs = Array.from(document.querySelectorAll('[data-demo-format]'));
  fmtTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const f = tab.dataset.demoFormat;
      if (!SAMPLES[f] || f === format) return;
      format = f;
      fmtTabs.forEach(t =>
        t.setAttribute('aria-pressed', t === tab ? 'true' : 'false'));
      if (mode === 'auto') {
        autoLoop();
      } else {
        if (editor) editor.value = SAMPLES[format].broken;
        runManual(SAMPLES[format].broken, cmdFor('broken'), 'broken');
      }
    });
  });

  // editing the editor hands control over immediately
  editor?.addEventListener('input', () => { if (mode === 'auto') takeOver(); });

  // Run button → lint whatever is in the editor
  runBtn?.addEventListener('click', () => {
    const text = editor ? editor.value : '';
    runManual(text, cmdFor('custom'), null);
  });

  // resume the auto demo
  resumeBtn?.addEventListener('click', () => { autoLoop(); });

  // "edit" toggle reveals the editor (collapsed by default to stay clean)
  editToggle?.addEventListener('click', () => {
    const open = editorWrap?.toggleAttribute('data-open');
    editToggle.setAttribute('aria-expanded', open ? 'true' : 'false');
    if (open && editor) {
      if (!editor.value.trim()) editor.value = SAMPLES[format].broken;
      editor.focus();
    }
  });

  // ── start only when scrolled into view (saves CPU) ────────────────────
  let started = false;
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (e.isIntersecting && !started) {
        started = true;
        autoLoop();
        io.disconnect();
      }
    }
  }, { threshold: 0.2 });
  io.observe(term);
})();
