// parallelogram.dev/workbench
// The page is a view over the engine in engine-worker.js. It holds no rule, no fix and no
// verdict of its own: every number on screen came back from the Python package.
import { Field, CLEAN, WARN, ERROR, GONE } from "/workbench/field.js";

const MODE = document.documentElement.dataset.mode;        // "bench" (sealed) or "operator"
const $ = (id) => document.getElementById(id);
const fmt = (n) => Number(n).toLocaleString("en-US");
const plural = (n, one, many = one + "s") => `${fmt(n)} ${n === 1 ? one : many}`;
const pad2 = (n) => String(n).padStart(2, "0");

// replaceChildren() turns a null argument into the text "null"; conditional children are
// everywhere here, so every container is filled through this instead.
export function fill(node, ...children) {
  node.replaceChildren(...children.flat().filter((child) => child != null && child !== false));
  return node;
}

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value == null) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child == null || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

// ── Engine client ───────────────────────────────────────────────────
const worker = new Worker("/workbench/engine-worker.js", { type: "module" });
const pending = new Map();
let sequence = 0, working = 0;
let engineReady;
const ready = new Promise((resolve, reject) => { engineReady = { resolve, reject }; });

worker.onmessage = ({ data }) => {
  if (data.type === "boot") return bootStep(data.step);
  if (data.type === "ready") return engineReady.resolve(data);
  if (data.type === "fatal") return engineReady.reject(new Error(data.message));
  const job = pending.get(data.id);
  if (!job) return;
  pending.delete(data.id);
  if (!data.ok) job.reject(new Error(data.error));
  else job.resolve(data.bytes ? data.bytes : JSON.parse(data.result));
};
worker.onerror = (event) => engineReady.reject(new Error(event.message || "The engine worker failed to start"));

// Long checks say how long they have been running: on a large file the engine takes
// tens of seconds, and a badge that never changes reads as a hang.
let busySince = 0, busyTimer = 0;
function busy(on) {
  const badge = $("busy"), label = $("busy-text");
  if (on && ++working === 1) {
    busySince = performance.now();
    label.textContent = "checking";
    badge.hidden = false;
    busyTimer = setInterval(() => {
      const s = Math.round((performance.now() - busySince) / 1000);
      if (s >= 2) label.textContent = `checking, ${s} s`;
    }, 500);
  } else if (!on && --working === 0) {
    clearInterval(busyTimer);
    badge.hidden = true;
  }
}

export function call(name, args = {}, files = [], quiet = false) {
  const id = ++sequence;
  if (!quiet) busy(true);
  const done = () => { if (!quiet) busy(false); };
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, name, args, files }, files.map((f) => f.bytes));
  }).finally(done);
}

// ── State ───────────────────────────────────────────────────────────
export const S = {
  snap: null, plan: null, index: new Map(), source: null,
  patches: [], selected: null, detail: null, tab: "conversation", editing: false,
  query: "", match: null, matched: 0, panel: null,
};
let field = null, planTicket = 0, operator = null;
export const wire = [];          // every request this page made to a model endpoint

function toast(message, ms = 5200) {
  const node = $("toast");
  node.textContent = message;
  node.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { node.hidden = true; }, ms);
}

// ── Boot ────────────────────────────────────────────────────────────
function bootStep(step) {
  $("boot-text").textContent = step === "runtime"
    ? "Python runtime, 9.6 MB, cached after the first visit"
    : "Unpacking the parallelogram package";
}

async function boot() {
  for (const [id, mode] of [["route-bench", "bench"], ["route-operator", "operator"]]) {
    if (mode === MODE) $(id).setAttribute("aria-current", "page");
  }
  $("wire-label").textContent = MODE === "bench" ? "sealed" : "wire: nothing sent";
  if (MODE === "operator") {
    $("claim-sealed").textContent = "This is the operator route: it can reach the model endpoint you configure, and nothing else. Records are redacted before they leave, you see every request, and nothing is sent until you press send.";
    document.querySelector(".claims dt").textContent = "You control the wire";
  }
  try {
    const info = await ready;
    S.engine = info;
    $("boot").classList.add("done");
    $("boot-line").textContent = `Engine ready: parallelogram ${info.engine}, in this tab`;
    $("boot-text").textContent = `parallelogram ${info.engine} on Python ${info.python} (Pyodide ${info.pyodide}), started in ${(info.ms / 1000).toFixed(1)} s`;
    $("open-sample").disabled = false;
    $("open-file").disabled = false;
    const carried = await takeHandoff();
    if (carried) await resume(carried);
    else if (new URLSearchParams(location.search).has("sample")) await openSample();
  } catch (error) {
    $("boot").classList.add("failed");
    $("boot-line").textContent = "The engine did not start";
    $("boot-text").textContent = String(error.message || error);
  }
}

// ── Opening datasets ────────────────────────────────────────────────
const EVAL_NAME = /(^|[^a-z])(eval|evaluation|test|val|valid|validation|dev|heldout|holdout)([^a-z]|$)/i;

async function openSample() {
  S.source = { sample: true };
  S.patches = [];
  await show(() => call("demo"));
}

async function openFiles(list) {
  const files = [...list];
  if (!files.length) return;
  let train = files[0], evalFile = files[1] || null;
  if (files.length > 1) {
    const named = files.find((f) => EVAL_NAME.test(f.name));
    if (named) { evalFile = named; train = files.find((f) => f !== named); }
  }
  const source = { train: { name: train.name, bytes: await train.arrayBuffer() } };
  if (evalFile) source.eval = { name: evalFile.name, bytes: await evalFile.arrayBuffer() };
  S.source = source;
  S.patches = [];
  await show(() => openSource(source));
}

function openSource(source) {
  if (source.sample) return call("demo");
  const files = [{ path: "/io/train", bytes: source.train.bytes.slice(0) }];
  const args = { train_name: source.train.name, train_path: "/io/train" };
  if (source.eval) {
    files.push({ path: "/io/eval", bytes: source.eval.bytes.slice(0) });
    Object.assign(args, { eval_name: source.eval.name, eval_path: "/io/eval" });
  }
  return call("open", args, files);
}

async function setEval(file) {
  if (!S.snap) return;
  const bytes = await file.arrayBuffer();
  if (S.source.sample) return toast("The sample comes with its own eval set. Open your own train file first.");
  S.source.eval = { name: file.name, bytes };
  await show(() => call("eval", { eval_name: file.name, eval_path: "/io/eval" },
    [{ path: "/io/eval", bytes: bytes.slice(0) }]));
}

async function show(load) {
  try {
    $("start").hidden = true;
    $("bench").hidden = false;
    $("files").hidden = false;
    if (!field) field = new Field($("field"), { onHover: hover, onSelect: (i) => select(S.snap.ids[i]) });
    setSnap(await load(), true);
  } catch (error) {
    if (!S.snap) { $("start").hidden = false; $("bench").hidden = true; $("files").hidden = true; }
    toast(error.message);
  }
}

function setSnap(snap, fresh = false) {
  S.snap = snap;
  S.plan = null;
  S.index = new Map(snap.ids.map((id, i) => [id, i]));
  if (fresh) { S.selected = null; S.detail = null; S.query = ""; $("q").value = ""; S.match = null; S.editing = false; }
  const links = new Map();
  for (const [dup, first] of Object.entries(snap.links.duplicates)) {
    if (S.index.has(+dup) && S.index.has(first)) links.set(S.index.get(+dup), S.index.get(first));
  }
  field.setData(snap, links, snap.options.max_seq_len);
  field.setSelected(S.selected == null ? -1 : S.index.get(S.selected));
  renderBar(); renderGate(); renderLedger(); renderLegend();
  applyQuery();
  if (S.selected != null) select(S.selected, true); else renderInspector();
  const ticket = ++planTicket;
  call("plan", {}, [], true).then((plan) => {
    if (ticket !== planTicket) return;
    S.plan = plan;
    renderGate();
    if (/unrepairable/i.test(S.query)) applyQuery();      // that filter is answered by the plan
    if (S.detail && !S.editing && !S.panel) renderInspector();
    if (operator) operator.planned();
  }).catch(() => {});
}

// ── Ledger operations ───────────────────────────────────────────────
export async function addPatch(patch) {
  S.patches.push({ id: Math.random().toString(36).slice(2, 10), enabled: true, ...patch });
  return syncPatches(() => S.patches.pop());
}

async function syncPatches(undo) {
  try {
    setSnap(await call("patches", { patches: S.patches }));
    return true;
  } catch (error) {
    if (undo) undo();
    toast(error.message);
    renderLedger();
    return false;
  }
}

const patchLabel = (p) => ({
  fix_all: "check --fix",
  fix: `fix ${p.rule}`,
  drop_rule: `drop ${p.rule}`,
  drop: p.rows && p.rows.length === 1 ? `drop line ${p.rows[0]}` : `drop ${p.rows ? p.rows.length : ""} records`,
  edit: `edit line ${Object.keys(p.rows || {})[0]}`,
  repair: "model repair",
}[p.kind]);

// ── Top bar ─────────────────────────────────────────────────────────
function renderBar() {
  const { snap } = S;
  fill($("file-train"), snap.file, h("span", { class: "dim", text: `${fmt(snap.ids.length)} records` }));
  fill($("file-eval"), ...(snap.eval
    ? [snap.eval, h("span", { class: "dim", text: `eval, ${fmt(snap.eval_records)}` })]
    : ["+ eval set"]));
  if (document.activeElement !== $("budget")) $("budget").value = snap.options.max_seq_len;
  $("rules-button").textContent = snap.options.disable.length ? `rules (${snap.options.disable.length} off)` : "rules";
}

function renderRules() {
  const { snap } = S;
  const off = new Set(snap.options.disable);
  fill($("rules-pop"), 
    h("h3", { text: "Rules" }),
    h("p", { text: "Every rule is on by default, as in the CLI. Turning one off is the CLI's --disable: the gate stops looking for that problem, and the command below says so." }),
    ...snap.rules.map((rule) => h("label", { class: "rule-toggle" },
      h("input", {
        type: "checkbox", checked: !off.has(rule.id), disabled: rule.id === "schema",
        onchange: (e) => {
          const next = new Set(off);
          if (e.target.checked) next.delete(rule.id); else next.add(rule.id);
          setOptions({ disable: [...next] });
        },
      }),
      h("b", { text: rule.id + (rule.id === "schema" ? "  (always on)" : "") }),
      h("span", { text: rule.description }))));
}

async function setOptions(options) {
  try { setSnap(await call("options", options)); if (!$("rules-pop").hidden) renderRules(); }
  catch (error) { toast(error.message); renderBar(); }
}

// ── Gate ────────────────────────────────────────────────────────────
function renderGate() {
  const { snap, plan } = S;
  const st = snap.stage;
  const exit = $("exit");
  exit.textContent = st.exit;
  exit.dataset.exit = st.exit;

  const removed = snap.gone.filter(Boolean).length;
  fill($("gate-line"), 
    h("b", { text: fmt(st.valid) }), ` of ${fmt(st.total)} records pass. `,
    st.errors ? h("b", { text: plural(st.errors, "error") }) : "No errors",
    st.warnings ? `, ${plural(st.warnings, "warning")}.` : ".",
    removed ? ` ${plural(removed, "record")} removed so far.` : "");

  const box = $("gate-fix");
  fill(box);
  if (st.exit === 0) {
    box.append(h("p", {}, h("b", { text: "This file exits 0. " }),
      "Download it below, then keep it that way: the same check runs in CI and before every training run."));
  } else if (!plan) {
    box.append(h("p", { text: "Working out what check --fix would do…" }));
  } else {
    const pr = plan.projection, n = plan.unrepairable.length;
    box.append(h("p", {}, h("code", { text: "check --fix" }), " would repair ", h("b", { text: plural(pr.fixed, "record") }),
      " and drop ", h("b", { text: fmt(pr.dropped + pr.unparseable) }), `, leaving ${fmt(pr.emitted)}. `,
      n ? h("b", { text: `${fmt(n)} of the dropped ${n === 1 ? "has" : "have"} no mechanical repair` }) : null,
      n ? ": an edit or a model can keep them." : null));
    const row = h("div", { class: "row" },
      h("button", { class: "btn small primary", type: "button", text: "Apply --fix", onclick: () => addPatch({ kind: "fix_all" }) }));
    if (n) {
      row.append(h("button", { class: "btn small", type: "button", text: n === 1 ? "Show it" : `Show the ${fmt(n)}`, onclick: () => setQuery("state:unrepairable") }));
      row.append(MODE === "operator"
        ? h("button", { class: "btn small accent", type: "button", text: "Repair with a model", onclick: () => openOperator(plan.unrepairable) })
        : h("a", { class: "btn small", href: "/workbench/operator/", text: "Operator route", title: "Model-assisted repair lives on a separate route", onclick: switchRoute }));
    }
    box.append(row);
  }

  const list = $("classes");
  fill(list);
  const active = /^rule:(\S+)$/.exec(S.query.trim());
  const classes = [...snap.classes].sort((a, b) => (b.errors > 0) - (a.errors > 0) || b.errors - a.errors || b.warnings - a.warnings);
  for (const c of classes) {
    const does = plan && plan.by_rule[c.id];
    const count = [c.errors && plural(c.errors, "error"), c.warnings && plural(c.warnings, "warning")].filter(Boolean).join(", ");
    const actions = h("div", { class: "class-actions" });
    if (c.fixable) {
      actions.append(h("button", { class: "btn small", type: "button", text: "Fix", title: c.fix_description,
        onclick: () => addPatch({ kind: "fix", rule: c.id }) }));
      actions.append(h("span", { class: "fine", text: c.fix_scope === "dataset"
        ? `drops ${plural(c.rows, "record")}`
        : does && does.would_drop ? `${c.id === "empty-content" ? "removes the empty turn; " : ""}--fix then drops ${fmt(does.would_drop)}` : c.fix_description.split(";")[0].replace(/\.$/, "").replace(/^./, (ch) => ch.toLowerCase()) }));
    } else {
      actions.append(h("button", { class: "btn small", type: "button", text: `Drop ${fmt(c.rows)}`, title: c.fix_description,
        onclick: () => addPatch({ kind: "drop_rule", rule: c.id }) }));
      actions.append(h("span", { class: "fine", text: "no mechanical repair" }));
    }
    list.append(h("li", { class: active && active[1] === c.id ? "active" : "" },
      h("button", { class: "class-top", type: "button", title: "Show these records",
        onclick: () => setQuery(active && active[1] === c.id ? "" : `rule:${c.id}`) },
        h("span", { class: `sev ${c.errors ? "error" : "warning"}` }), h("b", { text: c.id }), h("span", { class: "count", text: count })),
      h("p", { class: "class-desc", text: c.description }),
      actions));
  }
  if (!classes.length) {
    list.append(h("li", {}, h("p", { class: "gate-clean", text: "Nothing. Every enabled rule passes on every record." })));
  }
  const notes = [];
  if (!snap.tokens_info.exact) notes.push("Token counts here are approximate (about 4 characters per token). For exact counts run the CLI with --tokenizer.");
  if (!snap.eval) notes.push("No eval set is open, so train/eval overlap is not being checked.");
  if (snap.options.disable.length) notes.push(`Disabled: ${snap.options.disable.join(", ")}.`);
  $("gate-notes").textContent = notes.join(" ");
}

// ── Query ───────────────────────────────────────────────────────────
function setQuery(text) {
  $("q").value = text;
  S.query = text;
  applyQuery();
  renderGate();
}

let queryTicket = 0;
async function applyQuery() {
  const { snap } = S;
  const text = S.query.trim();
  const ticket = ++queryTicket;
  if (!text) {
    S.match = null; S.matched = 0;
    field.setMatch(null);
    return renderMatches();
  }
  const terms = text.match(/"[^"]*"|\S+/g) || [];
  const tests = [], words = [];
  const ruleBit = (name) => {
    const hit = snap.rules.filter((r) => r.id.startsWith(name));
    return hit.reduce((bits, r) => bits | r.bit, 0);
  };
  const unrepairable = new Set(S.plan ? S.plan.unrepairable : []);
  for (const raw of terms) {
    const term = raw.replace(/^"|"$/g, "");
    let m;
    if ((m = /^rule:(.+)$/i.exec(term))) { const bits = ruleBit(m[1].toLowerCase()); tests.push((i) => (snap.mask[i] & bits) !== 0 && snap.state[i] !== GONE); }
    else if ((m = /^state:(.+)$/i.exec(term))) {
      const want = m[1].toLowerCase();
      tests.push({
        error: (i) => snap.state[i] === ERROR, warning: (i) => snap.state[i] === WARN, clean: (i) => snap.state[i] === CLEAN,
        removed: (i) => snap.state[i] === GONE, changed: (i) => snap.touched[i] > 0 && snap.state[i] !== GONE,
        fixed: (i) => snap.touched[i] > 0 && snap.state[i] === CLEAN,
        unrepairable: (i) => unrepairable.has(snap.ids[i]),
      }[want] || (() => false));
    }
    else if ((m = /^tokens([<>])(\d+)$/i.exec(term))) { const n = +m[2]; tests.push(m[1] === ">" ? (i) => snap.tokens[i] > n : (i) => snap.tokens[i] < n); }
    else if ((m = /^line:(\d+)(?:-(\d+))?$/i.exec(term))) { const a = +m[1], b = m[2] ? +m[2] : a; tests.push((i) => snap.ids[i] >= a && snap.ids[i] <= b); }
    else if (/^dup(licates?)?$/i.test(term)) tests.push((i) => snap.ids[i] in snap.links.duplicates);
    else if (/^leak(s|ed)?$/i.test(term)) tests.push((i) => snap.ids[i] in snap.links.eval);
    else words.push(term.replace(/^text:/i, ""));
  }
  for (const word of words.filter(Boolean)) {
    const ids = new Set(await call("search", { text: word }, [], true));
    if (ticket !== queryTicket) return;
    tests.push((i) => ids.has(snap.ids[i]));
  }
  const match = new Uint8Array(snap.ids.length);
  let count = 0;
  for (let i = 0; i < match.length; i++) if (tests.every((t) => t(i))) { match[i] = 1; count++; }
  S.match = match; S.matched = count;
  field.setMatch(match);
  renderMatches();
}

function renderMatches() {
  const { snap } = S;
  const strip = $("matches");
  $("q-count").textContent = S.match ? `${fmt(S.matched)} of ${fmt(snap.ids.length)}` : "";
  if (!S.match) { strip.hidden = true; return; }
  strip.hidden = false;
  fill(strip);
  if (!S.matched) { strip.append(h("span", { class: "fine", text: "No record matches." })); return; }
  const shown = [];
  for (let i = 0; i < S.match.length && shown.length < 80; i++) if (S.match[i]) shown.push(snap.ids[i]);
  strip.append(h("span", { class: "fine", text: "lines" }));
  for (const id of shown) {
    strip.append(h("button", { type: "button", text: id, "aria-current": id === S.selected ? "true" : null, onclick: () => select(id) }));
  }
  if (S.matched > shown.length) strip.append(h("span", { class: "fine", text: `and ${fmt(S.matched - shown.length)} more` }));
  const live = [];
  for (let i = 0; i < S.match.length; i++) if (S.match[i] && snap.state[i] !== GONE) live.push(snap.ids[i]);
  if (live.length && live.length < snap.ids.length) {
    strip.append(h("button", { class: "btn small", type: "button", text: `Drop ${live.length === 1 ? "this record" : `these ${fmt(live.length)}`}`,
      onclick: () => addPatch({ kind: "drop", rows: live }) }));
    if (MODE === "operator" && live.length <= 200) {
      strip.append(h("button", { class: "btn small accent", type: "button", text: "Repair with a model", onclick: () => openOperator(live) }));
    }
  }
}

// ── Field: hover and legend ─────────────────────────────────────────
let peekTicket = 0;
function hover(index, event) {
  const peek = $("peek");
  const ticket = ++peekTicket;
  if (index < 0 || !event) { peek.hidden = true; return; }
  const { snap } = S, id = snap.ids[index], state = snap.state[index];
  const wrap = $("field-wrap").getBoundingClientRect();
  const x = Math.min(event.clientX - wrap.left + 16, wrap.width - 350), y = event.clientY - wrap.top + 18;
  peek.style.left = `${Math.max(0, x)}px`;
  peek.style.top = `${Math.min(y, wrap.height - 96)}px`;
  const label = state === GONE ? `removed in step ${pad2(snap.gone[index])}`
    : state === ERROR ? "blocks the run" : state === WARN ? "warning"
    : snap.touched[index] ? `repaired in step ${pad2(snap.touched[index])}` : "passes";
  const head = h("div", {}, h("b", { text: `line ${id}` }), h("span", { class: "num", text: `  ${label}  ·  ${fmt(snap.tokens[index])} tok` }));
  fill(peek, head);
  peek.hidden = false;
  clearTimeout(hover.timer);
  hover.timer = setTimeout(async () => {
    try {
      const info = await call("peek", { id }, [], true);
      if (ticket !== peekTicket) return;
      fill(peek, head,
        ...info.why.map((why) => h("p", { class: "why", text: why })),
        info.more ? h("p", { text: `and ${info.more} more` }) : null,
        h("p", { text: info.first }));
    } catch { /* the card keeps its first line */ }
  }, 70);
}

function renderLegend() {
  const { snap } = S;
  const count = (test) => snap.state.reduce((n, s, i) => n + (test(s, i) ? 1 : 0), 0);
  const repaired = count((s, i) => s === CLEAN && snap.touched[i] > 0);
  const items = [
    ["error", "blocks the run", count((s) => s === ERROR)],
    ["warn", "warning", count((s) => s === WARN)],
    ["", "passes", count((s, i) => s === CLEAN && !snap.touched[i])],
  ];
  if (repaired) items.push(["fixed", "repaired", repaired]);
  const gone = count((s) => s === GONE);
  if (gone) items.push(["gone", "removed", gone]);
  fill($("legend"), 
    ...items.map(([cls, label, n]) => h("span", {}, h("i", { class: cls }), `${label} ${fmt(n)}`)),
    h("span", { class: "right", text: field.per > 1 ? `one cell is up to ${field.per} records, showing the worst` : "one cell is one record" }));
}

// ── Inspector ───────────────────────────────────────────────────────
let rowTicket = 0;
export async function select(id, keep = false) {
  if (id == null || !S.index.has(id)) return;
  const ticket = ++rowTicket;
  S.selected = id;
  if (!keep) { S.editing = false; S.panel = null; }
  field.setSelected(S.index.get(id));
  if (S.match) renderMatches();
  try {
    const detail = await call("row", { id }, [], true);
    if (ticket !== rowTicket) return;
    S.detail = detail;
    if (S.tab === "changes" && !detail.changed) S.tab = "conversation";
    if (!S.panel) renderInspector();
  } catch (error) { toast(error.message); }
}

function stateTag(detail) {
  const i = S.index.get(detail.id), snap = S.snap;
  if (detail.gone) return h("span", { class: "state-tag gone", text: `removed in step ${pad2(detail.gone)}` });
  if (snap.state[i] === ERROR) return h("span", { class: "state-tag error", text: "blocks the run" });
  if (snap.state[i] === WARN) return h("span", { class: "state-tag", text: "warning" });
  if (detail.changed) return h("span", { class: "state-tag fixed", text: `repaired in step ${pad2(detail.changed)}, passes` });
  return h("span", { class: "state-tag", text: "passes" });
}

export function conversation(record, spans = {}, flagged = new Set()) {
  const out = [];
  if (!record || typeof record !== "object") return [h("p", { class: "fine", text: "This line is not a JSON record, so there is no conversation to show. See the raw line." })];
  const tools = record.tools || record.functions;
  if (tools) out.push(h("div", { class: "turn" }, h("div", { class: "role", text: "tools declared" }),
    h("pre", { text: (Array.isArray(tools) ? tools : [tools]).map((t) => (t && t.function ? t.function.name : t && t.name) || JSON.stringify(t)).join("\n") })));
  for (const key of ["messages", "rejected"]) {
    const messages = record[key];
    if (!Array.isArray(messages)) continue;
    if (key === "rejected") out.push(h("div", { class: "turn" }, h("div", { class: "role", text: "rejected answer (preference pair)" })));
    messages.forEach((msg, index) => {
      if (!msg || typeof msg !== "object") { out.push(h("div", { class: "turn" }, h("pre", { text: JSON.stringify(msg) }))); return; }
      const turn = h("div", { class: "turn" },
        h("div", { class: `role${key === "messages" && flagged.has(index) ? " flag" : ""}`, text: `${index}  ${msg.role ?? "(no role)"}` }));
      const content = msg.content;
      if (typeof content === "string" && content.length) {
        const text = h("div", { class: `text${content.length > 1800 ? " clip" : ""}` });
        let at = 0;
        for (const [start, end, rule, kind] of spans[`${key}:${index}`] || []) {
          text.append(content.slice(at, start), h("mark", { class: rule, title: kind, text: content.slice(start, end) }));
          at = end;
        }
        text.append(content.slice(at));
        turn.append(text);
      } else if (content == null || content === "") {
        if (!msg.tool_calls && !msg.function_call) turn.append(h("div", { class: "text empty", text: content === "" ? "empty string" : "no content" }));
      } else {
        turn.append(h("pre", { text: JSON.stringify(content, null, 2) }));
      }
      for (const extra of ["tool_calls", "function_call", "tool_call_id", "name"]) {
        if (msg[extra] != null) turn.append(h("pre", { text: `${extra}: ${typeof msg[extra] === "string" ? msg[extra] : JSON.stringify(msg[extra], null, 2)}` }));
      }
      out.push(turn);
    });
  }
  return out;
}

const pretty = (line) => { try { return JSON.stringify(JSON.parse(line), null, 2); } catch { return line; } };

// Word-level difference, enough to read one record against its earlier self.
export function diffNodes(before, after) {
  const a = before.match(/\s+|[^\s\w]|\w+/g) || [], b = after.match(/\s+|[^\s\w]|\w+/g) || [];
  let head = 0;
  while (head < a.length && head < b.length && a[head] === b[head]) head++;
  let tail = 0;
  while (tail < a.length - head && tail < b.length - head && a[a.length - 1 - tail] === b[b.length - 1 - tail]) tail++;
  const ma = a.slice(head, a.length - tail), mb = b.slice(head, b.length - tail);
  const nodes = [];
  const clip = (words, side) => {
    const text = words.join("");
    return text.length > 1400 ? `${text.slice(0, 600)} … ${fmt(text.length - 1200)} characters … ${text.slice(-600)}` : text;
  };
  const lead = a.slice(0, head).join(""), trail = a.slice(a.length - tail).join("");
  nodes.push(lead.length > 500 ? `… ${lead.slice(-420)}` : lead);
  if (ma.length * mb.length > 2_500_000) {
    if (ma.length) nodes.push(h("del", { text: clip(ma) }));
    if (mb.length) nodes.push(h("ins", { text: clip(mb) }));
  } else {
    const rows = ma.length + 1, cols = mb.length + 1, table = new Uint16Array(rows * cols);
    for (let i = ma.length - 1; i >= 0; i--) for (let j = mb.length - 1; j >= 0; j--) {
      table[i * cols + j] = ma[i] === mb[j] ? table[(i + 1) * cols + j + 1] + 1 : Math.max(table[(i + 1) * cols + j], table[i * cols + j + 1]);
    }
    let i = 0, j = 0, run = null;
    const push = (kind, word) => {
      if (run && run.kind === kind) run.words.push(word); else { run = { kind, words: [word] }; runs.push(run); }
    };
    const runs = [];
    while (i < ma.length && j < mb.length) {
      if (ma[i] === mb[j]) { push("same", ma[i]); i++; j++; }
      else if (table[(i + 1) * cols + j] >= table[i * cols + j + 1]) push("del", ma[i++]);
      else push("ins", mb[j++]);
    }
    while (i < ma.length) push("del", ma[i++]);
    while (j < mb.length) push("ins", mb[j++]);
    for (const r of runs) nodes.push(r.kind === "same" ? clip(r.words) : h(r.kind, { text: clip(r.words) }));
  }
  nodes.push(trail.length > 500 ? `${trail.slice(0, 420)} …` : trail);
  return nodes;
}

function renderInspector() {
  const root = $("inspector");
  if (S.panel === "operator" && operator) return operator.render(root);
  const detail = S.detail;
  if (!detail || S.selected == null) {
    fill(root, h("div", { class: "insp-empty" },
      h("h3", { text: "Pick a record" }),
      h("p", { text: "Each cell on the left is one record of the file, in the order the file has them. Orange cells stop the run; pale ones are warnings." }),
      h("p", {}, "Click a cell to read the conversation and what the engine found in it. ", h("kbd", { text: "←↑↓→" }), " move between cells, ",
        h("kbd", { text: "]" }), " jumps to the next record with a finding, ", h("kbd", { text: "/" }), " filters."),
      h("p", { text: "Nothing you do here touches your file. Every change is a step in the ledger at the bottom, and every step can be switched off." })));
    return;
  }
  const i = S.index.get(detail.id);
  const flagged = new Set(detail.issues.map((x) => x.context && x.context.message_index).filter((x) => x != null));
  const tabs = [["conversation", "Conversation"], ["raw", "Raw line"]];
  if (detail.changed) tabs.push(["changes", "Changes"]);

  const head = h("div", { class: "insp-head" },
    h("div", { class: "insp-title" },
      h("h3", {}, h("span", { text: "line" }), String(detail.id)),
      h("div", { class: "insp-meta" }, `${detail.format || "unreadable"}`, h("br"), `${detail.tokens != null ? `${S.snap.tokens_info.exact ? "" : "≈ "}${fmt(detail.tokens)} tok` : ""}${detail.line && detail.line !== detail.id ? `  ·  now line ${detail.line}` : ""}`)),
    stateTag(detail),
    S.editing ? null : h("div", { class: "tabs", role: "tablist" },
      tabs.map(([key, label]) => h("button", { type: "button", role: "tab", "aria-selected": String(S.tab === key), text: label,
        onclick: () => { S.tab = key; renderInspector(); } }))));

  const body = h("div", { class: "insp-body" });
  if (S.editing) return renderEditor(root, head, body, detail);

  if (detail.issues.length) {
    body.append(h("div", { class: "findings" }, detail.issues.map((x) => h("div", { class: "finding" },
      h("b", { class: x.severity, text: `${x.rule}${x.severity === "warning" ? "  warning" : ""}` }),
      h("p", { text: x.message }), x.detail ? h("p", { class: "fine", text: x.detail.replace(/--fix/g, "the fix") }) : null))));
  }
  const dropped = S.plan && S.plan.dropped[detail.id];
  if (dropped && !detail.gone) {
    const mechanical = S.plan.unrepairable.includes(detail.id);
    body.append(h("p", { class: "note" }, h("b", { text: "check --fix would drop this record. " }),
      mechanical ? `There is no mechanical repair for ${dropped.join(", ")}. Edit it here, or drop it.`
        : `Dropping it is the repair for ${dropped.join(", ")}.`));
  }
  if (detail.gone) {
    body.append(h("p", { class: "note" }, h("b", { text: `Removed in step ${pad2(detail.gone)}. ` }),
      "This is the record as it was when it was removed. Switch that step off in the ledger to bring it back."));
  }
  const dupOf = S.snap.links.duplicates[detail.id], leak = S.snap.links.eval[detail.id];
  if (dupOf && !detail.gone) body.append(h("p", { class: "note" }, "Same content as ", h("button", { class: "link", type: "button", text: `line ${dupOf}`, onclick: () => select(dupOf) }), ". The dashed line in the field joins the two."));
  if (leak && !detail.gone) body.append(h("p", { class: "note", text: `Also in the eval set, at line ${leak} of ${S.snap.eval}. A model graded on this record has already seen the answer.` }));

  if (S.tab === "conversation") body.append(...conversation(detail.record, detail.spans, flagged));
  else if (S.tab === "raw") body.append(h("pre", { class: "raw", text: pretty(detail.raw) }));
  else body.append(h("div", { class: "diff" }, diffNodes(pretty(detail.base), pretty(detail.raw))));

  if (!detail.gone) {
    const actions = h("div", { class: "insp-actions" },
      h("button", { class: "btn small", type: "button", text: "Edit", onclick: () => { S.editing = true; renderInspector(); } }),
      h("button", { class: "btn small", type: "button", text: "Drop this record", onclick: () => addPatch({ kind: "drop", rows: [detail.id] }) }));
    if (MODE === "operator" && S.snap.state[i] !== CLEAN) {
      actions.append(h("button", { class: "btn small accent", type: "button", text: "Repair with a model", onclick: () => openOperator([detail.id]) }));
    }
    body.append(actions);
  }
  fill(root, head, body);
}

function renderEditor(root, head, body, detail) {
  const area = h("textarea", { class: "editor", spellcheck: "false", "aria-label": `Record at line ${detail.id}, as JSON` });
  area.value = pretty(detail.raw);
  const verdict = h("div", { class: "verdict", text: "Checked as you type, by every enabled rule." });
  const preview = h("div", {});
  const apply = h("button", { class: "btn small primary", type: "button", text: "Apply as a step", onclick: async () => {
    if (await addPatch({ kind: "edit", rows: { [detail.id]: area.value } })) { S.editing = false; select(detail.id); }
  } });
  let ticket = 0;
  const check = async () => {
    const mine = ++ticket;
    try {
      const v = await call("validate", { id: detail.id, text: area.value }, [], true);
      if (mine !== ticket) return;
      verdict.className = `verdict ${v.errors ? "fail" : "pass"}`;
      fill(verdict, ...(v.issues.length
        ? v.issues.flatMap((x, n) => [n ? h("br") : null, h("b", { text: `${x.rule}  ` }), x.message + (x.detail && x.rule === "schema" ? ` (${x.detail})` : "")])
        : [h("b", { text: "passes  " }), "No rule objects to this record."]));
      if (!v.errors && v.warnings) verdict.prepend(h("b", { text: "passes with warnings  " }), h("br"));
      fill(preview, ...(v.record ? conversation(v.record, v.spans) : []));
    } catch (error) { verdict.className = "verdict fail"; verdict.textContent = error.message; }
  };
  area.addEventListener("input", () => { clearTimeout(check.timer); check.timer = setTimeout(check, 220); });
  body.append(
    h("p", { class: "note", text: "Edit the record as JSON. It is written back as one line. Duplicates are judged against the rest of the file as it stands now." }),
    area, verdict,
    h("div", { class: "insp-actions" }, apply,
      h("button", { class: "btn small", type: "button", text: "Cancel", onclick: () => { S.editing = false; renderInspector(); } })),
    preview);
  fill(root, head, body);
  check();
  area.focus();
}

// ── Ledger ──────────────────────────────────────────────────────────
function renderLedger() {
  const { snap } = S;
  const steps = $("steps");
  const after = (x) => h("div", { class: "after" }, "exit", h("b", { dataset: { exit: x.exit }, text: x.exit }),
    `${fmt(x.errors)} err · ${fmt(x.warnings)} warn`);
  fill(steps, h("li", { class: "step" },
    h("div", { class: "n", text: "00" }), h("div", { class: "what", text: "opened" }),
    h("div", { class: "did", text: `${fmt(snap.base.total)} records` }), after(snap.base)));
  snap.ledger.forEach((entry, index) => {
    const patch = S.patches[index];
    const did = !entry.enabled ? "switched off"
      : [entry.changed && `${fmt(entry.changed)} changed`, entry.dropped && `${fmt(entry.dropped)} removed`].filter(Boolean).join(", ") || "no effect";
    steps.append(h("li", { class: `step${entry.enabled ? "" : " off"}` },
      h("div", { class: "n", text: entry.enabled ? pad2(entry.number) : "--" }),
      h("div", { class: "what", text: patchLabel(patch), title: patch.model ? `model: ${patch.model}` : null }),
      h("div", { class: "did", text: patch.model && entry.enabled ? `${did} · ${patch.model}` : did }),
      entry.enabled ? after(entry) : null,
      h("div", { class: "step-tools" },
        h("button", { type: "button", title: entry.enabled ? "Switch this step off" : "Switch this step on", "aria-label": entry.enabled ? "Switch this step off" : "Switch this step on",
          text: entry.enabled ? "○" : "●", onclick: () => { patch.enabled = !entry.enabled; syncPatches(() => { patch.enabled = entry.enabled; }); } }),
        h("button", { type: "button", title: "Delete this step", "aria-label": "Delete this step", text: "×",
          onclick: () => { const [gone] = S.patches.splice(index, 1); syncPatches(() => S.patches.splice(index, 0, gone)); } }))));
  });
  steps.scrollLeft = steps.scrollWidth;
  $("ledger-hint").textContent = snap.active_patches
    ? "Your file is untouched. Switch a step off to see the dataset without it."
    : "Every change becomes a step here. Steps can be switched off or deleted.";

  const cli = snap.cli;
  fill($("cli"), 
    h("span", { text: !snap.active_patches ? "this check, in a terminal" : cli.exact ? "same bytes from the CLI" : "then gate the download" }),
    h("code", { text: cli.command, title: cli.command }),
    h("button", { class: "btn small", type: "button", text: "Copy", onclick: async (e) => {
      try { await navigator.clipboard.writeText(cli.command); e.target.textContent = "Copied"; setTimeout(() => { e.target.textContent = "Copy"; }, 1400); }
      catch { toast("Copying needs clipboard permission. The command is selectable."); }
    } }));
}

async function download(what) {
  const { snap } = S;
  const stem = snap.file.replace(/\.jsonl$/i, "");
  const onlyFix = S.patches.filter((p) => p.enabled).map((p) => p.kind).join() === "fix_all";
  const name = { working: snap.active_patches ? (onlyFix ? `${stem}.clean.jsonl` : `${stem}.workbench.jsonl`) : snap.file,
    gate: `${stem}.pass.jsonl`, recipe: `${stem}.recipe.json` }[what];
  try {
    const bytes = await call("export", { what, out_path: "/io/out" });
    const url = URL.createObjectURL(new Blob([bytes], { type: what === "recipe" ? "application/json" : "application/x-ndjson" }));
    const link = h("a", { href: url, download: name });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 4000);
  } catch (error) { toast(error.message); }
}

// ── Wire: what this page can and did send ───────────────────────────
async function renderWire() {
  const pop = $("wire-pop");
  let policy = null;
  try { policy = (await fetch(location.pathname, { method: "HEAD", cache: "no-store" })).headers.get("content-security-policy"); } catch { /* shown below */ }
  const connect = policy && (/connect-src ([^;]+)/.exec(policy) || [])[1];
  if (MODE === "bench") {
    fill(pop, 
      h("h3", { text: "Sealed" }),
      h("p", { text: "Your dataset is read by code running in this tab. The server sent this page with the policy below; the browser enforces it, not us." }),
      policy ? h("pre", { text: policy }) : h("p", { text: "This server sent no policy header (a local preview does not). On parallelogram.dev it is set in landing/vercel.json." }),
      connect ? h("p", {}, h("code", { text: `connect-src ${connect.trim()}` }), " means scripts here can open connections to this site only. The engine files are the only thing they fetch. The Network tab of your browser's developer tools shows the same.") : null,
      h("p", {}, "Model-assisted repair needs to reach a model, so it lives on a separate route with a different policy: ", h("a", { href: "/workbench/operator/", text: "the operator route", onclick: switchRoute }), "."));
  } else {
    const sent = wire.reduce((n, w) => n + w.bytes, 0);
    fill(pop, 
      h("h3", { text: wire.length ? `${plural(wire.length, "request")}, ${fmt(sent)} bytes sent` : "Nothing sent" }),
      h("p", { text: "This route may connect to the model endpoint you set, so its policy is wider than the sealed route's. Every request the page makes to a model is listed here, with what it carried." }),
      wire.length ? h("ol", { class: "wire-log" }, wire.map((w) => h("li", {},
        h("span", { class: "dim", text: w.time }), h("span", { text: `${w.host}  ${w.what}` }), h("span", { class: "dim", text: `${fmt(w.bytes)} B` })))) : null,
      policy ? h("pre", { text: policy }) : h("p", { text: "This server sent no policy header (a local preview does not)." }),
      h("p", {}, "Records are redacted by the engine before they are sent: credentials and personal details it can find are replaced with placeholders. For work that must not leave the tab at all, use ", h("a", { href: "/workbench/", text: "the sealed route", onclick: switchRoute }), "."));
  }
}

export function wireSent(entry) {
  wire.push({ time: new Date().toTimeString().slice(0, 8), ...entry });
  const sent = wire.reduce((n, w) => n + w.bytes, 0);
  $("wire").classList.add("sent");
  $("wire-label").textContent = `wire: ${plural(wire.length, "request")}, ${sent > 2048 ? `${(sent / 1024).toFixed(1)} KB` : `${sent} B`}`;
  if (!$("wire-pop").hidden) renderWire();
}

// ── Operator (loaded only on the operator route) ────────────────────
async function openOperator(ids) {
  if (MODE !== "operator") return;
  if (!operator) {
    const module = await import("/workbench/operator.js");
    operator = module.createOperator({ S, call, h, addPatch, wireSent, select, conversation, diffNodes, fmt, plural, toast,
      close: () => { S.panel = null; renderInspector(); }, rerender: () => renderInspector() });
  }
  operator.target(ids);
  S.panel = "operator";
  renderInspector();
}

// ── Moving between the sealed and operator routes with the work intact ──
const DB = "parallelogram-workbench";
function store(work) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open(DB, 1);
    open.onupgradeneeded = () => open.result.createObjectStore("handoff");
    open.onerror = () => reject(open.error);
    open.onsuccess = () => {
      const tx = open.result.transaction("handoff", "readwrite");
      const box = tx.objectStore("handoff");
      const request = work(box);
      tx.oncomplete = () => { open.result.close(); resolve(request && request.result); };
      tx.onerror = () => reject(tx.error);
    };
  });
}

async function switchRoute(event) {
  if (!S.snap) return;                      // nothing to carry: an ordinary link
  event.preventDefault();
  const href = event.currentTarget.getAttribute("href");
  try {
    await store((box) => box.put({
      source: S.source, patches: S.patches, options: S.snap.options, selected: S.selected, at: Date.now(),
    }, "work"));
  } catch { /* storage unavailable: arrive with an empty bench */ }
  location.href = href;
}

async function takeHandoff() {
  try {
    const work = await store((box) => { const get = box.get("work"); box.delete("work"); return get; });
    return work && Date.now() - work.at < 120000 ? work : null;
  } catch { return null; }
}

async function resume(work) {
  S.source = work.source;
  S.patches = [];
  await show(() => openSource(work.source));
  if (!S.snap) return;
  const { max_seq_len, disable } = work.options;
  if (max_seq_len !== 4096 || disable.length) await setOptions({ max_seq_len, disable });
  if (work.patches.length) { S.patches = work.patches; await syncPatches(() => { S.patches = []; }); }
  if (work.selected != null) select(work.selected);
}

// ── Events ──────────────────────────────────────────────────────────
function togglePop(button, pop, render) {
  const open = pop.hidden;
  for (const other of document.querySelectorAll(".pop")) other.hidden = true;
  for (const b of document.querySelectorAll("[aria-controls]")) b.setAttribute("aria-expanded", "false");
  if (open) { render(); pop.hidden = false; button.setAttribute("aria-expanded", "true"); }
}

function nextFinding(direction) {
  const { snap } = S;
  if (!snap) return;
  const n = snap.ids.length, start = S.selected == null ? (direction > 0 ? -1 : n) : S.index.get(S.selected);
  for (let k = 1; k <= n; k++) {
    const i = ((start + direction * k) % n + n) % n;
    if (snap.state[i] === ERROR || snap.state[i] === WARN) return select(snap.ids[i]);
  }
  toast("No record has a finding.", 2200);
}

function wireEvents() {
  $("open-sample").addEventListener("click", openSample);
  $("open-file").addEventListener("click", () => $("picker").click());
  $("file-train").addEventListener("click", () => $("picker").click());
  $("file-eval").addEventListener("click", () => $("picker-eval").click());
  $("picker").addEventListener("change", (e) => { openFiles(e.target.files); e.target.value = ""; });
  $("picker-eval").addEventListener("change", (e) => { if (e.target.files[0]) setEval(e.target.files[0]); e.target.value = ""; });
  $("budget").addEventListener("change", (e) => {
    const n = Math.floor(Number(e.target.value));
    if (n >= 1) setOptions({ max_seq_len: n }); else renderBar();
  });
  $("rules-button").addEventListener("click", (e) => togglePop(e.currentTarget, $("rules-pop"), renderRules));
  $("wire").addEventListener("click", (e) => togglePop(e.currentTarget, $("wire-pop"), renderWire));
  $("route-bench").addEventListener("click", (e) => { if (MODE !== "bench") switchRoute(e); else e.preventDefault(); });
  $("route-operator").addEventListener("click", (e) => { if (MODE !== "operator") switchRoute(e); else e.preventDefault(); });

  $("q").addEventListener("input", (e) => {
    S.query = e.target.value;
    clearTimeout(wireEvents.timer);
    wireEvents.timer = setTimeout(() => { applyQuery(); renderGate(); }, 140);
  });
  $("q").addEventListener("keydown", async (e) => {
    if (e.key === "Enter") {
      // Enter must act on what is typed now, not on the last debounced filter.
      clearTimeout(wireEvents.timer);
      S.query = e.target.value;
      await applyQuery();
      renderGate();
      if (S.match && S.matched) select(S.snap.ids[S.match.indexOf(1)]);
    }
    if (e.key === "Escape") { setQuery(""); e.target.blur(); }
  });
  for (const button of document.querySelectorAll("[data-arrange]")) {
    button.addEventListener("click", () => {
      for (const b of document.querySelectorAll("[data-arrange]")) b.setAttribute("aria-pressed", String(b === button));
      field.setArrange(button.dataset.arrange);
      renderLegend();
    });
  }
  for (const button of document.querySelectorAll("[data-export]")) {
    button.addEventListener("click", () => download(button.dataset.export));
  }

  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName);
    if (e.key === "Escape") { for (const pop of document.querySelectorAll(".pop")) pop.hidden = true; return; }
    if (typing || e.metaKey || e.ctrlKey || e.altKey || !S.snap) return;
    if (e.key === "/") { e.preventDefault(); $("q").focus(); }
    else if (e.key === "]") nextFinding(1);
    else if (e.key === "[") nextFinding(-1);
    else if (e.key === "e" && S.detail && !S.detail.gone && !S.panel) { e.preventDefault(); S.editing = true; renderInspector(); }
  });
  document.addEventListener("click", (e) => {
    if (e.target.closest(".pop, [aria-controls]")) return;
    for (const pop of document.querySelectorAll(".pop")) pop.hidden = true;
  });

  let depth = 0;
  const hasFiles = (e) => e.dataTransfer && [...e.dataTransfer.types].includes("Files");
  window.addEventListener("dragenter", (e) => { if (hasFiles(e)) { depth++; $("drop").hidden = false; e.preventDefault(); } });
  window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
  window.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; $("drop").hidden = true; } });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0; $("drop").hidden = true;
    if (S.engine) openFiles(e.dataTransfer.files); else toast("The engine is still starting. Drop the file again in a moment.");
  });
}

wireEvents();
boot();
