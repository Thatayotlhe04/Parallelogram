// The operator: a model proposes repairs, the engine decides.
// Loaded only on /workbench/operator/. The sealed route never imports this file, and its
// Content-Security-Policy would refuse the requests made here.
//
// The model is never trusted. Each proposal goes back to the engine, which checks it with
// the same rules as the gate; a proposal with an error cannot be accepted. What is sent is
// the engine's redacted copy of each record, and every request is entered on the wire log.

const PRESETS = {
  ollama: { label: "Ollama (this computer)", endpoint: "http://localhost:11434/v1" },
  lmstudio: { label: "LM Studio (this computer)", endpoint: "http://localhost:1234/v1" },
  openai: { label: "OpenAI", endpoint: "https://api.openai.com/v1" },
  openrouter: { label: "OpenRouter", endpoint: "https://openrouter.ai/api/v1" },
  custom: { label: "Another OpenAI-compatible endpoint", endpoint: "" },
};
const BATCH = 40;          // records per run; a run is something a person reviews
const PARALLEL = 3;
const ATTEMPTS = 2;        // the second attempt is told what the engine still objects to

export const SYSTEM = `You repair single records of a fine-tuning dataset so that they pass a strict validator. You are given one record as JSON and the validator's findings for it.

Rules:
- Make the smallest change that resolves the findings. Leave every other field, message and character exactly as it is, including placeholders such as [REDACTED_SECRET].
- Never invent facts, answers, tool results or message text. If the record can only pass by inventing content, for example a missing assistant reply, it is not repairable.
- Reply with exactly one JSON object and nothing else: either the corrected record, or {"unrepairable": true, "reason": "<one sentence>"}.`;

export function userPrompt(item, rules) {
  const ids = [...new Set(item.findings.map((f) => f.split(":")[0]))];
  const meaning = ids.map((id) => `- ${id}: ${(rules.find((r) => r.id === id) || {}).description || ""}`).join("\n");
  return `Findings:\n${item.findings.map((f) => `- ${f}`).join("\n")}\n\nWhat those rules require:\n${meaning}\n\nRecord:\n${item.row}`;
}

// The first JSON object in a model reply, tolerating code fences and stray prose.
export function extractObject(reply) {
  const text = String(reply || "").replace(/```(?:json)?/gi, "").trim();
  const start = text.indexOf("{"), end = text.lastIndexOf("}");
  if (start < 0 || end <= start) return null;
  try {
    const value = JSON.parse(text.slice(start, end + 1));
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch { return null; }
}

export function createOperator(ctx) {
  const { S, call, h, addPatch, wireSent, select, diffNodes, fmt, plural, toast, close, rerender } = ctx;
  const st = {
    ids: [], preset: "ollama", endpoint: PRESETS.ollama.endpoint, model: "", key: "", models: [],
    outbound: null, results: new Map(), accepted: new Set(), running: false, done: 0, note: "",
  };

  const host = () => { try { return new URL(st.endpoint).host; } catch { return "the endpoint"; } };

  async function request(path, body, what) {
    const headers = { "Content-Type": "application/json" };
    if (st.key) headers.Authorization = `Bearer ${st.key}`;
    const payload = body ? JSON.stringify(body) : null;
    if (payload) wireSent({ host: host(), what, bytes: new TextEncoder().encode(payload).length });
    let response;
    try {
      response = await fetch(st.endpoint.replace(/\/+$/, "") + path, { method: body ? "POST" : "GET", headers: body ? headers : (st.key ? { Authorization: headers.Authorization } : {}), body: payload });
    } catch {
      throw new Error(`Could not reach ${host()}. A local server has to be running and has to allow requests from ${location.origin} (Ollama: set OLLAMA_ORIGINS to that origin).`);
    }
    if (!response.ok) {
      const detail = (await response.text().catch(() => "")).slice(0, 200);
      throw new Error(`${host()} answered ${response.status}${detail ? `: ${detail}` : ""}`);
    }
    return response.json();
  }

  async function listModels() {
    try {
      const data = await request("/models");
      st.models = (data.data || []).map((m) => m.id).filter(Boolean).sort();
      if (!st.model && st.models.length) st.model = st.models[0];
      st.note = st.models.length ? `${plural(st.models.length, "model")} available at ${host()}.` : `${host()} lists no models.`;
    } catch (error) { st.note = error.message; }
    rerender();
  }

  async function prepare() {
    st.outbound = null;
    const ids = st.ids.slice(0, BATCH);
    st.outbound = await call("outbound", { ids }, [], true);
    rerender();
  }

  async function repairOne(item) {
    const messages = [{ role: "system", content: SYSTEM }, { role: "user", content: userPrompt(item, S.snap.rules) }];
    let last = null;
    for (let attempt = 1; attempt <= ATTEMPTS; attempt++) {
      const data = await request("/chat/completions", { model: st.model, temperature: 0, messages },
        `line ${item.id}${attempt > 1 ? `, attempt ${attempt}` : ""}`);
      const reply = data.choices && data.choices[0] && data.choices[0].message ? data.choices[0].message.content : "";
      const object = extractObject(reply);
      if (!object) { last = { id: item.id, kind: "unreadable", attempt, reason: "The reply was not a JSON object." }; }
      else if (object.unrepairable === true) {
        return { id: item.id, kind: "declined", attempt, reason: String(object.reason || "The model gave no reason.") };
      } else {
        const [verdict] = await call("verify", { proposals: { [item.id]: JSON.stringify(object) } }, [], true);
        last = { ...verdict, kind: verdict.ok ? "pass" : "fail", attempt };
        if (verdict.ok) return last;
      }
      if (attempt < ATTEMPTS) {
        messages.push({ role: "assistant", content: String(reply) });
        messages.push({ role: "user", content: last.kind === "unreadable"
          ? "That was not a single JSON object. Reply with the corrected record as one JSON object, or the unrepairable object."
          : `The validator still reports:\n${last.issues.filter((i) => i.severity === "error").map((i) => `- ${i.rule}: ${i.message}`).join("\n")}\n\nReply with the corrected record only, or the unrepairable object.` });
      }
    }
    return last;
  }

  async function run() {
    if (!st.model.trim()) { st.note = "Name the model to use."; return rerender(); }
    st.running = true; st.done = 0; st.results = new Map(); st.accepted = new Set(); st.note = "";
    rerender();
    const queue = [...st.outbound];
    const worker = async () => {
      for (let item; (item = queue.shift());) {
        let result;
        try { result = await repairOne(item); }
        catch (error) { result = { id: item.id, kind: "failed", reason: error.message }; }
        st.results.set(item.id, { ...result, sent: item.row });
        // Accept by default only what passes and adds little: new text is where invention hides.
        if (result.kind === "pass" && result.added <= 0.2) st.accepted.add(item.id);
        st.done++;
        rerender();
      }
    };
    await Promise.all(Array.from({ length: Math.min(PARALLEL, queue.length) }, worker));
    st.running = false;
    rerender();
  }

  async function apply() {
    const rows = {};
    for (const id of st.accepted) rows[id] = st.results.get(id).line;
    const count = Object.keys(rows).length;
    if (!count) return;
    if (await addPatch({ kind: "repair", rows, model: st.model, endpoint: host() })) {
      toast(`${plural(count, "repair")} applied as one step. The gate has re-checked the whole file.`, 4200);
      close();
      select(Number(Object.keys(rows)[0]));
    }
  }

  function field(label, control) { return [h("label", { text: label }), control]; }

  function render(root) {
    const total = st.ids.length, batch = Math.min(total, BATCH);
    const head = h("div", { class: "insp-head" },
      h("div", { class: "insp-title" }, h("h3", {}, h("span", { text: "operator" }), `${fmt(batch)} ${batch === 1 ? "record" : "records"}`),
        h("div", { class: "insp-meta" }, h("button", { class: "link", type: "button", text: "Back to the record", onclick: close }))));
    const body = h("div", { class: "insp-body op" });

    body.append(h("p", { class: "lead", text: "A model proposes a corrected record. The engine then checks each proposal with the same rules as the gate. A proposal with an error cannot be accepted, and accepted repairs become one ledger step you can switch off." }));
    if (total > BATCH) body.append(h("p", { class: "note", text: `${fmt(total)} records match; a run takes the first ${BATCH} so that a person can review it. Apply them and run again for the rest.` }));

    // 1. Where the model is
    const models = h("datalist", { id: "op-models" }, st.models.map((m) => h("option", { value: m })));
    body.append(h("div", { class: "op-grid" },
      ...field("endpoint", h("select", { "aria-label": "Endpoint preset", onchange: (e) => {
        st.preset = e.target.value;
        if (st.preset !== "custom") st.endpoint = PRESETS[st.preset].endpoint;
        st.models = []; st.note = "";
        rerender();
      } }, Object.entries(PRESETS).map(([key, p]) => h("option", { value: key, selected: key === st.preset, text: p.label })))),
      ...field("url", h("input", { type: "url", value: st.endpoint, placeholder: "https://…/v1", "aria-label": "Endpoint URL", spellcheck: "false",
        onchange: (e) => { st.endpoint = e.target.value.trim(); st.preset = "custom"; rerender(); } })),
      ...field("model", h("input", { type: "text", value: st.model, list: "op-models", placeholder: "model name", "aria-label": "Model", spellcheck: "false",
        oninput: (e) => { st.model = e.target.value; } })),
      ...field("key", h("input", { type: "password", value: st.key, placeholder: st.endpoint.startsWith("http://") ? "not needed for a local server" : "API key", "aria-label": "API key", autocomplete: "off",
        oninput: (e) => { st.key = e.target.value; } })),
      models));
    body.append(h("div", { class: "insp-actions" },
      h("button", { class: "btn small", type: "button", text: "List models", onclick: listModels }),
      h("span", { class: "fine", text: st.note || "The key is kept in this tab's memory and is not stored." })));

    // 2. What leaves
    const out = h("div", { class: "op-step" }, h("h4", { text: "What will be sent" }));
    if (!st.outbound) out.append(h("p", { text: "Preparing the redacted records…" }));
    else if (!st.outbound.length) out.append(h("p", { text: "None of these records is in the file as it stands now." }));
    else {
      const first = st.outbound[0];
      const sample = userPrompt(first, S.snap.rules);
      const bytes = st.outbound.reduce((n, item) => n + new TextEncoder().encode(SYSTEM + userPrompt(item, S.snap.rules)).length, 0);
      out.append(
        h("p", { text: `${plural(st.outbound.length, "request")} to ${host()}, about ${fmt(bytes)} bytes in total: the instructions below and, for each record, the engine's findings and its redacted copy of the record. Credentials and personal details the engine can find are already replaced with placeholders. Nothing else on this page is sent.` }),
        h("details", {}, h("summary", { class: "fine", text: "The instructions the model gets" }), h("pre", { text: SYSTEM })),
        h("pre", { "aria-label": `Request for line ${first.id}`, text: sample.length > 2400 ? `${sample.slice(0, 2400)}\n… ${fmt(sample.length - 2400)} more characters` : sample }),
        h("div", { class: "insp-actions" },
          h("button", { class: "btn small accent", type: "button", disabled: st.running, text: st.running ? `Working, ${st.done} of ${st.outbound.length}` : `Send ${plural(st.outbound.length, "record")} to ${host()}`, onclick: run })));
    }
    body.append(out);

    // 3. What came back, as judged by the engine
    if (st.results.size) {
      const review = h("div", { class: "op-step" }, h("h4", { text: "Proposals, as the engine judges them" }));
      const passing = [...st.results.values()].filter((r) => r.kind === "pass").length;
      review.append(h("p", { text: `${fmt(passing)} of ${fmt(st.results.size)} pass. Read the difference before accepting: the engine checks that a record is well-formed, not that the repair is what you meant.` }));
      for (const item of st.outbound) {
        const r = st.results.get(item.id);
        if (!r) continue;
        const card = h("div", { class: `proposal ${r.kind === "pass" ? "pass" : "fail"}` });
        const verdict = { pass: `passes${r.warnings ? `, ${plural(r.warnings, "warning")}` : ""}${r.attempt > 1 ? ", second attempt" : ""}`, fail: "rejected by the engine", declined: "the model declined", unreadable: "unreadable reply", failed: "request failed" }[r.kind];
        card.append(h("label", { class: "proposal-head" },
          h("input", { type: "checkbox", disabled: r.kind !== "pass", checked: st.accepted.has(item.id), "aria-label": `Accept the repair for line ${item.id}`,
            onchange: (e) => { if (e.target.checked) st.accepted.add(item.id); else st.accepted.delete(item.id); rerender(); } }),
          `line ${item.id}`, h("span", { class: "v", text: verdict })));
        if (r.kind === "pass" || r.kind === "fail") {
          card.append(h("p", {},
            h("span", { class: "meter", text: `keeps ${Math.round(r.kept * 100)}% of the original words` }),
            h("span", { class: `meter${r.added > 0.2 ? " warn" : ""}`, text: `${Math.round(r.added * 100)}% of the proposal is new` })));
          if (r.kind === "pass" && r.added > 0.2) card.append(h("p", { text: "This proposal adds a lot of text that was not in the record. It is not accepted unless you tick it." }));
          if (r.kind === "fail") card.append(h("p", { text: r.issues.filter((i) => i.severity === "error").map((i) => `${i.rule}: ${i.message}`).join("  ·  ") }));
          const before = (() => { try { return JSON.stringify(JSON.parse(r.sent), null, 2); } catch { return r.sent; } })();
          const after = (() => { try { return JSON.stringify(JSON.parse(r.line), null, 2); } catch { return r.line; } })();
          card.append(h("div", { class: "diff" }, diffNodes(before, after)));
        } else {
          card.append(h("p", { text: r.reason }));
        }
        review.append(card);
      }
      if (!st.running) {
        review.append(h("div", { class: "insp-actions" },
          h("button", { class: "btn small primary", type: "button", disabled: !st.accepted.size,
            text: st.accepted.size ? `Apply ${plural(st.accepted.size, "repair")} as a step` : "Nothing accepted", onclick: apply }),
          h("span", { class: "fine", text: "Declined and rejected records stay as they are. Drop them, or edit them by hand." })));
      }
      body.append(review);
    }
    root.replaceChildren(head, body);
  }

  return {
    target(ids) {
      st.ids = [...ids].sort((a, b) => a - b);
      st.results = new Map(); st.accepted = new Set(); st.outbound = null;
      prepare().catch((error) => toast(error.message));
    },
    render,
    planned() { /* the plan changed underneath; proposals already made stay valid against their own verification */ },
  };
}
