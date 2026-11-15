// End-to-end check of the workbench in a real browser, under the headers from vercel.json.
//
//   python scripts/serve_landing.py 8017 &
//   npm install --no-save playwright-core
//   node scripts/workbench_e2e.mjs http://127.0.0.1:8017 [screenshot-dir]
//
// Needs Chrome (CHROME=/path/to/chrome, default /usr/bin/google-chrome). Exits 1 on the
// first failed expectation. The operator route is exercised against a stub model server
// started here, so no real model is called.
import { chromium } from "playwright-core";
import http from "node:http";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import assert from "node:assert/strict";

const base = process.argv[2] || "http://127.0.0.1:8017";
const shots = process.argv[3] || null;
const browser = await chromium.launch({ executablePath: process.env.CHROME || "/usr/bin/google-chrome", args: ["--no-sandbox"] });
const context = await browser.newContext({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
const page = await context.newPage();
const problems = [], requests = [];
page.on("console", (m) => { if (m.type() === "error") problems.push(m.text()); });
page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
page.on("request", (r) => requests.push(r.url()));
const shot = async (name) => { if (shots) await page.screenshot({ path: path.join(shots, `${name}.png`) }); };
const text = (selector) => page.textContent(selector);
const settled = () => page.waitForFunction(() => document.getElementById("busy").hidden, null, { timeout: 60000 });
const exitIs = (code) => page.waitForFunction((c) => document.getElementById("exit").textContent.trim() === String(c), code, { timeout: 60000 });
const step = (name) => console.log(`ok  ${name}`);

// ── Sealed route ────────────────────────────────────────────────────
await page.goto(`${base}/workbench/`);
await page.waitForFunction(() => !document.getElementById("open-sample").disabled, null, { timeout: 120000 });
step(`engine started: ${await text("#boot-text")}`);
await shot("01-start");

await page.click("#open-sample");
await exitIs(2);
await page.waitForFunction(() => /Apply --fix/.test(document.getElementById("gate-fix").textContent), null, { timeout: 60000 });
assert.match(await text("#gate-line"), /2,402 of 2,412 records pass\. 10 errors, 6 warnings\./);
assert.match(await text("#gate-fix"), /would repair 7 records and drop 9, leaving 2,403/);
assert.equal(await page.locator("#classes li").count(), 9);
step("sample: exit 2, 10 errors, 6 warnings, nine finding classes; --fix projection 7 repaired / 9 dropped");
await page.waitForTimeout(500);
await shot("02-opened");

// The policy really is the sealed one, and the browser enforces it.
const csp = await page.evaluate(async () => (await fetch(location.pathname, { method: "HEAD" })).headers.get("content-security-policy"));
assert.match(csp, /connect-src 'self';/);
assert.doesNotMatch(csp, /unsafe-eval'(?<!wasm-unsafe-eval')/);
const blocked = await page.evaluate(() => fetch("https://example.com/", { mode: "no-cors" }).then(() => "sent", () => "blocked"));
assert.equal(blocked, "blocked");
step("sealed route: connect-src 'self'; a request to another origin is refused by the browser");

// Filter to one class, open the record, see the key underlined.
await page.fill("#q", "rule:secrets");
await page.waitForFunction(() => /1 of 2,412/.test(document.getElementById("q-count").textContent));
await page.press("#q", "Enter");
await page.waitForSelector("#inspector mark.secrets");
assert.match(await text("#inspector .insp-title"), /2131/);
assert.match(await text("#inspector mark.secrets"), /^sk-proj-/);
step("filter rule:secrets finds line 2131 and marks the key inside the message");
await shot("03-record");

// The card that follows the pointer names the record under it.
const cell = await page.evaluate(() => { const c = document.getElementById("field").getBoundingClientRect(); return { x: c.left + c.width / 2, y: c.top + c.height / 2 }; });
await page.mouse.move(cell.x, cell.y);
await page.waitForFunction(() => document.querySelectorAll("#peek p").length > 0);
assert.match(await text("#peek"), /^line \d+\s+passes/);
assert.doesNotMatch(await text("#peek"), /null/);
await page.mouse.move(5, 5);
step("hovering a cell shows its line, state and first message");

// Text search runs in the engine.
await page.fill("#q", "text:sig_mismatch");
await page.waitForFunction(() => /1 of 2,412/.test(document.getElementById("q-count").textContent));
await page.fill("#q", "");
step("text search finds the pasted webhook log");

// Fix one rule: a step appears, the record is redacted, the count drops.
await page.locator("#classes li", { hasText: "secrets" }).getByRole("button", { name: "Fix" }).click();
await settled();
await page.waitForFunction(() => document.querySelectorAll("#steps .step").length === 2);
assert.match(await text("#steps .step:nth-child(2)"), /fix secrets.*1 changed.*exit\s*2\s*9 err/s);
await page.waitForSelector("#inspector .state-tag.fixed");
await page.getByRole("tab", { name: "Changes" }).click();
assert.match(await text("#inspector .diff ins"), /REDACTED_SECRET/);
step("fix secrets: step 01, one record changed, nine errors left, the change is shown as a difference");
await shot("04-step");

// Edit a record by hand, with the engine judging as you type.
await page.fill("#q", "line:640");
await page.press("#q", "Enter");
await page.waitForFunction(() => /640/.test(document.querySelector("#inspector .insp-title")?.textContent || ""));
await page.keyboard.press("Escape");
await page.locator("#inspector").getByRole("button", { name: "Edit" }).click();
await page.waitForSelector("#inspector .verdict.fail");
assert.match(await text("#inspector .verdict"), /roles/);
assert.doesNotMatch(await text("#inspector .verdict"), /null/);
const edited = await page.evaluate(() => {
  const area = document.querySelector("#inspector textarea");
  const record = JSON.parse(area.value);
  record.messages.pop();                       // the trailing user turn nobody answered
  area.value = JSON.stringify(record, null, 2);
  area.dispatchEvent(new Event("input"));
  return record.messages.length;
});
assert.equal(edited, 2);
await page.waitForSelector("#inspector .verdict.pass");
await shot("05-edit");
await page.getByRole("button", { name: "Apply as a step" }).click();
await settled();
await page.waitForFunction(() => document.querySelectorAll("#steps .step").length === 3);
assert.match(await text("#steps .step:nth-child(3)"), /edit line 640.*1 changed.*8 err/s);
step("hand edit of line 640: live verdict goes from roles error to pass; applied as step 02");

// Switch a step off and on again.
await page.locator("#steps .step:nth-child(2)").hover();
await page.locator("#steps .step:nth-child(2)").getByRole("button", { name: "Switch this step off" }).click();
await settled();
await page.waitForFunction(() => document.querySelector("#steps .step:nth-child(2)").classList.contains("off"));
assert.match(await text("#steps .step:nth-child(3)"), /9 err/);
await page.locator("#steps .step:nth-child(2)").hover();
await page.locator("#steps .step:nth-child(2)").getByRole("button", { name: "Switch this step on" }).click();
await settled();
await page.waitForFunction(() => !document.querySelector("#steps .step:nth-child(2)").classList.contains("off"));
step("a step can be switched off (the error comes back) and on again");

// Finish with the engine's own --fix and download: the file must exit 0 in the real CLI.
await page.getByRole("button", { name: "Apply --fix" }).click();
await exitIs(0);
await settled();
assert.match(await text("#gate-line"), /2,404 of 2,404 records pass/);
await page.waitForTimeout(600);
await shot("06-exit0");
const dir = fs.mkdtempSync(path.join(os.tmpdir(), "wb-e2e-"));
const save = async (label) => {
  const [download] = await Promise.all([page.waitForEvent("download"), page.getByRole("button", { name: label }).click()]);
  const to = path.join(dir, download.suggestedFilename());
  await download.saveAs(to);
  return to;
};
const dataset = await save("Download dataset");
const recipe = JSON.parse(fs.readFileSync(await save("Recipe"), "utf8"));
assert.equal(path.basename(dataset), "train.workbench.jsonl");
assert.equal(fs.readFileSync(dataset, "utf8").trim().split("\n").length, 2404);
assert.deepEqual(recipe.steps.map((s) => s.kind), ["fix", "edit", "fix_all"]);
assert.equal(recipe.result.exit, 0);
step(`downloaded ${path.basename(dataset)} (2,404 records, one more than --fix alone keeps) and its recipe`);
console.log(`    dataset: ${dataset}`);

assert.deepEqual([...new Set(requests.map((u) => new URL(u).origin))], [new URL(base).origin]);
step("every request the sealed route made went to its own origin");

// ── Operator route, against a stub model ────────────────────────────
const seen = [];
const stub = http.createServer((req, res) => {
  const cors = { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "*", "Content-Type": "application/json" };
  if (req.method === "OPTIONS") { res.writeHead(204, cors); return res.end(); }
  if (req.url.endsWith("/models")) { res.writeHead(200, cors); return res.end(JSON.stringify({ data: [{ id: "stub-repairer" }] })); }
  let body = "";
  req.on("data", (chunk) => { body += chunk; });
  req.on("end", () => {
    const { messages } = JSON.parse(body);
    seen.push(body);
    const record = JSON.parse(messages[1].content.split("Record:\n")[1]);
    const retry = messages.length > 2;
    let answer;
    const last = record.messages[record.messages.length - 1];
    if (record.messages.some((m) => m.role === "assistant" && m.content === "")) {
      answer = { unrepairable: true, reason: "The assistant reply is missing; writing one would be inventing it." };
    } else if (last.role === "user") {
      record.messages.pop(); answer = record;
    } else if (record.tools && !retry) {
      answer = record;                                   // first attempt changes nothing: the engine must reject it
    } else if (record.tools) {
      record.tools.push({ type: "function", function: { name: "issue_refund", description: "Refund an invoice.",
        parameters: { type: "object", properties: { invoice_id: { type: "string" } }, required: ["invoice_id"] } } });
      answer = record;
    }
    res.writeHead(200, cors);
    res.end(JSON.stringify({ choices: [{ message: { role: "assistant", content: "```json\n" + JSON.stringify(answer) + "\n```" } }] }));
  });
});
await new Promise((resolve) => stub.listen(0, "127.0.0.1", resolve));
const stubUrl = `http://127.0.0.1:${stub.address().port}/v1`;

// Carry the work across: switch the two manual steps off first, so there is something to repair.
for (const n of [4, 3]) {
  await page.locator(`#steps .step:nth-child(${n})`).hover();
  await page.locator(`#steps .step:nth-child(${n})`).getByRole("button", { name: "Delete this step" }).click();
  await settled();
}
await exitIs(2);
await page.click("#route-operator");
await page.waitForURL(/\/workbench\/operator\/$/);
await page.waitForFunction(() => document.querySelectorAll("#steps .step").length === 2, null, { timeout: 120000 });
await exitIs(2);
assert.match(await text("#steps .step:nth-child(2)"), /fix secrets/);
step("switching to the operator route carries the dataset and the ledger across");

await page.waitForFunction(() => /Repair with a model/.test(document.getElementById("gate-fix").textContent), null, { timeout: 60000 });
await page.locator("#gate-fix").getByRole("button", { name: "Repair with a model" }).click();
await page.waitForSelector("#inspector .op pre[aria-label]");
const preview = await text("#inspector .op pre[aria-label]");
assert.match(preview, /roles: Message 2: Conversation must end on 'assistant'/);
await page.fill('#inspector input[aria-label="Endpoint URL"]', stubUrl);
await page.locator('#inspector input[aria-label="Endpoint URL"]').blur();
await page.getByRole("button", { name: "List models" }).click();
await page.waitForFunction(() => document.querySelector('#inspector input[aria-label="Model"]').value === "stub-repairer");
assert.equal(seen.length, 0);
step("operator: four unrepairable records queued, payload shown, nothing sent before the send button");
await shot("07-operator-preview");

await page.getByRole("button", { name: /^Send 4 records/ }).click();
await page.waitForFunction(() => /Apply 2 repairs as a step/.test(document.getElementById("inspector").textContent), null, { timeout: 60000 });
const cards = await page.locator("#inspector .proposal").allTextContents();
assert.equal(cards.length, 4);
assert.match(cards.find((c) => c.startsWith("line 640")), /passes/);
assert.match(cards.find((c) => c.startsWith("line 1450")), /passes, second attempt/);
assert.match(cards.find((c) => c.startsWith("line 1022")), /the model declined/);
assert.equal(seen.length, 5);                            // four records, one retry
assert.ok(seen.every((body) => !/sk-proj-[A-Za-z0-9_-]{20}/.test(body)));
step("operator: 2 proposals pass (one after the engine rejected the first attempt), 2 declined; 5 requests, no key in any");
await shot("08-operator-review");

await page.getByRole("button", { name: "Apply 2 repairs as a step" }).click();
await settled();
await page.waitForFunction(() => document.querySelectorAll("#steps .step").length === 3);
assert.match(await text("#steps .step:nth-child(3)"), /model repair.*2 changed.*stub-repairer/s);
assert.match(await text("#wire-label"), /wire: 6 requests/);      // the model list, then five carrying records
await page.getByRole("button", { name: "Apply --fix" }).click();
await exitIs(0);
await settled();
assert.match(await text("#gate-line"), /2,405 of 2,405 records pass/);
await page.waitForTimeout(600);
await shot("09-operator-exit0");
step("operator: repairs applied as one step; --fix then leaves 2,405 records at exit 0 (two more than --fix alone)");

assert.deepEqual(problems.filter((p) => !p.includes("example.com")), []);     // the refused probe above is expected
step("no console errors");
stub.close();
await browser.close();
