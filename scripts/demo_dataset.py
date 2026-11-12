"""Deterministic, realistic support-ticket SFT dataset for the landing demo.

Every name, company and contact detail is fictional: Fabrikam and Contoso are
Microsoft's documentation companies, +44 7700 900xxx is Ofcom's drama range,
and the API key is random characters in the OpenAI key shape.
"""
import json
import random


FIRST = ["Amara", "Lukas", "Mei", "Tomasz", "Ifeoma", "Rafael", "Hana", "Owen", "Zanele", "Mateo",
         "Leila", "Jonas", "Aiko", "Kwame", "Sofia", "Arjun", "Nora", "Elif", "Tariq", "Ingrid",
         "Kofi", "Camila", "Viktor", "Thandi", "Yusuf", "Freya", "Dario", "Naledi", "Henrik", "Priya"]
PLANS = ["Starter", "Growth", "Scale"]
CURRENCIES = ["EUR", "GBP", "USD", "ZAR", "CAD", "AUD"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
TOOLS = [{"type": "function", "function": {
    "name": "lookup_invoice",
    "description": "Fetch an invoice by number.",
    "parameters": {"type": "object", "properties": {"invoice_id": {"type": "string"}},
                   "required": ["invoice_id"]}}}]


def _inv(r):
    return f"INV-{r.randint(2024, 2026)}-{r.randint(1000, 99999):05d}"


def _amt(r):
    return f"{r.randint(40, 9800)}.{r.choice(['00', '50', '20', '75', '10', '99'])}"


def _day(r):
    return f"{r.randint(1, 28)} {r.choice(MONTHS)}"


def topics(r):
    """One support exchange: (user, assistant) or a follow-up pair appended."""
    inv, amt, cur, day = _inv(r), _amt(r), r.choice(CURRENCIES), _day(r)
    plan, plan2 = r.sample(PLANS, 2)
    n = r.randint(2, 40)
    seats = r.randint(3, 60)
    hook = r.choice(["invoice.paid", "invoice.overdue", "payout.sent", "payment.failed", "refund.updated"])
    return r.choice([
        (f"I was refunded for {inv} on {day} but I still don't see the {cur} {amt} on my statement.",
         f"Refunds for {inv} were issued on {day}. Card refunds usually take 5–10 business days to appear, "
         f"depending on your bank. If it hasn't arrived after 10 business days, reply here and we'll send you "
         f"the acquirer reference number to give your bank."),
        (f"How do I change the email address our invoices are sent to? We have {seats} people on the {plan} plan.",
         "Go to Settings → Billing → Invoice recipients, add the new address, and remove the old one. "
         "Changes apply from the next invoice; past invoices can be re-sent from Billing → Invoices."),
        (f"Our webhook endpoint keeps rejecting {hook} events with a signature error.",
         "Signature errors usually mean the raw request body was modified before verification. Verify against "
         "the exact bytes you received, use the signing secret for the same mode (test or live), and allow up to "
         "5 minutes of clock skew when checking the timestamp header."),
        (f"We're getting 429 responses after about {n} requests per second from our sync job.",
         f"The {plan} plan allows 25 requests per second per API key. Back off using the Retry-After header, "
         "batch list calls with the `limit` parameter, and contact sales if you need a higher limit."),
        (f"Can I add our VAT number to invoice {inv}? It was issued on {day}.",
         f"Yes. Add the VAT number under Settings → Tax details, then open {inv} and choose “Regenerate PDF”. "
         "The invoice number and amount stay the same."),
        (f"Our payout of {cur} {amt} scheduled for {day} hasn't arrived.",
         f"Payouts in {cur} typically land within 2 business days of the scheduled date. Check Payouts → History "
         "for the trace ID; if the status is “Sent” and it's been more than 3 business days, your bank can use "
         "that ID to locate it."),
        (f"How do I move from {plan} to {plan2} without losing our invoice history?",
         f"Switch under Settings → Plan. Moving from {plan} to {plan2} keeps all invoices, customers and payouts; "
         "the price difference is prorated on your next bill."),
        (f"A customer disputed {inv} ({cur} {amt}). What do we need to send?",
         "Open the dispute in Payments → Disputes and upload the signed contract or order confirmation, proof of "
         "delivery, and any correspondence with the customer. Submit at least 2 days before the deadline shown."),
        (f"Is there a way to export all transactions from {day} onwards as CSV?",
         "Yes. Go to Reports → Transactions, set the start date, and choose Export → CSV. Exports over 100,000 "
         "rows are emailed to the account owner as a download link."),
        (f"We need SAML SSO for our {seats}-person team. Which plans include it?",
         f"SAML SSO is included on Scale. On {plan} you can enable Google or Microsoft sign-in instead. "
         "After upgrading, set it up under Settings → Security → Single sign-on."),
        (f"Can recurring invoices for client {r.choice(FIRST)} go out on the 1st instead of the {r.randint(2, 28)}th?",
         "Open the recurring schedule, edit “Next issue date” to the 1st, and save. Future invoices follow the new "
         "date; the current cycle is not re-issued."),
        (f"I lost my phone and can't get past two-factor on our {plan} account.",
         "Use one of the recovery codes you saved when you enabled two-factor. If you don't have them, another "
         "admin on your account can reset two-factor for you under Settings → Team."),
        (f"Why was a {cur} {r.randint(1, 30)}.{r.randint(10, 99)} conversion fee added to {inv}?",
         f"{inv} was paid in a currency different from your payout currency, so a 1% conversion fee applies. "
         "Enable multi-currency payouts under Payouts → Currencies to receive that currency directly."),
        (f"Can a client pay {inv} in two parts?",
         f"Yes. Turn on “Allow partial payments” for {inv}. The invoice stays open and shows the remaining balance "
         "until it's fully paid."),
        ("How do I stop late-fee reminders for a client who's on a payment plan?",
         "Open the client's profile and switch off “Automatic reminders”, or exclude specific invoices from the "
         "reminder schedule under Invoices → Reminders."),
        (f"Our accountant wants invoice numbers to start at {r.randint(1000, 9000)} this year. Possible?",
         "Yes. Settings → Invoices → Numbering lets you set the next number and a prefix. Existing invoices keep "
         "their numbers."),
        ("Which API key should our staging server use?",
         "Use a test-mode key (it starts with `fab_test_`). Test keys never move real money and their webhooks are "
         "delivered only to endpoints registered in test mode."),
        (f"We were charged twice for the {plan} plan in {r.choice(MONTHS)}.",
         "I can see why that's worrying. Duplicate charges are usually an authorization hold that drops off within "
         "a few days. If both charges have settled, reply with the last four digits of the card and we'll refund "
         "the duplicate."),
    ])


def make_record(r, tools=False):
    user, assistant = topics(r)
    # Like most exports, no per-record system prompt: the trainer's chat template adds it.
    messages = [{"role": "user", "content": user}]
    if tools:
        inv = user.split("INV-")[1].split()[0].strip(".,?()") if "INV-" in user else None
        if inv:
            inv = "INV-" + inv
            call_id = f"call_{r.randrange(16**8):08x}"
            status = r.choice(["paid", "open", "refunded", "overdue"])
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": call_id, "type": "function",
                "function": {"name": "lookup_invoice", "arguments": json.dumps({"invoice_id": inv})}}]})
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": json.dumps({"invoice_id": inv, "status": status})})
    messages.append({"role": "assistant", "content": assistant})
    if r.random() < 0.18:
        follow = r.choice([
            ("Thanks — and will this affect our next invoice?",
             "No. Your next invoice is generated as usual on its scheduled date."),
            ("Great, can I do that from the mobile app too?",
             "Yes, the same option is under Settings in the mobile app."),
            ("Perfect. Is there anything I need to tell my customers?",
             "No action is needed on their side; they'll see the change on their next invoice."),
        ])
        messages += [{"role": "user", "content": follow[0]}, {"role": "assistant", "content": follow[1]}]
    record = {"messages": messages}
    if tools and any(m.get("tool_calls") for m in messages):
        record = {"tools": TOOLS, "messages": messages}
    return record


def build():
    r = random.Random(20261112)
    train, seen, prompts = [], set(), set()
    eval_rows = []

    def unique(tools=False):
        while True:
            rec = make_record(r, tools)
            key = json.dumps(rec["messages"], sort_keys=True)
            prompt = rec["messages"][0]["content"]
            if key not in seen and prompt not in prompts:
                seen.add(key)
                prompts.add(prompt)
                return rec

    for _ in range(240):
        eval_rows.append(unique())
    for i in range(2412):
        train.append(unique(tools=r.random() < 0.12))

    def at(line, record):
        train[line - 1] = record

    def sys_user_asst(user, assistant):
        return {"messages": [{"role": "user", "content": user},
                             {"role": "assistant", "content": assistant}]}

    # --- Defects a real helpdesk export picks up ---------------------------------
    # Eval leakage: two eval conversations were copied into train.
    at(377, json.loads(json.dumps(eval_rows[41])))
    at(1709, json.loads(json.dumps(eval_rows[188])))
    # A customer kept writing after the ticket was solved: ends on the user.
    rec = train[639]
    rec["messages"].append({"role": "user", "content": "Actually, one more thing — can you also resend last month's invoice?"})
    # A phone number in a customer message (Ofcom drama range).
    at(412, sys_user_asst(
        "Please call me back about the failed payout, I'm on +44 7700 900461 until 6pm.",
        "We don't make outbound calls, but I can help here. The payout failed because the receiving bank rejected "
        "the account name. Update it under Payouts → Bank account and we'll retry automatically within 24 hours."))
    # Windows-1252 mojibake from the CSV export.
    at(731, sys_user_asst(
        "Iâ€™m trying to download last quarterâ€™s invoices as a single ZIP. Is that possible?",
        "Yes. In Billing → Invoices, filter by date, select all, and choose Download → ZIP. Up to 500 invoices "
        "can be downloaded at once."))
    at(1576, sys_user_asst(
        "The customer portal says the link has expired and I canâ€™t pay invoice INV-2026-04417.",
        "Payment links expire after 30 days. Open INV-2026-04417 and choose “Send new link”; the invoice number "
        "and amount stay the same."))
    # The agent closed two tickets without replying.
    for line in (1022, 1967):
        rec = train[line - 1]
        rec["messages"] = [m for m in rec["messages"] if m["role"] != "tool" and not m.get("tool_calls")]
        rec.pop("tools", None)
        rec["messages"] = rec["messages"][:2]
        rec["messages"][1] = {"role": "assistant", "content": ""}
    # A customer pasted a live-looking API key into the ticket.
    kr = random.Random(7)
    key = "sk-proj-" + "".join(kr.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-") for _ in range(48))
    at(2131, sys_user_asst(
        f"401 on every call. Our key: {key}",
        "Thanks — please rotate that key now, since it's been shared in this conversation. Create a new one under "
        "Developers → API keys, update your sync job, then revoke the old key. A 401 usually means the key was "
        "revoked or belongs to the other mode (test vs live)."))
    # The agent forwarded the invoice to a named person: a work email in the answer.
    at(1843, sys_user_asst(
        "Can you send the corrected invoice to our finance lead? Same address as last time.",
        "Done. I've re-sent the corrected invoice to priya.raman@contoso.com and copied the account owner. It "
        "replaces the earlier version, so the invoice number is unchanged."))
    at(2210, sys_user_asst(
        "My login email changed to jonas.weber@contoso.com — how do I update it?",
        "Go to Profile → Email, enter the new address and confirm it from the verification email. Your password "
        "and two-factor settings stay the same."))
    # The model called a tool the record never declares.
    rec = train[1449]
    inv = _inv(r)
    rec.clear()
    rec.update({"tools": TOOLS, "messages": [
        {"role": "user", "content": f"Please refund {inv}, the client cancelled."},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_7f3a91c2", "type": "function",
            "function": {"name": "issue_refund", "arguments": json.dumps({"invoice_id": inv})}}]},
        {"role": "tool", "tool_call_id": "call_7f3a91c2", "content": json.dumps({"invoice_id": inv, "status": "refunded"})},
        {"role": "assistant", "content": f"{inv} has been refunded in full. The client will see it in 5–10 business days."}]})
    # A webhook log pasted into a ticket.
    log = "\n".join(
        f"2026-10-{d:02d}T{h:02d}:{m:02d}:{s:02d}Z POST /webhooks/fabrikam 400 sig_mismatch event=invoice.paid "
        f"id=evt_{random.Random(d * 1000 + h * 60 + m).randrange(16**12):012x} retry={k % 5}"
        for k, (d, h, m, s) in enumerate((d, h, m, (m * 7) % 60) for d in range(1, 31) for h in range(0, 24, 3) for m in (5, 35)))
    at(2077, sys_user_asst(
        "Every invoice.paid webhook is failing since yesterday. Full log below:\n" + log,
        "Every failure is `sig_mismatch`, which means the body is being modified before you verify it. Verify the "
        "raw request bytes with the live-mode signing secret, then use Webhooks → Resend to replay the failed events."))
    # The export job ran twice for three tickets at the end of the file.
    train[2397] = json.loads(json.dumps(train[2201]))
    train[2399] = json.loads(json.dumps(train[903]))
    train[2405] = json.loads(json.dumps(train[1187]))
    return train, eval_rows


def dumps(rows):
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1])
    (out / "data").mkdir(parents=True, exist_ok=True)
    train, ev = build()
    (out / "data/train.jsonl").write_text(dumps(train))
    (out / "data/eval.jsonl").write_text(dumps(ev))
    print(len(train), len(ev))
