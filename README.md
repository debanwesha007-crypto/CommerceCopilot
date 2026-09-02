# CommerceCopilot

An agent-readable merchant catalog and conversational checkout assistant,
built for the Razorpay AI Buildathon (Agentic Commerce track).

Renaming: "CommerceCopilot" is a placeholder — rename freely, it only
appears in this README and a couple of UI strings.

## What it does

A chat interface lets a buyer search a merchant's catalog, compare products,
ask review-grounded questions, add items to a cart, and check out — with
every step logged and every payment call bounded and explicitly confirmed.

Flow: **search → compare → ask_reviews (RAG) → add_to_cart → checkout (Razorpay test mode)**

## Architecture

```
Browser (Next.js chat UI)
        │
        ▼
FastAPI backend  ──►  Orchestrator
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      Rule-based agent      LLM agent (optional)
      (default, no key      (LangChain + Claude,
       required, always      only if USE_LLM_AGENT=true
       works)                 + ANTHROPIC_API_KEY set)
              │                     │
              └──────────┬──────────┘
                          ▼
                  6 shared tools
       search_products · get_product_details
       compare_products · ask_reviews
       add_to_cart · checkout
                          │
        ┌────────┬────────┴────────┬────────────┐
        ▼        ▼                 ▼             ▼
   Catalog    RAG review     Razorpay client   Audit log
   (JSON)     engine (TF-IDF) (mock or live,   (JSONL file,
                              test mode)        every call)
```

### Why it's built this way (worth keeping for the pitch video)

- **Rule-based agent is the default, not a fallback afterthought.** It has
  zero external dependencies and always works — important on demo day when
  wifi or an API key can fail you mid-pitch. The LLM agent is an upgrade
  layered on top, and if it throws for any reason, the orchestrator silently
  drops back to the rule-based path instead of breaking the conversation.
  This is a real, working instance of the "Failure Recovery" criterion, not
  just a slide about it.
- **Checkout is bounded and gated in code, not in a prompt.** The `checkout`
  tool refuses to run without `confirmed=true`, and refuses above a spend
  cap, regardless of what any agent (rule-based or LLM) asks it to do.
- **RAG answers are grounded or they don't answer.** `ask_reviews` returns
  `grounded: false` and a plain "I don't have a review-grounded answer for
  that" instead of letting anything improvise from a weak match.
- **Razorpay is mocked by default, same interface as the real client.**
  `RazorpayMockClient` and `RazorpayLiveClient` share method signatures —
  flip `RAZORPAY_LIVE=true` with real test-mode keys and nothing else in the
  app needs to change.

## Running it

### Backend

```bash
cd backend
pip install -r requirements.txt
# optional: cp .env.example .env and edit if you want the LLM agent or live Razorpay
uvicorn app.main:app --reload --port 8000
```

Runs fully offline by default: rule-based agent, mocked Razorpay, TF-IDF RAG
over the seeded catalog in `app/data/catalog.json`. No API keys required.

To enable the LLM agent:
```bash
pip install langchain-core langchain-anthropic
export USE_LLM_AGENT=true
export ANTHROPIC_API_KEY=sk-ant-...
```

To hit real Razorpay test mode instead of the mock:
```bash
pip install razorpay
export RAZORPAY_LIVE=true
export RAZORPAY_KEY_ID=rzp_test_...
export RAZORPAY_KEY_SECRET=...
```
Note: `RazorpayLiveClient.capture_payment` is a documented extension point —
a real conversational-checkout flow needs Razorpay Checkout on the client
plus signature verification on the backend, which is out of scope for this
prototype's mocked flow.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:3000. Set `NEXT_PUBLIC_API_BASE` (see
`.env.local.example`) if the backend isn't on `localhost:8000`.

### Try it

1. "find me a laptop under 50000"
2. "how is the battery life of the AeroBook 14 Slim Laptop" (RAG, cites reviews)
3. "add AeroBook 14 Slim Laptop to cart"
4. "checkout" → confirm in the UI or type "yes"

### Testing tools directly

Each tool is also exposed as its own REST endpoint under `/api/tools/*` for
debugging without going through the chat agent, and `/api/tools/audit_log`
returns the full audit trail as JSON.

## What's mocked vs. real

| Piece | Status |
|---|---|
| Catalog | Seeded JSON, 12 products — swap for a real merchant catalog API |
| RAG retrieval | TF-IDF (offline, zero setup) — swap for embeddings for better recall |
| Payments | Mocked Razorpay client with simulated failures — swap for live test-mode keys |
| Cart/session | In-memory — swap for Postgres/Supabase for persistence |
| LLM agent | Optional, gated behind env vars — off by default |

## Known failure mode found during the build

Adding "AeroBook 14 Slim Laptop" to cart via chat initially added 14 units —
the quantity regex was matching the "14" in the product name itself. Fixed
by requiring an explicit quantity marker (`qty 2`, `x3`, `2 units`) instead
of matching any bare number in the message. This is exactly the kind of
"what broke and how you fixed it" note the buildathon brief asks for —
swap in your own if you hit a different one while extending this.
