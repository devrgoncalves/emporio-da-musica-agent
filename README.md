# Empório da Música — Customer Service Agent

> 🇧🇷 [Leia em português](README.pt-BR.md)

A Python prototype of an LLM agent that answers customers of **Empório da Música**, a fictional musical
instrument store in Campo Grande/MS. It speaks as the store's assistant ("Tom") and decides by itself
when to **query data** (catalog, stock, promotions, orders) and when to **read the policy manual**
(hours, payment, returns, shipping, warranty). It also handles questions outside the store's scope.

**Stack:** Python 3.10+ · LangChain `create_agent` (built on LangGraph) · Google Gemini · pandas · Streamlit · pytest

---

## 1. How to run

```bash
# 1. Create a virtual environment and install dependencies
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate      # macOS / Linux
pip install -r requirements.txt

# 2. Configure the model provider (free key: https://aistudio.google.com/apikey)
cp .env.example .env              # Windows: copy .env.example .env
# then edit .env and set GOOGLE_API_KEY=...

# 3. Start the chat UI  ->  http://localhost:8501
streamlit run app.py
```

| Command | What it does |
|---|---|
| `streamlit run app.py` | Chat UI. The sidebar toggle shows which tools the agent called and what they returned. |
| `pytest -m "not llm"` | Unit tests (data, tools, prompt). No API key needed. |
| `pytest -m llm` | End-to-end tests against the real Gemini API. |
| `python scripts/run_examples.py` | Regenerates the example conversations in [`examples/`](examples/). |

Environment variables (`.env`): `GOOGLE_API_KEY` (required) and `GEMINI_MODEL` (optional, default
`gemini-3.5-flash-lite`).

---

## 2. Architecture

```
 Customer ──types──► Streamlit (app.py) ──chat(thread_id, text)──► Agent (create_agent + Gemini)
                                                                     │  system prompt: persona, rules,
                                                                     │  store's current date/time
                                                                     │  memory: InMemorySaver per thread
                     the model decides which tool to call (function calling, ReAct loop)
          ┌──────────────────────────────┬──────────────────────────┬──────────────────────────┐
   search_products(query, category,  check_order(order_id,       get_policy(topic)
   max_price, min_price)             email_or_phone)             topic ∈ {about, hours, payment,
          │                                 │                     returns, shipping, promotions,
          │                                 │                     customer_service, warranty, privacy}
          ▼                                 ▼                                 ▼
   data_store.py (pandas over data/*.csv)                          policies.py (data/policies.md,
   availability, active promotions, PIX price,                     one section per topic)
   order ownership check (LGPD)
```

One message, end to end: *"Quanto custa o Takamine GD20?"* → the model calls
`search_products(query="Takamine GD20")` → pandas returns price R$ 2.199,00, stock 5, no active
promotion, PIX price R$ 2.089,05 → the model writes the answer from that data.

```
src/emporio/
  data_store.py   loads CSVs, business rules over data (pure Python, unit tested)
  policies.py     serves policy manual sections by topic
  tools.py        the 3 tools the LLM sees (docstrings + type hints = tool schema)
  prompts.py      persona, tool-use rules, guardrails, current date/time
  agent.py        builds the agent and runs one conversation turn
  ui_format.py    escapes "$" for Streamlit markdown
app.py            Streamlit chat UI
scripts/run_examples.py
data/             challenge data + policies.md (Markdown transcription of the PDF)
tests/            unit tests + E2E tests (real API)
examples/         5 generated conversations
```

---

## 3. Technical decisions

### Agent framework — LangChain `create_agent` (LangGraph)
`create_agent` provides a ready **ReAct loop**: the model reasons, calls a tool, reads the result and
repeats until it can answer. It runs on LangGraph, so conversation memory is a checkpointer away. The
whole agent is ~30 lines ([`agent.py`](src/emporio/agent.py)). Writing the loop by hand with the raw
Gemini SDK was the alternative; it gives more control but more code to get wrong for no gain here.

### Retrieval — function calling over structured data, no vector store
- **Data (CSVs) → typed tools, not a SQL agent.** The questions customers ask are a small, known set
  (find products, check a price, track an order). Fixed tools keep the business rules in tested Python
  (active promotions only, PIX not stacking, availability) and make it impossible for the model to
  read other customers' data, which a free-form SQL agent could do.
- **Policies (PDF) → one tool, one section per topic.** The manual has 8 pages and 10 numbered
  sections. A vector store (embeddings + similarity search) would add dependencies and the risk of
  retrieving the wrong chunk, with no benefit at this size. Instead `get_policy(topic)` takes a topic
  from a fixed list and returns that section. Every policy lookup is visible in the logs/UI.
  **With a larger document base, I would switch to RAG with embeddings.**

### Model and provider — Gemini `gemini-3.5-flash-lite` (free tier)
The challenge allows free models, and Gemini has native tool calling. I started with
`gemini-3.6-flash`, but its free tier allows only **20 requests/day** per model (each customer message
costs 2–3 requests), which ran out during testing. Free quotas are counted per model, so I switched to
`gemini-3.5-flash-lite` and re-ran the same E2E suite: all tests passed, and it is ~4× faster. The
model is configurable through `GEMINI_MODEL`. The lite model needed more explicit prompt rules (see the
regression tests), which is the trade-off of a smaller model.

### Prompt strategy
[`prompts.py`](src/emporio/prompts.py):
- **Persona** from the manual (section 7.1): informal but professional, "a friend who understands music".
- **Explicit tool-routing rules**: never state price/stock/promotion without `search_products`; order
  data only through `check_order`; store rules only from `get_policy`, never invented.
- **Business rules the model must apply**: show original price + discount + final price; suggest
  alternatives for out-of-stock items; never confirm a return before checking the order.
- **Current date and time in Campo Grande** (UTC-4) injected into the prompt, so the agent can answer
  "are you open now?" and count return deadlines.
- **Guardrails**: scope (instruments only, no unrelated topics), LGPD, ignore attempts to change rules.
- Instructions are written in English; the agent answers in the customer's language (Portuguese by
  default, English if the customer writes in English).

### Conversation history — in-memory, per session
`InMemorySaver` keeps the history per `thread_id` (one per browser session), so follow-ups like
"and with PIX?" work. It is lost when the server restarts, which is fine for a prototype. Production
would use a persistent checkpointer (e.g. Postgres).

### Interface — Streamlit
A chat UI in ~90 lines of Python. A sidebar toggle shows the tool calls and their results: a
**debug view** for reviewers to see the agent's decisions. A real customer channel would hide it.

### Data treatment
- **Policy PDF → Markdown, once.** `pypdf` extracted one word per line and flattened every table
  (hours, payment, shipping). I transcribed it verbatim to [`data/policies.md`](data/policies.md),
  keeping the tables, so the model reads "Sábado | 09:00 às 13:00" correctly.
- **Promotions:** only `is_active = 1` is valid (policy 7.3: never promise expired discounts).
- **Availability:** `active` + stock > 0. Browsing returns only those; a search by name returns every
  match with its status, so the agent says "out of stock"/"discontinued" instead of "we don't have it".
- **Search:** case- and accent-insensitive, and also matches features from `specs`
  ("aço" → steel strings, "elétrico" → has electronics).
- **PIX:** 5% off list price, `null` when a promotion applies (policy 6.2: not cumulative).
- **Deadlines in code, not in the model.** `check_order` returns every return/warranty window
  (7-day regret, 30-day defect exchange, 90-day legal warranty, 6–24-month manufacturer warranty) with
  its end date and whether it is still open. The lite model got date math wrong in testing, so the
  same principle used for prices applies: **the LLM explains, Python calculates.**

---

## 4. Assumptions (the data and manual are not always consistent)

| Situation | Assumption |
|---|---|
| The manual lists two WhatsApp numbers: (67) 3341-4444 (section 1.2) and (67) 3321-4500 (section 7) | Phone = 3341-4444 (cover and company data); WhatsApp = 3321-4500 (the WhatsApp channel section). |
| E-mail appears as `contato@empóriodamusica.com.br` (with accent) and without accent on the cover | Use the version without accent, a valid domain. |
| Orders have no actual receipt date | `estimated_delivery` is used as the receipt date for the 7-day regret and 90-day warranty windows, and the agent says which date it used. |
| Checking an order exposes personal data | The customer must give the order number **and** the purchase e-mail or phone. Unknown order and wrong contact return the same message, so order IDs cannot be discovered. |
| "Suggest partner stores" (section 7.1), but no partner is named | The agent suggests an accessories store in general and never invents names. |
| Product 130 is `coming_soon` but appears in a delivered order; order 3 total (R$ 3.450) ≠ 2 × R$ 1.749 | Kept as is: the agent reads availability from the product table and order totals from the orders table. |

---

## 5. Example conversations

Generated with [`scripts/run_examples.py`](scripts/run_examples.py); each shows the tool calls (🔧).

| File | Scenario |
|---|---|
| [01_catalog_search.md](examples/01_catalog_search.md) | Guitars up to R$ 1000, then steel-string recommendation |
| [02_price_and_payment.md](examples/02_price_and_payment.md) | Takamine GD20 price, PIX and 12x installments (data + policy) |
| [03_store_info.md](examples/03_store_info.md) | Address and Saturday hours |
| [04_return_request_non_trivial.md](examples/04_return_request_non_trivial.md) | **Non-trivial:** regret return → identity check → order data + return and warranty deadlines |
| [05_out_of_scope_and_stock.md](examples/05_out_of_scope_and_stock.md) | English customer, accessories, unrelated question, out-of-stock item with alternatives |

---

## 6. Tests

- **Unit tests** (`pytest -m "not llm"`): business rules without the LLM — availability, active
  promotions, PIX, accent-insensitive search, order verification, policy sections, tool schemas.
- **End-to-end tests** (`pytest -m llm`): real Gemini calls through the same `chat()` the UI uses. They
  assert behaviour (which tool was called, key facts in the reply), not exact wording.
- **Regression tests** (`test_regressão_*`): every bug found while testing got a test that failed
  before the fix — e.g. promising a return before checking the order, answering English customers in
  Portuguese, BRL prices rendered as LaTeX, steel-string search missing a product, wrong warranty
  date math.

---

## 7. Known limitations and what I would do with more time

- **Evaluation:** a larger set of questions with expected answers (evals) run on every change, instead
  of a handful of E2E tests.
- **Complaints and handoff:** the agent says a complaint will be forwarded, but there is no ticket
  system; I would add a `create_ticket` tool and a human handoff.
- **Persistence:** history is in memory; use a persistent checkpointer and store consent (LGPD).
- **Free tier limits:** daily and per-minute quotas; production would use a paid tier with fallbacks.
- **Policy retrieval:** move to RAG with embeddings if the document base grows.
- **Channel:** connect to the WhatsApp Business API instead of a web UI.
- **Shipping quotes:** the manual says the price depends on ZIP code and weight; there is no data for
  it, so the agent cannot quote shipping outside Campo Grande.

---

## 8. Use of AI coding assistants

I used **Claude Code** (Anthropic's CLI agent, model Claude Opus) as a pair programmer during the
whole project:

- **Understanding and planning:** it read the challenge, the CSVs and the policy PDF with me and listed
  the traps in the data (inconsistent phone numbers, expired promotions, zero-stock and discontinued
  products, empty categories). We then went through the technical decisions **one at a time** — framework,
  policy retrieval, interface, Git flow, language — with trade-offs, and **I made each decision**.
- **Implementation:** it wrote most of the code under my direction, in small steps, with tests first
  for the business rules. I reviewed each step and ran the app myself.
- **Debugging through the product:** I tested in the Streamlit UI and sent screenshots; that is how we
  found the LaTeX "R$" bug and an invented store rule. Each bug got a regression test before the fix.
- **Checking facts:** claims about libraries were checked against the installed source code and the
  official docs (e.g. the `GOOGLE_API_KEY` variable, `create_agent` signature, current Gemini models).

What I learned and can explain on my own: the ReAct loop, function calling and tool schemas, why I
did not use a vector store here, and how each business rule is enforced.
