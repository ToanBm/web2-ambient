# Web2 Developer Loop — Micro-Challenges #1 through #7

## Quick Start

**Requirements:** Python 3.10+, pip, an [Ambient API key](https://app.ambient.xyz/keys)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure environment
cp .env.example .env
# Edit .env — set AMBIENT_API_KEY at minimum
```

To unlock the OpenAI comparison features (challenges #2 `--compare` and #4):
```bash
# Also set in .env:
OPENAI_ENABLED=true
OPENAI_API_KEY=your-openai-key-here
```

**Verify setup — run each challenge in order:**

```bash
python3 first_inference.py          # #1 — first call + receipt
python3 stream_response.py          # #2 — live streaming + latency
python3 verify_receipt.py --generate --tamper  # #3 — verify + tamper demo
python3 benchmark.py                # #4 — cost + latency table
python3 split_layers.py             # #5 — verifiable/interpretive split
python3 detect_refusal.py           # #6 — refusal detection
python3 expose_identity.py          # #7 — system identity card
```

---

## Challenge #7 — "Expose system identity in your app."

Prints a structured terminal interface — the **system identity card** — that
separates what Ambient guarantees from what it does not.  Optionally calls
the live API and attaches a per-response identity summary to the output.

### Guaranteed properties (7)

| Property | What it means |
|---|---|
| Cryptographic receipt | Every call produces an `events_hash` + `payload_hash`; any post-save mutation breaks verification |
| Declared model ID in every receipt | Model name is stamped into the receipt and cannot change without hash mismatch |
| Token usage reported | Prompt + completion counts are in the SSE stream, enabling cost auditing |
| Decentralised compute | Inference routes to GPU miners, not a single cloud provider |
| Proof of Logits (PoL) attestation | On-chain record that the declared model was run (settlement is async) |
| Open-weights model | Only publicly auditable checkpoints are served |
| Response integrity within a session | The streamed response matches the receipt; mutations detected by `verify_receipt.py` |

### Not-guaranteed properties (8)

| Property | Why it cannot be guaranteed |
|---|---|
| Which miner ran your inference | Routing is anonymised — not exposed in the API response |
| On-chain PoL settled during this session | Settlement is async; may lag behind the API response |
| Declared model ID matches weights executed | No in-band cryptographic proof of the exact weights used |
| Response not served from cache | A cached hit is structurally identical to a fresh completion |
| Response accuracy or factual correctness | Models hallucinate; verify claims independently |
| Real-time or up-to-date data | Open-weights models have a training cutoff |
| Financial, medical, or legal advice | Model outputs are not professional advice |
| Uptime or latency SLA | Decentralised miner availability and throughput vary |

### Run the system identity card (Web2 #7 deliverable)

```bash
# Print the static identity card (no API key required)
python3 expose_identity.py

# Call the API and attach a per-response identity summary
python3 expose_identity.py --live

# Custom prompt
python3 expose_identity.py --live --prompt "What is Solana?"

# Machine-readable JSON output (skip the card, emit JSON record)
python3 expose_identity.py --live --json --no-card
```

Optional params:
- `--live` — call the Ambient API and attach a per-response identity summary
- `--prompt` — prompt to send (default: asks the model to describe Ambient)
- `--model` — override `AMBIENT_MODEL`
- `--temperature` — default `0.0`
- `--max-tokens` — default `512`
- `--json` — emit a machine-readable JSON identity record instead of the terminal UI
- `--no-card` — skip the static identity card

### How the boundary was communicated

The interface has two layers:

**Static layer** (`print_identity_card()`): rendered at startup from two hard-coded
lists — `GUARANTEES` and `NOT_GUARANTEED` — each containing a short label and a
plain-English explanation.  No API call is needed.  The card uses `✓` (green) for
guarantees and `✗` (red) for non-guarantees, with dim-coloured detail text so the
boundary is scannable at a glance.

**Live layer** (`print_response_identity()`): after each real API call, a per-response
summary is appended.  It shows which guarantees were confirmed for this specific
call (model declared, local `events_hash`, receipt checkable, response non-empty,
token usage) alongside the properties that remain unverifiable (miner identity, PoL
on-chain settlement, actual weights, cache status, factual accuracy).

The `--json` flag emits the same information as a structured record — ready to be
attached to a log entry, UI tooltip, or API response envelope.

### Design principles applied

1. **Don't hide the system.** Every guarantee and every limitation is listed at the
   same level of prominence — nothing is buried in docs.
2. **Anchor to prior challenges.** The guarantee list references `verify_receipt.py`
   (#3), `split_layers.py` (#5), and `detect_refusal.py` (#6) so users know which
   tools enforce which properties.
3. **Separate static from dynamic.** The identity card is always available offline;
   the live summary augments it with per-call evidence.
4. **Machine-readable.** `--json` mode outputs the same identity as a JSON object,
   so the boundary can be surfaced in any UI — terminal, web, or API envelope.

---

## Challenge #6 — "Detect and handle refusal programmatically."

Calls Ambient via the streaming API, detects whether the model refused to answer,
and routes refused responses to a human review queue.

## Refusal States

| State | Meaning |
|---|---|
| `ANSWERED` | Model gave a confident response |
| `REFUSED_INSUFFICIENT_DATA` | Model said it lacks real-time or enough data |
| `REFUSED_AMBIGUOUS` | Model found the request unclear or multi-interpretable |
| `REFUSED_UNCERTAIN` | Model expressed doubt or declined to give financial/definitive advice |

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env — set AMBIENT_API_KEY
```

## Run refusal detection (Web2 #6 deliverable)

```bash
python3 detect_refusal.py
python3 detect_refusal.py --show-response
python3 detect_refusal.py --prompt "I have 2 months of data and unknown liabilities. Should I invest all my savings right now?"
```

Optional params:
- `--model` — override `AMBIENT_MODEL`
- `--temperature` — default `0.0`
- `--max-tokens` — default `256`
- `--show-response` — print full model response
- `--review-file` — path for refusal queue JSONL (default `data/human_review.jsonl`)

Refused responses are logged to `data/human_review.jsonl`.

## How refusal was detected

Lexical scan using regex patterns across four categories. The `detect_refusal()` function
counts hits per category and returns the dominant `RefusalState` with a confidence score
(0.60–0.97 based on number of matched patterns).

## How it was handled downstream

`route()` branches on the decision: `ANSWERED` responses log normally; any refusal state
calls `_escalate()`, which writes a structured record to `data/human_review.jsonl`. In
production, swap `_escalate()` for a Slack webhook, database write, or dead-letter queue.
Refusals are never silently dropped.

---

## Challenge #5 — "Split a response into verifiable and non-verifiable layers."

Calls Ambient and classifies each sentence of the response as one of three layers:

| Layer | Meaning |
|---|---|
| `VERIFIABLE` | Math, logic, definitions, measurements — can be checked externally |
| `INTERPRETIVE` | Summaries, advice, opinion, hedged language — subjective |
| `MIXED` | Contains signals from both layers |

## Run layer splitting (Web2 #5 deliverable)

```bash
python3 split_layers.py
python3 split_layers.py --show-response
python3 split_layers.py --prompt "Explain how RSA encryption works and give a key-size example."
```

Optional params: `--model`, `--api-url`, `--api-key`, `--max-tokens`, `--show-response`

## How the boundary was detected

Each sentence is scanned against two independent pattern sets:
- **Verifiable signals**: number–operator pairs, `=` followed by currency/digits, measurement units (%/$/kg), logical connectives (`therefore`, `hence`), formal definitions, inline code
- **Interpretive signals**: hedging adverbs (`typically`, `generally`), uncertainty modals (`might`, `could`), opinion markers (`I think`), recommendation verbs (`should`, `suggest`), summary phrases (`in summary`)

A sentence matching only verifiable patterns → `VERIFIABLE`.
A sentence matching only interpretive patterns → `INTERPRETIVE`.
A sentence matching both → `MIXED` (e.g., "You'll likely earn **$52.50** in year 2").

---

## Challenge #4 — "Cost + latency reality check."

Runs the same prompt through **Ambient** and **OpenAI** with identical constraints,
then prints a side-by-side comparison table covering cost, latency, and reliability.

| Metric | Description |
|---|---|
| Avg TTFB | Time to first byte (ms) |
| Avg TTC | Total time to completion (ms) |
| Prompt / Completion tokens | Average token counts across runs |
| Est. cost / call | Computed from public per-token rates |
| Stalls | Stream gaps exceeding threshold |
| Errors | Failed runs out of total |

## Setup (Challenge #4)

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env — set AMBIENT_API_KEY and OPENAI_API_KEY, set OPENAI_ENABLED=true
```

## Run benchmark (Web2 #4 deliverable)

```bash
python3 benchmark.py
python3 benchmark.py --runs 5
python3 benchmark.py --prompt "Explain RSA encryption briefly."
python3 benchmark.py --max-tokens 256 --temperature 0.0
```

Optional params: `--runs`, `--warmup`, `--prompt`, `--max-tokens`, `--temperature`

Cost rates default to known public pricing. Override per-provider in `.env`:
```
OPENAI_INPUT_RATE=0.15
OPENAI_OUTPUT_RATE=0.60
AMBIENT_INPUT_RATE=0.60
AMBIENT_OUTPUT_RATE=0.20
```

---

## Challenge #3 — "Verify or reject an inference."

Parses a saved Ambient inference receipt and runs a chain of integrity checks.
Demonstrates both a successful verification and a simulated tamper rejection.

| Check | What it tests |
|---|---|
| `events_hash` | `sha256(raw_events)` matches the stored hash — detects any post-save mutation |
| `payload_hash` | Request payload hash is present and well-formed |
| `event parsing` | Every SSE event deserializes as valid JSON |
| `content` | Response text can be fully reconstructed from deltas |
| `usage` | Token counts were reported in the stream |

## What verification guarantees — and what it does not

**Guarantees:**
- The receipt was not modified after it was saved
- The response was structurally valid (parseable SSE, non-empty content)
- Token usage was reported by the API

**Does not guarantee:**
- Which specific miner ran the inference
- That PoL settled on-chain (that is internal to Ambient's network)
- That the declared model ID matches the weights actually used
- That the response was not served from a cache

## Run receipt verification (Web2 #3 deliverable)

```bash
# Generate a fresh receipt from Ambient, then verify it
python3 verify_receipt.py --generate

# Verify the most recent saved receipt
python3 verify_receipt.py

# Verify a specific receipt file
python3 verify_receipt.py data/receipts/your-receipt.json

# Simulate a tampered receipt → shows REJECTED
python3 verify_receipt.py --tamper
python3 verify_receipt.py --generate --tamper
```

Optional params: `--receipt-dir`, `--max-tokens`, `--temperature`

To save receipts automatically from `main.py`, set in `.env`:
```
AMBIENT_RECEIPT_SAVE=true
AMBIENT_RECEIPT_DIR=data/receipts
```

---

## Challenge #2 — "Stream a response end-to-end."

Streams a live response from Ambient, printing tokens as they arrive,
and measures **time to first token (TTFB)** and **time to completion (TTC)**.

## Run streaming (Web2 #2 deliverable)

```bash
python3 stream_response.py
python3 stream_response.py --prompt "Explain RSA encryption briefly."
python3 stream_response.py --max-tokens 128 --temperature 0.7
```

## Stretch goal — compare with OpenAI

```bash
# Requires OPENAI_ENABLED=true and OPENAI_API_KEY in .env
python3 stream_response.py --compare
```

Runs the same prompt through both providers sequentially, then prints a latency summary:

```
  Metric               Ambient          OpenAI
  ─────────────────── ─────────────── ───────────────
  TTFB                      871 ms          743 ms
  TTC                     9,175 ms        5,873 ms
  Tokens (out)                 187             201
  Stalls                         0               0
```

Optional params: `--prompt`, `--max-tokens`, `--temperature`, `--compare`

---

## Challenge #1 — "Make your first verified inference call."

Sends a prompt to the Ambient API, prints the response live,
then saves and inspects the cryptographic receipt.

Standalone script — no internal dependencies beyond `requests`.

## Run first inference (Web2 #1 deliverable)

```bash
python3 first_inference.py
python3 first_inference.py --prompt "What is Solana?"
```

Output:
```
[Model]   zai-org/GLM-4.6
[Prompt]  What is Ambient Network on Solana?

Ambient is a Solana-compatible proof-of-work Layer 1...

──────────────────────────────────────────────────
  RECEIPT
──────────────────────────────────────────────────
  File     : data/receipts/1234_zai-org_GLM-4.6_abc123.json
  Model    : zai-org/GLM-4.6
  Saved    : 2026-02-24 12:34:56 UTC
  Events   : 187
  Hash     : b0ee9c52162cea2b...
  Payload  : 9885da34ebc9ed83...
──────────────────────────────────────────────────
```

Optional params: `--prompt`, `--max-tokens`

---

## Run the full client (streaming + benchmarking)

```bash
python3 main.py
```

Enable benchmarking in `.env`:

```
BENCH_ENABLED=true
BENCH_RUNS=3
```

## Project Structure

```
web2-dev/
├── first_inference.py       # Web2 #1 — first inference call + receipt inspection
├── stream_response.py       # Web2 #2 — live streaming + TTFB/TTC measurement
├── verify_receipt.py        # Web2 #3 — receipt integrity verification
├── benchmark.py             # Web2 #4 — cost + latency comparison table
├── split_layers.py          # Web2 #5 — verifiable/interpretive layer split
├── detect_refusal.py        # Web2 #6 — refusal detection & routing
├── expose_identity.py       # Web2 #7 — system identity card
├── main.py                  # Full streaming client entry point
├── requirements.txt
├── .env.example
└── ambient_client/
    ├── config.py            # .env loader
    ├── env_loader.py        # .env parser
    ├── streaming.py         # SSE streaming + receipt saving
    ├── utils.py             # Helpers
    └── app/
        ├── ambient.py       # Ambient provider config
        ├── openai_provider.py  # OpenAI provider config
        ├── bench.py         # Benchmarking utilities
        ├── prompt.py        # Prompt loader
        ├── provider_utils.py
        └── runner.py        # Orchestration
```
