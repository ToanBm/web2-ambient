# Web2 Developer Loop — Micro-Challenge #6
## "Detect and handle refusal programmatically."

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
python detect_refusal.py
python detect_refusal.py --show-response
python detect_refusal.py --prompt "I have 2 months of data and unknown liabilities. Should I invest all my savings right now?"
```

Optional params:
- `--model` — override `AMBIENT_MODEL`
- `--temperature` — default `0.0`
- `--max-tokens` — default `256`
- `--show-response` — print full model response
- `--review-file` — path for refusal queue JSONL (default `data/human_review.jsonl`)

Refused responses are logged to `data/human_review.jsonl`.

## Run the full client (streaming + benchmarking)

```bash
python main.py
```

Enable benchmarking in `.env`:

```
BENCH_ENABLED=true
BENCH_RUNS=3
```

## Project Structure

```
web2-dev/
├── detect_refusal.py        # Web2 #6 — refusal detection & routing
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
        ├── bench.py         # Benchmarking utilities
        ├── prompt.py        # Prompt loader
        ├── provider_utils.py
        └── runner.py        # Orchestration
```

## How refusal was detected

Lexical scan using regex patterns across four categories. The `detect_refusal()` function
counts hits per category and returns the dominant `RefusalState` with a confidence score
(0.60–0.97 based on number of matched patterns).

## How it was handled downstream

`route()` branches on the decision: `ANSWERED` responses log normally; any refusal state
calls `_escalate()`, which writes a structured record to `data/human_review.jsonl`. In
production, swap `_escalate()` for a Slack webhook, database write, or dead-letter queue.
Refusals are never silently dropped.
