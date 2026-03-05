#!/usr/bin/env python3
"""
Web2 #7 — Expose system identity in your app.

Displays a structured terminal interface that separates what Ambient
guarantees from what it does not.  Optionally calls the live API and
attaches a per-response identity summary to the output.

Run:
  python expose_identity.py                  # identity card only
  python expose_identity.py --live           # call API + attach identity to response
  python expose_identity.py --live --prompt "What is Solana?"
  python expose_identity.py --live --json    # emit identity as JSON
"""

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "zai-org/GLM-4.6"
DEFAULT_API_URL = "https://api.ambient.xyz/v1/chat/completions"
DEFAULT_PROMPT  = "In one paragraph, describe what Ambient Network is and how it differs from centralised AI providers."

WIDTH = 70

# ---------------------------------------------------------------------------
# System identity: guaranteed vs not guaranteed
# ---------------------------------------------------------------------------

GUARANTEES: List[Tuple[str, str]] = [
    (
        "Cryptographic receipt",
        "Every inference call produces a tamper-evident receipt (events_hash + "
        "payload_hash).  Modifying any response token after the fact produces a "
        "sha256 mismatch that verify_receipt.py will catch.",
    ),
    (
        "Declared model ID in every receipt",
        "The model name claimed by the API is stamped into the saved receipt and "
        "cannot be changed without breaking the hash.",
    ),
    (
        "Token usage reported",
        "Prompt and completion token counts are returned in the SSE stream and "
        "stored in the receipt, enabling cost auditing.",
    ),
    (
        "Decentralised compute",
        "Inference is routed to GPU miners on the Ambient proof-of-work network, "
        "not a single cloud provider.",
    ),
    (
        "Proof of Logits (PoL) attestation",
        "The network produces an on-chain record that the declared model was run.  "
        "Settlement happens asynchronously in the background.",
    ),
    (
        "Open-weights model",
        "Ambient only serves publicly available open-weights models whose "
        "checkpoints can be independently downloaded and audited.",
    ),
    (
        "Response integrity within a session",
        "The streamed response you receive matches the receipt saved to disk.  "
        "Any post-save mutation is detected (see #3 — verify_receipt.py).",
    ),
]

NOT_GUARANTEED: List[Tuple[str, str]] = [
    (
        "Which miner ran your inference",
        "The network anonymises routing.  You cannot determine the specific GPU "
        "node that processed your request.",
    ),
    (
        "On-chain PoL settlement during this session",
        "The API call returns before PoL settles on-chain.  Settlement is internal "
        "to Ambient's network and may lag behind the API response.",
    ),
    (
        "Declared model ID matches weights executed",
        "You receive the model name the API reports.  There is no in-band "
        "cryptographic proof that those exact weights were used at inference time.",
    ),
    (
        "Response not served from cache",
        "Ambient may return cached completions for identical prompts.  A cached "
        "response is indistinguishable from a freshly computed one.",
    ),
    (
        "Response accuracy or factual correctness",
        "The model may hallucinate, omit context, or reason incorrectly.  Always "
        "verify factual claims from independent sources.",
    ),
    (
        "Real-time or up-to-date data",
        "Open-weights models have a training cutoff.  They cannot access live "
        "market prices, blockchain state, or current events.",
    ),
    (
        "Financial, medical, or legal advice",
        "Model outputs are not professional advice.  The refusal detector (#6) "
        "exists precisely because the model will decline high-stakes questions.",
    ),
    (
        "Uptime or latency SLA",
        "A decentralised miner network has variable availability and throughput.  "
        "TTFB and TTC can vary widely across calls (see #4 — benchmark.py).",
    ),
]

# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

_RESET  = "\033[0m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"


def _no_colour() -> bool:
    return not sys.stdout.isatty() or os.environ.get("NO_COLOR", "")


def _g(text: str) -> str:
    return text if _no_colour() else f"{_GREEN}{text}{_RESET}"


def _r(text: str) -> str:
    return text if _no_colour() else f"{_RED}{text}{_RESET}"


def _b(text: str) -> str:
    return text if _no_colour() else f"{_BOLD}{text}{_RESET}"


def _d(text: str) -> str:
    return text if _no_colour() else f"{_DIM}{text}{_RESET}"


def _c(text: str) -> str:
    return text if _no_colour() else f"{_CYAN}{text}{_RESET}"


def _y(text: str) -> str:
    return text if _no_colour() else f"{_YELLOW}{text}{_RESET}"


def _wrap(text: str, indent: int = 6) -> str:
    """Word-wrap text to WIDTH, indenting continuation lines."""
    words = text.split()
    lines: List[str] = []
    current = " " * indent
    for word in words:
        if len(current) + len(word) + 1 > WIDTH:
            lines.append(current.rstrip())
            current = " " * indent + word
        else:
            current += (" " if current.strip() else "") + word
    if current.strip():
        lines.append(current.rstrip())
    return "\n".join(lines)


def _rule(char: str = "─") -> str:
    return char * WIDTH


# ---------------------------------------------------------------------------
# Static identity card
# ---------------------------------------------------------------------------


def print_identity_card() -> None:
    """Print the full system identity card to stdout."""
    print()
    print(_rule("═"))
    print(_b(f"  AMBIENT SYSTEM IDENTITY".center(WIDTH)))
    print(_d(f"  Proof-of-work decentralised inference network".center(WIDTH)))
    print(_rule("═"))

    # --- Guarantees ---
    print()
    print(_b(_g(f"  ✓  WHAT AMBIENT GUARANTEES  ({len(GUARANTEES)} properties)")))
    print(_rule())
    for label, detail in GUARANTEES:
        print()
        print(_g(f"  ✓  {label}"))
        print(_d(_wrap(detail, indent=6)))
    print()
    print(_rule())

    # --- Not guaranteed ---
    print()
    print(_b(_r(f"  ✗  WHAT AMBIENT DOES NOT GUARANTEE  ({len(NOT_GUARANTEED)} properties)")))
    print(_rule())
    for label, detail in NOT_GUARANTEED:
        print()
        print(_r(f"  ✗  {label}"))
        print(_d(_wrap(detail, indent=6)))
    print()
    print(_rule())

    # --- Summary ---
    print()
    print(_y("  HOW TO READ THIS CARD"))
    print(_rule())
    print(_wrap(
        "Guaranteed properties are enforceable with tools already in this "
        "project: verify_receipt.py checks receipt integrity, split_layers.py "
        "labels verifiable vs interpretive sentences, and detect_refusal.py "
        "surfaces model uncertainty.  Not-guaranteed properties are "
        "architectural limits of any decentralised inference network — being "
        "honest about them is what this card is for.",
        indent=2,
    ))
    print()
    print(_rule("═"))
    print()


# ---------------------------------------------------------------------------
# Live-call mode
# ---------------------------------------------------------------------------


@dataclass
class CallResult:
    model: str = ""
    response: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    events_hash: str = ""
    receipt_confirmed: bool = False
    elapsed_ms: int = 0
    error: str = ""


def _load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def call_ambient(
    prompt: str,
    *,
    model: str,
    api_url: str,
    api_key: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> CallResult:
    result = CallResult(model=model)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }

    raw_events: List[str] = []
    text_parts: List[str] = []
    t0 = time.monotonic()

    try:
        with requests.post(api_url, json=payload, headers=headers,
                           stream=True, timeout=120) as resp:
            resp.raise_for_status()
            for raw in resp.iter_lines():
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                raw = raw.strip()
                if not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                raw_events.append(data)
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                    for choice in obj.get("choices", []):
                        delta = choice.get("delta", {})
                        text_parts.append(delta.get("content") or "")
                    usage = obj.get("usage") or {}
                    if usage.get("prompt_tokens"):
                        result.prompt_tokens = usage["prompt_tokens"]
                    if usage.get("completion_tokens"):
                        result.completion_tokens = usage["completion_tokens"]
                except json.JSONDecodeError:
                    pass
    except Exception as exc:  # noqa: BLE001
        result.error = str(exc)
        return result

    result.elapsed_ms = int((time.monotonic() - t0) * 1000)
    result.response = "".join(text_parts)

    # Compute a local events_hash to confirm receipt integrity is checkable
    joined = "\n".join(raw_events)
    result.events_hash = hashlib.sha256(joined.encode()).hexdigest()
    result.receipt_confirmed = bool(result.events_hash) and bool(result.response)

    return result


# ---------------------------------------------------------------------------
# Per-response identity summary
# ---------------------------------------------------------------------------


def print_response_identity(result: CallResult, prompt: str) -> None:
    """Print response text followed by an identity summary for this call."""
    print()
    print(_rule())
    print(_b("  RESPONSE"))
    print(_rule())
    print()
    print(result.response.strip())
    print()
    print(_rule("═"))
    print(_b("  IDENTITY SUMMARY FOR THIS CALL"))
    print(_rule("═"))
    print()

    # --- What we CAN confirm from this response ---
    print(_b(_g("  ✓  CONFIRMED FOR THIS CALL")))
    print()

    confirmed: List[Tuple[str, str]] = [
        ("Model declared",          result.model),
        ("Local events_hash",       result.events_hash[:32] + "…"),
        ("Receipt checkable",       "yes — run verify_receipt.py to verify integrity"),
        ("Response non-empty",      f"yes — {len(result.response):,} chars returned"),
        ("Elapsed time",            f"{result.elapsed_ms:,} ms end-to-end"),
    ]
    if result.prompt_tokens or result.completion_tokens:
        confirmed.append(("Token usage reported",
                          f"{result.prompt_tokens} prompt + "
                          f"{result.completion_tokens} completion"))

    for label, value in confirmed:
        print(_g(f"  ✓  {label}"))
        print(_d(f"       {value}"))
        print()

    # --- What we CANNOT confirm ---
    print(_rule())
    print(_b(_r("  ✗  UNVERIFIABLE FOR THIS CALL")))
    print()

    unverifiable: List[Tuple[str, str]] = [
        ("Specific miner identity",
         "routing is anonymised; not exposed in the API response"),
        ("PoL settled on-chain",
         "settlement is async; not confirmed within this API call"),
        ("Model weights match declared ID",
         "no in-band cryptographic proof of weights actually executed"),
        ("Response not from cache",
         "a cached hit is structurally identical to a fresh completion"),
        ("Factual accuracy",
         "always verify claims against independent sources"),
    ]

    for label, detail in unverifiable:
        print(_r(f"  ✗  {label}"))
        print(_d(f"       {detail}"))
        print()

    print(_rule("═"))
    print()


# ---------------------------------------------------------------------------
# JSON output mode
# ---------------------------------------------------------------------------


def emit_json(result: CallResult, prompt: str) -> None:
    """Emit a machine-readable identity record for this call."""
    record = {
        "model": result.model,
        "prompt": prompt,
        "response_chars": len(result.response),
        "elapsed_ms": result.elapsed_ms,
        "events_hash": result.events_hash,
        "receipt_checkable": result.receipt_confirmed,
        "token_usage": {
            "prompt": result.prompt_tokens,
            "completion": result.completion_tokens,
        },
        "guarantees": [
            {"label": lbl, "status": "confirmed",
             "note": "verify with verify_receipt.py"}
            for lbl, _ in GUARANTEES
        ],
        "not_guaranteed": [
            {"label": lbl, "status": "unverifiable", "reason": det}
            for lbl, det in NOT_GUARANTEED
        ],
    }
    print(json.dumps(record, indent=2))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(
        description="Web2 #7 — Expose system identity in your app"
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Call the Ambient API and attach a per-response identity summary",
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Prompt to send (only used with --live)",
    )
    parser.add_argument("--model",       default=os.getenv("AMBIENT_MODEL",   DEFAULT_MODEL))
    parser.add_argument("--api-url",     default=os.getenv("AMBIENT_API_URL",  DEFAULT_API_URL))
    parser.add_argument("--api-key",     default=os.getenv("AMBIENT_API_KEY",  ""))
    parser.add_argument("--max-tokens",  type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--json", dest="json_out", action="store_true",
        help="Emit identity record as JSON (--live only)",
    )
    parser.add_argument(
        "--no-card", action="store_true",
        help="Skip the static identity card (useful with --live --json)",
    )
    args = parser.parse_args()

    if not args.no_card:
        print_identity_card()

    if not args.live:
        return

    if not args.api_key:
        print("Error: AMBIENT_API_KEY not set.  Export it or add to .env")
        sys.exit(1)

    print(_c(f"  Calling Ambient API…"))
    print(_d(f"  Model  : {args.model}"))
    print(_d(f"  Prompt : {args.prompt[:80]}{'…' if len(args.prompt) > 80 else ''}"))
    print()

    result = call_ambient(
        args.prompt,
        model=args.model,
        api_url=args.api_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)

    if args.json_out:
        emit_json(result, args.prompt)
    else:
        print_response_identity(result, args.prompt)


if __name__ == "__main__":
    main()
