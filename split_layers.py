#!/usr/bin/env python3
"""
Web2 Developer Loop — Micro-Challenge #5
"Split a response into verifiable and non-verifiable layers."

Calls Ambient, then classifies each sentence as:
  VERIFIABLE   — math, logic, definitions, measurements (can be checked externally)
  INTERPRETIVE — summaries, advice, opinion, hedged language (subjective)
  MIXED        — contains signals from both layers

Detection boundary:
  Verifiable signals   → numbers with operators, logical connectives, definitions,
                         measurements, code fragments, factual anchors (dates)
  Interpretive signals → hedging words, uncertainty modals, opinion markers,
                         recommendation verbs, summary phrases

Run:
  python split_layers.py
  python split_layers.py --prompt "Explain how RSA encryption works and give an example."
  python split_layers.py --show-response
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Defaults (overridden by .env or CLI flags)
# ---------------------------------------------------------------------------

DEFAULT_MODEL     = "zai-org/GLM-4.6"
DEFAULT_API_URL   = "https://api.ambient.xyz"
DEFAULT_MAX_TOKENS = 512
DEFAULT_PROMPT = (
    "What is compound interest? "
    "Calculate the value of $1,000 at 5% annual interest after 3 years, "
    "compounding annually. "
    "Then explain when compound interest works in a borrower's favor versus against them."
)

# ---------------------------------------------------------------------------
# Load .env (reuse ambient_client env loader if available)
# ---------------------------------------------------------------------------

def _load_env() -> dict:
    env: dict = {}
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return env
    with open(env_path) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if "=" in line:
                k, _, v = line.partition("=")
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env

_env = _load_env()

def _cfg(key: str, default: str = "") -> str:
    return os.environ.get(key) or _env.get(key) or default

# ---------------------------------------------------------------------------
# Verifiable patterns  (signal → kind label)
# ---------------------------------------------------------------------------

VERIFIABLE_PATTERNS: list[tuple[str, str]] = [
    # Math expressions: digits with operators or equals
    (r"\b\d[\d,]*\.?\d*\s*[\+\-\×\÷\*\/]\s*\d",         "math"),
    (r"\b\d[\d,]*\.?\d*\s*(?:equals?|=)\s*[\$\d]",       "math"),
    # Equation result with optional markdown: = **$1,050 or = $52.50
    (r"=\s*(?:\*\*)?[\$€£]\d[\d,]*",                     "math"),
    # Currency arithmetic with optional parenthetical: $1,000 (Principal) + $50
    (r"[\$€£]\d[\d,]*(?:\s*\([^)]*\))?\s*[\+\-]\s*[\$€£]\d", "math"),
    # Explicit calculation result
    (r"\b(?:result(?:s)? in|gives|yields|comes? to)\s+[\$\d]", "math"),
    # Percentages, currency, units of measurement
    (r"\b\d+\.?\d*\s*(?:%|percent|kg|km|m\b|Hz|USD|\$|SOL|°[CF])", "measurement"),
    # Logical connectives that signal a derived conclusion
    (r"\b(?:therefore|thus|hence|it follows that|consequently)\b", "logic"),
    # Formal definitions
    (r"\b(?:is defined as|refers to|is the process of|is a (?:mathematical|financial|logical))\b", "definition"),
    # Inline code or formula
    (r"`[^`]+`",                                          "code"),
    # Year-anchored facts ("in 1970", "since 2009")
    (r"\b(?:in|since|by|from)\s+(?:19|20)\d{2}\b",       "fact"),
    # Step-numbered reasoning ("Step 1:", "1.")
    (r"(?:^|\n)\s*(?:Step\s*)?\d+[\.\)]\s",              "logic"),
]

# ---------------------------------------------------------------------------
# Interpretive patterns
# ---------------------------------------------------------------------------

INTERPRETIVE_PATTERNS: list[tuple[str, str]] = [
    # Hedging / generalisation
    (r"\b(?:typically|generally|often|usually|tend(?:s)? to|in most cases|for the most part)\b", "hedging"),
    # Uncertainty modals
    (r"\b(?:might|may|could|possibly|perhaps|arguably|likely|unlikely|seem(?:s)? to)\b", "uncertainty"),
    # Opinion / belief markers
    (r"\b(?:I think|I believe|in my (?:view|opinion)|one might argue|it (?:could|can) be argued)\b", "opinion"),
    # Advice / recommendation
    (r"\b(?:should|consider|recommend(?:ed)?|suggest(?:ed)?|best practice|it(?:['']s| is) (?:better|best|wise|advisable))\b", "advice"),
    # Summary / conclusion markers
    (r"\b(?:in summary|in conclusion|to summarize|overall|in essence|to put it simply|in other words|all in all)\b", "summary"),
    # Comparative evaluation
    (r"\b(?:better|worse|more effective|less effective|superior|inferior|preferable|outweigh)\b", "evaluative"),
    # Vague subjective qualifiers
    (r"\b(?:significant(?:ly)?|important(?:ly)?|crucial(?:ly)?|key\b|major\b|minor\b|notable)\b", "subjective"),
]

# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

def _match_patterns(text: str, patterns: list[tuple[str, str]]) -> list[dict]:
    signals = []
    for pattern, kind in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if m:
            signals.append({"type": kind, "matched": m.group(0).strip()})
    return signals


def classify_sentence(sentence: str) -> tuple[str, list[dict]]:
    """Return (VERIFIABLE | INTERPRETIVE | MIXED, signals)."""
    v = _match_patterns(sentence, VERIFIABLE_PATTERNS)
    i = _match_patterns(sentence, INTERPRETIVE_PATTERNS)

    if v and i:
        label = "MIXED"
    elif v:
        label = "VERIFIABLE"
    elif i:
        label = "INTERPRETIVE"
    else:
        # Unclassified prose defaults to INTERPRETIVE (can't be checked externally)
        label = "INTERPRETIVE"
        i.append({"type": "default", "matched": "(no explicit signal — prose)"})

    return label, v + i


def split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries and newlines; skip very short fragments."""
    raw = re.split(r'(?<=[.?!])\s+|\n+', text)
    return [s.strip() for s in raw if s.strip() and len(s.strip()) > 10]

# ---------------------------------------------------------------------------
# Ambient API call (streaming)
# ---------------------------------------------------------------------------

def call_ambient(prompt: str, api_url: str, api_key: str,
                 model: str, max_tokens: int) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload = {
        "model": model,
        "stream": True,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    chunks: list[str] = []
    # api_url may already include the full path or just the base
    endpoint = api_url if api_url.endswith("/completions") else f"{api_url.rstrip('/')}/v1/chat/completions"
    with requests.post(
        endpoint,
        headers=headers,
        json=payload,
        stream=True,
        timeout=60,
    ) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    chunks.append(delta)
            except (json.JSONDecodeError, KeyError):
                pass
    return "".join(chunks)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(prompt: str, api_url: str, api_key: str,
        model: str, max_tokens: int, show_response: bool) -> None:

    print(f"\n[Model]  {model}")
    print(f"[Prompt] {prompt[:90]}{'...' if len(prompt) > 90 else ''}\n")

    print("Calling Ambient API...", flush=True)
    response = call_ambient(prompt, api_url, api_key, model, max_tokens)
    print("Done.\n")

    if show_response:
        print("─" * 62)
        print(response)
        print("─" * 62 + "\n")

    sentences = split_sentences(response)

    layers: dict[str, list] = {"VERIFIABLE": [], "INTERPRETIVE": [], "MIXED": []}
    for sent in sentences:
        label, signals = classify_sentence(sent)
        layers[label].append({"text": sent, "signals": signals})

    # ── Print layered report ─────────────────────────────────────────────────
    labels_order = ["VERIFIABLE", "INTERPRETIVE", "MIXED"]
    for label in labels_order:
        items = layers[label]
        if not items:
            continue
        print("─" * 62)
        print(f"  {label}  ({len(items)} sentence{'s' if len(items) != 1 else ''})")
        print("─" * 62)
        for item in items:
            print(f"  · {item['text']}")
            for sig in item["signals"]:
                print(f"      [{sig['type']}]  \"{sig['matched']}\"")
        print()

    total = sum(len(v) for v in layers.values())
    v_count = len(layers["VERIFIABLE"])
    i_count = len(layers["INTERPRETIVE"])
    m_count = len(layers["MIXED"])
    print(
        f"Total: {total} sentences  |  "
        f"VERIFIABLE: {v_count}  |  "
        f"INTERPRETIVE: {i_count}  |  "
        f"MIXED: {m_count}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split an Ambient API response into verifiable and interpretive layers."
    )
    parser.add_argument(
        "--prompt", default=DEFAULT_PROMPT,
        help="Prompt to send to the model (default: compound interest example)"
    )
    parser.add_argument("--model",      default=None, help="Override model name")
    parser.add_argument("--api-url",    default=None, help="Override API base URL")
    parser.add_argument("--api-key",    default=None, help="Override API key")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max tokens")
    parser.add_argument("--show-response", action="store_true",
                        help="Print the raw model response before layered output")
    args = parser.parse_args()

    api_key   = args.api_key   or _cfg("AMBIENT_API_KEY")
    api_url   = args.api_url   or _cfg("AMBIENT_API_URL", DEFAULT_API_URL)
    model     = args.model     or _cfg("AMBIENT_MODEL",   DEFAULT_MODEL)
    max_tokens = args.max_tokens or int(_cfg("AMBIENT_MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))

    if not api_key:
        print("ERROR: AMBIENT_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    run(args.prompt, api_url, api_key, model, max_tokens, args.show_response)
