#!/usr/bin/env python3
"""
Web2 #6 — Detect and handle refusal programmatically.

Calls Ambient, classifies the response into one of four states:
  ANSWERED                  — model provided a confident response
  REFUSED_INSUFFICIENT_DATA — model said it lacks enough information
  REFUSED_AMBIGUOUS         — model found the request unclear or multi-interpretable
  REFUSED_UNCERTAIN         — model expressed doubt or declined to guarantee

Refusals are routed to a human review queue (JSONL file).
"""

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PROMPT = (
    "Based on current on-chain data, should I invest in SOL right now? "
    "Give me a direct yes or no with reasoning."
)
DEFAULT_MODEL = "zai-org/GLM-4.6"
DEFAULT_API_URL = "https://api.ambient.xyz/v1/chat/completions"
DEFAULT_REVIEW_FILE = "data/human_review.jsonl"

# ---------------------------------------------------------------------------
# Refusal states & patterns
# ---------------------------------------------------------------------------


class RefusalState(Enum):
    ANSWERED = "ANSWERED"
    REFUSED_INSUFFICIENT_DATA = "REFUSED_INSUFFICIENT_DATA"
    REFUSED_AMBIGUOUS = "REFUSED_AMBIGUOUS"
    REFUSED_UNCERTAIN = "REFUSED_UNCERTAIN"


# Ordered by priority — first state with the most pattern hits wins
_PATTERNS: List[Tuple[RefusalState, List[str]]] = [
    (
        RefusalState.REFUSED_INSUFFICIENT_DATA,
        [
            r"not enough (information|data|context)",
            r"insufficient (data|information|context)",
            r"lack(ing)? (the )?(data|information|context)",
            r"(cannot|can't|unable to) (access|retrieve|fetch|get) (real.?time|live|current|up.?to.?date)",
            r"(no|don't have) access to (real.?time|live|current|market)",
            r"my (knowledge|training) (cutoff|data)",
            r"as of my (last|knowledge) (update|cutoff)",
        ],
    ),
    (
        RefusalState.REFUSED_AMBIGUOUS,
        [
            r"(unclear|ambiguous|vague|broad) (request|question|query|prompt)",
            r"(multiple|several|different) (interpretations|meanings|ways to (read|interpret))",
            r"could (mean|refer to|be interpreted as) (many|several|multiple)",
            r"depends on (what you mean|how you define|your definition)",
            r"(more )?(specific|context|detail|clarity|clarification) (needed|required|would help)",
            r"(clarify|specify|narrow down)",
        ],
    ),
    (
        RefusalState.REFUSED_UNCERTAIN,
        [
            r"(cannot|can't|unable to) (guarantee|predict|determine|say for certain|be sure)",
            r"(no|not a) (financial|investment) (advice|advisor|guidance)",
            r"(past performance|historical (data|returns)) (does not|doesn't|is not|isn't) (guarantee|predict)",
            r"(significant|high|considerable) (risk|uncertainty|volatility)",
            r"(should|must|need to) (consult|speak (with|to)|seek) (a |an )?(financial|professional|qualified)",
            r"i('m| am) not (able|in a position|qualified) to (recommend|advise|suggest)",
            r"this (is not|isn't) (financial|investment) advice",
            r"(market|price) (is|are|can be) (unpredictable|volatile|subject to)",
        ],
    ),
]

# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass
class RefusalDecision:
    state: RefusalState
    reasons: List[str] = field(default_factory=list)
    confidence: float = 0.0


def detect_refusal(text: str) -> RefusalDecision:
    """Classify a model response into a RefusalState."""
    lowered = text.lower()
    matches: List[Tuple[RefusalState, str]] = []

    for state, patterns in _PATTERNS:
        for pattern in patterns:
            if re.search(pattern, lowered):
                matches.append((state, pattern))

    if not matches:
        return RefusalDecision(state=RefusalState.ANSWERED, reasons=[], confidence=0.95)

    # tally hits per state
    counts: dict = {}
    for state, pattern in matches:
        counts.setdefault(state, []).append(pattern)

    dominant_state = max(counts, key=lambda s: len(counts[s]))
    matched_patterns = counts[dominant_state]

    # confidence scales with number of matched patterns (capped at 0.97)
    confidence = min(0.60 + 0.12 * len(matched_patterns), 0.97)

    return RefusalDecision(
        state=dominant_state,
        reasons=matched_patterns,
        confidence=round(confidence, 2),
    )


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route(
    prompt: str,
    response: str,
    decision: RefusalDecision,
    *,
    review_file: str = DEFAULT_REVIEW_FILE,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
) -> None:
    """Route the response: log normally or escalate to human review."""
    if decision.state == RefusalState.ANSWERED:
        if verbose:
            print(f"\n[OK — {decision.state.value}]  confidence={decision.confidence}")
            print(response[:300])
    else:
        _escalate(prompt, response, decision, review_file=review_file, model=model)
        if verbose:
            print(f"\n[ESCALATED — {decision.state.value}]  confidence={decision.confidence}")
            print(f"  Matched patterns : {decision.reasons}")
            print(f"  Logged to        : {review_file}")


def _escalate(
    prompt: str,
    response: str,
    decision: RefusalDecision,
    *,
    review_file: str,
    model: str,
) -> None:
    """Write the refused response to the human review queue (JSONL)."""
    Path(review_file).parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model,
        "refusal_state": decision.state.value,
        "confidence": decision.confidence,
        "matched_patterns": decision.reasons,
        "prompt": prompt,
        "response": response,
    }
    with open(review_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


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
    model: str = DEFAULT_MODEL,
    api_url: str = DEFAULT_API_URL,
    api_key: str = "",
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> str:
    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    text_parts: List[str] = []
    with requests.post(api_url, json=payload, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="replace")
            raw = raw.strip()
            if not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                for choice in obj.get("choices", []):
                    delta = choice.get("delta", {})
                    text_parts.append(delta.get("content") or "")
            except json.JSONDecodeError:
                continue

    return "".join(text_parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Detect and handle Ambient API refusals")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Prompt to send")
    parser.add_argument("--model", default=os.getenv("AMBIENT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-url", default=os.getenv("AMBIENT_API_URL", DEFAULT_API_URL))
    parser.add_argument("--api-key", default=os.getenv("AMBIENT_API_KEY", ""))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--review-file", default=os.getenv("REVIEW_FILE", DEFAULT_REVIEW_FILE))
    parser.add_argument("--show-response", action="store_true", help="Print full model response")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.api_key:
        print("Error: AMBIENT_API_KEY not set. Export it or add to .env")
        raise SystemExit(1)

    print(f"Prompt : {args.prompt[:80]}...")
    print(f"Model  : {args.model}")

    response = call_ambient(
        args.prompt,
        model=args.model,
        api_url=args.api_url,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    decision = detect_refusal(response)

    if args.show_response:
        print(f"\n--- Response ---\n{response}\n--- End ---")

    route(
        args.prompt,
        response,
        decision,
        review_file=args.review_file,
        model=args.model,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
