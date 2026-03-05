"""
Web2 Micro-Challenge #3 — Verify or reject an inference receipt.

Usage:
    python3 verify_receipt.py                          # verify most recent receipt
    python3 verify_receipt.py data/receipts/foo.json   # verify a specific receipt
    python3 verify_receipt.py --tamper                 # simulate tampered receipt → REJECTED
    python3 verify_receipt.py --generate               # call Ambient, save receipt, verify it
"""

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Check results
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class Check:
    name: str
    status: str       # PASS | FAIL | SKIP
    detail: str = ""


@dataclass
class VerifyResult:
    receipt_path: str
    checks: List[Check] = field(default_factory=list)
    model: str = ""
    event_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    content_chars: int = 0
    content_deltas: int = 0

    @property
    def verified(self) -> bool:
        return all(c.status != FAIL for c in self.checks)


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def verify_receipt(data: dict, receipt_path: str) -> VerifyResult:
    result = VerifyResult(receipt_path=receipt_path)
    result.model = data.get("model", "unknown")

    raw_events: List[str] = data.get("raw_events", [])
    result.event_count = len(raw_events)

    # --- Check 1: events_hash ---
    stored_events_hash = data.get("events_hash", "")
    derived_events_hash = _hash("\n".join(raw_events))
    if not stored_events_hash:
        result.checks.append(Check("events_hash", SKIP, "field missing in receipt"))
    elif derived_events_hash == stored_events_hash:
        result.checks.append(Check("events_hash", PASS,
            f"sha256 matches ({stored_events_hash[:16]}…)"))
    else:
        result.checks.append(Check("events_hash", FAIL,
            f"MISMATCH — expected {stored_events_hash[:16]}… got {derived_events_hash[:16]}…"))

    # --- Check 2: payload_hash ---
    stored_payload_hash = data.get("payload_hash", "")
    if not stored_payload_hash:
        result.checks.append(Check("payload_hash", SKIP, "field missing in receipt"))
    else:
        # We can't re-derive the payload from the receipt alone, but we can confirm
        # the field is present and is a valid sha256 hex string.
        if len(stored_payload_hash) == 64 and all(c in "0123456789abcdef" for c in stored_payload_hash):
            result.checks.append(Check("payload_hash", PASS,
                f"present and well-formed ({stored_payload_hash[:16]}…)"))
        else:
            result.checks.append(Check("payload_hash", FAIL,
                f"malformed hash value: {stored_payload_hash!r}"))

    # --- Check 3: event parsing ---
    parse_errors = 0
    parsed_events = []
    for ev in raw_events:
        if ev.strip() == "[DONE]":
            parsed_events.append(None)
            continue
        try:
            parsed_events.append(json.loads(ev))
        except json.JSONDecodeError:
            parse_errors += 1
            parsed_events.append(None)

    valid_count = result.event_count - parse_errors
    if parse_errors == 0:
        result.checks.append(Check("event parsing", PASS,
            f"{valid_count} / {result.event_count} valid JSON"))
    else:
        result.checks.append(Check("event parsing", FAIL,
            f"{parse_errors} parse error(s) out of {result.event_count} events"))

    # --- Check 4: content reconstruction ---
    content = ""
    for obj in parsed_events:
        if obj is None:
            continue
        for choice in obj.get("choices", []):
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            result.content_deltas += 1 if delta.get("content") else 0

    result.content_chars = len(content)
    if result.content_chars > 0:
        result.checks.append(Check("content", PASS,
            f"{result.content_deltas} deltas, {result.content_chars:,} chars reconstructed"))
    else:
        result.checks.append(Check("content", FAIL, "no content tokens found in events"))

    # --- Check 5: usage tokens ---
    for obj in reversed(parsed_events):
        if obj is None:
            continue
        usage = obj.get("usage") or {}
        pt = usage.get("prompt_tokens", 0)
        ct = usage.get("completion_tokens", 0)
        if pt or ct:
            result.prompt_tokens = pt
            result.completion_tokens = ct
            break

    if result.prompt_tokens > 0 or result.completion_tokens > 0:
        result.checks.append(Check("usage", PASS,
            f"{result.prompt_tokens} prompt + {result.completion_tokens} completion tokens"))
    else:
        result.checks.append(Check("usage", SKIP,
            "no usage block found (API may not return it in stream)"))

    return result


# ---------------------------------------------------------------------------
# Receipt file helpers
# ---------------------------------------------------------------------------

def _latest_receipt(receipt_dir: str) -> Optional[Path]:
    receipts = sorted(Path(receipt_dir).glob("*.json"), key=lambda p: p.stat().st_mtime)
    return receipts[-1] if receipts else None


def _load_receipt(path: str) -> Tuple[dict, str]:
    p = Path(path)
    if not p.exists():
        print(f"Error: receipt file not found: {path}")
        sys.exit(1)
    data = json.loads(p.read_text(encoding="utf-8"))
    return data, str(p)


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------

WIDTH = 62

def _print_result(result: VerifyResult, label: str = "") -> None:
    if label:
        print(f"\n{'─' * WIDTH}")
        print(f"  {label}")
    print(f"{'─' * WIDTH}")
    print(f"  Receipt  : {result.receipt_path}")
    print(f"  Model    : {result.model}")
    print(f"  Events   : {result.event_count}  |  "
          f"Tokens: {result.prompt_tokens} prompt + {result.completion_tokens} completion")
    print()

    for check in result.checks:
        icon = "✓" if check.status == PASS else ("✗" if check.status == FAIL else "–")
        tag  = f"[{check.status}]"
        print(f"  {icon} {tag:<6} {check.name:<16} {check.detail}")

    print()
    if result.verified:
        print("  Status: VERIFIED ✓")
    else:
        failed = [c for c in result.checks if c.status == FAIL]
        reasons = ", ".join(c.name for c in failed)
        print(f"  Status: REJECTED ✗  (reason: {reasons})")
    print(f"{'─' * WIDTH}")


# ---------------------------------------------------------------------------
# --generate mode
# ---------------------------------------------------------------------------

def _generate_and_verify(receipt_dir: str, temperature: float, max_tokens: int) -> str:
    from ambient_client.app.ambient import get_ambient_settings
    from ambient_client.config import load_env_file
    from ambient_client.streaming import stream_chat

    load_env_file()
    provider = get_ambient_settings()
    err = provider.validation_error()
    if err:
        print(f"Error: {err}")
        sys.exit(1)

    model = provider.models[0]
    prompt = os.getenv("AMBIENT_PROMPT",
        "What is compound interest? Give a brief definition and a worked example.")

    print(f"[Model]  {model}")
    print(f"[Prompt] {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
    print("\nCalling Ambient API (receipt will be saved)...")

    result = stream_chat(
        api_url=provider.api_url,
        api_key=provider.api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        save_receipt=True,
        receipt_dir=receipt_dir,
    )

    if result.error:
        print(f"Error: {result.error}")
        sys.exit(1)

    if not result.receipt_path:
        print("Error: receipt was not saved.")
        sys.exit(1)

    print(f"Receipt saved: {result.receipt_path}\n")
    return result.receipt_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Web2 #3 — Verify an Ambient inference receipt")
    p.add_argument("receipt",       nargs="?",      help="Path to receipt JSON (default: most recent)")
    p.add_argument("--tamper",      action="store_true", help="Simulate a tampered receipt → REJECTED")
    p.add_argument("--generate",    action="store_true", help="Call Ambient, save receipt, then verify")
    p.add_argument("--receipt-dir", default="data/receipts")
    p.add_argument("--max-tokens",  type=int,   default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()

    # Resolve receipt path
    if args.generate:
        receipt_path = _generate_and_verify(args.receipt_dir, args.temperature, args.max_tokens)
    elif args.receipt:
        receipt_path = args.receipt
    else:
        latest = _latest_receipt(args.receipt_dir)
        if not latest:
            print(f"No receipts found in '{args.receipt_dir}'. Run with --generate to create one.")
            sys.exit(1)
        receipt_path = str(latest)

    data, receipt_path = _load_receipt(receipt_path)

    # --- Normal verification ---
    result = verify_receipt(data, receipt_path)
    _print_result(result, label="VERIFICATION" if args.tamper else "")

    if not args.tamper:
        return

    # --- Tamper simulation ---
    print()
    tampered = dict(data)
    raw = list(data.get("raw_events", []))

    # Inject a fabricated token into the middle of the event stream
    fake_event = json.dumps({
        "choices": [{"delta": {"content": " [INJECTED_TOKEN]"}, "finish_reason": None}]
    })
    mid = len(raw) // 2
    raw.insert(mid, fake_event)
    tampered["raw_events"] = raw

    tampered_result = verify_receipt(tampered, receipt_path + " [tampered]")
    _print_result(tampered_result, label="TAMPER SIMULATION — injected fake token at event midpoint")


if __name__ == "__main__":
    main()
