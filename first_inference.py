"""
Web2 Micro-Challenge #1 — Make your first verified inference call.

Sends a prompt to the Ambient API, prints the response,
then saves and inspects the cryptographic receipt.

Usage:
    python3 first_inference.py
    python3 first_inference.py --prompt "What is Solana?"
"""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL   = "https://api.ambient.xyz/v1/chat/completions"
MODEL     = "zai-org/GLM-4.6"
RECEIPT_DIR = "data/receipts"


def _load_api_key() -> str:
    """Read AMBIENT_API_KEY from .env or environment."""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("AMBIENT_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("AMBIENT_API_KEY", "")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_ambient(api_key: str, prompt: str, max_tokens: int) -> tuple[str, list[str]]:
    """Call Ambient streaming API. Returns (response_text, raw_events)."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    text = ""
    raw_events: list[str] = []

    with requests.post(API_URL, json=payload, headers=headers, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if isinstance(line, bytes):
                line = line.decode("utf-8")
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            raw_events.append(data)
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
                for choice in obj.get("choices", []):
                    content = (choice.get("delta") or {}).get("content") or ""
                    if content:
                        text += content
                        print(content, end="", flush=True)
                    if choice.get("finish_reason") == "stop":
                        print()
            except json.JSONDecodeError:
                pass

    return text, raw_events


# ---------------------------------------------------------------------------
# Receipt
# ---------------------------------------------------------------------------

def save_receipt(prompt: str, raw_events: list[str]) -> str:
    raw_text = "\n".join(raw_events)
    events_hash = hashlib.sha256(raw_text.encode()).hexdigest()
    payload_hash = hashlib.sha256(
        json.dumps({"model": MODEL, "prompt": prompt}, sort_keys=True).encode()
    ).hexdigest()
    ts = int(time.time())
    filename = f"{ts}_{MODEL.replace('/', '_')}_{events_hash[:16]}.json"

    Path(RECEIPT_DIR).mkdir(parents=True, exist_ok=True)
    path = Path(RECEIPT_DIR) / filename
    receipt = {
        "model": MODEL,
        "timestamp": ts,
        "events_hash": events_hash,
        "payload_hash": payload_hash,
        "event_count": len(raw_events),
        "raw_events": raw_events,
    }
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return str(path)


def inspect_receipt(path: str) -> None:
    data = json.loads(Path(path).read_text())
    ts = time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime(data["timestamp"]))
    width = 50
    print(f"\n{'─' * width}")
    print("  RECEIPT")
    print(f"{'─' * width}")
    print(f"  File     : {path}")
    print(f"  Model    : {data['model']}")
    print(f"  Saved    : {ts}")
    print(f"  Events   : {data['event_count']}")
    print(f"  Hash     : {data['events_hash'][:32]}…")
    print(f"  Payload  : {data['payload_hash'][:32]}…")
    print(f"{'─' * width}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Web2 #1 — First Ambient inference call")
    p.add_argument("--prompt",     type=str, default="What is Ambient Network on Solana?")
    p.add_argument("--max-tokens", type=int, default=256)
    args = p.parse_args()

    api_key = _load_api_key()
    if not api_key:
        print("Error: AMBIENT_API_KEY not set. Add it to .env or export it.")
        return

    print(f"[Model]   {MODEL}")
    print(f"[Prompt]  {args.prompt}\n")

    text, raw_events = call_ambient(api_key, args.prompt, args.max_tokens)

    if not text:
        print("Error: empty response.")
        return

    receipt_path = save_receipt(args.prompt, raw_events)
    inspect_receipt(receipt_path)


if __name__ == "__main__":
    main()
