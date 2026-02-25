"""
Web2 Micro-Challenge #2 — Stream a response end-to-end.

Streams a response from Ambient (or any OpenAI-compatible provider),
printing tokens live as they arrive, then reports TTFB and TTC.

Usage:
    python3 stream_response.py
    python3 stream_response.py --prompt "Explain RSA encryption briefly."
    python3 stream_response.py --compare        # Ambient vs OpenAI (stretch goal)
    python3 stream_response.py --max-tokens 128
"""

import argparse
import os
import sys

from ambient_client.app.ambient import get_ambient_settings
from ambient_client.app.openai_provider import get_openai_settings
from ambient_client.app.provider_utils import ProviderSettings
from ambient_client.config import load_env_file
from ambient_client.streaming import StreamResult, stream_chat


# ---------------------------------------------------------------------------
# Single provider run — prints tokens live, reports latency
# ---------------------------------------------------------------------------

def run_provider(provider: ProviderSettings, prompt: str, max_tokens: int, temperature: float) -> StreamResult:
    model = provider.models[0]

    print(f"\n[{provider.name} / {model}]")
    print(f"[Prompt] {prompt[:80]}{'…' if len(prompt) > 80 else ''}\n")

    result = stream_chat(
        api_url=provider.api_url,
        api_key=provider.api_key,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        verbose=True,          # prints tokens to stdout as they arrive
    )

    print()  # newline after streamed content

    if result.error:
        print(f"\n[error] {result.error}")
    else:
        ttfb = f"{result.ttfb_ms:.0f}ms" if result.ttfb_ms is not None else "n/a"
        ttc  = f"{result.ttc_ms:.0f}ms"  if result.ttc_ms  is not None else "n/a"
        print(f"\n[ttfb={ttfb}  ttc={ttc}  "
              f"tokens={result.prompt_tokens}+{result.completion_tokens}  "
              f"stalls={result.stall_count}]")

    return result


# ---------------------------------------------------------------------------
# Compare mode — run both providers, print latency summary
# ---------------------------------------------------------------------------

def run_compare(prompt: str, max_tokens: int, temperature: float) -> None:
    providers = [get_ambient_settings(), get_openai_settings()]
    active = [p for p in providers if p.enabled]

    if len(active) < 2:
        print("Compare mode requires both AMBIENT_ENABLED=true and OPENAI_ENABLED=true in .env")
        sys.exit(1)

    results = []
    for provider in active:
        err = provider.validation_error()
        if err:
            print(f"Skipping {provider.name}: {err}")
            continue
        result = run_provider(provider, prompt, max_tokens, temperature)
        results.append((provider, result))

    if len(results) < 2:
        return

    # Summary table
    print(f"\n{'─' * 56}")
    print("  LATENCY SUMMARY")
    print(f"{'─' * 56}")
    header = f"  {'Metric':<20} {'Ambient':>14} {'OpenAI':>14}"
    print(header)
    print(f"  {'─'*20} {'─'*14} {'─'*14}")

    def ms(r: StreamResult, attr: str) -> str:
        v = getattr(r, attr)
        return f"{v:,.0f} ms" if v is not None else "n/a"

    ambient_r = results[0][1]
    openai_r  = results[1][1]

    rows = [
        ("TTFB",        ms(ambient_r, "ttfb_ms"),  ms(openai_r, "ttfb_ms")),
        ("TTC",         ms(ambient_r, "ttc_ms"),   ms(openai_r, "ttc_ms")),
        ("Tokens (out)", str(ambient_r.completion_tokens), str(openai_r.completion_tokens)),
        ("Stalls",      str(ambient_r.stall_count), str(openai_r.stall_count)),
        ("Errors",      str(ambient_r.error or "none"), str(openai_r.error or "none")),
    ]

    for label, a_val, o_val in rows:
        print(f"  {label:<20} {a_val:>14} {o_val:>14}")

    print(f"{'─' * 56}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Web2 #2 — Stream a response end-to-end")
    p.add_argument("--prompt",      type=str,   default=None)
    p.add_argument("--max-tokens",  type=int,   default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--compare",     action="store_true", help="Run Ambient + OpenAI side by side")
    args = p.parse_args()

    load_env_file()

    prompt = args.prompt or os.getenv(
        "AMBIENT_PROMPT",
        "What is compound interest? Give a brief definition and a worked example.",
    )

    if args.compare:
        run_compare(prompt, args.max_tokens, args.temperature)
        return

    provider = get_ambient_settings()
    err = provider.validation_error()
    if err:
        print(f"Error: {err}")
        sys.exit(1)

    run_provider(provider, prompt, args.max_tokens, args.temperature)


if __name__ == "__main__":
    main()
