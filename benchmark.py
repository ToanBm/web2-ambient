"""
Web2 Micro-Challenge #4 — Cost + Latency Reality Check
Runs the same prompt through Ambient and OpenAI, then prints a comparison table.

Usage:
    python3 benchmark.py
    python3 benchmark.py --runs 5
    python3 benchmark.py --prompt "Explain RSA encryption briefly."
    python3 benchmark.py --max-tokens 256 --temperature 0.0
"""

import argparse
import os
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ambient_client.app.ambient import get_ambient_settings
from ambient_client.app.openai_provider import get_openai_settings
from ambient_client.app.provider_utils import ProviderSettings
from ambient_client.config import load_env_file
from ambient_client.streaming import StreamResult, stream_chat

# ---------------------------------------------------------------------------
# Cost rates (USD per 1 million tokens)
# Override via OPENAI_INPUT_RATE / OPENAI_OUTPUT_RATE / AMBIENT_INPUT_RATE etc.
# ---------------------------------------------------------------------------

_DEFAULT_RATES: Dict[str, Dict[str, float]] = {
    # Ambient
    "zai-org/glm-4.6":   {"input": 0.60, "output": 0.20},
    # OpenAI
    "gpt-4o-mini":        {"input": 0.15, "output": 0.60},
    "gpt-4o":             {"input": 2.50, "output": 10.00},
    "gpt-3.5-turbo":      {"input": 0.50, "output": 1.50},
}


def _get_rates(provider_name: str, model: str) -> Dict[str, float]:
    prefix = provider_name.upper()
    env_in  = os.getenv(f"{prefix}_INPUT_RATE", "").strip()
    env_out = os.getenv(f"{prefix}_OUTPUT_RATE", "").strip()
    if env_in and env_out:
        return {"input": float(env_in), "output": float(env_out)}
    return _DEFAULT_RATES.get(model.lower(), {"input": 0.0, "output": 0.0})


def _estimate_cost(prompt_tokens: float, completion_tokens: float, rates: Dict[str, float]) -> float:
    return (prompt_tokens * rates["input"] + completion_tokens * rates["output"]) / 1_000_000


# ---------------------------------------------------------------------------
# Per-provider aggregation
# ---------------------------------------------------------------------------

@dataclass
class ProviderStats:
    name: str
    model: str
    ttfb_ms: List[float] = field(default_factory=list)
    ttc_ms: List[float] = field(default_factory=list)
    prompt_tokens: List[int] = field(default_factory=list)
    completion_tokens: List[int] = field(default_factory=list)
    stalls: int = 0
    errors: int = 0
    runs: int = 0

    def add(self, result: StreamResult) -> None:
        self.runs += 1
        if result.error:
            self.errors += 1
            return
        if result.ttfb_ms is not None:
            self.ttfb_ms.append(result.ttfb_ms)
        if result.ttc_ms is not None:
            self.ttc_ms.append(result.ttc_ms)
        self.prompt_tokens.append(result.prompt_tokens)
        self.completion_tokens.append(result.completion_tokens)
        self.stalls += result.stall_count

    def avg(self, values: List[float]) -> Optional[float]:
        return statistics.mean(values) if values else None

    def format_ms(self, values: List[float]) -> str:
        v = self.avg(values)
        return f"{v:,.0f} ms" if v is not None else "n/a"

    def format_tokens(self, values: List[int]) -> str:
        v = self.avg(values)
        return f"{v:,.1f}" if v is not None else "n/a"

    def format_cost(self, rates: Dict[str, float]) -> str:
        pt = self.avg(self.prompt_tokens)
        ct = self.avg(self.completion_tokens)
        if pt is None or ct is None:
            return "n/a"
        cost = _estimate_cost(pt, ct, rates)
        return f"${cost:.7f}"

    def format_errors(self) -> str:
        return f"{self.errors} / {self.runs}"


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def _col_width(values: List[str], header: str) -> int:
    return max(len(header), max((len(v) for v in values), default=0)) + 2


def _print_table(rows: List[tuple], headers: List[str]) -> None:
    col_widths = [
        _col_width([str(r[i]) for r in rows], headers[i])
        for i in range(len(headers))
    ]
    sep = "┼".join("─" * w for w in col_widths)
    top = "┬".join("─" * w for w in col_widths)
    bot = "┴".join("─" * w for w in col_widths)

    print(f"┌{top}┐")
    header_row = "│".join(f" {headers[i]:<{col_widths[i]-1}}" for i in range(len(headers)))
    print(f"│{header_row}│")
    print(f"├{sep}┤")
    for row in rows:
        line = "│".join(f" {str(row[i]):<{col_widths[i]-1}}" for i in range(len(headers)))
        print(f"│{line}│")
    print(f"└{bot}┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web2 #4 — Cost + Latency benchmark")
    p.add_argument("--runs",        type=int,   default=3,   help="Timed runs per provider (default 3)")
    p.add_argument("--warmup",      type=int,   default=1,   help="Warmup runs discarded (default 1)")
    p.add_argument("--prompt",      type=str,   default=None)
    p.add_argument("--max-tokens",  type=int,   default=256)
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main() -> None:
    load_env_file()
    args = _parse_args()

    prompt = args.prompt or os.getenv(
        "AMBIENT_PROMPT",
        "What is compound interest? Give a brief definition and a worked example.",
    )

    providers: List[ProviderSettings] = [get_ambient_settings(), get_openai_settings()]
    active = [p for p in providers if p.enabled]

    if not active:
        print("No providers enabled. Set AMBIENT_ENABLED=true or OPENAI_ENABLED=true in .env")
        return

    total_runs = args.warmup + args.runs
    messages = [{"role": "user", "content": prompt}]
    all_stats: List[ProviderStats] = []

    for provider in active:
        err = provider.validation_error()
        if err:
            print(f"Skipping {provider.name}: {err}")
            continue

        model = provider.models[0]
        stats = ProviderStats(name=provider.name, model=model)
        all_stats.append(stats)

        print(f"\n[{provider.name} / {model}]  {total_runs} run(s)  (warmup={args.warmup})")

        for i in range(total_runs):
            label = f"warmup {i+1}" if i < args.warmup else f"run {i - args.warmup + 1}/{args.runs}"
            print(f"  {label} ... ", end="", flush=True)

            result = stream_chat(
                api_url=provider.api_url,
                api_key=provider.api_key,
                model=model,
                messages=messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )

            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(f"ttfb={result.ttfb_ms:.0f}ms  ttc={result.ttc_ms:.0f}ms  "
                      f"tok={result.prompt_tokens}+{result.completion_tokens}")

            if i >= args.warmup:
                stats.add(result)

    if not all_stats:
        print("\nNo results to display.")
        return

    print(f"\n\n{'─'*64}")
    print("  BENCHMARK RESULTS")
    print(f"{'─'*64}")
    print(f"  Prompt : {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
    print(f"  Runs   : {args.runs}  |  max_tokens={args.max_tokens}  temperature={args.temperature}")
    print()

    headers = ["Metric"] + [f"{s.name} / {s.model}" for s in all_stats]
    rates_list = [_get_rates(s.name, s.model) for s in all_stats]

    rows = [
        ("Avg TTFB",              *[s.format_ms(s.ttfb_ms)         for s in all_stats]),
        ("Avg TTC (total)",       *[s.format_ms(s.ttc_ms)          for s in all_stats]),
        ("Prompt tokens (avg)",   *[s.format_tokens(s.prompt_tokens)     for s in all_stats]),
        ("Completion tokens",     *[s.format_tokens(s.completion_tokens) for s in all_stats]),
        ("Est. cost / call",      *[s.format_cost(r) for s, r in zip(all_stats, rates_list)]),
        ("Stalls",                *[str(s.stalls)                   for s in all_stats]),
        ("Errors",                *[s.format_errors()               for s in all_stats]),
    ]

    _print_table(rows, headers)

    print()
    for s, rates in zip(all_stats, rates_list):
        if rates["input"] == 0.0:
            print(f"  Note: no rate data for {s.name}/{s.model} — set {s.name.upper()}_INPUT_RATE / OUTPUT_RATE in .env")


if __name__ == "__main__":
    main()
