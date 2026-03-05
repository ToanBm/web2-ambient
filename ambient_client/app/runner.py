import os
from dataclasses import dataclass
from typing import Optional

from ..config import load_env_file
from ..streaming import stream_chat
from ..utils import is_enabled
from .ambient import get_ambient_settings
from .bench import (
    BenchRecorder,
    attach_result_metrics,
    build_bench_record,
    iter_run_specs,
)
from .prompt import load_prompt


@dataclass
class EnvConfig:
    temperature: Optional[float]
    max_tokens: Optional[int]
    top_p: Optional[float]
    verbose: bool
    save_receipt: bool
    receipt_dir: str
    stall_threshold_ms: float
    bench_enabled: bool
    bench_warmup: int
    bench_runs: int
    bench_output: str


def _load_env_config() -> EnvConfig:
    def _float(key: str) -> Optional[float]:
        v = os.getenv(key, "").strip()
        try:
            return float(v) if v else None
        except ValueError:
            return None

    def _int(key: str) -> Optional[int]:
        v = os.getenv(key, "").strip()
        try:
            return int(v) if v else None
        except ValueError:
            return None

    return EnvConfig(
        temperature=_float("REQUEST_TEMPERATURE"),
        max_tokens=_int("REQUEST_MAX_TOKENS"),
        top_p=_float("REQUEST_TOP_P"),
        verbose=is_enabled(os.getenv("REQUEST_VERBOSE"), default=True),
        save_receipt=is_enabled(os.getenv("AMBIENT_RECEIPT_SAVE"), default=False),
        receipt_dir=os.getenv("AMBIENT_RECEIPT_DIR", "data/receipts"),
        stall_threshold_ms=float(os.getenv("BENCH_STALL_THRESHOLD_MS", "2000")),
        bench_enabled=is_enabled(os.getenv("BENCH_ENABLED"), default=False),
        bench_warmup=int(os.getenv("BENCH_WARMUP_RUNS", "1")),
        bench_runs=int(os.getenv("BENCH_RUNS", "3")),
        bench_output=os.getenv("BENCH_OUTPUT", "data/bench.jsonl"),
    )


def run() -> None:
    load_env_file()
    cfg = _load_env_config()

    prompt = load_prompt()
    if not prompt:
        print("Error: no prompt available.")
        return

    providers = [get_ambient_settings()]

    for provider in providers:
        err = provider.validation_error()
        if err:
            print(f"Skipping {provider.name}: {err}")
            continue

        for model in provider.models:
            messages = [{"role": "user", "content": prompt}]
            specs = iter_run_specs(cfg.bench_enabled, cfg.bench_warmup, cfg.bench_runs)
            recorder = BenchRecorder(cfg.bench_output) if cfg.bench_enabled else None

            print(f"\n--- {provider.name} / {model} ---")

            for spec in specs:
                label = f"{provider.name}/{model}{spec.label_suffix}"
                print(f"\n[{label}]")

                result = stream_chat(
                    api_url=provider.api_url,
                    api_key=provider.api_key,
                    model=model,
                    messages=messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                    top_p=cfg.top_p,
                    stall_threshold_ms=cfg.stall_threshold_ms,
                    save_receipt=cfg.save_receipt,
                    receipt_dir=cfg.receipt_dir,
                    verbose=cfg.verbose,
                )

                if result.error:
                    print(f"\nError: {result.error}")
                else:
                    if not cfg.verbose:
                        print(result.text[:200])
                    print(
                        f"\n[ttfb={result.ttfb_ms:.0f}ms "
                        f"ttc={result.ttc_ms:.0f}ms "
                        f"stalls={result.stall_count}]"
                    )

                if recorder and not spec.warmup:
                    record = build_bench_record(provider.name, model, provider.api_url, spec)
                    record = attach_result_metrics(record, result)
                    recorder.write(record)

            if recorder:
                recorder.close()
