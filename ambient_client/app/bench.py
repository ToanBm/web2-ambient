import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..streaming import StreamResult


@dataclass(frozen=True)
class RunSpec:
    index: int
    total: int
    warmup: bool
    label_suffix: str


class BenchRecorder:
    def __init__(self, path: str) -> None:
        self._path = path
        self._file = None
        try:
            self._file = open(path, "a", encoding="utf-8")
        except OSError as exc:
            print(f"Warning: cannot open bench file '{path}': {exc}")

    def write(self, record: Dict[str, Any]) -> None:
        if self._file:
            try:
                self._file.write(json.dumps(record) + "\n")
                self._file.flush()
            except OSError as exc:
                print(f"Warning: bench write error: {exc}")

    def close(self) -> None:
        if self._file:
            self._file.close()


def iter_run_specs(bench_enabled: bool, warmup_runs: int, bench_runs: int) -> List[RunSpec]:
    if not bench_enabled:
        return [RunSpec(index=0, total=1, warmup=False, label_suffix="")]
    specs = []
    for i in range(warmup_runs):
        specs.append(RunSpec(
            index=i,
            total=warmup_runs + bench_runs,
            warmup=True,
            label_suffix=f" [warmup {i + 1}/{warmup_runs}]",
        ))
    for i in range(bench_runs):
        specs.append(RunSpec(
            index=warmup_runs + i,
            total=warmup_runs + bench_runs,
            warmup=False,
            label_suffix=f" [run {i + 1}/{bench_runs}]",
        ))
    return specs


def build_bench_meta(
    warmup_runs: int,
    bench_runs: int,
    stall_threshold_ms: float,
    prompt: str,
    params: Dict,
) -> Dict:
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "warmup_runs": warmup_runs,
        "bench_runs": bench_runs,
        "stall_threshold_ms": stall_threshold_ms,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
        "request_params": params,
    }


def build_bench_record(provider: str, model: str, api_url: str, spec: RunSpec) -> Dict:
    return {
        "provider": provider,
        "model": model,
        "api_url": api_url,
        "run_index": spec.index,
        "warmup": spec.warmup,
    }


def attach_result_metrics(record: Dict, result: StreamResult) -> Dict:
    record = dict(record)
    record.update({
        "ttfb_ms": round(result.ttfb_ms, 2) if result.ttfb_ms is not None else None,
        "ttc_ms": round(result.ttc_ms, 2) if result.ttc_ms is not None else None,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "text_chars": len(result.text),
        "reasoning_chars": len(result.reasoning),
        "stall_count": result.stall_count,
        "parse_errors": result.parse_errors,
        "error": result.error,
        "receipt_path": result.receipt_path,
    })
    return record
