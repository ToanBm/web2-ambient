import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import requests


@dataclass
class StreamResult:
    text: str = ""
    reasoning: str = ""
    ttfb_ms: Optional[float] = None   # time to first byte
    ttc_ms: Optional[float] = None    # time to completion
    prompt_tokens: int = 0
    completion_tokens: int = 0
    stall_count: int = 0
    parse_errors: int = 0
    error: Optional[str] = None
    receipt_path: Optional[str] = None


def stream_chat(
    api_url: str,
    api_key: str,
    model: str,
    messages: list,
    *,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    top_p: Optional[float] = None,
    stall_threshold_ms: float = 2000,
    save_receipt: bool = False,
    receipt_dir: str = "data/receipts",
    verbose: bool = False,
) -> StreamResult:
    result = StreamResult()
    payload = _build_payload(model, messages, temperature, max_tokens, top_p)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    raw_events: List[str] = []
    t_start = time.monotonic()
    last_chunk_time = t_start

    try:
        with requests.post(
            api_url, json=payload, headers=headers, stream=True, timeout=120
        ) as resp:
            resp.raise_for_status()
            for event_data in _iter_sse_data(resp.iter_lines()):
                now = time.monotonic()
                gap_ms = (now - last_chunk_time) * 1000
                if gap_ms > stall_threshold_ms and result.ttfb_ms is not None:
                    result.stall_count += 1
                last_chunk_time = now

                if result.ttfb_ms is None:
                    result.ttfb_ms = (now - t_start) * 1000

                raw_events.append(event_data)

                if event_data.strip() == "[DONE]":
                    break

                try:
                    obj = json.loads(event_data)
                except json.JSONDecodeError:
                    result.parse_errors += 1
                    continue

                content, reasoning = _extract_content_parts(obj)
                result.text += content
                result.reasoning += reasoning

                if verbose and content:
                    print(content, end="", flush=True)

                usage = obj.get("usage") or {}
                if usage.get("prompt_tokens"):
                    result.prompt_tokens = usage["prompt_tokens"]
                if usage.get("completion_tokens"):
                    result.completion_tokens = usage["completion_tokens"]

    except requests.RequestException as exc:
        result.error = str(exc)

    result.ttc_ms = (time.monotonic() - t_start) * 1000

    if save_receipt and raw_events:
        result.receipt_path = _write_receipt(receipt_dir, model, payload, raw_events)

    return result


# --- helpers ---

def _iter_sse_data(lines: Iterator) -> Iterator[str]:
    for line in lines:
        if isinstance(line, bytes):
            try:
                line = line.decode("utf-8")
            except UnicodeDecodeError:
                line = line.decode("utf-8", errors="replace")
        line = line.strip()
        if line.startswith("data:"):
            yield line[5:].strip()


def _extract_content_parts(obj: dict) -> Tuple[str, str]:
    content, reasoning = "", ""
    for choice in obj.get("choices", []):
        delta = choice.get("delta") or {}
        content += delta.get("content") or ""
        reasoning += delta.get("reasoning_content") or ""
    return content, reasoning


def _build_payload(
    model: str,
    messages: list,
    temperature: Optional[float],
    max_tokens: Optional[int],
    top_p: Optional[float],
) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if top_p is not None:
        payload["top_p"] = top_p
    return payload


def _write_receipt(
    receipt_dir: str,
    model: str,
    payload: dict,
    raw_events: List[str],
) -> str:
    Path(receipt_dir).mkdir(parents=True, exist_ok=True)
    raw_text = "\n".join(raw_events)
    events_hash = hashlib.sha256(raw_text.encode()).hexdigest()[:16]
    ts = int(time.time())
    safe_model = model.replace("/", "_").replace(":", "_")
    filename = f"{ts}_{safe_model}_{events_hash}.json"
    path = Path(receipt_dir) / filename
    receipt = {
        "model": model,
        "timestamp": ts,
        "events_hash": hashlib.sha256(raw_text.encode()).hexdigest(),
        "payload_hash": hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()
        ).hexdigest(),
        "event_count": len(raw_events),
        "raw_events": raw_events,
    }
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return str(path)
