import os
from dataclasses import dataclass
from typing import List, Optional

from ..utils import is_enabled


@dataclass
class ProviderSettings:
    name: str
    enabled: bool
    api_url: str
    api_key: str
    models: List[str]

    def validation_error(self) -> Optional[str]:
        if not self.enabled:
            return None
        if not self.api_key:
            return f"{self.name}: API key not set"
        if not self.models:
            return f"{self.name}: no models configured"
        return None


def build_chat_completions_url(
    api_url_env: str,
    base_url_env: str,
    default_url: str,
) -> str:
    explicit = os.getenv(api_url_env, "").strip()
    if explicit:
        return explicit
    base = os.getenv(base_url_env, "").strip()
    if base:
        base = base.rstrip("/")
        if not base.endswith("/chat/completions"):
            return f"{base}/chat/completions"
        return base
    return default_url


def parse_models(raw: str) -> List[str]:
    seen: set = set()
    result = []
    for m in raw.replace(",", "\n").splitlines():
        m = m.strip()
        if m and m not in seen:
            seen.add(m)
            result.append(m)
    return result


def model_flag_env_key(prefix: str, model: str) -> str:
    sanitized = "".join(c if c.isalnum() else "_" for c in model).upper()
    return f"{prefix}_MODEL_{sanitized}_ENABLED"


def filter_enabled_models(prefix: str, models: List[str]) -> List[str]:
    return [m for m in models if is_enabled(os.getenv(model_flag_env_key(prefix, m)))]


def get_provider_settings(
    name: str,
    prefix: str,
    enabled_env: str,
    api_url_env: str,
    base_url_env: str,
    default_url: str,
    api_key_envs: List[str],
    models_env: str,
    model_env: str,
    default_model: str,
    default_enabled: bool = True,
) -> ProviderSettings:
    enabled = is_enabled(os.getenv(enabled_env), default=default_enabled)
    api_url = build_chat_completions_url(api_url_env, base_url_env, default_url)

    api_key = ""
    for env_key in api_key_envs:
        api_key = os.getenv(env_key, "").strip()
        if api_key:
            break

    raw_models = os.getenv(models_env, "").strip()
    models = parse_models(raw_models) if raw_models else [os.getenv(model_env, default_model).strip()]
    models = filter_enabled_models(prefix, models)

    return ProviderSettings(
        name=name,
        enabled=enabled,
        api_url=api_url,
        api_key=api_key,
        models=models,
    )
