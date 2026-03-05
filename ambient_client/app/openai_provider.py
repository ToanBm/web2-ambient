from .provider_utils import ProviderSettings, get_provider_settings


def get_openai_settings() -> ProviderSettings:
    return get_provider_settings(
        name="OpenAI",
        prefix="OPENAI",
        enabled_env="OPENAI_ENABLED",
        api_url_env="OPENAI_API_URL",
        base_url_env="OPENAI_BASE_URL",
        default_url="https://api.openai.com/v1/chat/completions",
        api_key_envs=["OPENAI_API_KEY"],
        models_env="OPENAI_MODELS",
        model_env="OPENAI_MODEL",
        default_model="gpt-4o-mini",
        default_enabled=False,
    )
