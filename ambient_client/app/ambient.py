from .provider_utils import ProviderSettings, get_provider_settings


def get_ambient_settings() -> ProviderSettings:
    return get_provider_settings(
        name="Ambient",
        prefix="AMBIENT",
        enabled_env="AMBIENT_ENABLED",
        api_url_env="AMBIENT_API_URL",
        base_url_env="AMBIENT_BASE_URL",
        default_url="https://api.ambient.xyz/v1/chat/completions",
        api_key_envs=["AMBIENT_API_KEY"],
        models_env="AMBIENT_MODELS",
        model_env="AMBIENT_MODEL",
        default_model="zai-org/GLM-4.6",
        default_enabled=True,
    )
