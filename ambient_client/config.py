from pathlib import Path

from .env_loader import load_env

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_env_file() -> None:
    load_env(ENV_PATH)
