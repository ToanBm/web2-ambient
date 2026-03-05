import os
from pathlib import Path
from typing import Optional


def load_prompt() -> Optional[str]:
    prompt_file = os.getenv("AMBIENT_PROMPT_FILE", "").strip()
    if prompt_file:
        try:
            return Path(prompt_file).read_text(encoding="utf-8")
        except OSError as exc:
            print(f"Error: Unable to read AMBIENT_PROMPT_FILE '{prompt_file}': {exc}")
            return None
    return os.getenv("AMBIENT_PROMPT", "What is Ambient Network on Solana?")
