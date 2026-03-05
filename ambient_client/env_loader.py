import os
from pathlib import Path


def _strip_inline_comment(line: str) -> str:
    """Remove inline # comments, respecting quoted strings."""
    in_quote = None
    for i, ch in enumerate(line):
        if ch in ('"', "'"):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
        elif ch == "#" and in_quote is None:
            if i == 0 or line[i - 1].isspace():
                return line[:i].rstrip()
    return line


def load_env(path: Path) -> None:
    """Load .env file into os.environ (won't overwrite existing vars)."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        line = _strip_inline_comment(line)
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
