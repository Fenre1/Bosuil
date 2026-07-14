from __future__ import annotations

import os
from pathlib import Path


def load_dotenv_if_available() -> Path | None:
    """Load a local .env file without overriding real environment variables."""
    explicit_path = os.getenv("INVESTIGATION_RAG_ENV_FILE")
    try:
        from dotenv import find_dotenv, load_dotenv
    except ImportError:
        return None

    if explicit_path:
        env_path = Path(explicit_path).expanduser()
        if env_path.exists():
            load_dotenv(env_path, override=False)
            return env_path
        return None

    found = find_dotenv(filename=".env", usecwd=True)
    if not found:
        return None
    env_path = Path(found)
    load_dotenv(env_path, override=False)
    return env_path
