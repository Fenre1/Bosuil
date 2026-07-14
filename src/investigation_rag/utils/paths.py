from __future__ import annotations

import os
from pathlib import Path


def default_app_data_dir() -> Path:
    local_app_data = os.getenv("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "InvestigationRAG"
    return Path.home() / "AppData" / "Local" / "InvestigationRAG"


def normalize_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve()).casefold()
