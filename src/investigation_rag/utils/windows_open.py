from __future__ import annotations

import os
import subprocess
from pathlib import Path


def open_file(path: str | Path) -> None:
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(str(target))
    os.startfile(str(target))  # type: ignore[attr-defined]


def open_containing_folder(path: str | Path) -> None:
    target = Path(path)
    folder = target if target.is_dir() else target.parent
    if not folder.exists():
        raise FileNotFoundError(str(folder))
    os.startfile(str(folder))  # type: ignore[attr-defined]


def reveal_in_explorer(path: str | Path) -> None:
    target = Path(path)
    if target.exists() and target.is_file():
        subprocess.Popen(["explorer", "/select,", str(target)])
        return
    open_containing_folder(target)
