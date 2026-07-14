from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FileIdentity:
    stable_id: str | None
    size: int
    modified_ns: int


def get_file_identity(path: str | Path) -> FileIdentity:
    stat_result = os.stat(path)
    stable_id: str | None = None
    inode = getattr(stat_result, "st_ino", 0)
    device = getattr(stat_result, "st_dev", 0)
    if inode and device:
        stable_id = f"{device}:{inode}"
    return FileIdentity(
        stable_id=stable_id,
        size=int(stat_result.st_size),
        modified_ns=int(stat_result.st_mtime_ns),
    )
