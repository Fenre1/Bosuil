from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from investigation_rag.ingest.file_identity import get_file_identity
from investigation_rag.utils.paths import normalize_path


SUPPORTED_EXTENSIONS = {".txt", ".docx", ".pdf", ".msg"}


@dataclass(frozen=True)
class ScannedFile:
    path: Path
    normalized_path: str
    suffix: str
    stable_id: str | None
    size: int
    modified_ns: int


def iter_supported_files(roots: Iterable[Path]) -> Iterable[ScannedFile]:
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if root.is_file():
            candidates = [root]
        else:
            candidates = (p for p in root.rglob("*") if p.is_file())
        for path in candidates:
            suffix = path.suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                continue
            try:
                identity = get_file_identity(path)
                yield ScannedFile(
                    path=path.resolve(),
                    normalized_path=normalize_path(path),
                    suffix=suffix,
                    stable_id=identity.stable_id,
                    size=identity.size,
                    modified_ns=identity.modified_ns,
                )
            except OSError:
                continue
