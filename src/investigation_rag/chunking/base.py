from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Chunk:
    chunk_index: int
    text: str
    source_type: str
    page_number: int | None = None
    display_markdown: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Chunker:
    version = ""

    def chunk_document(self, document: Any) -> list[Chunk]:
        raise NotImplementedError
