from __future__ import annotations

from pathlib import Path

from investigation_rag.extractors.base import BaseExtractor, ExtractedDocument, ExtractedSection
from investigation_rag.extractors.structure import SourceBlock


class TxtExtractor(BaseExtractor):
    source_type = "txt"

    def extract(self, path: Path) -> ExtractedDocument:
        data = path.read_bytes()
        text = self._decode(data)
        stat = path.stat()
        return ExtractedDocument(
            source_type=self.source_type,
            sections=[ExtractedSection(text=text, source_type=self.source_type, blocks=[
                SourceBlock("text", text, "pre", {"char_start": 0, "char_end": len(text), "line_start": 1})
            ])],
            metadata={"path": str(path), "size_bytes": stat.st_size, "modified_ns": stat.st_mtime_ns},
        )

    @staticmethod
    def _decode(data: bytes) -> str:
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return data.decode(encoding)
            except UnicodeDecodeError:
                continue
        return data.decode("utf-8", errors="replace")
