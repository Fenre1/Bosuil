from __future__ import annotations

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.extractors.base import ExtractedDocument


class ExtractionCache:
    def __init__(self, db: SQLiteDB):
        self.db = db

    def get(self, content_hash: str, extractor_version: str, source_type: str) -> ExtractedDocument | None:
        payload = self.db.get_extraction_cache(content_hash, extractor_version, source_type)
        return ExtractedDocument.from_dict(payload) if payload else None

    def set(self, content_hash: str, extractor_version: str, source_type: str, document: ExtractedDocument) -> None:
        self.db.set_extraction_cache(content_hash, extractor_version, source_type, document.to_dict())
