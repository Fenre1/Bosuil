from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    chunk_id: str
    investigation_id: int
    file_id: int
    path: str
    chunk_text: str
    source_type: str
    chunk_index: int
    page_number: int | None
    score: float
    display_markdown: str | None = None
    email_subject: str | None = None
    email_sender: str | None = None
    email_recipients: str | None = None
    email_date: str | None = None
    attachment_name: str | None = None
    metadata: dict[str, Any] | None = None
    retrieval_source: str = ""
    retrieval_queries: tuple[str, ...] = ()
    keyword_rank: int | None = None
    vector_rank: int | None = None
    fused_rank: int | None = None
    rerank_score: float | None = None
    context_role: str = "primary"

    @property
    def locator(self) -> str:
        if self.page_number:
            return f"{self.path} page {self.page_number}"
        return f"{self.path} chunk {self.chunk_index + 1}"


def row_to_result(row: Any, score: float = 0.0) -> SearchResult:
    metadata: dict[str, Any] = {}
    try:
        raw_metadata = row["metadata_json"]
    except Exception:
        raw_metadata = None
    if raw_metadata:
        try:
            loaded = json.loads(raw_metadata)
            if isinstance(loaded, dict):
                metadata = loaded
        except json.JSONDecodeError:
            metadata = {}
    return SearchResult(
        chunk_id=str(row["id"]),
        investigation_id=int(row["investigation_id"]),
        file_id=int(row["file_id"]),
        path=str(row["path"]),
        chunk_text=str(row["chunk_text"]),
        source_type=str(row["source_type"]),
        chunk_index=int(row["chunk_index"]),
        page_number=row["page_number"],
        score=float(score),
        display_markdown=_optional_row_value(row, "display_markdown"),
        email_subject=row["email_subject"],
        email_sender=row["email_sender"],
        email_recipients=row["email_recipients"],
        email_date=row["email_date"],
        attachment_name=row["attachment_name"],
        metadata=metadata,
    )


def _optional_row_value(row: Any, key: str) -> Any:
    try:
        keys = row.keys()
    except AttributeError:
        keys = row
    if key not in keys:
        return None
    return row[key]
