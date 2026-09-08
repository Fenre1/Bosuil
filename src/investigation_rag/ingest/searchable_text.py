from __future__ import annotations

from pathlib import Path
from math import ceil


CONTEXT_FIELDS = (
    ("Title", "title"),
    ("Subject", "subject"),
    ("Subject", "email_subject"),
    ("Sender", "sender"),
    ("Sender", "email_sender"),
    ("Recipients", "recipients"),
    ("Recipients", "email_recipients"),
    ("Date", "date"),
    ("Date", "email_date"),
    ("Attachment", "attachment_name"),
    ("Parent subject", "parent_email_subject"),
)


def keyword_searchable_text(path: Path, chunk_text: str, metadata: dict[str, object]) -> str:
    """Stable FTS text; the separate indexed FTS path column holds path terms."""

    del path
    context = _metadata_context(metadata)
    return "\n".join([*context, "", str(chunk_text)])


def stable_embedding_text(chunk_text: str, metadata: dict[str, object]) -> str:
    """Semantic input that remains valid when a file is moved or renamed."""

    if "embedding_context" in metadata:
        context = str(metadata["embedding_context"])
        return f"{context}\n\n{chunk_text}" if context else str(chunk_text)
    context = _metadata_context(metadata)
    return "\n".join([*context, "", str(chunk_text)]) if context else str(chunk_text)


def embedding_context(metadata: dict[str, object]) -> str:
    context = _metadata_context(metadata)
    for key, label in (("heading_path_text", "Section"), ("table_header_context", "Table headers"),
                       ("overlap_text", "Previous passage context")):
        if metadata.get(key):
            context.append(f"{label}: {metadata[key]}")
    return "\n".join(context)


def estimate_embedding_tokens(text: str) -> int:
    # Conservative estimate, not a tokenizer. The server enforces its actual
    # model limit with truncate=False. UTF-8 length accounts for non-Latin text.
    return ceil(len(text.encode("utf-8")) / 3)


def _metadata_context(metadata: dict[str, object]) -> list[str]:
    context: list[str] = []
    seen: set[tuple[str, str]] = set()
    for label, key in CONTEXT_FIELDS:
        value = metadata.get(key)
        if value in (None, ""):
            continue
        rendered = str(value)
        marker = (label, rendered)
        if marker in seen:
            continue
        seen.add(marker)
        context.append(f"{label}: {rendered}")
    return context
