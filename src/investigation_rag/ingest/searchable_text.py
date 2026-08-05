from __future__ import annotations

from pathlib import Path


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

    context = _metadata_context(metadata)
    return "\n".join([*context, "", str(chunk_text)]) if context else str(chunk_text)


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
