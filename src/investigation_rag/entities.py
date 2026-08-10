from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from investigation_rag.db.sqlite_db import SQLiteDB


DEFAULT_ENTITY_LABELS = (
    "person",
    "organization",
    "location",
    "geopolitical entity",
    "facility",
    "event",
    "product",
    "law",
    "date",
    "money",
)
DEFAULT_GLINER_MODEL = "gliner-community/gliner_large-v2.5"


def normalize_entity_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def load_gliner_model(model_name: str = DEFAULT_GLINER_MODEL):
    try:
        from gliner import GLiNER
    except ImportError as exc:
        raise RuntimeError(
            'Entity extraction requires GLiNER. Install it with: pip install -e ".[entities]"'
        ) from exc
    return GLiNER.from_pretrained(model_name)


def extract_entities(
    db: SQLiteDB,
    investigation_id: int,
    labels: list[str],
    *,
    model: Any | None = None,
    threshold: float = 0.5,
    progress: Callable[[int, int], None] | None = None,
) -> int:
    labels = [label.strip() for label in labels if label.strip()]
    if not labels:
        raise ValueError("Provide at least one entity type.")
    active_chunks = db.get_active_chunks_for_entity_extraction(investigation_id)
    predictor = model or load_gliner_model()
    mentions: list[dict[str, Any]] = []
    total = len(active_chunks)
    for index, chunk in enumerate(active_chunks, start=1):
        text = str(chunk["chunk_text"])
        for found in predictor.predict_entities(text, labels, threshold=threshold):
            name = str(found.get("text", "")).strip()
            normalized = normalize_entity_name(name)
            if not normalized:
                continue
            mentions.append(
                {
                    "name": name,
                    "normalized_name": normalized,
                    "entity_type": str(found["label"]).strip().casefold(),
                    "file_id": int(chunk["file_id"]),
                    "chunk_id": str(chunk["id"]),
                    "mention_text": name,
                    "start_offset": int(found["start"]),
                    "end_offset": int(found["end"]),
                    "confidence": float(found.get("score", 0.0)),
                }
            )
        if progress is not None:
            progress(index, total)
    db.replace_entity_extraction(investigation_id, mentions)
    return len(mentions)
