from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from investigation_rag.chunking.deterministic import DeterministicChunker
from investigation_rag.extractors import get_extractor_version_for_suffix


METADATA_FTS_SIGNATURE = "chunk-metadata-fts-v2"
EMBEDDING_INPUT_SIGNATURE = "embedding-input-stable-v2"


@dataclass(frozen=True)
class PipelineSignatures:
    extraction: str
    chunking: str
    metadata: str
    embedding_input: str
    embedding_model: str


def pipeline_signatures(
    suffix: str,
    chunker: DeterministicChunker,
    embedding_model: str,
) -> PipelineSignatures:
    """Return every processing generation that can invalidate stored work.

    These signatures are deliberately stage-specific. Changing an embedding
    model must not force extraction, while changing any chunk-size setting must
    invalidate chunks even when the chunker algorithm name stays unchanged.
    """

    source_type = suffix.lower().lstrip(".")
    chunking_payload: dict[str, object] = {
        "algorithm": chunker.version_for_source_type(source_type),
        "source_type": source_type,
        "target_chars": chunker.target_chars,
        "max_chars": chunker.max_chars,
        "overlap_chars": chunker.overlap_chars,
    }
    if source_type == "pdf":
        chunking_payload["pdf"] = {
            "soft_min_chars": chunker.pdf_config.soft_min_chars,
            "overlap_mode": chunker.pdf_config.overlap_mode,
        }

    return PipelineSignatures(
        extraction=get_extractor_version_for_suffix(suffix),
        chunking=_versioned_digest("chunking-v2", chunking_payload),
        metadata=METADATA_FTS_SIGNATURE,
        embedding_input=EMBEDDING_INPUT_SIGNATURE,
        embedding_model=embedding_model.strip(),
    )


def _versioned_digest(label: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{label}:{digest}"
