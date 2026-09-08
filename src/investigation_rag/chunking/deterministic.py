from __future__ import annotations

from investigation_rag.chunking.base import Chunk, Chunker
from investigation_rag.chunking.pdf_layout import (
    DEFAULT_PDF_OVERLAP_MODE,
    DEFAULT_PDF_SOFT_MIN_CHARS,
    PdfChunkingConfig,
    chunk_pdf_document,
    has_pdf_layout,
    pdf_chunker_version,
)
from investigation_rag.extractors.base import ExtractedDocument
from investigation_rag.chunking.structured import chunk_structured_document, finalize_chunks


class DeterministicChunker(Chunker):
    version = "structured-v2"

    def __init__(
        self,
        target_chars: int = 2800,
        max_chars: int = 4800,
        overlap_chars: int = 240,
        pdf_soft_min_chars: int = DEFAULT_PDF_SOFT_MIN_CHARS,
        pdf_overlap_mode: str = DEFAULT_PDF_OVERLAP_MODE,
        max_tokens: int = 1400,
    ):
        self.target_chars = max(1, target_chars)
        self.max_chars = max(self.target_chars, max_chars)
        self.max_tokens = max(32, max_tokens)
        self.overlap_chars = max(0, overlap_chars)
        self.pdf_config = PdfChunkingConfig(
            target_chars=self.target_chars,
            max_chars=self.max_chars,
            soft_min_chars=pdf_soft_min_chars,
            overlap_mode=pdf_overlap_mode,
            overlap_chars=self.overlap_chars,
        ).normalized()
        self.pdf_version = pdf_chunker_version(self.pdf_config)

    def version_for_source_type(self, source_type: str) -> str:
        if source_type.lower().lstrip(".") == "pdf":
            return self.pdf_version
        return self.version

    def chunk_document(self, document: ExtractedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._chunk_one_document(document, chunks)
        def attachments(parent: ExtractedDocument) -> None:
            for attachment in parent.attachments:
                self._chunk_one_document(attachment, chunks)
                attachments(attachment)
        attachments(document)
        return finalize_chunks(chunks, self.max_chars, self.max_tokens, self.overlap_chars)

    def _chunk_one_document(self, document: ExtractedDocument, chunks: list[Chunk]) -> None:
        if has_pdf_layout(document):
            chunks.extend(chunk_pdf_document(document, self.pdf_config))
            return
        chunks.extend(chunk_structured_document(document, self.target_chars, self.max_chars))
