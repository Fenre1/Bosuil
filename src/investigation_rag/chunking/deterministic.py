from __future__ import annotations

import re

from investigation_rag.chunking.base import Chunk, Chunker
from investigation_rag.extractors.base import ExtractedDocument, ExtractedSection


class DeterministicChunker(Chunker):
    version = "deterministic-v1"

    def __init__(self, target_chars: int = 2800, max_chars: int = 4800, overlap_chars: int = 240):
        self.target_chars = target_chars
        self.max_chars = max(target_chars, max_chars)
        self.overlap_chars = max(0, overlap_chars)

    def chunk_document(self, document: ExtractedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        self._chunk_one_document(document, chunks)
        for attachment in document.attachments:
            self._chunk_one_document(attachment, chunks)
        for index, chunk in enumerate(chunks):
            chunk.chunk_index = index
        return chunks

    def _chunk_one_document(self, document: ExtractedDocument, chunks: list[Chunk]) -> None:
        for section in document.sections:
            chunks.extend(self._chunk_section(section, document.metadata))

    def _chunk_section(self, section: ExtractedSection, document_metadata: dict[str, object]) -> list[Chunk]:
        text = normalize_text(section.text)
        if not text:
            return []
        paragraphs = split_paragraphs(text)
        chunk_texts = pack_paragraphs(paragraphs, self.target_chars, self.max_chars, self.overlap_chars)
        chunk_markdown = chunk_display_markdown(section.display_markdown, paragraphs, chunk_texts)
        merged_metadata = dict(document_metadata)
        merged_metadata.update(section.metadata)
        return [
            Chunk(
                chunk_index=0,
                text=chunk_text,
                source_type=section.source_type,
                page_number=section.page_number,
                display_markdown=chunk_markdown[index] if index < len(chunk_markdown) else None,
                metadata=merged_metadata,
            )
            for index, chunk_text in enumerate(chunk_texts)
        ]


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    if len(parts) == 1 and len(parts[0]) > 2400:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", parts[0]) if part.strip()]
    return parts


def pack_paragraphs(paragraphs: list[str], target_chars: int, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            if current:
                chunks.append("\n\n".join(current).strip())
                current = []
                current_len = 0
            chunks.extend(split_large_text(paragraph, max_chars, overlap_chars))
            continue

        projected = current_len + len(paragraph) + (2 if current else 0)
        if current and projected > target_chars:
            chunk = "\n\n".join(current).strip()
            chunks.append(chunk)
            current = make_overlap_seed(chunk, overlap_chars)
            current_len = sum(len(item) for item in current) + max(0, len(current) - 1) * 2

        current.append(paragraph)
        current_len += len(paragraph) + (2 if current_len else 0)

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def split_large_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap_chars, start + 1)
    return [chunk for chunk in chunks if chunk]


def make_overlap_seed(previous_chunk: str, overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return []
    seed = previous_chunk[-overlap_chars:].strip()
    return [seed] if seed else []


def chunk_display_markdown(
    display_markdown: str | None,
    plain_paragraphs: list[str],
    chunk_texts: list[str],
) -> list[str | None]:
    if not display_markdown:
        return [None for _ in chunk_texts]

    markdown = normalize_text(display_markdown)
    if len(chunk_texts) == 1:
        return [markdown]

    markdown_paragraphs = split_paragraphs(markdown)
    if len(markdown_paragraphs) != len(plain_paragraphs):
        return pack_markdown_by_chunk_lengths(markdown_paragraphs, chunk_texts)

    chunks: list[str | None] = []
    paragraph_index = 0
    for chunk_text in chunk_texts:
        selected: list[str] = []
        selected_plain: list[str] = []
        while paragraph_index < len(plain_paragraphs):
            selected.append(markdown_paragraphs[paragraph_index])
            selected_plain.append(plain_paragraphs[paragraph_index])
            paragraph_index += 1
            if "\n\n".join(selected_plain).strip() == chunk_text:
                break
        chunks.append("\n\n".join(selected).strip() or None)

    if paragraph_index != len(plain_paragraphs):
        return pack_markdown_by_chunk_lengths(markdown_paragraphs, chunk_texts)
    return chunks


def pack_markdown_by_chunk_lengths(markdown_paragraphs: list[str], chunk_texts: list[str]) -> list[str | None]:
    if not markdown_paragraphs:
        return [None for _ in chunk_texts]

    chunks: list[str | None] = []
    paragraph_index = 0
    for chunk_index, chunk_text in enumerate(chunk_texts):
        selected: list[str] = []
        visible_len = 0
        target_len = len(chunk_text)
        while paragraph_index < len(markdown_paragraphs):
            paragraph = markdown_paragraphs[paragraph_index]
            selected.append(paragraph)
            visible_len += len(markdown_visible_text(paragraph))
            paragraph_index += 1
            remaining_chunks = len(chunk_texts) - chunk_index - 1
            remaining_paragraphs = len(markdown_paragraphs) - paragraph_index
            if remaining_chunks > 0 and remaining_paragraphs <= remaining_chunks:
                break
            if remaining_chunks > 0 and visible_len >= target_len * 0.75:
                break
        chunks.append("\n\n".join(selected).strip() or None)

    if paragraph_index < len(markdown_paragraphs) and chunks:
        remainder = "\n\n".join(markdown_paragraphs[paragraph_index:]).strip()
        chunks[-1] = "\n\n".join(part for part in (chunks[-1], remainder) if part).strip()
    return chunks


def markdown_visible_text(markdown: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", markdown, flags=re.MULTILINE)
    text = text.replace("**", "").replace("*", "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text
