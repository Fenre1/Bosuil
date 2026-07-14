from __future__ import annotations

from investigation_rag.retrieval.ranking import SearchResult


def citation_label(index: int, result: SearchResult) -> str:
    if result.page_number:
        location = f"page {result.page_number}"
    else:
        location = f"chunk {result.chunk_index + 1}"
    return f"[{index}] {result.path} ({location})"


def format_citations(results: list[SearchResult]) -> str:
    return "\n".join(citation_label(index, result) for index, result in enumerate(results, start=1))
