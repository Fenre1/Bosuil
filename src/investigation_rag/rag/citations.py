from __future__ import annotations

from investigation_rag.retrieval.ranking import SearchResult, result_location_label


def citation_label(index: int, result: SearchResult) -> str:
    return f"[{index}] {result.path} ({result_location_label(result)})"


def format_citations(results: list[SearchResult]) -> str:
    return "\n".join(citation_label(index, result) for index, result in enumerate(results, start=1))
