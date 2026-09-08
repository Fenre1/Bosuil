from __future__ import annotations

from investigation_rag.retrieval.ranking import SearchResult, result_location_label


def citation_label(index: int, result: SearchResult) -> str:
    chain = (result.metadata or {}).get("attachment_chain", [])
    attachment = " → ".join(str(part["name"]) for part in chain) if chain else result.attachment_name
    source = f"{result.path} → {attachment}" if attachment else result.path
    return f"[{index}] {source} ({result_location_label(result)})"


def format_citations(results: list[SearchResult]) -> str:
    return "\n".join(citation_label(index, result) for index, result in enumerate(results, start=1))
