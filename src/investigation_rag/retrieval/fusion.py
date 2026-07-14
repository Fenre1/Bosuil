from __future__ import annotations

from dataclasses import replace

from investigation_rag.retrieval.ranking import SearchResult


RRF_K = 60


def reciprocal_rank_fusion(
    keyword_results: list[SearchResult],
    vector_results: list[SearchResult],
    limit: int,
    rrf_k: int = RRF_K,
) -> list[SearchResult]:
    combined: dict[str, SearchResult] = {}
    scores: dict[str, float] = {}
    sources: dict[str, set[str]] = {}
    keyword_ranks: dict[str, int] = {}
    vector_ranks: dict[str, int] = {}
    queries: dict[str, set[str]] = {}

    add_ranked_results(keyword_results, "keyword", combined, scores, sources, queries, keyword_ranks, rrf_k)
    add_ranked_results(vector_results, "vector", combined, scores, sources, queries, vector_ranks, rrf_k)

    fused = []
    for chunk_id, result in combined.items():
        fused.append(
            replace(
                result,
                score=scores[chunk_id],
                retrieval_source="+".join(sorted(sources.get(chunk_id, set()))),
                retrieval_queries=tuple(sorted(queries.get(chunk_id, set()))),
                keyword_rank=keyword_ranks.get(chunk_id),
                vector_rank=vector_ranks.get(chunk_id),
            )
        )

    fused.sort(key=lambda item: item.score, reverse=True)
    trimmed = fused[:limit]
    return [replace(result, fused_rank=index) for index, result in enumerate(trimmed, start=1)]


def add_ranked_results(
    results: list[SearchResult],
    source: str,
    combined: dict[str, SearchResult],
    scores: dict[str, float],
    sources: dict[str, set[str]],
    queries: dict[str, set[str]],
    ranks: dict[str, int],
    rrf_k: int,
) -> None:
    seen_for_source: set[str] = set()
    for rank, result in enumerate(results, start=1):
        if result.chunk_id in seen_for_source:
            continue
        seen_for_source.add(result.chunk_id)
        combined.setdefault(result.chunk_id, result)
        scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        sources.setdefault(result.chunk_id, set()).add(source)
        ranks.setdefault(result.chunk_id, rank)
        result_queries = result.retrieval_queries or ()
        if not result_queries:
            result_queries = ("",)
        query_set = queries.setdefault(result.chunk_id, set())
        query_set.update(query for query in result_queries if query)
