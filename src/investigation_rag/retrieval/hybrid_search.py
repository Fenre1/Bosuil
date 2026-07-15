from __future__ import annotations

from investigation_rag.retrieval.keyword_search import KeywordSearcher
from investigation_rag.retrieval.fusion import reciprocal_rank_fusion
from investigation_rag.retrieval.ranking import SearchResult
from investigation_rag.retrieval.vector_search import VectorSearcher


class HybridSearcher:
    def __init__(self, keyword: KeywordSearcher, vector: VectorSearcher | None = None):
        self.keyword = keyword
        self.vector = vector

    def search(
        self,
        investigation_id: int,
        query: str,
        limit: int = 20,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        keyword_results = self.keyword.search(
            investigation_id,
            query,
            limit=limit,
            folder_paths=folder_paths,
            source_types=source_types,
        )
        vector_results: list[SearchResult] = []
        if self.vector is not None:
            vector_results = self.vector.search(
                investigation_id,
                query,
                limit=limit,
                folder_paths=folder_paths,
                source_types=source_types,
            )
        return reciprocal_rank_fusion(keyword_results, vector_results, limit=limit)
