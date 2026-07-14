from __future__ import annotations

from dataclasses import replace

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.retrieval.context_expansion import NeighborContextExpander
from investigation_rag.retrieval.fusion import reciprocal_rank_fusion
from investigation_rag.retrieval.keyword_search import KeywordSearcher
from investigation_rag.retrieval.profiles import RetrievalProfile, STANDARD_PROFILE
from investigation_rag.retrieval.ranking import SearchResult
from investigation_rag.retrieval.reranking import LexicalReranker, Reranker
from investigation_rag.retrieval.vector_search import VectorSearcher


class RetrievalPipeline:
    def __init__(
        self,
        keyword: KeywordSearcher,
        vector: VectorSearcher | None,
        db: SQLiteDB,
        profile: RetrievalProfile = STANDARD_PROFILE,
        reranker: Reranker | None = None,
        query_expander: object | None = None,
    ):
        self.keyword = keyword
        self.vector = vector
        self.db = db
        self.profile = profile
        self.reranker = reranker or LexicalReranker()
        self.query_expander = query_expander
        self.final_answer_chunks = profile.final_answer_chunks

    def search(
        self,
        investigation_id: int,
        query: str,
        limit: int | None = None,
        folder_paths: list[str] | None = None,
    ) -> list[SearchResult]:
        final_limit = limit or self.profile.final_answer_chunks
        queries = self.expand_queries(query)
        keyword_results: list[SearchResult] = []
        vector_results: list[SearchResult] = []

        for retrieval_query in queries:
            keyword_results.extend(
                tag_results(
                    self._search_with_scope(
                        self.keyword,
                        investigation_id,
                        retrieval_query,
                        self.profile.keyword_candidates,
                        folder_paths,
                    ),
                    retrieval_query,
                    "keyword",
                )
            )
            if self.vector is not None:
                vector_results.extend(
                    tag_results(
                        self._search_with_scope(
                            self.vector,
                            investigation_id,
                            retrieval_query,
                            self.profile.vector_candidates,
                            folder_paths,
                        ),
                        retrieval_query,
                        "vector",
                    )
                )

        keyword_results = dedupe_ranked(keyword_results)
        vector_results = dedupe_ranked(vector_results)
        fused = reciprocal_rank_fusion(
            keyword_results,
            vector_results,
            limit=self.profile.fusion_candidates,
        )
        if not fused:
            return []

        reranked = self.reranker.rerank(query, fused[: self.profile.rerank_candidates], self.profile.rerank_candidates)
        selected = reranked[:final_limit]

        if self.profile.context_expansion_enabled:
            expander = NeighborContextExpander(
                self.db,
                before=self.profile.neighbor_before,
                after=self.profile.neighbor_after,
            )
            return expander.expand(selected, budget=final_limit)
        return selected

    def expand_queries(self, query: str) -> list[str]:
        if not self.profile.query_expansion_enabled or self.query_expander is None:
            return [query]
        expanded = self.query_expander.expand(query)
        return expanded or [query]

    @staticmethod
    def _search_with_scope(
        searcher: object,
        investigation_id: int,
        query: str,
        limit: int,
        folder_paths: list[str] | None,
    ) -> list[SearchResult]:
        if folder_paths is None:
            return searcher.search(investigation_id, query, limit=limit)
        return searcher.search(investigation_id, query, limit=limit, folder_paths=folder_paths)


def tag_results(results: list[SearchResult], query: str, source: str) -> list[SearchResult]:
    tagged = []
    for result in results:
        tagged.append(
            replace(
                result,
                retrieval_source=source,
                retrieval_queries=(query,),
            )
        )
    return tagged


def dedupe_ranked(results: list[SearchResult]) -> list[SearchResult]:
    deduped: dict[str, SearchResult] = {}
    queries: dict[str, set[str]] = {}
    for result in results:
        if result.chunk_id not in deduped:
            deduped[result.chunk_id] = result
        queries.setdefault(result.chunk_id, set()).update(result.retrieval_queries)
    return [
        replace(result, retrieval_queries=tuple(sorted(queries.get(chunk_id, set()))))
        for chunk_id, result in deduped.items()
    ]
