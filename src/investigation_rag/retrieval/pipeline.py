from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.retrieval.context_expansion import NeighborContextExpander
from investigation_rag.retrieval.fusion import reciprocal_rank_fusion
from investigation_rag.retrieval.keyword_search import KeywordSearcher
from investigation_rag.retrieval.profiles import RetrievalProfile, STANDARD_PROFILE
from investigation_rag.retrieval.ranking import SearchResult
from investigation_rag.retrieval.reranking import LexicalReranker, Reranker
from investigation_rag.retrieval.vector_search import VectorSearcher
from investigation_rag.utils.timing import TimingStep


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
        self.last_timing_steps: list[TimingStep] = []
        self.last_process_feedback: list[str] = []

    def search(
        self,
        investigation_id: int,
        query: str,
        limit: int | None = None,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        self.last_timing_steps = []
        self.last_process_feedback = []
        timings: list[TimingStep] = []
        total_started = perf_counter()
        final_limit = limit or self.profile.final_answer_chunks
        if self.profile.query_expansion_enabled and self.query_expander is not None:
            started = perf_counter()
            queries = self.expand_queries(query)
            prompt_estimate = getattr(self.query_expander, "last_prompt_estimate", "")
            if isinstance(prompt_estimate, str) and prompt_estimate:
                self.last_process_feedback.append(prompt_estimate)
            timings.append(TimingStep("Query expansion", perf_counter() - started))
        else:
            queries = [query]
        keyword_results: list[SearchResult] = []
        vector_results: list[SearchResult] = []
        keyword_seconds = 0.0
        vector_seconds = 0.0

        for retrieval_query in queries:
            started = perf_counter()
            keyword_results.extend(
                tag_results(
                    self._search_with_scope(
                        self.keyword,
                        investigation_id,
                        retrieval_query,
                        self.profile.keyword_candidates,
                        folder_paths,
                        source_types,
                    ),
                    retrieval_query,
                    "keyword",
                )
            )
            keyword_seconds += perf_counter() - started
            if self.vector is not None:
                started = perf_counter()
                vector_results.extend(
                    tag_results(
                        self._search_with_scope(
                            self.vector,
                            investigation_id,
                            retrieval_query,
                            self.profile.vector_candidates,
                            folder_paths,
                            source_types,
                        ),
                        retrieval_query,
                        "vector",
                    )
                )
                vector_seconds += perf_counter() - started

        timings.append(TimingStep(step_label("Keyword retrieval", queries), keyword_seconds))
        if self.vector is not None:
            timings.append(TimingStep(step_label("Vector retrieval", queries), vector_seconds))

        started = perf_counter()
        keyword_results = dedupe_ranked(keyword_results)
        vector_results = dedupe_ranked(vector_results)
        fused = reciprocal_rank_fusion(
            keyword_results,
            vector_results,
            limit=self.profile.fusion_candidates,
        )
        timings.append(TimingStep("Result fusion", perf_counter() - started))
        if not fused:
            timings.append(TimingStep("Retrieval total", perf_counter() - total_started))
            self.last_timing_steps = timings
            return []

        started = perf_counter()
        reranked = self.reranker.rerank(query, fused[: self.profile.rerank_candidates], self.profile.rerank_candidates)
        timings.append(TimingStep("Rerank", perf_counter() - started))
        selected = reranked[:final_limit]

        if self.profile.context_expansion_enabled:
            started = perf_counter()
            expander = NeighborContextExpander(
                self.db,
                before=self.profile.neighbor_before,
                after=self.profile.neighbor_after,
                source_types=source_types,
            )
            selected = expander.expand(selected, budget=final_limit)
            timings.append(TimingStep("Context expansion", perf_counter() - started))
        timings.append(TimingStep("Retrieval total", perf_counter() - total_started))
        self.last_timing_steps = timings
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
        source_types: list[str] | None,
    ) -> list[SearchResult]:
        if folder_paths is None and source_types is None:
            return searcher.search(investigation_id, query, limit=limit)
        if source_types is None:
            return searcher.search(investigation_id, query, limit=limit, folder_paths=folder_paths)
        return searcher.search(
            investigation_id,
            query,
            limit=limit,
            folder_paths=folder_paths,
            source_types=source_types,
        )


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


def step_label(label: str, queries: list[str]) -> str:
    if len(queries) <= 1:
        return label
    return f"{label} ({len(queries)} queries)"
