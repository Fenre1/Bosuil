from __future__ import annotations

from dataclasses import replace

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.retrieval.ranking import SearchResult, row_to_result


class NeighborContextExpander:
    def __init__(
        self,
        db: SQLiteDB,
        before: int = 1,
        after: int = 1,
        source_types: list[str] | None = None,
    ):
        self.db = db
        self.before = before
        self.after = after
        self.source_types = source_types

    def expand(self, selected: list[SearchResult], budget: int) -> list[SearchResult]:
        packed: list[SearchResult] = []
        seen: set[str] = set()

        for result in selected:
            if len(packed) >= budget:
                break
            if result.chunk_id not in seen:
                packed.append(result)
                seen.add(result.chunk_id)
                if len(packed) >= budget:
                    break
            neighbors = self.db.get_neighbor_chunks(
                result.file_id,
                result.chunk_index,
                before=self.before,
                after=self.after,
                source_types=self.source_types,
            )
            if not neighbors:
                neighbors = []
            for row in neighbors:
                if len(packed) >= budget:
                    break
                neighbor = row_to_result(row, score=result.score)
                if neighbor.chunk_id in seen:
                    continue
                packed.append(
                    replace(
                        neighbor,
                        retrieval_source=result.retrieval_source,
                        retrieval_queries=result.retrieval_queries,
                        keyword_rank=result.keyword_rank,
                        vector_rank=result.vector_rank,
                        fused_rank=result.fused_rank,
                        rerank_score=result.rerank_score,
                        context_role="neighbor",
                    )
                )
                seen.add(neighbor.chunk_id)

        return packed[:budget]
