from __future__ import annotations

from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.embedding_cache import EmbeddingCache
from investigation_rag.retrieval.ranking import SearchResult, row_to_result


class VectorSearcher:
    def __init__(
        self,
        db: SQLiteDB,
        store: LanceDBStore,
        embeddings: EmbeddingCache,
        query_instruction: str = "",
    ):
        self.db = db
        self.store = store
        self.embeddings = embeddings
        self.query_instruction = query_instruction.strip()

    def search(
        self,
        investigation_id: int,
        query: str,
        limit: int = 20,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        if folder_paths == [] or source_types == []:
            return []
        query_vector = self.embeddings.embed_texts([format_embedding_query(query, self.query_instruction)])[0]
        vector_rows = self.store.search(
            query_vector,
            investigation_id=investigation_id,
            embedding_model=self.embeddings.model,
            limit=limit,
            oversample=8 if folder_paths is not None or source_types is not None else 3,
            source_types=source_types,
        )
        chunk_ids = [str(row["chunk_id"]) for row in vector_rows if row.get("chunk_id")]
        db_rows = self.db.get_chunks_by_ids_in_folders(
            chunk_ids,
            folder_paths,
            source_types=source_types,
            active_only=True,
        )
        distances = {str(row["chunk_id"]): float(row.get("_distance", 0.0)) for row in vector_rows}
        return [
            row_to_result(row, score=1.0 / (1.0 + distances.get(str(row["id"]), 0.0)))
            for row in db_rows[:limit]
        ]


def format_embedding_query(query: str, instruction: str = "") -> str:
    instruction = instruction.strip()
    query = query.strip()
    if not instruction or query.casefold().startswith("instruct:"):
        return query
    return f"Instruct: {instruction}\nQuery: {query}"
