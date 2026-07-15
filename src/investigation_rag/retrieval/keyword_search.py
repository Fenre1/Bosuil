from __future__ import annotations

import re

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.retrieval.ranking import SearchResult, row_to_result


class KeywordSearcher:
    def __init__(self, db: SQLiteDB):
        self.db = db

    def search(
        self,
        investigation_id: int,
        query: str,
        limit: int | None = 20,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> list[SearchResult]:
        fts_query = to_fts_query(query)
        if not fts_query:
            return []
        rows = self.db.keyword_search(
            investigation_id,
            fts_query,
            limit,
            folder_paths=folder_paths,
            source_types=source_types,
        )
        results: list[SearchResult] = []
        for row in rows:
            try:
                rank = float(row["rank"])
            except Exception:
                rank = 0.0
            results.append(row_to_result(row, score=-rank))
        return results


def to_fts_query(query: str) -> str:
    return " ".join(query_terms(query))


def query_terms(query: str) -> list[str]:
    tokens = re.findall(r"[\w\-]+", query, flags=re.UNICODE)
    terms = []
    seen = set()
    for token in tokens:
        key = token.casefold()
        if key in seen:
            continue
        seen.add(key)
        terms.append(token)
    return terms
