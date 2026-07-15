from __future__ import annotations

from pathlib import Path
from typing import Any


class LanceDBUnavailable(RuntimeError):
    pass


class LanceDBStore:
    def __init__(self, db_dir: Path, table_name: str = "chunks"):
        self.db_dir = db_dir
        self.table_name = table_name
        self._db = None

    def _connect(self) -> Any:
        if self._db is not None:
            return self._db
        try:
            import lancedb
        except ImportError as exc:
            raise LanceDBUnavailable("LanceDB is not installed. Install project dependencies to enable vector search.") from exc
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_dir))
        return self._db

    def _table_names(self) -> set[str]:
        db = self._connect()
        return set(db.table_names())

    def has_table(self) -> bool:
        return self.table_name in self._table_names()

    def upsert_chunks(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        db = self._connect()
        if self.table_name not in set(db.table_names()):
            db.create_table(self.table_name, data=records)
            return
        table = db.open_table(self.table_name)
        self._drop_deprecated_text_column(table)
        for record in records:
            chunk_id = str(record["chunk_id"]).replace("'", "''")
            table.delete(f"chunk_id = '{chunk_id}'")
        table.add(records)

    @staticmethod
    def _drop_deprecated_text_column(table: Any) -> None:
        field_names = set(table.schema.names)
        if "text" not in field_names:
            return
        if not hasattr(table, "drop_columns"):
            raise LanceDBUnavailable(
                "LanceDB table contains deprecated raw text. Upgrade LanceDB or rebuild the vector store."
            )
        table.drop_columns(["text"])

    def search(
        self,
        vector: list[float],
        investigation_id: int,
        embedding_model: str = "",
        limit: int = 10,
        oversample: int = 3,
        source_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if source_types == []:
            return []
        if not self.has_table():
            return []
        table = self._connect().open_table(self.table_name)
        query_limit = max(limit * oversample, limit)
        where_clause = f"investigation_id = {int(investigation_id)}"
        if embedding_model:
            safe_model = embedding_model.replace("'", "''")
            where_clause += f" AND embedding_model = '{safe_model}'"
        if source_types is not None:
            normalized = sorted(
                {
                    str(source_type).strip().lower().lstrip(".")
                    for source_type in source_types
                    if str(source_type).strip()
                }
            )
            if not normalized:
                return []
            safe_types = ", ".join("'" + source_type.replace("'", "''") + "'" for source_type in normalized)
            where_clause += f" AND source_type IN ({safe_types})"
        try:
            rows = (
                table.search(vector)
                .where(where_clause)
                .limit(query_limit)
                .to_list()
            )
        except Exception as exc:
            raise LanceDBUnavailable(f"LanceDB vector search failed: {exc}") from exc
        return rows[:query_limit]
