from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from threading import RLock
from typing import Any


class LanceDBUnavailable(RuntimeError):
    pass


class LanceDBStore:
    def __init__(self, db_dir: Path, table_name: str = "chunks"):
        self.db_dir = db_dir
        self.table_name = table_name
        self._db = None
        self._table = None
        self._lock = RLock()

    def _connect(self) -> Any:
        if self._db is not None:
            return self._db
        try:
            import lancedb
        except ImportError as exc:
            raise LanceDBUnavailable("LanceDB is not installed. Install project dependencies to enable vector search.") from exc
        self.db_dir.mkdir(parents=True, exist_ok=True)
        # The local table catalog is deliberately used here. A manifest-based
        # namespace can persist an absolute file:// URI and break after a move.
        self._db = lancedb.connect(str(self.db_dir), manifest_enabled=False)
        return self._db

    def has_table(self) -> bool:
        with self._lock:
            return self._open_table_if_exists() is not None

    def _open_table_if_exists(self) -> Any | None:
        """Return one cached table handle for this store's lifetime."""
        if self._table is not None:
            return self._table
        db = self._connect()
        if self.table_name not in set(db.table_names()):
            return None
        self._table = db.open_table(self.table_name)
        return self._table

    def _create_table(self, records: list[dict[str, Any]]) -> Any:
        db = self._connect()
        table = db.create_table(self.table_name, data=records)
        # Current LanceDB returns the table, but retain compatibility with
        # releases where create_table returned no handle.
        self._table = table if table is not None else db.open_table(self.table_name)
        return self._table

    def upsert_chunks(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                self._create_table(records)
                return
            self._drop_deprecated_text_column(table)
            chunk_ids = list(dict.fromkeys(str(record["chunk_id"]) for record in records))
            self._delete_string_values(table, "chunk_id", chunk_ids)
            table.add(records)

    def delete_vectors_for_files(
        self,
        file_ids: Iterable[int],
        *,
        investigation_id: int | None = None,
    ) -> None:
        """Delete all vector rows belonging to the supplied SQLite file IDs."""
        normalized_ids = sorted({int(file_id) for file_id in file_ids})
        if not normalized_ids:
            return
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return
            self._require_columns(table, {"file_id"})
            suffix = self._investigation_suffix(table, investigation_id)
            for start in range(0, len(normalized_ids), 500):
                values = ", ".join(str(file_id) for file_id in normalized_ids[start : start + 500])
                table.delete(f"file_id IN ({values}){suffix}")

    def delete_file(self, file_id: int, *, investigation_id: int | None = None) -> None:
        """Delete one file's vectors; convenience wrapper for scan integration."""
        self.delete_vectors_for_files([file_id], investigation_id=investigation_id)

    def set_files_active(
        self,
        file_ids: Iterable[int],
        active: bool,
        *,
        investigation_id: int | None = None,
    ) -> int:
        """Hide or restore file vectors without deleting or re-embedding them."""
        normalized_ids = sorted({int(file_id) for file_id in file_ids})
        if not normalized_ids:
            return 0
        updated = 0
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return 0
            self._require_columns(table, {"file_id", "is_active"})
            suffix = self._investigation_suffix(table, investigation_id)
            for start in range(0, len(normalized_ids), 500):
                values = ", ".join(str(file_id) for file_id in normalized_ids[start : start + 500])
                result = table.update(
                    where=f"file_id IN ({values}){suffix}",
                    values={"is_active": bool(active)},
                )
                updated += int(getattr(result, "rows_updated", 0))
        return updated

    def delete_vectors_for_chunks(
        self,
        chunk_ids: Iterable[str],
        *,
        investigation_id: int | None = None,
    ) -> None:
        """Delete vector rows for chunk IDs without touching embeddings or SQLite."""
        normalized_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids))
        if not normalized_ids:
            return
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return
            self._require_columns(table, {"chunk_id"})
            suffix = self._investigation_suffix(table, investigation_id)
            self._delete_string_values(table, "chunk_id", normalized_ids, suffix=suffix)

    def reconcile_active_chunks(self, investigation_id: int, active_chunk_ids: Iterable[str]) -> int:
        """Remove vectors that no longer have an active SQLite chunk.

        The caller supplies SQLite's active chunk IDs. No document content is
        opened and no embeddings are generated. The return value is the number
        of distinct stale chunk IDs removed.
        """
        active_ids = {str(chunk_id) for chunk_id in active_chunk_ids}
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return 0
            self._require_columns(table, {"chunk_id", "investigation_id"})
            selected_columns = ["chunk_id"]
            has_active_column = "is_active" in self._column_names(table)
            if has_active_column:
                selected_columns.append("is_active")
            rows = (
                table.search()
                .where(f"investigation_id = {int(investigation_id)}")
                .select(selected_columns)
                .to_list()
            )
            # Inactive rows may be pending disappeared-file review and must be
            # restorable without embedding. Final inactivation has an explicit
            # delete path; reconciliation only removes stale searchable rows.
            stored_ids = {
                str(row["chunk_id"])
                for row in rows
                if row.get("chunk_id") is not None and (not has_active_column or bool(row.get("is_active", True)))
            }
            stale_ids = sorted(stored_ids - active_ids)
            if stale_ids:
                self._delete_string_values(
                    table,
                    "chunk_id",
                    stale_ids,
                    suffix=f" AND investigation_id = {int(investigation_id)}",
                )
            return len(stale_ids)

    def update_file_path(self, file_id: int, path: Path | str, *, investigation_id: int | None = None) -> int:
        """Update mutable vector metadata after a move, without re-embedding."""
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return 0
            self._require_columns(table, {"file_id", "path"})
            where_clause = f"file_id = {int(file_id)}{self._investigation_suffix(table, investigation_id)}"
            result = table.update(where=where_clause, values={"path": str(path)})
            return int(getattr(result, "rows_updated", 0))

    def optimize(self) -> bool:
        """Run explicit vector-store maintenance.

        This is intentionally never invoked by writes or reconciliation. A
        caller can schedule it outside the foreground scan when appropriate.
        """
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return False
            optimize = getattr(table, "optimize", None)
            if not callable(optimize):
                return False
            optimize()
            return True

    def get_vectors_for_chunks(
        self,
        chunk_ids: Iterable[str],
        *,
        investigation_id: int | None = None,
        embedding_model: str = "",
    ) -> dict[str, list[float]]:
        """Return stored vectors for chunk IDs without performing a similarity search."""
        normalized_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids))
        if not normalized_ids:
            return {}
        vectors: dict[str, list[float]] = {}
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return vectors
            self._require_columns(table, {"chunk_id", "vector"})
            column_names = self._column_names(table)
            for start in range(0, len(normalized_ids), 500):
                batch = normalized_ids[start : start + 500]
                encoded = ", ".join("'" + value.replace("'", "''") + "'" for value in batch)
                where_clause = f"chunk_id IN ({encoded}){self._investigation_suffix(table, investigation_id)}"
                if "is_active" in column_names:
                    where_clause += " AND is_active = true"
                if embedding_model and "embedding_model" in column_names:
                    safe_model = embedding_model.replace("'", "''")
                    where_clause += f" AND embedding_model = '{safe_model}'"
                rows = (
                    table.search()
                    .where(where_clause)
                    .select(["chunk_id", "vector"])
                    .limit(len(batch))
                    .to_list()
                )
                for row in rows:
                    chunk_id = row.get("chunk_id")
                    vector = row.get("vector")
                    if chunk_id is not None and vector is not None:
                        vectors[str(chunk_id)] = [float(value) for value in vector]
        return vectors

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
        with self._lock:
            table = self._open_table_if_exists()
            if table is None:
                return []
            query_limit = max(limit * oversample, limit)
            where_clause = f"investigation_id = {int(investigation_id)}"
            if "is_active" in self._column_names(table):
                where_clause += " AND is_active = true"
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
                rows = table.search(vector).where(where_clause).limit(query_limit).to_list()
            except Exception as exc:
                raise LanceDBUnavailable(f"LanceDB vector search failed: {exc}") from exc
            return rows[:query_limit]

    @staticmethod
    def _column_names(table: Any) -> set[str]:
        return set(table.schema.names)

    @classmethod
    def _require_columns(cls, table: Any, required: set[str]) -> None:
        missing = required - cls._column_names(table)
        if missing:
            names = ", ".join(sorted(missing))
            raise LanceDBUnavailable(f"LanceDB table is missing required column(s): {names}")

    @classmethod
    def _investigation_suffix(cls, table: Any, investigation_id: int | None) -> str:
        if investigation_id is None:
            return ""
        cls._require_columns(table, {"investigation_id"})
        return f" AND investigation_id = {int(investigation_id)}"

    @staticmethod
    def _delete_string_values(table: Any, column: str, values: list[str], *, suffix: str = "") -> None:
        for start in range(0, len(values), 500):
            batch = values[start : start + 500]
            encoded = ", ".join("'" + value.replace("'", "''") + "'" for value in batch)
            table.delete(f"{column} IN ({encoded}){suffix}")
