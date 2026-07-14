from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Investigation:
    id: int
    name: str
    notes: str


class SQLiteDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        with self.transaction() as conn:
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            self._ensure_column(conn, "chunks", "metadata_json", "TEXT NOT NULL DEFAULT '{}'")
            self._ensure_column(conn, "chunks", "display_markdown", "TEXT")
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES(?, ?)",
                (1, utc_now()),
            )

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def create_investigation(self, name: str, roots: list[Path], notes: str = "") -> int:
        now = utc_now()
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO investigations(name, notes, created_at, updated_at)
                VALUES(?, ?, ?, ?)
                """,
                (name, notes, now, now),
            )
            investigation_id = int(cur.lastrowid)
            for root in roots:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO investigation_roots(investigation_id, root_path, created_at)
                    VALUES(?, ?, ?)
                    """,
                    (investigation_id, str(Path(root).resolve()), now),
                )
            return investigation_id

    def update_investigation(self, investigation_id: int, name: str, notes: str, roots: list[Path]) -> None:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                "UPDATE investigations SET name = ?, notes = ?, updated_at = ? WHERE id = ?",
                (name, notes, now, investigation_id),
            )
            conn.execute("DELETE FROM investigation_roots WHERE investigation_id = ?", (investigation_id,))
            for root in roots:
                conn.execute(
                    """
                    INSERT INTO investigation_roots(investigation_id, root_path, created_at)
                    VALUES(?, ?, ?)
                    """,
                    (investigation_id, str(Path(root).resolve()), now),
                )

    def list_investigations(self) -> list[Investigation]:
        with self.connection() as conn:
            rows = conn.execute("SELECT id, name, notes FROM investigations ORDER BY name").fetchall()
        return [Investigation(id=int(r["id"]), name=r["name"], notes=r["notes"]) for r in rows]

    def get_investigation(self, investigation_id: int) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM investigations WHERE id = ?", (investigation_id,)).fetchone()

    def get_investigation_by_name(self, name: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM investigations WHERE name = ?", (name,)).fetchone()

    def get_roots(self, investigation_id: int) -> list[Path]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT root_path FROM investigation_roots WHERE investigation_id = ? ORDER BY root_path",
                (investigation_id,),
            ).fetchall()
        return [Path(r["root_path"]) for r in rows]

    def start_scan(self, investigation_id: int) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO scan_runs(investigation_id, started_at, status) VALUES(?, ?, ?)",
                (investigation_id, utc_now(), "running"),
            )
            return int(cur.lastrowid)

    def finish_scan(self, scan_id: int, status: str, message: str = "") -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE scan_runs SET finished_at = ?, status = ?, message = ? WHERE id = ?",
                (utc_now(), status, message, scan_id),
            )

    def get_file_by_path(self, investigation_id: int, normalized_path: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT * FROM files
                WHERE investigation_id = ? AND normalized_path = ?
                """,
                (investigation_id, normalized_path),
            ).fetchone()

    def get_file_by_stable_id(self, investigation_id: int, stable_id: str | None) -> sqlite3.Row | None:
        if not stable_id:
            return None
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT * FROM files
                WHERE investigation_id = ? AND stable_id = ? AND status != 'inactive'
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (investigation_id, stable_id),
            ).fetchone()

    def get_active_file_by_content_hash(self, investigation_id: int, content_hash: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT * FROM files
                WHERE investigation_id = ? AND content_hash = ? AND status IN ('active', 'disappeared')
                ORDER BY updated_at DESC
                LIMIT 1
                """,
                (investigation_id, content_hash),
            ).fetchone()

    def upsert_file(
        self,
        investigation_id: int,
        path: Path,
        normalized_path: str,
        stable_id: str | None,
        size: int,
        modified_ns: int,
        content_hash: str | None,
        source_type: str,
        file_id: int | None = None,
    ) -> int:
        now = utc_now()
        with self.transaction() as conn:
            if file_id is not None:
                conn.execute(
                    """
                    UPDATE files
                    SET path = ?, normalized_path = ?, stable_id = ?, size = ?, modified_ns = ?,
                        content_hash = ?, source_type = ?, status = 'active', last_seen_at = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        str(path),
                        normalized_path,
                        stable_id,
                        size,
                        modified_ns,
                        content_hash,
                        source_type,
                        now,
                        now,
                        file_id,
                    ),
                )
                return file_id

            cur = conn.execute(
                """
                INSERT INTO files(
                    investigation_id, path, normalized_path, stable_id, size, modified_ns,
                    content_hash, source_type, status, first_seen_at, last_seen_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?)
                ON CONFLICT(investigation_id, normalized_path) DO UPDATE SET
                    stable_id = excluded.stable_id,
                    size = excluded.size,
                    modified_ns = excluded.modified_ns,
                    content_hash = excluded.content_hash,
                    source_type = excluded.source_type,
                    status = 'active',
                    last_seen_at = excluded.last_seen_at,
                    updated_at = excluded.updated_at
                """,
                (
                    investigation_id,
                    str(path),
                    normalized_path,
                    stable_id,
                    size,
                    modified_ns,
                    content_hash,
                    source_type,
                    now,
                    now,
                    now,
                ),
            )
            if cur.lastrowid:
                return int(cur.lastrowid)
            row = conn.execute(
                """
                SELECT id FROM files
                WHERE investigation_id = ? AND normalized_path = ?
                """,
                (investigation_id, normalized_path),
            ).fetchone()
            return int(row["id"])

    def mark_file_extraction(self, file_id: int, status: str, error: str | None, extractor_version: str) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE files
                SET extraction_status = ?, extraction_error = ?, extractor_version = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error, extractor_version, utc_now(), file_id),
            )

    def update_chunk_paths_for_file(self, file_id: int, path: Path) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE chunks SET path = ?, updated_at = ? WHERE file_id = ?",
                (str(path), utc_now(), file_id),
            )
            conn.execute("UPDATE chunks_fts SET path = ? WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)", (str(path), file_id))

    def set_chunks_inactive_for_file(self, file_id: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE chunks SET is_active = 0, updated_at = ? WHERE file_id = ? AND is_active = 1",
                (utc_now(), file_id),
            )
            conn.execute(
                "DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ? AND is_active = 0)",
                (file_id,),
            )

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        now = utc_now()
        with self.transaction() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks(
                        id, investigation_id, file_id, content_hash, chunk_index, chunk_text, display_markdown,
                        chunk_text_hash, source_type, page_number, email_subject, email_sender,
                        email_recipients, email_date, attachment_name, metadata_json, path, is_active,
                        extractor_version, chunker_version, embedding_model, created_at, updated_at
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["investigation_id"],
                        chunk["file_id"],
                        chunk["content_hash"],
                        chunk["chunk_index"],
                        chunk["chunk_text"],
                        chunk.get("display_markdown"),
                        chunk["chunk_text_hash"],
                        chunk["source_type"],
                        chunk.get("page_number"),
                        chunk.get("email_subject"),
                        chunk.get("email_sender"),
                        chunk.get("email_recipients"),
                        chunk.get("email_date"),
                        chunk.get("attachment_name"),
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                        chunk["path"],
                        1 if chunk.get("is_active", True) else 0,
                        chunk["extractor_version"],
                        chunk["chunker_version"],
                        chunk.get("embedding_model", ""),
                        now,
                        now,
                    ),
                )
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk["id"],))
                conn.execute(
                    """
                    INSERT INTO chunks_fts(chunk_id, investigation_id, chunk_text, path)
                    VALUES(?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["investigation_id"],
                        chunk.get("searchable_text", chunk["chunk_text"]),
                        chunk["path"],
                    ),
                )

    def update_chunk_metadata(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        with self.transaction() as conn:
            for chunk in chunks:
                conn.execute(
                    """
                    UPDATE chunks
                    SET display_markdown = ?, source_type = ?, page_number = ?, email_subject = ?, email_sender = ?,
                        email_recipients = ?, email_date = ?, attachment_name = ?,
                        metadata_json = ?, updated_at = ?
                    WHERE id = ? AND file_id = ? AND is_active = 1
                    """,
                    (
                        chunk.get("display_markdown"),
                        chunk["source_type"],
                        chunk.get("page_number"),
                        chunk.get("email_subject"),
                        chunk.get("email_sender"),
                        chunk.get("email_recipients"),
                        chunk.get("email_date"),
                        chunk.get("attachment_name"),
                        json.dumps(chunk.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                        utc_now(),
                        chunk["id"],
                        chunk["file_id"],
                    ),
                )
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (chunk["id"],))
                conn.execute(
                    """
                    INSERT INTO chunks_fts(chunk_id, investigation_id, chunk_text, path)
                    VALUES(?, ?, ?, ?)
                    """,
                    (
                        chunk["id"],
                        chunk["investigation_id"],
                        chunk.get("searchable_text", chunk["chunk_text"]),
                        chunk["path"],
                    ),
                )

    def active_chunks_missing_metadata(self, file_id: int) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT 1
                FROM chunks
                WHERE file_id = ? AND is_active = 1
                  AND (metadata_json IS NULL OR metadata_json = '' OR metadata_json = '{}')
                LIMIT 1
                """,
                (file_id,),
            ).fetchone()
        return row is not None

    def get_chunk(self, chunk_id: str) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,)).fetchone()

    def active_chunks_for_file(self, file_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM chunks WHERE file_id = ? AND is_active = 1 ORDER BY chunk_index",
                (file_id,),
            ).fetchall()

    def active_chunks_for_investigation(self, investigation_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT c.*
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                WHERE c.investigation_id = ?
                  AND c.is_active = 1
                  AND f.status = 'active'
                ORDER BY f.normalized_path, c.chunk_index
                """,
                (investigation_id,),
            ).fetchall()

    def active_chunk_summaries_for_investigation(self, investigation_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT
                  c.id,
                  c.file_id,
                  c.chunk_index,
                  c.source_type,
                  c.page_number,
                  c.email_subject,
                  c.attachment_name,
                  c.path
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                WHERE c.investigation_id = ?
                  AND c.is_active = 1
                  AND f.status = 'active'
                ORDER BY f.normalized_path, c.chunk_index
                """,
                (investigation_id,),
            ).fetchall()

    def active_chunk_count_for_file(self, file_id: int) -> int:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE file_id = ? AND is_active = 1",
                (file_id,),
            ).fetchone()
        return int(row["count"])

    def has_active_chunks_for_versions(self, file_id: int, extractor_version: str, chunker_version: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM chunks
                WHERE file_id = ?
                  AND is_active = 1
                  AND extractor_version = ?
                  AND chunker_version = ?
                """,
                (file_id, extractor_version, chunker_version),
            ).fetchone()
        return int(row["count"]) > 0

    def file_extraction_done_for_version(self, file_id: int, extractor_version: str) -> bool:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT extraction_status, extractor_version
                FROM files
                WHERE id = ?
                """,
                (file_id,),
            ).fetchone()
        return bool(
            row
            and row["extraction_status"] == "done"
            and row["extractor_version"] == extractor_version
        )

    def active_chunks_need_embedding_model(self, file_id: int, embedding_model: str) -> bool:
        if not embedding_model:
            return False
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT COUNT(*) AS count
                FROM chunks
                WHERE file_id = ?
                  AND is_active = 1
                  AND embedding_model != ?
                """,
                (file_id, embedding_model),
            ).fetchone()
        return int(row["count"]) > 0

    def update_embedding_model_for_chunks(self, chunk_ids: list[str], embedding_model: str) -> None:
        if not chunk_ids:
            return
        placeholders = ",".join("?" for _ in chunk_ids)
        with self.transaction() as conn:
            conn.execute(
                f"UPDATE chunks SET embedding_model = ?, updated_at = ? WHERE id IN ({placeholders})",
                [embedding_model, utc_now(), *chunk_ids],
            )

    def get_chunks_by_ids(self, chunk_ids: list[str], active_only: bool = True) -> list[sqlite3.Row]:
        if not chunk_ids:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        active_sql = "AND is_active = 1" if active_only else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({placeholders}) {active_sql}",
                chunk_ids,
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_chunks_by_ids_in_folders(
        self,
        chunk_ids: list[str],
        folder_paths: list[str] | None,
        active_only: bool = True,
    ) -> list[sqlite3.Row]:
        if folder_paths is None:
            return self.get_chunks_by_ids(chunk_ids, active_only=active_only)
        if not chunk_ids or not folder_paths:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        folder_sql, folder_params = self._folder_scope_sql(folder_paths, alias="f")
        active_sql = "AND c.is_active = 1" if active_only else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                WHERE c.id IN ({placeholders})
                  {active_sql}
                  AND ({folder_sql})
                """,
                [*chunk_ids, *folder_params],
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_neighbor_chunks(
        self,
        file_id: int,
        chunk_index: int,
        before: int = 1,
        after: int = 1,
    ) -> list[sqlite3.Row]:
        start = max(0, chunk_index - max(0, before))
        end = chunk_index + max(0, after)
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT *
                FROM chunks
                WHERE file_id = ?
                  AND is_active = 1
                  AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
                """,
                (file_id, start, end),
            ).fetchall()

    def get_extraction_cache(self, content_hash: str, extractor_version: str, source_type: str) -> dict[str, Any] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT extracted_json FROM extraction_cache
                WHERE content_hash = ? AND extractor_version = ? AND source_type = ?
                """,
                (content_hash, extractor_version, source_type),
            ).fetchone()
        return json.loads(row["extracted_json"]) if row else None

    def set_extraction_cache(
        self,
        content_hash: str,
        extractor_version: str,
        source_type: str,
        payload: dict[str, Any],
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO extraction_cache(content_hash, extractor_version, source_type, extracted_json, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (content_hash, extractor_version, source_type, json.dumps(payload), utc_now()),
            )

    def get_embedding(self, chunk_text_hash: str, embedding_model: str) -> list[float] | None:
        with self.connection() as conn:
            row = conn.execute(
                """
                SELECT vector_json FROM embedding_cache
                WHERE chunk_text_hash = ? AND embedding_model = ?
                """,
                (chunk_text_hash, embedding_model),
            ).fetchone()
        return json.loads(row["vector_json"]) if row else None

    def set_embedding(self, chunk_text_hash: str, embedding_model: str, vector: list[float]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache(chunk_text_hash, embedding_model, vector_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (chunk_text_hash, embedding_model, json.dumps(vector), utc_now()),
            )

    def keyword_search(
        self,
        investigation_id: int,
        query: str,
        limit: int | None,
        folder_paths: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        if folder_paths == []:
            return []
        folder_sql = ""
        folder_params: list[str] = []
        if folder_paths is not None:
            scoped_sql, folder_params = self._folder_scope_sql(folder_paths, alias="f")
            folder_sql = f"AND ({scoped_sql})"
        limit_sql = "LIMIT ?" if limit is not None else ""
        with self.connection() as conn:
            try:
                params: list[object] = [query, investigation_id, *folder_params]
                if limit is not None:
                    params.append(limit)
                return conn.execute(
                    f"""
                    SELECT c.*, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    JOIN chunks c ON c.id = chunks_fts.chunk_id
                    JOIN files f ON f.id = c.file_id
                    WHERE chunks_fts MATCH ?
                      AND c.investigation_id = ?
                      AND c.is_active = 1
                      {folder_sql}
                    ORDER BY rank
                    {limit_sql}
                    """,
                    params,
                ).fetchall()
            except sqlite3.OperationalError:
                like = f"%{query}%"
                params = [investigation_id, like, *folder_params]
                if limit is not None:
                    params.append(limit)
                return conn.execute(
                    f"""
                    SELECT c.*, 0.0 AS rank
                    FROM chunks c
                    JOIN files f ON f.id = c.file_id
                    WHERE c.investigation_id = ? AND c.is_active = 1 AND c.chunk_text LIKE ?
                      {folder_sql}
                    ORDER BY c.updated_at DESC
                    {limit_sql}
                    """,
                    params,
                ).fetchall()

    def list_active_document_folders(self, investigation_id: int) -> list[Path]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT path
                FROM files
                WHERE investigation_id = ?
                  AND status = 'active'
                  AND extraction_status = 'done'
                ORDER BY normalized_path
                """,
                (investigation_id,),
            ).fetchall()
        folders = {Path(str(row["path"])).parent for row in rows}
        return sorted(folders, key=lambda path: str(path).casefold())

    @staticmethod
    def _folder_scope_sql(folder_paths: list[str], alias: str = "files") -> tuple[str, list[str]]:
        clauses = []
        params = []
        for folder_path in folder_paths:
            normalized = str(Path(folder_path).expanduser().resolve()).casefold().rstrip("\\/")
            clauses.append(
                f"({alias}.normalized_path = ? OR {alias}.normalized_path LIKE ? OR {alias}.normalized_path LIKE ?)"
            )
            params.extend([normalized, f"{normalized}\\%", f"{normalized}/%"])
        return " OR ".join(clauses), params

    def active_files_for_investigation(self, investigation_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM files WHERE investigation_id = ? AND status = 'active'",
                (investigation_id,),
            ).fetchall()

    def add_disappeared_file(self, investigation_id: int, file_id: int, path: str, last_seen_at: str) -> None:
        with self.transaction() as conn:
            chunk_count_row = conn.execute(
                "SELECT COUNT(*) AS count FROM chunks WHERE file_id = ? AND is_active = 1",
                (file_id,),
            ).fetchone()
            chunk_count = int(chunk_count_row["count"])
            now = utc_now()
            conn.execute(
                """
                INSERT INTO disappeared_files(
                    investigation_id, file_id, path, last_seen_at, chunk_count, status, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, 'pending', ?, ?)
                ON CONFLICT(investigation_id, file_id) DO UPDATE SET
                    path = excluded.path,
                    last_seen_at = excluded.last_seen_at,
                    chunk_count = excluded.chunk_count,
                    status = CASE
                      WHEN disappeared_files.status = 'resolved' THEN 'pending'
                      ELSE disappeared_files.status
                    END,
                    updated_at = excluded.updated_at
                """,
                (investigation_id, file_id, path, last_seen_at, chunk_count, now, now),
            )
            conn.execute(
                "UPDATE files SET status = 'disappeared', updated_at = ? WHERE id = ? AND status = 'active'",
                (now, file_id),
            )

    def list_disappeared_files(self, investigation_id: int, status: str = "pending") -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT d.*, i.name AS investigation_name
                FROM disappeared_files d
                JOIN investigations i ON i.id = d.investigation_id
                WHERE d.investigation_id = ? AND d.status = ?
                ORDER BY d.updated_at DESC
                """,
                (investigation_id, status),
            ).fetchall()

    def resolve_disappeared(self, disappeared_id: int, action: str, new_path: str | None = None) -> None:
        now = utc_now()
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM disappeared_files WHERE id = ?", (disappeared_id,)).fetchone()
            if row is None:
                raise ValueError(f"Unknown disappeared file id {disappeared_id}")
            file_id = int(row["file_id"])
            if action == "inactive":
                conn.execute(
                    "UPDATE files SET status = 'inactive', updated_at = ? WHERE id = ?",
                    (now, file_id),
                )
                conn.execute(
                    "UPDATE chunks SET is_active = 0, updated_at = ? WHERE file_id = ?",
                    (now, file_id),
                )
                conn.execute("DELETE FROM chunks_fts WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)", (file_id,))
                status = "inactive"
            elif action == "keep":
                conn.execute(
                    "UPDATE files SET status = 'active', updated_at = ? WHERE id = ?",
                    (now, file_id),
                )
                status = "kept"
            elif action == "update_path":
                if not new_path:
                    raise ValueError("new_path is required")
                normalized = str(Path(new_path).expanduser().resolve()).casefold()
                conn.execute(
                    """
                    UPDATE files
                    SET path = ?, normalized_path = ?, status = 'active', updated_at = ?
                    WHERE id = ?
                    """,
                    (new_path, normalized, now, file_id),
                )
                conn.execute(
                    "UPDATE chunks SET path = ?, updated_at = ? WHERE file_id = ?",
                    (new_path, now, file_id),
                )
                conn.execute(
                    "UPDATE chunks_fts SET path = ? WHERE chunk_id IN (SELECT id FROM chunks WHERE file_id = ?)",
                    (new_path, file_id),
                )
                status = "resolved"
            else:
                raise ValueError(f"Unknown action {action}")

            conn.execute(
                "UPDATE disappeared_files SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, disappeared_id),
            )

    def resolve_disappeared_for_file(self, file_id: int) -> None:
        now = utc_now()
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE disappeared_files
                SET status = 'resolved', updated_at = ?
                WHERE file_id = ? AND status = 'pending'
                """,
                (now, file_id),
            )
