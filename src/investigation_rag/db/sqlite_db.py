from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import get_ident, local
from typing import Any, Collection, Iterator, Mapping


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Investigation:
    id: int
    name: str
    notes: str


@dataclass(frozen=True)
class FilePipelineState:
    file_id: int
    source_content_hash: str = ""
    extraction_signature: str = ""
    chunking_signature: str = ""
    metadata_signature: str = ""
    embedding_input_signature: str = ""
    embedding_model: str = ""
    extraction_status: str = "pending"
    chunk_status: str = "pending"
    metadata_status: str = "pending"
    vector_status: str = "pending"
    active_vector_count: int = 0
    updated_at: str = ""


@dataclass(frozen=True)
class ScanFileState:
    id: int
    path: str
    normalized_path: str
    stable_id: str | None
    size: int
    modified_ns: int
    content_hash: str | None
    status: str
    source_type: str
    active_chunk_count: int
    source_content_hash: str
    extraction_signature: str
    chunking_signature: str
    metadata_signature: str
    embedding_input_signature: str
    embedding_model: str
    extraction_status: str
    chunk_status: str
    metadata_status: str
    vector_status: str
    active_vector_count: int
    pipeline_state_exists: bool
    extractor_version: str | None
    extraction_error: str | None
    first_seen_at: str
    last_seen_at: str
    updated_at: str

    @property
    def file_id(self) -> int:
        """Alias used by file-oriented database methods."""

        return self.id


# Kept as a source-compatible name for code developed against the initial bulk
# manifest API.  New scan code should use ScanFileState.
ScanManifestEntry = ScanFileState


class SQLiteDB:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._scan_local = local()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        active = self._active_scan_connection()
        if active is not None:
            yield active
            return
        conn = self.connect()
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        active = self._active_scan_connection()
        if active is not None:
            counter = int(getattr(self._scan_local, "savepoint_counter", 0)) + 1
            self._scan_local.savepoint_counter = counter
            savepoint = f"scan_transaction_{counter}"
            active.execute(f"SAVEPOINT {savepoint}")
            try:
                yield active
                active.execute(f"RELEASE SAVEPOINT {savepoint}")
            except Exception:
                active.execute(f"ROLLBACK TO SAVEPOINT {savepoint}")
                active.execute(f"RELEASE SAVEPOINT {savepoint}")
                raise
            return
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _active_scan_connection(self) -> sqlite3.Connection | None:
        conn = getattr(self._scan_local, "connection", None)
        if conn is None:
            return None
        owner_thread = int(getattr(self._scan_local, "owner_thread", -1))
        if owner_thread != get_ident():
            raise RuntimeError("A scan database session may only be used by its owning thread.")
        return conn

    @contextmanager
    def scan_session(self, investigation_id: int) -> Iterator["ScanDBSession"]:
        """Open one worker-owned connection for all database work in a scan.

        While this context is active, the existing ``connection()`` and
        ``transaction()`` helpers transparently reuse the same connection.
        Existing SQLiteDB methods therefore gain connection reuse without
        needing scan-specific duplicates.  ``transaction()`` uses nested
        savepoints so callers may wrap several operations in ``session.batch()``.
        """

        if self._active_scan_connection() is not None:
            raise RuntimeError("A scan database session is already active on this thread.")
        conn = self.connect()
        self._scan_local.connection = conn
        self._scan_local.owner_thread = get_ident()
        self._scan_local.savepoint_counter = 0
        session = ScanDBSession(self, conn, investigation_id)
        try:
            yield session
            if conn.in_transaction:
                conn.commit()
        except Exception:
            if conn.in_transaction:
                conn.rollback()
            raise
        finally:
            self._scan_local.connection = None
            self._scan_local.owner_thread = None
            self._scan_local.savepoint_counter = 0
            conn.close()

    def initialize(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        with self.transaction() as conn:
            chunks_exists = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chunks'"
            ).fetchone()
            if chunks_exists is not None:
                chunk_columns = {
                    str(row["name"]) for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
                }
                if "fts_rowid" not in chunk_columns:
                    raise RuntimeError(
                        "This investigation uses the legacy ingest schema. "
                        "Recreate the investigation and ingest it as new."
                    )
            conn.executescript(schema_path.read_text(encoding="utf-8"))
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES(?, ?)",
                (1, utc_now()),
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES(?, ?)",
                (2, utc_now()),
            )
            conn.execute(
                "INSERT OR IGNORE INTO schema_version(version, applied_at) VALUES(?, ?)",
                (3, utc_now()),
            )

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

    def get_file(self, file_id: int) -> sqlite3.Row | None:
        with self.connection() as conn:
            return conn.execute("SELECT * FROM files WHERE id = ?", (file_id,)).fetchone()

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

    def update_chunk_paths_for_file(
        self,
        file_id: int,
        path: Path,
        *,
        size: int | None = None,
        modified_ns: int | None = None,
    ) -> None:
        with self.transaction() as conn:
            rowids = self._active_chunk_rowids_for_file(conn, file_id)
            metadata_sql = """
                json_set(
                    CASE WHEN json_valid(metadata_json) THEN metadata_json ELSE '{}' END,
                    '$.filename', ?,
                    '$.path', ?
                )
            """
            metadata_params: list[object] = [path.name, str(path)]
            if size is not None and modified_ns is not None:
                metadata_sql = """
                    json_set(
                        CASE WHEN json_valid(metadata_json) THEN metadata_json ELSE '{}' END,
                        '$.filename', ?,
                        '$.path', ?,
                        '$.size_bytes', ?,
                        '$.modified_ns', ?
                    )
                """
                metadata_params.extend([int(size), int(modified_ns)])
            conn.execute(
                f"""
                UPDATE chunks
                SET path = ?, metadata_json = {metadata_sql}, updated_at = ?
                WHERE file_id = ?
                """,
                [str(path), *metadata_params, utc_now(), file_id],
            )
            self._update_fts_paths(conn, rowids, path)

    def set_chunks_inactive_for_file(self, file_id: int) -> None:
        with self.transaction() as conn:
            rowids = self._active_chunk_rowids_for_file(conn, file_id)
            conn.execute(
                "UPDATE chunks SET is_active = 0, updated_at = ? WHERE file_id = ? AND is_active = 1",
                (utc_now(), file_id),
            )
            self._delete_fts_rowids(conn, rowids)
            conn.execute("UPDATE files SET active_chunk_count = 0 WHERE id = ?", (file_id,))

    def insert_chunks(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        now = utc_now()
        chunk_records = [
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
            )
            for chunk in chunks
        ]
        file_ids = sorted({int(chunk["file_id"]) for chunk in chunks})
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT INTO chunks(
                    id, investigation_id, file_id, content_hash, chunk_index, chunk_text, display_markdown,
                    chunk_text_hash, source_type, page_number, email_subject, email_sender,
                    email_recipients, email_date, attachment_name, metadata_json, path, is_active,
                    extractor_version, chunker_version, embedding_model, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    investigation_id = excluded.investigation_id,
                    file_id = excluded.file_id,
                    content_hash = excluded.content_hash,
                    chunk_index = excluded.chunk_index,
                    chunk_text = excluded.chunk_text,
                    display_markdown = excluded.display_markdown,
                    chunk_text_hash = excluded.chunk_text_hash,
                    source_type = excluded.source_type,
                    page_number = excluded.page_number,
                    email_subject = excluded.email_subject,
                    email_sender = excluded.email_sender,
                    email_recipients = excluded.email_recipients,
                    email_date = excluded.email_date,
                    attachment_name = excluded.attachment_name,
                    metadata_json = excluded.metadata_json,
                    path = excluded.path,
                    is_active = excluded.is_active,
                    extractor_version = excluded.extractor_version,
                    chunker_version = excluded.chunker_version,
                    embedding_model = excluded.embedding_model,
                    updated_at = excluded.updated_at
                """,
                chunk_records,
            )
            rowids = self._chunk_rowids_by_id(conn, [str(chunk["id"]) for chunk in chunks])
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks_fts(rowid, chunk_text, path)
                VALUES(?, ?, ?)
                """,
                [
                    (
                        rowids[str(chunk["id"])],
                        chunk.get("searchable_text", chunk["chunk_text"]),
                        chunk["path"],
                    )
                    for chunk in chunks
                ],
            )
            conn.executemany(
                """
                UPDATE files
                SET active_chunk_count = (
                    SELECT COUNT(*) FROM chunks
                    WHERE chunks.file_id = files.id AND chunks.is_active = 1
                )
                WHERE id = ?
                """,
                [(file_id,) for file_id in file_ids],
            )

    def update_chunk_metadata(self, chunks: list[dict[str, Any]]) -> None:
        if not chunks:
            return
        now = utc_now()
        metadata_records = [
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
                now,
                chunk["id"],
                chunk["file_id"],
            )
            for chunk in chunks
        ]
        with self.transaction() as conn:
            conn.executemany(
                """
                UPDATE chunks
                SET display_markdown = ?, source_type = ?, page_number = ?, email_subject = ?, email_sender = ?,
                    email_recipients = ?, email_date = ?, attachment_name = ?,
                    metadata_json = ?, updated_at = ?
                WHERE id = ? AND file_id = ? AND is_active = 1
                """,
                metadata_records,
            )
            rowids = self._chunk_rowids_by_id(
                conn,
                [str(chunk["id"]) for chunk in chunks],
                active_only=True,
            )
            conn.executemany(
                """
                INSERT OR REPLACE INTO chunks_fts(rowid, chunk_text, path)
                VALUES(?, ?, ?)
                """,
                [
                    (
                        rowids[str(chunk["id"])],
                        chunk.get("searchable_text", chunk["chunk_text"]),
                        chunk["path"],
                    )
                    for chunk in chunks
                    if str(chunk["id"]) in rowids
                ],
            )

    @staticmethod
    def _chunk_rowids_by_id(
        conn: sqlite3.Connection,
        chunk_ids: Collection[str],
        *,
        active_only: bool = False,
    ) -> dict[str, int]:
        result: dict[str, int] = {}
        unique_ids = list(dict.fromkeys(str(chunk_id) for chunk_id in chunk_ids))
        active_sql = "AND is_active = 1" if active_only else ""
        for start in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH_SIZE):
            batch = unique_ids[start : start + _SQLITE_PARAMETER_BATCH_SIZE]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT fts_rowid, id FROM chunks WHERE id IN ({placeholders}) {active_sql}",
                batch,
            ).fetchall()
            result.update({str(row["id"]): int(row["fts_rowid"]) for row in rows})
        return result

    @staticmethod
    def _active_chunk_rowids_for_file(conn: sqlite3.Connection, file_id: int) -> list[int]:
        rows = conn.execute(
            "SELECT fts_rowid FROM chunks WHERE file_id = ? AND is_active = 1",
            (file_id,),
        ).fetchall()
        return [int(row["fts_rowid"]) for row in rows]

    @staticmethod
    def _delete_fts_rowids(conn: sqlite3.Connection, rowids: Collection[int]) -> None:
        conn.executemany(
            "DELETE FROM chunks_fts WHERE rowid = ?",
            [(int(rowid),) for rowid in rowids],
        )

    @staticmethod
    def _update_fts_paths(
        conn: sqlite3.Connection,
        rowids: Collection[int],
        path: Path | str,
    ) -> None:
        conn.executemany(
            "UPDATE chunks_fts SET path = ? WHERE rowid = ?",
            [(str(path), int(rowid)) for rowid in rowids],
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

    def active_chunk_summaries_for_file(self, file_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT id, file_id, chunk_index, source_type, page_number,
                       email_subject, attachment_name, path
                FROM chunks
                WHERE file_id = ? AND is_active = 1
                ORDER BY chunk_index
                """,
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
        active_sql = "AND c.is_active = 1 AND f.status = 'active'" if active_only else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                WHERE c.id IN ({placeholders}) {active_sql}
                """,
                chunk_ids,
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_chunks_by_ids_in_folders(
        self,
        chunk_ids: list[str],
        folder_paths: list[str] | None,
        source_types: list[str] | None = None,
        active_only: bool = True,
    ) -> list[sqlite3.Row]:
        if folder_paths is None and source_types is None:
            return self.get_chunks_by_ids(chunk_ids, active_only=active_only)
        if not chunk_ids or folder_paths == [] or source_types == []:
            return []
        placeholders = ",".join("?" for _ in chunk_ids)
        filter_sql = []
        filter_params: list[str] = []
        if folder_paths is not None:
            folder_sql, folder_params = self._folder_scope_sql(folder_paths, alias="f")
            filter_sql.append(f"({folder_sql})")
            filter_params.extend(folder_params)
        if source_types is not None:
            source_type_sql, source_type_params = self._source_type_scope_sql(source_types, alias="c")
            filter_sql.append(f"({source_type_sql})")
            filter_params.extend(source_type_params)
        filters = f"AND {' AND '.join(filter_sql)}" if filter_sql else ""
        active_sql = "AND c.is_active = 1 AND f.status = 'active'" if active_only else ""
        with self.connection() as conn:
            rows = conn.execute(
                f"""
                SELECT c.*
                FROM chunks c
                JOIN files f ON f.id = c.file_id
                WHERE c.id IN ({placeholders})
                  {active_sql}
                  {filters}
                """,
                [*chunk_ids, *filter_params],
            ).fetchall()
        by_id = {row["id"]: row for row in rows}
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def get_neighbor_chunks(
        self,
        file_id: int,
        chunk_index: int,
        before: int = 1,
        after: int = 1,
        source_types: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        if source_types == []:
            return []
        start = max(0, chunk_index - max(0, before))
        end = chunk_index + max(0, after)
        source_type_sql = ""
        source_type_params: list[str] = []
        if source_types is not None:
            scoped_sql, source_type_params = self._source_type_scope_sql(source_types)
            source_type_sql = f"AND ({scoped_sql})"
        with self.connection() as conn:
            return conn.execute(
                f"""
                SELECT *
                FROM chunks
                WHERE file_id = ?
                  AND is_active = 1
                  AND chunk_index BETWEEN ? AND ?
                  {source_type_sql}
                ORDER BY chunk_index
                """,
                [file_id, start, end, *source_type_params],
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

    def get_embeddings(self, chunk_text_hashes: list[str], embedding_model: str) -> dict[str, list[float]]:
        if not chunk_text_hashes:
            return {}
        result: dict[str, list[float]] = {}
        unique_hashes = list(dict.fromkeys(chunk_text_hashes))
        with self.connection() as conn:
            for start in range(0, len(unique_hashes), 900):
                batch = unique_hashes[start : start + 900]
                placeholders = ",".join("?" for _ in batch)
                rows = conn.execute(
                    f"""
                    SELECT chunk_text_hash, vector_json
                    FROM embedding_cache
                    WHERE embedding_model = ?
                      AND chunk_text_hash IN ({placeholders})
                    """,
                    [embedding_model, *batch],
                ).fetchall()
                result.update(
                    {
                        str(row["chunk_text_hash"]): json.loads(row["vector_json"])
                        for row in rows
                    }
                )
        return result

    def set_embedding(self, chunk_text_hash: str, embedding_model: str, vector: list[float]) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO embedding_cache(chunk_text_hash, embedding_model, vector_json, created_at)
                VALUES(?, ?, ?, ?)
                """,
                (chunk_text_hash, embedding_model, json.dumps(vector), utc_now()),
            )

    def set_embeddings(self, embeddings: list[tuple[str, list[float]]], embedding_model: str) -> None:
        if not embeddings:
            return
        now = utc_now()
        with self.transaction() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO embedding_cache(
                    chunk_text_hash, embedding_model, vector_json, created_at
                )
                VALUES(?, ?, ?, ?)
                """,
                [
                    (text_hash, embedding_model, json.dumps(vector), now)
                    for text_hash, vector in embeddings
                ],
            )

    def keyword_search(
        self,
        investigation_id: int,
        query: str,
        limit: int | None,
        folder_paths: list[str] | None = None,
        source_types: list[str] | None = None,
    ) -> list[sqlite3.Row]:
        if folder_paths == [] or source_types == []:
            return []
        folder_sql = ""
        folder_params: list[str] = []
        if folder_paths is not None:
            scoped_sql, folder_params = self._folder_scope_sql(folder_paths, alias="f")
            folder_sql = f"AND ({scoped_sql})"
        source_type_sql = ""
        source_type_params: list[str] = []
        if source_types is not None:
            scoped_sql, source_type_params = self._source_type_scope_sql(source_types, alias="c")
            source_type_sql = f"AND ({scoped_sql})"
        limit_sql = "LIMIT ?" if limit is not None else ""
        with self.connection() as conn:
            try:
                params: list[object] = [query, investigation_id, *folder_params, *source_type_params]
                if limit is not None:
                    params.append(limit)
                return conn.execute(
                    f"""
                    SELECT c.*, bm25(chunks_fts) AS rank
                    FROM chunks_fts
                    JOIN chunks c ON c.fts_rowid = chunks_fts.rowid
                    JOIN files f ON f.id = c.file_id
                    WHERE chunks_fts MATCH ?
                      AND c.investigation_id = ?
                      AND c.is_active = 1
                      AND f.status = 'active'
                      {folder_sql}
                      {source_type_sql}
                    ORDER BY rank
                    {limit_sql}
                    """,
                    params,
                ).fetchall()
            except sqlite3.OperationalError:
                like = f"%{query}%"
                params = [investigation_id, like, *folder_params, *source_type_params]
                if limit is not None:
                    params.append(limit)
                return conn.execute(
                    f"""
                    SELECT c.*, 0.0 AS rank
                    FROM chunks c
                    JOIN files f ON f.id = c.file_id
                    WHERE c.investigation_id = ?
                      AND c.is_active = 1
                      AND f.status = 'active'
                      AND c.chunk_text LIKE ?
                      {folder_sql}
                      {source_type_sql}
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

    @staticmethod
    def _source_type_scope_sql(source_types: list[str], alias: str = "chunks") -> tuple[str, list[str]]:
        normalized = sorted(
            {
                str(source_type).strip().lower().lstrip(".")
                for source_type in source_types
                if str(source_type).strip()
            }
        )
        if not normalized:
            return "0", []
        placeholders = ",".join("?" for _ in normalized)
        return f"{alias}.source_type IN ({placeholders})", normalized

    def active_files_for_investigation(self, investigation_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                "SELECT * FROM files WHERE investigation_id = ? AND status = 'active'",
                (investigation_id,),
            ).fetchall()

    def file_browser_files_for_investigation(self, investigation_id: int) -> list[sqlite3.Row]:
        with self.connection() as conn:
            return conn.execute(
                """
                SELECT id, path, normalized_path, source_type, extraction_status,
                       active_chunk_count
                FROM files
                WHERE investigation_id = ? AND status = 'active'
                ORDER BY normalized_path
                """,
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
                    status = 'pending',
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
                rowids = self._active_chunk_rowids_for_file(conn, file_id)
                conn.execute(
                    "UPDATE files SET status = 'inactive', updated_at = ? WHERE id = ?",
                    (now, file_id),
                )
                conn.execute(
                    "UPDATE chunks SET is_active = 0, updated_at = ? WHERE file_id = ?",
                    (now, file_id),
                )
                conn.execute("UPDATE files SET active_chunk_count = 0 WHERE id = ?", (file_id,))
                self._delete_fts_rowids(conn, rowids)
                conn.execute("DELETE FROM file_pipeline_state WHERE file_id = ?", (file_id,))
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
                rowids = self._active_chunk_rowids_for_file(conn, file_id)
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
                    """
                    UPDATE chunks
                    SET path = ?,
                        metadata_json = json_set(
                            CASE WHEN json_valid(metadata_json) THEN metadata_json ELSE '{}' END,
                            '$.filename', ?,
                            '$.path', ?
                        ),
                        updated_at = ?
                    WHERE file_id = ?
                    """,
                    (new_path, Path(new_path).name, new_path, now, file_id),
                )
                self._update_fts_paths(conn, rowids, new_path)
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


_PIPELINE_STATE_COLUMNS = frozenset(
    {
        "source_content_hash",
        "extraction_signature",
        "chunking_signature",
        "metadata_signature",
        "embedding_input_signature",
        "embedding_model",
        "extraction_status",
        "chunk_status",
        "metadata_status",
        "vector_status",
        "active_vector_count",
    }
)
_SQLITE_PARAMETER_BATCH_SIZE = 800


class ScanDBSession:
    """Scan-scoped bulk APIs backed by one worker-owned SQLite connection."""

    def __init__(self, db: SQLiteDB, connection: sqlite3.Connection, investigation_id: int):
        self._db = db
        self._connection = connection
        self.investigation_id = investigation_id
        self._owner_thread = get_ident()

    def _assert_owner(self) -> None:
        if get_ident() != self._owner_thread:
            raise RuntimeError("A scan database session may only be used by its owning thread.")
        if self._db._active_scan_connection() is not self._connection:
            raise RuntimeError("The scan database session is no longer active.")

    @contextmanager
    def batch(self) -> Iterator["ScanDBSession"]:
        """Commit several scan writes atomically using a nestable savepoint."""

        self._assert_owner()
        with self._db.transaction() as conn:
            if conn is not self._connection:
                raise RuntimeError("The scan database connection changed unexpectedly.")
            yield self

    def load_manifest(self) -> list[ScanFileState]:
        """Load the complete file manifest and staged state in one query."""

        self._assert_owner()
        rows = self._connection.execute(
            """
            SELECT
                f.id AS file_id,
                f.path,
                f.normalized_path,
                f.stable_id,
                f.size,
                f.modified_ns,
                f.content_hash,
                f.status,
                f.source_type,
                f.extractor_version,
                f.extraction_status,
                f.extraction_error,
                f.active_chunk_count,
                f.first_seen_at,
                f.last_seen_at,
                f.updated_at,
                ps.file_id AS state_file_id,
                ps.source_content_hash,
                ps.extraction_signature,
                ps.chunking_signature,
                ps.metadata_signature,
                ps.embedding_input_signature,
                ps.embedding_model,
                ps.extraction_status AS pipeline_extraction_status,
                ps.chunk_status,
                ps.metadata_status,
                ps.vector_status,
                ps.active_vector_count
            FROM files AS f
            LEFT JOIN file_pipeline_state AS ps ON ps.file_id = f.id
            WHERE f.investigation_id = ?
            ORDER BY f.id
            """,
            (self.investigation_id,),
        ).fetchall()
        manifest: list[ScanFileState] = []
        for row in rows:
            state_exists = row["state_file_id"] is not None
            manifest.append(
                ScanFileState(
                    id=int(row["file_id"]),
                    path=str(row["path"]),
                    normalized_path=str(row["normalized_path"]),
                    stable_id=str(row["stable_id"]) if row["stable_id"] is not None else None,
                    size=int(row["size"]),
                    modified_ns=int(row["modified_ns"]),
                    content_hash=str(row["content_hash"]) if row["content_hash"] is not None else None,
                    status=str(row["status"]),
                    source_type=str(row["source_type"]),
                    active_chunk_count=int(row["active_chunk_count"]),
                    source_content_hash=(
                        str(row["source_content_hash"]) if state_exists else ""
                    ),
                    extraction_signature=(
                        str(row["extraction_signature"]) if state_exists else ""
                    ),
                    chunking_signature=(
                        str(row["chunking_signature"]) if state_exists else ""
                    ),
                    metadata_signature=(
                        str(row["metadata_signature"]) if state_exists else ""
                    ),
                    embedding_input_signature=(
                        str(row["embedding_input_signature"]) if state_exists else ""
                    ),
                    embedding_model=str(row["embedding_model"]) if state_exists else "",
                    extraction_status=(
                        str(row["pipeline_extraction_status"])
                        if state_exists
                        else "pending"
                    ),
                    chunk_status=str(row["chunk_status"]) if state_exists else "pending",
                    metadata_status=(
                        str(row["metadata_status"]) if state_exists else "pending"
                    ),
                    vector_status=str(row["vector_status"]) if state_exists else "pending",
                    active_vector_count=(
                        int(row["active_vector_count"]) if state_exists else 0
                    ),
                    pipeline_state_exists=state_exists,
                    extractor_version=(
                        str(row["extractor_version"])
                        if row["extractor_version"] is not None
                        else None
                    ),
                    extraction_error=(
                        str(row["extraction_error"])
                        if row["extraction_error"] is not None
                        else None
                    ),
                    first_seen_at=str(row["first_seen_at"]),
                    last_seen_at=str(row["last_seen_at"]),
                    updated_at=str(row["updated_at"]),
                )
            )
        return manifest

    def vector_eligible_chunk_ids(self) -> set[str]:
        """Return active chunk IDs that may remain in the vector store.

        Disappeared files intentionally remain eligible until the investigator
        resolves them as inactive, preserving the existing Keep workflow.
        """

        self._assert_owner()
        rows = self._connection.execute(
            """
            SELECT c.id
            FROM chunks AS c
            JOIN files AS f ON f.id = c.file_id
            WHERE c.investigation_id = ?
              AND c.is_active = 1
              AND f.status IN ('active', 'disappeared')
            """,
            (self.investigation_id,),
        ).fetchall()
        return {str(row["id"]) for row in rows}

    def set_pipeline_state(self, state: FilePipelineState) -> None:
        self.set_pipeline_states([state])

    def set_pipeline_states(self, states: Collection[FilePipelineState]) -> None:
        """Upsert complete state records as one atomic batch."""

        self._assert_owner()
        if not states:
            return
        normalized = list(states)
        for state in normalized:
            self._validate_vector_count(state.active_vector_count)
        now = utc_now()
        parameters = [
            (
                state.file_id,
                state.source_content_hash,
                state.extraction_signature,
                state.chunking_signature,
                state.metadata_signature,
                state.embedding_input_signature,
                state.embedding_model,
                state.extraction_status,
                state.chunk_status,
                state.metadata_status,
                state.vector_status,
                state.active_vector_count,
                now,
                state.file_id,
                self.investigation_id,
            )
            for state in normalized
        ]
        with self.batch():
            self._connection.executemany(
                """
                INSERT INTO file_pipeline_state(
                    file_id, source_content_hash, extraction_signature,
                    chunking_signature, metadata_signature,
                    embedding_input_signature, embedding_model,
                    extraction_status, chunk_status, metadata_status,
                    vector_status, active_vector_count,
                    updated_at
                )
                SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                FROM files
                WHERE id = ? AND investigation_id = ?
                ON CONFLICT(file_id) DO UPDATE SET
                    source_content_hash = excluded.source_content_hash,
                    extraction_signature = excluded.extraction_signature,
                    chunking_signature = excluded.chunking_signature,
                    metadata_signature = excluded.metadata_signature,
                    embedding_input_signature = excluded.embedding_input_signature,
                    embedding_model = excluded.embedding_model,
                    extraction_status = excluded.extraction_status,
                    chunk_status = excluded.chunk_status,
                    metadata_status = excluded.metadata_status,
                    vector_status = excluded.vector_status,
                    active_vector_count = excluded.active_vector_count,
                    updated_at = excluded.updated_at
                """,
                parameters,
            )

    def update_pipeline_state(
        self,
        file_id: int,
        changes: Mapping[str, object] | None = None,
        **fields: object,
    ) -> None:
        """Update selected staged-state fields, creating a pending row if needed."""

        self._assert_owner()
        merged = dict(changes or {})
        overlap = merged.keys() & fields.keys()
        if overlap:
            duplicate = ", ".join(sorted(overlap))
            raise ValueError(f"Pipeline-state field supplied twice: {duplicate}")
        merged.update(fields)
        if not merged:
            return
        unknown = merged.keys() - _PIPELINE_STATE_COLUMNS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(f"Unknown pipeline-state field(s): {names}")
        if "active_vector_count" in merged:
            self._validate_vector_count(int(merged["active_vector_count"]))

        now = utc_now()
        assignments = ", ".join(f"{column} = ?" for column in merged)
        values = list(merged.values())
        with self.batch():
            self._connection.execute(
                """
                INSERT INTO file_pipeline_state(file_id, updated_at)
                SELECT id, ? FROM files
                WHERE id = ? AND investigation_id = ?
                ON CONFLICT(file_id) DO NOTHING
                """,
                (now, file_id, self.investigation_id),
            )
            cursor = self._connection.execute(
                f"""
                UPDATE file_pipeline_state
                SET {assignments}, updated_at = ?
                WHERE file_id = ?
                  AND EXISTS (
                      SELECT 1 FROM files
                      WHERE files.id = file_pipeline_state.file_id
                        AND files.investigation_id = ?
                  )
                """,
                [*values, now, file_id, self.investigation_id],
            )
            if cursor.rowcount != 1:
                raise ValueError(
                    f"File {file_id} does not belong to investigation {self.investigation_id}."
                )

    def clear_pipeline_state(self, file_id: int) -> bool:
        """Remove staged state after content replacement; chunks are unaffected."""

        self._assert_owner()
        with self.batch():
            cursor = self._connection.execute(
                """
                DELETE FROM file_pipeline_state
                WHERE file_id = ?
                  AND EXISTS (
                      SELECT 1 FROM files
                      WHERE files.id = file_pipeline_state.file_id
                        AND files.investigation_id = ?
                  )
                """,
                (file_id, self.investigation_id),
            )
        return cursor.rowcount == 1

    def touch_seen_files(
        self,
        file_ids: Collection[int],
        *,
        seen_at: str | None = None,
    ) -> int:
        """Record presence for a complete scan without per-file write calls."""

        self._assert_owner()
        unique_ids = sorted({int(file_id) for file_id in file_ids})
        if not unique_ids:
            return 0
        timestamp = seen_at or utc_now()
        touched = 0
        with self.batch():
            for start in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH_SIZE):
                batch = unique_ids[start : start + _SQLITE_PARAMETER_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                cursor = self._connection.execute(
                    f"""
                    UPDATE files
                    SET last_seen_at = ?
                    WHERE investigation_id = ?
                      AND id IN ({placeholders})
                    """,
                    [timestamp, self.investigation_id, *batch],
                )
                touched += max(0, cursor.rowcount)
        return touched

    def mark_disappeared(self, file_ids: Collection[int]) -> int:
        """Queue explicitly absent active files and return the number changed.

        The caller performs the inventory diff and must omit IDs belonging to
        roots or subtrees whose inventory did not complete successfully.
        """

        self._assert_owner()
        unique_ids = sorted({int(file_id) for file_id in file_ids})
        if not unique_ids:
            return 0
        now = utc_now()
        changed = 0
        with self.batch():
            for start in range(0, len(unique_ids), _SQLITE_PARAMETER_BATCH_SIZE):
                batch = unique_ids[start : start + _SQLITE_PARAMETER_BATCH_SIZE]
                placeholders = ",".join("?" for _ in batch)
                self._connection.execute(
                    f"""
                    INSERT INTO disappeared_files(
                        investigation_id, file_id, path, last_seen_at,
                        chunk_count, status, created_at, updated_at
                    )
                    SELECT
                        f.investigation_id, f.id, f.path, f.last_seen_at,
                        f.active_chunk_count, 'pending', ?, ?
                    FROM files AS f
                    WHERE f.investigation_id = ?
                      AND f.status = 'active'
                      AND f.id IN ({placeholders})
                    ON CONFLICT(investigation_id, file_id) DO UPDATE SET
                        path = excluded.path,
                        last_seen_at = excluded.last_seen_at,
                        chunk_count = excluded.chunk_count,
                        status = 'pending',
                        updated_at = excluded.updated_at
                    """,
                    [now, now, self.investigation_id, *batch],
                )
                cursor = self._connection.execute(
                    f"""
                    UPDATE files
                    SET status = 'disappeared', updated_at = ?
                    WHERE investigation_id = ?
                      AND status = 'active'
                      AND id IN ({placeholders})
                    """,
                    [now, self.investigation_id, *batch],
                )
                changed += max(0, cursor.rowcount)
        return changed

    @staticmethod
    def _validate_vector_count(active_vector_count: int) -> None:
        if active_vector_count < 0:
            raise ValueError("The active vector count cannot be negative.")
