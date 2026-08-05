from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from investigation_rag.db.sqlite_db import FilePipelineState, SQLiteDB


class CountingSQLiteDB(SQLiteDB):
    def __init__(self, path: Path):
        super().__init__(path)
        self.connection_opens = 0

    def connect(self) -> sqlite3.Connection:
        self.connection_opens += 1
        return super().connect()


def add_file(
    db: SQLiteDB,
    investigation_id: int,
    path: Path,
    *,
    stable_id: str,
    content_hash: str = "content-hash",
) -> int:
    return db.upsert_file(
        investigation_id=investigation_id,
        path=path,
        normalized_path=str(path).casefold(),
        stable_id=stable_id,
        size=10,
        modified_ns=20,
        content_hash=content_hash,
        source_type="txt",
    )


def test_scan_session_reuses_one_connection_for_existing_db_methods(tmp_path):
    db = CountingSQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    investigation_id = db.create_investigation("Case", [root])
    file_id = add_file(db, investigation_id, root / "note.txt", stable_id="stable")
    db.connection_opens = 0

    with db.scan_session(investigation_id) as session:
        assert session.load_manifest()[0].file_id == file_id
        assert db.get_file_by_path(investigation_id, str(root / "note.txt").casefold()) is not None
        assert db.active_chunk_count_for_file(file_id) == 0
        db.mark_file_extraction(file_id, "done", None, "extractor-v1")
        assert db.connection_opens == 1

    assert db.connection_opens == 1


def test_manifest_load_is_one_query_and_does_not_infer_missing_stage_state(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    investigation_id = db.create_investigation("Case", [root])
    for index in range(100):
        add_file(
            db,
            investigation_id,
            root / f"file-{index}.txt",
            stable_id=f"stable-{index}",
            content_hash=f"hash-{index}",
        )
    # Initialization never infers staged state from file/chunk rows.
    db.initialize()

    with db.scan_session(investigation_id) as session:
        statements: list[str] = []
        with db.connection() as connection:
            connection.set_trace_callback(statements.append)
        manifest = session.load_manifest()
        with db.connection() as connection:
            connection.set_trace_callback(None)

    selects = [statement for statement in statements if statement.lstrip().upper().startswith("SELECT")]
    assert len(manifest) == 100
    assert len(selects) == 1
    assert all(entry.pipeline_state_exists is False for entry in manifest)
    assert all(entry.extraction_signature == "" for entry in manifest)


def test_pipeline_state_supports_full_batch_and_partial_updates(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    investigation_id = db.create_investigation("Case", [root])
    first_id = add_file(db, investigation_id, root / "first.txt", stable_id="first")
    second_id = add_file(db, investigation_id, root / "second.txt", stable_id="second")

    with db.scan_session(investigation_id) as session:
        session.set_pipeline_states(
            [
                FilePipelineState(
                    file_id=first_id,
                    source_content_hash="first-hash",
                    extraction_signature="extract-v2",
                    chunking_signature="chunk-v2:size=1000",
                    metadata_signature="metadata-v2",
                    embedding_input_signature="embedding-input-v2",
                    embedding_model="model-a",
                    extraction_status="done",
                    chunk_status="done",
                    metadata_status="done",
                    vector_status="done",
                    active_vector_count=3,
                ),
                FilePipelineState(
                    file_id=second_id,
                    source_content_hash="second-hash",
                    extraction_status="done",
                ),
            ]
        )
        session.update_pipeline_state(
            second_id,
            chunking_signature="chunk-v2:size=1000",
            chunk_status="done",
        )
        by_id = {entry.file_id: entry for entry in session.load_manifest()}

        assert by_id[first_id].pipeline_state_exists is True
        assert by_id[first_id].embedding_model == "model-a"
        assert by_id[first_id].active_vector_count == 3
        assert by_id[second_id].pipeline_state_exists is True
        assert by_id[second_id].extraction_status == "done"
        assert by_id[second_id].chunking_signature == "chunk-v2:size=1000"
        assert by_id[second_id].chunk_status == "done"
        assert session.clear_pipeline_state(first_id) is True
        assert {entry.file_id: entry for entry in session.load_manifest()}[
            first_id
        ].pipeline_state_exists is False


def test_scan_batch_rolls_back_nested_existing_db_operations(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    investigation_id = db.create_investigation("Case", [root])
    normalized_path = str(root / "rolled-back.txt").casefold()

    with db.scan_session(investigation_id) as session:
        with pytest.raises(RuntimeError, match="abort batch"):
            with session.batch():
                file_id = add_file(
                    db,
                    investigation_id,
                    root / "rolled-back.txt",
                    stable_id="rolled-back",
                )
                session.update_pipeline_state(file_id, extraction_status="done")
                raise RuntimeError("abort batch")

        assert db.get_file_by_path(investigation_id, normalized_path) is None


def test_bulk_disappearance_only_changes_explicit_active_files(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    first_root = tmp_path / "first-case"
    second_root = tmp_path / "second-case"
    first_root.mkdir()
    second_root.mkdir()
    first_investigation = db.create_investigation("First", [first_root])
    second_investigation = db.create_investigation("Second", [second_root])
    missing_id = add_file(db, first_investigation, first_root / "missing.txt", stable_id="missing")
    present_id = add_file(db, first_investigation, first_root / "present.txt", stable_id="present")
    other_id = add_file(db, second_investigation, second_root / "other.txt", stable_id="other")
    db.insert_chunks(
        [
            {
                "id": "missing-chunk",
                "investigation_id": first_investigation,
                "file_id": missing_id,
                "content_hash": "content-hash",
                "chunk_index": 0,
                "chunk_text": "evidence",
                "chunk_text_hash": "text-hash",
                "source_type": "txt",
                "path": str(first_root / "missing.txt"),
                "extractor_version": "extractor-v1",
                "chunker_version": "chunker-v1",
                "embedding_model": "",
            }
        ]
    )

    with db.scan_session(first_investigation) as session:
        assert session.mark_disappeared([missing_id, other_id]) == 1
        assert session.mark_disappeared([missing_id]) == 0
        assert session.vector_eligible_chunk_ids() == {"missing-chunk"}

    missing = db.get_file_by_path(first_investigation, str(first_root / "missing.txt").casefold())
    present = db.get_file_by_path(first_investigation, str(first_root / "present.txt").casefold())
    other = db.get_file_by_path(second_investigation, str(second_root / "other.txt").casefold())
    pending = db.list_disappeared_files(first_investigation)
    assert missing["status"] == "disappeared"
    assert present["status"] == "active"
    assert other["status"] == "active"
    assert len(pending) == 1
    assert pending[0]["file_id"] == missing_id
    assert pending[0]["chunk_count"] == 1

    db.resolve_disappeared(int(pending[0]["id"]), "inactive")
    with db.scan_session(first_investigation) as session:
        assert session.vector_eligible_chunk_ids() == set()


def test_fresh_fts_index_matches_filename_terms(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    investigation_id = db.create_investigation("Case", [root])
    file_path = root / "smokinggun-report.txt"
    file_id = add_file(db, investigation_id, file_path, stable_id="stable")
    db.insert_chunks(
        [
            {
                "id": "path-search-chunk",
                "investigation_id": investigation_id,
                "file_id": file_id,
                "content_hash": "content-hash",
                "chunk_index": 0,
                "chunk_text": "The body deliberately omits the filename token.",
                "chunk_text_hash": "path-search-hash",
                "source_type": "txt",
                "path": str(file_path),
                "extractor_version": "extractor-v1",
                "chunker_version": "chunker-v1",
                "embedding_model": "",
            }
        ]
    )

    results = db.keyword_search(investigation_id, "smokinggun", limit=10)

    assert [row["id"] for row in results] == ["path-search-chunk"]


def test_fts_rows_share_stable_chunk_keys_and_replace_tokens_in_place(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    investigation_id = db.create_investigation("Case", [root])
    file_path = root / "original.txt"
    file_id = add_file(db, investigation_id, file_path, stable_id="stable")

    def record(text: str, path: Path) -> dict[str, object]:
        return {
            "id": "stable-fts-chunk",
            "investigation_id": investigation_id,
            "file_id": file_id,
            "content_hash": "content-hash",
            "chunk_index": 0,
            "chunk_text": text,
            "searchable_text": text,
            "chunk_text_hash": f"hash-{text}",
            "source_type": "txt",
            "path": str(path),
            "extractor_version": "extractor-v1",
            "chunker_version": "chunker-v1",
            "embedding_model": "",
        }

    db.insert_chunks([record("obsoletetoken", file_path)])
    with db.connection() as conn:
        first = conn.execute(
            """
            SELECT c.fts_rowid AS chunk_key, f.rowid AS fts_key
            FROM chunks c JOIN chunks_fts f ON f.rowid = c.fts_rowid
            WHERE c.id = 'stable-fts-chunk'
            """
        ).fetchone()

    db.insert_chunks([record("replacementtoken", file_path)])
    moved_path = root / "renamed.txt"
    db.update_chunk_paths_for_file(file_id, moved_path)

    with db.connection() as conn:
        after = conn.execute(
            """
            SELECT c.fts_rowid AS chunk_key, f.rowid AS fts_key
            FROM chunks c JOIN chunks_fts f ON f.rowid = c.fts_rowid
            WHERE c.id = 'stable-fts-chunk'
            """
        ).fetchall()
        fts_count = conn.execute("SELECT COUNT(*) AS count FROM chunks_fts").fetchone()

    assert int(first["chunk_key"]) == int(first["fts_key"])
    assert len(after) == 1
    assert int(after[0]["chunk_key"]) == int(first["chunk_key"])
    assert fts_count["count"] == 1
    assert db.keyword_search(investigation_id, "obsoletetoken", limit=10) == []
    assert [row["id"] for row in db.keyword_search(investigation_id, "replacementtoken", limit=10)] == [
        "stable-fts-chunk"
    ]
    assert [row["id"] for row in db.keyword_search(investigation_id, "renamed", limit=10)] == [
        "stable-fts-chunk"
    ]

    db.set_chunks_inactive_for_file(file_id)
    with db.connection() as conn:
        assert conn.execute("SELECT COUNT(*) AS count FROM chunks_fts").fetchone()["count"] == 0


def test_initialize_rejects_legacy_chunks_schema_without_backfill(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()

    with pytest.raises(RuntimeError, match="legacy ingest schema"):
        SQLiteDB(path).initialize()
