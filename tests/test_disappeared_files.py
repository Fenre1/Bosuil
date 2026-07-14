from pathlib import Path

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.ingest.disappeared import queue_disappeared_files
from investigation_rag.utils.paths import normalize_path


def test_disappeared_files_are_queued_and_can_be_inactivated(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    file_path = root / "note.txt"
    file_path.write_text("evidence", encoding="utf-8")
    investigation_id = db.create_investigation("Case", [root])
    file_id = db.upsert_file(
        investigation_id=investigation_id,
        path=file_path,
        normalized_path=normalize_path(file_path),
        stable_id="stable",
        size=file_path.stat().st_size,
        modified_ns=file_path.stat().st_mtime_ns,
        content_hash="hash",
        source_type="txt",
    )
    db.insert_chunks(
        [
            {
                "id": "chunk-1",
                "investigation_id": investigation_id,
                "file_id": file_id,
                "content_hash": "hash",
                "chunk_index": 0,
                "chunk_text": "evidence",
                "chunk_text_hash": "text-hash",
                "source_type": "txt",
                "path": str(file_path),
                "extractor_version": "extractors-v1",
                "chunker_version": "deterministic-v1",
                "embedding_model": "",
            }
        ]
    )

    count = queue_disappeared_files(db, investigation_id, seen_file_ids=set())

    assert count == 1
    pending = db.list_disappeared_files(investigation_id)
    assert len(pending) == 1
    assert pending[0]["chunk_count"] == 1

    db.resolve_disappeared(int(pending[0]["id"]), "inactive")

    inactive_chunk = db.get_chunk("chunk-1")
    assert inactive_chunk["is_active"] == 0
