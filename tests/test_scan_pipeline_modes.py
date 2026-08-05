from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import replace
from pathlib import Path

import investigation_rag.ingest.pipeline as pipeline_module
from investigation_rag.config import AppConfig
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.ingest.pipeline import ScanPipeline
from investigation_rag.ingest.scan_mode import ScanMode
from investigation_rag.ingest.scanner import InventoryError, scan_supported_files


class CountingSQLiteDB(SQLiteDB):
    def __init__(self, path: Path):
        super().__init__(path)
        self.connection_opens = 0

    def connect(self) -> sqlite3.Connection:
        self.connection_opens += 1
        return super().connect()


class RecordingVectorStore:
    def __init__(self):
        self.path_updates: list[tuple[int, str, int | None]] = []
        self.activity_updates: list[tuple[tuple[int, ...], bool, int | None]] = []
        self.file_deletes: list[tuple[tuple[int, ...], int | None]] = []
        self.reconciliations: list[tuple[int, set[str]]] = []

    def update_file_path(self, file_id, path, *, investigation_id=None):
        self.path_updates.append((int(file_id), str(path), investigation_id))
        return 1

    def set_files_active(self, file_ids, active, *, investigation_id=None):
        normalized = tuple(int(file_id) for file_id in file_ids)
        self.activity_updates.append((normalized, bool(active), investigation_id))
        return len(normalized)

    def delete_vectors_for_files(self, file_ids, *, investigation_id=None):
        self.file_deletes.append(
            (tuple(int(file_id) for file_id in file_ids), investigation_id)
        )

    def reconcile_active_chunks(self, investigation_id, active_chunk_ids):
        self.reconciliations.append((int(investigation_id), set(active_chunk_ids)))
        return 0


def app_config(tmp_path: Path, *, verbose_file_limit: int = 100) -> AppConfig:
    return AppConfig(
        app_data_dir=tmp_path,
        sqlite_path=tmp_path / "app.sqlite3",
        lancedb_dir=tmp_path / "lancedb",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_allow_remote=False,
        ollama_timeout_seconds=120,
        ollama_num_ctx=0,
        ollama_embed_num_ctx=0,
        ollama_llm_model="",
        ollama_embed_model="",
        ollama_embed_query_instruction="",
        chunk_target_chars=2800,
        chunk_max_chars=4800,
        chunk_overlap_chars=240,
        rag_keyword_candidates=64,
        rag_vector_candidates=64,
        rag_fusion_candidates=80,
        rag_rerank_candidates=80,
        rag_final_answer_chunks=8,
        rag_deep_keyword_candidates=150,
        rag_deep_vector_candidates=150,
        rag_deep_fusion_candidates=200,
        rag_deep_rerank_candidates=200,
        rag_deep_final_answer_chunks=12,
        gui_lazy_load_file_chunk_text=True,
        scan_discovery_workers=4,
        scan_hash_workers=2,
        scan_verbose_file_limit=verbose_file_limit,
    )


def create_case(db: SQLiteDB, tmp_path: Path, file_count: int = 1) -> tuple[Path, int]:
    root = tmp_path / "case"
    root.mkdir()
    for index in range(file_count):
        (root / f"note-{index:03d}.txt").write_text(
            f"evidence {index}", encoding="utf-8"
        )
    return root, db.create_investigation("Case", [root])


def test_unchanged_fast_rescan_uses_one_connection_and_no_hash_or_extraction(
    tmp_path, monkeypatch
):
    db = CountingSQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    _, investigation_id = create_case(db, tmp_path, file_count=40)
    config = app_config(tmp_path, verbose_file_limit=10)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    db.connection_opens = 0
    logs: list[str] = []

    monkeypatch.setattr(
        pipeline_module,
        "sha256_file",
        lambda path: (_ for _ in ()).throw(AssertionError(f"hashed {path}")),
    )
    monkeypatch.setattr(
        ScanPipeline,
        "_extract_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extracted")),
    )

    summary = ScanPipeline(config, db, log=logs.append).scan_investigation(
        investigation_id, mode=ScanMode.FAST
    )

    assert summary.total_files == 40
    assert summary.unchanged == 40
    assert summary.hashed == 0
    assert summary.extracted == 0
    assert summary.failed == 0
    assert db.connection_opens == 1
    assert not any(message.startswith("Processing ") for message in logs)


def test_verify_hashes_every_file_without_rebuilding_current_stages(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    _, investigation_id = create_case(db, tmp_path, file_count=3)
    config = app_config(tmp_path)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    real_sha256_file = pipeline_module.sha256_file
    hashed_paths: list[Path] = []

    def observed_hash(path: Path) -> str:
        hashed_paths.append(path)
        return real_sha256_file(path)

    monkeypatch.setattr(pipeline_module, "sha256_file", observed_hash)
    monkeypatch.setattr(
        ScanPipeline,
        "_extract_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extracted")),
    )

    summary = ScanPipeline(config, db).scan_investigation(
        investigation_id, mode=ScanMode.VERIFY
    )

    assert summary.hashed == 3
    assert len(hashed_paths) == 3
    assert summary.unchanged == 3
    assert summary.extracted == 0
    assert summary.failed == 0


def test_verify_detects_same_size_content_change_with_preserved_mtime(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    path = root / "note.txt"
    path.write_text("AAAA", encoding="utf-8")
    investigation_id = db.create_investigation("Case", [root])
    config = app_config(tmp_path)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    original_mtime = path.stat().st_mtime_ns
    path.write_text("BBBB", encoding="utf-8")
    os.utime(path, ns=(original_mtime, original_mtime))

    fast = ScanPipeline(config, db).scan_investigation(investigation_id, mode=ScanMode.FAST)
    verified = ScanPipeline(config, db).scan_investigation(
        investigation_id, mode=ScanMode.VERIFY
    )

    assert fast.unchanged == 1
    assert fast.hashed == 0
    assert verified.hashed == 1
    assert verified.extracted == 1
    assert db.active_chunks_for_investigation(investigation_id)[0]["chunk_text"] == "BBBB"


def test_rebuild_verifies_content_and_recreates_stages_for_unchanged_files(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    _, investigation_id = create_case(db, tmp_path, file_count=2)
    config = app_config(tmp_path)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    real_extract = ScanPipeline._extract_document
    extracted_paths: list[Path] = []

    def observed_extract(self, path, content_hash, source_type, use_cache=True):
        extracted_paths.append(path)
        return real_extract(self, path, content_hash, source_type, use_cache=use_cache)

    monkeypatch.setattr(ScanPipeline, "_extract_document", observed_extract)

    summary = ScanPipeline(config, db).scan_investigation(
        investigation_id, mode=ScanMode.REBUILD
    )

    assert summary.hashed == 2
    assert summary.extracted == 2
    assert len(extracted_paths) == 2
    assert summary.failed == 0


def test_identical_files_remain_distinct_occurrences(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    (root / "first.txt").write_text("same evidence", encoding="utf-8")
    (root / "second.txt").write_text("same evidence", encoding="utf-8")
    investigation_id = db.create_investigation("Case", [root])

    summary = ScanPipeline(app_config(tmp_path), db).scan_investigation(investigation_id)
    files = db.active_files_for_investigation(investigation_id)

    assert summary.extracted == 2
    assert len(files) == 2
    assert len({int(row["id"]) for row in files}) == 2
    assert len({str(row["content_hash"]) for row in files}) == 1
    assert sum(int(row["active_chunk_count"]) for row in files) == 2


def test_rename_is_hash_confirmed_without_reextracting_or_reembedding(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    config = app_config(tmp_path)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    before = db.active_files_for_investigation(investigation_id)[0]
    old_path = root / "note-000.txt"
    new_path = root / "renamed.txt"
    old_path.rename(new_path)

    monkeypatch.setattr(
        ScanPipeline,
        "_extract_document",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("extracted")),
    )

    summary = ScanPipeline(config, db).scan_investigation(investigation_id)
    after = db.active_files_for_investigation(investigation_id)[0]
    chunks = db.active_chunks_for_file(int(after["id"]))

    assert summary.hashed == 1
    assert summary.metadata_only == 1
    assert summary.extracted == 0
    assert int(after["id"]) == int(before["id"])
    assert after["path"] == str(new_path)
    assert {row["path"] for row in chunks} == {str(new_path)}
    metadata = json.loads(chunks[0]["metadata_json"])
    assert metadata["filename"] == "renamed.txt"
    assert metadata["path"] == str(new_path)
    assert [row["id"] for row in db.keyword_search(investigation_id, "renamed", limit=10)] == [
        chunks[0]["id"]
    ]


def test_incomplete_inventory_never_marks_absent_files_disappeared(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path, file_count=2)
    config = app_config(tmp_path)
    store = RecordingVectorStore()
    ScanPipeline(config, db, store).scan_investigation(investigation_id)
    missing_path = root / "note-001.txt"
    missing_path.unlink()
    real_inventory = scan_supported_files([root])
    error = InventoryError(
        root=root,
        path=root / "unreadable-subtree",
        operation="scandir",
        message="simulated provider error",
    )
    incomplete_inventory = replace(
        real_inventory,
        root_results=tuple(
            replace(result, complete=False, errors=(error,))
            for result in real_inventory.root_results
        ),
        errors=(error,),
    )
    monkeypatch.setattr(pipeline_module, "scan_supported_files", lambda *args, **kwargs: incomplete_inventory)

    summary = ScanPipeline(config, db, store).scan_investigation(investigation_id)
    files = db.active_files_for_investigation(investigation_id)

    assert summary.inventory_errors == 1
    assert summary.disappeared == 0
    assert len(files) == 2
    assert db.list_disappeared_files(investigation_id) == []
    assert store.activity_updates == []


def test_fast_rescan_bulk_updates_last_seen_without_per_file_upserts(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    _, investigation_id = create_case(db, tmp_path, file_count=3)
    config = app_config(tmp_path)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    with db.transaction() as conn:
        conn.execute(
            "UPDATE files SET last_seen_at = '2000-01-01T00:00:00+00:00' WHERE investigation_id = ?",
            (investigation_id,),
        )
    monkeypatch.setattr(
        db,
        "upsert_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("per-file upsert")),
    )

    summary = ScanPipeline(config, db).scan_investigation(investigation_id)
    files = db.active_files_for_investigation(investigation_id)

    assert summary.unchanged == 3
    assert {row["last_seen_at"] for row in files} != {"2000-01-01T00:00:00+00:00"}


def test_kept_file_that_is_still_missing_returns_to_pending_review(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    config = app_config(tmp_path)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    (root / "note-000.txt").unlink()
    ScanPipeline(config, db).scan_investigation(investigation_id)
    first_pending = db.list_disappeared_files(investigation_id)
    db.resolve_disappeared(int(first_pending[0]["id"]), "keep")

    ScanPipeline(config, db).scan_investigation(investigation_id)
    pending_again = db.list_disappeared_files(investigation_id)

    assert len(pending_again) == 1
    assert int(pending_again[0]["file_id"]) == int(first_pending[0]["file_id"])


def test_inactive_occurrence_rebuilds_if_the_same_path_reappears(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    config = app_config(tmp_path)
    path = root / "note-000.txt"
    ScanPipeline(config, db).scan_investigation(investigation_id)
    path.unlink()
    ScanPipeline(config, db).scan_investigation(investigation_id)
    pending = db.list_disappeared_files(investigation_id)
    file_id = int(pending[0]["file_id"])
    db.resolve_disappeared(int(pending[0]["id"]), "inactive")
    path.write_text("evidence 0", encoding="utf-8")

    summary = ScanPipeline(config, db).scan_investigation(investigation_id)

    assert summary.extracted == 1
    assert db.active_chunk_count_for_file(file_id) == 1
    assert [row["id"] for row in db.keyword_search(investigation_id, "evidence", limit=10)]


def test_pending_disappeared_file_is_hidden_from_keyword_results(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    config = app_config(tmp_path)
    ScanPipeline(config, db).scan_investigation(investigation_id)
    assert db.keyword_search(investigation_id, "evidence", limit=10)
    (root / "note-000.txt").unlink()

    ScanPipeline(config, db).scan_investigation(investigation_id)

    assert db.keyword_search(investigation_id, "evidence", limit=10) == []


def test_new_copy_never_reuses_a_finalized_inactive_occurrence(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    config = app_config(tmp_path)
    original = root / "note-000.txt"
    ScanPipeline(config, db).scan_investigation(investigation_id)
    original_id = int(db.active_files_for_investigation(investigation_id)[0]["id"])
    original.unlink()
    ScanPipeline(config, db).scan_investigation(investigation_id)
    pending = db.list_disappeared_files(investigation_id)[0]
    db.resolve_disappeared(int(pending["id"]), "inactive")
    (root / "new-copy.txt").write_text("evidence 0", encoding="utf-8")

    ScanPipeline(config, db).scan_investigation(investigation_id)
    active = db.active_files_for_investigation(investigation_id)

    assert len(active) == 1
    assert int(active[0]["id"]) != original_id


def test_multiple_path_misses_with_one_stable_id_never_collapse(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    config = app_config(tmp_path)
    original = root / "note-000.txt"
    ScanPipeline(config, db).scan_investigation(investigation_id)
    first_link = root / "first-link.txt"
    second_link = root / "second-link.txt"
    original.rename(first_link)
    os.link(first_link, second_link)

    summary = ScanPipeline(config, db).scan_investigation(investigation_id)
    active = db.active_files_for_investigation(investigation_id)

    assert summary.extracted == 2
    assert len(active) == 2
    assert len({int(row["id"]) for row in active}) == 2
    assert {Path(row["path"]).name for row in active} == {"first-link.txt", "second-link.txt"}


def test_file_mutation_during_hash_is_rejected_and_not_ingested(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    path = root / "note-000.txt"
    real_hash = pipeline_module.sha256_file

    def hash_then_mutate(candidate: Path) -> str:
        content_hash = real_hash(candidate)
        candidate.write_text("changed while hashing", encoding="utf-8")
        return content_hash

    monkeypatch.setattr(pipeline_module, "sha256_file", hash_then_mutate)

    summary = ScanPipeline(app_config(tmp_path), db).scan_investigation(investigation_id)

    assert summary.failed == 1
    assert summary.extracted == 0
    assert db.active_files_for_investigation(investigation_id) == []


def test_file_mutation_during_extraction_is_rejected_before_chunk_writes(tmp_path, monkeypatch):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    real_extract = ScanPipeline._extract_document

    def extract_then_mutate(self, path, content_hash, source_type, use_cache=True):
        document = real_extract(self, path, content_hash, source_type, use_cache=use_cache)
        path.write_text("changed while extracting", encoding="utf-8")
        return document

    monkeypatch.setattr(ScanPipeline, "_extract_document", extract_then_mutate)

    summary = ScanPipeline(app_config(tmp_path), db).scan_investigation(investigation_id)
    files = db.active_files_for_investigation(investigation_id)

    assert summary.failed == 1
    assert len(files) == 1
    assert files[0]["extraction_status"] == "failed"
    assert db.active_chunk_count_for_file(int(files[0]["id"])) == 0


def test_vector_lifecycle_tracks_moves_disappearance_reappearance_and_rebuild(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root, investigation_id = create_case(db, tmp_path)
    config = app_config(tmp_path)
    store = RecordingVectorStore()
    original = root / "note-000.txt"
    moved = root / "moved.txt"
    ScanPipeline(config, db, store).scan_investigation(investigation_id)
    file_id = int(db.active_files_for_investigation(investigation_id)[0]["id"])
    original.rename(moved)

    ScanPipeline(config, db, store).scan_investigation(investigation_id)
    assert store.path_updates[-1] == (file_id, str(moved), investigation_id)

    moved.unlink()
    ScanPipeline(config, db, store).scan_investigation(investigation_id)
    assert store.activity_updates[-1] == ((file_id,), False, investigation_id)

    moved.write_text("evidence 0", encoding="utf-8")
    ScanPipeline(config, db, store).scan_investigation(investigation_id)
    assert store.activity_updates[-1] == ((file_id,), True, investigation_id)

    ScanPipeline(config, db, store).scan_investigation(
        investigation_id, mode=ScanMode.REBUILD
    )
    eligible = {
        str(row["id"]) for row in db.active_chunks_for_investigation(investigation_id)
    }
    assert store.file_deletes[-1] == ((file_id,), investigation_id)
    assert store.reconciliations[-1] == (investigation_id, eligible)
