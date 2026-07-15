from pathlib import Path

from investigation_rag.config import AppConfig
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.ingest.pipeline import ScanPipeline, ScanSummary


def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        app_data_dir=tmp_path,
        sqlite_path=tmp_path / "app.sqlite3",
        lancedb_dir=tmp_path / "lancedb",
        ollama_base_url="http://127.0.0.1:11434",
        ollama_allow_remote=False,
        ollama_timeout_seconds=120,
        ollama_num_ctx=0,
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
    )


def test_scan_logs_total_files_progress_and_ingested_counts(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite3")
    db.initialize()
    root = tmp_path / "case"
    root.mkdir()
    (root / "first.txt").write_text("first note", encoding="utf-8")
    (root / "second.txt").write_text("second note", encoding="utf-8")
    (root / "ignored.jpg").write_text("not supported", encoding="utf-8")
    investigation_id = db.create_investigation("Case", [root])
    logs: list[str] = []

    summary = ScanPipeline(app_config(tmp_path), db, log=logs.append).scan_investigation(investigation_id)

    assert summary.total_files == 2
    assert summary.scanned == 2
    assert summary.ingested == 2
    assert any("Discovered 2 supported file(s)" in message for message in logs)
    assert any("Processing 1/2" in message for message in logs)
    assert any("Progress: files=2/2" in message and "ingested=2" in message for message in logs)
    assert any("Scan complete. scanned=2/2" in message and "ingested=2" in message for message in logs)


def test_progress_text_reports_eta_and_rate():
    summary = ScanSummary(total_files=4, total_bytes=4096, scanned=2, scanned_bytes=2048, extracted=1)

    message = ScanPipeline._progress_text(summary, elapsed_seconds=10)

    assert "files=2/4 (50.0%)" in message
    assert "bytes=2.0 KB/4.0 KB" in message
    assert "ingested=1" in message
    assert "eta~10s" in message
    assert "rate=12.0 files/min" in message
