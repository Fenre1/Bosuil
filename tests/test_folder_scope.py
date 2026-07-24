from pathlib import Path

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.retrieval.keyword_search import KeywordSearcher


def test_keyword_search_can_be_limited_to_selected_folder(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite")
    db.initialize()
    root = tmp_path / "case"
    investigation_id = db.create_investigation("case", [root])
    alpha_file = root / "alpha" / "report.txt"
    beta_file = root / "beta" / "report.txt"
    alpha_id = insert_file(db, investigation_id, alpha_file)
    beta_id = insert_file(db, investigation_id, beta_file)
    db.insert_chunks(
        [
            chunk(investigation_id, alpha_id, alpha_file, "alpha", "payment from alpha"),
            chunk(investigation_id, beta_id, beta_file, "beta", "payment from beta"),
        ]
    )

    searcher = KeywordSearcher(db)

    scoped = searcher.search(investigation_id, "payment", folder_paths=[str(root / "alpha")])
    unscoped = searcher.search(investigation_id, "payment")
    none_selected = searcher.search(investigation_id, "payment", folder_paths=[])

    assert [result.path for result in scoped] == [str(alpha_file)]
    assert {result.path for result in unscoped} == {str(alpha_file), str(beta_file)}
    assert none_selected == []


def test_keyword_search_can_be_limited_to_selected_file_types(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite")
    db.initialize()
    root = tmp_path / "case"
    investigation_id = db.create_investigation("case", [root])
    pdf_file = root / "report.pdf"
    txt_file = root / "notes.txt"
    pdf_id = insert_file(db, investigation_id, pdf_file, source_type="pdf")
    txt_id = insert_file(db, investigation_id, txt_file, source_type="txt")
    db.insert_chunks(
        [
            chunk(investigation_id, pdf_id, pdf_file, "pdf", "payment from pdf", source_type="pdf"),
            chunk(investigation_id, txt_id, txt_file, "txt", "payment from txt", source_type="txt"),
        ]
    )

    searcher = KeywordSearcher(db)

    pdf_only = searcher.search(investigation_id, "payment", source_types=["pdf"])
    mixed = searcher.search(investigation_id, "payment", source_types=[".pdf", "txt"])
    none_selected = searcher.search(investigation_id, "payment", source_types=[])

    assert [result.path for result in pdf_only] == [str(pdf_file)]
    assert {result.path for result in mixed} == {str(pdf_file), str(txt_file)}
    assert none_selected == []


def test_active_chunks_for_investigation_returns_active_file_chunks(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite")
    db.initialize()
    root = tmp_path / "case"
    investigation_id = db.create_investigation("case", [root])
    active_file = root / "active" / "report.txt"
    inactive_file = root / "inactive" / "report.txt"
    active_id = insert_file(db, investigation_id, active_file)
    inactive_id = insert_file(db, investigation_id, inactive_file)
    db.insert_chunks(
        [
            chunk(investigation_id, active_id, active_file, "active-1", "first active chunk"),
            chunk(investigation_id, active_id, active_file, "active-2", "second active chunk", chunk_index=1),
            chunk(investigation_id, inactive_id, inactive_file, "inactive", "inactive file chunk"),
        ]
    )
    db.add_disappeared_file(investigation_id, inactive_id, str(inactive_file), "2026-01-01T00:00:00+00:00")

    rows = db.active_chunks_for_investigation(investigation_id)

    assert [row["id"] for row in rows] == ["chunk-active-1", "chunk-active-2"]


def test_active_chunk_summaries_for_investigation_omit_chunk_text(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite")
    db.initialize()
    root = tmp_path / "case"
    investigation_id = db.create_investigation("case", [root])
    file_id = insert_file(db, investigation_id, root / "report.txt")
    db.insert_chunks([chunk(investigation_id, file_id, root / "report.txt", "summary", "body text")])

    rows = db.active_chunk_summaries_for_investigation(investigation_id)

    assert len(rows) == 1
    assert rows[0]["id"] == "chunk-summary"
    assert "chunk_text" not in rows[0].keys()


def test_file_browser_projection_stores_chunk_counts_and_loads_summaries_per_file(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite")
    db.initialize()
    root = tmp_path / "case"
    investigation_id = db.create_investigation("case", [root])
    report = root / "report.txt"
    file_id = insert_file(db, investigation_id, report)
    db.insert_chunks(
        [
            chunk(investigation_id, file_id, report, "one", "first"),
            chunk(investigation_id, file_id, report, "two", "second", chunk_index=1),
        ]
    )

    files = db.file_browser_files_for_investigation(investigation_id)
    summaries = db.active_chunk_summaries_for_file(file_id)

    assert files[0]["active_chunk_count"] == 2
    assert [row["id"] for row in summaries] == ["chunk-one", "chunk-two"]
    assert "chunk_text" not in summaries[0].keys()

    db.set_chunks_inactive_for_file(file_id)
    assert db.file_browser_files_for_investigation(investigation_id)[0]["active_chunk_count"] == 0


def test_keyword_search_returns_display_markdown(tmp_path):
    db = SQLiteDB(tmp_path / "app.sqlite")
    db.initialize()
    root = tmp_path / "case"
    investigation_id = db.create_investigation("case", [root])
    report = root / "report.docx"
    file_id = insert_file(db, investigation_id, report)
    db.insert_chunks(
        [
            chunk(
                investigation_id,
                file_id,
                report,
                "docx",
                "Important heading",
                display_markdown="# Important heading",
            )
        ]
    )

    results = KeywordSearcher(db).search(investigation_id, "Important")

    assert len(results) == 1
    assert results[0].chunk_text == "Important heading"
    assert results[0].display_markdown == "# Important heading"


def insert_file(db: SQLiteDB, investigation_id: int, path: Path, source_type: str = "txt") -> int:
    return db.upsert_file(
        investigation_id=investigation_id,
        path=path,
        normalized_path=str(path.resolve()).casefold(),
        stable_id=None,
        size=10,
        modified_ns=1,
        content_hash=f"hash-{path.parent.name}",
        source_type=source_type,
    )


def chunk(
    investigation_id: int,
    file_id: int,
    path: Path,
    suffix: str,
    text: str,
    chunk_index: int = 0,
    display_markdown: str | None = None,
    source_type: str = "txt",
) -> dict[str, object]:
    return {
        "id": f"chunk-{suffix}",
        "investigation_id": investigation_id,
        "file_id": file_id,
        "content_hash": f"hash-{suffix}",
        "chunk_index": chunk_index,
        "chunk_text": text,
        "display_markdown": display_markdown,
        "chunk_text_hash": f"text-hash-{suffix}",
        "source_type": source_type,
        "page_number": None,
        "email_subject": None,
        "email_sender": None,
        "email_recipients": None,
        "email_date": None,
        "attachment_name": None,
        "path": str(path),
        "is_active": True,
        "extractor_version": "test",
        "chunker_version": "test",
        "embedding_model": "",
    }
