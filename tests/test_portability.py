from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from investigation_rag.ingest.hashing import sha256_text
from investigation_rag.ingest.searchable_text import stable_embedding_text
from investigation_rag.portability import export_workspace, import_workspace
from investigation_rag.workspace import WorkspaceCatalog


def test_export_import_restores_vectors_without_source_files(tmp_path: Path):
    source_catalog = WorkspaceCatalog(tmp_path / "source-data")
    source_catalog.initialize()
    missing_root = tmp_path / "source-files-are-not-present"
    workspace = source_catalog.create("Portable case", [missing_root])

    missing_file = missing_root / "evidence.txt"
    file_id = workspace.db.upsert_file(
        investigation_id=workspace.investigation_id,
        path=missing_file,
        normalized_path=str(missing_file).casefold(),
        stable_id="portable-file",
        size=123,
        modified_ns=456,
        content_hash="content-hash",
        source_type="txt",
    )
    text = "The source file is gone, but this extracted evidence remains."
    metadata = {"filename": missing_file.name, "path": str(missing_file)}
    workspace.db.insert_chunks(
        [
            {
                "id": "portable-chunk",
                "investigation_id": workspace.investigation_id,
                "file_id": file_id,
                "content_hash": "content-hash",
                "chunk_index": 0,
                "chunk_text": text,
                "chunk_text_hash": sha256_text(text),
                "source_type": "txt",
                "metadata": metadata,
                "path": str(missing_file),
                "extractor_version": "test-extractor",
                "chunker_version": "test-chunker",
                "embedding_model": "portable-model",
            }
        ]
    )
    cache_key = sha256_text(stable_embedding_text(text, metadata))
    workspace.db.set_embedding(cache_key, "portable-model", [0.1, 0.2, 0.3])

    archive_path = tmp_path / "portable-case.bosuil"
    export_workspace(workspace.entry, archive_path)

    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {
            "export.json",
            "manifest.json",
            "investigation.sqlite3",
        }

    target_catalog = WorkspaceCatalog(tmp_path / "target-data")
    target_catalog.initialize()
    imported = import_workspace(
        target_catalog,
        archive_path,
        preferred_embedding_model="portable-model",
    )

    assert imported.vector_warning == ""
    assert imported.vector_rebuild is not None
    assert imported.vector_rebuild.restored_vectors == 1
    assert not missing_file.exists()
    assert imported.workspace.db.get_chunk("portable-chunk")["chunk_text"] == text
    vectors = imported.workspace.store.get_vectors_for_chunks(
        ["portable-chunk"],
        investigation_id=imported.workspace.investigation_id,
        embedding_model="portable-model",
    )
    assert vectors["portable-chunk"] == pytest.approx([0.1, 0.2, 0.3])
    assert not (imported.workspace.entry.workspace_path / "lancedb" / "__manifest").exists()
