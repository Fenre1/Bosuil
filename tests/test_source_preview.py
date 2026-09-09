from hashlib import sha256
from pathlib import Path
import os

import pytest

from investigation_rag.chunking.deterministic import DeterministicChunker
from investigation_rag.extractors.pdf_extractor import PdfExtractor
from investigation_rag.gui.source_document import load_source


def indexed_pdf(path):
    chunks = DeterministicChunker().chunk_document(PdfExtractor().extract(path))
    for chunk in chunks:
        chunk.metadata["source_content_hash"] = sha256(path.read_bytes()).hexdigest()
    return chunks


@pytest.fixture(scope="module")
def qt_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_pdf_render_highlights_follow_rotation_and_crop(tmp_path, rotation):
    import fitz
    path = tmp_path / "rotated.pdf"
    with fitz.open() as document:
        page = document.new_page(width=600, height=800)
        page.insert_text((100, 150), "Selected evidence passage.")
        page.set_cropbox(fitz.Rect(30, 40, 560, 750))
        page.set_rotation(rotation)
        document.save(path)
    chunk = indexed_pdf(path)[0]
    rendered = load_source(str(path), chunk.metadata)
    assert rendered["kind"] == "pdf"
    assert rendered["rectangles"]
    for x0, y0, x1, y1 in rendered["rectangles"]:
        assert 0 <= x0 < x1 <= rendered["width"] + 1
        assert 0 <= y0 < y1 <= rendered["height"] + 1
    from PyQt5.QtGui import QImage
    image = QImage.fromData(rendered["png"])
    # Highlight boxes must cover actual ink, not just fall inside the page.
    x0, y0, x1, y1 = rendered["rectangles"][0]
    assert any(image.pixelColor(x, y).lightness() < 150
               for x in range(int(x0), int(x1)) for y in range(int(y0), int(y1)))


def test_changed_or_missing_source_never_renders_stale_highlights(tmp_path):
    import fitz
    path = tmp_path / "source.pdf"
    with fitz.open() as document:
        document.new_page().insert_text((60, 90), "Original evidence.")
        document.save(path)
    chunk = indexed_pdf(path)[0]
    assert load_source(str(path), chunk.metadata)["kind"] == "pdf"
    path.write_bytes(b"changed source")
    result = load_source(str(path), chunk.metadata)
    assert result["kind"] == "error" and "changed" in result["status"]
    path.unlink()
    assert "unavailable" in load_source(str(path), chunk.metadata)["status"]


def test_text_source_preserves_indent_and_highlights_exact_range(tmp_path):
    from investigation_rag.extractors.txt_extractor import TxtExtractor
    path = tmp_path / "source.txt"
    path.write_text("    First line.\n\tSecond line.\nThird line.", encoding="utf-8")
    chunk = DeterministicChunker(target_chars=20, max_chars=40, overlap_chars=0).chunk_document(TxtExtractor().extract(path))[1]
    chunk.metadata["source_content_hash"] = sha256(path.read_bytes()).hexdigest()
    result = load_source(str(path), chunk.metadata)
    assert result["kind"] == "text"
    assert "First line." in result["html"] and "Third line." in result["html"]
    assert 'background-color:#fde68a' in result["html"]
    assert 'name="selection"' in result["html"]


def test_preview_modes_switch_without_losing_formatting(qt_app):
    from investigation_rag.gui.chunk_preview import ChunkPreview
    preview = ChunkPreview()
    preview.set_chunk("Plain first")  # Legacy chunks have no source rendering.
    preview.set_chunk("Styled passage", metadata={"display_html": "<p><b>Styled</b> passage</p>", "embedding_context": "Section: Findings"})
    preview.formatted_button.click()
    assert preview.stack.currentWidget() is preview.viewer
    assert preview.toPlainText() == "Styled passage"
    assert "font-weight:600" in preview.viewer.toHtml()
    preview.plain_button.click()
    assert preview.toPlainText() == "Section: Findings\n\nStyled passage"
    preview.source_button.click()
    assert preview.stack.currentWidget() is preview.source_view
    assert "rescan" in preview.source_view.status.text()
    preview.clear()
    assert not preview.earlier_button.isEnabled()
    assert not preview.source_view.path
    preview.close()


def test_clear_invalidates_pending_source_result(qt_app):
    from concurrent.futures import Future
    from investigation_rag.gui.source_view import SourceView
    view = SourceView()
    future = Future()
    view.future = future
    view.request_generation = view.generation
    view.clear()
    future.set_result({"kind": "text", "html": "Old document", "status": "Old"})
    view.poll()
    assert view.status.text() == ""
    assert view.text.toPlainText() == ""
    view.close()


def test_workspace_switch_routes_all_preview_neighbors_to_active_database(qt_app, tmp_path):
    from investigation_rag.config import AppConfig
    from investigation_rag.gui.main_window import MainWindow
    from investigation_rag.workspace import WorkspaceCatalog

    catalog = WorkspaceCatalog(tmp_path / "data")
    catalog.initialize()
    workspaces = [catalog.create(name, []) for name in ("Alpha", "Beta")]
    for workspace in workspaces:
        with workspace.db.transaction() as conn:
            conn.execute(
                """INSERT INTO files (id, investigation_id, path, normalized_path,
                   first_seen_at, last_seen_at, updated_at)
                   VALUES (1, 1, 'evidence.txt', 'evidence.txt', '', '', '')"""
            )
            conn.execute(
                """INSERT INTO chunks (id, investigation_id, file_id, content_hash,
                   chunk_index, chunk_text, chunk_text_hash, source_type, path,
                   extractor_version, chunker_version, created_at, updated_at)
                   VALUES ('neighbor', 1, 1, 'hash', 1, ?, 'hash', 'txt',
                           'evidence.txt', 'test', 'test', '', '')""",
                (workspace.entry.name,),
            )

    config = AppConfig.from_env().for_workspace(tmp_path / "inactive")
    window = MainWindow(config, catalog)
    previews = (window.files_panel.preview, window.search_panel.preview,
                window.rag_panel.chunk_preview)
    try:
        # The startup catalog deliberately has no chunks table. Each workspace
        # reuses file/chunk IDs, so a stale connection could show another case.
        for workspace in workspaces:
            window._apply_workspace_dependencies(workspace)
            for preview in previews:
                preview.set_mode("embedding")
                preview.set_chunk("Selected evidence", chunk_index=0,
                                  metadata={"file_id": 1})
                assert not preview.earlier_button.isEnabled()
                assert preview.later_button.isEnabled()
                preview.later_button.click()
                assert preview.toPlainText() == workspace.entry.name
    finally:
        window.close()


def test_source_page_navigation_can_leave_the_selected_passage(qt_app, tmp_path):
    import fitz
    from investigation_rag.gui.source_view import SourceView
    path = tmp_path / "two-pages.pdf"
    with fitz.open() as document:
        document.new_page().insert_text((60, 90), "Selected passage.")
        document.new_page().insert_text((60, 90), "Surrounding source context.")
        document.save(path)
    chunk = indexed_pdf(path)[0]
    metadata = dict(chunk.metadata)
    metadata["source_spans"] = [s for s in metadata["source_spans"] if s["page_number"] == 1]
    metadata["page_start"] = 1
    view = SourceView()
    view.metadata = metadata
    requested = []
    view.page.valueChanged.connect(requested.append)
    rendered = load_source(str(path), metadata, 2)
    assert not rendered["rectangles"]
    view.show_result(rendered)
    assert view.page.value() == 2
    assert not requested
    assert view.future is None
    view.close()


def test_attachment_source_is_reopened_through_its_parent_message(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import sys
    from investigation_rag.gui.source_document import source_bytes
    path = tmp_path / "outer.msg"
    path.write_bytes(b"outer message")
    closed = []

    class Message:
        def __init__(self, data):
            self.data = data
            content = (SimpleNamespace(exportBytes=lambda: b"nested message")
                       if data == b"outer message" else b"attachment evidence")
            name = "inner.msg" if data == b"outer message" else "evidence.txt"
            self.attachments = [SimpleNamespace(longFilename=name, data=content)]

        def close(self):
            closed.append(self.data)

    monkeypatch.setitem(sys.modules, "extract_msg", SimpleNamespace(Message=Message))
    chain = [{"index": 0, "name": "inner.msg"}, {"index": 0, "name": "evidence.txt"}]
    with source_bytes(str(path), sha256(path.read_bytes()).hexdigest(), chain) as (data, suffix):
        assert data == b"attachment evidence"
        assert suffix == ".txt"
    assert closed == [b"outer message", b"nested message"]
