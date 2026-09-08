from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication

from investigation_rag.fact_ledger import FactLedgerFailure, FactLedgerItem
from investigation_rag.gui.fact_ledger_panel import FactLedgerPanel


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def document_row(file_id: int, path: str) -> dict[str, object]:
    return {
        "id": file_id,
        "path": path,
        "normalized_path": path.casefold(),
        "source_type": "pdf",
        "active_chunk_count": 7,
    }


def test_document_tree_selects_files_but_not_folders_and_filters_by_path(app) -> None:
    panel = FactLedgerPanel(SimpleNamespace(ollama_llm_model=""), SimpleNamespace())
    panel.investigation_id = 1
    root_path = Path("C:/Onderzoek")
    root = panel.add_folder_item(panel.document_tree, "Onderzoek", root_path)
    reports = panel.add_folder_item(root, "Rapporten", root_path / "Rapporten")
    notes = panel.add_folder_item(root, "Notities", root_path / "Notities")
    alpha = panel.add_document_item(
        reports, document_row(1, "C:/Onderzoek/Rapporten/Alpha.pdf")
    )
    beta = panel.add_document_item(
        notes, document_row(2, "C:/Onderzoek/Notities/Beta.pdf")
    )
    panel._document_count = 2

    panel.document_tree.setCurrentItem(reports)
    assert panel.selected_document() is None

    panel.document_tree.setCurrentItem(alpha)
    assert panel.selected_file_id() == 1

    panel.document_search.setText("rapporten")
    assert not alpha.isHidden()
    assert beta.isHidden()
    assert not reports.isHidden()
    assert panel.document_status_label.text() == "1 van 2 documenten zichtbaar"

    panel.close()


def test_ledger_row_opens_source_chunks_and_failure_is_visible(app) -> None:
    chunks = {
        "c1": {
            "chunk_text": "Tekst van de eerste bronchunk.",
            "display_markdown": None,
            "path": "C:/Onderzoek/Rapport.pdf",
            "chunk_index": 0,
            "metadata_json": "{}",
        },
        "c2": {
            "chunk_text": "Tekst van de tweede bronchunk.",
            "display_markdown": None,
            "path": "C:/Onderzoek/Rapport.pdf",
            "chunk_index": 1,
            "metadata_json": "{}",
        },
    }
    database = SimpleNamespace(get_chunk=lambda chunk_id: chunks.get(chunk_id))
    panel = FactLedgerPanel(SimpleNamespace(ollama_llm_model=""), database)
    panel.populate_table([FactLedgerItem("Een verklaring.", ("c1", "c2"))])

    panel.table.selectRow(0)
    app.processEvents()
    assert panel.chunk_preview.toPlainText() == "Tekst van de eerste bronchunk."
    assert panel.chunk_preview.next_button.isEnabled()

    panel.move_preview_chunk(1)
    assert panel.chunk_preview.toPlainText() == "Tekst van de tweede bronchunk."
    assert panel.chunk_preview.previous_button.isEnabled()

    panel.apply_failure_for_run(
        panel._run_id,
        FactLedgerFailure("extractie", ("c1", "c2"), (1, 2), "ongeldig", 1),
    )
    assert not panel.failure_label.isHidden()
    assert "chunks 1, 2" in panel.failure_label.text()
    panel.close()
