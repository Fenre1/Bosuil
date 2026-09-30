from dataclasses import replace
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt5.QtCore import Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication

from investigation_rag.config import AppConfig
from investigation_rag.gui.chunk_preview import ChunkPreview
from investigation_rag.gui.evaluation_panel import EvaluationPanel
from investigation_rag.gui.entities_panel import EntitiesPanel
from investigation_rag.gui.main_window import MainWindow
from investigation_rag.gui.rag_panel import RagPanel
from investigation_rag.workspace import WorkspaceCatalog


@pytest.fixture
def window(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    # Navigation and layout must work without contacting a model server.
    monkeypatch.setattr(RagPanel, "refresh_answer_models", lambda self: None)
    monkeypatch.setattr(EvaluationPanel, "refresh_answer_models", lambda self: None)
    monkeypatch.setattr(EntitiesPanel, "refresh_entity_models", lambda self: None)
    config = replace(AppConfig.from_env(), app_data_dir=tmp_path)
    catalog = WorkspaceCatalog(tmp_path)
    catalog.initialize()
    window = MainWindow(config, catalog)
    window.resize(1200, 780)
    window.show()
    app.processEvents()
    yield window
    window.close()


def test_navigation_reaches_every_page_and_keeps_heading_in_sync(window):
    assert window.page_title.text() == "Onderzoeken"
    window.navigation.setFocus()
    for index, (panel, title, description) in enumerate(window.pages):
        assert window.tabs.currentIndex() == index
        assert window.tabs.widget(index).widget() is panel
        assert panel.isVisible()
        assert window.page_title.text() == title
        assert window.page_description.text() == description
        if index < len(window.pages) - 1:
            QTest.keyClick(window.navigation, Qt.Key_Down)
    window.tabs.setCurrentIndex(2)
    assert window.navigation.currentRow() == 2
    assert window.page_title.text() == "Bestanden"


def test_empty_state_gives_way_to_selectable_investigation(window, tmp_path):
    assert window.investigation_empty_state.isVisible()
    assert not window.investigation_list.isVisible()
    assert not window.load_button.isEnabled()
    workspace = window.catalog.create("Test investigation", [tmp_path / "evidence"])
    window.reload_investigations(select_id=workspace.entry.id)
    assert not window.investigation_empty_state.isVisible()
    assert window.investigation_list.isVisible()
    assert window.selected_workspace_id() == workspace.entry.id
    assert window.load_button.isEnabled()
    # Selecting a workspace must not silently load it or start indexing.
    assert window.active_workspace_id() is None
    window.refresh_button.click()
    assert window.selected_workspace_id() == workspace.entry.id


def test_clear_preview_restores_instructions_and_disables_navigation(window):
    preview = ChunkPreview("Select a source", navigation_label="citation")
    preview.set_chunk("Evidence", has_previous=True, has_next=True)
    assert preview.stack.currentWidget() is preview.source_view
    preview.clear()
    assert preview.stack.currentWidget() is preview.viewer
    assert preview.viewer.placeholderText() == "Select a source"
    assert not preview.previous_button.isEnabled()
    assert not preview.next_button.isEnabled()
    preview.close()
