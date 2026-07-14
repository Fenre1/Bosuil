from __future__ import annotations

from PyQt5.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.gui.disappeared_files_panel import DisappearedFilesPanel
from investigation_rag.gui.files_panel import FilesPanel
from investigation_rag.gui.folder_scope import FolderScope, FolderScopePanel
from investigation_rag.gui.investigation_dialog import InvestigationDialog
from investigation_rag.gui.rag_panel import RagPanel
from investigation_rag.gui.scan_panel import ScanPanel
from investigation_rag.gui.search_panel import SearchPanel


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, db: SQLiteDB, store: LanceDBStore):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.folder_scope = FolderScope()
        self.setWindowTitle("Local Investigation RAG")

        self.investigation_combo = QComboBox()
        self.investigation_combo.currentIndexChanged.connect(self.on_investigation_changed)
        create_button = QPushButton("New")
        create_button.clicked.connect(self.create_investigation)
        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self.edit_investigation)
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.reload_investigations)

        top = QHBoxLayout()
        top.addWidget(self.investigation_combo, 1)
        top.addWidget(create_button)
        top.addWidget(edit_button)
        top.addWidget(refresh_button)

        self.tabs = QTabWidget()
        self.scan_panel = ScanPanel(config, db, store)
        self.folder_scope_panel = FolderScopePanel(db, self.folder_scope)
        self.files_panel = FilesPanel(db, lazy_load_chunk_text=config.gui_lazy_load_file_chunk_text)
        self.search_panel = SearchPanel(config, db, store, self.folder_scope)
        self.rag_panel = RagPanel(config, db, store, self.folder_scope)
        self.disappeared_panel = DisappearedFilesPanel(db)
        self.tabs.addTab(self.scan_panel, "Scan")
        self.tabs.addTab(self.folder_scope_panel, "Folders")
        self.tabs.addTab(self.files_panel, "Files")
        self.tabs.addTab(self.search_panel, "Search")
        self.tabs.addTab(self.rag_panel, "RAG")
        self.tabs.addTab(self.disappeared_panel, "Disappeared")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addLayout(top)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(container)

        self.reload_investigations()

    def active_investigation_id(self) -> int | None:
        data = self.investigation_combo.currentData()
        return int(data) if data is not None else None

    def reload_investigations(self) -> None:
        current = self.active_investigation_id()
        self.investigation_combo.blockSignals(True)
        self.investigation_combo.clear()
        for investigation in self.db.list_investigations():
            self.investigation_combo.addItem(investigation.name, investigation.id)
        if current is not None:
            index = self.investigation_combo.findData(current)
            if index >= 0:
                self.investigation_combo.setCurrentIndex(index)
        self.investigation_combo.blockSignals(False)
        self.on_investigation_changed()

    def on_investigation_changed(self) -> None:
        investigation_id = self.active_investigation_id()
        for panel in (
            self.scan_panel,
            self.folder_scope_panel,
            self.files_panel,
            self.search_panel,
            self.rag_panel,
            self.disappeared_panel,
        ):
            panel.set_investigation_id(investigation_id)

    def create_investigation(self) -> None:
        dialog = InvestigationDialog(self)
        if dialog.exec_() != dialog.Accepted:
            return
        name, notes, roots = dialog.values()
        if not name or not roots:
            QMessageBox.warning(self, "Investigation", "Name and at least one root folder are required.")
            return
        try:
            investigation_id = self.db.create_investigation(name, roots, notes)
        except Exception as exc:
            QMessageBox.critical(self, "Investigation", str(exc))
            return
        self.reload_investigations()
        index = self.investigation_combo.findData(investigation_id)
        if index >= 0:
            self.investigation_combo.setCurrentIndex(index)

    def edit_investigation(self) -> None:
        investigation_id = self.active_investigation_id()
        if investigation_id is None:
            return
        row = self.db.get_investigation(investigation_id)
        if row is None:
            return
        dialog = InvestigationDialog(
            self,
            name=row["name"],
            notes=row["notes"],
            roots=self.db.get_roots(investigation_id),
        )
        if dialog.exec_() != dialog.Accepted:
            return
        name, notes, roots = dialog.values()
        if not name or not roots:
            QMessageBox.warning(self, "Investigation", "Name and at least one root folder are required.")
            return
        try:
            self.db.update_investigation(investigation_id, name, notes, roots)
        except Exception as exc:
            QMessageBox.critical(self, "Investigation", str(exc))
            return
        self.reload_investigations()

    def closeEvent(self, event) -> None:
        self.scan_panel.shutdown()
        super().closeEvent(event)
