from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
from investigation_rag.gui.file_type_scope import FileTypeScope, FileTypeScopePanel
from investigation_rag.gui.files_panel import FilesPanel
from investigation_rag.gui.folder_scope import FolderScope, FolderScopePanel
from investigation_rag.gui.investigation_dialog import InvestigationDialog
from investigation_rag.gui.rag_panel import RagPanel
from investigation_rag.gui.scan_panel import ScanPanel
from investigation_rag.gui.search_panel import SearchPanel


def app_icon_path() -> Path | None:
    module_path = Path(__file__).resolve()
    candidates = [
        module_path.with_name("icon_small.png"),
        module_path.parents[3] / "icon_small.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def app_icon() -> QIcon:
    icon_path = app_icon_path()
    if icon_path is None:
        return QIcon()
    return QIcon(str(icon_path))


def crop_transparent_padding(pixmap: QPixmap) -> QPixmap:
    image = pixmap.toImage()
    if image.isNull() or not image.hasAlphaChannel():
        return pixmap

    left = image.width()
    top = image.height()
    right = -1
    bottom = -1
    for y in range(image.height()):
        for x in range(image.width()):
            if image.pixelColor(x, y).alpha() > 0:
                left = min(left, x)
                top = min(top, y)
                right = max(right, x)
                bottom = max(bottom, y)

    if right < left or bottom < top:
        return pixmap
    return QPixmap.fromImage(image.copy(left, top, right - left + 1, bottom - top + 1))


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, db: SQLiteDB, store: LanceDBStore):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.folder_scope = FolderScope()
        self.file_type_scope = FileTypeScope()
        self._active_investigation_id: int | None = None
        self._loading_investigation_id: int | None = None
        self.setWindowTitle("Bosuil, de lokale antwoordvogel")
        icon = app_icon()
        if not icon.isNull():
            self.setWindowIcon(icon)

        self.investigation_list = QListWidget()
        self.investigation_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.investigation_list.itemSelectionChanged.connect(self.update_investigation_actions)
        self.investigation_list.itemDoubleClicked.connect(lambda _: self.load_selected_investigation())
        self.create_button = QPushButton("New")
        self.create_button.setObjectName("PrimaryButton")
        self.create_button.clicked.connect(self.create_investigation)
        self.edit_button = QPushButton("Edit")
        self.edit_button.setObjectName("GhostButton")
        self.edit_button.clicked.connect(self.edit_investigation)
        self.load_button = QPushButton("Load")
        self.load_button.setObjectName("PrimaryButton")
        self.load_button.clicked.connect(self.load_selected_investigation)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setObjectName("GhostButton")
        self.refresh_button.clicked.connect(lambda: self.reload_investigations())

        title = QLabel("Bosuil, de lokale antwoordvogel")
        title.setObjectName("AppTitle")
        self.header_investigation_label = QLabel("")
        self.header_investigation_label.setObjectName("HeaderInvestigationName")

        top = QHBoxLayout()
        top.setContentsMargins(18, 14, 18, 16)
        top.setSpacing(10)
        title_icon = self.build_title_icon()
        if title_icon is not None:
            top.addWidget(title_icon)
        top.addWidget(title)
        top.addWidget(self.header_investigation_label)
        top.addStretch(1)

        header = QWidget()
        header.setObjectName("AppHeader")
        header.setLayout(top)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.tabBar().setExpanding(False)
        self.tabs.tabBar().setElideMode(Qt.ElideNone)
        self.tabs.tabBar().setUsesScrollButtons(True)
        self.scan_panel = ScanPanel(config, db, store)
        self.scan_panel.scan_button.clicked.connect(self.scan_selected_investigation)
        self.scan_panel.set_investigation_selector(self.build_investigation_selector())
        self.folder_scope_panel = FolderScopePanel(db, self.folder_scope)
        self.file_type_scope_panel = FileTypeScopePanel(self.file_type_scope)
        self.filters_panel = self.build_filters_panel()
        self.files_panel = FilesPanel(db, lazy_load_chunk_text=config.gui_lazy_load_file_chunk_text)
        self.files_panel.load_started.connect(self.on_investigation_data_load_started)
        self.files_panel.load_finished.connect(self.on_investigation_data_load_finished)
        self.search_panel = SearchPanel(config, db, store, self.folder_scope, self.file_type_scope)
        self.rag_panel = RagPanel(config, db, store, self.folder_scope, self.file_type_scope)
        self.tabs.addTab(self.scan_panel, "Onderzoeken")
        self.tabs.addTab(self.filters_panel, "Filters")
        self.tabs.addTab(self.files_panel, "Bestanden")
        self.tabs.addTab(self.search_panel, "Zoeken")
        self.tabs.addTab(self.rag_panel, "De Uil Antwoord")

        container = QWidget()
        container.setObjectName("AppRoot")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(container)

        self.reload_investigations()
        self.on_investigation_changed()

    def build_filters_panel(self) -> QWidget:
        self.file_type_scope_panel.layout().setContentsMargins(0, 0, 0, 0)
        self.folder_scope_panel.layout().setContentsMargins(0, 0, 0, 0)

        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addWidget(self.file_type_scope_panel)
        layout.addWidget(self.folder_scope_panel, 1)
        return panel

    def build_title_icon(self) -> QLabel | None:
        icon_path = app_icon_path()
        if icon_path is None:
            return None
        pixmap = QPixmap(str(icon_path))
        if pixmap.isNull():
            return None
        pixmap = crop_transparent_padding(pixmap)
        icon_label = QLabel()
        icon_label.setObjectName("AppIcon")
        icon_label.setFixedSize(58, 58)
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setPixmap(pixmap.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return icon_label

    def build_investigation_selector(self) -> QWidget:
        investigation_label = QLabel("Investigation")
        investigation_label.setObjectName("FieldLabel")

        self.loaded_investigation_label = QLabel("No investigation loaded")
        self.loaded_investigation_label.setObjectName("StatusLabel")

        selector = QWidget()
        selector.setObjectName("InvestigationPanel")
        selector.setMinimumWidth(320)
        selector_layout = QVBoxLayout(selector)
        selector_layout.setContentsMargins(14, 12, 14, 14)
        selector_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        header_row.addWidget(investigation_label)
        header_row.addStretch(1)
        header_row.addWidget(self.refresh_button)

        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        action_row.addWidget(self.create_button)
        action_row.addWidget(self.edit_button)
        action_row.addWidget(self.load_button)

        scan_row = QHBoxLayout()
        scan_row.setSpacing(8)
        scan_row.addWidget(self.scan_panel.scan_button)

        selector_layout.addLayout(header_row)
        selector_layout.addWidget(self.investigation_list, 1)
        selector_layout.addWidget(self.loaded_investigation_label)
        selector_layout.addLayout(action_row)
        selector_layout.addLayout(scan_row)
        return selector

    def active_investigation_id(self) -> int | None:
        return self._active_investigation_id

    def selected_investigation_id(self) -> int | None:
        item = self.investigation_list.currentItem()
        if item is None:
            return None
        data = item.data(Qt.UserRole)
        return int(data) if data is not None else None

    def reload_investigations(self, select_id: int | None = None) -> None:
        current = select_id if select_id is not None else self.selected_investigation_id()
        loaded_id = self.active_investigation_id()
        seen_ids = set()
        self.investigation_list.blockSignals(True)
        self.investigation_list.clear()
        for investigation in self.db.list_investigations():
            seen_ids.add(investigation.id)
            item = QListWidgetItem(investigation.name)
            item.setData(Qt.UserRole, investigation.id)
            self.investigation_list.addItem(item)
        if loaded_id is not None and loaded_id not in seen_ids:
            self._active_investigation_id = None
        if current is not None:
            self.select_investigation(current)
        else:
            self.investigation_list.setCurrentRow(-1)
        self.investigation_list.blockSignals(False)
        self.update_investigation_actions()
        self.refresh_loaded_investigation_marker()
        if loaded_id != self.active_investigation_id():
            self.on_investigation_changed()

    def select_investigation(self, investigation_id: int) -> None:
        for row in range(self.investigation_list.count()):
            item = self.investigation_list.item(row)
            if item is not None and item.data(Qt.UserRole) == investigation_id:
                self.investigation_list.setCurrentRow(row)
                return
        self.investigation_list.setCurrentRow(-1)

    def load_selected_investigation(self) -> None:
        investigation_id = self.selected_investigation_id()
        if investigation_id is None:
            return
        self.set_active_investigation(investigation_id)

    def set_active_investigation(self, investigation_id: int | None) -> None:
        self._active_investigation_id = investigation_id
        self._loading_investigation_id = investigation_id
        self.refresh_loaded_investigation_marker()
        QApplication.processEvents()
        self.on_investigation_changed()

    def on_investigation_changed(self) -> None:
        investigation_id = self.active_investigation_id()
        for panel in (
            self.scan_panel,
            self.folder_scope_panel,
            self.file_type_scope_panel,
            self.files_panel,
            self.search_panel,
            self.rag_panel,
        ):
            panel.set_investigation_id(investigation_id)
        self.update_investigation_actions()

    def update_investigation_actions(self) -> None:
        has_selection = self.selected_investigation_id() is not None
        self.edit_button.setEnabled(has_selection)
        self.load_button.setEnabled(has_selection)
        if self.scan_panel.thread is None:
            self.scan_panel.scan_button.setEnabled(has_selection or self.active_investigation_id() is not None)

    def refresh_loaded_investigation_marker(self) -> None:
        loaded_id = self.active_investigation_id()
        loaded_name = None
        for row in range(self.investigation_list.count()):
            item = self.investigation_list.item(row)
            if item is None:
                continue
            is_loaded = item.data(Qt.UserRole) == loaded_id
            font = item.font()
            font.setBold(is_loaded)
            item.setFont(font)
            if is_loaded:
                loaded_name = item.text()
        if loaded_name:
            self.header_investigation_label.setText(f"Investigation: {loaded_name}")
            if self._loading_investigation_id == loaded_id:
                self.loaded_investigation_label.setText(f"Loading data for: {loaded_name}...")
            else:
                self.loaded_investigation_label.setText(f"Loaded: {loaded_name}")
        else:
            self.header_investigation_label.setText("")
            self.loaded_investigation_label.setText("No investigation loaded")

    def on_investigation_data_load_started(self, investigation_id: int) -> None:
        if investigation_id != self.active_investigation_id():
            return
        self._loading_investigation_id = investigation_id
        self.refresh_loaded_investigation_marker()

    def on_investigation_data_load_finished(self, investigation_id: int, _error: str) -> None:
        if self._loading_investigation_id != investigation_id:
            return
        self._loading_investigation_id = None
        self.refresh_loaded_investigation_marker()

    def scan_selected_investigation(self) -> None:
        selected_id = self.selected_investigation_id()
        if selected_id is not None and selected_id != self.active_investigation_id():
            self.set_active_investigation(selected_id)
        self.scan_panel.start_scan()

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
        self.reload_investigations(select_id=investigation_id)
        self.set_active_investigation(investigation_id)

    def edit_investigation(self) -> None:
        investigation_id = self.selected_investigation_id()
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
        self.reload_investigations(select_id=investigation_id)
        if self.active_investigation_id() == investigation_id:
            self.on_investigation_changed()

    def closeEvent(self, event) -> None:
        self.scan_panel.shutdown()
        self.files_panel.shutdown()
        self.rag_panel.shutdown()
        super().closeEvent(event)
