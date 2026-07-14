from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.gui.folder_scope import FolderScopePanel
from investigation_rag.utils.windows_open import open_containing_folder, open_file


class FilesPanel(QWidget):
    def __init__(self, db: SQLiteDB, lazy_load_chunk_text: bool = True):
        super().__init__()
        self.db = db
        self.lazy_load_chunk_text = lazy_load_chunk_text
        self.investigation_id: int | None = None
        self.status_label = QLabel()

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File / chunk", "Type", "Details"])
        self.tree.itemSelectionChanged.connect(self.update_preview)

        self.preview = QTextEdit()
        self.preview.setReadOnly(True)

        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.reload)
        open_button = QPushButton("Open file")
        open_button.clicked.connect(self.open_selected_file)
        folder_button = QPushButton("Open folder")
        folder_button.clicked.connect(self.open_selected_folder)

        controls = QHBoxLayout()
        controls.addWidget(refresh_button)
        controls.addWidget(open_button)
        controls.addWidget(folder_button)
        controls.addStretch(1)
        controls.addWidget(self.status_label)

        splitter = QSplitter()
        splitter.addWidget(self.tree)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(splitter, 1)

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.reload()

    def reload(self) -> None:
        self.tree.clear()
        self.preview.clear()
        if self.investigation_id is None:
            self.update_status(0, 0)
            return

        roots = self.db.get_roots(self.investigation_id)
        files = self.db.active_files_for_investigation(self.investigation_id)
        if self.lazy_load_chunk_text:
            chunks = self.db.active_chunk_summaries_for_investigation(self.investigation_id)
        else:
            chunks = self.db.active_chunks_for_investigation(self.investigation_id)
        chunks_by_file_id: dict[int, list] = {}
        for chunk in chunks:
            chunks_by_file_id.setdefault(int(chunk["file_id"]), []).append(chunk)

        self.populate_tree(roots, files, chunks_by_file_id)
        self.tree.expandToDepth(0)
        self.tree.resizeColumnToContents(0)
        self.update_status(len(files), len(chunks))

    def populate_tree(self, roots: list[Path], files: list, chunks_by_file_id: dict[int, list]) -> None:
        root_items: dict[str, QTreeWidgetItem] = {}
        items_by_path: dict[str, QTreeWidgetItem] = {}
        for root in roots:
            root_key = FolderScopePanel.path_key(root)
            item = self.add_folder_item(self.tree, root.name or str(root), root)
            root_items[root_key] = item
            items_by_path[root_key] = item

        for file_row in sorted(files, key=lambda row: str(row["normalized_path"])):
            file_path = Path(str(file_row["path"]))
            parent = self.ensure_folder_item(file_path.parent, roots, root_items, items_by_path)
            file_item = self.add_file_item(parent, file_row)
            for chunk in chunks_by_file_id.get(int(file_row["id"]), []):
                self.add_chunk_item(file_item, chunk)

    def ensure_folder_item(
        self,
        folder: Path,
        roots: list[Path],
        root_items: dict[str, QTreeWidgetItem],
        items_by_path: dict[str, QTreeWidgetItem],
    ) -> QTreeWidgetItem | QTreeWidget:
        root = FolderScopePanel.best_root(folder, roots)
        if root is None:
            key = FolderScopePanel.path_key(folder)
            if key not in items_by_path:
                items_by_path[key] = self.add_folder_item(self.tree, str(folder), folder)
            return items_by_path[key]

        parent = root_items[FolderScopePanel.path_key(root)]
        current = root
        try:
            relative_parts = folder.relative_to(root).parts
        except ValueError:
            relative_parts = ()
        for part in relative_parts:
            current = current / part
            key = FolderScopePanel.path_key(current)
            if key not in items_by_path:
                items_by_path[key] = self.add_folder_item(parent, part, current)
            parent = items_by_path[key]
        return parent

    def add_folder_item(self, parent: QTreeWidget | QTreeWidgetItem, label: str, path: Path) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent)
        item.setText(0, label)
        item.setText(1, "folder")
        item.setData(0, Qt.UserRole, {"kind": "folder", "path": str(path)})
        item.setToolTip(0, str(path))
        return item

    def add_file_item(self, parent: QTreeWidget | QTreeWidgetItem, file_row) -> QTreeWidgetItem:
        path = str(file_row["path"])
        item = QTreeWidgetItem(parent)
        item.setText(0, Path(path).name)
        item.setText(1, str(file_row["source_type"]))
        item.setText(2, str(file_row["extraction_status"]))
        item.setData(0, Qt.UserRole, {"kind": "file", "path": path})
        item.setToolTip(0, path)
        return item

    def add_chunk_item(self, parent: QTreeWidgetItem, chunk) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent)
        item.setText(0, f"Chunk {int(chunk['chunk_index']) + 1}")
        item.setText(1, str(chunk["source_type"]))
        item.setText(2, self.chunk_location(chunk))
        item.setData(
            0,
            Qt.UserRole,
            {
                "kind": "chunk",
                "chunk_id": str(chunk["id"]),
                "path": str(chunk["path"]),
                "text": None if self.lazy_load_chunk_text else str(chunk["chunk_text"]),
            },
        )
        item.setToolTip(0, "Click to show chunk content")
        return item

    def update_preview(self) -> None:
        data = self.selected_item_data()
        if data is None or data.get("kind") != "chunk":
            self.preview.clear()
            return
        text = data.get("text")
        if text is None:
            chunk_id = data.get("chunk_id")
            if not isinstance(chunk_id, str):
                self.preview.clear()
                return
            chunk = self.db.get_chunk(chunk_id)
            if chunk is None:
                self.preview.setPlainText("Chunk text is no longer available.")
                return
            text = str(chunk["chunk_text"])
            data["text"] = text
            item = self.tree.currentItem()
            if item is not None:
                item.setData(0, Qt.UserRole, data)
        self.preview.setPlainText(str(text))

    def selected_item_data(self) -> dict | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        data = item.data(0, Qt.UserRole)
        return data if isinstance(data, dict) else None

    def open_selected_file(self) -> None:
        data = self.selected_item_data()
        if data is None or data.get("kind") == "folder":
            return
        try:
            open_file(str(data["path"]))
        except Exception as exc:
            QMessageBox.warning(self, "Open file", str(exc))

    def open_selected_folder(self) -> None:
        data = self.selected_item_data()
        if data is None:
            return
        try:
            open_containing_folder(str(data["path"]))
        except Exception as exc:
            QMessageBox.warning(self, "Open folder", str(exc))

    def update_status(self, file_count: int, chunk_count: int) -> None:
        self.status_label.setText(f"{file_count} files, {chunk_count} chunks")

    @staticmethod
    def chunk_location(chunk) -> str:
        if chunk["page_number"]:
            return f"page {chunk['page_number']}"
        if chunk["email_subject"]:
            return str(chunk["email_subject"])
        if chunk["attachment_name"]:
            return f"attachment: {chunk['attachment_name']}"
        return ""
