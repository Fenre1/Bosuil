from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.gui.chunk_preview import ChunkPreview
from investigation_rag.gui.folder_scope import FolderScopePanel
from investigation_rag.utils.windows_open import open_containing_folder, open_file


@dataclass(frozen=True)
class FilesLoadResult:
    request_id: int
    investigation_id: int
    roots: list[Path]
    files: list[dict]
    query_seconds: float = 0.0
    error: str = ""


@dataclass
class TreePopulateState:
    request_id: int
    roots: list[Path]
    files: list[dict]
    root_items: dict[str, QTreeWidgetItem]
    items_by_path: dict[str, QTreeWidgetItem]
    next_file_index: int = 0
    folder_lookup_seconds: float = 0.0
    widget_seconds: float = 0.0


class FilesLoadWorker(QObject):
    finished = pyqtSignal(object)

    def __init__(self, db: SQLiteDB, investigation_id: int, lazy_load_chunk_text: bool, request_id: int):
        super().__init__()
        self.db = db
        self.investigation_id = investigation_id
        self.lazy_load_chunk_text = lazy_load_chunk_text
        self.request_id = request_id

    def run(self) -> None:
        started = perf_counter()
        try:
            roots = self.db.get_roots(self.investigation_id)
            files = [dict(row) for row in self.db.file_browser_files_for_investigation(self.investigation_id)]
            result = FilesLoadResult(
                self.request_id,
                self.investigation_id,
                roots,
                files,
                query_seconds=perf_counter() - started,
            )
        except Exception as exc:
            result = FilesLoadResult(
                self.request_id,
                self.investigation_id,
                [],
                [],
                query_seconds=perf_counter() - started,
                error=str(exc),
            )
        self.finished.emit(result)


class FilesPanel(QWidget):
    load_started = pyqtSignal(int)
    load_finished = pyqtSignal(int, str)

    POPULATE_BATCH_SIZE = 120

    def __init__(self, db: SQLiteDB, lazy_load_chunk_text: bool = True):
        super().__init__()
        self.db = db
        self.lazy_load_chunk_text = lazy_load_chunk_text
        self.investigation_id: int | None = None
        self.status_label = QLabel()
        self.status_label.setObjectName("StatusLabel")
        self.load_thread: QThread | None = None
        self.load_worker: FilesLoadWorker | None = None
        self._load_request_id = 0
        self._reload_pending = False
        self._populate_state: TreePopulateState | None = None
        self._load_started_at: float | None = None
        self._populate_started_at: float | None = None
        self.last_load_timing: dict[str, float | int] = {}

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["File / chunk", "Type", "Details"])
        self.tree.itemSelectionChanged.connect(self.update_preview)
        self.tree.itemExpanded.connect(self.load_file_chunks)

        self.preview = ChunkPreview(db=db)
        self.preview.previous_requested.connect(lambda: self.select_adjacent_chunk(-1))
        self.preview.next_requested.connect(lambda: self.select_adjacent_chunk(1))

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)
        self.open_button = QPushButton("Open file")
        self.open_button.clicked.connect(self.open_selected_file)
        self.folder_button = QPushButton("Open folder")
        self.folder_button.clicked.connect(self.open_selected_folder)
        self.refresh_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.folder_button.setEnabled(False)

        controls = QHBoxLayout()
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.open_button)
        controls.addWidget(self.folder_button)
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
        self._load_request_id += 1
        request_id = self._load_request_id
        self._populate_state = None
        self.tree.clear()
        self.preview.clear()
        if self.investigation_id is None:
            self._reload_pending = False
            self.set_loading(False)
            self.update_status(0, 0)
            return

        self.set_loading(True, "Loading files and chunks...")
        self._load_started_at = perf_counter()
        self.load_started.emit(self.investigation_id)
        if self.load_thread is not None:
            self._reload_pending = True
            return
        self.start_load_thread(self.investigation_id, request_id)

    def start_load_thread(self, investigation_id: int, request_id: int) -> None:
        thread = QThread(self)
        worker = FilesLoadWorker(self.db, investigation_id, self.lazy_load_chunk_text, request_id)
        self.load_thread = thread
        self.load_worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self.apply_load_result)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.load_thread_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def apply_load_result(self, result: FilesLoadResult) -> None:
        if result.request_id != self._load_request_id or result.investigation_id != self.investigation_id:
            return
        if result.error:
            self.set_loading(False)
            self.tree.clear()
            error_item = QTreeWidgetItem(self.tree)
            error_item.setText(0, "Could not load file list.")
            error_item.setText(1, "error")
            self.status_label.setText("Unable to load files")
            self.load_finished.emit(result.investigation_id, result.error)
            QMessageBox.warning(self, "Files", result.error)
            return
        self.start_tree_population(result)

    def load_thread_finished(self) -> None:
        self.load_thread = None
        self.load_worker = None
        if not self._reload_pending:
            return
        self._reload_pending = False
        if self.investigation_id is None:
            self.set_loading(False)
            self.update_status(0, 0)
            return
        self.start_load_thread(self.investigation_id, self._load_request_id)

    def start_tree_population(self, result: FilesLoadResult) -> None:
        self._populate_started_at = perf_counter()
        self.last_load_timing = {
            "query_seconds": result.query_seconds,
            "file_count": len(result.files),
        }
        root_items: dict[str, QTreeWidgetItem] = {}
        items_by_path: dict[str, QTreeWidgetItem] = {}
        self.tree.clear()
        self.tree.setUpdatesEnabled(False)
        try:
            for root in result.roots:
                root_key = FolderScopePanel.path_key(root)
                item = self.add_folder_item(self.tree, root.name or str(root), root)
                root_items[root_key] = item
                items_by_path[root_key] = item
        finally:
            self.tree.setUpdatesEnabled(True)

        files = sorted(result.files, key=lambda row: str(row["normalized_path"]))
        self._populate_state = TreePopulateState(
            request_id=result.request_id,
            roots=result.roots,
            files=files,
            root_items=root_items,
            items_by_path=items_by_path,
        )
        self.status_label.setText(f"Loading files... 0/{len(files)}")
        QTimer.singleShot(0, self.populate_next_batch)

    def populate_next_batch(self) -> None:
        state = self._populate_state
        if state is None or state.request_id != self._load_request_id:
            return
        end_index = min(state.next_file_index + self.POPULATE_BATCH_SIZE, len(state.files))
        self.tree.setUpdatesEnabled(False)
        try:
            for file_row in state.files[state.next_file_index:end_index]:
                file_path = Path(str(file_row["path"]))
                phase_started = perf_counter()
                parent = self.ensure_folder_item(file_path.parent, state.roots, state.root_items, state.items_by_path)
                state.folder_lookup_seconds += perf_counter() - phase_started
                phase_started = perf_counter()
                self.add_file_item(parent, file_row)
                state.widget_seconds += perf_counter() - phase_started
        finally:
            self.tree.setUpdatesEnabled(True)

        state.next_file_index = end_index
        self.status_label.setText(f"Loading files... {end_index}/{len(state.files)}")
        if end_index < len(state.files):
            QTimer.singleShot(0, self.populate_next_batch)
            return

        self.tree.expandToDepth(0)
        self.tree.setColumnWidth(0, 360)
        chunk_count = sum(int(file_row["active_chunk_count"]) for file_row in state.files)
        self._populate_state = None
        populate_seconds = perf_counter() - self._populate_started_at if self._populate_started_at is not None else 0.0
        total_seconds = perf_counter() - self._load_started_at if self._load_started_at is not None else populate_seconds
        self.last_load_timing.update(
            {
                "populate_seconds": populate_seconds,
                "folder_lookup_seconds": state.folder_lookup_seconds,
                "widget_seconds": state.widget_seconds,
                "total_seconds": total_seconds,
                "chunk_count": chunk_count,
            }
        )
        self.set_loading(False)
        self.update_status(len(state.files), chunk_count)
        if self.investigation_id is not None:
            self.load_finished.emit(self.investigation_id, "")

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
            file_item.takeChildren()
            for chunk in chunks_by_file_id.get(int(file_row["id"]), []):
                self.add_chunk_item(file_item, chunk)

    def ensure_folder_item(
        self,
        folder: Path,
        roots: list[Path],
        root_items: dict[str, QTreeWidgetItem],
        items_by_path: dict[str, QTreeWidgetItem],
    ) -> QTreeWidgetItem | QTreeWidget:
        folder_key = FolderScopePanel.path_key(folder)
        existing_item = items_by_path.get(folder_key)
        if existing_item is not None:
            return existing_item
        root = FolderScopePanel.best_root(folder, roots)
        if root is None:
            item = self.add_folder_item(self.tree, str(folder), folder)
            items_by_path[folder_key] = item
            return item

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
        chunk_count = int(file_row["active_chunk_count"])
        item = QTreeWidgetItem(parent)
        item.setText(0, Path(path).name)
        item.setText(1, str(file_row["source_type"]))
        item.setText(2, f"{file_row['extraction_status']}; {chunk_count} chunks")
        item.setData(
            0,
            Qt.UserRole,
            {
                "kind": "file",
                "file_id": int(file_row["id"]),
                "path": path,
                "chunks_loaded": chunk_count == 0,
            },
        )
        item.setToolTip(0, path)
        if chunk_count:
            placeholder = QTreeWidgetItem(item)
            placeholder.setText(0, f"{chunk_count} chunks — expand to load")
            placeholder.setText(1, "loading")
        return item

    def load_file_chunks(self, item: QTreeWidgetItem) -> None:
        data = item.data(0, Qt.UserRole)
        if not isinstance(data, dict) or data.get("kind") != "file" or data.get("chunks_loaded"):
            return
        file_id = data.get("file_id")
        if not isinstance(file_id, int):
            return
        try:
            if self.lazy_load_chunk_text:
                chunks = self.db.active_chunk_summaries_for_file(file_id)
            else:
                chunks = self.db.active_chunks_for_file(file_id)
        except Exception as exc:
            item.takeChildren()
            error_item = QTreeWidgetItem(item)
            error_item.setText(0, f"Could not load chunks: {exc}")
            error_item.setText(1, "error")
            return
        item.takeChildren()
        for chunk in chunks:
            self.add_chunk_item(item, chunk)
        data["chunks_loaded"] = True
        item.setData(0, Qt.UserRole, data)

    def add_chunk_item(self, parent: QTreeWidgetItem, chunk) -> QTreeWidgetItem:
        try:
            metadata = json.loads(chunk["metadata_json"] or "{}") if not self.lazy_load_chunk_text else {}
        except (json.JSONDecodeError, TypeError):
            metadata = {}
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
                "markdown": None if self.lazy_load_chunk_text else chunk["display_markdown"],
                "chunk_index": int(chunk["chunk_index"]),
                "metadata": metadata,
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
        markdown = data.get("markdown")
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
            markdown = chunk["display_markdown"]
            try:
                metadata = json.loads(chunk["metadata_json"] or "{}")
            except (json.JSONDecodeError, TypeError):
                metadata = {}
            data["text"] = text
            data["markdown"] = markdown
            data["metadata"] = metadata
            item = self.tree.currentItem()
            if item is not None:
                item.setData(0, Qt.UserRole, data)
        item = self.tree.currentItem()
        parent = item.parent() if item is not None else None
        row = parent.indexOfChild(item) if parent is not None and item is not None else -1
        self.preview.set_chunk(
            str(text),
            markdown if isinstance(markdown, str) else None,
            path=str(data.get("path", "")),
            chunk_index=data.get("chunk_index"),
            metadata=data.get("metadata") if isinstance(data.get("metadata"), dict) else {},
            has_previous=row > 0,
            has_next=parent is not None and row + 1 < parent.childCount(),
        )

    def select_adjacent_chunk(self, offset: int) -> None:
        item = self.tree.currentItem()
        parent = item.parent() if item is not None else None
        if item is None or parent is None:
            return
        target = parent.indexOfChild(item) + offset
        if 0 <= target < parent.childCount():
            self.tree.setCurrentItem(parent.child(target))

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

    def set_loading(self, is_loading: bool, message: str = "") -> None:
        has_investigation = self.investigation_id is not None
        self.tree.setEnabled(not is_loading)
        self.preview.setEnabled(not is_loading)
        self.refresh_button.setEnabled(not is_loading and has_investigation)
        self.open_button.setEnabled(not is_loading and has_investigation)
        self.folder_button.setEnabled(not is_loading and has_investigation)
        if is_loading:
            self.setCursor(Qt.WaitCursor)
            self.tree.clear()
            loading_item = QTreeWidgetItem(self.tree)
            loading_item.setText(0, message or "Loading files and chunks...")
            loading_item.setText(1, "loading")
            self.status_label.setText(message or "Loading files and chunks...")
        else:
            self.unsetCursor()

    def shutdown(self, wait_ms: int = 5000) -> None:
        self._load_request_id += 1
        self._reload_pending = False
        self._populate_state = None
        if self.load_thread is None or not self.load_thread.isRunning():
            return
        self.load_thread.quit()
        self.load_thread.wait(wait_ms)

    @staticmethod
    def chunk_location(chunk) -> str:
        if chunk["page_number"]:
            return f"page {chunk['page_number']}"
        if chunk["email_subject"]:
            return str(chunk["email_subject"])
        if chunk["attachment_name"]:
            return f"attachment: {chunk['attachment_name']}"
        return ""
