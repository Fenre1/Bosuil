from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.config import AppConfig
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.embeddings.ollama_client import OllamaClient
from investigation_rag.fact_ledger import (
    FACT_BATCH_SIZE,
    FACT_BATCH_OVERLAP,
    FactLedgerCancelled,
    FactLedgerFailure,
    FactLedgerItem,
    FactLedgerModelOutput,
    build_document_fact_ledger,
    default_export_filename,
    export_fact_ledger,
    overlapping_chunk_batches,
)
from investigation_rag.gui.rag_panel import DEFAULT_ANSWER_MODEL
from investigation_rag.gui.folder_scope import FolderScopePanel
from investigation_rag.gui.chunk_preview import ChunkPreview


@dataclass
class DocumentPopulateState:
    request_id: int
    roots: list[Path]
    files: list[dict[str, object]]
    root_items: dict[str, QTreeWidgetItem]
    items_by_path: dict[str, QTreeWidgetItem]
    preferred_file_id: int | None
    next_file_index: int = 0


class FactLedgerWorker(QObject):
    progress = pyqtSignal(int, int, str)
    ledger_updated = pyqtSignal(object)
    batch_failed = pyqtSignal(object)
    model_output_received = pyqtSignal(object)
    finished = pyqtSignal(object, str, bool)

    def __init__(
        self,
        config: AppConfig,
        db: SQLiteDB,
        file_id: int,
        document_path: str,
        model_name: str,
    ):
        super().__init__()
        self.config = config
        self.db = db
        self.file_id = file_id
        self.document_path = document_path
        self.model_name = model_name
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        client: OllamaClient | None = None
        try:
            chunks = self.db.active_chunks_for_file(self.file_id)
            batch_count = len(overlapping_chunk_batches(chunks))
            completed_steps = 0

            def report(message: str) -> None:
                nonlocal completed_steps
                batch_match = re.match(r"Ronde (\d+) van \d+:", message)
                if batch_match:
                    completed_steps = max(completed_steps, int(batch_match.group(1)))
                elif message.startswith("Alle "):
                    completed_steps = batch_count
                self.progress.emit(completed_steps, batch_count + 1, message)

            client = OllamaClient(
                self.config.ollama_base_url,
                timeout=self.config.ollama_timeout_seconds,
                allow_remote=self.config.ollama_allow_remote,
                num_ctx=max(self.config.ollama_num_ctx, self.config.entity_num_ctx),
            )
            ledger = build_document_fact_ledger(
                client,
                self.model_name,
                Path(self.document_path).name,
                chunks,
                progress=report,
                ledger_updated=self.ledger_updated.emit,
                failure_callback=self.batch_failed.emit,
                model_output_callback=self.model_output_received.emit,
                cancelled=self.cancel_event.is_set,
                output_tokens=self.config.fact_ledger_output_tokens,
            )
            self.finished.emit(ledger, "", False)
        except FactLedgerCancelled:
            self.finished.emit([], "", True)
        except Exception as exc:
            self.finished.emit([], str(exc), False)
        finally:
            if client is not None:
                client.close()


class FactLedgerPanel(QWidget):
    POPULATE_BATCH_SIZE = 150

    def __init__(self, config: AppConfig, db: SQLiteDB):
        super().__init__()
        self.config = config
        self.db = db
        self.investigation_id: int | None = None
        self.thread: QThread | None = None
        self.worker: FactLedgerWorker | None = None
        self.ledger: list[FactLedgerItem] = []
        self.failures: list[FactLedgerFailure] = []
        self.model_outputs: list[FactLedgerModelOutput] = []
        self.document_path = ""
        self._run_complete = False
        self._preview_chunk_ids: list[str] = []
        self._preview_chunk_position = -1
        self._run_id = 0
        self._document_load_request_id = 0
        self._document_loading = False
        self._populate_state: DocumentPopulateState | None = None
        self._document_count = 0

        intro = QLabel(
            "Selecteer één document. De Uil extraheert alle verklaringen uit opeenvolgende sets "
            f"van {FACT_BATCH_SIZE} chunks met {FACT_BATCH_OVERLAP} overlappende chunk en controleert "
            "daarna het volledige feitenoverzicht op duplicaten en passende combinaties."
        )
        intro.setWordWrap(True)

        self.document_search = QLineEdit()
        self.document_search.setPlaceholderText("Zoek op bestandsnaam of pad")
        self.document_search.setClearButtonEnabled(True)
        self.document_search.textChanged.connect(self.apply_document_filter)
        self.document_tree = QTreeWidget()
        self.document_tree.setHeaderLabels(["Document", "Type", "Details"])
        self.document_tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.document_tree.itemSelectionChanged.connect(self.selection_changed)
        self.document_tree.itemDoubleClicked.connect(self.document_double_clicked)
        self.document_tree.setMinimumWidth(380)
        self.document_status_label = QLabel("Geen onderzoek geladen")
        self.document_status_label.setObjectName("StatusLabel")
        self.model_edit = QLineEdit(config.ollama_llm_model or DEFAULT_ANSWER_MODEL)
        self.model_edit.setMinimumWidth(210)
        self.model_edit.setPlaceholderText(DEFAULT_ANSWER_MODEL)
        self.start_button = QPushButton("Feiten extraheren")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_extraction)
        self.stop_button = QPushButton("Stoppen")
        self.stop_button.clicked.connect(self.stop_extraction)
        self.export_button = QPushButton("Exporteren naar Excel")
        self.export_button.setObjectName("GhostButton")
        self.export_button.clicked.connect(self.export_excel)
        self.refresh_button = QPushButton("Documenten vernieuwen")
        self.refresh_button.setObjectName("GhostButton")
        self.refresh_button.clicked.connect(self.reload_documents)
        for widget in (
            self.model_edit,
            self.start_button,
            self.stop_button,
            self.export_button,
            self.refresh_button,
        ):
            widget.setFixedHeight(34)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(QLabel("Model:"))
        controls.addWidget(self.model_edit)
        controls.addStretch(1)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.export_button)
        controls.addWidget(self.refresh_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.status_label = QLabel("Geen onderzoek geladen")
        self.status_label.setObjectName("StatusLabel")
        self.failure_label = QLabel()
        self.failure_label.setWordWrap(True)
        self.failure_label.setStyleSheet("color: #b45309;")
        self.failure_label.setVisible(False)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Nr.", "Verklaring", "Bronlocatie(s)", "Bronchunk-ID('s)"])
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.preview_selected_ledger_source)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)

        document_controls = QHBoxLayout()
        document_controls.setContentsMargins(0, 0, 0, 0)
        document_controls.setSpacing(6)
        document_controls.addWidget(QLabel("Document:"))
        document_controls.addWidget(self.document_search, 1)

        document_panel = QWidget()
        document_layout = QVBoxLayout(document_panel)
        document_layout.setContentsMargins(0, 0, 0, 0)
        document_layout.setSpacing(6)
        document_layout.addLayout(document_controls)
        document_layout.addWidget(self.document_status_label)
        document_layout.addWidget(self.document_tree, 1)

        self.chunk_preview = ChunkPreview(
            "Selecteer een ledgerregel om de bijbehorende chunktekst te bekijken"
        )
        self.chunk_preview.previous_requested.connect(lambda: self.move_preview_chunk(-1))
        self.chunk_preview.next_requested.connect(lambda: self.move_preview_chunk(1))

        ledger_content = QSplitter(Qt.Vertical)
        ledger_content.addWidget(self.table)
        ledger_content.addWidget(self.chunk_preview)
        ledger_content.setStretchFactor(0, 3)
        ledger_content.setStretchFactor(1, 2)

        ledger_panel = QWidget()
        ledger_layout = QVBoxLayout(ledger_panel)
        ledger_layout.setContentsMargins(0, 0, 0, 0)
        ledger_layout.setSpacing(6)
        ledger_layout.addWidget(QLabel("Evidence ledger"))
        ledger_layout.addWidget(ledger_content, 1)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(document_panel)
        splitter.addWidget(ledger_panel)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([460, 760])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(10)
        layout.addWidget(intro)
        layout.addLayout(controls)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        layout.addWidget(self.failure_label)
        layout.addWidget(splitter, 1)
        self.update_actions()

    def set_investigation_id(self, investigation_id: int | None) -> None:
        changed = investigation_id != self.investigation_id
        if self.thread is not None and changed:
            self.stop_extraction()
        if changed:
            self._run_id += 1
        self.investigation_id = investigation_id
        self.clear_result()
        if changed:
            self.document_tree.blockSignals(True)
            self.document_tree.setCurrentItem(None)
            self.document_tree.blockSignals(False)
        self.reload_documents()

    def set_data_sources(self, config: AppConfig, db: SQLiteDB) -> None:
        self.config = config
        self.db = db

    def reload_documents(self) -> None:
        previous_file_id = self.selected_file_id()
        self._document_load_request_id += 1
        request_id = self._document_load_request_id
        self._populate_state = None
        self._document_loading = False
        self.document_tree.blockSignals(True)
        self.document_tree.clear()
        if self.investigation_id is None:
            self.document_tree.blockSignals(False)
            self.document_status_label.setText("Geen onderzoek geladen")
            self.update_actions()
            return

        self._document_loading = True
        self.document_status_label.setText("Documenten laden...")
        try:
            roots = self.db.get_roots(self.investigation_id)
            files = [
                dict(row)
                for row in self.db.file_browser_files_for_investigation(self.investigation_id)
                if int(row["active_chunk_count"]) > 0
            ]
        except Exception as exc:
            self._document_loading = False
            self.document_tree.blockSignals(False)
            self.document_status_label.setText("Documenten konden niet worden geladen")
            QMessageBox.warning(self, "Feitenoverzicht", str(exc))
            self.update_actions()
            return

        root_items: dict[str, QTreeWidgetItem] = {}
        items_by_path: dict[str, QTreeWidgetItem] = {}
        for root in roots:
            key = FolderScopePanel.path_key(root)
            item = self.add_folder_item(self.document_tree, root.name or str(root), root)
            root_items[key] = item
            items_by_path[key] = item
        self._document_count = len(files)
        self._populate_state = DocumentPopulateState(
            request_id=request_id,
            roots=roots,
            files=sorted(files, key=lambda row: str(row["normalized_path"])),
            root_items=root_items,
            items_by_path=items_by_path,
            preferred_file_id=previous_file_id,
        )
        self.update_actions()
        QTimer.singleShot(0, self.populate_next_document_batch)

    def populate_next_document_batch(self) -> None:
        state = self._populate_state
        if state is None or state.request_id != self._document_load_request_id:
            return
        end = min(state.next_file_index + self.POPULATE_BATCH_SIZE, len(state.files))
        self.document_tree.setUpdatesEnabled(False)
        try:
            for row in state.files[state.next_file_index:end]:
                path = Path(str(row["path"]))
                parent = self.ensure_folder_item(
                    path.parent, state.roots, state.root_items, state.items_by_path
                )
                item = self.add_document_item(parent, row)
                if state.preferred_file_id == int(row["id"]):
                    self.document_tree.setCurrentItem(item)
        finally:
            self.document_tree.setUpdatesEnabled(True)
        state.next_file_index = end
        self.document_status_label.setText(f"Documenten laden... {end}/{len(state.files)}")
        if end < len(state.files):
            QTimer.singleShot(0, self.populate_next_document_batch)
            return

        self._populate_state = None
        self._document_loading = False
        self.document_tree.blockSignals(False)
        self.document_tree.expandToDepth(0)
        self.document_tree.setColumnWidth(0, 320)
        self.document_tree.setColumnWidth(1, 70)
        self.apply_document_filter()
        if self.selected_document() is None and not self.ledger:
            self.status_label.setText("Selecteer een document en start de extractie")
        self.update_actions()

    def add_folder_item(
        self,
        parent: QTreeWidget | QTreeWidgetItem,
        label: str,
        path: Path,
    ) -> QTreeWidgetItem:
        item = QTreeWidgetItem(parent)
        item.setText(0, label)
        item.setText(1, "map")
        item.setData(0, Qt.UserRole, {"kind": "folder", "path": str(path)})
        item.setToolTip(0, str(path))
        return item

    def ensure_folder_item(
        self,
        folder: Path,
        roots: list[Path],
        root_items: dict[str, QTreeWidgetItem],
        items_by_path: dict[str, QTreeWidgetItem],
    ) -> QTreeWidgetItem | QTreeWidget:
        key = FolderScopePanel.path_key(folder)
        existing = items_by_path.get(key)
        if existing is not None:
            return existing
        root = FolderScopePanel.best_root(folder, roots)
        if root is None:
            item = self.add_folder_item(self.document_tree, str(folder), folder)
            items_by_path[key] = item
            return item
        parent = root_items[FolderScopePanel.path_key(root)]
        current = root
        try:
            parts = folder.relative_to(root).parts
        except ValueError:
            parts = ()
        for part in parts:
            current = current / part
            current_key = FolderScopePanel.path_key(current)
            if current_key not in items_by_path:
                items_by_path[current_key] = self.add_folder_item(parent, part, current)
            parent = items_by_path[current_key]
        return parent

    def add_document_item(
        self, parent: QTreeWidget | QTreeWidgetItem, row: dict[str, object]
    ) -> QTreeWidgetItem:
        path = str(row["path"])
        chunk_count = int(row["active_chunk_count"])
        item = QTreeWidgetItem(parent)
        item.setText(0, Path(path).name)
        item.setText(1, str(row["source_type"]))
        item.setText(2, f"{chunk_count} chunks")
        item.setData(
            0,
            Qt.UserRole,
            {
                "kind": "document",
                "file_id": int(row["id"]),
                "path": path,
                "chunk_count": chunk_count,
            },
        )
        item.setToolTip(0, path)
        return item

    def apply_document_filter(self) -> None:
        query = self.document_search.text().strip().casefold()
        visible = 0
        for index in range(self.document_tree.topLevelItemCount()):
            visible += self.filter_document_item(
                self.document_tree.topLevelItem(index), query, ancestor_matches=False
            )
        if self._document_loading:
            return
        if query:
            self.document_status_label.setText(
                f"{visible} van {self._document_count} documenten zichtbaar"
            )
        elif self.investigation_id is not None:
            self.document_status_label.setText(f"{self._document_count} documenten")

    def filter_document_item(
        self, item: QTreeWidgetItem, query: str, *, ancestor_matches: bool
    ) -> int:
        data = item.data(0, Qt.UserRole)
        path = str(data.get("path", "")) if isinstance(data, dict) else ""
        own_matches = not query or query in item.text(0).casefold() or query in path.casefold()
        matches_context = ancestor_matches or own_matches
        if isinstance(data, dict) and data.get("kind") == "document":
            visible = matches_context
            item.setHidden(not visible)
            return int(visible)
        visible_documents = 0
        for child_index in range(item.childCount()):
            visible_documents += self.filter_document_item(
                item.child(child_index), query, ancestor_matches=matches_context
            )
        item.setHidden(not (own_matches or visible_documents))
        if query and visible_documents:
            item.setExpanded(True)
        return visible_documents

    def selected_document(self) -> dict[str, object] | None:
        item = self.document_tree.currentItem()
        data = item.data(0, Qt.UserRole) if item is not None else None
        if isinstance(data, dict) and data.get("kind") == "document":
            return data
        return None

    def selected_file_id(self) -> int | None:
        selected = self.selected_document()
        return int(selected["file_id"]) if selected is not None else None

    def selection_changed(self) -> None:
        if self.thread is None:
            self.clear_result()
            selected = self.selected_document()
            if selected is None:
                self.status_label.setText("Selecteer een document en start de extractie")
            else:
                self.status_label.setText(
                    f"Geselecteerd: {Path(str(selected['path'])).name} "
                    f"({selected['chunk_count']} chunks)"
                )
        self.update_actions()

    def document_double_clicked(self, _item: QTreeWidgetItem, _column: int) -> None:
        if self.start_button.isEnabled() and self.selected_document() is not None:
            self.start_extraction()

    def clear_result(self) -> None:
        self.ledger = []
        self.failures = []
        self.model_outputs = []
        self.document_path = ""
        self._run_complete = False
        self._preview_chunk_ids = []
        self._preview_chunk_position = -1
        self.table.setRowCount(0)
        self.chunk_preview.clear()
        self.failure_label.clear()
        self.failure_label.setVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

    def start_extraction(self) -> None:
        selected = self.selected_document()
        model_name = self.model_edit.text().strip()
        if self.investigation_id is None or selected is None or self.thread is not None:
            return
        if not model_name:
            QMessageBox.warning(self, "Feitenoverzicht", "Vul een geïnstalleerd Ollama-model in.")
            return
        self.clear_result()
        self.document_path = str(selected["path"])
        batch_count = len(
            overlapping_chunk_batches([None] * int(selected["chunk_count"]))
        )
        self.progress_bar.setRange(0, batch_count + 1)
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Verbinden met Ollama-model {model_name}...")
        self._run_id += 1
        run_id = self._run_id

        thread = QThread(self)
        worker = FactLedgerWorker(
            self.config,
            self.db,
            int(selected["file_id"]),
            self.document_path,
            model_name,
        )
        self.thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda value, maximum, message: self.apply_progress_for_run(
                run_id, value, maximum, message
            )
        )
        worker.ledger_updated.connect(lambda ledger: self.apply_ledger_for_run(run_id, ledger))
        worker.batch_failed.connect(lambda failure: self.apply_failure_for_run(run_id, failure))
        worker.model_output_received.connect(
            lambda output: self.apply_model_output_for_run(run_id, output)
        )
        worker.finished.connect(
            lambda ledger, error, cancelled: self.extraction_finished_for_run(
                run_id, ledger, error, cancelled
            )
        )
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.update_actions()
        thread.start()

    def apply_progress(self, value: int, maximum: int, message: str) -> None:
        self.progress_bar.setRange(0, max(1, maximum))
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def apply_progress_for_run(self, run_id: int, value: int, maximum: int, message: str) -> None:
        if run_id == self._run_id:
            self.apply_progress(value, maximum, message)

    def apply_ledger(self, ledger: object) -> None:
        if not isinstance(ledger, list):
            return
        items = [item for item in ledger if isinstance(item, FactLedgerItem)]
        self.populate_table(items)

    def apply_ledger_for_run(self, run_id: int, ledger: object) -> None:
        if run_id == self._run_id:
            self.apply_ledger(ledger)

    def apply_failure_for_run(self, run_id: int, failure: object) -> None:
        if run_id != self._run_id or not isinstance(failure, FactLedgerFailure):
            return
        self.failures.append(failure)
        self.refresh_failure_label()

    def apply_model_output_for_run(self, run_id: int, output: object) -> None:
        if run_id == self._run_id and isinstance(output, FactLedgerModelOutput):
            self.model_outputs.append(output)

    def refresh_failure_label(self) -> None:
        if not self.failures:
            self.failure_label.clear()
            self.failure_label.setVisible(False)
            return
        descriptions = []
        for failure in self.failures:
            chunks = ", ".join(str(number) for number in failure.chunk_numbers)
            if failure.stage == "extractie":
                descriptions.append(
                    f"Ronde {failure.batch_number}, chunks {chunks}: na twee ongeldige antwoorden overgeslagen"
                )
            else:
                descriptions.append(
                    "Eindcontrole: na twee ongeldige antwoorden is het voorlopige overzicht behouden"
                )
        self.failure_label.setText("Let op — " + "; ".join(descriptions))
        self.failure_label.setToolTip("\n".join(failure.error for failure in self.failures))
        self.failure_label.setVisible(True)

    def populate_table(self, items: list[FactLedgerItem]) -> None:
        self.chunk_preview.clear()
        self._preview_chunk_ids = []
        self._preview_chunk_position = -1
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(len(items))
            for row, item in enumerate(items):
                number = QTableWidgetItem()
                number.setData(Qt.DisplayRole, row + 1)
                self.table.setItem(row, 0, number)
                statement_item = QTableWidgetItem(item.statement)
                statement_item.setData(Qt.UserRole, list(item.source_chunk_ids))
                self.table.setItem(row, 1, statement_item)
                self.table.setItem(row, 2, QTableWidgetItem("; ".join(item.source_locations)))
                self.table.setItem(row, 3, QTableWidgetItem("; ".join(item.source_chunk_ids)))
        finally:
            self.table.setUpdatesEnabled(True)

    def preview_selected_ledger_source(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self._preview_chunk_ids = []
            self._preview_chunk_position = -1
            self.chunk_preview.clear()
            return
        statement_item = self.table.item(rows[0].row(), 1)
        source_ids = statement_item.data(Qt.UserRole) if statement_item is not None else None
        self._preview_chunk_ids = [str(value) for value in source_ids] if isinstance(source_ids, list) else []
        self._preview_chunk_position = 0 if self._preview_chunk_ids else -1
        self.show_preview_chunk()

    def move_preview_chunk(self, offset: int) -> None:
        if not self._preview_chunk_ids:
            return
        target = self._preview_chunk_position + offset
        if 0 <= target < len(self._preview_chunk_ids):
            self._preview_chunk_position = target
            self.show_preview_chunk()

    def show_preview_chunk(self) -> None:
        if not (0 <= self._preview_chunk_position < len(self._preview_chunk_ids)):
            self.chunk_preview.clear()
            return
        chunk_id = self._preview_chunk_ids[self._preview_chunk_position]
        try:
            chunk = self.db.get_chunk(chunk_id)
        except Exception as exc:
            self.chunk_preview.setPlainText(f"Chunk kon niet worden geladen: {exc}")
            return
        if chunk is None:
            self.chunk_preview.setPlainText("Deze chunk is niet meer beschikbaar.")
            return
        try:
            metadata = json.loads(chunk["metadata_json"] or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        self.chunk_preview.set_chunk(
            str(chunk["chunk_text"]),
            chunk["display_markdown"],
            path=str(chunk["path"]),
            chunk_index=int(chunk["chunk_index"]),
            metadata=metadata,
            has_previous=self._preview_chunk_position > 0,
            has_next=self._preview_chunk_position + 1 < len(self._preview_chunk_ids),
        )

    def extraction_finished(self, ledger: object, error: str, cancelled: bool) -> None:
        if cancelled:
            self.status_label.setText("Extractie gestopt; het onvolledige resultaat is niet exporteerbaar")
            return
        if error:
            self.status_label.setText("Extractie mislukt")
            QMessageBox.critical(self, "Feitenoverzicht", error)
            return
        self.ledger = [item for item in ledger if isinstance(item, FactLedgerItem)] if isinstance(ledger, list) else []
        self._run_complete = True
        self.populate_table(self.ledger)
        self.progress_bar.setValue(self.progress_bar.maximum())
        failure_suffix = (
            f"; {len(self.failures)} waarschuwing{'en' if len(self.failures) != 1 else ''}"
            if self.failures
            else ""
        )
        self.status_label.setText(
            f"Gereed: {len(self.ledger)} verklaringen uit {Path(self.document_path).name}"
            f"{failure_suffix}"
        )

    def extraction_finished_for_run(
        self, run_id: int, ledger: object, error: str, cancelled: bool
    ) -> None:
        if run_id == self._run_id:
            self.extraction_finished(ledger, error, cancelled)

    def stop_extraction(self) -> None:
        if self.worker is None:
            return
        self.worker.cancel()
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stoppen na het huidige Ollama-verzoek...")

    def thread_finished(self) -> None:
        self.thread = None
        self.worker = None
        self.update_actions()

    def export_excel(self) -> None:
        if not self._run_complete or not self.document_path:
            return
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Feitenoverzicht exporteren",
            default_export_filename(self.document_path),
            "Excel-werkmap (*.xlsx)",
        )
        if not destination:
            return
        if not destination.lower().endswith(".xlsx"):
            destination += ".xlsx"
        try:
            exported = export_fact_ledger(
                destination,
                self.document_path,
                self.ledger,
                self.failures,
                self.model_outputs,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Excel-export", str(exc))
            return
        self.status_label.setText(f"Geëxporteerd naar {exported}")

    def update_actions(self) -> None:
        running = self.thread is not None
        has_document = self.selected_document() is not None
        documents_available = self.investigation_id is not None and not self._document_loading
        self.document_tree.setEnabled(not running and documents_available)
        self.document_search.setEnabled(not running and documents_available)
        self.model_edit.setEnabled(not running and self.investigation_id is not None)
        self.start_button.setEnabled(not running and not self._document_loading and has_document)
        self.stop_button.setEnabled(running)
        self.refresh_button.setEnabled(not running and documents_available)
        self.export_button.setEnabled(
            not running and self._run_complete and bool(self.document_path)
        )

    def shutdown(self) -> None:
        self._document_load_request_id += 1
        self._populate_state = None
        if self.thread is None:
            return
        if self.worker is not None:
            self.worker.cancel()
        self.thread.requestInterruption()
        self.thread.quit()
        self.thread.wait(2000)
