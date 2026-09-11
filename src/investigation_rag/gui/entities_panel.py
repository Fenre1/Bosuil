from __future__ import annotations

from threading import Event

from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.config import AppConfig
from investigation_rag.entities import (
    DEFAULT_ENTITY_LABELS,
    DEFAULT_ENTITY_MODEL,
    DEFAULT_ENTITY_NUM_CTX,
    DEFAULT_ENTITY_OUTPUT_TOKENS,
    EntityExtractionCancelled,
    extract_entities,
)
from investigation_rag.gui.checkable_type_filter import CheckableTypeFilter
from investigation_rag.gui.entity_graph import EntityGraphPanel
from investigation_rag.gui.entity_merge_dialog import EntityMergeDialog


class EntityExtractionWorker(QObject):
    progress = pyqtSignal(int, int)
    message = pyqtSignal(str)
    finished = pyqtSignal(int, str)

    def __init__(
        self,
        db: SQLiteDB,
        investigation_id: int,
        labels: list[str],
        model_name: str,
        config: AppConfig | None,
    ):
        super().__init__()
        self.db = db
        self.investigation_id = investigation_id
        self.labels = labels
        self.model_name = model_name
        self.config = config
        self.cancel_event = Event()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            count = extract_entities(
                self.db,
                self.investigation_id,
                self.labels,
                model_name=self.model_name,
                base_url=(
                    self.config.ollama_base_url
                    if self.config is not None
                    else "http://127.0.0.1:11434"
                ),
                allow_remote=(self.config.ollama_allow_remote if self.config is not None else False),
                timeout=(self.config.ollama_timeout_seconds if self.config is not None else 120),
                num_ctx=(self.config.entity_num_ctx if self.config is not None else DEFAULT_ENTITY_NUM_CTX),
                output_tokens=(
                    self.config.entity_output_tokens
                    if self.config is not None
                    else DEFAULT_ENTITY_OUTPUT_TOKENS
                ),
                progress=self.progress.emit,
                notice=self.message.emit,
                cancelled=self.cancel_event.is_set,
            )
            self.finished.emit(count, "")
        except EntityExtractionCancelled:
            self.finished.emit(-1, "")
        except Exception as exc:
            self.finished.emit(0, str(exc))


class EntitiesPanel(QWidget):
    def __init__(
        self,
        db: SQLiteDB,
        store: LanceDBStore | None = None,
        config: AppConfig | None = None,
    ):
        super().__init__()
        self.db = db
        self.config = config
        self.investigation_id: int | None = None
        self.thread: QThread | None = None
        self.worker: EntityExtractionWorker | None = None

        intro = QLabel(
            "Extract unique named entities with one local Qwen request per document, then "
            "link exact occurrences back to the indexed chunks."
        )
        intro.setWordWrap(True)
        self.model_edit = QLineEdit(
            config.entity_llm_model if config is not None else DEFAULT_ENTITY_MODEL
        )
        self.model_edit.setPlaceholderText(DEFAULT_ENTITY_MODEL)
        self.model_edit.setFixedHeight(34)
        self.model_edit.setMinimumWidth(180)
        self.labels_edit = QLineEdit(", ".join(DEFAULT_ENTITY_LABELS))
        self.labels_edit.setPlaceholderText("person, organization, location")
        self.labels_edit.setFixedHeight(34)
        self.extract_button = QPushButton("Extract entities")
        self.extract_button.setObjectName("PrimaryButton")
        self.extract_button.clicked.connect(self.start_extraction)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_extraction)
        self.stop_button.setEnabled(False)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.reload)
        self.merge_button = QPushButton("Merge selected")
        self.merge_button.clicked.connect(self.merge_selected)
        for button in (self.extract_button, self.stop_button, self.refresh_button, self.merge_button):
            button.setFixedHeight(34)
        self.status_label = QLabel("No investigation loaded")
        self.status_label.setObjectName("StatusLabel")

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(6)
        controls.addWidget(QLabel("Model:"))
        controls.addWidget(self.model_edit)
        controls.addWidget(QLabel("Entity types:"))
        controls.addWidget(self.labels_edit, 1)
        controls.addWidget(self.extract_button)
        controls.addWidget(self.stop_button)
        controls.addWidget(self.refresh_button)
        controls.addWidget(self.merge_button)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search entity names")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setFixedHeight(34)
        self.search_edit.textChanged.connect(self.apply_filters)
        self.type_filter = CheckableTypeFilter()
        self.type_filter.setFixedHeight(34)
        self.type_filter.setMinimumWidth(125)
        self.type_filter.selectionChanged.connect(self.apply_filters)
        self.minimum_mentions = QSpinBox()
        self.minimum_mentions.setRange(0, 999_999)
        self.minimum_mentions.setSpecialValueText("Any")
        self.minimum_mentions.setFixedHeight(34)
        self.minimum_mentions.setFixedWidth(90)
        self.minimum_mentions.valueChanged.connect(self.apply_filters)
        self.clear_filters_button = QPushButton("Clear filters")
        self.clear_filters_button.setObjectName("GhostButton")
        self.clear_filters_button.setFixedHeight(34)
        self.clear_filters_button.clicked.connect(self.clear_filters)

        filters = QHBoxLayout()
        filters.setContentsMargins(0, 0, 0, 0)
        filters.setSpacing(6)
        filters.addWidget(QLabel("Filter:"))
        filters.addWidget(self.search_edit, 1)
        filters.addWidget(QLabel("Type:"))
        filters.addWidget(self.type_filter)
        filters.addWidget(QLabel("Minimum mentions:"))
        filters.addWidget(self.minimum_mentions)
        filters.addWidget(self.clear_filters_button)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Entity", "Type", "Mentions", "Documents"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setSortIndicator(0, Qt.AscendingOrder)
        self.table.horizontalHeader().setSortIndicatorShown(True)
        self.table.horizontalHeader().setToolTip("Click a column heading to sort")
        self.table.horizontalHeader().sortIndicatorChanged.connect(
            lambda *_: QTimer.singleShot(0, self.sync_graph_context)
        )
        self.table.itemSelectionChanged.connect(self.update_actions)

        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        results_layout.setContentsMargins(0, 8, 0, 0)
        results_layout.setSpacing(8)
        results_layout.addWidget(self.status_label)
        results_layout.addWidget(self.table, 1)

        self.graph_panel = EntityGraphPanel(db, store)
        self.result_tabs = QTabWidget()
        self.result_tabs.setDocumentMode(True)
        self.result_tabs.addTab(results_page, "Results")
        self.result_tabs.addTab(self.graph_panel, "Graph")
        self.result_tabs.currentChanged.connect(self.result_tab_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 12)
        layout.setSpacing(6)
        layout.addWidget(intro)
        layout.addLayout(controls)
        layout.addLayout(filters)
        layout.addWidget(self.result_tabs, 1)
        self.update_actions()

    def set_investigation_id(self, investigation_id: int | None) -> None:
        changed = investigation_id != self.investigation_id
        self.investigation_id = investigation_id
        if changed:
            self.clear_filters()
        self.reload()

    def set_data_sources(self, db: SQLiteDB, store: LanceDBStore | None) -> None:
        self.db = db
        self.graph_panel.set_data_sources(db, store)

    def reload(self) -> None:
        sort_column = self.table.horizontalHeader().sortIndicatorSection()
        sort_order = self.table.horizontalHeader().sortIndicatorOrder()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if self.investigation_id is None:
            self.update_type_filter([])
            self.table.setSortingEnabled(True)
            self.status_label.setText("No investigation loaded")
            self.sync_graph_context()
            self.update_actions()
            return
        rows = self.db.list_entities(self.investigation_id)
        self.update_type_filter(sorted({str(row["entity_type"]) for row in rows}, key=str.casefold))
        self.table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            name_item = QTableWidgetItem(str(row["name"]))
            name_item.setData(Qt.UserRole, int(row["id"]))
            self.table.setItem(row_index, 0, name_item)
            self.table.setItem(row_index, 1, QTableWidgetItem(str(row["entity_type"])))
            mention_item = QTableWidgetItem()
            mention_item.setData(Qt.DisplayRole, int(row["mention_count"]))
            self.table.setItem(row_index, 2, mention_item)
            document_item = QTableWidgetItem()
            document_item.setData(Qt.DisplayRole, int(row["document_count"]))
            self.table.setItem(row_index, 3, document_item)
        self.table.setSortingEnabled(True)
        self.table.sortItems(sort_column, sort_order)
        self.table.resizeColumnsToContents()
        self.apply_filters()
        self.graph_panel.refresh()
        self.update_actions()

    def update_type_filter(self, entity_types: list[str]) -> None:
        self.type_filter.set_entity_types(entity_types)

    def apply_filters(self) -> None:
        query = self.search_edit.text().strip().casefold()
        selected_types = self.type_filter.selected_types()
        type_filter_active = self.type_filter.is_filter_active()
        minimum_mentions = self.minimum_mentions.value()
        visible_count = 0
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            type_item = self.table.item(row, 1)
            mentions_item = self.table.item(row, 2)
            name = name_item.text() if name_item is not None else ""
            row_type = type_item.text() if type_item is not None else ""
            mentions = int(mentions_item.data(Qt.DisplayRole)) if mentions_item is not None else 0
            visible = (
                query in name.casefold()
                and (not type_filter_active or row_type in selected_types)
                and mentions >= minimum_mentions
            )
            self.table.setRowHidden(row, not visible)
            visible_count += int(visible)

        if any(self.table.isRowHidden(index.row()) for index in self.table.selectionModel().selectedRows()):
            self.table.clearSelection()

        total_count = self.table.rowCount()
        filters_active = bool(query or type_filter_active or minimum_mentions)
        if self.investigation_id is None:
            self.status_label.setText("No investigation loaded")
        elif filters_active:
            self.status_label.setText(f"{visible_count} of {total_count} unique entities")
        else:
            self.status_label.setText(f"{total_count} unique entities")
        self.clear_filters_button.setEnabled(filters_active)
        self.sync_graph_context()
        self.update_actions()

    def visible_entity_ids(self) -> list[int]:
        entity_ids = []
        for row in range(self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            item = self.table.item(row, 0)
            if item is not None:
                entity_ids.append(int(item.data(Qt.UserRole)))
        return entity_ids

    def sync_graph_context(self) -> None:
        self.graph_panel.set_context(self.investigation_id, self.visible_entity_ids())

    def result_tab_changed(self, index: int) -> None:
        self.sync_graph_context()
        self.graph_panel.set_active(self.result_tabs.widget(index) is self.graph_panel)

    def clear_filters(self) -> None:
        self.search_edit.clear()
        self.type_filter.select_all()
        self.minimum_mentions.setValue(0)
        self.apply_filters()

    def start_extraction(self) -> None:
        if self.investigation_id is None or self.thread is not None:
            return
        labels = [value.strip() for value in self.labels_edit.text().split(",") if value.strip()]
        if not labels:
            QMessageBox.warning(self, "Entities", "Enter at least one entity type.")
            return
        model_name = self.model_edit.text().strip()
        if not model_name:
            QMessageBox.warning(self, "Entities", "Enter an installed Ollama model.")
            return
        thread = QThread(self)
        worker = EntityExtractionWorker(
            self.db,
            self.investigation_id,
            labels,
            model_name,
            self.config,
        )
        self.thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(
            lambda done, total: self.status_label.setText(
                f"Extracting entities: {done}/{total} documents"
            )
        )
        worker.message.connect(self.status_label.setText)
        worker.finished.connect(self.extraction_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(self.thread_finished)
        thread.finished.connect(thread.deleteLater)
        self.extract_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.model_edit.setEnabled(False)
        self.status_label.setText(f"Connecting to Ollama model {model_name}...")
        thread.start()

    def extraction_finished(self, mention_count: int, error: str) -> None:
        if mention_count < 0:
            self.status_label.setText("Entity extraction stopped; previous results were preserved")
        elif error:
            QMessageBox.critical(self, "Entity extraction", error)
            self.status_label.setText("Entity extraction failed")
        else:
            self.reload()
            self.status_label.setText(
                f"Extraction complete: {mention_count} mentions, {self.table.rowCount()} unique entities"
            )

    def stop_extraction(self) -> None:
        if self.worker is None:
            return
        self.worker.cancel()
        self.stop_button.setEnabled(False)
        self.status_label.setText(
            "Stopping after the current Ollama document request completes..."
        )

    def thread_finished(self) -> None:
        self.thread = None
        self.worker = None
        self.model_edit.setEnabled(True)
        self.update_actions()

    def selected_entity_ids(self) -> list[int]:
        return [entity_id for entity_id, _, _ in self.selected_entities()]

    def selected_entities(self) -> list[tuple[int, str, str]]:
        entities = []
        for index in sorted(self.table.selectionModel().selectedRows(), key=lambda value: value.row()):
            item = self.table.item(index.row(), 0)
            type_item = self.table.item(index.row(), 1)
            if item is not None and type_item is not None:
                entities.append((int(item.data(Qt.UserRole)), item.text(), type_item.text()))
        return entities

    def merge_selected(self) -> None:
        if self.investigation_id is None:
            return
        selected = self.selected_entities()
        if len(selected) < 2:
            return
        dialog = EntityMergeDialog([(name, entity_type) for _, name, entity_type in selected], self)
        if not dialog.exec_():
            return
        name, entity_type = dialog.values()
        try:
            self.db.merge_entities_as(
                self.investigation_id,
                [entity_id for entity_id, _, _ in selected],
                name,
                entity_type,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Merge entities", str(exc))
            return
        self.reload()

    def update_actions(self) -> None:
        running = self.thread is not None
        available = self.investigation_id is not None and not running
        self.extract_button.setEnabled(available)
        self.stop_button.setEnabled(running)
        self.refresh_button.setEnabled(available)
        self.merge_button.setEnabled(available and len(self.selected_entity_ids()) >= 2)
        self.model_edit.setEnabled(available)

    def shutdown(self) -> None:
        self.graph_panel.shutdown()
        if self.thread is not None:
            if self.worker is not None:
                self.worker.cancel()
            self.thread.quit()
            self.thread.wait(2000)
