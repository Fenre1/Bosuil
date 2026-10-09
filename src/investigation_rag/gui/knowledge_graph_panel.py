from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from threading import Event

from PyQt5.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog,
    QFormLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QPlainTextEdit, QPushButton, QSpinBox, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from investigation_rag.config import AppConfig
from investigation_rag.db.sqlite_db import SQLiteDB, utc_now
from investigation_rag.gui.chunk_preview import ChunkPreview
from investigation_rag.gui.knowledge_graph_view import KnowledgeGraphView
from investigation_rag.knowledge_graph import (
    KGModelClient, KGSettings, KGStore, NODE_TYPES, RELATIONS, RELATION_GROUPS, build_graph, settings_from_saved,
)


class KGWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(object, str)

    def __init__(self, config: AppConfig, store: KGStore, investigation_id: int | None,
                 file_ids: list[int], settings: KGSettings, force: bool = False, models: bool = False,
                 import_path: str | None = None):
        super().__init__()
        self.config, self.store, self.investigation_id = config, store, investigation_id
        self.file_ids, self.settings, self.force, self.models = file_ids, settings, force, models
        self.import_path = import_path
        self.cancel_event = Event()

    def cancel(self):
        self.cancel_event.set()

    def run(self):
        client = None
        try:
            if self.import_path:
                payload = json.loads(Path(self.import_path).read_text(encoding="utf-8-sig"))
                result = self.store.import_graph(self.investigation_id, payload, cancel=self.cancel_event, progress=self.progress.emit)
                self.finished.emit(result, "")
                return
            client = KGModelClient(self.config.ollama_base_url, timeout=self.settings.request_seconds,
                                   allow_remote=self.config.ollama_allow_remote)
            if self.models:
                self.finished.emit(client.list_models(), "")
            else:
                result = build_graph(self.store, self.investigation_id, self.file_ids, self.settings,
                                     client, self.cancel_event, self.progress.emit, force=self.force)
                self.finished.emit(result, "")
        except Exception as exc:
            self.finished.emit(None, str(exc))
        finally:
            if client is not None:
                client.close()


class KnowledgeGraphPanel(QWidget):
    def __init__(self, config: AppConfig, db: SQLiteDB):
        super().__init__()
        self.config, self.db, self.store = config, db, KGStore(db)
        self.investigation_id: int | None = None
        self.thread: QThread | None = None
        self.worker: KGWorker | None = None
        self._generation = 0
        self._seconds = 0
        self._worker_generation = 0
        self._claims: list[dict] = []
        self._visible: list[dict] = []
        self._orphan_nodes: list[dict] = []
        self._selected: dict | None = None
        self._relations = tuple(RELATIONS)

        layout = QVBoxLayout(self)
        intro = QLabel("Build source-backed relationships with a separate local extraction pipeline. "
                       "Model suggestions require review. Select documents, then build.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        model_row = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.setMinimumWidth(180)
        self.model.addItem(config.ollama_llm_model or KGSettings().model)
        self.refresh_models_button = QPushButton("Refresh models")
        self.refresh_models_button.clicked.connect(self.refresh_models)
        self.settings_button = QPushButton("Extraction settings")
        self.settings_button.clicked.connect(self.edit_settings)
        self.diagnostics_button = QPushButton("Diagnostics")
        self.diagnostics_button.clicked.connect(self.show_diagnostics)
        self.build_button = QPushButton("Build graph")
        self.build_button.setToolTip("Build relationships from the selected visible documents only")
        self.build_button.setObjectName("PrimaryButton")
        self.build_button.clicked.connect(self.start_build)
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop)
        model_row.addWidget(QLabel("Model"))
        model_row.addWidget(self.model, 1)
        for button in (self.refresh_models_button, self.settings_button, self.build_button, self.stop_button):
            model_row.addWidget(button)
        layout.addLayout(model_row)
        recommendation = QLabel('Use a current local instruction model and compare source accuracy on a small sample. '
                                'Research guides the extraction method, not a ranking of current models. '
                                '<a href="https://aclanthology.org/2025.emnlp-main.783/">Research</a>')
        recommendation.setOpenExternalLinks(True)
        recommendation.setWordWrap(True)
        layout.addWidget(recommendation)

        defaults = KGSettings()
        self.num_ctx, self.output_tokens, self.request_seconds = defaults.num_ctx, defaults.output_tokens, defaults.request_seconds
        self.verify, self.thinking = True, "off"
        self.force = QCheckBox("Reprocess cached windows")
        self.status = QLabel("Load an investigation to begin.")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        status_row = QHBoxLayout()
        status_row.addWidget(self.status, 1)
        status_row.addWidget(self.force)
        layout.addLayout(status_row)

        split = QSplitter(Qt.Horizontal)
        sources = QWidget()
        source_layout = QVBoxLayout(sources)
        source_layout.addWidget(QLabel("Indexed documents"))
        self.document_search = QLineEdit()
        self.document_search.setPlaceholderText("Filter documents…")
        self.document_search.textChanged.connect(self.filter_documents)
        source_layout.addWidget(self.document_search)
        self.documents = QListWidget()
        self.documents.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.documents.itemSelectionChanged.connect(self.update_actions)
        source_layout.addWidget(self.documents)
        source_buttons = QHBoxLayout()
        select = QPushButton("Select visible")
        select.clicked.connect(self.select_visible_documents)
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        source_buttons.addWidget(select)
        source_buttons.addWidget(refresh)
        source_layout.addLayout(source_buttons)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        filter_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find an entity, relationship or quote…")
        self.search.textChanged.connect(self.apply_filters)
        self.review_filter = QComboBox()
        self.review_filter.addItems(["Suggested + accepted", "Accepted", "Suggested", "Rejected", "All"])
        self.review_filter.currentIndexChanged.connect(self.apply_filters)
        self.focus = QComboBox()
        self.focus.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.focus.setMinimumContentsLength(12)
        self.focus.setToolTip("Focus the graph and table on one node; All nodes clears the focus.")
        self.focus.addItem("All nodes", None)
        self.focus.currentIndexChanged.connect(self.apply_filters)
        filter_row.addWidget(self.search, 2)
        filter_row.addWidget(self.review_filter)
        filter_row.addWidget(self.focus, 1)
        workspace_layout.addLayout(filter_row)

        self.views = QTabWidget()
        self.views.addTab(sources, "Documents")
        graph_page = QWidget()
        graph_layout = QVBoxLayout(graph_page)
        self.graph = KnowledgeGraphView()
        self.graph.node_selected.connect(self.focus_node)
        self.graph.claim_selected.connect(self.select_claim)
        self.graph_status = QLabel()
        self.graph_status.setWordWrap(True)
        graph_layout.addWidget(self.graph, 1)
        display_tools = QHBoxLayout()
        self.node_limit = QSpinBox()
        self.node_limit.setRange(0, 1)  # Updated to the available node count when the graph is drawn.
        self.node_limit.setSpecialValueText("All")
        self.node_limit.setValue(0)
        self.node_limit.setToolTip("Maximum matching nodes to display. All removes the display limit; the relationship table stays complete.")
        self.node_limit.setKeyboardTracking(False)
        self.node_limit.valueChanged.connect(self.change_graph_display)
        self.graph_layout_choice = QComboBox()
        self.graph_layout_choice.addItem("Connected groups", "connected")
        self.graph_layout_choice.addItem("Compact grid", "grid")
        self.graph_layout_choice.currentIndexChanged.connect(self.change_graph_layout)
        self.arrange_button = QPushButton("Rearrange")
        self.arrange_button.setToolTip("Reset dragged positions and arrange nodes using the selected layout")
        self.arrange_button.clicked.connect(self.rearrange_graph)
        fit = QPushButton("Fit graph")
        fit.clicked.connect(self.graph.fit_graph)
        display_tools.addWidget(QLabel("Nodes shown"))
        display_tools.addWidget(self.node_limit)
        display_tools.addWidget(self.graph_layout_choice, 1)
        display_tools.addWidget(self.arrange_button)
        display_tools.addWidget(fit)
        graph_layout.addLayout(display_tools)
        graph_tools = QHBoxLayout()
        graph_tools.addWidget(self.graph_status, 1)
        graph_layout.addLayout(graph_tools)
        self.views.addTab(graph_page, "Graph")

        self.table = QTableWidget(0, 9)
        self.table.setHorizontalHeaderLabels(["Subject", "Relationship", "Object / value", "Property", "Polarity", "Attributed to", "Time", "Review", "Source"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.preview_selection)
        self.views.addTab(self.table, "Relationships")
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(2000)
        self.log_page = QWidget()
        log_layout = QVBoxLayout(self.log_page)
        log_layout.addWidget(self.log, 1)
        log_layout.addWidget(self.diagnostics_button, 0, Qt.AlignRight)
        self.views.addTab(self.log_page, "Build log")
        workspace_layout.addWidget(self.views, 1)
        actions = QHBoxLayout()
        self.accept_button = QPushButton("Accept")
        self.reject_button = QPushButton("Reject")
        self.reset_button = QPushButton("Reset review")
        self.edit_button = QPushButton("Edit claim")
        self.merge_button = QPushButton("Merge nodes")
        self.node_button = QPushButton("Edit node")
        self.export_button = QPushButton("Export graph")
        self.import_button = QPushButton("Import graph")
        self.accept_button.clicked.connect(lambda: self.review("accepted"))
        self.reject_button.clicked.connect(lambda: self.review("rejected"))
        self.reset_button.clicked.connect(lambda: self.review("suggested"))
        self.edit_button.clicked.connect(self.edit_claim)
        self.merge_button.clicked.connect(self.merge_nodes)
        self.node_button.clicked.connect(self.edit_node)
        self.export_button.clicked.connect(self.export_graph)
        self.import_button.clicked.connect(self.import_graph)
        for button in (self.accept_button, self.reject_button, self.reset_button, self.edit_button):
            actions.addWidget(button)
        workspace_layout.addLayout(actions)
        tools = QHBoxLayout()
        for button in (self.node_button, self.merge_button, self.import_button, self.export_button):
            tools.addWidget(button)
        workspace_layout.addLayout(tools)
        split.addWidget(workspace)

        inspector = QWidget()
        inspector_layout = QVBoxLayout(inspector)
        self.claim_details = QPlainTextEdit()
        self.claim_details.setReadOnly(True)
        self.claim_details.setMaximumHeight(210)
        self.claim_details.setPlaceholderText("Select an edge or relationship to inspect its exact evidence quote.")
        inspector_layout.addWidget(self.claim_details)
        self.preview = ChunkPreview("Select a relationship to view its source", db=db)
        self.preview.set_mode("readable")
        inspector_layout.addWidget(self.preview, 1)
        split.addWidget(inspector)
        split.setSizes([650, 390])
        layout.addWidget(split, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.tick)
        self.views.currentChanged.connect(self.graph_page_changed)
        self.update_actions()

    def settings(self) -> KGSettings:
        return KGSettings(self.model.currentText().strip(), self.num_ctx, self.output_tokens,
                          self.request_seconds, self.verify, self.thinking, self._relations)

    def set_data_sources(self, config: AppConfig, db: SQLiteDB):
        self.config, self.db, self.store = config, db, KGStore(db)
        self.preview.db = db

    def set_investigation_id(self, investigation_id: int | None):
        self._generation += 1
        if self.worker:
            self.worker.cancel()
        self.investigation_id = investigation_id
        self._selected = None
        self.graph.positions.clear()
        self.log.clear()
        self.status.setText("Select documents to build their knowledge graph." if investigation_id else "Load an investigation to begin.")
        if investigation_id is not None:
            with self.db.connection() as conn:
                row = conn.execute("SELECT value FROM settings WHERE key='knowledge_graph_settings'").fetchone()
            if row:
                try:
                    data = json.loads(row[0])
                    settings = settings_from_saved(data)
                    self.apply_settings(settings)
                except (ValueError, KeyError, TypeError):
                    self.apply_settings(KGSettings(model=self.config.ollama_llm_model or KGSettings().model))
            else:
                self.apply_settings(KGSettings(model=self.config.ollama_llm_model or KGSettings().model))
        self.load_graph_display()
        self.reload()

    def apply_settings(self, settings: KGSettings):
        self.model.setCurrentText(settings.model)
        self.num_ctx, self.output_tokens = settings.num_ctx, settings.output_tokens
        self.request_seconds, self.verify = settings.request_seconds, settings.verify
        self.thinking, self._relations = settings.thinking, settings.relations

    def reload(self):
        selected = {i.data(Qt.UserRole) for i in self.documents.selectedItems()}
        self.documents.clear()
        if self.investigation_id is not None:
            for row in self.store.documents(self.investigation_id):
                item = QListWidgetItem(Path(row["path"]).name + f" ({row['passages']} passages)")
                item.setToolTip(row["path"])
                item.setData(Qt.UserRole, row["id"])
                self.documents.addItem(item)
                item.setSelected(row["id"] in selected)
        self.filter_documents()
        self.reload_claims()

    def filter_documents(self):
        text = self.document_search.text().casefold()
        for index in range(self.documents.count()):
            item = self.documents.item(index)
            item.setHidden(text not in item.toolTip().casefold())
        self.update_actions()

    def select_visible_documents(self):
        for index in range(self.documents.count()):
            item = self.documents.item(index)
            item.setSelected(not item.isHidden())

    def reload_claims(self):
        selected_id = self._selected["id"] if self._selected else None
        self._claims = self.store.claims(self.investigation_id) if self.investigation_id is not None else []
        self._orphan_nodes = self.store.orphan_nodes(self.investigation_id) if self.investigation_id is not None else []
        focus_id = self.focus.currentData()
        self.focus.blockSignals(True)
        self.focus.clear()
        self.focus.addItem("All nodes", None)
        names = {}
        for claim in self._claims:
            for role in ("subject", "object"):
                names[claim[role + "_id"]] = f"{claim[role]} · {claim[role + '_type']} · #{claim[role + '_id']}"
        for node in self._orphan_nodes:
            names[node["id"]] = f"{node['name']} · {node['node_type']} · #{node['id']}"
        for node_id, name in sorted(names.items(), key=lambda entry: entry[1].casefold()):
            self.focus.addItem(name, node_id)
        self.focus.setCurrentIndex(max(0, self.focus.findData(focus_id)))
        self.focus.blockSignals(False)
        self.apply_filters()
        if selected_id is not None:
            self.select_claim(selected_id, switch=False)

    def apply_filters(self):
        text, focus = self.search.text().casefold(), self.focus.currentData()
        allowed = {0: {"suggested", "accepted"}, 1: {"accepted"}, 2: {"suggested"}, 3: {"rejected"},
                   4: {"suggested", "accepted", "rejected"}}[self.review_filter.currentIndex()]
        self._visible = [c for c in self._claims if c["review_status"] in allowed and
                         (focus is None or focus in (c["subject_id"], c["object_id"])) and
                         (not text or text in " ".join(str(c[k]) for k in ("subject", "predicate", "object", "quote", "path", "attribution", "time_text", "attribute")).casefold())]
        self.table.blockSignals(True)
        self.table.setRowCount(len(self._visible))
        for row, claim in enumerate(self._visible):
            for col, value in enumerate([claim["subject"], claim["predicate"], claim["object"], claim["attribute"], claim["polarity"],
                                         claim["attribution"], claim["time_text"], claim["review_status"], Path(claim["path"]).name +
                                         (" [imported quote]" if claim.get("imported") and not claim["source_current"] else "")]):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.UserRole, claim["id"])
                item.setToolTip(str(value))
                self.table.setItem(row, col, item)
        self.table.clearSelection()
        self.table.blockSignals(False)
        self._selected = None
        self.claim_details.clear()
        self.preview.clear()
        self.redraw_graph()
        self.update_actions()

    def redraw_graph(self):
        text, focus = self.search.text().casefold(), self.focus.currentData()
        orphans = [node for node in self._orphan_nodes if (focus is None or node["id"] == focus) and
                   (not text or text in node["name"].casefold() or text in node["node_type"].casefold())]
        total = len({c[role + "_id"] for c in self._visible for role in ("subject", "object")} | {n["id"] for n in orphans})
        self.node_limit.blockSignals(True)
        self.node_limit.setMaximum(max(1, total, self.node_limit.value()))
        self.node_limit.blockSignals(False)
        n, e = self.graph.draw_claims(self._visible, self.node_limit.value(), layout=self.graph_layout_choice.currentData(), extra_nodes=orphans)
        self.graph_status.setText(f"Showing {n} of {total} matching nodes / {e} labelled edges. {len(self._visible)} source claims in table.")

    def load_graph_display(self):
        saved = {}
        if self.investigation_id is not None:
            with self.db.connection() as conn:
                row = conn.execute("SELECT value FROM settings WHERE key='knowledge_graph_view_settings'").fetchone()
            if row:
                try:
                    saved = json.loads(row[0])
                    if not isinstance(saved, dict):
                        saved = {}
                except ValueError:
                    pass
        limit = saved.get("node_limit", 0)
        if type(limit) is not int or not 0 <= limit <= 2147483647:
            limit = 0
        self.node_limit.blockSignals(True)
        self.graph_layout_choice.blockSignals(True)
        self.node_limit.setMaximum(max(1, limit))
        self.node_limit.setValue(limit)
        self.graph_layout_choice.setCurrentIndex(max(0, self.graph_layout_choice.findData(saved.get("layout", "connected"))))
        self.node_limit.blockSignals(False)
        self.graph_layout_choice.blockSignals(False)

    def save_graph_display(self):
        if self.investigation_id is not None:
            with self.db.transaction() as conn:
                conn.execute("INSERT OR REPLACE INTO settings VALUES('knowledge_graph_view_settings',?,?)",
                             (json.dumps({"node_limit": self.node_limit.value(), "layout": self.graph_layout_choice.currentData()}), utc_now()))

    def change_graph_display(self):
        self.save_graph_display()
        self.redraw_graph()
        self.graph.fit_graph()

    def change_graph_layout(self):
        self.save_graph_display()
        self.rearrange_graph()

    def rearrange_graph(self):
        self.graph.positions.clear()
        self.redraw_graph()
        self.graph.fit_graph()

    def graph_page_changed(self, index):
        if self.views.widget(index) is not self.graph.parentWidget():
            return
        generation = self._generation
        def fit_visible_graph():
            if generation == self._generation and self.views.currentWidget() is self.graph.parentWidget():
                # Tab activation gives the layout its actual viewport dimensions.
                self.redraw_graph()
                self.graph.fit_graph()
        QTimer.singleShot(0, fit_visible_graph)

    def focus_node(self, node_id: int):
        index = self.focus.findData(node_id)
        if index >= 0:
            self.focus.setCurrentIndex(index)

    def select_claim(self, claim_id: int, switch: bool = True):
        for row, claim in enumerate(self._visible):
            if claim["id"] == claim_id:
                self.table.selectRow(row)
                if switch:
                    self.views.setCurrentWidget(self.table)
                break

    def preview_selection(self):
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            self._selected = None
            self.update_actions()
            return
        claim = self._visible[selected[0].row()]
        self._selected = claim
        self.preview.clear()
        evidence = ""
        if claim.get("imported"):
            evidence = ("Imported claim · linked indexed evidence" if claim["source_current"] else
                        "Imported quote · local source unavailable or changed")
            evidence += f"\nOriginal source: {claim['path']}\n"
        self.claim_details.setPlainText(
            f"{claim['subject']} → {claim['predicate']} → {claim['object']}\n"
            f"{claim['polarity']} · {claim['review_status']} · {claim['verification']}\n"
            f"Property: {claim['attribute'] or 'not stated'}\n"
            f"Attributed to: {claim['attribution'] or 'not stated'}\nTime: {claim['time_text'] or 'not stated'}\n"
            f"Model: {claim['model'] or 'not recorded'} · run {claim['run_id'] if claim['run_id'] is not None else 'not recorded'}\n"
            f"{evidence}\n{claim['quote']}\n\n{claim['review_note']}")
        chunk = self.db.get_chunk(claim["chunk_id"]) if claim["source_current"] else None
        if chunk:
            metadata = json.loads(chunk["metadata_json"] or "{}")
            metadata.update(file_id=chunk["file_id"], chunk_id=chunk["id"])
            metadata.setdefault("source_content_hash", chunk["content_hash"])
            self.preview.set_chunk(chunk["chunk_text"], chunk["display_markdown"], path=chunk["path"],
                                   chunk_index=chunk["chunk_index"], metadata=metadata)
        elif claim.get("imported"):
            self.preview.setPlainText("Imported evidence quote\n\n" + claim["quote"])
        self.update_actions()

    def edit_settings(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Knowledge graph extraction settings")
        form = QFormLayout(dialog)
        context, output, timeout = QSpinBox(), QSpinBox(), QSpinBox()
        for field, low, high, value in ((context, 4096, 131072, self.num_ctx), (output, 512, 8192, self.output_tokens),
                                      (timeout, 15, 900, self.request_seconds)):
            field.setRange(low, high)
            field.setValue(value)
        thinking = QComboBox()
        for label, value in (("Off (recommended)", "off"), ("On", "on"), ("Low (models with named thinking levels)", "low")):
            thinking.addItem(label, value)
        thinking.setCurrentIndex(thinking.findData(self.thinking))
        verify = QCheckBox("Check candidates in a separate model pass")
        verify.setChecked(self.verify)
        relations = QListWidget()
        for category, names in RELATION_GROUPS.items():
            heading = QListWidgetItem(category)
            heading.setFlags(Qt.ItemIsEnabled)
            font = heading.font()
            font.setBold(True)
            heading.setFont(font)
            relations.addItem(heading)
            for name in names:
                item = QListWidgetItem(name)
                item.setToolTip(RELATIONS[name])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if name in self._relations else Qt.Unchecked)
                relations.addItem(item)
        form.addRow("Context tokens", context)
        form.addRow("Maximum output tokens / call", output)
        form.addRow("Request time limit (seconds)", timeout)
        form.addRow("Thinking", thinking)
        form.addRow(verify)
        form.addRow("Relationships", relations)
        info = QLabel("English instructions; original source names and quotes. Windows contain up to 4,000 source characters. "
                      "Truncated windows retry as smaller sections. Completed extraction and verification are reused when budgets change. "
                      "Verification normally adds one call per nonempty window. Stopping waits for the next stream update or network timeout.")
        info.setWordWrap(True)
        form.addRow(info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            settings = KGSettings(self.model.currentText(), context.value(), output.value(), timeout.value(), verify.isChecked(),
                                  thinking.currentData(), tuple(relations.item(i).text() for i in range(relations.count()) if relations.item(i).checkState() == Qt.Checked))
            try:
                settings.validate()
            except ValueError as exc:
                QMessageBox.warning(self, "Extraction settings", str(exc))
                return
            self.apply_settings(settings)

    def start_build(self):
        if self.thread is not None or self.investigation_id is None:
            return
        file_ids = [i.data(Qt.UserRole) for i in self.documents.selectedItems() if not i.isHidden()]
        if not file_ids:
            return
        settings = self.settings()
        try:
            settings.validate()
        except ValueError as exc:
            QMessageBox.warning(self, "Knowledge graph", str(exc))
            return
        with self.db.transaction() as conn:
            conn.execute("INSERT OR REPLACE INTO settings(key,value,updated_at) VALUES('knowledge_graph_settings',?,?)",
                         (json.dumps(asdict(settings)), utc_now()))
        self.start_worker(KGWorker(self.config, self.store, self.investigation_id, file_ids, settings, self.force.isChecked()))

    def refresh_models(self):
        if self.thread is None:
            settings = KGSettings(request_seconds=30)
            self.start_worker(KGWorker(self.config, self.store, self.investigation_id, [], settings, models=True))

    def start_worker(self, worker: KGWorker):
        self.worker = worker
        self.thread = QThread(self)
        generation = self._generation
        self._worker_generation = generation
        worker.moveToThread(self.thread)
        self.thread.started.connect(worker.run)
        worker.progress.connect(lambda message: self.apply_progress(generation, message))
        worker.finished.connect(lambda result, error: self.worker_finished(generation, worker.models, result, error))
        worker.finished.connect(self.thread.quit)
        worker.finished.connect(worker.deleteLater)
        self.thread.finished.connect(self.thread_finished)
        self.thread.finished.connect(self.thread.deleteLater)
        self._seconds = 0
        self.status.setText("Importing knowledge graph…" if worker.import_path else
                            "Refreshing local models…" if worker.models else "Starting knowledge graph build…")
        self.timer.start()
        self.thread.start()
        self.update_actions()

    def tick(self):
        self._seconds += 1
        self.stop_button.setToolTip(f"Elapsed {self._seconds}s. Stop at next stream update; waiting requests use the network timeout.")
        if self._seconds % 5 == 0 and self.worker:
            action = "importing and checking source links" if self.worker.import_path else "waiting for model or processing current window"
            self.apply_progress(self._worker_generation, f"Elapsed {self._seconds}s total; {action}.", log_only=True)

    def apply_progress(self, generation: int, message: str, log_only: bool = False):
        if generation != self._generation:
            return
        self.log.appendPlainText(message)
        if not log_only:
            self.status.setText(message)

    def worker_finished(self, generation: int, models: bool, result: object, error: str):
        if generation != self._generation:
            return
        if error:
            self.status.setText(error)
            self.log.appendPlainText(error)
        elif models:
            selected = self.model.currentText()
            self.model.clear()
            for item in result:
                name = item.get("name", "")
                if name:
                    self.model.addItem(name)
            self.model.setCurrentText(selected)
            self.status.setText(f"Found {self.model.count()} installed models. Choose a generation model.")
        elif result.get("operation") == "import":
            message = (f"Imported {result['nodes']} new nodes and {result['claims']} new claims; "
                       f"{result['existing']} already present. {result['linked']} claims have current indexed evidence; "
                       f"{result['unlinked']} retain imported quotes.")
            self.status.setText(message)
            self.log.appendPlainText(message)
            self.reload_claims()
            self.views.setCurrentWidget(self.graph.parentWidget())
            QTimer.singleShot(0, self.graph.fit_graph)
        else:
            message = (f"Build {result['status']} · {result['completed']} windows saved · {result['cached']} cached · "
                       f"{result['claims']} validated suggestions · {result['rejected']} discarded · {len(result['failures'])} failed sections.")
            message += (f" Extraction reused {result.get('extraction_cached', 0)} times; "
                        f"verification reused {result.get('verification_cached', 0)} times; "
                        f"{result.get('splits', 0)} recovery subdivisions.")
            self.status.setText(message)
            self.log.appendPlainText(message)
            self.reload_claims()
            if result["failures"]:
                self.log.appendPlainText("First error: " + result["failures"][0]["error"])
            if result["status"] == "failed":
                self.views.setCurrentWidget(self.log_page)
            else:
                self.views.setCurrentWidget(self.graph.parentWidget())
                QTimer.singleShot(0, self.graph.fit_graph)

    def thread_finished(self):
        self.timer.stop()
        self.thread, self.worker = None, None
        self.update_actions()

    def stop(self):
        if self.worker:
            self.worker.cancel()
            self.stop_button.setEnabled(False)
            if self.worker.import_path:
                self.status.setText("Cancelling knowledge graph import…")
            else:
                self.status.setText("Stopping at the next stream update or network timeout; completed batches are retained.")

    def review(self, status: str):
        if self._selected and self.investigation_id is not None:
            self.store.review(self.investigation_id, self._selected["id"], status, self._selected["review_note"])
            self.reload_claims()

    def edit_claim(self):
        if not self._selected:
            return
        claim = self._selected
        dialog = QDialog(self)
        dialog.setWindowTitle("Correct relationship")
        form = QFormLayout(dialog)
        predicate, polarity = QComboBox(), QComboBox()
        predicate.addItems(list(RELATIONS))
        predicate.setCurrentText(claim["predicate"])
        polarity.addItems(["affirmed", "negated", "uncertain"])
        polarity.setCurrentText(claim["polarity"])
        attribution, time, note = QLineEdit(claim["attribution"]), QLineEdit(claim["time_text"]), QLineEdit(claim["review_note"])
        attribute = QLineEdit(claim["attribute"])
        for label, field in (("Relationship", predicate), ("Property", attribute), ("Polarity", polarity), ("Attributed to", attribution), ("Time", time), ("Review note", note)):
            form.addRow(label, field)
        form.addRow(QLabel("Editing resets review to suggested; the original source quote is retained."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            self.store.edit_claim(self.investigation_id, claim["id"], predicate.currentText(), polarity.currentText(),
                                  attribution.text(), time.text(), note.text(), attribute.text())
            self.reload_claims()

    def merge_nodes(self):
        if self.investigation_id is None:
            return
        nodes = self.store.nodes(self.investigation_id)
        if len(nodes) < 2:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Merge knowledge graph nodes")
        form = QFormLayout(dialog)
        source, target = QComboBox(), QComboBox()
        for node in nodes:
            label = f"{node['name']} · {node['node_type']} · #{node['id']}"
            source.addItem(label, node["id"])
            target.addItem(label, node["id"])
        target.setCurrentIndex(1)
        info = QLabel("Merge only when the source passages establish the same identity. "
                      "All references and source-specific aliases will point to the target node. This merge persists on subsequent builds.")
        info.setWordWrap(True)
        form.addRow(info)
        form.addRow("Merge", source)
        form.addRow("Into", target)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            try:
                self.store.merge_nodes(self.investigation_id, source.currentData(), target.currentData())
                self.reload_claims()
            except ValueError as exc:
                QMessageBox.warning(self, "Merge nodes", str(exc))

    def edit_node(self):
        if self.investigation_id is None:
            return
        nodes = self.store.nodes(self.investigation_id)
        dialog = QDialog(self)
        dialog.setWindowTitle("Correct knowledge graph node")
        form = QFormLayout(dialog)
        choice, node_type, name = QComboBox(), QComboBox(), QLineEdit()
        node_type.addItems(list(NODE_TYPES))
        for node in nodes:
            choice.addItem(f"{node['name']} · {node['node_type']} · #{node['id']}", node["id"])
        def populate(index):
            if 0 <= index < len(nodes):
                name.setText(nodes[index]["name"])
                node_type.setCurrentText(nodes[index]["node_type"])
        choice.currentIndexChanged.connect(populate)
        populate(0)
        form.addRow("Node", choice)
        form.addRow("Canonical name", name)
        form.addRow("Type", node_type)
        info = QLabel("Source names remain as aliases for future extraction. Connected accepted claims return to suggested for review.")
        info.setWordWrap(True)
        form.addRow(info)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() == QDialog.Accepted:
            try:
                self.store.edit_node(self.investigation_id, choice.currentData(), name.text(), node_type.currentText())
                self.reload_claims()
            except ValueError as exc:
                QMessageBox.warning(self, "Edit node", str(exc))

    def export_graph(self):
        if self.investigation_id is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export knowledge graph", "knowledge-graph.json", "JSON (*.json)")
        if not path:
            return
        try:
            payload = self.store.export_graph(self.investigation_id)
            Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status.setText(f"Exported {len(payload['nodes'])} nodes and {len(payload['claims'])} source claims, including review and evidence statuses.")
        except OSError as exc:
            QMessageBox.warning(self, "Export graph", str(exc))

    def import_graph(self):
        if self.investigation_id is None or self.thread is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Import knowledge graph", "", "Bosuil knowledge graph (*.json)")
        if path:
            self.start_worker(KGWorker(self.config, self.store, self.investigation_id, [], self.settings(), import_path=path))

    def show_diagnostics(self):
        if self.investigation_id is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("KG run diagnostics")
        dialog.resize(850, 650)
        layout = QVBoxLayout(dialog)
        info = QLabel("Durations are in nanoseconds. load_duration: loading the model; prompt_eval_duration: reading input; "
                      "eval_duration: generating tokens. Thinking-character counts reveal emitted reasoning despite the requested setting. "
                      "These records include source text in model output.")
        info.setWordWrap(True)
        layout.addWidget(info)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setPlainText(json.dumps(self.store.diagnostics(self.investigation_id), ensure_ascii=False, indent=2))
        layout.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec_()

    def update_actions(self):
        busy, loaded = self.thread is not None, self.investigation_id is not None
        self.build_button.setEnabled(loaded and not busy and any(not i.isHidden() for i in self.documents.selectedItems()))
        self.stop_button.setEnabled(busy)
        for control in (self.model, self.refresh_models_button, self.settings_button, self.force, self.documents,
                        self.node_limit, self.graph_layout_choice, self.arrange_button):
            control.setEnabled(not busy)
        for button in (self.accept_button, self.reject_button, self.reset_button, self.edit_button):
            button.setEnabled(loaded and self._selected is not None and not busy)
        has_nodes = bool(self._claims or self._orphan_nodes)
        self.merge_button.setEnabled(loaded and has_nodes and not busy)
        self.node_button.setEnabled(loaded and has_nodes and not busy)
        self.export_button.setEnabled(loaded and has_nodes and not busy)
        self.import_button.setEnabled(loaded and not busy)
        self.diagnostics_button.setEnabled(loaded)

    def shutdown(self) -> bool:
        if self.thread is not None:
            self.stop()
            return False
        self.preview.source_view.clear()
        return True
