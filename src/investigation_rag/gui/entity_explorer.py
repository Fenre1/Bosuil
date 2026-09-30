"""Investigation-oriented graph UI; rendering limits never limit evidence queries."""
from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from threading import Event, Thread

import numpy as np
from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPainterPathStroker, QPen, QTextCharFormat, QTextCursor, QTransform
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox, QGraphicsEllipseItem,
    QGraphicsItem, QGraphicsPathItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QPushButton, QSplitter, QTabWidget,
    QTableWidget, QTableWidgetItem, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from investigation_rag.entity_network import (
    ExplorationRepository, connection_layout, load_network, rank_connections, separate_topic_positions,
)
from investigation_rag.gui.checkable_type_filter import CheckableTypeFilter
from investigation_rag.gui.chunk_preview import ChunkPreview
from investigation_rag.gui.entity_merge_dialog import EntityMergeDialog
from investigation_rag.retrieval.ranking import row_to_result


class _Emitter(QObject):
    finished = pyqtSignal(int, object, str)


class EntityNode(QGraphicsEllipseItem):
    def __init__(self, entity, color, pinned, moved):
        super().__init__(-10, -10, 20, 20)
        self.entity_id = entity.id
        self.moved = moved
        self.setBrush(QColor(color))
        self.setPen(QPen(QColor("#0f172a"), 3 if pinned else 1))
        self.setFlags(QGraphicsItem.ItemIsMovable | QGraphicsItem.ItemIsSelectable | QGraphicsItem.ItemSendsGeometryChanges)
        self.setZValue(2)
        self.setAcceptHoverEvents(True)
        font = QFont("Segoe UI", 9)
        self.label = QGraphicsSimpleTextItem(QFontMetrics(font).elidedText(entity.name, Qt.ElideRight, 140), self)
        self.label.setFont(font)
        self.label.setPos(-self.label.boundingRect().width() / 2, 14)
        self.label.setBrush(QColor("#172033"))
        self.label.setAcceptedMouseButtons(Qt.NoButton)
        self.setToolTip(f"{entity.name}\n{entity.entity_type}\n" + ("Pinned\n" if pinned else "") + "Double-click to expand; right-click for actions")

    def itemChange(self, change, value):
        result = super().itemChange(change, value)
        if change == QGraphicsItem.ItemPositionHasChanged and self.scene():
            self.moved(self.entity_id)
        return result

    def hoverEnterEvent(self, event):
        self.label.show()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        for view in self.scene().views():
            view.update_labels()
        super().hoverLeaveEvent(event)


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, edge, nodes, mode):
        super().__init__()
        self.edge = edge
        self.nodes = nodes
        pen = QPen(QColor("#8c9bb0"), min(5.0, 1 + math.log1p(edge.unique_documents)))
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setZValue(0)
        self.setToolTip(f"Mentioned in the same {mode}\n{edge.unique_documents} distinct document contents; "
                        f"{edge.unique_passages} distinct shared passage texts\nClick to inspect evidence")
        self.update_path()

    def update_path(self):
        path = QPainterPath(self.nodes[self.edge.first].pos())
        path.lineTo(self.nodes[self.edge.second].pos())
        self.setPath(path)

    def shape(self):
        stroker = QPainterPathStroker()
        stroker.setWidth(12)
        return stroker.createStroke(self.path())


class NetworkView(QGraphicsView):
    entity_clicked = pyqtSignal(int)
    edge_clicked = pyqtSignal(object)
    expand_requested = pyqtSignal(int)
    action_requested = pyqtSignal(str, int)
    positions_changed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#f8fafc"))
        self.nodes, self.edges, self.adjacent = {}, [], defaultdict(list)
        self.pinned, self.focus = set(), set()
        self._drawing = False

    def positions(self):
        return {i: (n.pos().x(), n.pos().y()) for i, n in self.nodes.items()}

    def viewport_state(self):
        center = self.mapToScene(self.viewport().rect().center())
        return {"scale": self.transform().m11(), "center": [center.x(), center.y()]}

    def restore_viewport(self, state):
        self.setTransform(QTransform.fromScale(state.get("scale", 1), state.get("scale", 1)))
        self.centerOn(*state.get("center", [0, 0]))
        self.update_labels()

    def draw(self, network, ids, edges, positions, pinned):
        from investigation_rag.gui.entity_graph import entity_type_color
        self._drawing = True
        camera = self.viewport_state()
        self.scene().clear()
        self.nodes, self.edges, self.adjacent = {}, [], defaultdict(list)
        self.pinned = set(pinned)
        for i in sorted(ids):
            entity = network.entities[i]
            node = EntityNode(entity, entity_type_color(entity.entity_type), i in pinned, self._moved)
            node.setPos(*positions[i])
            self.scene().addItem(node)
            self.nodes[i] = node
        for edge in edges:
            item = ConnectionItem(edge, self.nodes, self.relation_mode)
            self.scene().addItem(item)
            self.edges.append(item)
            self.adjacent[edge.first].append(item)
            self.adjacent[edge.second].append(item)
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-120, -120, 120, 120))
        self._drawing = False
        self.restore_viewport(camera)

    def _moved(self, entity_id):
        if self._drawing:
            return
        for edge in self.adjacent[entity_id]:
            edge.update_path()

    def fit_view(self):
        if self.nodes:
            self.fitInView(self.scene().itemsBoundingRect().adjusted(-40, -40, 40, 40), Qt.KeepAspectRatio)
            if self.transform().m11() > 1.5:
                self.scale(1.5 / self.transform().m11(), 1.5 / self.transform().m11())
        self.update_labels()

    def update_labels(self):
        zoom = self.transform().m11()
        for i, node in self.nodes.items():
            node.label.setVisible(zoom >= 0.65 or i in self.focus or i in self.pinned or node.isSelected())
            node.label.setFlag(QGraphicsItem.ItemIgnoresTransformations, zoom < 0.65)

    def set_focus(self, ids):
        self.focus = set(ids)
        for i, node in self.nodes.items():
            node.setPen(QPen(QColor("#0f172a"), 3 if i in self.focus or i in self.pinned else 1))
        for edge in self.edges:
            pen = edge.pen()
            pen.setColor(QColor("#3b5f91" if {edge.edge.first, edge.edge.second} & self.focus else "#c7d0dc"))
            edge.setPen(pen)
        self.update_labels()

    def wheelEvent(self, event):
        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        self.scale(factor, factor)
        self.update_labels()

    def node_at(self, position):
        for item in self.items(position):
            if isinstance(item, EntityNode):
                return item
            if isinstance(item.parentItem(), EntityNode):
                return item.parentItem()
        return None

    def mousePressEvent(self, event):
        node = self.node_at(event.pos())
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            if node:
                self.entity_clicked.emit(node.entity_id)
            else:
                edge = next((i for i in self.items(event.pos()) if isinstance(i, ConnectionItem)), None)
                if edge:
                    self.edge_clicked.emit(edge.edge)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.scene().setSceneRect(self.scene().itemsBoundingRect().adjusted(-120, -120, 120, 120))
        self.positions_changed.emit()

    def mouseDoubleClickEvent(self, event):
        node = self.node_at(event.pos())
        if node:
            self.expand_requested.emit(node.entity_id)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        node = self.node_at(event.pos())
        if not node:
            return
        menu = QMenu(self)
        for label, action in [("Expand connections", "expand"), ("Pin / unpin", "pin"),
                              ("Hide from this view", "hide"), ("Correct type", "type"),
                              ("Merge selected aliases", "merge")]:
            menu.addAction(label, lambda checked=False, a=action: self.action_requested.emit(a, node.entity_id))
        menu.exec_(event.globalPos())


class EntityGraphPanel(QWidget):
    entities_changed = pyqtSignal()

    def __init__(self, db, store=None):
        super().__init__()
        self.db, self.store = db, store
        self.investigation_id = None
        self.network = None
        self._active, self._dirty, self._closed = False, True, False
        self._generation = 0
        self._jobs = {}
        self.visible, self.pinned, self.hidden, self.seeds = set(), set(), set(), set()
        self.selected = None
        self.selected_edge = None
        self.scope = None
        self._history, self._history_index = [], -1
        self._preview_ids, self._preview_index, self._highlight_names = [], -1, []
        self._pending_ids = []
        self._positions = {}
        self._pending_selection = None
        self._pending_evidence = None
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._load)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a person, organisation, location, asset, or alias")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._search)
        self.search.returnPressed.connect(self._open_search)
        self.search_results = QListWidget()
        self.search_results.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.search_results.setMaximumHeight(180)
        self.search_results.itemDoubleClicked.connect(self._open_search)
        self.search_results.hide()
        self.scope_button = QPushButton("Documents: all")
        self.scope_button.clicked.connect(self.choose_scope)
        self.type_filter = CheckableTypeFilter()
        self.type_filter.selectionChanged.connect(self._scope_changed)
        self.relation = QComboBox()
        self.relation.addItem("Same passage", "passage")
        self.relation.addItem("Same document", "document")
        self.relation.currentIndexChanged.connect(self._relation_changed)
        self.ranking = QComboBox()
        for label, value in [("Most shared documents", "documents"), ("Most shared passages", "passages"), ("Most specific connections", "specificity")]:
            self.ranking.addItem(label, value)
        self.ranking.setToolTip("Counts discount exact copies. Specificity = shared document contents / document contents mentioning either entity (Jaccard). It is not confidence or independence.")
        self.ranking.currentIndexChanged.connect(self._ranking_changed)
        self.layout_mode = QComboBox()
        self.layout_mode.addItems(["Connections", "Topic map"])
        self.layout_mode.currentIndexChanged.connect(lambda: self._render(reset_layout=True))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        search_row = QHBoxLayout()
        search_row.addWidget(self.search, 1)
        self.open_button = self._button("Explore selected", self._open_search, search_row)
        search_row.addWidget(self.scope_button)
        search_row.addWidget(self.type_filter)
        layout.addLayout(search_row)
        layout.addWidget(self.search_results)
        options = QHBoxLayout()
        options.addWidget(QLabel("Connection:"))
        options.addWidget(self.relation)
        options.addWidget(QLabel("Rank:"))
        options.addWidget(self.ranking)
        options.addWidget(QLabel("Layout:"))
        options.addWidget(self.layout_mode)
        options.addStretch()
        layout.addLayout(options)
        navigation = QHBoxLayout()
        self.back_button = self._button("Back", lambda: self.navigate(-1), navigation)
        self.forward_button = self._button("Forward", lambda: self.navigate(1), navigation)
        self._button("Fit view", lambda: self.view.fit_view(), navigation)
        self._button("Overview", self.show_overview, navigation)
        self._button("Show all entities", self.show_all, navigation)
        exploration = self._menu("Exploration", navigation)
        self.undo_button = exploration.addAction("Undo view change", lambda: self.navigate(-1))
        exploration.addAction("Save view…", self.save_view)
        exploration.addAction("Load view…", self.load_view)
        exploration.addAction("Refresh collection", self.refresh)
        self.cancel_button = self._button("Cancel", self._cancel_requested, navigation)
        navigation.addStretch()
        layout.addLayout(navigation)
        self.status = QLabel("Open the graph to explore the collection")
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        layout.addWidget(self.legend)
        self.view = NetworkView()
        self.view.relation_mode = "passage"
        self.view.entity_clicked.connect(self.select_entity)
        self.view.edge_clicked.connect(self.select_edge)
        self.view.expand_requested.connect(self.expand)
        self.view.action_requested.connect(self.node_action)

        self.overview = QTreeWidget()
        self.overview.setHeaderLabels(["Connected groups · double-click to explore", "Entities"])
        self.overview.itemDoubleClicked.connect(self._open_group)
        self.canvas_tabs = QTabWidget()
        self.canvas_tabs.addTab(self.view, "Network")
        self.canvas_tabs.addTab(self.overview, "Collection overview")
        details = QWidget()
        detail_layout = QVBoxLayout(details)
        detail_layout.setContentsMargins(4, 0, 0, 0)
        self.details_label = QLabel("Find an entity or select a document to start. Ctrl-click nodes to select several.")
        self.details_label.setWordWrap(True)
        self.details_label.setTextFormat(Qt.PlainText)
        detail_layout.addWidget(self.details_label)
        actions = QHBoxLayout()
        self.expand_button = self._button("Expand all", lambda: self.expand(self.selected), actions)
        self.follow_button = self._button("Follow", self.follow_connection, actions)
        self.shared_button = self._button("Shared connections", self.shared_connections, actions)
        corrections = self._menu("Entity actions", actions)
        self.pin_button = corrections.addAction("Pin / unpin", self.pin_selected)
        self.hide_button = corrections.addAction("Hide from this view", self.hide_selected)
        corrections.addAction("Correct type…", self.correct_type)
        corrections.addAction("Merge selected aliases…", self.merge_selected)
        corrections.addAction("Undo correction", self.undo_correction)
        detail_layout.addLayout(actions)
        self.detail_tabs = QTabWidget()
        self.connections = QTableWidget(0, 4)
        self.connections.setHorizontalHeaderLabels(["Connected entity", "Docs*", "Passages*", "Specificity"])
        self.connections.horizontalHeaderItem(2).setToolTip("Distinct passage texts mentioning both entities, even when Same document is selected")
        self.connections.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.connections.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.connections.setSortingEnabled(True)
        self.connections.horizontalHeader().setStretchLastSection(True)
        self.connections.cellClicked.connect(self._connection_clicked)
        self.connections.cellDoubleClicked.connect(self._connection_opened)
        self.evidence = QTreeWidget()
        self.evidence.setHeaderLabels(["Source document / passage", "Entities present"])
        self.evidence.header().setSectionResizeMode(QHeaderView.Stretch)
        self.evidence.itemClicked.connect(self._evidence_clicked)
        self.detail_tabs.addTab(self.connections, "Connections")
        self.detail_tabs.addTab(self.evidence, "Evidence")
        self.detail_tabs.setMinimumHeight(160)
        detail_layout.addWidget(self.detail_tabs, 1)
        self.count_note = QLabel("* Ranking counts discount exact copies.")
        self.count_note.setToolTip("Counts do not establish source independence. All source copies and single-occurrence entities remain accessible.")
        self.count_note.setWordWrap(True)
        detail_layout.addWidget(self.count_note)
        self.preview = ChunkPreview("Select a source passage", db=db, navigation_label="evidence passage")
        self.preview.set_mode("readable")
        self.preview.rendered.connect(self.highlight_preview)
        self.preview.previous_requested.connect(lambda: self.adjacent_evidence(-1))
        self.preview.next_requested.connect(lambda: self.adjacent_evidence(1))
        self.inspector = QSplitter(Qt.Vertical)
        self.inspector.setChildrenCollapsible(False)
        self.inspector.addWidget(details)
        self.inspector.addWidget(self.preview)
        self.inspector.setSizes([480, 320])
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.addWidget(self.canvas_tabs)
        self.splitter.addWidget(self.inspector)
        self.splitter.setSizes([850, 520])
        layout.addWidget(self.splitter, 1)
        self._update_actions()

    @staticmethod
    def _button(label, callback, layout):
        button = QPushButton(label)
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    @staticmethod
    def _menu(label, layout):
        button = QToolButton()
        button.setText(label)
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        button.setMenu(menu)
        layout.addWidget(button)
        return menu

    def _types(self):
        return self.type_filter.selected_types() if self.type_filter.is_filter_active() else None

    def set_context(self, investigation_id, visible_entity_ids=()):
        # Table search and sorting are intentionally independent of graph scope.
        if investigation_id == self.investigation_id:
            return
        self.cancel()
        self.investigation_id = investigation_id
        self.network = None
        self.visible, self.pinned, self.hidden, self.seeds = set(), set(), set(), set()
        self.scope, self.selected, self.selected_edge = None, None, None
        self._positions, self._history, self._history_index = {}, [], -1
        self._pending_ids = []
        self.scope_button.setText("Documents: all")
        self.search.clear()
        self.type_filter.blockSignals(True)
        self.type_filter.set_entity_types([])
        self.type_filter.blockSignals(False)
        self._clear_display()
        self._dirty = True
        if self._active:
            self._debounce.start(0)

    def set_data_sources(self, db, store):
        changed = db is not self.db
        self.cancel()
        self.db, self.store = db, store
        self.preview.db = db
        if changed:
            self.set_context(None)
        self.network = None
        self._dirty = True

    def set_active(self, active):
        self._active = active
        if active and self._dirty:
            self._debounce.start(0)

    def refresh(self):
        self._dirty = True
        if self._active:
            self._debounce.start(0)

    def _clear_display(self):
        self._preview_ids, self._preview_index, self._highlight_names = [], -1, []
        self._pending_selection, self._pending_evidence = None, None
        self.view.scene().clear()
        self.view.nodes, self.view.edges, self.view.adjacent = {}, [], defaultdict(list)
        self.overview.clear()
        self.connections.setRowCount(0)
        self.evidence.clear()
        self.preview.clear()
        self.legend.clear()
        self.details_label.setText("Find an entity or select a document to start.")
        self.status.setText("No investigation loaded" if self.investigation_id is None else "Loading collection…")

    def _run(self, work, done, message):
        self.cancel()
        generation = self._generation
        event = Event()
        emitter = _Emitter(self)
        def finish(gen, result, error):
            self._jobs.pop(gen, None)
            emitter.deleteLater()
            if self._closed or gen != self._generation:
                return
            self._set_busy(False)
            if error:
                if self.network is not None and self._history_index >= 0:
                    self._restore(self._history[self._history_index], redraw=False)
                self.status.setText(error)
            else:
                done(result)
        emitter.finished.connect(finish, Qt.QueuedConnection)
        def run():
            try:
                result = work(event.is_set)
                error = ""
            except InterruptedError:
                result, error = None, "Cancelled"
            except Exception as exc:
                result, error = None, f"Could not build graph: {exc}"
            try:
                emitter.finished.emit(generation, result if not event.is_set() else None,
                                      error if not event.is_set() else "Cancelled")
            except RuntimeError:
                pass  # The parent widget may have been destroyed during shutdown.
        thread = Thread(target=run, daemon=True, name="entity-explorer")
        self._jobs[generation] = (event, emitter, thread)
        self.status.setText(message)
        self._set_busy(True)
        thread.start()

    def _set_busy(self, busy):
        # Keep Cancel available while a complete query/layout is being built.
        # Disable commands that could save or edit an unfinished drawing.
        for widget in (self.search, self.scope_button, self.type_filter, self.relation,
                       self.ranking, self.layout_mode, self.canvas_tabs, self.inspector):
            widget.setEnabled(not busy)
        for widget in self.findChildren(QPushButton) + self.findChildren(QToolButton):
            if not self.inspector.isAncestorOf(widget) and not self.canvas_tabs.isAncestorOf(widget):
                widget.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        if not busy:
            self._update_actions()

    def cancel(self):
        self._generation += 1
        for event, _, _ in self._jobs.values():
            event.set()
        self.cancel_button.setEnabled(False) if hasattr(self, "cancel_button") else None
        if hasattr(self, "inspector"):
            self._set_busy(False)
        if hasattr(self, "status") and self._jobs:
            self.status.setText("Cancelled. The previous drawing remains available; Refresh retries the collection load.")

    def _cancel_requested(self):
        self.cancel()
        self._set_busy(False)
        if self.network is not None and self._history_index >= 0:
            self._restore(self._history[self._history_index], redraw=False)
            self._status(len(self.view.edges))
            self.status.setText(self.status.text() + " Cancelled the pending change.")

    def shutdown(self):
        self._closed = True
        self._debounce.stop()
        self.cancel()

    def _load(self):
        if self.investigation_id is None or self._closed:
            return
        db, investigation = self.db, self.investigation_id
        self._run(lambda cancelled: load_network(db, investigation, cancelled), self._loaded,
                  "Indexing all active entity memberships…")

    def _loaded(self, network):
        self.network = network
        self._dirty = False
        valid = set(network.entities)
        self.visible &= valid
        self.pinned &= valid
        self.seeds &= valid
        self.hidden &= valid
        if self.scope is not None:
            self.scope &= set(network.documents)
        self.type_filter.blockSignals(True)
        self.type_filter.set_entity_types(sorted({e.entity_type for e in network.entities.values()}))
        self.type_filter.blockSignals(False)
        self._search()
        self._update_overview()
        if not self.visible and not self._pending_ids:
            self.canvas_tabs.setCurrentWidget(self.overview)
        if self._pending_ids:
            ids, self._pending_ids = self._pending_ids, []
            self.explore_entities(ids)
        else:
            self._render(fit=not self.view.nodes)

    def _status(self, edges=0):
        if self.network is None:
            return
        eligible = self.network.eligible(self.scope, self._types())
        docs = self.scope if self.scope is not None else set(self.network.documents)
        with_entities = sum(bool(self.network.by_document[d]) for d in docs)
        self.status.setText(f"Collection: {len(self.network.entities):,} entities / {self.network.indexed_files:,} indexed files · "
            f"Scope: {len(docs):,} documents (including attachments), {len(eligible):,} entities · "
            f"Displayed: {len(self.visible):,} entities, {edges:,} connections · {len(self.hidden):,} hidden. "
            f"All matching memberships counted; {with_entities:,} scoped documents have entities. Extraction completeness is not recorded.")
        from html import escape
        from investigation_rag.gui.entity_graph import entity_type_color
        self.legend.setText(" &nbsp; ".join(f'<span style="color:{entity_type_color(t)}">●</span> {escape(t)}'
                            for t in sorted({self.network.entities[i].entity_type for i in self.visible})))

    def _render(self, *, fit=False, reset_layout=False, record=True, camera=None):
        if self.network is None:
            return
        self.visible &= self.network.eligible(self.scope, self._types())
        self.visible -= self.hidden
        network, ids = self.network, set(self.visible)
        mode, scope = self.relation.currentData(), self.scope
        old = {} if reset_layout else {**self._positions, **self.view.positions()}
        if reset_layout:
            old = {i: p for i, p in self.view.positions().items() if i in self.pinned}
        topic = self.layout_mode.currentText() == "Topic map"
        selection = self._pending_selection if self._pending_selection is not None else self.selected_ids()
        self._pending_selection = None
        evidence_state, self._pending_evidence = self._pending_evidence, None
        store = self.store
        def work(cancelled):
            edges = network.edges(ids, mode, scope, cancelled)
            positions = connection_layout(ids, edges, old, cancelled)
            warning = ""
            if topic and ids and (reset_layout or ids - set(old)):
                positions, warning = self._topic_positions(network, ids, scope, store, positions, cancelled)
                positions = separate_topic_positions(positions, old, cancelled)
            if cancelled():
                raise InterruptedError()
            return edges, positions, warning
        def done(result):
            edges, positions, warning = result
            self._positions.update(positions)
            self.view.relation_mode = mode
            self.view.draw(network, ids, edges, positions, self.pinned)
            for i in selection & ids:
                self.view.nodes[i].setSelected(True)
            if camera:
                self.view.restore_viewport(camera)
            elif fit:
                self.view.fit_view()
            self._status(len(edges))
            if topic:
                self.status.setText(self.status.text() + " Topic map: proximity means similar text, not a documented relationship. " + warning)
            if self.selected_edge and all(i in ids for i in self.selected_edge):
                edge = network.connection(*self.selected_edge, mode, scope)
                if edge:
                    self.select_edge(edge, record=False)
                else:
                    self.selected_edge = None
                    self.select_entity(self.selected, record=False)
            elif self.selected in ids:
                self.select_entity(self.selected, record=False)
            else:
                self.selected, self.selected_edge = None, None
                self.details_label.setText("Find an entity or select a document to start. Ctrl-click nodes for shared connections or merging.")
                self.connections.setRowCount(0)
                self.evidence.clear()
                self.preview.clear()
            if evidence_state and evidence_state[0] in self._preview_ids:
                self.preview.set_mode(evidence_state[1])
                self.show_chunk(evidence_state[0])
            if record:
                self._record()
            self._update_actions()
        self._run(work, done, f"Building {len(ids):,} entities with complete {mode} connections…")

    @staticmethod
    def _topic_positions(network, ids, scope, store, fallback, cancelled):
        if store is None:
            return fallback, "No stored embeddings available; using connections."
        from investigation_rag.gui.entity_graph import _manifold_2d, _pca_2d
        chunks = {c for i in ids for c in network.scoped_chunks(i, scope)}
        models = defaultdict(list)
        for c in chunks:
            models[network.chunks[c].embedding_model].append(c)
        model = max((m for m in models if m), key=lambda m: len(models[m]), default=None)
        if model is None:
            return fallback, "No recorded embedding model; using connections."
        vectors = {}
        for start in range(0, len(models[model]), 900):
            if cancelled():
                raise InterruptedError()
            vectors.update(store.get_vectors_for_chunks(models[model][start:start + 900], embedding_model=model))
        entity_ids, centroids = [], []
        dimension = None
        for i in sorted(ids):
            values = []
            for c in network.scoped_chunks(i, scope):
                if c in vectors:
                    v = np.asarray(vectors[c], dtype=np.float32)
                    if v.ndim == 1 and v.size and np.isfinite(v).all():
                        dimension = dimension or len(v)
                        if len(v) == dimension:
                            values.append(v)
            if values:
                entity_ids.append(i)
                centroids.append(np.mean(values, axis=0))
        if not centroids:
            return fallback, "No compatible embeddings; using connections."
        matrix = np.stack(centroids)
        try:
            coords = _manifold_2d(matrix) if len(matrix) >= 4 else _pca_2d(matrix)
        except Exception:
            coords = _pca_2d(matrix)
        extent = np.max(np.abs(coords)) or 1
        scale = max(300, math.sqrt(len(ids)) * 100)
        output = dict(fallback)
        for i, xy in zip(entity_ids, coords / extent * scale):
            output[i] = (float(xy[0]), float(xy[1]))
        return output, f"Embeddings cover {len(entity_ids):,} of {len(ids):,} entities; remaining positions use connections."

    def _search(self, *_):
        self.search_results.clear()
        query = self.search.text().strip().casefold()
        if not query or self.network is None:
            self.search_results.hide()
            return
        for i in sorted(self.network.eligible(self.scope, self._types()), key=lambda i: self.network.entities[i].name.casefold()):
            e = self.network.entities[i]
            if any(query in name.casefold() for name in (e.name, *e.aliases)):
                item = QListWidgetItem(f"{e.name} [{e.entity_type}] · {len(self.network.entity_documents[i]):,} documents")
                item.setData(Qt.UserRole, i)
                self.search_results.addItem(item)
        self.search_results.setVisible(True)
        if self.search_results.count():
            self.search_results.setCurrentRow(0)

    def _open_search(self, *_):
        ids = [i.data(Qt.UserRole) for i in self.search_results.selectedItems()]
        if ids:
            self.explore_entities(ids)
            self.search_results.hide()

    def explore_entities(self, ids):
        if self.network is None:
            self._pending_ids = list(ids)
            self.set_active(True)
            return
        self._capture_current()
        ids = set(ids) & self.network.eligible(self.scope, self._types())
        self.seeds = ids
        self.visible = ids | (self.pinned & self.network.eligible(self.scope, self._types()))
        self.hidden -= ids
        # Starting neighbourhood is a navigation default, never a hard graph cap.
        for i in ids:
            ranked = self._neighbors(i)
            self.visible.update(self._other(c, i) for c in ranked[:20])
        self.selected = min(ids) if ids else None
        self.selected_edge = None
        self.canvas_tabs.setCurrentWidget(self.view)
        self._render(fit=True)

    @staticmethod
    def _other(connection, entity_id):
        return connection.second if connection.first == entity_id else connection.first

    def _neighbors(self, entity_id):
        return rank_connections(self.network.neighbors(entity_id, self.relation.currentData(), self.scope, self._types()), self.ranking.currentData())

    def expand(self, entity_id=None):
        if self.network is None or entity_id not in self.network.entities:
            return
        self._capture_current()
        connections = self._neighbors(entity_id)
        self.visible.add(entity_id)
        self.visible.update(self._other(c, entity_id) for c in connections)
        self.hidden -= self.visible
        self.selected, self.selected_edge = entity_id, None
        self._render()

    def show_all(self):
        if self.network is None:
            return
        self._capture_current()
        self.hidden.clear()
        self.visible = self.network.eligible(self.scope, self._types())
        self.canvas_tabs.setCurrentWidget(self.view)
        self._render(fit=True)

    def selected_ids(self):
        ids = {i for i, n in self.view.nodes.items() if n.isSelected()}
        return ids or ({self.selected} if self.selected is not None else set())

    def shared_connections(self):
        ids = self.selected_ids()
        if self.network is None or len(ids) < 2:
            self.details_label.setText("Ctrl-click at least two entities, then choose Shared connections.")
            return
        self._capture_current()
        neighbors = [{self._other(c, i) for c in self._neighbors(i)} for i in ids]
        shared = set.intersection(*neighbors) - ids
        self.seeds = ids
        self.visible = ids | shared | self.pinned
        self.hidden -= self.visible
        self._render(fit=True)

    def pin_selected(self):
        self._capture_current()
        for i in self.selected_ids():
            if i in self.pinned:
                self.pinned.remove(i)
            else:
                self.pinned.add(i)
        self._render()

    def hide_selected(self):
        self._capture_current()
        ids = self.selected_ids()
        self.hidden.update(ids)
        self.visible -= ids
        self.pinned -= ids
        self.selected, self.selected_edge = None, None
        self._render()

    def node_action(self, action, entity_id):
        if action != "merge":
            self.view.scene().clearSelection()
            self.view.nodes[entity_id].setSelected(True)
        self.select_entity(entity_id)
        {"expand": lambda: self.expand(entity_id), "pin": self.pin_selected, "hide": self.hide_selected,
         "type": self.correct_type, "merge": self.merge_selected}[action]()

    def select_entity(self, entity_id, *, record=True):
        if self.network is None or entity_id not in self.network.entities:
            return
        if record:
            self._capture_current()
        self.selected, self.selected_edge = entity_id, None
        e = self.network.entities[entity_id]
        neighbors = self._neighbors(entity_id)
        displayed = sum(self._other(c, entity_id) in self.visible for c in neighbors)
        self.details_label.setText(f"{e.name} [{e.entity_type}]\nAliases: {', '.join(e.aliases) or 'none'}\n"
            f"{len(self.network.scoped_chunks(entity_id, self.scope)):,} passages · "
            f"{len(self.network.entity_documents[entity_id] & self.scope) if self.scope is not None else len(self.network.entity_documents[entity_id]):,} documents\n"
            f"{displayed:,} of {len(neighbors):,} connections displayed. Expand adds all; the list includes every connection.")
        self.view.set_focus([entity_id])
        self._populate_connections(neighbors, entity_id)
        self._populate_evidence(self.network.scoped_chunks(entity_id, self.scope), [entity_id])
        self.detail_tabs.setCurrentWidget(self.connections)
        self.preview.clear()
        if record:
            self._record()
        self._update_actions()

    def _populate_connections(self, connections, entity_id):
        self.connections.setSortingEnabled(False)
        self.connections.setRowCount(len(connections))
        for row, c in enumerate(connections):
            other = self._other(c, entity_id)
            e = self.network.entities[other]
            values = [f"{e.name} [{e.entity_type}]", c.unique_documents, c.unique_passages, round(c.specificity, 3)]
            for col, value in enumerate(values):
                item = QTableWidgetItem()
                item.setData(Qt.DisplayRole, value)
                item.setData(Qt.UserRole, (entity_id, other))
                self.connections.setItem(row, col, item)
        # The chosen ranking establishes initial order; header sorting stays available.
        sort_column = {"documents": 1, "passages": 2, "specificity": 3}[self.ranking.currentData()]
        self.connections.horizontalHeader().setSortIndicator(sort_column, Qt.DescendingOrder)
        self.connections.setSortingEnabled(True)
        self.connections.resizeColumnsToContents()

    def _connection_clicked(self, row, _column):
        pair = self.connections.item(row, 0).data(Qt.UserRole)
        edge = self.network.connection(*pair, self.relation.currentData(), self.scope)
        if edge:
            self.select_edge(edge)

    def _connection_opened(self, row, _column):
        pair = self.connections.item(row, 0).data(Qt.UserRole)
        self.explore_entities([pair[1]])

    def select_edge(self, edge, *, record=True):
        if record:
            self._capture_current()
        self.selected_edge = (edge.first, edge.second)
        other = self._other(edge, self.selected) if self.selected in self.selected_edge else edge.second
        self.follow_button.setToolTip(f"Explore {self.network.entities[other].name} and its neighbours")
        names = [self.network.entities[i].name for i in self.selected_edge]
        self.details_label.setText(f"{' ↔ '.join(names)}\nMentioned in the same {self.relation.currentData()}. "
            f"{len(edge.documents):,} source documents; {edge.unique_documents:,} distinct document contents; "
            f"{edge.unique_passages:,} distinct shared passage texts.")
        self.details_label.setToolTip("Co-occurrence does not establish a relationship, responsibility, or causation. Inspect the source passages to assess the connection.")
        self.view.set_focus(self.selected_edge)
        self._populate_evidence(edge.chunks, self.selected_edge)
        self.detail_tabs.setCurrentWidget(self.evidence)
        self.preview.clear()
        if record:
            self._record()
        self._update_actions()

    def follow_connection(self):
        if self.selected_edge:
            first, second = self.selected_edge
            self.explore_entities([first if self.selected == second else second])

    def _populate_evidence(self, chunk_ids, entity_ids):
        self.evidence.clear()
        self._preview_ids = sorted(chunk_ids, key=lambda c: (self.network.chunks[c].path.casefold(), self.network.chunks[c].document, self.network.chunks[c].index, c))
        self._preview_index = -1
        self._highlight_names = [(self.network.entities[i].name, *self.network.entities[i].aliases) for i in entity_ids]
        docs = {}
        for c in self._preview_ids:
            chunk = self.network.chunks[c]
            if chunk.document not in docs:
                document = self.network.documents[chunk.document]
                label = Path(document.path).name + (f" · {document.identity}" if document.identity != "root" else "")
                parent = QTreeWidgetItem([label, ""])
                parent.setToolTip(0, document.path)
                self.evidence.addTopLevelItem(parent)
                docs[chunk.document] = parent
            present = [self.network.entities[i].name for i in entity_ids if c in self.network.entities[i].chunks]
            label = f"Passage {chunk.index + 1}" + (f" · page {chunk.page}" if chunk.page else "")
            item = QTreeWidgetItem([label, ", ".join(present)])
            item.setData(0, Qt.UserRole, c)
            docs[chunk.document].addChild(item)
        self.evidence.expandToDepth(0)

    def _evidence_clicked(self, item, _column):
        chunk_id = item.data(0, Qt.UserRole)
        if chunk_id:
            self.show_chunk(chunk_id)

    def show_chunk(self, chunk_id):
        row = self.db.get_chunk(chunk_id)
        if row is None or not row["is_active"]:
            self.preview.setPlainText("This passage is no longer active. Refresh the collection.")
            return
        result = row_to_result(row)
        if result.investigation_id != self.investigation_id:
            return
        self._preview_index = self._preview_ids.index(chunk_id) if chunk_id in self._preview_ids else -1
        self.preview.set_chunk(result.chunk_text, result.display_markdown, path=result.path,
            chunk_index=result.chunk_index, metadata=result.metadata or {},
            has_previous=self._preview_index > 0, has_next=0 <= self._preview_index < len(self._preview_ids) - 1)

    def adjacent_evidence(self, delta):
        index = self._preview_index + delta
        if 0 <= index < len(self._preview_ids):
            self.show_chunk(self._preview_ids[index])

    def highlight_preview(self):
        text = self.preview.toPlainText()
        colors = ["#fff1a8", "#bce8ff"]
        for index, aliases in enumerate(self._highlight_names):
            pattern = "|".join(re.escape(name) for name in sorted(set(aliases), key=len, reverse=True) if name)
            if not pattern:
                continue
            cursor = self.preview.textCursor()
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(colors[index % len(colors)]))
            fmt.setForeground(QColor("#172033"))
            for match in re.finditer(rf"(?<!\w)(?:{pattern})(?!\w)", text, re.IGNORECASE):
                # QTextCursor offsets use UTF-16, not Python's code point indexing.
                cursor.setPosition(len(text[:match.start()].encode("utf-16-le")) // 2)
                cursor.setPosition(len(text[:match.end()].encode("utf-16-le")) // 2, QTextCursor.KeepAnchor)
                cursor.mergeCharFormat(fmt)

    def choose_scope(self):
        if self.network is None:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Document scope")
        dialog.resize(660, 500)
        layout = QVBoxLayout(dialog)
        search = QLineEdit()
        search.setPlaceholderText("Find a document or attachment")
        layout.addWidget(search)
        choices = QListWidget()
        for d in sorted(self.network.documents.values(), key=lambda d: (d.path.casefold(), d.identity)):
            item = QListWidgetItem(d.path + (f" · {d.identity}" if d.identity != "root" else ""))
            item.setData(Qt.UserRole, d.id)
            item.setCheckState(Qt.Checked if self.scope is None or d.id in self.scope else Qt.Unchecked)
            choices.addItem(item)
        search.textChanged.connect(lambda query: [choices.item(i).setHidden(query.casefold() not in choices.item(i).text().casefold()) for i in range(choices.count())])
        layout.addWidget(choices)
        actions = QHBoxLayout()
        for label, checked in [("Select all", True), ("Clear", False)]:
            self._button(label, lambda _=False, value=checked: [choices.item(i).setCheckState(Qt.Checked if value else Qt.Unchecked) for i in range(choices.count()) if not choices.item(i).isHidden()], actions)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            self._capture_current()
            scope = {choices.item(i).data(Qt.UserRole) for i in range(choices.count()) if choices.item(i).checkState() == Qt.Checked}
            self.scope = None if scope == set(self.network.documents) else frozenset(scope)
            self.scope_button.setText("Documents: all" if self.scope is None else f"Documents: {len(scope)}")
            eligible = self.network.eligible(self.scope, self._types())
            if not self.visible & eligible:
                self.visible = eligible
            self._scope_changed(fit=True)

    def _scope_changed(self, *_, fit=False):
        if self.network:
            self._search()
            self._update_overview()
            self._render(fit=fit)

    def _relation_changed(self, *_):
        self.selected_edge = None
        self._scope_changed()

    def _ranking_changed(self, *_):
        if self.network and self.selected in self.network.entities:
            self.select_entity(self.selected)

    def _update_overview(self):
        self.overview.clear()
        if self.network is None:
            return
        for group in self.network.groups(self.relation.currentData(), self.scope, self._types()):
            ordered = sorted(group, key=lambda i: (-len(self.network.entity_documents[i]), self.network.entities[i].name.casefold()))
            label = ", ".join(self.network.entities[i].name for i in ordered[:3])
            item = QTreeWidgetItem([label, str(len(group))])
            item.setData(0, Qt.UserRole, sorted(group))
            item.setToolTip(0, "Connected by co-occurrence; an automatic navigation group, not an investigative finding.")
            self.overview.addTopLevelItem(item)
            by_type = defaultdict(list)
            for i in ordered:
                by_type[self.network.entities[i].entity_type].append(i)
            for entity_type, ids in sorted(by_type.items()):
                type_item = QTreeWidgetItem([entity_type, str(len(ids))])
                type_item.setData(0, Qt.UserRole, ids)
                item.addChild(type_item)
                for i in ids:
                    child = QTreeWidgetItem([self.network.entities[i].name, "1"])
                    child.setData(0, Qt.UserRole, [i])
                    type_item.addChild(child)
        self.overview.resizeColumnToContents(0)

    def show_overview(self):
        self._update_overview()
        self.canvas_tabs.setCurrentWidget(self.overview)

    def _open_group(self, item, _column):
        ids = item.data(0, Qt.UserRole)
        if len(ids) == 1:
            self.explore_entities(ids)
            return
        self._capture_current()
        self.visible = set(ids) | self.pinned
        self.hidden -= self.visible
        self.canvas_tabs.setCurrentWidget(self.view)
        self._render(fit=True)

    def _state(self):
        return {"version": 1, "visible": sorted(self.visible), "pinned": sorted(self.pinned),
            "hidden": sorted(self.hidden), "seeds": sorted(self.seeds), "selected": self.selected,
            "edge": self.selected_edge, "scope": sorted(self.scope) if self.scope is not None else None,
            "types": sorted(self._types()) if self._types() is not None else None,
            "relation": self.relation.currentData(), "ranking": self.ranking.currentData(),
            "layout": self.layout_mode.currentText(), "positions": {str(i): list(p) for i, p in {**self._positions, **self.view.positions()}.items()},
            "camera": self.view.viewport_state(), "selection": sorted(self.selected_ids()),
            "tab": self.canvas_tabs.currentIndex(),
            "evidence": self._preview_ids[self._preview_index] if 0 <= self._preview_index < len(self._preview_ids) else None,
            "preview_mode": self.preview._mode}

    def _capture_current(self):
        if 0 <= self._history_index < len(self._history):
            self._history[self._history_index] = self._state()

    def _record(self):
        state = self._state()
        if self._history_index >= 0 and self._history[self._history_index] == state:
            return
        self._history = self._history[:self._history_index + 1]
        self._history.append(state)
        self._history_index = len(self._history) - 1
        self._update_actions()

    def _restore(self, state, *, redraw=True):
        valid = set(self.network.entities)
        for attr in ("visible", "pinned", "hidden", "seeds"):
            setattr(self, attr, set(state.get(attr, [])) & valid)
        self.selected = state.get("selected")
        self.selected_edge = state.get("edge")
        self._pending_selection = set(state.get("selection", [])) & valid
        self._pending_evidence = (state.get("evidence"), state.get("preview_mode", "readable"))
        self.canvas_tabs.setCurrentIndex(state.get("tab", 0))
        self.scope = frozenset(state["scope"]) & set(self.network.documents) if state.get("scope") is not None else None
        self.scope_button.setText("Documents: all" if self.scope is None else f"Documents: {len(self.scope)}")
        self.type_filter.blockSignals(True)
        if state.get("types") is None:
            self.type_filter.select_all()
        else:
            self.type_filter.set_selected_types(state["types"])
        self.type_filter.blockSignals(False)
        for combo, key in [(self.relation, "relation"), (self.ranking, "ranking")]:
            combo.blockSignals(True)
            combo.setCurrentIndex(max(0, combo.findData(state.get(key))))
            combo.blockSignals(False)
        self.layout_mode.blockSignals(True)
        self.layout_mode.setCurrentText(state.get("layout", "Connections"))
        self.layout_mode.blockSignals(False)
        self._positions = {int(i): tuple(p) for i, p in state.get("positions", {}).items() if int(i) in valid}
        # The restored positions take precedence over the current drawing.
        if redraw:
            self.view.scene().clear()
            self.view.nodes, self.view.edges = {}, []
        self._update_overview()
        self._search()
        if redraw:
            self._render(record=False, camera=state.get("camera"))

    def navigate(self, delta):
        index = self._history_index + delta
        if 0 <= index < len(self._history):
            self._capture_current()
            self._history_index = index
            self._restore(self._history[index])

    def save_view(self):
        if self.network is None:
            return
        name, ok = QInputDialog.getText(self, "Save exploration", "View name (an existing name replaces that view):")
        if ok and name.strip():
            try:
                ExplorationRepository(self.db).save(self.investigation_id, name.strip(), self._state())
                self.status.setText(f"Saved exploration: {name.strip()}")
            except Exception as exc:
                QMessageBox.warning(self, "Save exploration", str(exc))

    def load_view(self):
        if self.network is None:
            return
        try:
            saved = ExplorationRepository(self.db).saved(self.investigation_id)
            if not saved:
                self.status.setText("No saved explorations for this investigation.")
                return
            name, ok = QInputDialog.getItem(self, "Load exploration", "View:", list(saved), editable=False)
            if ok:
                self._capture_current()
                self._restore(saved[name])
                self._history = self._history[:self._history_index + 1] + [saved[name]]
                self._history_index = len(self._history) - 1
        except Exception as exc:
            QMessageBox.warning(self, "Load exploration", str(exc))

    def _update_actions(self):
        self.back_button.setEnabled(self._history_index > 0)
        self.undo_button.setEnabled(self._history_index > 0)
        self.forward_button.setEnabled(self._history_index + 1 < len(self._history))
        self.expand_button.setEnabled(self.selected is not None)
        self.follow_button.setEnabled(self.selected_edge is not None)
        self.pin_button.setEnabled(self.selected is not None)
        self.hide_button.setEnabled(self.selected is not None)

    def correct_type(self):
        if self.network is None or self.selected not in self.network.entities:
            return
        e = self.network.entities[self.selected]
        value, ok = QInputDialog.getText(self, "Correct entity type", f"Type for {e.name}:", text=e.entity_type)
        if ok and value.strip() and value.strip().casefold() != e.entity_type:
            self._edit([e.id], e.name, value)

    def merge_selected(self):
        ids = sorted(self.selected_ids())
        if self.network is None or len(ids) < 2:
            self.details_label.setText("Ctrl-click at least two nodes to merge their aliases.")
            return
        dialog = EntityMergeDialog([(self.network.entities[i].name, self.network.entities[i].entity_type) for i in ids], self)
        if dialog.exec_() == QDialog.Accepted:
            self._edit(ids, *dialog.values())

    def _edit(self, ids, name, entity_type):
        try:
            target = ExplorationRepository(self.db).edit(self.investigation_id, ids, name, entity_type)
            self.visible -= set(ids)
            self.visible.add(target)
            if self.pinned & set(ids):
                self.pinned -= set(ids)
                self.pinned.add(target)
            self.selected, self.selected_edge = target, None
            self.entities_changed.emit()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Entity correction", str(exc))

    def undo_correction(self):
        if self.investigation_id is None:
            return
        try:
            ExplorationRepository(self.db).undo_edit(self.investigation_id)
            self.entities_changed.emit()
            self.refresh()
        except Exception as exc:
            QMessageBox.warning(self, "Undo correction", str(exc))
