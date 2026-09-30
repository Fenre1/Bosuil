from __future__ import annotations

import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from html import escape
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QModelIndex, QRect, QRectF, QSize, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QTextCharFormat, QTextCursor, QTextDocument
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QSplitter,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from investigation_rag.cli import build_searcher
from investigation_rag.config import AppConfig
from investigation_rag.db.lancedb_store import LanceDBStore
from investigation_rag.db.sqlite_db import SQLiteDB
from investigation_rag.gui.file_type_scope import FileTypeScope
from investigation_rag.gui.folder_scope import FolderScope
from investigation_rag.gui.chunk_preview import ChunkPreview
from investigation_rag.retrieval.keyword_search import query_terms
from investigation_rag.retrieval.ranking import SearchResult, result_location_label
from investigation_rag.utils.windows_open import open_containing_folder, open_file


RESULT_DATA_ROLE = Qt.UserRole
TITLE_ROLE = Qt.UserRole + 1
SUBTITLE_ROLE = Qt.UserRole + 2
SNIPPET_ROLE = Qt.UserRole + 3
TERMS_ROLE = Qt.UserRole + 4
RELEVANCE_ROLE = Qt.UserRole + 5
RELEVANCE_LABEL_ROLE = Qt.UserRole + 6
BADGES_ROLE = Qt.UserRole + 7
DOCUMENT_ROLE = Qt.UserRole + 8
ROW_HEIGHT_ROLE = Qt.UserRole + 9

DOCUMENT_ROW_HEIGHT = 64
MATCH_ROW_HEIGHT = 108


class SearchPanel(QWidget):
    def __init__(
        self,
        config: AppConfig,
        db: SQLiteDB,
        store: LanceDBStore,
        folder_scope: FolderScope | None = None,
        file_type_scope: FileTypeScope | None = None,
    ):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.folder_scope = folder_scope
        self.file_type_scope = file_type_scope
        self.investigation_id: int | None = None
        self.results: list[SearchResult] = []

        self.query_edit = QLineEdit()
        self.query_edit.setPlaceholderText("Search documents")
        self.query_edit.setClearButtonEnabled(True)
        self.query_edit.returnPressed.connect(self.run_search)
        self.query_edit.textChanged.connect(self.update_search_button_state)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["keyword", "semantic", "hybrid"])
        self.search_button = QPushButton("Search")
        self.search_button.setObjectName("PrimaryButton")
        self.search_button.clicked.connect(self.run_search)
        self.open_button = QPushButton("Open file")
        self.open_button.setObjectName("GhostButton")
        self.open_button.clicked.connect(self.open_selected_file)
        self.folder_button = QPushButton("Open folder")
        self.folder_button.setObjectName("GhostButton")
        self.folder_button.clicked.connect(self.open_selected_folder)
        self.status_label = QLabel()
        self.status_label.setObjectName("StatusLabel")

        top = QHBoxLayout()
        top.setSpacing(8)
        top.addWidget(self.query_edit, 1)
        top.addWidget(self.mode_combo)
        top.addWidget(self.search_button)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Result", "Relevance", "Type", "Location"])
        self.table.setItemDelegate(SearchResultDelegate(self.table))
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setShowGrid(False)
        self.table.setWordWrap(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(MATCH_ROW_HEIGHT)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Fixed)
        self.table.setColumnWidth(1, 130)
        self.table.setColumnWidth(2, 204)
        self.table.setColumnWidth(3, 280)
        self.table.itemSelectionChanged.connect(self.update_preview)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Document / match", "Relevance", "Type", "Location"])
        self.tree.itemSelectionChanged.connect(self.update_preview)
        self.tree.setItemDelegate(SearchResultDelegate(self.tree))
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(False)
        self.tree.setWordWrap(True)
        self.tree.setIndentation(18)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.tree.header().setSectionResizeMode(1, QHeaderView.Fixed)
        self.tree.header().setSectionResizeMode(2, QHeaderView.Fixed)
        self.tree.header().setSectionResizeMode(3, QHeaderView.Fixed)
        self.tree.setColumnWidth(1, 130)
        self.tree.setColumnWidth(2, 204)
        self.tree.setColumnWidth(3, 280)

        self.result_stack = QStackedWidget()
        self.result_stack.addWidget(self.table)
        self.result_stack.addWidget(self.tree)

        list_button = QPushButton("List")
        list_button.setCheckable(True)
        list_button.setProperty("segment", True)
        list_button.setProperty("segmentPosition", "left")
        list_button.clicked.connect(lambda: self.set_result_view(0))
        hierarchy_button = QPushButton("Hierarchy")
        hierarchy_button.setCheckable(True)
        hierarchy_button.setChecked(True)
        hierarchy_button.setProperty("segment", True)
        hierarchy_button.setProperty("segmentPosition", "right")
        hierarchy_button.clicked.connect(lambda: self.set_result_view(1))
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        self.view_button_group.addButton(list_button)
        self.view_button_group.addButton(hierarchy_button)
        self.result_stack.setCurrentIndex(1)

        view_controls = QHBoxLayout()
        view_controls.setSpacing(8)
        view_controls.addWidget(list_button)
        view_controls.addWidget(hierarchy_button)
        view_controls.addSpacing(12)
        view_controls.addWidget(self.status_label)
        view_controls.addStretch(1)
        view_controls.addWidget(self.open_button)
        view_controls.addWidget(self.folder_button)

        self.metadata_view = QTextEdit()
        self.metadata_view.setReadOnly(True)
        self.metadata_view.setPlaceholderText("Select a result to view metadata")
        self.preview = ChunkPreview("Select a result to preview matching text", db=db, navigation_label="result")
        self.preview.rendered.connect(lambda: self.highlight_preview_keywords(self.query_edit.text()))
        self.preview.previous_requested.connect(lambda: self.select_adjacent_result(-1))
        self.preview.next_requested.connect(lambda: self.select_adjacent_result(1))

        preview_splitter = QSplitter(Qt.Vertical)
        preview_splitter.addWidget(self.metadata_view)
        preview_splitter.addWidget(self.preview)
        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 3)

        splitter = QSplitter(Qt.Vertical)
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addLayout(view_controls)
        results_layout.addWidget(self.result_stack, 1)

        splitter.addWidget(results_panel)
        splitter.addWidget(preview_splitter)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([360, 360])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 18)
        layout.setSpacing(12)
        layout.addLayout(top)
        layout.addWidget(splitter, 1)
        self.update_search_button_state()
        self.update_action_buttons()
        self.update_status()

    def set_investigation_id(self, investigation_id: int | None) -> None:
        self.investigation_id = investigation_id
        self.results = []
        self.table.setRowCount(0)
        self.tree.clear()
        self.metadata_view.clear()
        self.preview.clear()
        self.update_search_button_state()
        self.update_action_buttons()
        self.update_status()

    def run_search(self) -> None:
        if self.investigation_id is None:
            return
        query = self.query_edit.text().strip()
        if not query:
            return
        try:
            searcher = build_searcher(self.config, self.db, self.mode_combo.currentText())
            folder_paths = self.folder_scope.folder_paths() if self.folder_scope is not None else None
            source_types = self.file_type_scope.source_types() if self.file_type_scope is not None else None
            limit = None if self.mode_combo.currentText() == "keyword" else 30
            self.results = searcher.search(
                self.investigation_id,
                query,
                limit=limit,
                folder_paths=folder_paths,
                source_types=source_types,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Search", str(exc))
            return
        self.populate_table()
        self.populate_tree()
        self.update_status()
        self.update_action_buttons()

    def update_search_button_state(self) -> None:
        self.search_button.setEnabled(self.investigation_id is not None and bool(self.query_edit.text().strip()))

    def update_action_buttons(self) -> None:
        has_selection = self.selected_result() is not None
        self.open_button.setEnabled(has_selection)
        self.folder_button.setEnabled(has_selection)

    def update_status(self) -> None:
        if self.investigation_id is None:
            self.status_label.setText("No investigation")
        elif not self.results:
            self.status_label.setText("No results")
        elif len(self.results) == 1:
            self.status_label.setText("1 result")
        else:
            document_count = len({result.file_id for result in self.results})
            self.status_label.setText(f"{len(self.results)} results in {document_count} documents")

    def set_result_view(self, view_index: int) -> None:
        self.result_stack.setCurrentIndex(view_index)
        if view_index == 0 and self.results and self.table.currentRow() < 0:
            self.table.selectRow(0)
        if view_index == 1 and self.results and self.tree.currentItem() is None:
            first_document = self.tree.topLevelItem(0)
            if first_document is not None and first_document.childCount():
                self.tree.setCurrentItem(first_document.child(0))
        self.update_preview()

    def populate_table(self) -> None:
        self.table.setRowCount(len(self.results))
        query = self.query_edit.text()
        for row, result in enumerate(self.results):
            result_item = QTableWidgetItem(display_filename(result.path))
            result_item.setToolTip(result.path)
            result_item.setData(RESULT_DATA_ROLE, row)
            result_item.setData(TITLE_ROLE, display_filename(result.path))
            result_item.setData(SUBTITLE_ROLE, result_context_label(result))
            result_item.setData(SNIPPET_ROLE, result_snippet(result, query))
            result_item.setData(TERMS_ROLE, query_terms(query))
            result_item.setData(ROW_HEIGHT_ROLE, MATCH_ROW_HEIGHT)
            self.table.setItem(row, 0, result_item)

            relevance_item = QTableWidgetItem(relevance_label(row))
            relevance_item.setToolTip(f"Raw score: {result.score:.3f}")
            relevance_item.setData(RESULT_DATA_ROLE, row)
            relevance_item.setData(RELEVANCE_ROLE, relevance_for_rank(row, len(self.results)))
            relevance_item.setData(RELEVANCE_LABEL_ROLE, relevance_label(row))
            relevance_item.setData(ROW_HEIGHT_ROLE, MATCH_ROW_HEIGHT)
            self.table.setItem(row, 1, relevance_item)

            why_item = QTableWidgetItem(", ".join(type_badges_for_result(result)))
            why_item.setData(RESULT_DATA_ROLE, row)
            why_item.setData(BADGES_ROLE, type_badges_for_result(result))
            why_item.setData(ROW_HEIGHT_ROLE, MATCH_ROW_HEIGHT)
            self.table.setItem(row, 2, why_item)

            location_item = QTableWidgetItem(result_file_location(result))
            location_item.setToolTip(result.path)
            location_item.setData(RESULT_DATA_ROLE, row)
            location_item.setData(ROW_HEIGHT_ROLE, MATCH_ROW_HEIGHT)
            self.table.setItem(row, 3, location_item)
            self.table.setRowHeight(row, MATCH_ROW_HEIGHT)
        if self.results and self.result_stack.currentIndex() == 0:
            self.table.selectRow(0)

    def populate_tree(self) -> None:
        self.tree.clear()
        query = self.query_edit.text()
        for group in grouped_results(self.results):
            first_index, first_result = group[0]
            best_index = min(result_index for result_index, _ in group)
            document_item = QTreeWidgetItem(self.tree)
            document_item.setText(0, display_filename(first_result.path))
            document_item.setText(1, relevance_label(best_index, prefix="Best "))
            document_item.setText(2, ", ".join(document_type_badges_for_group(group)))
            document_item.setText(3, result_file_location(first_result))
            document_item.setToolTip(0, first_result.path)
            document_item.setToolTip(3, first_result.path)
            document_item.setData(0, RESULT_DATA_ROLE, {"kind": "document", "result_index": first_index})
            document_item.setData(0, TITLE_ROLE, display_filename(first_result.path))
            document_item.setData(0, SUBTITLE_ROLE, document_group_subtitle(group))
            document_item.setData(0, DOCUMENT_ROLE, True)
            document_item.setData(1, RELEVANCE_ROLE, relevance_for_rank(best_index, len(self.results)))
            document_item.setData(1, RELEVANCE_LABEL_ROLE, relevance_label(best_index, prefix="Best "))
            document_item.setData(2, BADGES_ROLE, document_type_badges_for_group(group))
            set_tree_item_row_height(document_item, DOCUMENT_ROW_HEIGHT)

            for result_index, result in sorted(group, key=lambda item: (item[1].chunk_index, item[0])):
                item = QTreeWidgetItem(document_item)
                item.setText(0, result_match_title(result))
                item.setText(1, relevance_label(result_index))
                item.setText(2, ", ".join(type_badges_for_result(result)))
                item.setText(3, result_file_location(result))
                item.setToolTip(0, result.path)
                item.setToolTip(3, result.path)
                item.setData(0, RESULT_DATA_ROLE, {"kind": "result", "result_index": result_index})
                item.setData(0, TITLE_ROLE, result_match_title(result))
                item.setData(0, SUBTITLE_ROLE, result_match_subtitle(result))
                item.setData(0, SNIPPET_ROLE, result_snippet(result, query))
                item.setData(0, TERMS_ROLE, query_terms(query))
                item.setData(1, RELEVANCE_ROLE, relevance_for_rank(result_index, len(self.results)))
                item.setData(1, RELEVANCE_LABEL_ROLE, relevance_label(result_index))
                item.setData(2, BADGES_ROLE, type_badges_for_result(result))
                set_tree_item_row_height(item, MATCH_ROW_HEIGHT)
        self.tree.expandToDepth(0)
        if self.results and self.result_stack.currentIndex() == 1:
            first_document = self.tree.topLevelItem(0)
            if first_document is not None and first_document.childCount():
                self.tree.setCurrentItem(first_document.child(0))

    def selected_result(self) -> SearchResult | None:
        if self.result_stack.currentIndex() == 1:
            item = self.tree.currentItem()
            if item is None:
                return None
            data = item.data(0, RESULT_DATA_ROLE)
            if not isinstance(data, dict):
                return None
            result_index = data.get("result_index")
            if not isinstance(result_index, int) or result_index < 0 or result_index >= len(self.results):
                return None
            return self.results[result_index]

        row = self.table.currentRow()
        if row < 0 or row >= len(self.results):
            return None
        return self.results[row]

    def update_preview(self) -> None:
        result = self.selected_result()
        self.update_action_buttons()
        if result is None:
            self.metadata_view.clear()
            self.preview.clear()
            return
        self.metadata_view.setHtml(metadata_html(result, self.mode_combo.currentText()))
        result_index = self.results.index(result)
        self.preview.set_chunk(
            result.chunk_text,
            result.display_markdown,
            path=result.path,
            chunk_index=result.chunk_index,
            metadata=result.metadata or {},
            has_previous=result_index > 0,
            has_next=result_index + 1 < len(self.results),
        )
        self.highlight_preview_keywords(self.query_edit.text())

    def select_adjacent_result(self, offset: int) -> None:
        result = self.selected_result()
        if result is None:
            return
        target = self.results.index(result) + offset
        if not 0 <= target < len(self.results):
            return
        if self.result_stack.currentIndex() == 0:
            self.table.selectRow(target)
            return
        iterator = self.tree.findItems("", Qt.MatchContains | Qt.MatchRecursive, 0)
        for item in iterator:
            data = item.data(0, RESULT_DATA_ROLE)
            if isinstance(data, dict) and data.get("kind") == "result" and data.get("result_index") == target:
                self.tree.setCurrentItem(item)
                return

    def highlight_preview_keywords(self, query: str, start_position: int = 0) -> None:
        terms = query_terms(query)
        if not terms:
            return

        escaped_terms = [re.escape(term) for term in sorted(terms, key=len, reverse=True)]
        pattern = re.compile(
            rf"(?<![\w\-])(?:{'|'.join(escaped_terms)})(?![\w\-])",
            flags=re.IGNORECASE | re.UNICODE,
        )
        text = self.preview.toPlainText()
        cursor = self.preview.textCursor()
        highlight_format = QTextCharFormat()
        highlight_format.setBackground(QColor("#fff59d"))
        highlight_format.setForeground(QColor("#111827"))
        first_match_start: int | None = None

        for match in pattern.finditer(text[start_position:]):
            if first_match_start is None:
                first_match_start = start_position + match.start()
            cursor.setPosition(start_position + match.start())
            cursor.setPosition(start_position + match.end(), QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(highlight_format)

        cursor.clearSelection()
        cursor.setPosition(first_match_start or 0)
        self.preview.setTextCursor(cursor)
        if first_match_start is not None:
            self.preview.ensureCursorVisible()

    def open_selected_file(self) -> None:
        result = self.selected_result()
        if result is None:
            return
        try:
            open_file(result.path)
        except Exception as exc:
            QMessageBox.warning(self, "Open file", str(exc))

    def open_selected_folder(self) -> None:
        result = self.selected_result()
        if result is None:
            return
        try:
            open_containing_folder(result.path)
        except Exception as exc:
            QMessageBox.warning(self, "Open folder", str(exc))


class SearchResultDelegate(QStyledItemDelegate):
    def initStyleOption(self, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        super().initStyleOption(option, index)
        option.textElideMode = Qt.ElideRight

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        if index.data(TITLE_ROLE) is not None:
            self.paint_result_text(painter, option, index)
            return
        if index.data(RELEVANCE_ROLE) is not None:
            self.paint_relevance(painter, option, index)
            return
        if index.data(BADGES_ROLE):
            self.paint_badges(painter, option, index)
            return
        super().paint(painter, option, index)

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        base = super().sizeHint(option, index)
        row_height = index.data(ROW_HEIGHT_ROLE)
        if row_height is not None:
            return QSize(base.width(), int(row_height))
        if index.data(SNIPPET_ROLE) is not None:
            return QSize(base.width(), MATCH_ROW_HEIGHT)
        if index.data(DOCUMENT_ROLE):
            return QSize(base.width(), DOCUMENT_ROW_HEIGHT)
        return base

    def paint_result_text(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, opt, painter, opt.widget)

        title = str(index.data(TITLE_ROLE) or "")
        subtitle = str(index.data(SUBTITLE_ROLE) or "")
        snippet = str(index.data(SNIPPET_ROLE) or "")
        terms = index.data(TERMS_ROLE) or []
        document = bool(index.data(DOCUMENT_ROLE))
        selected = bool(opt.state & QStyle.State_Selected)

        rect = opt.rect.adjusted(8, 5, -8, -5)
        html = result_cell_html(title, subtitle, snippet, terms, document=document, selected=selected)
        doc = QTextDocument()
        doc.setDefaultFont(opt.font)
        doc.setDocumentMargin(0)
        doc.setTextWidth(rect.width())
        doc.setHtml(html)

        painter.save()
        painter.translate(rect.topLeft())
        doc.drawContents(painter, QRectF(0, 0, rect.width(), rect.height()))
        painter.restore()

    def paint_relevance(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, opt, painter, opt.widget)

        value = clamp_float(index.data(RELEVANCE_ROLE), 0.0, 1.0)
        label = str(index.data(RELEVANCE_LABEL_ROLE) or "")
        rect = opt.rect.adjusted(8, 9, -8, -9)

        painter.save()
        font = QFont(opt.font)
        font.setPointSize(max(8, font.pointSize() - 1))
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#334155"))
        painter.drawText(QRect(rect.left(), rect.top(), rect.width(), 18), Qt.AlignLeft | Qt.AlignVCenter, label)

        bar_rect = QRect(rect.left(), rect.top() + 25, rect.width(), 8)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#e2e8f0"))
        painter.drawRoundedRect(bar_rect, 4, 4)
        if value > 0:
            fill_rect = QRect(bar_rect.left(), bar_rect.top(), max(4, int(bar_rect.width() * value)), bar_rect.height())
            painter.setBrush(QColor("#2563eb"))
            painter.drawRoundedRect(fill_rect, 4, 4)
        painter.restore()

    def paint_badges(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawPrimitive(QStyle.PE_PanelItemViewItem, opt, painter, opt.widget)

        badges = [str(badge) for badge in (index.data(BADGES_ROLE) or []) if str(badge).strip()]
        rect = opt.rect.adjusted(8, 9, -8, -7)
        painter.save()
        font = QFont(opt.font)
        font.setPointSize(max(8, font.pointSize() - 1))
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        x = rect.left()
        y = rect.top()
        line_height = 22

        for badge in badges:
            label = badge_label(badge)
            width = metrics.horizontalAdvance(label) + 16
            if x > rect.left() and x + width > rect.right():
                x = rect.left()
                y += line_height
            if y + line_height > rect.bottom() + 4:
                break
            background, foreground = badge_colors(badge)
            badge_rect = QRect(x, y + 1, min(width, rect.right() - x + 1), 18)
            painter.setPen(QPen(background.darker(108)))
            painter.setBrush(background)
            painter.drawRoundedRect(badge_rect, 5, 5)
            painter.setPen(foreground)
            painter.drawText(badge_rect.adjusted(7, 0, -7, 0), Qt.AlignCenter, metrics.elidedText(label, Qt.ElideRight, badge_rect.width() - 10))
            x += width + 5
        painter.restore()


def display_filename(path: str) -> str:
    path_parts = split_path_parts(path)
    return path_parts[-1] if path_parts else Path(path).name


def set_tree_item_row_height(item: QTreeWidgetItem, height: int) -> None:
    for column in range(item.columnCount()):
        item.setData(column, ROW_HEIGHT_ROLE, height)
        item.setSizeHint(column, QSize(0, height))


def result_cell_html(
    title: str,
    subtitle: str,
    snippet: str,
    terms: list[str],
    *,
    document: bool = False,
    selected: bool = False,
) -> str:
    title_color = "#0f172a"
    subtitle_color = "#64748b"
    snippet_color = "#1f2937" if not selected else "#111827"
    weight = "700" if document else "600"
    title_size = "10.5pt" if document else "10pt"
    parts = [
        f'<div style="font-size:{title_size}; font-weight:{weight}; color:{title_color};">'
        f"{escape(title)}</div>"
    ]
    if subtitle:
        parts.append(f'<div style="font-size:8.5pt; color:{subtitle_color}; margin-top:1px;">{escape(subtitle)}</div>')
    if snippet:
        parts.append(
            f'<div style="font-size:9pt; color:{snippet_color}; margin-top:5px; line-height:125%;">'
            f"{highlight_terms_html(snippet, terms)}</div>"
        )
    return "".join(parts)


def highlight_terms_html(text: str, terms: list[str]) -> str:
    pattern = terms_pattern(terms)
    if pattern is None:
        return escape(text)

    html_parts: list[str] = []
    cursor = 0
    for match in pattern.finditer(text):
        html_parts.append(escape(text[cursor : match.start()]))
        html_parts.append(
            '<span style="background-color:#fff59d; color:#111827; font-weight:600;">'
            f"{escape(match.group(0))}</span>"
        )
        cursor = match.end()
    html_parts.append(escape(text[cursor:]))
    return "".join(html_parts)


def result_snippet(result: SearchResult, query: str, max_chars: int = 260) -> str:
    text = compact_text(result.chunk_text)
    if not text:
        return ""
    terms = query_terms(query)
    pattern = terms_pattern(terms)
    match = pattern.search(text) if pattern is not None else None
    if match is None:
        snippet = text[:max_chars].strip()
        return f"{snippet} ..." if len(text) > max_chars else snippet

    start = max(0, match.start() - 80)
    end = min(len(text), start + max_chars)
    start = max(0, min(start, max(0, end - max_chars)))
    snippet = text[start:end].strip()
    if start > 0:
        snippet = f"... {snippet}"
    if end < len(text):
        snippet = f"{snippet} ..."
    return snippet


def result_match_subtitle(result: SearchResult) -> str:
    if result.email_subject:
        return result.email_subject
    if result.attachment_name:
        return f"attachment: {result.attachment_name}"
    return ""


def result_match_title(result: SearchResult) -> str:
    location = result_location_label(result)
    if location.startswith("page"):
        return location.capitalize()
    if result.source_type == "msg":
        return "Email match"
    if result.attachment_name:
        return "Attachment match"
    return "Text match"


def result_file_location(result: SearchResult) -> str:
    return display_reversed_folder_path(result.path, parts=4)


def compact_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def terms_pattern(terms: list[str]) -> re.Pattern[str] | None:
    clean_terms = [term for term in terms if term.strip()]
    if not clean_terms:
        return None
    escaped_terms = [re.escape(term) for term in sorted(clean_terms, key=len, reverse=True)]
    return re.compile(
        rf"(?<![\w\-])(?:{'|'.join(escaped_terms)})(?![\w\-])",
        flags=re.IGNORECASE | re.UNICODE,
    )


def relevance_for_rank(result_index: int, result_count: int) -> float:
    if result_count <= 1:
        return 1.0
    return max(0.18, 1.0 - (result_index / (result_count - 1)) * 0.72)


def relevance_label(result_index: int, prefix: str = "") -> str:
    return f"{prefix}#{result_index + 1}"


def type_badges_for_result(result: SearchResult) -> list[str]:
    badge = source_type_badge(result.source_type)
    return [badge] if badge else []


def document_type_badges_for_group(group: list[tuple[int, SearchResult]]) -> list[str]:
    badges = []
    for _, result in group:
        badges.extend(type_badges_for_result(result))
    badges = unique_badges(badges)
    badges.append(match_count_label(len(group)))
    return badges


def source_badges_for_result(result: SearchResult, fallback_mode: str | None = None) -> list[str]:
    badges: list[str] = []
    source = result.retrieval_source.lower()
    if "keyword" in source or result.keyword_rank is not None:
        badges.append("keyword")
    if "vector" in source or result.vector_rank is not None:
        badges.append("semantic")
    if not badges and source:
        badges.append(source.replace("+", " + "))
    if not badges and fallback_mode:
        badges.append(search_mode_badge(fallback_mode))
    if result.context_role and result.context_role != "primary":
        badges.append("context")
    return unique_badges(badges)


def document_group_subtitle(group: list[tuple[int, SearchResult]]) -> str:
    return match_count_label(len(group))


def match_count_label(count: int) -> str:
    return "1 match" if count == 1 else f"{count} matches"


def search_mode_badge(mode: str) -> str:
    mode = mode.strip().lower()
    if mode == "semantic":
        return "semantic"
    if mode == "keyword":
        return "keyword"
    if mode == "hybrid":
        return "hybrid"
    return mode or "search"


def source_type_badge(source_type: str) -> str:
    source_type = source_type.strip().lower()
    if source_type == "msg":
        return "email"
    return source_type


def unique_badges(badges: list[str]) -> list[str]:
    unique = []
    seen = set()
    for badge in badges:
        key = badge.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(key)
    return unique


def badge_label(badge: str) -> str:
    labels = {
        "context": "context",
        "docx": "DOCX",
        "email": "email",
        "hybrid": "hybrid",
        "keyword": "keyword",
        "pdf": "PDF",
        "semantic": "semantic",
        "txt": "TXT",
    }
    return labels.get(badge, badge)


def badge_colors(badge: str) -> tuple[QColor, QColor]:
    if badge == "keyword":
        return QColor("#dbeafe"), QColor("#1d4ed8")
    if badge == "semantic":
        return QColor("#dcfce7"), QColor("#166534")
    if badge == "hybrid":
        return QColor("#ede9fe"), QColor("#5b21b6")
    if badge == "context":
        return QColor("#ffedd5"), QColor("#9a3412")
    if badge.endswith("match") or badge.endswith("matches"):
        return QColor("#f1f5f9"), QColor("#475569")
    return QColor("#eef2f7"), QColor("#334155")


def clamp_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = minimum
    return max(minimum, min(maximum, number))


def grouped_results(results: list[SearchResult]) -> list[list[tuple[int, SearchResult]]]:
    groups: list[list[tuple[int, SearchResult]]] = []
    groups_by_file_id: dict[int, list[tuple[int, SearchResult]]] = {}
    for index, result in enumerate(results):
        group = groups_by_file_id.get(result.file_id)
        if group is None:
            group = []
            groups_by_file_id[result.file_id] = group
            groups.append(group)
        group.append((index, result))
    return groups


def result_location(result: SearchResult) -> str:
    location = result_location_label(result)
    if location.startswith("page"):
        return f"{location.capitalize()}, chunk {result.chunk_index + 1}"
    return f"Chunk {result.chunk_index + 1}"


def result_context_label(result: SearchResult) -> str:
    if result.email_subject:
        return result.email_subject
    if result.attachment_name:
        return f"attachment: {result.attachment_name}"
    location = result_location_label(result)
    if location.startswith("page"):
        return location
    return ""


def display_path_tail(path: str, parts: int = 4) -> str:
    path_parts = split_path_parts(path)
    if not path_parts:
        return path
    return "\\".join(path_parts[-parts:])


def display_reversed_folder_path(path: str, parts: int = 4) -> str:
    path_parts = split_path_parts(path)
    if len(path_parts) <= 1:
        return ""
    folder_parts = path_parts[:-1]
    return "\\".join(reversed(folder_parts[-parts:]))


def split_path_parts(path: str) -> list[str]:
    return [part for part in re.split(r"[\\/]+", path) if part]


def metadata_text(result: SearchResult) -> str:
    filename, lines = metadata_lines(result)
    if not lines:
        return filename
    return "\n".join([filename, *lines])


def metadata_html(result: SearchResult, fallback_mode: str | None = None) -> str:
    filename, metadata_rows = metadata_items(result)
    body = [
        '<div style="color:#111827;">',
        f'<div style="font-size:12pt; font-weight:700; margin-bottom:8px;">{escape(filename)}</div>',
    ]
    retrieval_rows = retrieval_detail_rows(result, fallback_mode)
    if retrieval_rows:
        body.append(detail_section_html("Retrieval", retrieval_rows))
    if metadata_rows:
        body.append(detail_section_html("Details", metadata_rows))
    body.append("</div>")
    return "".join(body)


def metadata_lines(result: SearchResult) -> tuple[str, list[str]]:
    filename, metadata_rows = metadata_items(result)
    rows = [*retrieval_detail_rows(result), *metadata_rows]
    return filename, [f"{label}: {value}" for label, value in rows]


def metadata_items(result: SearchResult) -> tuple[str, list[tuple[str, str]]]:
    metadata = dict(result.metadata or {})
    metadata.pop("filename", None)
    filename = display_filename(result.path)
    metadata["path"] = result.path
    metadata.setdefault("file_type", result.source_type)
    if result.page_number:
        metadata.setdefault("page_number", result.page_number)
    if result.email_subject:
        metadata.setdefault("subject", result.email_subject)
    if result.email_sender:
        metadata.setdefault("sender", result.email_sender)
    if result.email_recipients:
        metadata.setdefault("recipients", result.email_recipients)
    if result.email_date:
        metadata.setdefault("date", result.email_date)
    if result.attachment_name:
        metadata.setdefault("attachment_name", result.attachment_name)

    key_order = metadata_keys_for_type(result.source_type)
    rows: list[tuple[str, str]] = []
    for key in key_order:
        if key == "filename":
            continue
        value = metadata.pop(key, None)
        if value not in (None, ""):
            rows.append((metadata_label(key), format_metadata_value(key, value)))
    for key in sorted(metadata):
        if key in hidden_metadata_keys():
            continue
        value = metadata[key]
        if value not in (None, ""):
            rows.append((metadata_label(key), format_metadata_value(key, value)))
    return filename, rows


def retrieval_detail_rows(result: SearchResult, fallback_mode: str | None = None) -> list[tuple[str, str]]:
    rows = [
        ("Location", result_location(result)),
        ("Relevance score", f"{result.score:.3f}"),
    ]
    source = retrieval_source_label(result, fallback_mode)
    if source:
        rows.append(("Matched by", source))
    ranks = ranking_summary(result)
    if ranks:
        rows.append(("Ranks", ranks))
    if result.rerank_score is not None:
        rows.append(("Rerank score", f"{result.rerank_score:.3f}"))
    if result.context_role and result.context_role != "primary":
        rows.append(("Context role", result.context_role))
    return rows


def detail_section_html(title: str, rows: list[tuple[str, str]]) -> str:
    body = [
        f'<div style="font-size:9pt; font-weight:700; color:#334155; margin-top:8px; margin-bottom:4px;">'
        f"{escape(title)}</div>",
        '<table cellspacing="0" cellpadding="3" style="width:100%;">',
    ]
    for label, value in rows:
        body.append(
            "<tr>"
            '<td style="white-space:nowrap; color:#64748b; font-weight:600; padding-right:10px;">'
            f"{escape(label)}</td>"
            '<td style="color:#111827;">'
            f"{escape(value)}</td>"
            "</tr>"
        )
    body.append("</table>")
    return "".join(body)


def retrieval_source_label(result: SearchResult, fallback_mode: str | None = None) -> str:
    badges = [badge for badge in source_badges_for_result(result, fallback_mode) if badge != "context"]
    if not badges:
        return ""
    return " + ".join(badge_label(badge) for badge in badges)


def ranking_summary(result: SearchResult) -> str:
    ranks = []
    if result.keyword_rank is not None:
        ranks.append(f"keyword #{result.keyword_rank}")
    if result.vector_rank is not None:
        ranks.append(f"semantic #{result.vector_rank}")
    if result.fused_rank is not None:
        ranks.append(f"fusion #{result.fused_rank}")
    return ", ".join(ranks)


def metadata_keys_for_type(source_type: str) -> list[str]:
    common = ["filename", "file_type", "path"]
    if source_type == "pdf":
        return common + [
            "page_start",
            "page_end",
            "pages",
            "page_number",
            "heading_path_text",
            "content_type",
            "contains_figure",
            "page_count",
            "title",
            "author",
            "subject",
            "keywords",
            "creator",
            "producer",
            "creation_date",
            "modification_date",
        ]
    if source_type == "docx":
        return common + [
            "title",
            "author",
            "subject",
            "keywords",
            "category",
            "comments",
            "last_modified_by",
            "revision",
            "creation_date",
            "modification_date",
        ]
    if source_type == "msg":
        return common + ["subject", "sender", "recipients", "cc", "date", "attachment_name"]
    return common + ["size_bytes", "modified_ns", "attachment_name", "parent_email_subject", "parent_email_sender"]


def hidden_metadata_keys() -> set[str]:
    return {
        "display_html",
        "embedding_context",
        "source_content_hash",
        "source_reference_version",
        "source_document_id",
        "source_section",
        "file_id",
        "attachment_chain",
        "heading_path",
        "source_spans",
        "retrieval_prefix",
        "retrieval_prefix_char_count",
        "overlap_source_spans",
        "pdf_chunk_debug",
        "raw_text",
        "layout_blocks",
        "pdf_layout_version",
    }


def metadata_label(key: str) -> str:
    labels = {
        "cc": "CC",
        "docx": "DOCX",
        "file_type": "Type",
        "modified_ns": "Modified",
        "content_type": "Content type",
        "heading_path_text": "Section",
        "page_count": "Pages",
        "page_number": "Page",
        "page_start": "Page start",
        "page_end": "Page end",
        "parent_email_sender": "Parent sender",
        "parent_email_subject": "Parent subject",
        "size_bytes": "Size",
    }
    return labels.get(key, key.replace("_", " ").capitalize())


def format_metadata_value(key: str, value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if key == "size_bytes":
        try:
            return format_bytes(int(value))
        except (TypeError, ValueError):
            return str(value)
    if key == "modified_ns":
        try:
            return format_display_date(datetime.fromtimestamp(int(value) / 1_000_000_000))
        except (OSError, TypeError, ValueError):
            return str(value)
    if is_date_key(key):
        formatted = parse_display_date(str(value))
        if formatted:
            return formatted
    return str(value)


def is_date_key(key: str) -> bool:
    return key == "date" or key.endswith("_date")


def parse_display_date(value: str) -> str | None:
    text = value.strip()
    if not text:
        return None

    pdf_match = re.match(r"^D:(\d{4})(\d{2})?(\d{2})?", text)
    if pdf_match:
        year = int(pdf_match.group(1))
        month = int(pdf_match.group(2) or "1")
        day = int(pdf_match.group(3) or "1")
        try:
            return format_display_date(datetime(year, month, day))
        except ValueError:
            return None

    iso_text = text.replace("Z", "+00:00")
    try:
        return format_display_date(datetime.fromisoformat(iso_text))
    except ValueError:
        pass

    try:
        return format_display_date(parsedate_to_datetime(text))
    except (TypeError, ValueError):
        return None


def format_display_date(value: datetime) -> str:
    return f"{value.day} {value.strftime('%B')} {value.year}"


def format_bytes(byte_count: int) -> str:
    value = float(max(byte_count, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{byte_count} B"
