from __future__ import annotations

import re
from datetime import datetime
from email.utils import parsedate_to_datetime
from html import escape
from pathlib import Path
from typing import Any

from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtGui import QColor, QTextCharFormat, QTextCursor
from PyQt5.QtWidgets import (
    QAbstractItemView,
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
from investigation_rag.gui.folder_scope import FolderScope
from investigation_rag.retrieval.keyword_search import query_terms
from investigation_rag.retrieval.ranking import SearchResult
from investigation_rag.utils.windows_open import open_containing_folder, open_file


class SearchPanel(QWidget):
    def __init__(self, config: AppConfig, db: SQLiteDB, store: LanceDBStore, folder_scope: FolderScope | None = None):
        super().__init__()
        self.config = config
        self.db = db
        self.store = store
        self.folder_scope = folder_scope
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
        self.table.setHorizontalHeaderLabels(["Score", "Filename", "Path", "Type"])
        self.table.setItemDelegate(SearchResultDelegate(self.table))
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.itemSelectionChanged.connect(self.update_preview)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Document / result", "Score", "Location", "Type"])
        self.tree.itemSelectionChanged.connect(self.update_preview)
        self.tree.setItemDelegate(SearchResultDelegate(self.tree))
        self.tree.setAlternatingRowColors(True)
        self.tree.header().setSectionResizeMode(0, QHeaderView.Stretch)

        self.result_stack = QStackedWidget()
        self.result_stack.addWidget(self.table)
        self.result_stack.addWidget(self.tree)

        list_button = QPushButton("List")
        list_button.setCheckable(True)
        list_button.setChecked(True)
        list_button.setProperty("segment", True)
        list_button.setProperty("segmentPosition", "left")
        list_button.clicked.connect(lambda: self.set_result_view(0))
        hierarchy_button = QPushButton("Hierarchy")
        hierarchy_button.setCheckable(True)
        hierarchy_button.setProperty("segment", True)
        hierarchy_button.setProperty("segmentPosition", "right")
        hierarchy_button.clicked.connect(lambda: self.set_result_view(1))
        self.view_button_group = QButtonGroup(self)
        self.view_button_group.setExclusive(True)
        self.view_button_group.addButton(list_button)
        self.view_button_group.addButton(hierarchy_button)

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
        self.preview = QTextEdit()
        self.preview.setReadOnly(True)
        self.preview.setPlaceholderText("Select a result to preview matching text")

        preview_splitter = QSplitter(Qt.Vertical)
        preview_splitter.addWidget(self.metadata_view)
        preview_splitter.addWidget(self.preview)
        preview_splitter.setStretchFactor(0, 1)
        preview_splitter.setStretchFactor(1, 3)

        splitter = QSplitter()
        results_panel = QWidget()
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        results_layout.addLayout(view_controls)
        results_layout.addWidget(self.result_stack, 1)

        splitter.addWidget(results_panel)
        splitter.addWidget(preview_splitter)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)

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
            limit = None if self.mode_combo.currentText() == "keyword" else 30
            self.results = searcher.search(self.investigation_id, query, limit=limit, folder_paths=folder_paths)
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
            self.status_label.setText(f"{len(self.results)} results")

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
        for row, result in enumerate(self.results):
            score_item = QTableWidgetItem(f"{result.score:.3f}")
            score_item.setData(Qt.UserRole, row)
            self.table.setItem(row, 0, score_item)
            filename_item = QTableWidgetItem(display_filename(result.path))
            filename_item.setData(Qt.UserRole, row)
            self.table.setItem(row, 1, filename_item)
            path_item = QTableWidgetItem(display_reversed_folder_path(result.path, parts=4))
            path_item.setToolTip(result.path)
            path_item.setData(Qt.UserRole, row)
            self.table.setItem(row, 2, path_item)
            type_item = QTableWidgetItem(result.source_type)
            type_item.setData(Qt.UserRole, row)
            self.table.setItem(row, 3, type_item)
        if self.results and self.result_stack.currentIndex() == 0:
            self.table.selectRow(0)

    def populate_tree(self) -> None:
        self.tree.clear()
        for group in grouped_results(self.results):
            first_index, first_result = group[0]
            document_item = QTreeWidgetItem(self.tree)
            document_item.setText(0, display_filename(first_result.path))
            document_item.setText(1, f"{max(result.score for _, result in group):.3f}")
            document_item.setText(2, f"{len(group)} result(s)")
            document_item.setText(3, first_result.source_type)
            document_item.setToolTip(0, first_result.path)
            document_item.setData(0, Qt.UserRole, {"kind": "document", "result_index": first_index})

            for result_index, result in sorted(group, key=lambda item: (item[1].chunk_index, item[0])):
                item = QTreeWidgetItem(document_item)
                item.setText(0, result_location(result))
                item.setText(1, f"{result.score:.3f}")
                item.setText(2, result_context_label(result))
                item.setText(3, result.source_type)
                item.setToolTip(0, "Click to show result content")
                item.setData(0, Qt.UserRole, {"kind": "result", "result_index": result_index})
        self.tree.expandToDepth(0)
        self.tree.resizeColumnToContents(1)
        self.tree.resizeColumnToContents(2)
        if self.results and self.result_stack.currentIndex() == 1:
            first_document = self.tree.topLevelItem(0)
            if first_document is not None and first_document.childCount():
                self.tree.setCurrentItem(first_document.child(0))

    def selected_result(self) -> SearchResult | None:
        if self.result_stack.currentIndex() == 1:
            item = self.tree.currentItem()
            if item is None:
                return None
            data = item.data(0, Qt.UserRole)
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
        self.metadata_view.setHtml(metadata_html(result))
        set_preview_text(self.preview, result.display_markdown, result.chunk_text)
        self.highlight_preview_keywords(self.query_edit.text())

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

        for match in pattern.finditer(text[start_position:]):
            cursor.setPosition(start_position + match.start())
            cursor.setPosition(start_position + match.end(), QTextCursor.KeepAnchor)
            cursor.mergeCharFormat(highlight_format)

        cursor.clearSelection()
        cursor.setPosition(0)
        self.preview.setTextCursor(cursor)

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
        option.textElideMode = Qt.ElideLeft if index.column() == 2 else Qt.ElideRight


def display_filename(path: str) -> str:
    path_parts = split_path_parts(path)
    return path_parts[-1] if path_parts else Path(path).name


def set_preview_text(preview: QTextEdit, display_markdown: str | None, plain_text: str) -> None:
    if display_markdown:
        set_markdown = getattr(preview, "setMarkdown", None)
        if callable(set_markdown):
            set_markdown(display_markdown)
            return
    preview.setPlainText(plain_text)


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
    if result.page_number:
        return f"Page {result.page_number}, chunk {result.chunk_index + 1}"
    return f"Chunk {result.chunk_index + 1}"


def result_context_label(result: SearchResult) -> str:
    if result.email_subject:
        return result.email_subject
    if result.attachment_name:
        return f"attachment: {result.attachment_name}"
    if result.page_number:
        return f"page {result.page_number}"
    return display_reversed_folder_path(result.path, parts=3)


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


def metadata_html(result: SearchResult) -> str:
    filename, lines = metadata_lines(result)
    body = [f"<p><b>{escape(filename)}</b></p>"]
    if lines:
        body.append("<p>" + "<br>".join(escape(line) for line in lines) + "</p>")
    return "".join(body)


def metadata_lines(result: SearchResult) -> tuple[str, list[str]]:
    metadata = dict(result.metadata or {})
    filename = str(metadata.pop("filename", "") or display_filename(result.path))
    metadata.setdefault("path", result.path)
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
    lines = []
    for key in key_order:
        if key == "filename":
            continue
        value = metadata.pop(key, None)
        if value not in (None, ""):
            lines.append(f"{metadata_label(key)}: {format_metadata_value(key, value)}")
    for key in sorted(metadata):
        value = metadata[key]
        if value not in (None, ""):
            lines.append(f"{metadata_label(key)}: {format_metadata_value(key, value)}")
    return filename, lines


def metadata_keys_for_type(source_type: str) -> list[str]:
    common = ["filename", "file_type", "path"]
    if source_type == "pdf":
        return common + [
            "page_number",
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


def metadata_label(key: str) -> str:
    labels = {
        "cc": "CC",
        "docx": "DOCX",
        "file_type": "Type",
        "modified_ns": "Modified",
        "page_count": "Pages",
        "page_number": "Page",
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
